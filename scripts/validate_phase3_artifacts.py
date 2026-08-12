from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import read_jsonl
from src.parsing import split_steps, validate_score_alignment
from src.task_profiles import get_task_profile


_YES_NO_FINAL_RE = re.compile(
    r"^Final answer:\s*(yes|no)\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_records(path: Path) -> List[Dict[str, Any]]:
    require(path.exists(), f"Missing artifact: {path}")
    records = list(read_jsonl(path))
    require(bool(records), f"Artifact is empty: {path}")
    ids = [record.get("id") for record in records]
    require(all(ids), f"One or more records in {path} have no id")
    require(len(ids) == len(set(ids)), f"Duplicate ids found in {path}")
    return records


def validate_against_dataset(
    records: List[Dict[str, Any]],
    dataset_path: Path,
    expected_count: int,
    task: str,
) -> None:
    dataset = load_records(dataset_path)
    require(len(dataset) == expected_count, f"Dataset count is {len(dataset)}, expected {expected_count}")
    require(len(records) == expected_count, f"Artifact count is {len(records)}, expected {expected_count}")
    require(
        [record["id"] for record in records] == [record["id"] for record in dataset],
        "Artifact ids/order do not match the frozen dataset",
    )
    dataset_by_id = {record["id"]: record for record in dataset}
    for record in records:
        source = dataset_by_id[record["id"]]
        require(record.get("task") == task, f"{record['id']}: expected task={task}")
        require(record.get("question") == source.get("question"), f"{record['id']}: question changed")
        require(record.get("gold_answer") == source.get("gold_answer"), f"{record['id']}: gold answer changed")


def validate_answer(task: str, answer: Any, record_id: str) -> None:
    profile = get_task_profile(task)
    require(profile.normalize_answer(answer) is not None, f"{record_id}: invalid final answer {answer!r}")


def validate_score(score: Dict[str, Any], record_id: str, label: str) -> None:
    steps = score.get("steps", [])
    scores = score.get("scores", {})
    verifier = scores.get("verifier", [])
    contradiction = scores.get("contradiction", [])
    validate_score_alignment(steps, verifier, contradiction)
    require(len(score.get("risks", [])) == len(steps), f"{record_id}: {label} risk count mismatch")
    require("risk_summary" in score, f"{record_id}: {label} has no risk summary")


def validate_generated(
    artifact_path: Path,
    dataset_path: Path,
    expected_count: int,
    task: str,
) -> Dict[str, Any]:
    records = load_records(artifact_path)
    validate_against_dataset(records, dataset_path, expected_count, task)

    step_counts = []
    for record in records:
        record_id = record["id"]
        trace = record.get("model_trace", "")
        steps = split_steps(trace)
        require(bool(steps), f"{record_id}: no numbered reasoning steps parsed")
        validate_answer(task, record.get("model_answer"), record_id)
        if task == "strategyqa":
            require(
                len(_YES_NO_FINAL_RE.findall(trace)) == 1,
                f"{record_id}: trace must contain exactly one explicit yes/no final-answer line",
            )
        step_counts.append(len(steps))

    return {
        "kind": "generated",
        "path": str(artifact_path),
        "records": len(records),
        "task": task,
        "min_steps": min(step_counts),
        "max_steps": max(step_counts),
        "status": "PASS",
    }


def validate_candidate_pool(
    artifact_path: Path,
    dataset_path: Path,
    expected_count: int,
    task: str,
) -> Dict[str, Any]:
    records = load_records(artifact_path)
    validate_against_dataset(records, dataset_path, expected_count, task)

    proposal_count = 0
    judge_results = 0
    judge_errors = 0
    for record in records:
        record_id = record["id"]
        pool = record.get("candidate_pool")
        require(isinstance(pool, dict), f"{record_id}: missing candidate_pool")
        require(pool.get("construction_policy") == "risk_only_chain", f"{record_id}: unexpected pool policy")
        validate_score(pool.get("original_score", {}), record_id, "original")
        proposals = pool.get("proposals", [])
        require(isinstance(proposals, list), f"{record_id}: proposals is not a list")

        expected_parent_trace = record.get("model_trace")
        expected_parent_answer = record.get("model_answer")
        for proposal in proposals:
            proposal_count += 1
            require(proposal.get("parent_trace") == expected_parent_trace, f"{record_id}: parent trace mismatch")
            require(proposal.get("parent_answer") == expected_parent_answer, f"{record_id}: parent answer mismatch")
            candidate_trace = proposal.get("candidate_trace")
            require(bool(candidate_trace), f"{record_id}: candidate trace missing")
            require(bool(split_steps(candidate_trace)), f"{record_id}: candidate has no parsed steps")
            validate_answer(task, proposal.get("candidate_answer"), record_id)
            validate_score(proposal.get("candidate_score", {}), record_id, "candidate")

            improvement = float(proposal.get("improvement"))
            if improvement >= 0.0:
                has_judge = isinstance(proposal.get("judge"), dict)
                has_error = bool(proposal.get("judge_error"))
                require(has_judge or has_error, f"{record_id}: improved candidate has no frozen judge result or error")
                judge_results += int(has_judge)
                judge_errors += int(has_error)

            expected_parent_trace = candidate_trace
            expected_parent_answer = proposal.get("candidate_answer")

    return {
        "kind": "candidate_pool",
        "path": str(artifact_path),
        "records": len(records),
        "task": task,
        "proposals": proposal_count,
        "judge_results": judge_results,
        "judge_errors": judge_errors,
        "status": "PASS",
    }


def validate_policy_output(
    artifact_path: Path,
    dataset_path: Path,
    expected_count: int,
    task: str,
) -> Dict[str, Any]:
    records = load_records(artifact_path)
    validate_against_dataset(records, dataset_path, expected_count, task)

    attempts = 0
    accepted = 0
    rejected = 0
    modes = set()
    for record in records:
        record_id = record["id"]
        validate_answer(task, record.get("final_answer"), record_id)
        log = record.get("logs", {}).get("iterative_repair", {})
        require(log.get("source") == "frozen_candidate_pool", f"{record_id}: output did not use frozen pool")
        modes.add(log.get("acceptance_mode"))
        iterations = log.get("iterations", [])
        require(bool(iterations) and iterations[0].get("kind") == "original", f"{record_id}: missing original log")
        for iteration in iterations[1:]:
            attempts += 1
            require(bool(iteration.get("candidate_trace")), f"{record_id}: logged candidate trace missing")
            require("candidate_answer" in iteration, f"{record_id}: logged candidate answer missing")
            require(isinstance(iteration.get("candidate_score"), dict), f"{record_id}: candidate score missing")
            require(isinstance(iteration.get("acceptance"), dict), f"{record_id}: acceptance decision missing")
            was_accepted = bool(iteration.get("accepted"))
            accepted += int(was_accepted)
            rejected += int(not was_accepted)

    require(len(modes) == 1, f"Policy output contains multiple modes: {sorted(modes)}")
    return {
        "kind": "policy_output",
        "path": str(artifact_path),
        "records": len(records),
        "task": task,
        "mode": next(iter(modes)),
        "attempts": attempts,
        "accepted": accepted,
        "rejected": rejected,
        "status": "PASS",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline Phase 3 artifact validation")
    parser.add_argument("--kind", choices=("generated", "candidate-pool", "policy-output"), required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--task", choices=("math", "strategyqa"), required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validators = {
        "generated": validate_generated,
        "candidate-pool": validate_candidate_pool,
        "policy-output": validate_policy_output,
    }
    summary = validators[args.kind](
        args.artifact,
        args.dataset,
        args.expected_count,
        args.task,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
