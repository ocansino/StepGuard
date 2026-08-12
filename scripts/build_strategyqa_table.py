from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import read_jsonl, write_jsonl
from src.task_profiles import answers_match


@dataclass(frozen=True)
class ResultSpec:
    label: str
    run_name: str
    pool_name: str


SPECS = (
    ResultSpec(
        "Aggressive risk-only",
        "strategyqa_openai_100_aggressive_risk_only",
        "strategyqa_openai_100_aggressive_shared",
    ),
    ResultSpec(
        "Calibrated risk-only",
        "strategyqa_openai_100_calibrated_risk_only",
        "strategyqa_openai_100_calibrated_shared",
    ),
    ResultSpec(
        "Calibrated oracle guard",
        "strategyqa_openai_100_oracle",
        "strategyqa_openai_100_calibrated_shared",
    ),
    ResultSpec(
        "Strict judge guard",
        "strategyqa_openai_100_strict_judge",
        "strategyqa_openai_100_calibrated_shared",
    ),
    ResultSpec(
        "Updated judge guard",
        "strategyqa_openai_100_updated_judge",
        "strategyqa_openai_100_calibrated_shared",
    ),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> Dict[str, Any]:
    require(path.exists(), f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    require(path.exists(), f"Missing file: {path}")
    records = list(read_jsonl(path))
    require(len(records) == 100, f"Expected 100 records in {path}, found {len(records)}")
    return records


def proposal_key(record_id: str, proposal: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        record_id,
        proposal.get("iter"),
        proposal.get("parent_trace"),
        proposal.get("candidate_trace"),
    )


def classify_proposal(record: Dict[str, Any], proposal: Dict[str, Any]) -> str:
    parent_correct = answers_match(
        record.get("task"),
        proposal.get("parent_answer"),
        record.get("gold_answer"),
    )
    candidate_correct = answers_match(
        record.get("task"),
        proposal.get("candidate_answer"),
        record.get("gold_answer"),
    )
    if parent_correct and not candidate_correct:
        return "harmful_correct_to_wrong"
    if not parent_correct and candidate_correct:
        return "beneficial_wrong_to_correct"
    if parent_correct and candidate_correct:
        return "safe_correct_to_correct"
    return "unchanged_wrong_to_wrong"


def pool_proposals(pool_records: List[Dict[str, Any]]) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    proposals: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for record in pool_records:
        for proposal in record.get("candidate_pool", {}).get("proposals", []):
            key = proposal_key(record["id"], proposal)
            require(key not in proposals, f"Duplicate candidate proposal key for {record['id']}")
            proposals[key] = {
                "record": record,
                "proposal": proposal,
                "classification": classify_proposal(record, proposal),
            }
    return proposals


def accepted_decisions(
    output_records: List[Dict[str, Any]],
    valid_proposals: Dict[Tuple[Any, ...], Dict[str, Any]],
) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    decisions = {}
    for record in output_records:
        iterations = record.get("logs", {}).get("iterative_repair", {}).get("iterations", [])
        for iteration in iterations[1:]:
            key = (
                record["id"],
                iteration.get("iter"),
                iteration.get("parent_trace"),
                iteration.get("candidate_trace"),
            )
            require(key in valid_proposals, f"Policy output used a non-frozen candidate for {record['id']}")
            decisions[key] = {
                "accepted": bool(iteration.get("accepted")),
                "reason": iteration.get("acceptance", {}).get("reason"),
            }
    return decisions


def measured_risk_reduction_percent(metrics: Dict[str, Any]) -> float:
    risk = metrics["risk"]
    original = float(risk["mean_original_avg_risk"])
    delta = float(risk["mean_risk_delta"])
    return 100.0 * delta / original if original else 0.0


def format_percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def build_outputs(runs_root: Path, output_dir: Path) -> None:
    pools: Dict[str, List[Dict[str, Any]]] = {}
    proposal_maps: Dict[str, Dict[Tuple[Any, ...], Dict[str, Any]]] = {}
    for pool_name in sorted({spec.pool_name for spec in SPECS}):
        records = load_jsonl(runs_root / pool_name / "candidate_pool.jsonl")
        pools[pool_name] = records
        proposal_maps[pool_name] = pool_proposals(records)

    calibrated = pools["strategyqa_openai_100_calibrated_shared"]
    aggressive = pools["strategyqa_openai_100_aggressive_shared"]
    for calibrated_record, aggressive_record in zip(calibrated, aggressive):
        require(calibrated_record["id"] == aggressive_record["id"], "Pool id/order mismatch")
        require(calibrated_record["model_trace"] == aggressive_record["model_trace"], "Original trace mismatch")
        require(calibrated_record.get("model_answer") == aggressive_record.get("model_answer"), "Original answer mismatch")
        calibrated_scores = calibrated_record["candidate_pool"]["original_score"]["scores"]
        aggressive_scores = aggressive_record["candidate_pool"]["original_score"]["scores"]
        require(calibrated_scores == aggressive_scores, "Aggressive pool did not reuse frozen original scores")

    rows = []
    decisions_by_run: Dict[str, Dict[Tuple[Any, ...], Dict[str, Any]]] = {}
    original_accuracy = None
    for spec in SPECS:
        run_dir = runs_root / spec.run_name
        metrics = load_json(run_dir / "metrics.json")
        output_records = load_jsonl(run_dir / "iterative_repaired.jsonl")
        decisions = accepted_decisions(output_records, proposal_maps[spec.pool_name])
        decisions_by_run[spec.run_name] = decisions

        correctness = metrics["correctness"]
        repair_loop = metrics["repair_loop"]
        current_original_accuracy = float(correctness["original_accuracy"])
        if original_accuracy is None:
            original_accuracy = current_original_accuracy
        require(current_original_accuracy == original_accuracy, "Original accuracy differs across configurations")

        proposal_counts = {
            "harmful_generated": 0,
            "harmful_evaluated": 0,
            "harmful_accepted": 0,
            "harmful_blocked": 0,
            "harmful_unreached": 0,
            "beneficial_generated": 0,
            "beneficial_evaluated": 0,
            "beneficial_accepted": 0,
            "beneficial_blocked": 0,
            "beneficial_unreached": 0,
        }
        for key, item in proposal_maps[spec.pool_name].items():
            classification = item["classification"]
            decision = decisions.get(key)
            evaluated = decision is not None
            accepted = bool(decision and decision.get("accepted", False))
            if classification == "harmful_correct_to_wrong":
                proposal_counts["harmful_generated"] += 1
                proposal_counts["harmful_evaluated"] += int(evaluated)
                proposal_counts["harmful_accepted"] += int(accepted)
                proposal_counts["harmful_blocked"] += int(evaluated and not accepted)
                proposal_counts["harmful_unreached"] += int(not evaluated)
            elif classification == "beneficial_wrong_to_correct":
                proposal_counts["beneficial_generated"] += 1
                proposal_counts["beneficial_evaluated"] += int(evaluated)
                proposal_counts["beneficial_accepted"] += int(accepted)
                proposal_counts["beneficial_blocked"] += int(evaluated and not accepted)
                proposal_counts["beneficial_unreached"] += int(not evaluated)

        rows.append(
            {
                "configuration": spec.label,
                "original_accuracy": current_original_accuracy,
                "final_accuracy": float(correctness["final_accuracy"]),
                "accuracy_delta": float(correctness["accuracy_delta"]),
                "correct_to_wrong": int(correctness["outcomes"]["correct_to_wrong"]),
                "wrong_to_correct": int(correctness["outcomes"]["wrong_to_correct"]),
                "repair_attempt_rate": float(repair_loop["repair_attempt_rate"]),
                "accepted_repair_rate": float(repair_loop["accepted_repair_record_rate"]),
                "measured_risk_reduction_percent": measured_risk_reduction_percent(metrics),
                **proposal_counts,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "strategyqa_table.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "strategyqa_table.csv").open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "| Configuration | Original acc. | Final acc. | Delta | C→W | W→C | Attempt rate | Accepted rate | Measured risk reduction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            "| {configuration} | {original} | {final} | {delta:+.1f} pp | {ctw} | {wtc} | {attempt} | {accepted} | {risk:.1f}% |".format(
                configuration=row["configuration"],
                original=format_percent(row["original_accuracy"]),
                final=format_percent(row["final_accuracy"]),
                delta=100.0 * row["accuracy_delta"],
                ctw=row["correct_to_wrong"],
                wtc=row["wrong_to_correct"],
                attempt=format_percent(row["repair_attempt_rate"]),
                accepted=format_percent(row["accepted_repair_rate"]),
                risk=row["measured_risk_reduction_percent"],
            )
        )
    (output_dir / "strategyqa_table.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    proposal_rows = []
    for pool_name, proposals in proposal_maps.items():
        applicable_specs = [spec for spec in SPECS if spec.pool_name == pool_name]
        for key, item in proposals.items():
            record = item["record"]
            proposal = item["proposal"]
            proposal_rows.append(
                {
                    "pool": pool_name,
                    "id": record["id"],
                    "question": record["question"],
                    "gold_answer": record.get("gold_answer"),
                    "classification": item["classification"],
                    "parent_answer": proposal.get("parent_answer"),
                    "candidate_answer": proposal.get("candidate_answer"),
                    "parent_trace": proposal.get("parent_trace"),
                    "candidate_trace": proposal.get("candidate_trace"),
                    "old_avg_risk": proposal.get("old_avg_risk"),
                    "new_avg_risk": proposal.get("new_avg_risk"),
                    "improvement": proposal.get("improvement"),
                    "judge": proposal.get("judge"),
                    "policy_decisions": {
                        spec.label: decisions_by_run[spec.run_name].get(key)
                        for spec in applicable_specs
                    },
                }
            )
    write_jsonl(output_dir / "proposal_analysis.jsonl", proposal_rows)

    print("\n".join(markdown))
    print(f"\nVerified outputs written to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and verify the StrategyQA Table 3 equivalent")
    parser.add_argument("--runs-root", type=Path, default=Path("data/runs"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/runs/strategyqa_100_results"),
    )
    args = parser.parse_args()
    build_outputs(args.runs_root, args.output_dir)


if __name__ == "__main__":
    main()
