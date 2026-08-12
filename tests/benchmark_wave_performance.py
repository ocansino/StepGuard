from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Dict, List
from unittest.mock import patch

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.cli import build_candidate_pool
from tests.test_wave_candidate_pool import FakeNLIScorer, FakeWaveClient


DEFAULT_CONCURRENCY_LEVELS = (1, 2, 3, 4)


class TimedFakeWaveClient(FakeWaveClient):
    """Deterministic fake provider with controlled network/model latency."""

    def judge_steps(self, *args: Any, **kwargs: Any):
        sleep(0.050)
        return super().judge_steps(*args, **kwargs)

    def repair_suffix(self, *args: Any, **kwargs: Any):
        sleep(0.075)
        return super().repair_suffix(*args, **kwargs)

    def judge_repair_candidate(self, *args: Any, **kwargs: Any):
        sleep(0.050)
        return super().judge_repair_candidate(*args, **kwargs)


def make_records(count: int) -> List[Dict[str, Any]]:
    return [
        {
            "id": f"benchmark-q{index:03d}",
            "question": f"Is benchmark statement {index} true?",
            "model_trace": (
                f"Step 1: Original reasoning for benchmark item {index}.\n"
                "Final answer: no"
            ),
            "model_answer": "no",
            "gold_answer": "yes",
            "task": "strategyqa",
            "source": "offline-wave-benchmark",
        }
        for index in range(1, count + 1)
    ]


def write_config(
    *,
    path: Path,
    run_name: str,
    dataset_path: Path,
    output_dir: Path,
    max_concurrency: int,
) -> None:
    config = {
        "run_name": run_name,
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "task_profile": "strategyqa",
        "model": {
            "provider": "openai",
            "name": "offline-fake-model",
            "temperature": 0.0,
            "max_output_tokens": 100,
        },
        "execution": {
            "metrics_enabled": True,
            "max_concurrency": max_concurrency,
            "request_timeout_seconds": 45.0,
            "max_retries": 1,
            "retry_base_delay_seconds": 0.5,
        },
        "scoring": {
            "risk_threshold": 0.2,
            "improvement_threshold": 0.02,
            "max_iters": 1,
            "risk_formula": "weighted",
            "verifier_weight": 0.75,
            "contradiction_weight": 0.25,
        },
        "repair_acceptance": {
            "mode": "risk_only",
            "support_tolerance": 0.05,
            "max_regression_risk": 0.35,
        },
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stage_seconds(metrics: Dict[str, Any], name: str) -> float:
    return float(metrics["stages"].get(name, {}).get("elapsed_seconds", 0.0))


def run_once(
    *,
    temp_path: Path,
    dataset_path: Path,
    output_dir: Path,
    record_count: int,
    concurrency: int,
    repetition: int,
) -> Dict[str, Any]:
    run_name = f"wave-benchmark-c{concurrency}-r{repetition}"
    config_path = temp_path / f"{run_name}.yaml"
    write_config(
        path=config_path,
        run_name=run_name,
        dataset_path=dataset_path,
        output_dir=output_dir,
        max_concurrency=concurrency,
    )

    def make_client(*, model, metrics, **execution_options):
        return TimedFakeWaveClient(model=model, metrics=metrics)

    started = perf_counter()
    with (
        patch(
            "src.cli.OpenAIClientWrapper",
            side_effect=make_client,
        ),
        patch("src.cli.NLIScorer", FakeNLIScorer),
        patch(
            "src.providers.openai_client.OpenAI",
            side_effect=AssertionError(
                "Offline benchmark attempted to construct a real OpenAI client"
            ),
        ),
    ):
        build_candidate_pool(
            config=str(config_path),
            input_path=str(dataset_path),
            output_path=None,
            reuse_original_scores=False,
            resume=False,
        )
    measured_wall_seconds = perf_counter() - started

    run_dir = output_dir / run_name
    artifact = read_jsonl(run_dir / "candidate_pool.jsonl")
    metrics = json.loads(
        (run_dir / "metrics.build_candidate_pool.json").read_text(
            encoding="utf-8"
        )
    )

    operations = metrics["operations"]
    verification_calls = int(
        operations["openai.step_verification"]["logical_calls"]
    )
    repair_calls = int(
        operations["openai.suffix_repair"]["logical_calls"]
    )
    acceptance_calls = int(
        operations["openai.acceptance_judging"]["logical_calls"]
    )
    logical_calls = verification_calls + repair_calls + acceptance_calls

    expected_counts = {
        "verification_calls": record_count * 2,
        "repair_calls": record_count,
        "acceptance_calls": record_count,
        "logical_calls": record_count * 4,
    }
    actual_counts = {
        "verification_calls": verification_calls,
        "repair_calls": repair_calls,
        "acceptance_calls": acceptance_calls,
        "logical_calls": logical_calls,
    }
    if actual_counts != expected_counts:
        raise RuntimeError(
            "Benchmark workload changed: "
            f"expected {expected_counts}, observed {actual_counts}"
        )

    if len(artifact) != record_count:
        raise RuntimeError(
            f"Expected {record_count} records, observed {len(artifact)}"
        )
    if not all(
        len(record["candidate_pool"]["proposals"]) == 1
        for record in artifact
    ):
        raise RuntimeError("Every benchmark record must have one proposal")

    return {
        "concurrency": concurrency,
        "repetition": repetition,
        "wall_seconds": measured_wall_seconds,
        "metrics_wall_seconds": float(metrics["wall_time_seconds"]),
        "records_per_second": record_count / measured_wall_seconds,
        **actual_counts,
        "stage_seconds": {
            "original_verification": stage_seconds(
                metrics,
                "original_verification_wave",
            ),
            "repair": stage_seconds(metrics, "repair_wave"),
            "candidate_verification": stage_seconds(
                metrics,
                "candidate_verification_wave",
            ),
            "acceptance": stage_seconds(metrics, "acceptance_wave"),
            "checkpoint_write": stage_seconds(
                metrics,
                "checkpoint_write",
            ),
            "artifact_write": stage_seconds(metrics, "artifact_write"),
        },
        "artifact": artifact,
    }


def median_result(
    runs: List[Dict[str, Any]],
    *,
    baseline_wall_seconds: float,
) -> Dict[str, Any]:
    wall_seconds = statistics.median(run["wall_seconds"] for run in runs)
    metrics_wall_seconds = statistics.median(
        run["metrics_wall_seconds"] for run in runs
    )
    stage_names = runs[0]["stage_seconds"]
    stages = {
        name: statistics.median(
            run["stage_seconds"][name]
            for run in runs
        )
        for name in stage_names
    }

    return {
        "concurrency": runs[0]["concurrency"],
        "repetitions": len(runs),
        "median_wall_seconds": wall_seconds,
        "median_metrics_wall_seconds": metrics_wall_seconds,
        "speedup_vs_concurrency_1": baseline_wall_seconds / wall_seconds,
        "median_records_per_second": statistics.median(
            run["records_per_second"] for run in runs
        ),
        "logical_calls_per_run": runs[0]["logical_calls"],
        "median_stage_seconds": stages,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline deterministic StepGuard wave benchmark"
    )
    parser.add_argument("--records", type=int, default=24)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONCURRENCY_LEVELS),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.records < 1:
        raise ValueError("--records must be at least 1")
    if args.repetitions < 1:
        raise ValueError("--repetitions must be at least 1")
    if 1 not in args.concurrency:
        raise ValueError("--concurrency must include 1 as the baseline")
    if any(level < 1 or level > 32 for level in args.concurrency):
        raise ValueError("concurrency levels must be between 1 and 32")

    concurrency_levels = sorted(set(args.concurrency))
    all_runs: Dict[int, List[Dict[str, Any]]] = {
        level: [] for level in concurrency_levels
    }
    reference_artifact = None

    with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
        temp_path = Path(temp_dir)
        dataset_path = temp_path / "generated.jsonl"
        output_dir = temp_path / "runs"
        records = make_records(args.records)
        dataset_path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )

        for repetition in range(1, args.repetitions + 1):
            for concurrency in concurrency_levels:
                run = run_once(
                    temp_path=temp_path,
                    dataset_path=dataset_path,
                    output_dir=output_dir,
                    record_count=args.records,
                    concurrency=concurrency,
                    repetition=repetition,
                )

                artifact = run.pop("artifact")
                if reference_artifact is None:
                    reference_artifact = artifact
                elif artifact != reference_artifact:
                    raise RuntimeError(
                        "Candidate-pool artifacts differ across benchmark runs"
                    )

                all_runs[concurrency].append(run)

    baseline_wall_seconds = statistics.median(
        run["wall_seconds"] for run in all_runs[1]
    )
    summaries = [
        median_result(
            all_runs[level],
            baseline_wall_seconds=baseline_wall_seconds,
        )
        for level in concurrency_levels
    ]

    report = {
        "benchmark": "stepguard_offline_wave_performance",
        "records_per_run": args.records,
        "repetitions": args.repetitions,
        "concurrency_levels": concurrency_levels,
        "expected_logical_calls_per_run": args.records * 4,
        "artifact_equivalence": "PASS",
        "network_access": "BLOCKED_BY_TEST_GUARD",
        "results": summaries,
    }

    print()
    print("Concurrency | Median wall | Speedup | Records/sec | Calls")
    print("------------|-------------|---------|-------------|------")
    for result in summaries:
        print(
            f"{result['concurrency']:>11} | "
            f"{result['median_wall_seconds']:>9.3f}s | "
            f"{result['speedup_vs_concurrency_1']:>7.2f}x | "
            f"{result['median_records_per_second']:>11.2f} | "
            f"{result['logical_calls_per_run']:>5}"
        )

    print()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
