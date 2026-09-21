from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acceptance import SUPPORTED_ACCEPTANCE_MODES
from src.candidate_pool import apply_candidate_pool_record
from src.io_utils import read_jsonl
from src.task_profiles import answers_match


METRICS_WITH_INTERVALS = (
    "final_accuracy",
    "accuracy_delta",
    "wrong_to_correct_rate",
    "correct_to_wrong_rate",
    "accepted_record_rate",
    "acceptance_rate_among_attempted",
    "mean_risk_reduction",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_protocol(path: Path) -> Dict[str, Any]:
    require(path.exists(), f"Missing protocol: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "Protocol must be a YAML object")
    return value


def load_candidate_pools(paths: Sequence[Path], dataset: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen_ids = set()
    scoring_signature: Optional[str] = None
    for path in paths:
        require(path.exists(), f"Missing candidate pool: {path}")
        for record in read_jsonl(path):
            record_id = str(record.get("id"))
            require(record_id not in seen_ids, f"Duplicate {dataset} record id: {record_id}")
            seen_ids.add(record_id)
            require(record.get("gold_answer") is not None, f"{record_id} has no gold answer")
            pool = record.get("candidate_pool")
            require(isinstance(pool, dict), f"{record_id} has no candidate_pool object")
            require(
                pool.get("candidate_generation_mode") == "suffix_repair",
                f"{record_id} is not a suffix-repair candidate pool",
            )
            require(pool.get("max_iters") == 1, f"{record_id} is not a one-shot pool")
            proposals = pool.get("proposals", [])
            require(isinstance(proposals, list), f"{record_id} has invalid proposals")
            require(len(proposals) <= 1, f"{record_id} contains more than one proposal")
            signature = json.dumps(pool.get("scoring", {}), sort_keys=True)
            if scoring_signature is None:
                scoring_signature = signature
            require(
                signature == scoring_signature,
                f"{dataset} candidate pools do not share one scoring configuration",
            )
            records.append(record)
    require(bool(records), f"No records loaded for {dataset}")
    return records


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else 0.0


def record_result(
    dataset: str,
    source: Mapping[str, Any],
    applied: Mapping[str, Any],
) -> Dict[str, Any]:
    task = source.get("task")
    gold = source.get("gold_answer")
    original_correct = answers_match(task, source.get("model_answer"), gold)
    final_correct = answers_match(task, applied.get("final_answer"), gold)
    if not original_correct and final_correct:
        outcome = "wrong_to_correct"
    elif original_correct and not final_correct:
        outcome = "correct_to_wrong"
    elif original_correct:
        outcome = "correct_to_correct"
    else:
        outcome = "wrong_to_wrong"

    pool = source["candidate_pool"]
    original_risks = pool["original_score"].get("risks", [])
    final_risks = applied.get("final_risks", [])
    iterations = applied.get("logs", {}).get("iterative_repair", {}).get("iterations", [])
    repair_iterations = [item for item in iterations if item.get("kind") == "repair"]
    accepted_iterations = [item for item in repair_iterations if item.get("accepted") is True]
    acceptance_reason = None
    if repair_iterations:
        acceptance_reason = repair_iterations[-1].get("acceptance", {}).get("reason")

    return {
        "dataset": dataset,
        "id": str(source.get("id")),
        "original_answer": source.get("model_answer"),
        "final_answer": applied.get("final_answer"),
        "original_correct": original_correct,
        "final_correct": final_correct,
        "outcome": outcome,
        "attempted": bool(repair_iterations),
        "accepted": bool(accepted_iterations),
        "num_attempts": len(repair_iterations),
        "num_accepted": len(accepted_iterations),
        "acceptance_reason": acceptance_reason,
        "original_avg_risk": mean(float(value) for value in original_risks),
        "final_avg_risk": mean(float(value) for value in final_risks),
        "risk_reduction": (
            mean(float(value) for value in original_risks)
            - mean(float(value) for value in final_risks)
        ),
        "stop_reason": applied.get("logs", {})
        .get("iterative_repair", {})
        .get("stop_reason"),
    }


def apply_policies(
    datasets: Mapping[str, Sequence[Mapping[str, Any]]],
    policies: Sequence[str],
    acceptance: Mapping[str, Any],
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for dataset, records in datasets.items():
        results[dataset] = {}
        for policy in policies:
            rows = []
            for source in records:
                applied = apply_candidate_pool_record(
                    dict(source),
                    mode=policy,
                    support_tolerance=float(acceptance["support_tolerance"]),
                    max_regression_risk=float(acceptance["max_regression_risk"]),
                    improvement_threshold=float(acceptance["improvement_threshold"]),
                )
                rows.append(record_result(dataset, source, applied))
            results[dataset][policy] = rows
    return results


def calculate_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    original_correct = sum(bool(row["original_correct"]) for row in rows)
    final_correct = sum(bool(row["final_correct"]) for row in rows)
    originally_wrong = n - original_correct
    wrong_to_correct = sum(row["outcome"] == "wrong_to_correct" for row in rows)
    correct_to_wrong = sum(row["outcome"] == "correct_to_wrong" for row in rows)
    attempted = sum(bool(row["attempted"]) for row in rows)
    accepted = sum(bool(row["accepted"]) for row in rows)
    return {
        "n": n,
        "original_correct": original_correct,
        "final_correct": final_correct,
        "original_accuracy": original_correct / n if n else None,
        "final_accuracy": final_correct / n if n else None,
        "accuracy_delta": (final_correct - original_correct) / n if n else None,
        "originally_wrong": originally_wrong,
        "wrong_to_correct": wrong_to_correct,
        "wrong_to_correct_rate": (
            wrong_to_correct / originally_wrong if originally_wrong else None
        ),
        "originally_correct": original_correct,
        "correct_to_wrong": correct_to_wrong,
        "correct_to_wrong_rate": (
            correct_to_wrong / original_correct if original_correct else None
        ),
        "attempted_records": attempted,
        "accepted_records": accepted,
        "accepted_record_rate": accepted / n if n else None,
        "acceptance_rate_among_attempted": accepted / attempted if attempted else None,
        "mean_risk_reduction": mean(float(row["risk_reduction"]) for row in rows),
        "stop_reasons": dict(Counter(str(row["stop_reason"]) for row in rows)),
    }


def percentile(sorted_values: Sequence[float], quantile: float) -> float:
    require(bool(sorted_values), "Cannot calculate a percentile of no values")
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower]
        + fraction * (sorted_values[upper] - sorted_values[lower])
    )


def interval(values: Sequence[float], confidence_level: float) -> Dict[str, float]:
    ordered = sorted(values)
    alpha = 1.0 - confidence_level
    return {
        "lower": percentile(ordered, alpha / 2.0),
        "upper": percentile(ordered, 1.0 - alpha / 2.0),
    }


def aligned_rows(
    results: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    datasets: Sequence[str],
    policies: Sequence[str],
) -> Dict[str, List[Mapping[str, Any]]]:
    aligned: Dict[str, List[Mapping[str, Any]]] = {policy: [] for policy in policies}
    for dataset in datasets:
        expected_ids: Optional[List[str]] = None
        for policy in policies:
            rows = list(results[dataset][policy])
            ids = [str(row["id"]) for row in rows]
            if expected_ids is None:
                expected_ids = ids
            require(ids == expected_ids, f"Policy row mismatch for {dataset}: {policy}")
            aligned[policy].extend(rows)
    return aligned


def paired_stratified_bootstrap(
    policy_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    reference_policy: str,
    resamples: int,
    seed: int,
    confidence_level: float,
) -> Dict[str, Any]:
    policies = list(policy_rows)
    n = len(policy_rows[reference_policy])
    require(all(len(policy_rows[policy]) == n for policy in policies), "Policy lengths differ")
    strata: Dict[Tuple[str, bool], List[int]] = defaultdict(list)
    for index, row in enumerate(policy_rows[reference_policy]):
        strata[(str(row["dataset"]), bool(row["original_correct"]))].append(index)
    require(bool(strata), "Bootstrap requires records")

    samples: Dict[str, Dict[str, List[float]]] = {
        policy: {metric: [] for metric in METRICS_WITH_INTERVALS}
        for policy in policies
    }
    paired_differences: Dict[str, Dict[str, List[float]]] = {
        policy: {metric: [] for metric in METRICS_WITH_INTERVALS}
        for policy in policies
        if policy != reference_policy
    }
    rng = random.Random(seed)
    for _ in range(resamples):
        indices: List[int] = []
        for stratum_indices in strata.values():
            indices.extend(rng.choice(stratum_indices) for _ in stratum_indices)
        replicate = {
            policy: calculate_metrics([policy_rows[policy][index] for index in indices])
            for policy in policies
        }
        for policy, metrics in replicate.items():
            for metric in METRICS_WITH_INTERVALS:
                value = metrics[metric]
                if value is not None:
                    samples[policy][metric].append(float(value))
        reference = replicate[reference_policy]
        for policy, differences in paired_differences.items():
            for metric in METRICS_WITH_INTERVALS:
                value = replicate[policy][metric]
                reference_value = reference[metric]
                if value is not None and reference_value is not None:
                    differences[metric].append(float(value) - float(reference_value))

    return {
        "policy_intervals": {
            policy: {
                metric: interval(values, confidence_level)
                for metric, values in metrics.items()
                if values
            }
            for policy, metrics in samples.items()
        },
        "paired_differences_from_reference": {
            policy: {
                metric: interval(values, confidence_level)
                for metric, values in metrics.items()
                if values
            }
            for policy, metrics in paired_differences.items()
        },
        "stratum_sizes": {
            f"{dataset}|original_{'correct' if correctness else 'wrong'}": len(indices)
            for (dataset, correctness), indices in strata.items()
        },
    }


def utility_table(
    metrics_by_policy: Mapping[str, Mapping[str, Any]], penalties: Sequence[float]
) -> Dict[str, Dict[str, float]]:
    return {
        policy: {
            str(penalty): float(metrics["wrong_to_correct"])
            - float(penalty) * float(metrics["correct_to_wrong"])
            for penalty in penalties
        }
        for policy, metrics in metrics_by_policy.items()
    }


def fmt_pct(value: Optional[float]) -> str:
    return "NA" if value is None else f"{100.0 * value:.1f}%"


def fmt_ci(interval_value: Optional[Mapping[str, float]], *, percent: bool = True) -> str:
    if interval_value is None:
        return "NA"
    factor = 100.0 if percent else 1.0
    return f"[{factor * interval_value['lower']:.1f}, {factor * interval_value['upper']:.1f}]"


def fmt_signed_ci(interval_value: Optional[Mapping[str, float]]) -> str:
    if interval_value is None:
        return "NA"
    return f"[{100.0 * interval_value['lower']:+.1f}, {100.0 * interval_value['upper']:+.1f}]"


def markdown_report(report: Mapping[str, Any]) -> str:
    bootstrap = report["bootstrap"]
    lines = [
        "# Category 3: acceptance-policy sensitivity analysis",
        "",
        "All policies were replayed offline on the same frozen, one-shot suffix-repair candidates. No model, NLI, or provider calls were made. The oracle guard uses gold labels and is diagnostic only.",
        "",
        "Intervals are {level:.0f}% stratified nonparametric percentile-bootstrap intervals ({resamples:,} paired resamples; fixed seed {seed}). Exact transition counts remain primary because some events are rare.".format(
            level=100.0 * float(bootstrap["confidence_level"]),
            resamples=int(bootstrap["resamples"]),
            seed=int(bootstrap["seed"]),
        ),
        "",
    ]
    labels = report["policy_labels"]
    for dataset, section in report["datasets"].items():
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| Policy | Final accuracy (95% CI) | Delta | W->C | C->W | Accepted / attempted | Mean risk reduction |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for policy, metrics in section["policies"].items():
            ci = section["bootstrap"]["policy_intervals"][policy]
            lines.append(
                "| {label} | {accuracy} {accuracy_ci} | {delta:+.1f} pp {delta_ci} | "
                "{wtc}/{wrong} ({wtc_rate}) | {ctw}/{correct} ({ctw_rate}) | "
                "{accepted}/{attempted} | {risk:.4f} |".format(
                    label=labels[policy],
                    accuracy=fmt_pct(metrics["final_accuracy"]),
                    accuracy_ci=fmt_ci(ci.get("final_accuracy")),
                    delta=100.0 * metrics["accuracy_delta"],
                    delta_ci=fmt_ci(ci.get("accuracy_delta")),
                    wtc=metrics["wrong_to_correct"],
                    wrong=metrics["originally_wrong"],
                    wtc_rate=fmt_pct(metrics["wrong_to_correct_rate"]),
                    ctw=metrics["correct_to_wrong"],
                    correct=metrics["originally_correct"],
                    ctw_rate=fmt_pct(metrics["correct_to_wrong_rate"]),
                    accepted=metrics["accepted_records"],
                    attempted=metrics["attempted_records"],
                    risk=metrics["mean_risk_reduction"],
                )
            )
        lines.extend(["", "Utility sensitivity: `U(lambda) = W->C - lambda * C->W`.", ""])
        penalties = report["utility_penalties"]
        lines.append(
            "| Policy | " + " | ".join(f"lambda={penalty:g}" for penalty in penalties) + " |"
        )
        lines.append("|---|" + "---:|" * len(penalties))
        for policy, values in section["utility"].items():
            lines.append(
                f"| {labels[policy]} | "
                + " | ".join(f"{values[str(penalty)]:g}" for penalty in penalties)
                + " |"
            )
        lines.extend(
            [
                "",
                f"Paired differences relative to {labels[report['reference_policy']]} (policy minus reference):",
                "",
                "| Policy | Final accuracy difference | W->C rate difference | C->W rate difference | Accepted-record-rate difference | Risk-reduction difference |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for policy, intervals in section["bootstrap"][
            "paired_differences_from_reference"
        ].items():
            policy_metrics = section["policies"][policy]
            reference_metrics = section["policies"][report["reference_policy"]]
            lines.append(
                "| {label} | {accuracy:+.1f} pp {accuracy_ci} | {wtc:+.1f} pp {wtc_ci} | "
                "{ctw:+.1f} pp {ctw_ci} | {accepted:+.1f} pp {accepted_ci} | "
                "{risk:+.4f} [{risk_low:+.4f}, {risk_high:+.4f}] |".format(
                    label=labels[policy],
                    accuracy=100.0
                    * (policy_metrics["final_accuracy"] - reference_metrics["final_accuracy"]),
                    accuracy_ci=fmt_signed_ci(intervals.get("final_accuracy")),
                    wtc=100.0
                    * (
                        (policy_metrics["wrong_to_correct_rate"] or 0.0)
                        - (reference_metrics["wrong_to_correct_rate"] or 0.0)
                    ),
                    wtc_ci=fmt_signed_ci(intervals.get("wrong_to_correct_rate")),
                    ctw=100.0
                    * (
                        (policy_metrics["correct_to_wrong_rate"] or 0.0)
                        - (reference_metrics["correct_to_wrong_rate"] or 0.0)
                    ),
                    ctw_ci=fmt_signed_ci(intervals.get("correct_to_wrong_rate")),
                    accepted=100.0
                    * (
                        policy_metrics["accepted_record_rate"]
                        - reference_metrics["accepted_record_rate"]
                    ),
                    accepted_ci=fmt_signed_ci(intervals.get("accepted_record_rate")),
                    risk=policy_metrics["mean_risk_reduction"]
                    - reference_metrics["mean_risk_reduction"],
                    risk_low=intervals["mean_risk_reduction"]["lower"],
                    risk_high=intervals["mean_risk_reduction"]["upper"],
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation guardrails",
            "",
            "- `risk_only` here is an acceptance rule under one fixed detector and candidate pool. It is not the paper's old aggressive operating point, which changed candidate opportunities as well as acceptance.",
            "- `oracle_guard` is not deployable and should not be described as a production baseline or as a universal accuracy upper bound.",
            "- A policy is not declared universally best. The tables expose the safety-utility tradeoff, and the paired intervals show how uncertain the small observed differences remain.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_report(protocol: Mapping[str, Any], project_root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    policies = [str(policy) for policy in protocol["policies"]]
    require(len(policies) == len(set(policies)), "Policies must be unique")
    require(
        all(policy in SUPPORTED_ACCEPTANCE_MODES for policy in policies),
        "Protocol includes an unsupported acceptance policy",
    )
    reference_policy = str(protocol["reference_policy"])
    require(reference_policy in policies, "Reference policy must be included in policies")
    bootstrap_cfg = protocol["bootstrap"]
    require(bootstrap_cfg.get("paired_policy_comparisons") is True, "Bootstrap must be paired")
    require(
        bootstrap_cfg.get("strata") == "dataset_and_original_answer_correctness",
        "Unexpected bootstrap strata",
    )

    datasets: Dict[str, List[Dict[str, Any]]] = {}
    source_paths: Dict[str, List[str]] = {}
    for dataset, dataset_cfg in protocol["datasets"].items():
        configured_paths = [Path(value) for value in dataset_cfg["candidate_pools"]]
        resolved_paths = [path if path.is_absolute() else project_root / path for path in configured_paths]
        datasets[str(dataset)] = load_candidate_pools(resolved_paths, str(dataset))
        source_paths[str(dataset)] = [str(path) for path in configured_paths]

    results = apply_policies(datasets, policies, protocol["acceptance"])
    report: Dict[str, Any] = {
        "analysis_name": protocol["analysis_name"],
        "source_candidate_pools": source_paths,
        "policy_labels": {
            "risk_only": "Risk-only",
            "strict_judge_guard": "Strict judge guard",
            "judge_guard": "Updated judge guard",
            "oracle_guard": "Oracle guard (diagnostic)",
        },
        "reference_policy": reference_policy,
        "acceptance": dict(protocol["acceptance"]),
        "bootstrap": dict(bootstrap_cfg),
        "utility_definition": protocol["utility"]["definition"],
        "utility_penalties": [float(value) for value in protocol["utility"]["regression_penalties"]],
        "datasets": {},
        "audit_records": results,
    }

    analysis_groups = {dataset: [dataset] for dataset in datasets}
    analysis_groups["pooled_descriptive"] = list(datasets)
    for label, included_datasets in analysis_groups.items():
        aligned = aligned_rows(results, included_datasets, policies)
        policy_metrics = {
            policy: calculate_metrics(rows) for policy, rows in aligned.items()
        }
        bootstrap = paired_stratified_bootstrap(
            aligned,
            reference_policy=reference_policy,
            resamples=int(bootstrap_cfg["resamples"]),
            seed=int(bootstrap_cfg["seed"]),
            confidence_level=float(bootstrap_cfg["confidence_level"]),
        )
        report["datasets"][label] = {
            "included_datasets": included_datasets,
            "policies": policy_metrics,
            "utility": utility_table(policy_metrics, report["utility_penalties"]),
            "bootstrap": bootstrap,
            "pooled_results_are_descriptive": label == "pooled_descriptive",
        }
    return report


def write_report(protocol_path: Path, bootstrap_resamples: Optional[int] = None) -> Dict[str, Any]:
    protocol = load_protocol(protocol_path)
    if bootstrap_resamples is not None:
        require(bootstrap_resamples > 0, "Bootstrap resamples must be positive")
        protocol["bootstrap"]["resamples"] = bootstrap_resamples
    report = build_report(protocol)
    output_dir = Path(protocol["output_dir"])
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "acceptance_policy_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    markdown = markdown_report(report)
    (output_dir / "acceptance_policy_report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Reports written to {output_dir}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay acceptance policies offline on frozen one-shot candidate pools"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/acceptance_policy_analysis.yaml"),
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        help="Override the configured count (useful for fast tests only)",
    )
    args = parser.parse_args()
    write_report(args.config, bootstrap_resamples=args.bootstrap_resamples)


if __name__ == "__main__":
    main()
