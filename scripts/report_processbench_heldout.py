from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import read_json, read_jsonl, write_json_atomic
from src.risk import compute_risks

from scripts.analyze_processbench_detector import _binary_metrics, evaluate_detector
from scripts.processbench import evaluate_labels
from scripts.processbench_runner import GOLD_FIELDS, PREDICTION_SCHEMA_VERSION


REPORT_SCHEMA_VERSION = 1
PLAN_SCHEMA_VERSION = 1
RECEIPT_SCHEMA_VERSION = 1
DETECTOR_NAMES = ("primary_submitted", "development_selected")
FORBIDDEN_SEARCH_KEYS = {
    "weight_grid",
    "threshold_grid",
    "evaluated_configurations",
    "top_10",
}

LOCALIZATION_METRICS = {
    "official_processbench_f1": (
        "official_processbench",
        "f1",
    ),
    "error_localization_accuracy": (
        "official_processbench",
        "error_localization_accuracy",
    ),
    "correct_trace_accuracy": (
        "official_processbench",
        "correct_trace_accuracy",
    ),
    "overall_exact_label_accuracy": (
        "additional_metrics",
        "overall_exact_label_accuracy",
    ),
    "within_one_localization_accuracy_on_error_traces": (
        "additional_metrics",
        "within_one_localization_accuracy_on_error_traces",
    ),
    "trace_error_detection_precision": (
        "additional_metrics",
        "trace_error_detection",
        "precision",
    ),
    "trace_error_detection_recall": (
        "additional_metrics",
        "trace_error_detection",
        "recall",
    ),
    "trace_error_detection_f1": (
        "additional_metrics",
        "trace_error_detection",
        "f1",
    ),
    "trace_error_detection_false_positive_rate": (
        "additional_metrics",
        "trace_error_detection",
        "false_positive_rate",
    ),
}

RISK_METRICS = (
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "roc_auc",
    "average_precision",
    "brier_score",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record["id"]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _find_forbidden_search_key(value: Any) -> Optional[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in FORBIDDEN_SEARCH_KEYS:
                return str(key)
            found = _find_forbidden_search_key(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_forbidden_search_key(child)
            if found is not None:
                return found
    return None


def _validate_detector_config(name: str, config: Mapping[str, Any]) -> None:
    scoring = config.get("scoring")
    threshold = config.get("risk_threshold")
    _require(isinstance(scoring, Mapping), f"{name}: scoring must be an object")
    _require(
        isinstance(threshold, (int, float)) and not isinstance(threshold, bool),
        f"{name}: risk_threshold must be numeric",
    )
    _require(0 <= float(threshold) <= 1, f"{name}: risk_threshold is outside [0, 1]")
    formula = scoring.get("risk_formula")
    _require(
        formula in {"weighted", "verifier_only", "contradiction_only"},
        f"{name}: unsupported risk formula {formula!r}",
    )
    if formula == "weighted":
        verifier_weight = scoring.get("verifier_weight")
        contradiction_weight = scoring.get("contradiction_weight")
        _require(
            isinstance(verifier_weight, (int, float))
            and not isinstance(verifier_weight, bool)
            and isinstance(contradiction_weight, (int, float))
            and not isinstance(contradiction_weight, bool),
            f"{name}: weighted scoring requires numeric weights",
        )
        _require(
            math.isclose(
                float(verifier_weight) + float(contradiction_weight),
                1.0,
                abs_tol=1e-12,
            ),
            f"{name}: detector weights must sum to 1",
        )


def validate_frozen_plan(plan: Mapping[str, Any]) -> Dict[str, Any]:
    _require(
        plan.get("schema_version") == PLAN_SCHEMA_VERSION,
        f"Unsupported frozen-plan schema: {plan.get('schema_version')!r}",
    )
    _require(plan.get("status") == "frozen", "Detector plan is not frozen")
    _require(
        plan.get("selection_source") == "development_only",
        "Detector selection must come from development only",
    )
    forbidden = _find_forbidden_search_key(plan)
    _require(
        forbidden is None,
        f"Frozen held-out plan contains forbidden search key {forbidden!r}",
    )
    detectors = plan.get("detectors")
    _require(isinstance(detectors, Mapping), "Frozen plan is missing detectors")
    _require(
        tuple(detectors.keys()) == DETECTOR_NAMES,
        f"Frozen plan must contain detectors in order {DETECTOR_NAMES!r}",
    )
    for name in DETECTOR_NAMES:
        _validate_detector_config(name, detectors[name])

    partition = plan.get("heldout_partition")
    _require(isinstance(partition, Mapping), "Frozen plan is missing heldout_partition")
    _require(
        isinstance(partition.get("records"), int)
        and not isinstance(partition.get("records"), bool)
        and partition["records"] > 0,
        "heldout_partition.records must be a positive integer",
    )
    _require(
        isinstance(partition.get("labels_sha256"), str)
        and len(partition["labels_sha256"]) == 64,
        "heldout_partition.labels_sha256 must be present",
    )

    bootstrap = plan.get("bootstrap")
    _require(isinstance(bootstrap, Mapping), "Frozen plan is missing bootstrap settings")
    _require(
        isinstance(bootstrap.get("resamples"), int)
        and not isinstance(bootstrap.get("resamples"), bool)
        and bootstrap["resamples"] > 0,
        "bootstrap.resamples must be a positive integer",
    )
    _require(
        isinstance(bootstrap.get("seed"), int)
        and not isinstance(bootstrap.get("seed"), bool),
        "bootstrap.seed must be an integer",
    )
    level = bootstrap.get("confidence_level")
    _require(
        isinstance(level, (int, float))
        and not isinstance(level, bool)
        and 0 < float(level) < 1,
        "bootstrap.confidence_level must be between 0 and 1",
    )
    _require(
        bootstrap.get("strata") == "gold_error_vs_correct_trace",
        "Bootstrap strata must be gold_error_vs_correct_trace",
    )
    _require(
        bootstrap.get("paired_detector_comparison") is True,
        "Detector comparison must use paired resampling",
    )
    return dict(plan)


def _config_identity(config: Mapping[str, Any]) -> Dict[str, Any]:
    scoring = config["scoring"]
    result = {
        "risk_formula": scoring["risk_formula"],
        "risk_threshold": float(config["risk_threshold"]),
    }
    if scoring["risk_formula"] == "weighted":
        result["verifier_weight"] = float(scoring["verifier_weight"])
        result["contradiction_weight"] = float(scoring["contradiction_weight"])
    return result


def _prediction_identity(record: Mapping[str, Any]) -> Dict[str, Any]:
    identity = record.get("detector_identity")
    _require(isinstance(identity, Mapping), f"{record.get('id')}: missing detector identity")
    scoring = identity.get("scoring")
    _require(isinstance(scoring, Mapping), f"{record.get('id')}: missing scoring identity")
    config = {
        "scoring": dict(scoring),
        "risk_threshold": identity.get("risk_threshold"),
    }
    return _config_identity(config)


def validate_blind_predictions(
    predictions: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> None:
    _require(bool(predictions), "Blind prediction file is empty")
    expected_records = int(plan["heldout_partition"]["records"])
    _require(
        len(predictions) == expected_records,
        f"Found {len(predictions)} predictions; expected {expected_records}",
    )
    expected_identity = _config_identity(plan["detectors"]["primary_submitted"])
    seen = set()
    for index, record in enumerate(predictions):
        record_id = record.get("id")
        _require(isinstance(record_id, str) and record_id, f"Row {index}: missing id")
        _require(record_id not in seen, f"Duplicate prediction id: {record_id}")
        seen.add(record_id)
        _require(
            record.get("prediction_schema_version") == PREDICTION_SCHEMA_VERSION,
            f"{record_id}: unsupported prediction schema",
        )
        _require(
            not GOLD_FIELDS.intersection(record),
            f"{record_id}: prediction file is not blinded",
        )
        _require(
            _prediction_identity(record) == expected_identity,
            f"{record_id}: prediction detector differs from frozen primary detector",
        )
        score = record.get("stepguard_score")
        _require(isinstance(score, Mapping), f"{record_id}: missing StepGuard score")
        steps = score.get("steps")
        scores = score.get("scores")
        judgments = score.get("raw_verifier_judgments")
        _require(isinstance(steps, list) and steps, f"{record_id}: missing scored steps")
        _require(isinstance(scores, Mapping), f"{record_id}: missing score arrays")
        _require(
            isinstance(scores.get("verifier"), list)
            and isinstance(scores.get("contradiction"), list)
            and len(scores["verifier"]) == len(steps)
            and len(scores["contradiction"]) == len(steps),
            f"{record_id}: score arrays do not align",
        )
        _require(
            isinstance(judgments, list) and len(judgments) == len(steps),
            f"{record_id}: incomplete verifier audit",
        )


def freeze_prediction_receipt(
    *,
    predictions_path: Path,
    plan_path: Path,
) -> Dict[str, Any]:
    plan = validate_frozen_plan(read_json(plan_path))
    predictions = list(read_jsonl(predictions_path))
    validate_blind_predictions(predictions, plan)
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "artifact": "processbench_blind_prediction_freeze",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "unblinded": False,
        "plan": {
            "path": str(plan_path),
            "sha256": sha256_file(plan_path),
        },
        "predictions": {
            "path": str(predictions_path),
            "sha256": sha256_file(predictions_path),
            "ids_sha256": ids_sha256(predictions),
            "records": len(predictions),
            "prediction_detector": "primary_submitted",
        },
        "heldout_partition": dict(plan["heldout_partition"]),
    }


def _load_and_validate_receipt(
    *,
    receipt_path: Path,
    plan_path: Path,
    predictions_path: Path,
    predictions: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    receipt = read_json(receipt_path)
    _require(
        receipt.get("schema_version") == RECEIPT_SCHEMA_VERSION,
        "Unsupported prediction-freeze receipt schema",
    )
    _require(receipt.get("unblinded") is False, "Prediction receipt is not pre-unblinding")
    _require(
        receipt.get("plan", {}).get("sha256") == sha256_file(plan_path),
        "Frozen detector plan hash differs from prediction receipt",
    )
    frozen_predictions = receipt.get("predictions", {})
    _require(
        frozen_predictions.get("sha256") == sha256_file(predictions_path),
        "Prediction file hash differs from prediction receipt",
    )
    _require(
        frozen_predictions.get("ids_sha256") == ids_sha256(predictions),
        "Prediction ID hash differs from prediction receipt",
    )
    _require(
        frozen_predictions.get("records") == len(predictions),
        "Prediction record count differs from prediction receipt",
    )
    return receipt


def join_blind_labels(
    predictions: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    _require(
        len(predictions) == len(labels),
        "Blind prediction and label counts differ",
    )
    evaluated = []
    for index, (prediction, label) in enumerate(zip(predictions, labels)):
        _require(
            prediction.get("id") == label.get("id"),
            f"Blind prediction/label id mismatch at index {index}",
        )
        _require(
            set(label).issubset(
                {"id", "gold_label", "gold_earliest_bad_step", "final_answer_correct"}
            ),
            f"{label.get('id')}: unexpected held-out label fields",
        )
        gold = label.get("gold_label")
        _require(
            isinstance(gold, int) and not isinstance(gold, bool),
            f"{label.get('id')}: invalid held-out gold label",
        )
        evaluated.append({**dict(prediction), "gold_label": gold})
    return evaluated


def _nested_value(value: Mapping[str, Any], path: Sequence[str]) -> Optional[float]:
    current: Any = value
    for key in path:
        current = current[key]
    return current


def _flat_metrics(
    localization: Mapping[str, Any],
    trace_metrics: Mapping[str, Any],
    step_metrics: Mapping[str, Any],
) -> Dict[str, Optional[float]]:
    flat = {
        name: _nested_value(localization, path)
        for name, path in LOCALIZATION_METRICS.items()
    }
    for name in RISK_METRICS:
        flat[f"trace_{name}"] = trace_metrics[name]
        flat[f"step_{name}"] = step_metrics[name]
    return flat


def _detector_view(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    gold_labels = []
    predicted_labels = []
    trace_scores = []
    per_record_step_labels = []
    per_record_step_scores = []
    threshold = float(config["risk_threshold"])
    scoring = dict(config["scoring"])
    for record in records:
        gold = int(record["gold_label"])
        scores = record["stepguard_score"]["scores"]
        risks = compute_risks(
            list(scores["verifier"]),
            list(scores["contradiction"]),
            scoring,
        )
        prediction = next(
            (index for index, risk in enumerate(risks) if risk > threshold),
            -1,
        )
        gold_labels.append(gold)
        predicted_labels.append(prediction)
        trace_scores.append(max(risks) if risks else 0.0)
        last_evaluable = len(risks) - 1 if gold == -1 else gold
        per_record_step_labels.append(
            [int(gold != -1 and index == gold) for index in range(last_evaluable + 1)]
        )
        per_record_step_scores.append(risks[: last_evaluable + 1])
    return {
        "threshold": threshold,
        "gold_labels": gold_labels,
        "predicted_labels": predicted_labels,
        "trace_scores": trace_scores,
        "step_labels": per_record_step_labels,
        "step_scores": per_record_step_scores,
    }


def _metrics_for_indices(view: Mapping[str, Any], indices: Sequence[int]) -> Dict[str, Any]:
    gold = [view["gold_labels"][index] for index in indices]
    predicted = [view["predicted_labels"][index] for index in indices]
    trace_labels = [int(label != -1) for label in gold]
    trace_scores = [view["trace_scores"][index] for index in indices]
    step_labels = [
        label
        for index in indices
        for label in view["step_labels"][index]
    ]
    step_scores = [
        score
        for index in indices
        for score in view["step_scores"][index]
    ]
    localization = evaluate_labels(gold, predicted)
    trace = _binary_metrics(trace_labels, trace_scores, view["threshold"])
    step = _binary_metrics(step_labels, step_scores, view["threshold"])
    return _flat_metrics(localization, trace, step)


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    _require(bool(sorted_values), "Cannot compute a percentile of no values")
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


def _interval(values: Sequence[float], confidence_level: float) -> Dict[str, float]:
    ordered = sorted(values)
    alpha = 1 - confidence_level
    return {
        "lower": _percentile(ordered, alpha / 2),
        "upper": _percentile(ordered, 1 - alpha / 2),
    }


def paired_stratified_bootstrap(
    records: Sequence[Mapping[str, Any]],
    detector_configs: Mapping[str, Mapping[str, Any]],
    *,
    resamples: int,
    seed: int,
    confidence_level: float,
) -> Dict[str, Any]:
    error_indices = [
        index for index, record in enumerate(records) if record["gold_label"] != -1
    ]
    correct_indices = [
        index for index, record in enumerate(records) if record["gold_label"] == -1
    ]
    _require(bool(error_indices), "Bootstrap requires at least one error trace")
    _require(bool(correct_indices), "Bootstrap requires at least one correct trace")
    views = {
        name: _detector_view(records, config)
        for name, config in detector_configs.items()
    }
    samples: Dict[str, Dict[str, List[float]]] = {
        name: {} for name in detector_configs
    }
    deltas: Dict[str, List[float]] = {}
    rng = random.Random(seed)
    for _ in range(resamples):
        indices = [rng.choice(error_indices) for _ in error_indices]
        indices.extend(rng.choice(correct_indices) for _ in correct_indices)
        replicate = {
            name: _metrics_for_indices(view, indices)
            for name, view in views.items()
        }
        for name, metrics in replicate.items():
            for metric, value in metrics.items():
                if value is not None:
                    samples[name].setdefault(metric, []).append(float(value))
        primary = replicate["primary_submitted"]
        selected = replicate["development_selected"]
        for metric in primary:
            if primary[metric] is not None and selected[metric] is not None:
                deltas.setdefault(metric, []).append(
                    float(selected[metric]) - float(primary[metric])
                )
    return {
        "detector_intervals": {
            name: {
                metric: _interval(values, confidence_level)
                for metric, values in metric_samples.items()
            }
            for name, metric_samples in samples.items()
        },
        "paired_delta_intervals": {
            metric: _interval(values, confidence_level)
            for metric, values in deltas.items()
        },
    }


def build_heldout_report(
    *,
    records: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    bootstrap_resamples: Optional[int] = None,
) -> Dict[str, Any]:
    validated_plan = validate_frozen_plan(plan)
    configs = validated_plan["detectors"]
    detector_results = {
        name: evaluate_detector(
            records,
            scoring_cfg=config["scoring"],
            threshold=float(config["risk_threshold"]),
        )
        for name, config in configs.items()
    }
    bootstrap_cfg = validated_plan["bootstrap"]
    resamples = (
        int(bootstrap_cfg["resamples"])
        if bootstrap_resamples is None
        else bootstrap_resamples
    )
    _require(resamples > 0, "Bootstrap resamples must be positive")
    bootstrap = paired_stratified_bootstrap(
        records,
        configs,
        resamples=resamples,
        seed=int(bootstrap_cfg["seed"]),
        confidence_level=float(bootstrap_cfg["confidence_level"]),
    )
    point_flat = {
        name: _flat_metrics(
            result["localization"],
            result["risk_association"]["trace_level"],
            result["risk_association"]["step_level"],
        )
        for name, result in detector_results.items()
    }
    point_deltas = {
        metric: (
            None
            if point_flat["primary_submitted"][metric] is None
            or point_flat["development_selected"][metric] is None
            else point_flat["development_selected"][metric]
            - point_flat["primary_submitted"][metric]
        )
        for metric in point_flat["primary_submitted"]
    }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report": "processbench_fixed_detector_heldout_evaluation",
        "selection_performed": False,
        "records": len(records),
        "detectors": {
            name: {
                **result,
                "confidence_intervals": bootstrap["detector_intervals"][name],
            }
            for name, result in detector_results.items()
        },
        "paired_comparison": {
            "direction": "development_selected_minus_primary_submitted",
            "point_deltas": point_deltas,
            "confidence_intervals": bootstrap["paired_delta_intervals"],
        },
        "bootstrap": {
            "method": "stratified_nonparametric_percentile_bootstrap",
            "strata": "gold_error_vs_correct_trace",
            "paired_detector_comparison": True,
            "resamples": resamples,
            "seed": int(bootstrap_cfg["seed"]),
            "confidence_level": float(bootstrap_cfg["confidence_level"]),
        },
    }


def report_from_files(
    *,
    predictions_path: Path,
    labels_path: Path,
    plan_path: Path,
    receipt_path: Path,
) -> Dict[str, Any]:
    plan = validate_frozen_plan(read_json(plan_path))
    predictions = list(read_jsonl(predictions_path))
    validate_blind_predictions(predictions, plan)
    receipt = _load_and_validate_receipt(
        receipt_path=receipt_path,
        plan_path=plan_path,
        predictions_path=predictions_path,
        predictions=predictions,
    )
    expected_label_hash = plan["heldout_partition"]["labels_sha256"].lower()
    actual_label_hash = sha256_file(labels_path)
    _require(
        actual_label_hash == expected_label_hash,
        "Held-out label hash differs from the frozen detector plan",
    )
    labels = list(read_jsonl(labels_path))
    records = join_blind_labels(predictions, labels)
    report = build_heldout_report(records=records, plan=plan)
    report["inputs"] = {
        "plan_sha256": sha256_file(plan_path),
        "prediction_receipt_sha256": sha256_file(receipt_path),
        "predictions_sha256": sha256_file(predictions_path),
        "predictions_ids_sha256": ids_sha256(predictions),
        "labels_sha256": actual_label_hash,
        "pre_unblinding_receipt_confirmed": receipt.get("unblinded") is False,
    }
    report["heldout_partition"] = dict(plan["heldout_partition"])
    return report


def _freeze(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite prediction receipt: {args.output}")
    receipt = freeze_prediction_receipt(
        predictions_path=args.predictions,
        plan_path=args.plan,
    )
    write_json_atomic(args.output, receipt)
    print(json.dumps(receipt, indent=2))


def _report(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite held-out report: {args.output}")
    report = report_from_files(
        predictions_path=args.predictions,
        labels_path=args.labels,
        plan_path=args.plan,
        receipt_path=args.receipt,
    )
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze and report fixed ProcessBench held-out detectors offline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser(
        "freeze-predictions",
        help="Hash and validate blind predictions before labels are opened",
    )
    freeze.add_argument("--predictions", type=Path, required=True)
    freeze.add_argument("--plan", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.set_defaults(handler=_freeze)

    report = subparsers.add_parser(
        "report",
        help="Evaluate exactly the two frozen detectors with paired intervals",
    )
    report.add_argument("--predictions", type=Path, required=True)
    report.add_argument("--labels", type=Path, required=True)
    report.add_argument("--plan", type=Path, required=True)
    report.add_argument("--receipt", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.set_defaults(handler=_report)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
