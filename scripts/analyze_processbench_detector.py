from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import read_jsonl, write_json_atomic
from src.risk import compute_risks

from scripts.processbench import evaluate_labels
from scripts.processbench_runner import PREDICTION_SCHEMA_VERSION


SUBMITTED_SCORING = {
    "risk_formula": "weighted",
    "verifier_weight": 0.75,
    "contradiction_weight": 0.25,
}
SUBMITTED_THRESHOLD = 0.20
DEFAULT_WEIGHT_GRID = tuple(index / 20 for index in range(21))
DEFAULT_THRESHOLD_GRID = tuple(index / 20 for index in range(1, 20))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _safe_ratio(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def _harmonic_mean(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None or right is None:
        return None
    if left + right == 0:
        return 0.0
    return 2 * left * right / (left + right)


def _binary_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
) -> Dict[str, Any]:
    _require(len(labels) == len(scores), "Binary label and score counts differ")
    _require(bool(labels), "No binary examples to evaluate")
    predictions = [score > threshold for score in scores]
    true_positive = sum(label == 1 and prediction for label, prediction in zip(labels, predictions))
    false_positive = sum(label == 0 and prediction for label, prediction in zip(labels, predictions))
    false_negative = sum(label == 1 and not prediction for label, prediction in zip(labels, predictions))
    true_negative = sum(label == 0 and not prediction for label, prediction in zip(labels, predictions))
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    return {
        "examples": len(labels),
        "positive_examples": sum(labels),
        "negative_examples": len(labels) - sum(labels),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": _harmonic_mean(precision, recall),
        "false_positive_rate": _safe_ratio(false_positive, false_positive + true_negative),
        "accuracy": (true_positive + true_negative) / len(labels),
        "roc_auc": roc_auc(labels, scores),
        "average_precision": average_precision(labels, scores),
        "brier_score": sum((score - label) ** 2 for label, score in zip(labels, scores)) / len(labels),
        "calibration_bins": calibration_bins(labels, scores),
    }


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    ranked = sorted(zip(scores, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        positive_rank_sum += average_rank * sum(label for _, label in ranked[index:end])
        index = end
    return (
        positive_rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)


def average_precision(
    labels: Sequence[int],
    scores: Sequence[float],
) -> Optional[float]:
    positives = sum(labels)
    if positives == 0:
        return None
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    true_positive = 0
    processed = 0
    previous_recall = 0.0
    area = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        group = ranked[index:end]
        true_positive += sum(label for _, label in group)
        processed += len(group)
        recall = true_positive / positives
        precision = true_positive / processed
        area += (recall - previous_recall) * precision
        previous_recall = recall
        index = end
    return area


def calibration_bins(
    labels: Sequence[int],
    scores: Sequence[float],
    bin_count: int = 10,
) -> List[Dict[str, Any]]:
    bins = []
    for bin_index in range(bin_count):
        lower = bin_index / bin_count
        upper = (bin_index + 1) / bin_count
        members = [
            (label, score)
            for label, score in zip(labels, scores)
            if lower <= score < upper or (bin_index == bin_count - 1 and score == 1)
        ]
        if not members:
            continue
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "mean_score": sum(score for _, score in members) / len(members),
                "empirical_error_rate": sum(label for label, _ in members) / len(members),
            }
        )
    return bins


def _validate_prediction_rows(records: Sequence[Mapping[str, Any]]) -> None:
    _require(bool(records), "Development prediction file is empty")
    seen_ids = set()
    for index, record in enumerate(records):
        record_id = record.get("id")
        _require(isinstance(record_id, str) and record_id, f"Row {index}: missing id")
        _require(record_id not in seen_ids, f"Duplicate prediction id: {record_id}")
        seen_ids.add(record_id)
        _require(
            record.get("prediction_schema_version") == PREDICTION_SCHEMA_VERSION,
            f"{record_id}: prediction schema is not audit version {PREDICTION_SCHEMA_VERSION}",
        )
        gold = record.get("gold_label")
        score = record.get("stepguard_score")
        _require(isinstance(gold, int) and not isinstance(gold, bool), f"{record_id}: invalid gold label")
        _require(isinstance(score, Mapping), f"{record_id}: missing StepGuard score")
        steps = score.get("steps")
        verifier = (score.get("scores") or {}).get("verifier")
        contradiction = (score.get("scores") or {}).get("contradiction")
        judgments = score.get("raw_verifier_judgments")
        _require(isinstance(steps, list) and steps, f"{record_id}: missing steps")
        _require(
            isinstance(verifier, list)
            and isinstance(contradiction, list)
            and len(verifier) == len(steps)
            and len(contradiction) == len(steps),
            f"{record_id}: score arrays do not align",
        )
        _require(
            isinstance(judgments, list) and len(judgments) == len(steps),
            f"{record_id}: incomplete raw verifier audit",
        )
        _require(gold == -1 or 0 <= gold < len(steps), f"{record_id}: gold label is outside steps")


def _risks_for_record(
    record: Mapping[str, Any],
    scoring_cfg: Mapping[str, Any],
) -> List[float]:
    scores = record["stepguard_score"]["scores"]
    return compute_risks(
        list(scores["verifier"]),
        list(scores["contradiction"]),
        dict(scoring_cfg),
    )


def evaluate_detector(
    records: Sequence[Mapping[str, Any]],
    *,
    scoring_cfg: Mapping[str, Any],
    threshold: float,
) -> Dict[str, Any]:
    _validate_prediction_rows(records)
    gold_labels = []
    predicted_labels = []
    step_labels: List[int] = []
    step_scores: List[float] = []
    trace_labels: List[int] = []
    trace_scores: List[float] = []

    for record in records:
        gold = int(record["gold_label"])
        risks = _risks_for_record(record, scoring_cfg)
        prediction = next((index for index, risk in enumerate(risks) if risk > threshold), -1)
        gold_labels.append(gold)
        predicted_labels.append(prediction)
        trace_labels.append(int(gold != -1))
        trace_scores.append(max(risks) if risks else 0.0)

        last_evaluable_index = len(risks) - 1 if gold == -1 else gold
        for step_index in range(last_evaluable_index + 1):
            step_labels.append(int(gold != -1 and step_index == gold))
            step_scores.append(risks[step_index])

    return {
        "configuration": {
            "scoring": dict(scoring_cfg),
            "risk_threshold": float(threshold),
        },
        "localization": evaluate_labels(gold_labels, predicted_labels),
        "risk_association": {
            "trace_level": _binary_metrics(trace_labels, trace_scores, threshold),
            "step_level": _binary_metrics(step_labels, step_scores, threshold),
            "step_label_scope": (
                "all steps on gold-correct traces; pre-error and first-error steps "
                "on gold-error traces; post-error steps excluded as unlabeled"
            ),
        },
    }


def _objective(result: Mapping[str, Any]) -> Tuple[float, float, float, float, float]:
    official = result["localization"]["official_processbench"]
    additional = result["localization"]["additional_metrics"]
    step = result["risk_association"]["step_level"]
    f1 = official["f1"] if official["f1"] is not None else -1.0
    exact = additional["overall_exact_label_accuracy"]
    step_f1 = step["f1"] if step["f1"] is not None else -1.0
    false_positive_rate = step["false_positive_rate"]
    if false_positive_rate is None:
        false_positive_rate = 1.0
    config = result["configuration"]
    weight = float(config["scoring"].get("verifier_weight", 0.75))
    threshold = float(config["risk_threshold"])
    distance_from_submitted = abs(weight - 0.75) + abs(threshold - 0.20)
    return (f1, exact, step_f1, -false_positive_rate, -distance_from_submitted)


def analyze_development(
    records: Sequence[Mapping[str, Any]],
    *,
    weight_grid: Sequence[float] = DEFAULT_WEIGHT_GRID,
    threshold_grid: Sequence[float] = DEFAULT_THRESHOLD_GRID,
) -> Dict[str, Any]:
    submitted = evaluate_detector(
        records,
        scoring_cfg=SUBMITTED_SCORING,
        threshold=SUBMITTED_THRESHOLD,
    )
    fixed_ablations = {
        "verifier_only": evaluate_detector(
            records,
            scoring_cfg={"risk_formula": "verifier_only"},
            threshold=SUBMITTED_THRESHOLD,
        ),
        "contradiction_only": evaluate_detector(
            records,
            scoring_cfg={"risk_formula": "contradiction_only"},
            threshold=SUBMITTED_THRESHOLD,
        ),
    }

    candidates = []
    for verifier_weight in weight_grid:
        _require(0 <= verifier_weight <= 1, "Weight grid values must be between 0 and 1")
        scoring = {
            "risk_formula": "weighted",
            "verifier_weight": float(verifier_weight),
            "contradiction_weight": float(1 - verifier_weight),
        }
        for threshold in threshold_grid:
            _require(0 <= threshold <= 1, "Threshold grid values must be between 0 and 1")
            candidates.append(
                evaluate_detector(
                    records,
                    scoring_cfg=scoring,
                    threshold=float(threshold),
                )
            )
    ranked = sorted(candidates, key=_objective, reverse=True)
    return {
        "analysis": "processbench_detector_development_calibration",
        "records": len(records),
        "submitted_detector": submitted,
        "fixed_threshold_ablations": fixed_ablations,
        "calibration": {
            "selection_objective": [
                "maximize official ProcessBench F1",
                "maximize exact label accuracy",
                "maximize evaluable-step F1",
                "minimize evaluable-step false-positive rate",
                "minimize distance from submitted weight and threshold",
            ],
            "weight_grid": list(weight_grid),
            "threshold_grid": list(threshold_grid),
            "evaluated_configurations": len(candidates),
            "selected": ranked[0],
            "top_10": ranked[:10],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline ProcessBench detector analysis")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite analysis: {args.output}")
    result = analyze_development(list(read_jsonl(args.predictions)))
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
