from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import read_jsonl, write_json_atomic, write_jsonl
from src.parsing import split_steps


DATASET_REPO = "Qwen/ProcessBench"
PINNED_REVISION = "3bdcd5371ed567559a78f559c01c13a6deee7604"
SUPPORTED_SPLITS = ("gsm8k", "math", "olympiadbench", "omnimath")
EXPECTED_SPLIT_COUNTS = {
    "gsm8k": 400,
    "math": 1000,
    "olympiadbench": 1000,
    "omnimath": 1000,
}
REQUIRED_FIELDS = {
    "id",
    "generator",
    "problem",
    "steps",
    "final_answer_correct",
    "label",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_record(record: Mapping[str, Any], split: str) -> None:
    prefix = f"{split}/{record.get('id', '<missing-id>')}"
    missing = REQUIRED_FIELDS.difference(record)
    _require(not missing, f"{prefix}: missing fields {sorted(missing)}")

    _require(
        isinstance(record["id"], str) and bool(record["id"].strip()),
        f"{prefix}: id must be a non-empty string",
    )
    _require(
        isinstance(record["generator"], str),
        f"{prefix}: generator must be a string",
    )
    _require(
        isinstance(record["problem"], str) and bool(record["problem"].strip()),
        f"{prefix}: problem must be a non-empty string",
    )

    steps = record["steps"]
    _require(
        isinstance(steps, list) and bool(steps),
        f"{prefix}: steps must be a non-empty list",
    )
    _require(
        all(isinstance(step, str) and bool(step.strip()) for step in steps),
        f"{prefix}: every step must be a non-empty string",
    )
    _require(
        isinstance(record["final_answer_correct"], bool),
        f"{prefix}: final_answer_correct must be a boolean",
    )

    label = record["label"]
    _require(
        isinstance(label, int) and not isinstance(label, bool),
        f"{prefix}: label must be an integer",
    )
    _require(
        label == -1 or 0 <= label < len(steps),
        f"{prefix}: label {label} is outside -1 or [0, {len(steps) - 1}]",
    )


def validate_source_records(
    records: Sequence[Mapping[str, Any]],
    split: str,
) -> None:
    _require(split in SUPPORTED_SPLITS, f"Unsupported ProcessBench split: {split}")
    _require(bool(records), f"{split}: source file is empty")
    seen_ids = set()
    for record in records:
        _require(isinstance(record, Mapping), f"{split}: each row must be an object")
        validate_source_record(record, split)
        record_id = record["id"]
        _require(record_id not in seen_ids, f"{split}: duplicate id {record_id!r}")
        seen_ids.add(record_id)


def format_numbered_trace(steps: Sequence[str]) -> str:
    return "\n\n".join(
        f"Step {index}: {step.strip()}"
        for index, step in enumerate(steps, start=1)
    )


def convert_record(
    record: Mapping[str, Any],
    split: str,
    revision: str = PINNED_REVISION,
) -> Dict[str, Any]:
    validate_source_record(record, split)
    steps = list(record["steps"])
    model_trace = format_numbered_trace(steps)
    parsed_steps = split_steps(model_trace)
    _require(
        len(parsed_steps) == len(steps),
        (
            f"{split}/{record['id']}: numbered trace parses into "
            f"{len(parsed_steps)} steps, but the benchmark contains {len(steps)}"
        ),
    )

    label = int(record["label"])
    return {
        "id": f"processbench:{split}:{record['id']}",
        "task": "math",
        "question": record["problem"],
        "model_trace": model_trace,
        "benchmark_steps": steps,
        "generator": record["generator"],
        "gold_label": label,
        "gold_earliest_bad_step": None if label == -1 else label,
        "final_answer_correct": record["final_answer_correct"],
        "benchmark": {
            "name": "ProcessBench",
            "repository": DATASET_REPO,
            "revision": revision,
            "split": split,
            "source_id": record["id"],
            "label_indexing": "zero_based_minus_one_for_no_error",
        },
    }


def load_source(path: Path, split: str) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    _require(isinstance(payload, list), f"{path}: expected a JSON list")
    validate_source_records(payload, split)
    return [dict(record) for record in payload]


def prepare_split(
    source_path: Path,
    output_path: Path,
    split: str,
    revision: str = PINNED_REVISION,
    enforce_official_count: bool = True,
) -> Dict[str, Any]:
    records = load_source(source_path, split)
    if enforce_official_count:
        expected = EXPECTED_SPLIT_COUNTS[split]
        _require(
            len(records) == expected,
            f"{split}: found {len(records)} records, expected {expected}",
        )
    converted = [convert_record(record, split, revision) for record in records]
    write_jsonl(output_path, converted)
    return {
        "split": split,
        "records": len(converted),
        "source_path": str(source_path),
        "source_sha256": _sha256(source_path),
        "output_path": str(output_path),
        "output_sha256": _sha256(output_path),
    }


def select_smoke_subset(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    error_count: int,
    correct_count: int,
) -> List[Dict[str, Any]]:
    _require(error_count >= 0, "error_count must not be negative")
    _require(correct_count >= 0, "correct_count must not be negative")
    _require(error_count + correct_count > 0, "Subset must not be empty")

    def stable_key(record: Mapping[str, Any]) -> str:
        value = f"{seed}:{record['id']}".encode("utf-8")
        return hashlib.sha256(value).hexdigest()

    error_rows = sorted(
        (record for record in records if record.get("gold_label") != -1),
        key=stable_key,
    )
    correct_rows = sorted(
        (record for record in records if record.get("gold_label") == -1),
        key=stable_key,
    )
    _require(
        len(error_rows) >= error_count,
        f"Only {len(error_rows)} error traces are available",
    )
    _require(
        len(correct_rows) >= correct_count,
        f"Only {len(correct_rows)} correct traces are available",
    )
    selected = error_rows[:error_count] + correct_rows[:correct_count]
    return [dict(record) for record in sorted(selected, key=stable_key)]


def write_smoke_subset(
    *,
    source_path: Path,
    output_path: Path,
    manifest_path: Path,
    seed: int,
    error_count: int,
    correct_count: int,
) -> Dict[str, Any]:
    records = list(read_jsonl(source_path))
    selected = select_smoke_subset(
        records,
        seed=seed,
        error_count=error_count,
        correct_count=correct_count,
    )
    write_jsonl(output_path, selected)
    manifest = {
        "benchmark": "ProcessBench",
        "purpose": "diagnostic_smoke_test_not_reported_evaluation",
        "selection": "sha256(seed:id), stratified by gold error presence",
        "seed": seed,
        "requested_error_traces": error_count,
        "requested_correct_traces": correct_count,
        "source_path": str(source_path),
        "source_sha256": _sha256(source_path),
        "output_path": str(output_path),
        "output_sha256": _sha256(output_path),
        "record_ids": [record["id"] for record in selected],
        "records": len(selected),
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


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


def evaluate_labels(
    gold_labels: Sequence[int],
    predicted_labels: Sequence[int],
) -> Dict[str, Any]:
    _require(
        len(gold_labels) == len(predicted_labels),
        "Gold and prediction counts differ",
    )
    _require(bool(gold_labels), "No predictions to evaluate")
    for name, labels in (("gold", gold_labels), ("predicted", predicted_labels)):
        _require(
            all(isinstance(label, int) and not isinstance(label, bool) for label in labels),
            f"Every {name} label must be an integer",
        )
        _require(
            all(label >= -1 for label in labels),
            f"Every {name} label must be -1 or a non-negative step index",
        )

    error_count = sum(label != -1 for label in gold_labels)
    correct_count = len(gold_labels) - error_count
    error_exact = sum(
        gold != -1 and prediction == gold
        for gold, prediction in zip(gold_labels, predicted_labels)
    )
    correct_exact = sum(
        gold == -1 and prediction == -1
        for gold, prediction in zip(gold_labels, predicted_labels)
    )
    exact_total = sum(
        gold == prediction
        for gold, prediction in zip(gold_labels, predicted_labels)
    )

    error_localization_accuracy = _safe_ratio(error_exact, error_count)
    correct_trace_accuracy = _safe_ratio(correct_exact, correct_count)

    true_positive = sum(
        gold != -1 and prediction != -1
        for gold, prediction in zip(gold_labels, predicted_labels)
    )
    false_positive = sum(
        gold == -1 and prediction != -1
        for gold, prediction in zip(gold_labels, predicted_labels)
    )
    false_negative = sum(
        gold != -1 and prediction == -1
        for gold, prediction in zip(gold_labels, predicted_labels)
    )
    true_negative = correct_exact
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    detection_f1 = _harmonic_mean(precision, recall)
    within_one = sum(
        gold != -1 and prediction != -1 and abs(gold - prediction) <= 1
        for gold, prediction in zip(gold_labels, predicted_labels)
    )

    return {
        "records": len(gold_labels),
        "official_processbench": {
            "error_trace_count": error_count,
            "correct_trace_count": correct_count,
            "error_localization_accuracy": error_localization_accuracy,
            "correct_trace_accuracy": correct_trace_accuracy,
            "f1": _harmonic_mean(
                error_localization_accuracy,
                correct_trace_accuracy,
            ),
        },
        "additional_metrics": {
            "overall_exact_label_accuracy": exact_total / len(gold_labels),
            "within_one_localization_accuracy_on_error_traces": _safe_ratio(
                within_one,
                error_count,
            ),
            "trace_error_detection": {
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_negative": true_negative,
                "precision": precision,
                "recall": recall,
                "f1": detection_f1,
                "false_positive_rate": _safe_ratio(
                    false_positive,
                    false_positive + true_negative,
                ),
            },
        },
    }


def evaluate_prediction_records(
    records: Iterable[Mapping[str, Any]],
    gold_key: str = "gold_label",
    prediction_key: str = "predicted_label",
) -> Dict[str, Any]:
    rows = list(records)
    _require(bool(rows), "Prediction file is empty")
    for index, row in enumerate(rows, start=1):
        _require(gold_key in row, f"Row {index}: missing {gold_key!r}")
        _require(prediction_key in row, f"Row {index}: missing {prediction_key!r}")
    return evaluate_labels(
        [row[gold_key] for row in rows],
        [row[prediction_key] for row in rows],
    )


def _prepare(args: argparse.Namespace) -> None:
    summaries = []
    for split in args.splits:
        summaries.append(
            prepare_split(
                args.source_dir / f"{split}.json",
                args.output_dir / f"{split}.jsonl",
                split,
            )
        )
    manifest = {
        "benchmark": "ProcessBench",
        "repository": DATASET_REPO,
        "revision": PINNED_REVISION,
        "splits": summaries,
    }
    write_json_atomic(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def _evaluate(args: argparse.Namespace) -> None:
    metrics = evaluate_prediction_records(
        read_jsonl(args.predictions),
        gold_key=args.gold_key,
        prediction_key=args.prediction_key,
    )
    if args.output:
        write_json_atomic(args.output, metrics)
    print(json.dumps(metrics, indent=2))


def _subset(args: argparse.Namespace) -> None:
    manifest = write_smoke_subset(
        source_path=args.input,
        output_path=args.output,
        manifest_path=args.manifest,
        seed=args.seed,
        error_count=args.error_count,
        correct_count=args.correct_count,
    )
    print(json.dumps(manifest, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline ProcessBench preparation and evaluation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument(
        "--splits",
        nargs="+",
        choices=SUPPORTED_SPLITS,
        default=list(SUPPORTED_SPLITS),
    )
    prepare.set_defaults(handler=_prepare)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--gold-key", default="gold_label")
    evaluate.add_argument("--prediction-key", default="predicted_label")
    evaluate.add_argument("--output", type=Path)
    evaluate.set_defaults(handler=_evaluate)

    subset = subparsers.add_parser("subset")
    subset.add_argument("--input", type=Path, required=True)
    subset.add_argument("--output", type=Path, required=True)
    subset.add_argument("--manifest", type=Path, required=True)
    subset.add_argument("--seed", type=int, required=True)
    subset.add_argument("--error-count", type=int, required=True)
    subset.add_argument("--correct-count", type=int, required=True)
    subset.set_defaults(handler=_subset)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
