from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import read_jsonl, write_json_atomic, write_jsonl

from scripts.processbench import PINNED_REVISION


PROTOCOL_VERSION = 1
DEFAULT_SEED = 20260920
DEVELOPMENT_RECORDS = 100
DEVELOPMENT_ERROR_TRACES = 52
DEVELOPMENT_CORRECT_TRACES = 48
EXPECTED_SPLIT_COUNTS = {
    "gsm8k": 400,
    "math": 1000,
    "olympiadbench": 1000,
    "omnimath": 1000,
}
GOLD_FIELDS = {
    "gold_label",
    "gold_earliest_bad_step",
    "final_answer_correct",
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


def _stable_key(record_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{record_id}".encode("utf-8")).hexdigest()


def select_development_records(
    records: Sequence[Mapping[str, Any]],
    *,
    required_ids: Iterable[str],
    seed: int = DEFAULT_SEED,
    error_count: int = DEVELOPMENT_ERROR_TRACES,
    correct_count: int = DEVELOPMENT_CORRECT_TRACES,
) -> List[Dict[str, Any]]:
    by_id = {str(record["id"]): record for record in records}
    _require(len(by_id) == len(records), "GSM8K source contains duplicate ids")
    required = set(required_ids)
    missing = sorted(required.difference(by_id))
    _require(not missing, f"Required smoke ids are missing: {missing}")

    required_errors = sum(by_id[record_id]["gold_label"] != -1 for record_id in required)
    required_correct = len(required) - required_errors
    _require(required_errors <= error_count, "Too many required error traces")
    _require(required_correct <= correct_count, "Too many required correct traces")

    remaining = [record for record in records if record["id"] not in required]
    error_candidates = sorted(
        (record for record in remaining if record["gold_label"] != -1),
        key=lambda record: _stable_key(str(record["id"]), seed),
    )
    correct_candidates = sorted(
        (record for record in remaining if record["gold_label"] == -1),
        key=lambda record: _stable_key(str(record["id"]), seed),
    )
    selected_ids = set(required)
    selected_ids.update(
        record["id"]
        for record in error_candidates[:error_count - required_errors]
    )
    selected_ids.update(
        record["id"]
        for record in correct_candidates[:correct_count - required_correct]
    )
    selected = [dict(record) for record in records if record["id"] in selected_ids]
    _require(
        len(selected) == error_count + correct_count,
        "Development selection has the wrong record count",
    )
    _require(
        sum(record["gold_label"] != -1 for record in selected) == error_count,
        "Development selection has the wrong error-trace count",
    )
    return selected


def blind_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in GOLD_FIELDS
    }


def label_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "id": record["id"],
        "gold_label": record["gold_label"],
        "gold_earliest_bad_step": record["gold_earliest_bad_step"],
        "final_answer_correct": record["final_answer_correct"],
    }


def _write_partition(
    *,
    name: str,
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    blinded: bool,
) -> Dict[str, Any]:
    if blinded:
        input_path = output_dir / f"{name}_inputs.jsonl"
        labels_path = output_dir / f"{name}_labels.jsonl"
        write_jsonl(input_path, (blind_record(record) for record in records))
        write_jsonl(labels_path, (label_record(record) for record in records))
        return {
            "name": name,
            "records": len(records),
            "error_traces": sum(record["gold_label"] != -1 for record in records),
            "correct_traces": sum(record["gold_label"] == -1 for record in records),
            "blinded": True,
            "input_path": str(input_path),
            "input_sha256": _sha256(input_path),
            "labels_path": str(labels_path),
            "labels_sha256": _sha256(labels_path),
        }

    path = output_dir / f"{name}.jsonl"
    write_jsonl(path, records)
    return {
        "name": name,
        "records": len(records),
        "error_traces": sum(record["gold_label"] != -1 for record in records),
        "correct_traces": sum(record["gold_label"] == -1 for record in records),
        "blinded": False,
        "path": str(path),
        "sha256": _sha256(path),
    }


def freeze_protocol(
    *,
    prepared_dir: Path,
    smoke_path: Path,
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
) -> Dict[str, Any]:
    manifest_path = output_dir / "protocol_manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite frozen protocol: {manifest_path}"
        )
    sources: Dict[str, List[Dict[str, Any]]] = {}
    source_manifest = {}
    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        path = prepared_dir / f"{split}.jsonl"
        records = list(read_jsonl(path))
        _require(
            len(records) == expected_count,
            f"{split}: found {len(records)} records, expected {expected_count}",
        )
        sources[split] = records
        source_manifest[split] = {
            "path": str(path),
            "sha256": _sha256(path),
            "records": len(records),
        }

    smoke_records = list(read_jsonl(smoke_path))
    smoke_ids = [str(record["id"]) for record in smoke_records]
    _require(len(smoke_ids) == 5, "Expected exactly five prior smoke records")

    development = select_development_records(
        sources["gsm8k"],
        required_ids=smoke_ids,
        seed=seed,
    )
    development_ids = {record["id"] for record in development}
    gsm8k_heldout = [
        record
        for record in sources["gsm8k"]
        if record["id"] not in development_ids
    ]
    _require(len(gsm8k_heldout) == 300, "GSM8K held-out partition must contain 300 records")
    _require(
        not development_ids.intersection(record["id"] for record in gsm8k_heldout),
        "Development and held-out GSM8K partitions overlap",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    partitions = [
        _write_partition(
            name="gsm8k_development_100",
            records=development,
            output_dir=output_dir,
            blinded=False,
        ),
        _write_partition(
            name="gsm8k_heldout_300",
            records=gsm8k_heldout,
            output_dir=output_dir,
            blinded=True,
        ),
    ]
    for split in ("math", "olympiadbench", "omnimath"):
        partitions.append(
            _write_partition(
                name=f"{split}_transfer_1000",
                records=sources[split],
                output_dir=output_dir,
                blinded=True,
            )
        )

    manifest = {
        "protocol": "stepguard_processbench_detector_evaluation",
        "protocol_version": PROTOCOL_VERSION,
        "dataset_repository": "Qwen/ProcessBench",
        "dataset_revision": PINNED_REVISION,
        "selection_seed": seed,
        "created_from_sources": source_manifest,
        "prior_smoke": {
            "path": str(smoke_path),
            "sha256": _sha256(smoke_path),
            "record_ids": smoke_ids,
            "all_in_development_partition": required_ids_are_in_partition(
                smoke_ids,
                development,
            ),
        },
        "development_policy": {
            "split": "gsm8k",
            "records": DEVELOPMENT_RECORDS,
            "error_traces": DEVELOPMENT_ERROR_TRACES,
            "correct_traces": DEVELOPMENT_CORRECT_TRACES,
            "selection": (
                "include all prior smoke records, then rank remaining records "
                "within error/correct strata by sha256(seed:id)"
            ),
            "allowed_uses": [
                "threshold calibration",
                "risk-weight calibration",
                "prompt debugging",
                "ablation selection",
            ],
        },
        "heldout_policy": {
            "gold_fields_removed_from_scoring_inputs": sorted(GOLD_FIELDS),
            "configuration_must_be_frozen_before_unblinding": True,
            "primary_submitted_detector": {
                "risk_formula": "weighted",
                "verifier_weight": 0.75,
                "contradiction_weight": 0.25,
                "risk_threshold": 0.20,
            },
            "primary_metrics": [
                "official_processbench_f1",
                "error_localization_accuracy",
                "correct_trace_accuracy",
                "trace_error_detection_precision",
                "trace_error_detection_recall",
                "trace_error_detection_f1",
                "false_positive_rate",
            ],
            "secondary_metrics": [
                "overall_exact_label_accuracy",
                "within_one_localization_accuracy_on_error_traces",
            ],
            "reporting": {
                "per_split_results_required": True,
                "macro_average_across_splits_required": True,
                "pooled_results_are_secondary": True,
                "raw_counts_required": True,
            },
            "confidence_intervals": {
                "level": 0.95,
                "method": "stratified_nonparametric_percentile_bootstrap",
                "strata": "gold error trace versus gold correct trace",
                "resamples": 10000,
                "seed": 20260920,
                "paired_resampling_for_detector_comparisons": True,
            },
        },
        "partitions": partitions,
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def required_ids_are_in_partition(
    required_ids: Iterable[str],
    records: Sequence[Mapping[str, Any]],
) -> bool:
    partition_ids = {str(record["id"]) for record in records}
    return set(required_ids).issubset(partition_ids)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze ProcessBench evaluation protocol v1")
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--smoke-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = freeze_protocol(
        prepared_dir=args.prepared_dir,
        smoke_path=args.smoke_path,
        output_dir=args.output_dir,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
