from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .io_utils import read_json, write_json_atomic


CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_KIND = "stepguard_generation_checkpoint"


class CheckpointValidationError(ValueError):
    """The generation checkpoint is structurally invalid."""


class CheckpointCompatibilityError(ValueError):
    """The checkpoint belongs to a different input or configuration."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _canonical_json_sha256(
    payload: Dict[str, Any],
) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def make_generation_checkpoint_identity(
    *,
    run_name: str,
    input_path: str | Path,
    config_raw: Dict[str, Any],
) -> Dict[str, Any]:
    resolved_input_path = Path(input_path).resolve()

    if not resolved_input_path.exists():
        raise FileNotFoundError(
            f"Generation input does not exist: "
            f"{resolved_input_path}"
        )

    return {
        "run_name": str(run_name),
        "input_path": str(resolved_input_path),
        "input_sha256": _file_sha256(
            resolved_input_path
        ),
        "config_sha256": _canonical_json_sha256(
            config_raw
        ),
    }


def _validate_identity(
    actual: Any,
    expected: Dict[str, Any],
) -> None:
    if not isinstance(actual, dict):
        raise CheckpointValidationError(
            "Checkpoint identity must be a JSON object"
        )

    differing_fields = sorted(
        key
        for key in set(actual) | set(expected)
        if actual.get(key) != expected.get(key)
    )

    if differing_fields:
        raise CheckpointCompatibilityError(
            "Checkpoint is incompatible with the current run; "
            "differing identity fields: "
            + ", ".join(differing_fields)
        )


def _validate_checkpoint_payload(
    payload: Dict[str, Any],
    *,
    expected_identity: Dict[str, Any],
    expected_record_count: Optional[int],
) -> None:
    if (
        payload.get("schema_version")
        != CHECKPOINT_SCHEMA_VERSION
    ):
        raise CheckpointCompatibilityError(
            "Unsupported generation checkpoint schema: "
            f"{payload.get('schema_version')!r}"
        )

    if payload.get("kind") != CHECKPOINT_KIND:
        raise CheckpointValidationError(
            "Unexpected checkpoint kind: "
            f"{payload.get('kind')!r}"
        )

    _validate_identity(
        payload.get("identity"),
        expected_identity,
    )

    saved_at_utc = payload.get("saved_at_utc")

    if not isinstance(saved_at_utc, str) or not saved_at_utc:
        raise CheckpointValidationError(
            "Checkpoint has no valid saved_at_utc value"
        )

    next_source_index = payload.get(
        "next_source_index"
    )

    if (
        not isinstance(next_source_index, int)
        or isinstance(next_source_index, bool)
        or next_source_index < 0
    ):
        raise CheckpointValidationError(
            "Checkpoint next_source_index must be "
            "a nonnegative integer"
        )

    records = payload.get("records")

    if not isinstance(records, list):
        raise CheckpointValidationError(
            "Checkpoint records must be a list"
        )

    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise CheckpointValidationError(
                f"Checkpoint record {record_index} "
                "must be an object"
            )

    if len(records) != next_source_index:
        raise CheckpointValidationError(
            "Checkpoint record count must equal "
            "next_source_index"
        )

    if (
        expected_record_count is not None
        and next_source_index > expected_record_count
    ):
        raise CheckpointValidationError(
            "Checkpoint next_source_index exceeds "
            "the current dataset record count"
        )


def save_generation_checkpoint(
    path: str | Path,
    *,
    identity: Dict[str, Any],
    next_source_index: int,
    records: List[Dict[str, Any]],
) -> None:
    payload: Dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": CHECKPOINT_KIND,
        "saved_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "identity": dict(identity),
        "next_source_index": next_source_index,
        "records": records,
    }

    _validate_checkpoint_payload(
        payload,
        expected_identity=identity,
        expected_record_count=None,
    )
    write_json_atomic(path, payload)


def load_generation_checkpoint(
    path: str | Path,
    *,
    expected_identity: Dict[str, Any],
    expected_record_count: int,
) -> Dict[str, Any]:
    checkpoint_path = Path(path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No generation checkpoint exists at "
            f"{checkpoint_path}"
        )

    if (
        not isinstance(expected_record_count, int)
        or isinstance(expected_record_count, bool)
        or expected_record_count < 0
    ):
        raise ValueError(
            "expected_record_count must be "
            "a nonnegative integer"
        )

    payload = read_json(checkpoint_path)

    _validate_checkpoint_payload(
        payload,
        expected_identity=expected_identity,
        expected_record_count=expected_record_count,
    )

    return payload


def delete_generation_checkpoint(
    path: str | Path,
) -> None:
    Path(path).unlink(missing_ok=True)