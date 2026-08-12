from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .io_utils import read_json, write_json_atomic


CHECKPOINT_SCHEMA_VERSION = 2
CHECKPOINT_KIND = "stepguard_candidate_pool_checkpoint"

_VALID_PHASES = {
    "original_scoring_chunk_complete",
    "original_scoring_complete",
    "iteration_chunk_complete",
    "iteration_complete",
}


class CheckpointValidationError(ValueError):
    """The checkpoint is incomplete or structurally invalid."""


class CheckpointCompatibilityError(ValueError):
    """The checkpoint belongs to a different input or configuration."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _canonical_json_sha256(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def make_candidate_pool_checkpoint_identity(
    *,
    run_name: str,
    input_path: str | Path,
    config_raw: Dict[str, Any],
    reuse_original_scores: bool,
) -> Dict[str, Any]:
    resolved_input_path = Path(input_path).resolve()

    if not resolved_input_path.exists():
        raise FileNotFoundError(
            f"Candidate-pool input does not exist: "
            f"{resolved_input_path}"
        )

    return {
        "run_name": str(run_name),
        "input_path": str(resolved_input_path),
        "input_sha256": _file_sha256(resolved_input_path),
        "config_sha256": _canonical_json_sha256(config_raw),
        "reuse_original_scores": bool(reuse_original_scores),
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


def _validate_state(
    state: Any,
    expected_index: int,
) -> None:
    if not isinstance(state, dict):
        raise CheckpointValidationError(
            f"Checkpoint state {expected_index} must be an object"
        )

    if state.get("state_index") != expected_index:
        raise CheckpointValidationError(
            f"Checkpoint state {expected_index} has invalid "
            f"state_index={state.get('state_index')!r}"
        )

    required_types = {
        "record": dict,
        "task": str,
        "current_trace": str,
        "current_score": dict,
        "original_score": dict,
        "original_score_source": str,
        "proposals": list,
        "generation_errors": list,
    }

    for field_name, expected_type in required_types.items():
        value = state.get(field_name)
        if not isinstance(value, expected_type):
            raise CheckpointValidationError(
                f"Checkpoint state {expected_index} field "
                f"{field_name!r} must be "
                f"{expected_type.__name__}"
            )

    if "scheduled_iteration" not in state:
        raise CheckpointValidationError(
            f"Checkpoint state {expected_index} has no "
            "scheduled_iteration field"
        )

    scheduled_iteration = state["scheduled_iteration"]

    if (
        scheduled_iteration is not None
        and (
            not isinstance(scheduled_iteration, int)
            or isinstance(scheduled_iteration, bool)
            or scheduled_iteration < 1
        )
    ):
        raise CheckpointValidationError(
            f"Checkpoint state {expected_index} field "
            "'scheduled_iteration' must be a positive "
            "integer or null"
        )


def _validate_checkpoint_payload(
    payload: Dict[str, Any],
    *,
    expected_identity: Dict[str, Any],
) -> None:
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointCompatibilityError(
            "Unsupported candidate-pool checkpoint schema: "
            f"{payload.get('schema_version')!r}"
        )

    if payload.get("kind") != CHECKPOINT_KIND:
        raise CheckpointValidationError(
            f"Unexpected checkpoint kind: {payload.get('kind')!r}"
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

    phase = payload.get("phase")
    if phase not in _VALID_PHASES:
        raise CheckpointValidationError(
            f"Invalid checkpoint phase: {phase!r}"
        )

    next_iteration = payload.get("next_iteration")
    if (
        not isinstance(next_iteration, int)
        or isinstance(next_iteration, bool)
        or next_iteration < 1
    ):
        raise CheckpointValidationError(
            "Checkpoint next_iteration must be a positive integer"
        )

    states = payload.get("states")
    if not isinstance(states, list):
        raise CheckpointValidationError(
            "Checkpoint states must be a list"
        )

    for expected_index, state in enumerate(states):
        _validate_state(state, expected_index)

    active_state_indexes = payload.get("active_state_indexes")
    if not isinstance(active_state_indexes, list):
        raise CheckpointValidationError(
            "Checkpoint active_state_indexes must be a list"
        )

    if len(active_state_indexes) != len(set(active_state_indexes)):
        raise CheckpointValidationError(
            "Checkpoint active_state_indexes contains duplicates"
        )

    for state_index in active_state_indexes:
        if (
            not isinstance(state_index, int)
            or isinstance(state_index, bool)
            or state_index < 0
            or state_index >= len(states)
        ):
            raise CheckpointValidationError(
                "Checkpoint active_state_indexes contains invalid "
                f"state index {state_index!r}"
            )


def save_candidate_pool_checkpoint(
    path: str | Path,
    *,
    identity: Dict[str, Any],
    phase: str,
    next_iteration: int,
    active_state_indexes: List[int],
    states: List[Dict[str, Any]],
) -> None:
    payload: Dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": CHECKPOINT_KIND,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "identity": dict(identity),
        "phase": phase,
        "next_iteration": next_iteration,
        "active_state_indexes": list(active_state_indexes),
        "states": states,
    }

    _validate_checkpoint_payload(
        payload,
        expected_identity=identity,
    )
    write_json_atomic(path, payload)


def load_candidate_pool_checkpoint(
    path: str | Path,
    *,
    expected_identity: Dict[str, Any],
) -> Dict[str, Any]:
    checkpoint_path = Path(path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No candidate-pool checkpoint exists at "
            f"{checkpoint_path}"
        )

    payload = read_json(checkpoint_path)
    _validate_checkpoint_payload(
        payload,
        expected_identity=expected_identity,
    )
    return payload


def delete_candidate_pool_checkpoint(
    path: str | Path,
) -> None:
    Path(path).unlink(missing_ok=True)