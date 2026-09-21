from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import (
    assemble_trace_scores_batch,
    create_gemini_rate_limiter,
    create_mistral_rate_limiter,
    create_openai_rate_limiter,
    resolve_openai_execution_options,
)
from src.config import RunConfig, load_config
from src.execution_metrics import ExecutionMetrics
from src.io_utils import read_json, read_jsonl, write_json_atomic, write_jsonl
from src.manifest import make_manifest, write_manifest
from src.providers.gemini_client import GeminiClient
from src.providers.mistral_client import MistralClientWrapper
from src.providers.openai_client import OpenAIClientWrapper
from src.scorers.nli import NLIScorer

from scripts.processbench import evaluate_prediction_records
from scripts.freeze_processbench_protocol import GOLD_FIELDS


ScoreBatch = Callable[[Sequence[Mapping[str, Any]]], List[Dict[str, Any]]]
PREDICTION_SCHEMA_VERSION = 2
DEFAULT_VERIFIER_SCHEMA_MAX_RETRIES = 2


class VerifierJudgmentValidationError(ValueError):
    """The provider response parsed, but did not match the step contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_verifier(condition: bool, message: str) -> None:
    if not condition:
        raise VerifierJudgmentValidationError(message)


def detector_identity(
    *,
    tau: float,
    scoring_cfg: Mapping[str, Any],
    context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    identity = {
        "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
        "risk_threshold": float(tau),
        "scoring": dict(scoring_cfg),
        "prediction_convention": "zero_based_minus_one_for_no_error",
    }
    if context:
        identity["context"] = dict(context)
    return identity


def validate_verifier_judgments(
    judgments: Any,
    *,
    step_count: int,
    record_id: str,
) -> List[Dict[str, Any]]:
    """Validate and preserve the parsed objects returned by the verifier."""
    _require_verifier(
        isinstance(judgments, list),
        f"{record_id}: verifier judgments must be a list",
    )
    _require_verifier(
        len(judgments) == step_count,
        (
            f"{record_id}: verifier returned {len(judgments)} judgments "
            f"for {step_count} steps"
        ),
    )

    preserved = []
    seen_indexes = set()
    for position, judgment in enumerate(judgments):
        _require_verifier(
            isinstance(judgment, Mapping),
            f"{record_id}: verifier judgment {position} must be an object",
        )
        step_index = judgment.get("step_index")
        _require_verifier(
            isinstance(step_index, int)
            and not isinstance(step_index, bool)
            and 0 <= step_index < step_count,
            f"{record_id}: invalid verifier step_index {step_index!r}",
        )
        _require_verifier(
            step_index not in seen_indexes,
            f"{record_id}: duplicate verifier step_index {step_index}",
        )
        p_wrong = judgment.get("p_wrong")
        _require_verifier(
            isinstance(p_wrong, (int, float))
            and not isinstance(p_wrong, bool)
            and math.isfinite(float(p_wrong))
            and 0 <= float(p_wrong) <= 1,
            f"{record_id}: invalid p_wrong for step {step_index}: {p_wrong!r}",
        )
        seen_indexes.add(step_index)
        preserved.append(dict(judgment))

    _require_verifier(
        seen_indexes == set(range(step_count)),
        f"{record_id}: verifier judgments do not cover every step",
    )
    return preserved


def _schema_retry_question(question: str, step_count: int) -> str:
    return (
        question.rstrip()
        + "\n\n[VERIFIER OUTPUT CONTRACT - NOT PART OF THE MATH PROBLEM]\n"
        + f"The supplied STEPS list contains exactly {step_count} steps. "
        + f"Return exactly {step_count} result objects, using step_index 0 "
        + f"through {step_count - 1} exactly once each. Do not split a supplied "
        + "step into additional results and do not add a summary result."
    )


def validate_detector_input_record(record: Mapping[str, Any], index: int) -> None:
    prefix = f"Record {index}"
    _require(
        isinstance(record.get("id"), str) and bool(record["id"]),
        f"{prefix}: missing id",
    )
    _require(
        isinstance(record.get("question"), str) and bool(record["question"]),
        f"{prefix}: missing question",
    )
    steps = record.get("benchmark_steps")
    _require(
        isinstance(steps, list) and bool(steps),
        f"{prefix}: benchmark_steps must be a non-empty list",
    )
    _require(
        all(isinstance(step, str) and bool(step.strip()) for step in steps),
        f"{prefix}: benchmark_steps contains an invalid step",
    )


def validate_prepared_record(record: Mapping[str, Any], index: int) -> None:
    validate_detector_input_record(record, index)
    prefix = f"Record {index}"
    steps = record["benchmark_steps"]
    label = record.get("gold_label")
    _require(
        isinstance(label, int) and not isinstance(label, bool),
        f"{prefix}: gold_label must be an integer",
    )
    _require(
        label == -1 or 0 <= label < len(steps),
        f"{prefix}: gold_label is outside the step range",
    )


def make_stepguard_score_batch(
    *,
    judge_client: Any,
    nli: Any,
    tau: float,
    scoring_cfg: Mapping[str, Any],
    verifier_max_output_tokens: int = 1200,
    verifier_schema_max_retries: int = DEFAULT_VERIFIER_SCHEMA_MAX_RETRIES,
) -> ScoreBatch:
    """Bind injected verifier/NLI dependencies to the frozen StepGuard scorer."""

    _require(
        isinstance(verifier_schema_max_retries, int)
        and not isinstance(verifier_schema_max_retries, bool)
        and verifier_schema_max_retries >= 0,
        "verifier_schema_max_retries must be a non-negative integer",
    )

    def score_batch(
        records: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        items = []
        for record in records:
            steps = list(record["benchmark_steps"])
            schema_errors = []
            preserved_judgments = None
            for schema_attempt in range(verifier_schema_max_retries + 1):
                question = str(record["question"])
                if schema_attempt > 0:
                    question = _schema_retry_question(question, len(steps))
                # Benchmark evaluation is strict: provider failures propagate.
                # Only parsed responses that violate the one-result-per-step
                # contract receive this bounded formatting retry.
                judge_results = judge_client.judge_steps(
                    question=question,
                    steps=steps,
                    task="math",
                    max_output_tokens=verifier_max_output_tokens,
                )
                try:
                    preserved_judgments = validate_verifier_judgments(
                        judge_results,
                        step_count=len(steps),
                        record_id=str(record["id"]),
                    )
                except VerifierJudgmentValidationError as error:
                    schema_errors.append(str(error))
                    if schema_attempt >= verifier_schema_max_retries:
                        raise VerifierJudgmentValidationError(
                            f"{record['id']}: verifier schema remained invalid "
                            f"after {schema_attempt + 1} attempts; "
                            f"last error: {error}"
                        ) from error
                    continue
                break

            _require(
                preserved_judgments is not None,
                f"{record['id']}: verifier schema retry ended without judgments",
            )
            items.append(
                {
                    "steps": steps,
                    "judge_results": preserved_judgments,
                    "schema_errors": schema_errors,
                }
            )

        scores = assemble_trace_scores_batch(
            items=items,
            nli=nli,
            tau=float(tau),
            scoring_cfg=dict(scoring_cfg),
        )
        for score, item in zip(scores, items):
            score["raw_verifier_judgments"] = [
                dict(judgment)
                for judgment in item["judge_results"]
            ]
            score["verifier_audit"] = {
                "expected_steps": len(item["steps"]),
                "received_judgments": len(item["judge_results"]),
                "complete": True,
                "schema_attempts": len(item["schema_errors"]) + 1,
                "schema_retries": len(item["schema_errors"]),
                "schema_retry_contract": (
                    "expected_cardinality_v1"
                    if item["schema_errors"]
                    else None
                ),
                "rejected_schema_errors": list(item["schema_errors"]),
            }
        return scores

    return score_batch


def _validate_score_for_prediction(
    *,
    source: Mapping[str, Any],
    score: Mapping[str, Any],
) -> tuple[List[Dict[str, Any]], Dict[str, Any], Optional[int]]:
    source_steps = list(source["benchmark_steps"])
    scored_steps = score.get("steps")
    _require(
        scored_steps == source_steps,
        f"{source['id']}: scorer steps differ from benchmark_steps",
    )
    raw_judgments = validate_verifier_judgments(
        score.get("raw_verifier_judgments"),
        step_count=len(source_steps),
        record_id=str(source["id"]),
    )
    verifier_audit = score.get("verifier_audit")
    _require(
        isinstance(verifier_audit, Mapping)
        and verifier_audit.get("expected_steps") == len(source_steps)
        and verifier_audit.get("received_judgments") == len(source_steps)
        and verifier_audit.get("complete") is True,
        f"{source['id']}: incomplete verifier audit metadata",
    )

    earliest = score.get("earliest_bad_step")
    _require(
        earliest is None
        or (
            isinstance(earliest, int)
            and not isinstance(earliest, bool)
            and 0 <= earliest < len(source_steps)
        ),
        f"{source['id']}: invalid earliest_bad_step {earliest!r}",
    )
    return raw_judgments, dict(verifier_audit), earliest


def _prediction_score(
    score: Mapping[str, Any],
    raw_judgments: List[Dict[str, Any]],
    verifier_audit: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        **dict(score),
        "raw_verifier_judgments": raw_judgments,
        "verifier_audit": dict(verifier_audit),
    }


def make_prediction_record(
    *,
    source: Mapping[str, Any],
    score: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> Dict[str, Any]:
    raw_judgments, verifier_audit, earliest = _validate_score_for_prediction(
        source=source,
        score=score,
    )
    predicted_label = -1 if earliest is None else earliest
    gold_label = source["gold_label"]

    return {
        "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
        "id": source["id"],
        "benchmark": dict(source.get("benchmark", {})),
        "generator": source.get("generator"),
        "gold_label": gold_label,
        "predicted_label": predicted_label,
        "gold_earliest_bad_step": source.get("gold_earliest_bad_step"),
        "predicted_earliest_bad_step": earliest,
        "exact_label_match": predicted_label == gold_label,
        "final_answer_correct": source.get("final_answer_correct"),
        "detector_identity": dict(identity),
        "stepguard_score": _prediction_score(score, raw_judgments, verifier_audit),
    }


def make_blind_prediction_record(
    *,
    source: Mapping[str, Any],
    score: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> Dict[str, Any]:
    _require(
        not GOLD_FIELDS.intersection(source),
        f"{source['id']}: blinded input contains gold fields",
    )
    raw_judgments, verifier_audit, earliest = _validate_score_for_prediction(
        source=source,
        score=score,
    )
    return {
        "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
        "id": source["id"],
        "benchmark": dict(source.get("benchmark", {})),
        "generator": source.get("generator"),
        "predicted_label": -1 if earliest is None else earliest,
        "predicted_earliest_bad_step": earliest,
        "detector_identity": dict(identity),
        "stepguard_score": _prediction_score(score, raw_judgments, verifier_audit),
    }


def _load_resume_prefix(
    *,
    output_path: Path,
    source_records: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
    blinded: bool = False,
) -> List[Dict[str, Any]]:
    completed = list(read_jsonl(output_path))
    _require(
        len(completed) <= len(source_records),
        "Resume output contains more records than the source dataset",
    )
    for index, prior in enumerate(completed):
        source = source_records[index]
        _require(
            prior.get("id") == source.get("id"),
            f"Resume output id/order mismatch at index {index}",
        )
        if not blinded:
            _require(
                prior.get("gold_label") == source.get("gold_label"),
                f"Resume output gold label mismatch for {source.get('id')}",
            )
        _require(
            prior.get("detector_identity") == identity,
            f"Resume output detector configuration mismatch for {source.get('id')}",
        )
    return completed


def run_detector(
    *,
    records: Iterable[Mapping[str, Any]],
    score_batch: ScoreBatch,
    output_path: Path,
    tau: float,
    scoring_cfg: Mapping[str, Any],
    chunk_size: int = 25,
    resume: bool = False,
    metrics_path: Optional[Path] = None,
    identity_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    _require(
        isinstance(chunk_size, int)
        and not isinstance(chunk_size, bool)
        and chunk_size >= 1,
        "chunk_size must be a positive integer",
    )
    source_records = list(records)
    _require(bool(source_records), "ProcessBench input is empty")
    for index, record in enumerate(source_records):
        validate_prepared_record(record, index)
    ids = [record["id"] for record in source_records]
    _require(len(ids) == len(set(ids)), "ProcessBench input contains duplicate ids")

    identity = detector_identity(
        tau=tau,
        scoring_cfg=scoring_cfg,
        context=identity_context,
    )
    if output_path.exists():
        if not resume:
            raise FileExistsError(
                f"Refusing to overwrite existing predictions: {output_path}"
            )
        predictions = _load_resume_prefix(
            output_path=output_path,
            source_records=source_records,
            identity=identity,
        )
    else:
        predictions = []

    for start in range(len(predictions), len(source_records), chunk_size):
        chunk = source_records[start:start + chunk_size]
        scores = score_batch(chunk)
        _require(
            len(scores) == len(chunk),
            f"Scorer returned {len(scores)} results for a {len(chunk)}-record chunk",
        )
        chunk_predictions = [
            make_prediction_record(
                source=source,
                score=score,
                identity=identity,
            )
            for source, score in zip(chunk, scores)
        ]
        predictions.extend(chunk_predictions)
        write_jsonl(output_path, predictions)

    metrics = evaluate_prediction_records(predictions)
    if metrics_path is not None:
        write_json_atomic(metrics_path, metrics)
    return metrics


def run_blind_detector(
    *,
    records: Iterable[Mapping[str, Any]],
    score_batch: ScoreBatch,
    output_path: Path,
    tau: float,
    scoring_cfg: Mapping[str, Any],
    chunk_size: int = 25,
    resume: bool = False,
    identity_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    _require(
        isinstance(chunk_size, int)
        and not isinstance(chunk_size, bool)
        and chunk_size >= 1,
        "chunk_size must be a positive integer",
    )
    source_records = list(records)
    _require(bool(source_records), "ProcessBench input is empty")
    for index, record in enumerate(source_records):
        validate_detector_input_record(record, index)
        _require(
            not GOLD_FIELDS.intersection(record),
            f"{record['id']}: blinded input contains gold fields",
        )
    ids = [record["id"] for record in source_records]
    _require(len(ids) == len(set(ids)), "ProcessBench input contains duplicate ids")

    identity = detector_identity(
        tau=tau,
        scoring_cfg=scoring_cfg,
        context={**dict(identity_context or {}), "blinded": True},
    )
    if output_path.exists():
        if not resume:
            raise FileExistsError(
                f"Refusing to overwrite existing predictions: {output_path}"
            )
        predictions = _load_resume_prefix(
            output_path=output_path,
            source_records=source_records,
            identity=identity,
            blinded=True,
        )
    else:
        predictions = []

    for start in range(len(predictions), len(source_records), chunk_size):
        chunk = source_records[start:start + chunk_size]
        scores = score_batch(chunk)
        _require(
            len(scores) == len(chunk),
            f"Scorer returned {len(scores)} results for a {len(chunk)}-record chunk",
        )
        predictions.extend(
            make_blind_prediction_record(
                source=source,
                score=score,
                identity=identity,
            )
            for source, score in zip(chunk, scores)
        )
        write_jsonl(output_path, predictions)

    return {
        "records": len(predictions),
        "blinded": True,
        "predictions_path": str(output_path),
        "metrics_computed": False,
    }


def evaluate_blind_predictions(
    *,
    predictions: Iterable[Mapping[str, Any]],
    labels: Iterable[Mapping[str, Any]],
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    prediction_rows = list(predictions)
    label_rows = list(labels)
    _require(bool(prediction_rows), "Blind prediction file is empty")
    _require(
        len(prediction_rows) == len(label_rows),
        "Blind prediction and label counts differ",
    )
    evaluated = []
    for index, (prediction, label) in enumerate(zip(prediction_rows, label_rows)):
        _require(
            prediction.get("id") == label.get("id"),
            f"Blind prediction/label id mismatch at index {index}",
        )
        _require(
            not GOLD_FIELDS.intersection(prediction),
            f"{prediction.get('id')}: blind prediction unexpectedly contains gold fields",
        )
        gold_label = label.get("gold_label")
        _require(
            isinstance(gold_label, int) and not isinstance(gold_label, bool),
            f"{label.get('id')}: invalid held-out gold label",
        )
        evaluated.append(
            {
                **dict(prediction),
                "gold_label": gold_label,
                "gold_earliest_bad_step": label.get("gold_earliest_bad_step"),
                "final_answer_correct": label.get("final_answer_correct"),
                "exact_label_match": prediction.get("predicted_label") == gold_label,
            }
        )
    return evaluate_prediction_records(evaluated), evaluated


def export_scored_records(
    *,
    records: Iterable[Mapping[str, Any]],
    output_path: Path,
    tau: float,
    scoring_cfg: Mapping[str, Any],
    score_key: str = "stepguard_score",
) -> Dict[str, Any]:
    """Export already-scored records without invoking verifier or NLI models."""
    identity = detector_identity(tau=tau, scoring_cfg=scoring_cfg)
    predictions = []
    for index, record in enumerate(records):
        validate_prepared_record(record, index)
        score = record.get(score_key)
        _require(
            isinstance(score, Mapping),
            f"{record['id']}: missing score object at {score_key!r}",
        )
        predictions.append(
            make_prediction_record(source=record, score=score, identity=identity)
        )
    _require(bool(predictions), "Scored input is empty")
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing predictions: {output_path}"
        )
    write_jsonl(output_path, predictions)
    return evaluate_prediction_records(predictions)


def _parse_json_object(value: str, option_name: str) -> Dict[str, Any]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{option_name} must be valid JSON") from error
    _require(isinstance(payload, dict), f"{option_name} must be a JSON object")
    return payload


def validate_runtime_config(cfg: RunConfig) -> Dict[str, Any]:
    _require(cfg.raw.get("task_profile") == "math", "ProcessBench task_profile must be math")
    model_cfg = cfg.raw.get("model")
    _require(isinstance(model_cfg, dict), "model configuration must be a mapping")
    provider = model_cfg.get("provider")
    _require(
        provider in {"openai", "mistral", "gemini"},
        f"Unsupported ProcessBench provider: {provider!r}",
    )
    model_name = model_cfg.get("name")
    _require(
        isinstance(model_name, str) and bool(model_name.strip()),
        "model.name must be a non-empty string",
    )

    execution_cfg = cfg.raw.get("execution") or {}
    _require(isinstance(execution_cfg, dict), "execution must be a mapping")
    max_concurrency = execution_cfg.get("max_concurrency", 1)
    _require(
        max_concurrency == 1,
        "Initial ProcessBench runs require max_concurrency: 1",
    )
    chunk_size = execution_cfg.get("processbench_chunk_size", 1)
    _require(
        isinstance(chunk_size, int)
        and not isinstance(chunk_size, bool)
        and chunk_size >= 1,
        "execution.processbench_chunk_size must be a positive integer",
    )
    nli_batch_size = execution_cfg.get("nli_batch_size", 32)
    _require(
        isinstance(nli_batch_size, int)
        and not isinstance(nli_batch_size, bool)
        and nli_batch_size >= 1,
        "execution.nli_batch_size must be a positive integer",
    )

    scoring_cfg = cfg.raw.get("scoring") or {}
    _require(isinstance(scoring_cfg, dict), "scoring must be a mapping")
    tau = scoring_cfg.get("risk_threshold", scoring_cfg.get("tau"))
    _require(
        isinstance(tau, (int, float)) and not isinstance(tau, bool),
        "scoring.risk_threshold must be numeric",
    )
    _require(0 <= float(tau) <= 1, "scoring.risk_threshold must be between 0 and 1")

    processbench_cfg = cfg.raw.get("processbench") or {}
    _require(isinstance(processbench_cfg, dict), "processbench must be a mapping")
    blinded = processbench_cfg.get("blinded", False)
    _require(isinstance(blinded, bool), "processbench.blinded must be a boolean")
    expected_records = processbench_cfg.get("expected_records")
    if expected_records is not None:
        _require(
            isinstance(expected_records, int)
            and not isinstance(expected_records, bool)
            and expected_records >= 1,
            "processbench.expected_records must be a positive integer",
        )

    dataset_path = Path(cfg.dataset_path)
    _require(dataset_path.exists(), f"ProcessBench dataset does not exist: {dataset_path}")
    records = list(read_jsonl(dataset_path))
    _require(bool(records), "ProcessBench dataset is empty")
    for index, record in enumerate(records):
        if blinded:
            validate_detector_input_record(record, index)
            _require(
                not GOLD_FIELDS.intersection(record),
                f"{record['id']}: blinded input contains gold fields",
            )
        else:
            validate_prepared_record(record, index)
    if expected_records is not None:
        _require(
            len(records) == expected_records,
            f"ProcessBench dataset has {len(records)} records, expected {expected_records}",
        )

    return {
        "provider": provider,
        "model_name": model_name,
        "records": len(records),
        "chunk_size": chunk_size,
        "nli_batch_size": nli_batch_size,
        "risk_threshold": float(tau),
        "blinded": blinded,
    }


def _make_rate_limiter(
    provider: str,
    execution_cfg: Dict[str, Any],
    metrics: ExecutionMetrics,
) -> Any:
    if provider == "openai":
        return create_openai_rate_limiter(execution_cfg, metrics)
    if provider == "mistral":
        return create_mistral_rate_limiter(execution_cfg, metrics)
    return create_gemini_rate_limiter(execution_cfg, metrics)


def _make_judge_client(
    *,
    provider: str,
    model_name: str,
    execution_cfg: Dict[str, Any],
    metrics: ExecutionMetrics,
) -> Any:
    options = resolve_openai_execution_options(execution_cfg)
    rate_limiter = _make_rate_limiter(provider, execution_cfg, metrics)
    client_type = {
        "openai": OpenAIClientWrapper,
        "mistral": MistralClientWrapper,
        "gemini": GeminiClient,
    }[provider]
    return client_type(
        model=model_name,
        metrics=metrics,
        rate_limiter=rate_limiter,
        **options,
    )


def _execution_metrics_session_paths(metrics_path: Path) -> List[Path]:
    return sorted(
        metrics_path.parent.glob(f"{metrics_path.stem}.session_*.json")
    )


def archive_execution_metrics_for_resume(metrics_path: Path) -> Optional[Path]:
    """Preserve the completed/failed session before ExecutionMetrics replaces it."""
    if not metrics_path.exists():
        return None
    index = 1
    while True:
        destination = metrics_path.with_name(
            f"{metrics_path.stem}.session_{index:03d}.json"
        )
        if not destination.exists():
            break
        index += 1
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(metrics_path, temporary)
    temporary.replace(destination)
    return destination


def _sum_numeric_fields(
    snapshots: Sequence[Mapping[str, Any]],
    field: str,
) -> Dict[str, Any]:
    keys = set()
    for snapshot in snapshots:
        value = snapshot.get(field) or {}
        if isinstance(value, Mapping):
            keys.update(
                key
                for key, item in value.items()
                if isinstance(item, (int, float)) and not isinstance(item, bool)
            )
    return {
        key: sum((snapshot.get(field) or {}).get(key, 0) for snapshot in snapshots)
        for key in sorted(keys)
    }


def _merge_error_counts(entries: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    merged: Dict[str, int] = {}
    for entry in entries:
        for name, count in (entry.get("error_types") or {}).items():
            merged[str(name)] = merged.get(str(name), 0) + int(count)
    return merged


def aggregate_execution_metric_snapshots(
    snapshots: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    _require(bool(snapshots), "No execution metric sessions to aggregate")
    run_name = snapshots[0].get("run_name")
    command = snapshots[0].get("command")
    _require(
        all(snapshot.get("run_name") == run_name for snapshot in snapshots),
        "Execution metric sessions have different run names",
    )
    _require(
        all(snapshot.get("command") == command for snapshot in snapshots),
        "Execution metric sessions have different commands",
    )

    operation_keys = sorted(
        {
            key
            for snapshot in snapshots
            for key in (snapshot.get("operations") or {})
        }
    )
    operations = {}
    for key in operation_keys:
        entries = [
            snapshot["operations"][key]
            for snapshot in snapshots
            if key in (snapshot.get("operations") or {})
        ]
        template = entries[-1]
        numeric = _sum_numeric_fields(
            [{"entry": entry} for entry in entries],
            "entry",
        )
        operations[key] = {
            "provider": template.get("provider"),
            "operation": template.get("operation"),
            **numeric,
            "error_types": _merge_error_counts(entries),
        }

    wait_keys = sorted(
        {
            key
            for snapshot in snapshots
            for key in (snapshot.get("waits") or {})
        }
    )
    waits = {}
    for key in wait_keys:
        entries = [
            snapshot["waits"][key]
            for snapshot in snapshots
            if key in (snapshot.get("waits") or {})
        ]
        waits[key] = {
            "name": entries[-1].get("name", key),
            "events": sum(int(entry.get("events", 0)) for entry in entries),
            "elapsed_seconds": sum(
                float(entry.get("elapsed_seconds", 0)) for entry in entries
            ),
            "max_elapsed_seconds": max(
                (float(entry.get("max_elapsed_seconds", 0)) for entry in entries),
                default=0.0,
            ),
        }

    stage_keys = sorted(
        {
            key
            for snapshot in snapshots
            for key in (snapshot.get("stages") or {})
        }
    )
    stages = {}
    for key in stage_keys:
        entries = [
            snapshot["stages"][key]
            for snapshot in snapshots
            if key in (snapshot.get("stages") or {})
        ]
        numeric = _sum_numeric_fields(
            [{"entry": entry} for entry in entries],
            "entry",
        )
        stages[key] = {
            "name": entries[-1].get("name", key),
            **numeric,
            "error_types": _merge_error_counts(entries),
        }

    stage_events = []
    for session_index, snapshot in enumerate(snapshots, start=1):
        for event in snapshot.get("stage_events") or []:
            stage_events.append({**dict(event), "session_index": session_index})

    current = snapshots[-1]
    return {
        "schema_version": 1,
        "aggregate": True,
        "sessions": len(snapshots),
        "run_name": run_name,
        "command": command,
        "started_at_utc": snapshots[0].get("started_at_utc"),
        "completed_at_utc": current.get("completed_at_utc"),
        "wall_time_seconds": sum(
            float(snapshot.get("wall_time_seconds", 0)) for snapshot in snapshots
        ),
        "records_processed": max(
            int(snapshot.get("records_processed", 0)) for snapshot in snapshots
        ),
        "execution_config": dict(current.get("execution_config") or {}),
        "context": dict(current.get("context") or {}),
        "totals": _sum_numeric_fields(snapshots, "totals"),
        "wait_totals": _sum_numeric_fields(snapshots, "wait_totals"),
        "waits": waits,
        "operations": operations,
        "stages": stages,
        "stage_events": stage_events,
        "session_summaries": [
            {
                "session_index": index,
                "started_at_utc": snapshot.get("started_at_utc"),
                "completed_at_utc": snapshot.get("completed_at_utc"),
                "wall_time_seconds": snapshot.get("wall_time_seconds"),
                "records_processed": snapshot.get("records_processed"),
                "totals": dict(snapshot.get("totals") or {}),
            }
            for index, snapshot in enumerate(snapshots, start=1)
        ],
    }


def write_aggregate_execution_metrics(metrics_path: Path) -> Optional[Path]:
    session_paths = _execution_metrics_session_paths(metrics_path)
    if not session_paths or not metrics_path.exists():
        return None
    snapshots = [read_json(path) for path in session_paths]
    snapshots.append(read_json(metrics_path))
    aggregate = aggregate_execution_metric_snapshots(snapshots)
    aggregate["session_files"] = [str(path) for path in session_paths]
    aggregate["current_session_file"] = str(metrics_path)
    output_path = metrics_path.with_name(
        f"{metrics_path.stem}.aggregate.json"
    )
    write_json_atomic(output_path, aggregate)
    return output_path


def run_from_config(config_path: Path, *, resume: bool = False) -> Dict[str, Any]:
    cfg = load_config(config_path)
    resolved = validate_runtime_config(cfg)
    outdir = Path(cfg.output_dir) / cfg.run_name
    predictions_path = outdir / "processbench_predictions.jsonl"
    benchmark_metrics_path = outdir / "processbench_metrics.json"
    execution_metrics_path = outdir / "metrics.processbench_score.json"
    manifest_path = outdir / "manifest.processbench_score.json"
    if predictions_path.exists() and not resume:
        raise FileExistsError(
            f"Refusing to overwrite existing predictions: {predictions_path}"
        )

    archived_metrics_path = (
        archive_execution_metrics_for_resume(execution_metrics_path)
        if resume
        else None
    )

    execution_cfg = dict(cfg.raw.get("execution") or {})
    model_cfg = dict(cfg.raw["model"])
    scoring_cfg = dict(cfg.raw.get("scoring") or {})
    processbench_cfg = dict(cfg.raw.get("processbench") or {})
    metrics = ExecutionMetrics(
        run_name=cfg.run_name,
        command="processbench-score",
        output_path=execution_metrics_path,
        execution_config=execution_cfg,
        context={
            "dataset_path": cfg.dataset_path,
            "task_profile": "math",
            "model": model_cfg,
            "processbench": processbench_cfg,
            "resume": {
                "enabled": resume,
                "archived_prior_metrics": (
                    str(archived_metrics_path)
                    if archived_metrics_path is not None
                    else None
                ),
            },
        },
        enabled=bool(execution_cfg.get("metrics_enabled", True)),
        flush_interval_events=execution_cfg.get("metrics_flush_interval_events", 1),
    )
    if not manifest_path.exists():
        write_manifest(
            manifest_path,
            make_manifest(
                cfg.run_name,
                "processbench-score",
                str(config_path),
                cfg.raw,
            ),
        )

    try:
        with metrics.stage(
            "nli_initialization",
            metadata={"model": "FacebookAI/roberta-large-mnli"},
        ):
            nli = NLIScorer(
                model_name="FacebookAI/roberta-large-mnli",
                metrics=metrics,
                batch_size=resolved["nli_batch_size"],
            )
        judge_client = _make_judge_client(
            provider=resolved["provider"],
            model_name=resolved["model_name"],
            execution_cfg=execution_cfg,
            metrics=metrics,
        )
        score_batch = make_stepguard_score_batch(
            judge_client=judge_client,
            nli=nli,
            tau=resolved["risk_threshold"],
            scoring_cfg=scoring_cfg,
            verifier_max_output_tokens=int(
                processbench_cfg.get("verifier_max_output_tokens", 1200)
            ),
            verifier_schema_max_retries=DEFAULT_VERIFIER_SCHEMA_MAX_RETRIES,
        )
        with metrics.stage(
            "processbench_detector_scoring",
            metadata={
                "records": resolved["records"],
                "chunk_size": resolved["chunk_size"],
                "strict_provider_errors": True,
            },
        ):
            runner = run_blind_detector if resolved["blinded"] else run_detector
            runner_options = {
                "records": read_jsonl(cfg.dataset_path),
                "score_batch": score_batch,
                "output_path": predictions_path,
                "tau": resolved["risk_threshold"],
                "scoring_cfg": scoring_cfg,
                "chunk_size": resolved["chunk_size"],
                "resume": resume,
                "identity_context": {
                    "provider": resolved["provider"],
                    "model": resolved["model_name"],
                    "dataset_path": cfg.dataset_path,
                },
            }
            if not resolved["blinded"]:
                runner_options["metrics_path"] = benchmark_metrics_path
            benchmark_metrics = runner(**runner_options)
    finally:
        completed = (
            len(list(read_jsonl(predictions_path)))
            if predictions_path.exists()
            else 0
        )
        metrics.finish(records_processed=completed)
        if resume:
            write_aggregate_execution_metrics(execution_metrics_path)
    return benchmark_metrics


def _export(args: argparse.Namespace) -> None:
    scoring_cfg = _parse_json_object(args.scoring, "--scoring")
    metrics = export_scored_records(
        records=read_jsonl(args.input),
        output_path=args.output,
        tau=args.risk_threshold,
        scoring_cfg=scoring_cfg,
        score_key=args.score_key,
    )
    if args.metrics:
        write_json_atomic(args.metrics, metrics)
    print(json.dumps(metrics, indent=2))


def _validate_config(args: argparse.Namespace) -> None:
    resolved = validate_runtime_config(load_config(args.config))
    print(json.dumps(resolved, indent=2))


def _score(args: argparse.Namespace) -> None:
    metrics = run_from_config(args.config, resume=args.resume)
    print(json.dumps(metrics, indent=2))


def _evaluate_blind(args: argparse.Namespace) -> None:
    metrics, evaluated = evaluate_blind_predictions(
        predictions=read_jsonl(args.predictions),
        labels=read_jsonl(args.labels),
    )
    if args.evaluated_records:
        if args.evaluated_records.exists():
            raise FileExistsError(
                f"Refusing to overwrite evaluated records: {args.evaluated_records}"
            )
        write_jsonl(args.evaluated_records, evaluated)
    if args.metrics.exists():
        raise FileExistsError(f"Refusing to overwrite metrics: {args.metrics}")
    write_json_atomic(args.metrics, metrics)
    print(json.dumps(metrics, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or export strict ProcessBench detector predictions",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser(
        "export",
        help="Convert already-scored ProcessBench rows to prediction JSONL",
    )
    export.add_argument("--input", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--metrics", type=Path)
    export.add_argument("--score-key", default="stepguard_score")
    export.add_argument("--risk-threshold", type=float, required=True)
    export.add_argument("--scoring", required=True)
    export.set_defaults(handler=_export)

    validate = subparsers.add_parser(
        "validate-config",
        help="Validate a ProcessBench run config without initializing models",
    )
    validate.add_argument("--config", type=Path, required=True)
    validate.set_defaults(handler=_validate_config)

    score = subparsers.add_parser(
        "score",
        help="Run verifier and local NLI scoring from a validated config",
    )
    score.add_argument("--config", type=Path, required=True)
    score.add_argument("--resume", action="store_true")
    score.set_defaults(handler=_score)

    evaluate_blind = subparsers.add_parser(
        "evaluate-blind",
        help="Join frozen labels to blind predictions and compute metrics",
    )
    evaluate_blind.add_argument("--predictions", type=Path, required=True)
    evaluate_blind.add_argument("--labels", type=Path, required=True)
    evaluate_blind.add_argument("--metrics", type=Path, required=True)
    evaluate_blind.add_argument("--evaluated-records", type=Path)
    evaluate_blind.set_defaults(handler=_evaluate_blind)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
