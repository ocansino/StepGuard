from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Dict, Iterator, Optional


_TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_input_tokens",
    "reasoning_output_tokens",
)


@dataclass
class ExecutionMetrics:
    run_name: str
    command: str
    output_path: Path
    execution_config: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    flush_interval_events: int = 1

    started_at_utc: str = field(init=False)
    completed_at_utc: Optional[str] = field(default=None, init=False)
    records_processed: int = field(default=0, init=False)

    _started_clock: float = field(init=False, repr=False)
    _lock: Any = field(init=False, repr=False)
    _operations: Dict[str, Dict[str, Any]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _waits: Dict[str, Dict[str, Any]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _stages: Dict[str, Dict[str, Any]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _stage_events: list[Dict[str, Any]] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _pending_snapshot_events: int = field(
        default=0,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.flush_interval_events, int)
            or isinstance(self.flush_interval_events, bool)
            or self.flush_interval_events < 1
        ):
            raise ValueError(
                "flush_interval_events must be a positive integer"
            )

        self.output_path = Path(self.output_path)
        self.started_at_utc = datetime.now(timezone.utc).isoformat()
        self._started_clock = perf_counter()
        self._lock = Lock()

        if self.enabled:
            with self._lock:
                self._write_snapshot_locked()

    def record_operation(
        self,
        *,
        provider: str,
        operation: str,
        elapsed_seconds: float,
        success: bool,
        usage: Optional[Dict[str, int]] = None,
        error_type: Optional[str] = None,
        logical_call: bool = True,
        retry_delay_seconds: float = 0.0,
    ) -> None:
        if not self.enabled:
            return

        key = f"{provider}.{operation}"
        usage = usage or {}

        with self._lock:
            entry = self._operations.setdefault(
                key,
                {
                    "provider": provider,
                    "operation": operation,
                    "logical_calls": 0,
                    "attempts": 0,
                    "retries": 0,
                    "successes": 0,
                    "failures": 0,
                    "elapsed_seconds": 0.0,
                    "retry_delay_seconds": 0.0,
                    "usage_reported_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cached_input_tokens": 0,
                    "reasoning_output_tokens": 0,
                    "error_types": {},
                },
            )

            if logical_call:
                entry["logical_calls"] += 1
            else:
                entry["retries"] += 1

            entry["attempts"] += 1
            entry["elapsed_seconds"] += max(
                0.0,
                float(elapsed_seconds),
            )
            entry["retry_delay_seconds"] += max(
                0.0,
                float(retry_delay_seconds),
            )

            if success:
                entry["successes"] += 1
            else:
                entry["failures"] += 1
                label = error_type or "UnknownError"
                entry["error_types"][label] = (
                    entry["error_types"].get(label, 0) + 1
                )

            usage_was_reported = False
            for token_field in _TOKEN_FIELDS:
                value = usage.get(token_field)
                if isinstance(value, (int, float)):
                    entry[token_field] += int(value)
                    usage_was_reported = True

            if usage_was_reported:
                entry["usage_reported_calls"] += 1

            self._mark_snapshot_dirty_locked()

    def record_wait(
        self,
        *,
        name: str,
        elapsed_seconds: float,
    ) -> None:
        if not self.enabled:
            return

        normalized_name = str(name).strip()

        if not normalized_name:
            raise ValueError("wait name must not be empty")

        elapsed = max(0.0, float(elapsed_seconds))

        if elapsed == 0:
            return

        with self._lock:
            entry = self._waits.setdefault(
                normalized_name,
                {
                    "name": normalized_name,
                    "events": 0,
                    "elapsed_seconds": 0.0,
                    "max_elapsed_seconds": 0.0,
                },
            )

            entry["events"] += 1
            entry["elapsed_seconds"] += elapsed
            entry["max_elapsed_seconds"] = max(
                entry["max_elapsed_seconds"],
                elapsed,
            )

            self._mark_snapshot_dirty_locked()

    def record_stage(
        self,
        *,
        name: str,
        elapsed_seconds: float,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None,
        error_type: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return

        normalized_name = str(name).strip()
        if not normalized_name:
            raise ValueError("stage name must not be empty")

        elapsed = max(0.0, float(elapsed_seconds))
        stage_metadata = dict(metadata or {})

        with self._lock:
            entry = self._stages.setdefault(
                normalized_name,
                {
                    "name": normalized_name,
                    "calls": 0,
                    "successes": 0,
                    "failures": 0,
                    "elapsed_seconds": 0.0,
                    "error_types": {},
                },
            )

            entry["calls"] += 1
            entry["elapsed_seconds"] += elapsed

            if success:
                entry["successes"] += 1
            else:
                entry["failures"] += 1
                label = error_type or "UnknownError"
                entry["error_types"][label] = (
                    entry["error_types"].get(label, 0) + 1
                )

            self._stage_events.append(
                {
                    "name": normalized_name,
                    "elapsed_seconds": elapsed,
                    "success": success,
                    "error_type": error_type,
                    "metadata": stage_metadata,
                }
            )

            self._mark_snapshot_dirty_locked()

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        started = perf_counter()

        try:
            yield
        except Exception as error:
            self.record_stage(
                name=name,
                elapsed_seconds=perf_counter() - started,
                success=False,
                metadata=metadata,
                error_type=type(error).__name__,
            )
            raise
        else:
            self.record_stage(
                name=name,
                elapsed_seconds=perf_counter() - started,
                success=True,
                metadata=metadata,
            )

    def set_records_processed(self, count: int) -> None:
        if not self.enabled:
            return

        with self._lock:
            self.records_processed = max(0, int(count))
            self._mark_snapshot_dirty_locked()

    def finish(self, *, records_processed: Optional[int] = None) -> None:
        if not self.enabled:
            return

        with self._lock:
            if records_processed is not None:
                self.records_processed = max(
                    0,
                    int(records_processed),
                )

            self.completed_at_utc = datetime.now(
                timezone.utc
            ).isoformat()

            if self._write_snapshot_locked():
                self._pending_snapshot_events = 0
    
    def flush(self) -> None:
        if not self.enabled:
            return

        with self._lock:
            if self._pending_snapshot_events == 0:
                return

            if self._write_snapshot_locked():
                self._pending_snapshot_events = 0

    def _mark_snapshot_dirty_locked(self) -> None:
        self._pending_snapshot_events += 1

        if (
            self._pending_snapshot_events
            >= self.flush_interval_events
        ):
            if self._write_snapshot_locked():
                self._pending_snapshot_events = 0

    def _totals_locked(self) -> Dict[str, Any]:
        totals: Dict[str, Any] = {
            "logical_calls": 0,
            "attempts": 0,
            "retries": 0,
            "successes": 0,
            "failures": 0,
            "elapsed_seconds": 0.0,
            "retry_delay_seconds": 0.0,
            "usage_reported_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_input_tokens": 0,
            "reasoning_output_tokens": 0,
        }

        for operation in self._operations.values():
            for key in totals:
                totals[key] += operation[key]

        return totals

    def _wait_totals_locked(self) -> Dict[str, Any]:
        return {
            "events": sum(
                entry["events"]
                for entry in self._waits.values()
            ),
            "elapsed_seconds": sum(
                entry["elapsed_seconds"]
                for entry in self._waits.values()
            ),
        }

    def _snapshot_locked(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "run_name": self.run_name,
            "command": self.command,
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "wall_time_seconds": max(
                0.0,
                perf_counter() - self._started_clock,
            ),
            "records_processed": self.records_processed,
            "execution_config": dict(self.execution_config),
            "context": dict(self.context),
            "totals": self._totals_locked(),
            "wait_totals": self._wait_totals_locked(),
            "waits": {
                key: dict(value)
                for key, value in self._waits.items()
            },
            "operations": {
                key: {
                    **value,
                    "error_types": dict(value["error_types"]),
                }
                for key, value in self._operations.items()
            },
            "stages": {
                key: {
                    **value,
                    "error_types": dict(value["error_types"]),
                }
                for key, value in self._stages.items()
            },
            "stage_events": [
                {
                    **event,
                    "metadata": dict(event["metadata"]),
                }
                for event in self._stage_events
            ],
        }

    def _write_snapshot_locked(self) -> bool:
        try:
            self.output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            temporary_path = self.output_path.with_suffix(
                self.output_path.suffix + ".tmp"
            )

            with temporary_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    self._snapshot_locked(),
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

            temporary_path.replace(self.output_path)
            return True
        except Exception:
            return False