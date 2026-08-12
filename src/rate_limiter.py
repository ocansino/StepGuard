from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from time import monotonic, sleep
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from src.execution_metrics import ExecutionMetrics


@dataclass
class RequestRateLimiter:
    requests_per_minute: Optional[float]
    headroom_fraction: float = 0.8
    metrics: Optional["ExecutionMetrics"] = None
    wait_name: str = "openai.request_rate_limit"
    clock: Callable[[], float] = monotonic
    sleeper: Callable[[float], None] = sleep

    _lock: Lock = field(init=False, repr=False)
    _next_request_time: float = field(
        default=0.0,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._lock = Lock()

        if self.requests_per_minute is not None:
            self.requests_per_minute = float(
                self.requests_per_minute
            )

            if self.requests_per_minute <= 0:
                raise ValueError(
                    "requests_per_minute must be greater than zero"
                )

        self.headroom_fraction = float(self.headroom_fraction)

        if not 0 < self.headroom_fraction <= 1:
            raise ValueError(
                "headroom_fraction must be greater than zero "
                "and no greater than one"
            )

        self.wait_name = str(self.wait_name).strip()

        if not self.wait_name:
            raise ValueError("wait_name must not be empty")

    @property
    def enabled(self) -> bool:
        return self.requests_per_minute is not None

    @property
    def effective_requests_per_minute(self) -> Optional[float]:
        if self.requests_per_minute is None:
            return None

        return (
            self.requests_per_minute
            * self.headroom_fraction
        )

    @property
    def minimum_interval_seconds(self) -> float:
        effective_rpm = self.effective_requests_per_minute

        if effective_rpm is None:
            return 0.0

        return 60.0 / effective_rpm

    def acquire(self) -> float:
        if not self.enabled:
            return 0.0

        with self._lock:
            current_time = float(self.clock())
            scheduled_time = max(
                current_time,
                self._next_request_time,
            )
            wait_seconds = max(
                0.0,
                scheduled_time - current_time,
            )
            self._next_request_time = (
                scheduled_time
                + self.minimum_interval_seconds
            )

        if wait_seconds > 0:
            self.sleeper(wait_seconds)

            if self.metrics is not None:
                self.metrics.record_wait(
                    name=self.wait_name,
                    elapsed_seconds=wait_seconds,
                )

        return wait_seconds

    def defer(self, seconds: float) -> None:
        delay_seconds = max(0.0, float(seconds))

        if not self.enabled or delay_seconds == 0:
            return

        with self._lock:
            deferred_until = (
                float(self.clock()) + delay_seconds
            )
            self._next_request_time = max(
                self._next_request_time,
                deferred_until,
            )