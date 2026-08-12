import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.execution_metrics import ExecutionMetrics
from src.rate_limiter import RequestRateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.current = 0.0
        self.sleeps = []

    def now(self) -> float:
        return self.current

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.current += seconds


class RequestRateLimiterTests(unittest.TestCase):
    def test_disabled_limiter_never_waits(self) -> None:
        clock = FakeClock()
        limiter = RequestRateLimiter(
            requests_per_minute=None,
            clock=clock.now,
            sleeper=clock.sleep,
        )

        self.assertEqual(limiter.acquire(), 0.0)
        self.assertEqual(limiter.acquire(), 0.0)
        self.assertEqual(clock.sleeps, [])

    def test_headroom_produces_even_request_spacing(self) -> None:
        clock = FakeClock()
        limiter = RequestRateLimiter(
            requests_per_minute=120.0,
            headroom_fraction=0.5,
            clock=clock.now,
            sleeper=clock.sleep,
        )

        waits = [limiter.acquire() for _ in range(3)]

        self.assertEqual(waits, [0.0, 1.0, 1.0])
        self.assertEqual(clock.sleeps, [1.0, 1.0])
        self.assertEqual(limiter.effective_requests_per_minute, 60.0)
        self.assertEqual(limiter.minimum_interval_seconds, 1.0)

    def test_shared_limiter_reserves_unique_slots_across_threads(self) -> None:
        recorded_sleeps = []
        limiter = RequestRateLimiter(
            requests_per_minute=60.0,
            headroom_fraction=1.0,
            clock=lambda: 0.0,
            sleeper=recorded_sleeps.append,
        )

        with ThreadPoolExecutor(max_workers=4) as executor:
            waits = list(executor.map(lambda _: limiter.acquire(), range(4)))

        self.assertEqual(sorted(waits), [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(sorted(recorded_sleeps), [1.0, 2.0, 3.0])

    def test_defer_delays_future_request_slots(self) -> None:
        clock = FakeClock()
        limiter = RequestRateLimiter(
            requests_per_minute=60.0,
            headroom_fraction=1.0,
            clock=clock.now,
            sleeper=clock.sleep,
        )

        self.assertEqual(limiter.acquire(), 0.0)
        limiter.defer(5.0)
        self.assertEqual(limiter.acquire(), 5.0)
        self.assertEqual(clock.sleeps, [5.0])

    def test_wait_time_is_written_to_execution_metrics(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            output_path = Path(temp_dir) / "metrics.json"
            metrics = ExecutionMetrics(
                run_name="rate-limit-test",
                command="test",
                output_path=output_path,
            )
            clock = FakeClock()
            limiter = RequestRateLimiter(
                requests_per_minute=60.0,
                headroom_fraction=1.0,
                metrics=metrics,
                clock=clock.now,
                sleeper=clock.sleep,
            )

            limiter.acquire()
            limiter.acquire()
            metrics.finish()

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            wait_entry = payload["waits"]["openai.request_rate_limit"]

            self.assertEqual(wait_entry["events"], 1)
            self.assertEqual(wait_entry["elapsed_seconds"], 1.0)
            self.assertEqual(wait_entry["max_elapsed_seconds"], 1.0)
            self.assertEqual(payload["wait_totals"]["events"], 1)
            self.assertEqual(payload["wait_totals"]["elapsed_seconds"], 1.0)

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RequestRateLimiter(requests_per_minute=0)

        with self.assertRaises(ValueError):
            RequestRateLimiter(
                requests_per_minute=60,
                headroom_fraction=0,
            )

        with self.assertRaises(ValueError):
            RequestRateLimiter(
                requests_per_minute=60,
                headroom_fraction=1.01,
            )


if __name__ == "__main__":
    unittest.main()
