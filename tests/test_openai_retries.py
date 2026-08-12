import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from openai import APIConnectionError, RateLimitError

from src.execution_metrics import ExecutionMetrics
from src.providers.openai_client import OpenAIClientWrapper


def fake_response():
    return SimpleNamespace(
        output_text="Step 1: Answer the question.\nFinal answer: Yes",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            input_tokens_details=None,
            output_tokens_details=None,
        ),
    )


def connection_error():
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    return APIConnectionError(request=request)


def rate_limit_error(retry_after=None):
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    headers = {}
    if retry_after is not None:
        headers["retry-after"] = str(retry_after)
    response = httpx.Response(
        429,
        request=request,
        headers=headers,
    )
    return RateLimitError(
        "rate limited",
        response=response,
        body=None,
    )


class SequencedResponses:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def create(self, **request):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeOpenAIClient:
    def __init__(self, outcomes):
        self.responses = SequencedResponses(outcomes)


class FakeRateLimiter:
    def __init__(self):
        self.acquire_calls = 0
        self.deferred_seconds = []

    def acquire(self):
        self.acquire_calls += 1
        return 0.0

    def defer(self, seconds):
        self.deferred_seconds.append(seconds)


class OpenAIRetryTests(unittest.TestCase):
    def make_metrics(self, output_path):
        return ExecutionMetrics(
            run_name="retry-test",
            command="generate-traces",
            output_path=output_path,
        )

    def test_retryable_failure_is_retried_and_measured(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            metrics_path = Path(temp_dir) / "metrics.json"
            metrics = self.make_metrics(metrics_path)
            fake_client = FakeOpenAIClient(
                [connection_error(), fake_response()]
            )

            with (
                patch(
                    "src.providers.openai_client.OpenAI",
                    return_value=fake_client,
                ) as openai_constructor,
                patch("src.providers.openai_client.sleep") as sleep_mock,
            ):
                client = OpenAIClientWrapper(
                    model="fake-model",
                    metrics=metrics,
                    request_timeout_seconds=9.0,
                    max_retries=2,
                    retry_base_delay_seconds=0.25,
                    retry_jitter_fraction=0.0,
                )
                trace, answer = client.generate_trace(
                    "Is this an offline test?",
                    task="strategyqa",
                )

            metrics.finish(records_processed=1)
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            operation = payload["operations"]["openai.trace_generation"]

            self.assertIn("Final answer: Yes", trace)
            self.assertEqual(answer, "Yes")
            self.assertEqual(fake_client.responses.calls, 2)
            openai_constructor.assert_called_once_with(
                timeout=9.0,
                max_retries=0,
            )
            sleep_mock.assert_called_once_with(0.25)
            self.assertEqual(operation["logical_calls"], 1)
            self.assertEqual(operation["attempts"], 2)
            self.assertEqual(operation["retries"], 1)
            self.assertEqual(operation["successes"], 1)
            self.assertEqual(operation["failures"], 1)
            self.assertEqual(operation["retry_delay_seconds"], 0.25)
            self.assertEqual(payload["totals"]["retries"], 1)

    def test_nonretryable_failure_is_not_retried(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            metrics_path = Path(temp_dir) / "metrics.json"
            metrics = self.make_metrics(metrics_path)
            fake_client = FakeOpenAIClient([ValueError("bad response")])

            with (
                patch(
                    "src.providers.openai_client.OpenAI",
                    return_value=fake_client,
                ),
                patch("src.providers.openai_client.sleep") as sleep_mock,
            ):
                client = OpenAIClientWrapper(
                    model="fake-model",
                    metrics=metrics,
                    max_retries=2,
                )
                with self.assertRaisesRegex(ValueError, "bad response"):
                    client.generate_trace(
                        "Is this an offline test?",
                        task="strategyqa",
                    )

            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            operation = payload["operations"]["openai.trace_generation"]

            self.assertEqual(fake_client.responses.calls, 1)
            sleep_mock.assert_not_called()
            self.assertEqual(operation["logical_calls"], 1)
            self.assertEqual(operation["attempts"], 1)
            self.assertEqual(operation["retries"], 0)
            self.assertEqual(operation["failures"], 1)

    def test_retryable_failure_stops_after_configured_limit(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            metrics_path = Path(temp_dir) / "metrics.json"
            metrics = self.make_metrics(metrics_path)
            fake_client = FakeOpenAIClient(
                [connection_error(), connection_error(), connection_error()]
            )

            with (
                patch(
                    "src.providers.openai_client.OpenAI",
                    return_value=fake_client,
                ),
                patch("src.providers.openai_client.sleep") as sleep_mock,
            ):
                client = OpenAIClientWrapper(
                    model="fake-model",
                    metrics=metrics,
                    max_retries=2,
                    retry_base_delay_seconds=0.5,
                    retry_jitter_fraction=0.0,
                )
                with self.assertRaises(APIConnectionError):
                    client.generate_trace(
                        "Is this an offline test?",
                        task="strategyqa",
                    )

            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            operation = payload["operations"]["openai.trace_generation"]

            self.assertEqual(fake_client.responses.calls, 3)
            self.assertEqual(
                [call.args[0] for call in sleep_mock.call_args_list],
                [0.5, 1.0],
            )
            self.assertEqual(operation["logical_calls"], 1)
            self.assertEqual(operation["attempts"], 3)
            self.assertEqual(operation["retries"], 2)
            self.assertEqual(operation["successes"], 0)
            self.assertEqual(operation["failures"], 3)
            self.assertEqual(operation["retry_delay_seconds"], 1.5)

    def test_rate_limiter_is_acquired_for_every_attempt(self):
        fake_client = FakeOpenAIClient(
            [connection_error(), fake_response()]
        )
        rate_limiter = FakeRateLimiter()

        with (
            patch(
                "src.providers.openai_client.OpenAI",
                return_value=fake_client,
            ),
            patch("src.providers.openai_client.sleep"),
        ):
            client = OpenAIClientWrapper(
                model="fake-model",
                rate_limiter=rate_limiter,
                max_retries=1,
                retry_base_delay_seconds=0.0,
                retry_jitter_fraction=0.0,
            )
            client.generate_trace(
                "Is this an offline test?",
                task="strategyqa",
            )

        self.assertEqual(rate_limiter.acquire_calls, 2)
        self.assertEqual(rate_limiter.deferred_seconds, [])

    def test_rate_limit_response_defers_shared_limiter(self):
        fake_client = FakeOpenAIClient(
            [rate_limit_error(retry_after=2.5), fake_response()]
        )
        rate_limiter = FakeRateLimiter()

        with (
            patch(
                "src.providers.openai_client.OpenAI",
                return_value=fake_client,
            ),
            patch("src.providers.openai_client.sleep") as sleep_mock,
        ):
            client = OpenAIClientWrapper(
                model="fake-model",
                rate_limiter=rate_limiter,
                max_retries=1,
                retry_base_delay_seconds=0.25,
                retry_jitter_fraction=0.0,
            )
            client.generate_trace(
                "Is this an offline test?",
                task="strategyqa",
            )

        self.assertEqual(rate_limiter.acquire_calls, 2)
        self.assertEqual(rate_limiter.deferred_seconds, [2.5])
        sleep_mock.assert_called_once_with(2.5)

    def test_retry_jitter_changes_exponential_delay(self):
        fake_client = FakeOpenAIClient(
            [connection_error(), fake_response()]
        )

        with (
            patch(
                "src.providers.openai_client.OpenAI",
                return_value=fake_client,
            ),
            patch(
                "src.providers.openai_client.random.uniform",
                return_value=1.2,
            ) as uniform_mock,
            patch("src.providers.openai_client.sleep") as sleep_mock,
        ):
            client = OpenAIClientWrapper(
                model="fake-model",
                max_retries=1,
                retry_base_delay_seconds=2.0,
                retry_jitter_fraction=0.2,
            )
            client.generate_trace(
                "Is this an offline test?",
                task="strategyqa",
            )

        uniform_mock.assert_called_once_with(0.8, 1.2)
        sleep_mock.assert_called_once_with(2.4)

    def test_invalid_retry_jitter_is_rejected(self):
        with self.assertRaises(ValueError):
            OpenAIClientWrapper(retry_jitter_fraction=-0.01)

        with self.assertRaises(ValueError):
            OpenAIClientWrapper(retry_jitter_fraction=1.01)


if __name__ == "__main__":
    unittest.main()
