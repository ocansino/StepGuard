import importlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from src.execution_metrics import ExecutionMetrics


ADAPTER_MODULE = "src.providers.mistral_client"
ADAPTER_AVAILABLE = importlib.util.find_spec(ADAPTER_MODULE) is not None


class FakeMistralError(Exception):
    def __init__(self, status_code, *, headers=None, message="Mistral error"):
        super().__init__(message)
        self.status_code = status_code
        self.headers = headers or {}
        self.message = message


class FakeBackoffStrategy:
    def __init__(
        self,
        initial_interval,
        max_interval,
        exponent,
        max_elapsed_time,
    ):
        self.initial_interval = initial_interval
        self.max_interval = max_interval
        self.exponent = exponent
        self.max_elapsed_time = max_elapsed_time


class FakeRetryConfig:
    def __init__(self, strategy, backoff, retry_connection_errors):
        self.strategy = strategy
        self.backoff = backoff
        self.retry_connection_errors = retry_connection_errors


def load_adapter_with_fake_sdk():
    mistralai_package = ModuleType("mistralai")
    client_module = ModuleType("mistralai.client")
    utils_module = ModuleType("mistralai.client.utils")

    client_module.Mistral = object
    client_module.errors = SimpleNamespace(MistralError=FakeMistralError)
    utils_module.BackoffStrategy = FakeBackoffStrategy
    utils_module.RetryConfig = FakeRetryConfig

    fake_modules = {
        "mistralai": mistralai_package,
        "mistralai.client": client_module,
        "mistralai.client.utils": utils_module,
    }

    sys.modules.pop(ADAPTER_MODULE, None)
    with patch.dict(sys.modules, fake_modules):
        return importlib.import_module(ADAPTER_MODULE)


if ADAPTER_AVAILABLE:
    mistral_client = load_adapter_with_fake_sdk()
else:
    mistral_client = None


def fake_response(text, *, prompt_tokens=10, completion_tokens=5):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


class SequencedChat:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def complete(self, **request):
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeMistralSDK:
    def __init__(self, outcomes):
        self.chat = SequencedChat(outcomes)


class FakeRateLimiter:
    def __init__(self):
        self.acquire_calls = 0
        self.deferred_seconds = []

    def acquire(self):
        self.acquire_calls += 1
        return 0.0

    def defer(self, seconds):
        self.deferred_seconds.append(seconds)


@unittest.skipUnless(
    ADAPTER_AVAILABLE,
    "Apply src/providers/mistral_client.py to activate these contract tests",
)
class MistralClientContractTests(unittest.TestCase):
    def test_all_stepguard_operations_use_pinned_model_and_expected_modes(self):
        sdk = FakeMistralSDK(
            [
                fake_response(
                    "Step 1: Reason carefully.\nFinal answer: yes"
                ),
                fake_response(
                    json.dumps(
                        {
                            "results": [
                                {
                                    "step_index": 0,
                                    "verdict": "valid",
                                    "p_wrong": 0.05,
                                }
                            ]
                        }
                    )
                ),
                fake_response(
                    "Step 2: Repair the suffix.\nFinal answer: no"
                ),
                fake_response(
                    json.dumps(
                        {
                            "prefer_repaired": True,
                            "original_answer_support": 0.4,
                            "repaired_answer_support": 0.8,
                            "regression_risk": 0.1,
                            "confidence": 0.9,
                            "reason": "The repair is better supported.",
                        }
                    )
                ),
            ]
        )

        with (
            patch.dict(os.environ, {"MISTRAL_API_KEY": "offline-test-key"}),
            patch.object(
                mistral_client,
                "Mistral",
                return_value=sdk,
            ) as constructor,
        ):
            client = mistral_client.MistralClientWrapper()
            trace, answer = client.generate_trace(
                "Is this a contract test?",
                task="strategyqa",
            )
            judgments = client.judge_steps(
                "Is this a contract test?",
                ["Step 1: Reason carefully."],
                task="strategyqa",
            )
            suffix, repaired_answer = client.repair_suffix(
                "Is this a contract test?",
                ["Step 1: Preserve this."],
                2,
                task="strategyqa",
            )
            acceptance = client.judge_repair_candidate(
                question="Is this a contract test?",
                original_trace=trace,
                original_answer=answer,
                repaired_trace=suffix,
                repaired_answer=repaired_answer,
                task="strategyqa",
            )

        constructor.assert_called_once()
        constructor_kwargs = constructor.call_args.kwargs
        self.assertEqual(constructor_kwargs["api_key"], "offline-test-key")
        self.assertEqual(
            constructor_kwargs["retry_config"].strategy,
            "none",
        )

        self.assertEqual(answer, "yes")
        self.assertEqual(judgments[0]["p_wrong"], 0.05)
        self.assertEqual(repaired_answer, "no")
        self.assertTrue(acceptance["prefer_repaired"])

        self.assertEqual(len(sdk.chat.requests), 4)
        self.assertTrue(
            all(
                request["model"] == "mistral-small-2603"
                for request in sdk.chat.requests
            )
        )
        self.assertEqual(
            [request["temperature"] for request in sdk.chat.requests],
            [0.2, 0.0, 0.2, 0.0],
        )
        self.assertTrue(
            all(
                request["timeout_ms"] == 120_000
                for request in sdk.chat.requests
            )
        )
        self.assertEqual(
            sdk.chat.requests[0]["response_format"],
            {"type": "text"},
        )
        self.assertEqual(
            sdk.chat.requests[1]["response_format"],
            {"type": "json_object"},
        )
        self.assertEqual(
            sdk.chat.requests[2]["response_format"],
            {"type": "text"},
        )
        self.assertEqual(
            sdk.chat.requests[3]["response_format"],
            {"type": "json_object"},
        )

    def test_generation_records_mistral_usage_without_network_access(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            metrics_path = Path(temp_dir) / "metrics.json"
            metrics = ExecutionMetrics(
                run_name="mistral-provider-test",
                command="generate-traces",
                output_path=metrics_path,
            )
            sdk = FakeMistralSDK(
                [
                    fake_response(
                        "Step 1: Add the values.\nFinal answer: 4",
                        prompt_tokens=12,
                        completion_tokens=8,
                    )
                ]
            )

            with (
                patch.dict(
                    os.environ,
                    {"MISTRAL_API_KEY": "offline-test-key"},
                ),
                patch.object(mistral_client, "Mistral", return_value=sdk),
            ):
                client = mistral_client.MistralClientWrapper(
                    metrics=metrics,
                )
                _, answer = client.generate_trace("What is 2 + 2?")

            metrics.finish(records_processed=1)
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            operation = payload["operations"]["mistral.trace_generation"]

            self.assertEqual(answer, "4")
            self.assertEqual(operation["logical_calls"], 1)
            self.assertEqual(operation["attempts"], 1)
            self.assertEqual(operation["input_tokens"], 12)
            self.assertEqual(operation["output_tokens"], 8)
            self.assertEqual(operation["total_tokens"], 20)

    def test_429_is_retried_measured_and_deferred(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            metrics_path = Path(temp_dir) / "metrics.json"
            metrics = ExecutionMetrics(
                run_name="mistral-retry-test",
                command="generate-traces",
                output_path=metrics_path,
            )
            sdk = FakeMistralSDK(
                [
                    FakeMistralError(
                        429,
                        headers={"retry-after": "2.5"},
                    ),
                    fake_response(
                        "Step 1: Retry safely.\nFinal answer: yes"
                    ),
                ]
            )
            limiter = FakeRateLimiter()

            with (
                patch.dict(
                    os.environ,
                    {"MISTRAL_API_KEY": "offline-test-key"},
                ),
                patch.object(mistral_client, "Mistral", return_value=sdk),
                patch.object(mistral_client, "sleep") as sleep_mock,
            ):
                client = mistral_client.MistralClientWrapper(
                    metrics=metrics,
                    rate_limiter=limiter,
                    max_retries=1,
                    retry_base_delay_seconds=0.25,
                    retry_jitter_fraction=0.0,
                )
                client.generate_trace(
                    "Is this a retry test?",
                    task="strategyqa",
                )

            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            operation = payload["operations"]["mistral.trace_generation"]

            self.assertEqual(len(sdk.chat.requests), 2)
            self.assertEqual(limiter.acquire_calls, 2)
            self.assertEqual(limiter.deferred_seconds, [2.5])
            sleep_mock.assert_called_once_with(2.5)
            self.assertEqual(operation["attempts"], 2)
            self.assertEqual(operation["retries"], 1)
            self.assertEqual(operation["failures"], 1)
            self.assertEqual(operation["successes"], 1)
            self.assertEqual(operation["error_types"], {"HTTP429": 1})

    def test_nonretryable_400_is_not_retried(self):
        sdk = FakeMistralSDK([FakeMistralError(400)])

        with (
            patch.dict(os.environ, {"MISTRAL_API_KEY": "offline-test-key"}),
            patch.object(mistral_client, "Mistral", return_value=sdk),
            patch.object(mistral_client, "sleep") as sleep_mock,
        ):
            client = mistral_client.MistralClientWrapper(max_retries=2)
            with self.assertRaises(FakeMistralError):
                client.generate_trace("Bad request test")

        self.assertEqual(len(sdk.chat.requests), 1)
        sleep_mock.assert_not_called()

    def test_missing_api_key_fails_before_client_construction(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(mistral_client, "Mistral") as constructor,
        ):
            with self.assertRaisesRegex(RuntimeError, "MISTRAL_API_KEY"):
                mistral_client.MistralClientWrapper()

        constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
