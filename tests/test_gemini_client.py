import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from google.genai import errors as genai_errors

from src import cli
from src.execution_metrics import ExecutionMetrics
from src.providers import gemini_client


REQUIRED_METHODS = (
    "generate_trace",
    "judge_steps",
    "repair_suffix",
    "judge_repair_candidate",
)
CONSTRUCTOR_PARAMETERS = inspect.signature(
    gemini_client.GeminiClient
).parameters
FULL_ADAPTER_READY = (
    all(
        hasattr(gemini_client.GeminiClient, method)
        for method in REQUIRED_METHODS
    )
    and "metrics" in CONSTRUCTOR_PARAMETERS
    and "rate_limiter" in CONSTRUCTOR_PARAMETERS
)
RATE_LIMITER_READY = hasattr(cli, "create_gemini_rate_limiter")


def fake_response(
    text,
    *,
    prompt_tokens=10,
    candidate_tokens=5,
    thought_tokens=0,
):
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=candidate_tokens,
            total_token_count=(
                prompt_tokens + candidate_tokens + thought_tokens
            ),
            cached_content_token_count=0,
            thoughts_token_count=thought_tokens,
        ),
    )


class SequencedModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def generate_content(self, **request):
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeGeminiSDK:
    def __init__(self, outcomes):
        self.models = SequencedModels(outcomes)


class FakeRateLimiter:
    def __init__(self):
        self.acquire_calls = 0
        self.deferred_seconds = []

    def acquire(self):
        self.acquire_calls += 1
        return 0.0

    def defer(self, seconds):
        self.deferred_seconds.append(seconds)


def rate_limit_error(retry_after="2.5"):
    request = httpx.Request(
        "POST",
        "https://generativelanguage.googleapis.com/v1beta/models",
    )
    response = httpx.Response(
        429,
        request=request,
        headers={"retry-after": retry_after},
    )
    return genai_errors.ClientError(
        429,
        {
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "message": "offline rate-limit test",
            }
        },
        response,
    )


@unittest.skipUnless(
    FULL_ADAPTER_READY,
    "Apply the Gemini provider replacement to activate these contract tests",
)
class GeminiClientContractTests(unittest.TestCase):
    def test_full_stepguard_contract_uses_stable_model_and_output_modes(self):
        sdk = FakeGeminiSDK(
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
            patch.dict(
                os.environ,
                {"GEMINI_API_KEY": "offline-test-key"},
                clear=True,
            ),
            patch.object(
                gemini_client.genai,
                "Client",
                return_value=sdk,
            ) as constructor,
        ):
            client = gemini_client.GeminiClient()
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
        self.assertEqual(
            constructor_kwargs["api_key"],
            "offline-test-key",
        )
        self.assertEqual(
            constructor_kwargs["http_options"].timeout,
            120_000,
        )
        self.assertEqual(
            constructor_kwargs[
                "http_options"
            ].retry_options.attempts,
            1,
        )

        self.assertEqual(answer, "yes")
        self.assertEqual(judgments[0]["p_wrong"], 0.05)
        self.assertEqual(repaired_answer, "no")
        self.assertTrue(acceptance["prefer_repaired"])

        self.assertEqual(len(sdk.models.requests), 4)
        self.assertTrue(
            all(
                request["model"] == "gemini-3.5-flash-lite"
                for request in sdk.models.requests
            )
        )
        self.assertEqual(
            [
                request["config"].temperature
                for request in sdk.models.requests
            ],
            [0.2, 0.0, 0.2, 0.0],
        )
        self.assertIsNone(
            sdk.models.requests[0]["config"].response_mime_type
        )
        self.assertEqual(
            sdk.models.requests[1]["config"].response_mime_type,
            "application/json",
        )
        self.assertIsNone(
            sdk.models.requests[2]["config"].response_mime_type
        )
        self.assertEqual(
            sdk.models.requests[3]["config"].response_mime_type,
            "application/json",
        )

    def test_generation_records_usage_without_network_access(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            metrics_path = Path(temp_dir) / "metrics.json"
            metrics = ExecutionMetrics(
                run_name="gemini-provider-test",
                command="generate-traces",
                output_path=metrics_path,
            )
            sdk = FakeGeminiSDK(
                [
                    fake_response(
                        "Step 1: Add the values.\nFinal answer: 4",
                        prompt_tokens=12,
                        candidate_tokens=8,
                        thought_tokens=3,
                    )
                ]
            )

            with (
                patch.dict(
                    os.environ,
                    {"GEMINI_API_KEY": "offline-test-key"},
                    clear=True,
                ),
                patch.object(
                    gemini_client.genai,
                    "Client",
                    return_value=sdk,
                ),
            ):
                client = gemini_client.GeminiClient(metrics=metrics)
                _, answer = client.generate_trace("What is 2 + 2?")

            metrics.finish(records_processed=1)
            payload = json.loads(
                metrics_path.read_text(encoding="utf-8")
            )
            operation = payload["operations"][
                "gemini.trace_generation"
            ]

            self.assertEqual(answer, "4")
            self.assertEqual(operation["logical_calls"], 1)
            self.assertEqual(operation["attempts"], 1)
            self.assertEqual(operation["input_tokens"], 12)
            self.assertEqual(operation["output_tokens"], 8)
            self.assertEqual(operation["total_tokens"], 23)
            self.assertEqual(operation["reasoning_output_tokens"], 3)

    def test_429_is_retried_measured_and_defers_shared_limiter(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            metrics_path = Path(temp_dir) / "metrics.json"
            metrics = ExecutionMetrics(
                run_name="gemini-retry-test",
                command="generate-traces",
                output_path=metrics_path,
            )
            sdk = FakeGeminiSDK(
                [
                    rate_limit_error(),
                    fake_response(
                        "Step 1: Retry safely.\nFinal answer: yes"
                    ),
                ]
            )
            limiter = FakeRateLimiter()

            with (
                patch.dict(
                    os.environ,
                    {"GEMINI_API_KEY": "offline-test-key"},
                    clear=True,
                ),
                patch.object(
                    gemini_client.genai,
                    "Client",
                    return_value=sdk,
                ),
                patch.object(gemini_client, "sleep") as sleep_mock,
            ):
                client = gemini_client.GeminiClient(
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

            payload = json.loads(
                metrics_path.read_text(encoding="utf-8")
            )
            operation = payload["operations"][
                "gemini.trace_generation"
            ]

            self.assertEqual(len(sdk.models.requests), 2)
            self.assertEqual(limiter.acquire_calls, 2)
            self.assertEqual(limiter.deferred_seconds, [2.5])
            sleep_mock.assert_called_once_with(2.5)
            self.assertEqual(operation["attempts"], 2)
            self.assertEqual(operation["retries"], 1)
            self.assertEqual(operation["failures"], 1)
            self.assertEqual(operation["successes"], 1)
            self.assertEqual(operation["error_types"], {"HTTP429": 1})

    def test_nonretryable_400_is_not_retried(self):
        request = httpx.Request(
            "POST",
            "https://generativelanguage.googleapis.com/v1beta/models",
        )
        response = httpx.Response(400, request=request)
        error = genai_errors.ClientError(
            400,
            {
                "error": {
                    "code": 400,
                    "status": "INVALID_ARGUMENT",
                    "message": "offline bad-request test",
                }
            },
            response,
        )
        sdk = FakeGeminiSDK([error])

        with (
            patch.dict(
                os.environ,
                {"GEMINI_API_KEY": "offline-test-key"},
                clear=True,
            ),
            patch.object(
                gemini_client.genai,
                "Client",
                return_value=sdk,
            ),
            patch.object(gemini_client, "sleep") as sleep_mock,
        ):
            client = gemini_client.GeminiClient(max_retries=2)
            with self.assertRaises(genai_errors.ClientError):
                client.generate_trace("Bad request test")

        self.assertEqual(len(sdk.models.requests), 1)
        sleep_mock.assert_not_called()

    def test_missing_api_key_fails_before_client_construction(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(
                gemini_client.genai,
                "Client",
            ) as constructor,
        ):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                gemini_client.GeminiClient()

        constructor.assert_not_called()


@unittest.skipUnless(
    RATE_LIMITER_READY,
    "Apply the Gemini CLI rate-limiter block to activate these tests",
)
class GeminiRateLimiterTests(unittest.TestCase):
    def test_account_limits_are_required(self):
        for missing in (
            "gemini_requests_per_minute",
            "gemini_tokens_per_minute",
            "gemini_requests_per_day",
        ):
            execution = {
                "gemini_requests_per_minute": 10,
                "gemini_tokens_per_minute": 250000,
                "gemini_requests_per_day": 250,
            }
            execution[missing] = None

            with self.assertRaisesRegex(ValueError, missing):
                cli.create_gemini_rate_limiter(
                    execution,
                    metrics=None,
                )

    def test_verified_rpm_controls_request_cadence(self):
        limiter = cli.create_gemini_rate_limiter(
            {
                "gemini_requests_per_minute": 10,
                "gemini_tokens_per_minute": 250000,
                "gemini_requests_per_day": 250,
                "rate_limit_headroom_fraction": 0.2,
            },
            metrics=None,
        )

        self.assertEqual(limiter.requests_per_minute, 10)
        self.assertEqual(limiter.effective_requests_per_minute, 2)
        self.assertEqual(
            limiter.wait_name,
            "gemini.request_rate_limit",
        )


if __name__ == "__main__":
    unittest.main()
