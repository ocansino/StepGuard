from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from time import perf_counter, sleep
from typing import Any, Dict, List, Optional, Tuple

import httpx

try:
    from mistralai.client import Mistral
    from mistralai.client.utils import BackoffStrategy, RetryConfig
except ImportError:
    Mistral = None
    BackoffStrategy = None
    RetryConfig = None

from ..execution_metrics import ExecutionMetrics
from ..rate_limiter import RequestRateLimiter
from ..task_profiles import get_task_profile


def extract_final_answer(text: str) -> str:
    match = re.search(
        r"^Final answer:\s*(.*)\s*$",
        text,
        flags=re.MULTILINE,
    )
    if match:
        return match.group(1).strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _read_value(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _extract_mistral_usage(response: Any) -> Dict[str, int]:
    usage = _read_value(response, "usage")
    if usage is None:
        return {}

    candidates = {
        "input_tokens": _read_value(usage, "prompt_tokens"),
        "output_tokens": _read_value(usage, "completion_tokens"),
        "total_tokens": _read_value(usage, "total_tokens"),
    }

    return {
        key: int(value)
        for key, value in candidates.items()
        if isinstance(value, (int, float))
    }


def _extract_mistral_text(response: Any) -> str:
    choices = _read_value(response, "choices")
    if not isinstance(choices, list) or not choices:
        return ""

    message = _read_value(choices[0], "message")
    content = _read_value(message, "content")

    if isinstance(content, str):
        return content.strip()

    if not isinstance(content, list):
        return ""

    parts: List[str] = []

    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue

        text = _read_value(item, "text")
        if isinstance(text, str):
            parts.append(text)

    return "\n".join(parts).strip()


def _parse_json_object(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Mistral returned JSON that is not an object")

    return parsed


def _status_code(error: Exception) -> Optional[int]:
    value = getattr(error, "status_code", None)
    return value if isinstance(value, int) else None


def _is_retryable_mistral_error(error: Exception) -> bool:
    if isinstance(error, (httpx.RequestError, httpx.TimeoutException)):
        return True

    status_code = _status_code(error)
    return (
        status_code in {408, 409, 429}
        or (
            isinstance(status_code, int)
            and status_code >= 500
        )
    )


def _is_rate_limit_error(error: Exception) -> bool:
    return _status_code(error) == 429


def _extract_retry_after_seconds(
    error: Exception,
) -> Optional[float]:
    if not _is_rate_limit_error(error):
        return None

    headers = getattr(error, "headers", None)

    if headers is None:
        raw_response = getattr(error, "raw_response", None)
        headers = getattr(raw_response, "headers", None)

    if headers is None:
        return None

    retry_after = headers.get("retry-after")
    if retry_after is None:
        return None

    try:
        seconds = float(retry_after)
    except (TypeError, ValueError):
        return None

    return seconds if seconds >= 0 else None


def _metric_error_type(error: Exception) -> str:
    status_code = _status_code(error)
    if status_code is not None:
        return f"HTTP{status_code}"

    return type(error).__name__


@dataclass
class MistralClientWrapper:
    model: str = "mistral-small-2603"
    metrics: Optional[ExecutionMetrics] = None
    request_timeout_seconds: float = 120.0
    max_retries: int = 2
    retry_base_delay_seconds: float = 1.0
    retry_jitter_fraction: float = 0.2
    rate_limiter: Optional[RequestRateLimiter] = None

    def __post_init__(self) -> None:
        if self.request_timeout_seconds <= 0:
            raise ValueError(
                "request_timeout_seconds must be greater than zero"
            )

        if (
            not isinstance(self.max_retries, int)
            or isinstance(self.max_retries, bool)
            or self.max_retries < 0
            or self.max_retries > 10
        ):
            raise ValueError(
                "max_retries must be an integer between 0 and 10"
            )

        if self.retry_base_delay_seconds < 0:
            raise ValueError(
                "retry_base_delay_seconds must not be negative"
            )

        if not 0 <= self.retry_jitter_fraction <= 1:
            raise ValueError(
                "retry_jitter_fraction must be between 0 and 1"
            )

        api_key = os.environ.get("MISTRAL_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "MISTRAL_API_KEY must be set in the process environment"
            )

        if (
            Mistral is None
            or BackoffStrategy is None
            or RetryConfig is None
        ):
            raise RuntimeError(
                "The Mistral provider requires the mistralai package"
            )

        # StepGuard owns retry behavior so every attempt and 429 is measured.
        retry_config = RetryConfig(
            "none",
            BackoffStrategy(1, 1, 1.0, 0),
            False,
        )

        self._client = Mistral(
            api_key=api_key,
            retry_config=retry_config,
        )

    def _complete(
        self,
        operation: str,
        **request: Any,
    ) -> Any:
        request.setdefault(
            "timeout_ms",
            int(self.request_timeout_seconds * 1000),
        )
        total_attempts = self.max_retries + 1

        for attempt_index in range(total_attempts):
            if self.rate_limiter is not None:
                self.rate_limiter.acquire()

            started = perf_counter()

            try:
                response = self._client.chat.complete(**request)
            except Exception as error:
                will_retry = (
                    _is_retryable_mistral_error(error)
                    and attempt_index < self.max_retries
                )
                retry_delay = 0.0

                if will_retry:
                    retry_delay = self._calculate_retry_delay(
                        attempt_index=attempt_index,
                        error=error,
                    )

                if (
                    self.rate_limiter is not None
                    and _is_rate_limit_error(error)
                ):
                    retry_after = _extract_retry_after_seconds(error)
                    cooldown_delay = retry_delay

                    if retry_after is not None:
                        cooldown_delay = max(
                            cooldown_delay,
                            retry_after,
                        )

                    if cooldown_delay > 0:
                        self.rate_limiter.defer(cooldown_delay)

                if self.metrics is not None:
                    self.metrics.record_operation(
                        provider="mistral",
                        operation=operation,
                        elapsed_seconds=perf_counter() - started,
                        success=False,
                        error_type=_metric_error_type(error),
                        logical_call=(attempt_index == 0),
                        retry_delay_seconds=retry_delay,
                    )

                if not will_retry:
                    raise

                sleep(retry_delay)
                continue

            if self.metrics is not None:
                self.metrics.record_operation(
                    provider="mistral",
                    operation=operation,
                    elapsed_seconds=perf_counter() - started,
                    success=True,
                    usage=_extract_mistral_usage(response),
                    logical_call=(attempt_index == 0),
                )

            return response

        raise RuntimeError("Mistral request exhausted without a result")

    def _calculate_retry_delay(
        self,
        *,
        attempt_index: int,
        error: Exception,
    ) -> float:
        delay = (
            self.retry_base_delay_seconds
            * (2 ** attempt_index)
        )

        if delay > 0 and self.retry_jitter_fraction > 0:
            delay *= random.uniform(
                1.0 - self.retry_jitter_fraction,
                1.0 + self.retry_jitter_fraction,
            )

        retry_after = _extract_retry_after_seconds(error)
        if retry_after is not None:
            delay = max(delay, retry_after)

        return delay

    def generate_trace(
        self,
        question: str,
        *,
        task: str = "math",
        temperature: float = 0.2,
        max_output_tokens: int = 800,
    ) -> Tuple[str, str]:
        profile = get_task_profile(task)
        prompt = profile.build_generation_prompt(
            question,
            provider="mistral",
        )

        response = self._complete(
            "trace_generation",
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_output_tokens,
            response_format={"type": "text"},
        )

        text = _extract_mistral_text(response)
        return text, extract_final_answer(text)

    def judge_steps(
        self,
        question: str,
        steps: List[str],
        *,
        task: str = "math",
        max_output_tokens: int = 1200,
    ) -> List[Dict[str, Any]]:
        profile = get_task_profile(task)
        prompt = profile.build_verifier_prompt(
            question,
            steps,
            provider="mistral",
        )

        response = self._complete(
            "step_verification",
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_output_tokens,
            response_format={"type": "json_object"},
        )

        parsed = _parse_json_object(
            _extract_mistral_text(response)
        )
        results = parsed.get("results", [])

        if not isinstance(results, list):
            raise ValueError(
                "Judge returned invalid JSON: 'results' is not a list"
            )

        return results

    def repair_suffix(
        self,
        question: str,
        prefix_steps: List[str],
        next_step_number: int,
        *,
        task: str = "math",
        temperature: float = 0.2,
        max_output_tokens: int = 800,
    ) -> Tuple[str, str]:
        profile = get_task_profile(task)
        prompt = profile.build_repair_prompt(
            question,
            prefix_steps,
            next_step_number,
        )

        response = self._complete(
            "suffix_repair",
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_output_tokens,
            response_format={"type": "text"},
        )

        text = _extract_mistral_text(response)
        return text, extract_final_answer(text)

    def judge_repair_candidate(
        self,
        *,
        question: str,
        original_trace: str,
        original_answer: str,
        repaired_trace: str,
        repaired_answer: str,
        task: str = "math",
        max_output_tokens: int = 800,
    ) -> Dict[str, Any]:
        profile = get_task_profile(task)
        prompt = profile.build_acceptance_prompt(
            question=question,
            original_trace=original_trace,
            original_answer=original_answer,
            repaired_trace=repaired_trace,
            repaired_answer=repaired_answer,
        )

        response = self._complete(
            "acceptance_judging",
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_output_tokens,
            response_format={"type": "json_object"},
        )

        return _parse_json_object(
            _extract_mistral_text(response)
        )