from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from time import perf_counter, sleep
from typing import Any, Dict, List, Optional, Tuple


from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from ..task_profiles import get_task_profile
from ..execution_metrics import ExecutionMetrics
from ..rate_limiter import RequestRateLimiter


def extract_final_answer(text: str) -> str:
    m = re.search(r"^Final answer:\s*(.*)\s*$", text, flags=re.MULTILINE)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""

def _read_value(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _extract_openai_usage(response: Any) -> Dict[str, int]:
    usage = _read_value(response, "usage")
    if usage is None:
        return {}

    input_details = _read_value(usage, "input_tokens_details")
    output_details = _read_value(usage, "output_tokens_details")

    candidates = {
        "input_tokens": _read_value(usage, "input_tokens"),
        "output_tokens": _read_value(usage, "output_tokens"),
        "total_tokens": _read_value(usage, "total_tokens"),
        "cached_input_tokens": _read_value(
            input_details,
            "cached_tokens",
        ),
        "reasoning_output_tokens": _read_value(
            output_details,
            "reasoning_tokens",
        ),
    }

    return {
        key: int(value)
        for key, value in candidates.items()
        if isinstance(value, (int, float))
    }

def _is_retryable_openai_error(error: Exception) -> bool:
    if isinstance(
        error,
        (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
        ),
    ):
        return True

    if isinstance(error, APIStatusError):
        status_code = getattr(error, "status_code", None)
        return (
            status_code in {408, 409, 429}
            or (
                isinstance(status_code, int)
                and status_code >= 500
            )
        )

    return False

def _is_rate_limit_error(error: Exception) -> bool:
    if isinstance(error, RateLimitError):
        return True

    if isinstance(error, APIStatusError):
        return getattr(error, "status_code", None) == 429

    return False


def _extract_retry_after_seconds(
    error: Exception,
) -> Optional[float]:
    if not _is_rate_limit_error(error):
        return None

    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)

    if headers is None:
        return None

    retry_after = headers.get("retry-after")

    if retry_after is None:
        return None

    try:
        seconds = float(retry_after)
    except (TypeError, ValueError):
        return None

    if seconds < 0:
        return None

    return seconds

@dataclass
class OpenAIClientWrapper:
    model: str = "gpt-5.4-mini"
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

        # StepGuard owns retry behavior so that attempts are bounded and
        # visible in its execution metrics. Disable the SDK's internal retries.
        self._client = OpenAI(
            timeout=self.request_timeout_seconds,
            max_retries=0,
        )

    def _create_response(
        self,
        operation: str,
        **request: Any,
    ) -> Any:
        total_attempts = self.max_retries + 1

        for attempt_index in range(total_attempts):
            if self.rate_limiter is not None:
                self.rate_limiter.acquire()

            started = perf_counter()

            try:
                response = self._client.responses.create(**request)
            except Exception as error:
                will_retry = (
                    _is_retryable_openai_error(error)
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
                    retry_after = _extract_retry_after_seconds(
                        error
                    )
                    cooldown_delay = retry_delay

                    if retry_after is not None:
                        cooldown_delay = max(
                            cooldown_delay,
                            retry_after,
                        )

                    if cooldown_delay > 0:
                        self.rate_limiter.defer(
                            cooldown_delay
                        )

                if self.metrics is not None:
                    self.metrics.record_operation(
                        provider="openai",
                        operation=operation,
                        elapsed_seconds=(
                            perf_counter() - started
                        ),
                        success=False,
                        error_type=type(error).__name__,
                        logical_call=(attempt_index == 0),
                        retry_delay_seconds=retry_delay,
                    )

                if not will_retry:
                    raise

                sleep(retry_delay)
                continue

            if self.metrics is not None:
                self.metrics.record_operation(
                    provider="openai",
                    operation=operation,
                    elapsed_seconds=(
                        perf_counter() - started
                    ),
                    success=True,
                    usage=_extract_openai_usage(response),
                    logical_call=(attempt_index == 0),
                )

            return response

        raise RuntimeError(
            "OpenAI request exhausted without a result"
        ) 

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
            minimum_factor = 1.0 - self.retry_jitter_fraction
            maximum_factor = 1.0 + self.retry_jitter_fraction
            delay *= random.uniform(
                minimum_factor,
                maximum_factor,
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
        prompt = profile.build_generation_prompt(question, provider="openai")

        resp = self._create_response(
            "trace_generation",
            model=self.model,
            input=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        text = (resp.output_text or "").strip()
        answer = extract_final_answer(text)
        return text, answer

    def judge_steps(
        self,
        question: str,
        steps: List[str],
        *,
        task: str = "math",
        max_output_tokens: int = 1200,
    ) -> List[Dict[str, Any]]:
        """
        Returns: [{"step_index": int, "verdict": "...", "p_wrong": float}, ...]
        """
        profile = get_task_profile(task)
        prompt = profile.build_verifier_prompt(question, steps, provider="openai")

        resp = self._create_response(
            "step_verification",
            model=self.model,
            input=prompt,
            temperature=0.0,
            max_output_tokens=max_output_tokens,
        )

        text = (resp.output_text or "").strip()

        # Parse JSON (fail-soft handled in caller)
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

        obj = json.loads(text)
        results = obj.get("results", [])
        if not isinstance(results, list):
            raise ValueError("Judge returned invalid JSON: 'results' is not a list")
        return results

    def repair_suffix(
        self,
        question: str,
        prefix_steps: list[str],
        next_step_number: int,
        *,
        task: str = "math",
        temperature: float = 0.2,
        max_output_tokens: int = 800,
    ) -> tuple[str, str]:
        """
        Returns: (suffix_text, repaired_answer)
        suffix_text should include Step <next_step_number>: ... lines and end with Final answer: ...
        """
        profile = get_task_profile(task)
        prompt = profile.build_repair_prompt(
            question,
            prefix_steps,
            next_step_number,
        )

        resp = self._create_response(
            "suffix_repair",
            model=self.model,
            input=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        text = (resp.output_text or "").strip()
        answer = extract_final_answer(text)
        return text, answer

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

        resp = self._create_response(
            "acceptance_judging",
            model=self.model,
            input=prompt,
            temperature=0.0,
            max_output_tokens=max_output_tokens,
        )

        text = (resp.output_text or "").strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

        return json.loads(text)
