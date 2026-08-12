from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any

from google import genai
from google.genai import types

from ..task_profiles import get_task_profile


def extract_final_answer(text: str) -> str:
    m = re.search(r"^Final answer:\s*(.*)\s*$", text, flags=re.MULTILINE)
    if m:
        return m.group(1).strip()
    # Fallback: last non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""

def _extract_json(text: str) -> Dict[str, Any]:
    """
    Tries to parse JSON from a model response.
    If the response contains extra text, attempts to extract the first {...} block.
    """
    text = text.strip()
    if not text:
        raise ValueError("Empty judge response")
    # Strip common markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass
        # fallback: extract the first JSON object block

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError(f"Could not parse JSON from judge response: {text[:200]}")

@dataclass
class GeminiClient:
    model: str

    def __post_init__(self) -> None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in the environment.")
        self._client = genai.Client(api_key=api_key)

    def generate_trace(
        self,
        question: str,
        *,
        task: str = "math",
        temperature: float = 0.2,
        max_output_tokens: int = 800,
    ) -> Tuple[str, str]:
        profile = get_task_profile(task)
        prompt = profile.build_generation_prompt(question, provider="gemini")

        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )

        text = (resp.text or "").strip()
        answer = extract_final_answer(text)
        return text, answer
    def judge_steps(
        self,
        question: str,
        steps: list[str],
        *,
        task: str = "math",
        temperature: float = 0.0,
        max_output_tokens: int = 2000,
    ) -> list[dict]:
        """
        Returns a list of dicts, one per step:
          {"step_index": int, "verdict": "valid|invalid|uncertain", "p_wrong": float, "notes": str}
        """
        profile = get_task_profile(task)
        prompt = profile.build_verifier_prompt(question, steps, provider="gemini")

        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                response_mime_type="application/json",
            ),
        )

        text = (resp.text or "").strip()
        print("JUDGE RAW OUTPUT:\n", text)
        obj = _extract_json(text)

        results = obj.get("results", [])
        if not isinstance(results, list):
            raise ValueError("Gemini judge returned invalid JSON: 'results' is not a list")
        return results
