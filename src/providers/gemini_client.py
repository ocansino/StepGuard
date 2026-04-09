from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Tuple

from google import genai
from google.genai import types


_SYSTEM_INSTRUCTIONS = """You are a reasoning assistant.

Output MUST follow this exact format:

Step 1: ...
Step 2: ...
...
Final answer: <answer>

Rules:
- One step per line.
- No extra commentary outside the steps.
- The final line MUST start with exactly: "Final answer: "
"""


def extract_final_answer(text: str) -> str:
    m = re.search(r"^Final answer:\s*(.*)\s*$", text, flags=re.MULTILINE)
    if m:
        return m.group(1).strip()
    # Fallback: last non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


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
        temperature: float = 0.2,
        max_output_tokens: int = 800,
    ) -> Tuple[str, str]:
        prompt = _SYSTEM_INSTRUCTIONS + "\n\nProblem:\n" + question.strip()

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