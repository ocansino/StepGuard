from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from openai import OpenAI


_SYSTEM_INSTRUCTIONS = """You are solving a problem.

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
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


@dataclass
class OpenAIClientWrapper:
    model: str = "gpt-5.4-mini"

    def __post_init__(self) -> None:
        self._client = OpenAI()

    def generate_trace(
        self,
        question: str,
        *,
        temperature: float = 0.2,
        max_output_tokens: int = 800,
    ) -> Tuple[str, str]:
        prompt = _SYSTEM_INSTRUCTIONS + "\n\nProblem:\n" + question.strip()

        resp = self._client.responses.create(
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
        max_output_tokens: int = 1200,
    ) -> List[Dict[str, Any]]:
        """
        Returns: [{"step_index": int, "verdict": "...", "p_wrong": float}, ...]
        """
        # Keep JSON small and robust (no notes).
        judge_instructions = """You are a strict step-by-step verifier for math reasoning.

Given a QUESTION and numbered STEPS, evaluate each step's validity given the question and earlier steps.

Return JSON ONLY in this schema (minified, one line):
{"results":[{"step_index":0,"verdict":"valid|invalid|uncertain","p_wrong":0.0}, ...]}

Rules:
- p_wrong is in [0,1] and represents probability the step is wrong.
- Do not add extra keys.
"""

        steps_block = "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps)])
        prompt = (
            judge_instructions
            + "\n\nQUESTION:\n" + question.strip()
            + "\n\nSTEPS:\n" + steps_block
        )

        resp = self._client.responses.create(
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
        temperature: float = 0.2,
        max_output_tokens: int = 800,
    ) -> tuple[str, str]:
        """
        Returns: (suffix_text, repaired_answer)
        suffix_text should include Step <next_step_number>: ... lines and end with Final answer: ...
        """
        prefix_block = "\n".join(prefix_steps)

        prompt = f"""You are fixing a step-by-step solution.

    You are given:
    1) The QUESTION.
    2) The VERIFIED-CORRECT PREFIX steps that must be kept.
    Your job: continue the solution starting at Step {next_step_number} and finish.

    Output MUST follow this exact format:
    Step {next_step_number}: ...
    Step {next_step_number + 1}: ...
    ...
    Final answer: <answer>

    Rules:
    - Do NOT repeat or modify the prefix steps.
    - Do NOT introduce new assumptions.
    - Each step must be logically valid.
    - Keep steps concise.

    QUESTION:
    {question.strip()}

    PREFIX (do not change):
    {prefix_block}
    """

        resp = self._client.responses.create(
            model=self.model,
            input=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        text = (resp.output_text or "").strip()
        answer = extract_final_answer(text)
        return text, answer