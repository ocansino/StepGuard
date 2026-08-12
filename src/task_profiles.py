from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")
_FINAL_ANSWER_PREFIX_RE = re.compile(r"^final\s+answer\s*:\s*", flags=re.IGNORECASE)


def normalize_math_answer(text: Optional[str]) -> Optional[str]:
    """Return the canonical numeric form used by the existing GSM8K evaluation."""
    if text is None:
        return None

    value = str(text).strip().lower()
    if not value:
        return None

    value = value.replace(",", "").replace("$", "")
    numbers = _NUM_RE.findall(value)
    if not numbers:
        return None

    try:
        number = float(numbers[-1])
    except ValueError:
        return None

    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.10f}".rstrip("0").rstrip(".")


def normalize_strategyqa_answer(text: Optional[str]) -> Optional[str]:
    """Normalize only an explicit yes/no value from the stored final-answer field."""
    if text is None:
        return None

    value = _FINAL_ANSWER_PREFIX_RE.sub("", str(text).strip())
    value = value.strip(" \t\r\n`*_\"'.,!?;:()[]{}")
    value = value.lower()

    if value in {"yes", "true"}:
        return "yes"
    if value in {"no", "false"}:
        return "no"
    return None


@dataclass(frozen=True)
class TaskProfile:
    name: str
    answer_normalizer: Callable[[Optional[str]], Optional[str]]

    def normalize_answer(self, text: Optional[str]) -> Optional[str]:
        return self.answer_normalizer(text)

    def answers_match(self, prediction: Optional[str], gold: Optional[str]) -> bool:
        normalized_prediction = self.normalize_answer(prediction)
        normalized_gold = self.normalize_answer(gold)
        return (
            normalized_prediction is not None
            and normalized_gold is not None
            and normalized_prediction == normalized_gold
        )

    def build_generation_prompt(self, question: str, *, provider: str = "openai") -> str:
        if self.name == "math":
            if provider == "gemini":
                instructions = """You are a reasoning assistant.

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
            else:
                instructions = """You are solving a problem.

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
            return instructions + "\n\nProblem:\n" + question.strip()

        return f"""You are answering a closed-book yes/no reasoning question.

Use concise numbered reasoning steps based only on generally available knowledge. Do not claim to browse, retrieve evidence, or consult external sources. Do not invent citations.

Output MUST follow this exact format:

Step 1: ...
Step 2: ...
...
Final answer: yes

Rules:
- One numbered reasoning step per line.
- No extra commentary outside the steps.
- The final line MUST be exactly "Final answer: yes" or "Final answer: no".

QUESTION:
{question.strip()}"""

    def build_verifier_prompt(
        self,
        question: str,
        steps: List[str],
        *,
        provider: str = "openai",
    ) -> str:
        if self.name == "math":
            if provider == "gemini":
                instructions = """You are a strict step-by-step verifier for math reasoning.

You will be given a QUESTION and a list of reasoning STEPS.
Your job is to assess each step ONLY based on:
- the question
- earlier steps
- basic math/logic

For each step, output:
- verdict: "valid", "invalid", or "uncertain"
- p_wrong: a number in [0,1] meaning probability the step is wrong/invalid

Rules:
- Output JSON ONLY. No markdown. No extra text.
- Output MUST be minified JSON on a single line (no pretty printing).

JSON schema:
{
  "results": [
    {"step_index": 0, "verdict": "...", "p_wrong": 0.0},
    ...
  ]
}
"""
            else:
                instructions = """You are a strict step-by-step verifier for math reasoning.

Given a QUESTION and numbered STEPS, evaluate each step's validity given the question and earlier steps.

Return JSON ONLY in this schema (minified, one line):
{"results":[{"step_index":0,"verdict":"valid|invalid|uncertain","p_wrong":0.0}, ...]}

Rules:
- p_wrong is in [0,1] and represents probability the step is wrong.
- Do not add extra keys.
"""
        else:
            instructions = """You are a strict step-by-step verifier for a closed-book yes/no reasoning question.

Evaluate every numbered step using the question and earlier steps. Consider whether the step is relevant, factually plausible based on general knowledge, internally consistent, logically supported, and free of unsupported assumptions. Also consider whether it supports the final yes/no conclusion.

Return JSON ONLY in this schema (minified, one line):
{"results":[{"step_index":0,"verdict":"valid|invalid|uncertain","p_wrong":0.0}, ...]}

Rules:
- p_wrong is in [0,1] and estimates the risk that the step is invalid or unsupported.
- This is a closed-book estimate, not an evidence-grounded factuality judgment.
- Evaluate every supplied step exactly once.
- Do not add extra keys.
"""

        steps_block = "\n".join(f"{index + 1}. {step}" for index, step in enumerate(steps))
        return instructions + "\n\nQUESTION:\n" + question.strip() + "\n\nSTEPS:\n" + steps_block

    def build_repair_prompt(
        self,
        question: str,
        prefix_steps: List[str],
        next_step_number: int,
    ) -> str:
        prefix_block = "\n".join(prefix_steps)

        if self.name == "math":
            return f"""You are fixing a step-by-step solution.

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

        return f"""You are repairing the suffix of a closed-book yes/no reasoning trace.

You are given the QUESTION and a RETAINED LOW-RISK PREFIX that must not be changed. Continue the reasoning beginning at Step {next_step_number} and finish the answer.

Output MUST follow this exact format:
Step {next_step_number}: ...
Step {next_step_number + 1}: ...
...
Final answer: yes

Rules:
- Do not repeat or modify the retained prefix.
- Do not introduce unsupported factual assumptions.
- Do not claim to browse, retrieve evidence, or consult external sources.
- Do not invent citations.
- Keep the reasoning concise and logically consistent.
- The final line MUST be exactly "Final answer: yes" or "Final answer: no".

QUESTION:
{question.strip()}

RETAINED LOW-RISK PREFIX (do not change):
{prefix_block}"""

    def build_acceptance_prompt(
        self,
        *,
        question: str,
        original_trace: str,
        original_answer: str,
        repaired_trace: str,
        repaired_answer: str,
    ) -> str:
        if self.name == "math":
            problem_description = "the same math problem"
            comparison_guidance = (
                "Your job is to estimate whether accepting the repaired solution would risk "
                "making the answer worse. The repaired solution does not need to be better than "
                "the original; it only needs to be similarly supported and low-risk."
            )
        else:
            problem_description = "the same closed-book yes/no reasoning question"
            comparison_guidance = (
                "Estimate whether accepting the repaired solution would make the answer less "
                "defensible. Consider answer support, logical consistency, unsupported factual "
                "assumptions, and the risk that the repair made the answer worse. Do not assume "
                "that either trace has access to retrieved evidence."
            )

        return f"""You are a strict repair acceptance judge.

You will compare an ORIGINAL solution and a REPAIRED solution for {problem_description}.

{comparison_guidance}

Return JSON ONLY in this schema:
{{
  "prefer_repaired": true,
  "original_answer_support": 0.0,
  "repaired_answer_support": 0.0,
  "regression_risk": 0.0,
  "confidence": 0.0,
  "reason": "short explanation"
}}

Definitions:
- original_answer_support: probability from 0 to 1 that the original answer is supported by the reasoning.
- repaired_answer_support: probability from 0 to 1 that the repaired answer is supported by the reasoning.
- regression_risk: probability from 0 to 1 that the repair made the answer worse.
- prefer_repaired: true if the repaired solution is clearly better than the original. If both are similarly correct, prefer_repaired may be false.

QUESTION:
{question.strip()}

ORIGINAL TRACE:
{original_trace.strip()}

ORIGINAL ANSWER:
{original_answer}

REPAIRED TRACE:
{repaired_trace.strip()}

REPAIRED ANSWER:
{repaired_answer}
"""


_PROFILES: Dict[str, TaskProfile] = {
    "math": TaskProfile(name="math", answer_normalizer=normalize_math_answer),
    "strategyqa": TaskProfile(
        name="strategyqa",
        answer_normalizer=normalize_strategyqa_answer,
    ),
}


def get_task_profile(task: Optional[str]) -> TaskProfile:
    """Resolve a canonical profile; missing legacy task values remain math-compatible."""
    profile_name = "math" if task is None else str(task).strip().lower()
    try:
        return _PROFILES[profile_name]
    except KeyError as exc:
        supported = ", ".join(sorted(_PROFILES))
        raise ValueError(
            f"Unknown task profile {task!r}. Supported profiles: {supported}"
        ) from exc


def normalize_answer(task: Optional[str], text: Optional[str]) -> Optional[str]:
    return get_task_profile(task).normalize_answer(text)


def answers_match(
    task: Optional[str],
    prediction: Optional[str],
    gold: Optional[str],
) -> bool:
    return get_task_profile(task).answers_match(prediction, gold)
