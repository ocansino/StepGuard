from __future__ import annotations

import re
from typing import List


_STEP_HEADER_RE = re.compile(r"^step\s+\d+\s*:", flags=re.IGNORECASE)
_FINAL_ANSWER_RE = re.compile(r"^final\s+answer\s*:", flags=re.IGNORECASE)


def split_steps(trace: str) -> List[str]:
    """Parse numbered reasoning blocks while excluding the final-answer line."""
    steps: List[str] = []
    current: List[str] = []

    for raw_line in str(trace or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if _FINAL_ANSWER_RE.match(line):
            if current:
                steps.append("\n".join(current))
                current = []
            break

        if _STEP_HEADER_RE.match(line):
            if current:
                steps.append("\n".join(current))
            current = [line]
            continue

        if current:
            current.append(line)

    if current:
        steps.append("\n".join(current))

    return steps


def validate_score_alignment(
    steps: List[str],
    verifier: List[float],
    contradiction: List[float],
) -> None:
    """Fail clearly if a scorer produces a different number of values than steps."""
    expected = len(steps)
    if len(verifier) != expected or len(contradiction) != expected:
        raise ValueError(
            "Step-score alignment failure: "
            f"steps={expected}, verifier={len(verifier)}, "
            f"contradiction={len(contradiction)}"
        )
