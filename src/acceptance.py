from __future__ import annotations

from typing import Any, Dict, Optional

from .task_profiles import answers_match


JUDGE_MODES = {"strict_judge_guard", "judge_guard"}
SUPPORTED_ACCEPTANCE_MODES = {"risk_only", "oracle_guard", *JUDGE_MODES}


def evaluate_repair_acceptance(
    *,
    mode: str,
    task: Optional[str],
    gold_answer: Optional[str],
    current_answer: Optional[str],
    candidate_answer: Optional[str],
    old_avg_risk: float,
    new_avg_risk: float,
    judge_result: Optional[Dict[str, Any]] = None,
    judge_error: Optional[str] = None,
    support_tolerance: float = 0.05,
    max_regression_risk: float = 0.35,
) -> Dict[str, Any]:
    """Apply an acceptance policy using frozen scores and an optional frozen judge result."""
    if mode not in SUPPORTED_ACCEPTANCE_MODES:
        supported = ", ".join(sorted(SUPPORTED_ACCEPTANCE_MODES))
        raise ValueError(f"Unknown repair acceptance mode: {mode}. Supported: {supported}")

    improvement = old_avg_risk - new_avg_risk
    risk_improved = improvement >= 0.0
    decision: Dict[str, Any] = {
        "accepted": risk_improved,
        "reason": "risk_improved" if risk_improved else "risk_worsened",
        "acceptance_mode": mode,
        "risk_improvement": improvement,
    }

    if mode == "risk_only":
        return decision

    if mode == "oracle_guard":
        if not risk_improved:
            return decision

        original_correct = answers_match(task, current_answer, gold_answer)
        candidate_correct = answers_match(task, candidate_answer, gold_answer)
        creates_regression = original_correct and not candidate_correct
        decision.update(
            {
                "oracle_original_correct": original_correct,
                "oracle_candidate_correct": candidate_correct,
                "oracle_creates_regression": creates_regression,
            }
        )
        if creates_regression:
            decision["accepted"] = False
            decision["reason"] = "oracle_regression_guard"
        return decision

    if not risk_improved:
        return decision

    if judge_error is not None:
        decision.update(
            {
                "accepted": False,
                "reason": f"{mode}_error",
                "judge_error": judge_error,
            }
        )
        return decision

    if judge_result is None:
        decision.update(
            {
                "accepted": False,
                "reason": f"{mode}_missing_judge",
            }
        )
        return decision

    prefer_repaired = bool(judge_result.get("prefer_repaired", False))
    original_support = float(judge_result.get("original_answer_support", 0.0))
    repaired_support = float(judge_result.get("repaired_answer_support", 0.0))
    regression_risk = float(judge_result.get("regression_risk", 1.0))

    support_ok = repaired_support >= original_support - support_tolerance
    regression_ok = regression_risk <= max_regression_risk
    preference_ok = prefer_repaired if mode == "strict_judge_guard" else True
    accepted = risk_improved and support_ok and regression_ok and preference_ok

    if accepted:
        reason = f"{mode}_passed"
    elif not support_ok:
        reason = f"{mode}_support_rejected"
    elif not regression_ok:
        reason = f"{mode}_regression_rejected"
    elif not preference_ok:
        reason = "strict_judge_guard_preference_rejected"
    else:
        reason = f"{mode}_rejected"

    decision.update(
        {
            "accepted": accepted,
            "reason": reason,
            "judge": judge_result,
            "judge_prefer_repaired": prefer_repaired,
            "judge_support_ok": support_ok,
            "judge_regression_ok": regression_ok,
            "judge_preference_required": mode == "strict_judge_guard",
            "judge_preference_ok": preference_ok,
        }
    )
    return decision
