from __future__ import annotations

from typing import Any, Dict, List

from .parsing import validate_score_alignment


def summarize_risk(risks: List[float], tau: float) -> Dict[str, Any]:
    if not risks:
        return {"avg_risk": 0.0, "max_risk": 0.0, "num_risky_steps": 0}
    return {
        "avg_risk": sum(risks) / len(risks),
        "max_risk": max(risks),
        "num_risky_steps": sum(1 for risk in risks if risk > tau),
    }


def compute_risks(
    verifier: List[float],
    contradiction: List[float],
    scoring_cfg: Dict[str, Any],
) -> List[float]:
    formula = scoring_cfg.get("risk_formula", "verifier_heavy")

    if formula == "verifier_only":
        return [min(1.0, value) for value in verifier]
    if formula == "contradiction_only":
        return [min(1.0, value) for value in contradiction]
    if formula == "additive":
        return [min(1.0, v + c) for v, c in zip(verifier, contradiction)]
    if formula == "weighted":
        verifier_weight = float(scoring_cfg.get("verifier_weight", 0.75))
        contradiction_weight = float(scoring_cfg.get("contradiction_weight", 0.25))
        return [
            min(1.0, verifier_weight * v + contradiction_weight * c)
            for v, c in zip(verifier, contradiction)
        ]
    if formula == "verifier_heavy":
        return [
            min(1.0, 0.75 * v + 0.25 * c)
            for v, c in zip(verifier, contradiction)
        ]
    raise ValueError(f"Unknown risk_formula: {formula}")


def rescore_frozen_trace_score(
    frozen_score: Dict[str, Any],
    *,
    scoring_cfg: Dict[str, Any],
    tau: float,
) -> Dict[str, Any]:
    """Recompute risk and localization while preserving frozen step-model outputs."""
    steps = list(frozen_score.get("steps", []))
    scores = dict(frozen_score.get("scores", {}))
    verifier = list(scores.get("verifier", []))
    contradiction = list(scores.get("contradiction", []))
    validate_score_alignment(steps, verifier, contradiction)

    risks = compute_risks(verifier, contradiction, scoring_cfg)
    earliest = next((index for index, risk in enumerate(risks) if risk > tau), None)
    return {
        **frozen_score,
        "steps": steps,
        "scores": scores,
        "risks": risks,
        "earliest_bad_step": earliest,
        "risk_summary": summarize_risk(risks, tau),
    }
