from __future__ import annotations

from typing import Any, Dict

from .acceptance import evaluate_repair_acceptance


def _risk_summary(score: Dict[str, Any]) -> Dict[str, Any]:
    return dict(score.get("risk_summary", {}))


def apply_candidate_pool_record(
    record: Dict[str, Any],
    *,
    mode: str,
    support_tolerance: float = 0.05,
    max_regression_risk: float = 0.35,
    improvement_threshold: float = 0.02,
) -> Dict[str, Any]:
    """Apply one acceptance policy to a frozen candidate-pool record without model calls."""
    pool = record.get("candidate_pool")
    if not isinstance(pool, dict):
        raise ValueError(f"Record {record.get('id')!r} has no candidate_pool object")

    original_score = pool.get("original_score")
    if not isinstance(original_score, dict):
        raise ValueError(f"Record {record.get('id')!r} has no frozen original score")

    proposals = pool.get("proposals", [])
    if not isinstance(proposals, list):
        raise ValueError(f"Record {record.get('id')!r} has invalid candidate proposals")

    current_trace = record["model_trace"]
    current_answer = record.get("model_answer")
    final_score = original_score
    iteration_logs = [
        {
            "iter": 0,
            "kind": "original",
            "answer": current_answer,
            "trace": current_trace,
            "earliest_bad_step": original_score.get("earliest_bad_step"),
            **_risk_summary(original_score),
        }
    ]

    stop_reason = pool.get("generation_stop_reason", "candidate_pool_exhausted")

    for proposal in proposals:
        parent_trace = proposal.get("parent_trace")
        parent_answer = proposal.get("parent_answer")
        if parent_trace != current_trace or parent_answer != current_answer:
            stop_reason = "candidate_parent_mismatch"
            break

        candidate_score = proposal.get("candidate_score")
        if not isinstance(candidate_score, dict):
            raise ValueError(
                f"Record {record.get('id')!r} proposal {proposal.get('iter')} "
                "has no candidate_score"
            )

        decision = evaluate_repair_acceptance(
            mode=mode,
            task=record.get("task"),
            gold_answer=record.get("gold_answer"),
            current_answer=current_answer,
            candidate_answer=proposal.get("candidate_answer"),
            old_avg_risk=float(proposal["old_avg_risk"]),
            new_avg_risk=float(proposal["new_avg_risk"]),
            judge_result=proposal.get("judge"),
            judge_error=proposal.get("judge_error"),
            support_tolerance=support_tolerance,
            max_regression_risk=max_regression_risk,
        )
        accepted = bool(decision["accepted"])
        improvement = float(proposal["improvement"])

        iteration_logs.append(
            {
                "iter": proposal.get("iter"),
                "kind": "repair",
                "k": proposal.get("k"),
                "next_step_number": proposal.get("next_step_number"),
                "parent_trace": parent_trace,
                "parent_answer": parent_answer,
                "candidate_trace": proposal.get("candidate_trace"),
                "candidate_answer": proposal.get("candidate_answer"),
                "candidate_score": candidate_score,
                "old_avg_risk": float(proposal["old_avg_risk"]),
                "new_avg_risk": float(proposal["new_avg_risk"]),
                "improvement": improvement,
                "accepted": accepted,
                "acceptance": decision,
                "continue": accepted and improvement >= improvement_threshold,
                "earliest_bad_step": candidate_score.get("earliest_bad_step"),
                **_risk_summary(candidate_score),
            }
        )

        if not accepted:
            stop_reason = decision["reason"]
            break

        current_trace = proposal["candidate_trace"]
        current_answer = proposal.get("candidate_answer")
        final_score = candidate_score

        if improvement < improvement_threshold:
            stop_reason = "improvement_below_threshold"
            break
    else:
        if proposals:
            stop_reason = pool.get("generation_stop_reason", "candidate_pool_exhausted")

    result = dict(record)
    result.update(
        {
            "final_trace": current_trace,
            "final_answer": current_answer,
            "final_steps": final_score.get("steps", []),
            "final_scores": final_score.get("scores", {}),
            "final_risks": final_score.get("risks", []),
            "final_earliest_bad_step": final_score.get("earliest_bad_step"),
            "logs": {
                **record.get("logs", {}),
                "iterative_repair": {
                    "source": "frozen_candidate_pool",
                    "max_iters": pool.get("max_iters"),
                    "improvement_threshold": improvement_threshold,
                    "risk_threshold": pool.get("risk_threshold"),
                    "acceptance_mode": mode,
                    "support_tolerance": support_tolerance,
                    "max_regression_risk": max_regression_risk,
                    "stop_reason": stop_reason,
                    "iterations": iteration_logs,
                },
            },
        }
    )
    return result
