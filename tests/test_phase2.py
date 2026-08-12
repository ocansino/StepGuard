import unittest

from src.acceptance import evaluate_repair_acceptance
from src.candidate_pool import apply_candidate_pool_record


def make_score(avg_risk: float, earliest_bad_step=0):
    return {
        "steps": ["Step 1: Example reasoning."],
        "scores": {"verifier": [avg_risk], "contradiction": [0.0]},
        "risks": [avg_risk],
        "earliest_bad_step": earliest_bad_step,
        "risk_summary": {
            "avg_risk": avg_risk,
            "max_risk": avg_risk,
            "num_risky_steps": int(earliest_bad_step is not None),
        },
    }


def make_pool_record(judge):
    original_trace = "Step 1: Original reasoning.\nFinal answer: yes"
    candidate_trace = "Step 1: Repaired reasoning.\nFinal answer: no"
    return {
        "id": "example",
        "task": "strategyqa",
        "question": "Example question?",
        "gold_answer": "yes",
        "model_trace": original_trace,
        "model_answer": "yes",
        "candidate_pool": {
            "construction_policy": "risk_only_chain",
            "risk_threshold": 0.2,
            "improvement_threshold": 0.02,
            "max_iters": 1,
            "original_score": make_score(0.6),
            "generation_stop_reason": "max_iters",
            "generation_errors": [],
            "proposals": [
                {
                    "iter": 1,
                    "k": 0,
                    "next_step_number": 1,
                    "parent_trace": original_trace,
                    "parent_answer": "yes",
                    "parent_score": make_score(0.6),
                    "candidate_trace": candidate_trace,
                    "candidate_answer": "no",
                    "candidate_score": make_score(0.1, earliest_bad_step=None),
                    "old_avg_risk": 0.6,
                    "new_avg_risk": 0.1,
                    "improvement": 0.5,
                    "judge": judge,
                    "judge_error": None,
                }
            ],
        },
    }


class AcceptancePolicyTests(unittest.TestCase):
    def test_strict_requires_preference_but_updated_does_not(self):
        judge = {
            "prefer_repaired": False,
            "original_answer_support": 0.8,
            "repaired_answer_support": 0.79,
            "regression_risk": 0.1,
        }
        common = {
            "task": "strategyqa",
            "gold_answer": "yes",
            "current_answer": "yes",
            "candidate_answer": "no",
            "old_avg_risk": 0.6,
            "new_avg_risk": 0.1,
            "judge_result": judge,
        }
        strict = evaluate_repair_acceptance(mode="strict_judge_guard", **common)
        updated = evaluate_repair_acceptance(mode="judge_guard", **common)

        self.assertFalse(strict["accepted"])
        self.assertEqual(strict["reason"], "strict_judge_guard_preference_rejected")
        self.assertTrue(updated["accepted"])
        self.assertTrue(strict["judge_preference_required"])
        self.assertFalse(updated["judge_preference_required"])

    def test_support_and_regression_checks_remain_required(self):
        low_support = {
            "prefer_repaired": True,
            "original_answer_support": 0.9,
            "repaired_answer_support": 0.5,
            "regression_risk": 0.1,
        }
        result = evaluate_repair_acceptance(
            mode="judge_guard",
            task="strategyqa",
            gold_answer="yes",
            current_answer="yes",
            candidate_answer="yes",
            old_avg_risk=0.6,
            new_avg_risk=0.1,
            judge_result=low_support,
        )
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "judge_guard_support_rejected")

    def test_oracle_blocks_strategyqa_correct_to_wrong_regression(self):
        result = evaluate_repair_acceptance(
            mode="oracle_guard",
            task="strategyqa",
            gold_answer="yes",
            current_answer="YES.",
            candidate_answer="false",
            old_avg_risk=0.6,
            new_avg_risk=0.1,
        )
        self.assertFalse(result["accepted"])
        self.assertTrue(result["oracle_creates_regression"])

    def test_risk_only_uses_no_judge_result(self):
        result = evaluate_repair_acceptance(
            mode="risk_only",
            task="strategyqa",
            gold_answer="yes",
            current_answer="yes",
            candidate_answer="no",
            old_avg_risk=0.6,
            new_avg_risk=0.1,
        )
        self.assertTrue(result["accepted"])
        self.assertNotIn("judge", result)


class CandidatePoolTests(unittest.TestCase):
    def test_same_candidate_is_reused_by_strict_and_updated_policies(self):
        judge = {
            "prefer_repaired": False,
            "original_answer_support": 0.8,
            "repaired_answer_support": 0.79,
            "regression_risk": 0.1,
        }
        record = make_pool_record(judge)
        strict = apply_candidate_pool_record(record, mode="strict_judge_guard")
        updated = apply_candidate_pool_record(record, mode="judge_guard")

        strict_log = strict["logs"]["iterative_repair"]["iterations"][1]
        updated_log = updated["logs"]["iterative_repair"]["iterations"][1]
        self.assertEqual(strict_log["candidate_trace"], updated_log["candidate_trace"])
        self.assertFalse(strict_log["accepted"])
        self.assertTrue(updated_log["accepted"])
        self.assertEqual(strict["final_answer"], "yes")
        self.assertEqual(updated["final_answer"], "no")

    def test_rejected_candidate_artifact_is_complete(self):
        judge = {
            "prefer_repaired": False,
            "original_answer_support": 0.8,
            "repaired_answer_support": 0.79,
            "regression_risk": 0.1,
        }
        result = apply_candidate_pool_record(
            make_pool_record(judge),
            mode="strict_judge_guard",
        )
        log = result["logs"]["iterative_repair"]["iterations"][1]
        self.assertIn("candidate_trace", log)
        self.assertIn("candidate_answer", log)
        self.assertIn("candidate_score", log)
        self.assertIn("judge", log["acceptance"])
        self.assertFalse(log["accepted"])

    def test_oracle_and_risk_only_apply_without_fresh_judge_calls(self):
        record = make_pool_record(judge=None)
        oracle = apply_candidate_pool_record(record, mode="oracle_guard")
        risk_only = apply_candidate_pool_record(record, mode="risk_only")
        self.assertEqual(oracle["final_answer"], "yes")
        self.assertEqual(risk_only["final_answer"], "no")


if __name__ == "__main__":
    unittest.main()
