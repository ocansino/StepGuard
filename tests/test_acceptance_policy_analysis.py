import json
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.analyze_acceptance_policies import build_report


def score(risk):
    return {
        "steps": ["Step 1: synthetic"],
        "scores": {"verifier": [risk], "contradiction": [0.0]},
        "risks": [risk],
        "earliest_bad_step": 0,
        "risk_summary": {
            "avg_risk": risk,
            "max_risk": risk,
            "num_risky_steps": 1,
        },
    }


def record(record_id, original, gold, candidate=None, judge=None, old=0.4, new=0.1):
    original_trace = f"Step 1: original {record_id}\nFinal answer: {original}"
    proposals = []
    stop_reason = "no_bad_step"
    if candidate is not None:
        proposals = [
            {
                "iter": 1,
                "generation_mode": "suffix_repair",
                "k": 0,
                "next_step_number": 1,
                "parent_trace": original_trace,
                "parent_answer": original,
                "parent_score": score(old),
                "candidate_trace": f"Step 1: candidate {record_id}\nFinal answer: {candidate}",
                "candidate_answer": candidate,
                "candidate_score": score(new),
                "old_avg_risk": old,
                "new_avg_risk": new,
                "improvement": old - new,
                "judge": judge,
                "judge_error": None,
            }
        ]
        stop_reason = "max_iters_reached"
    return {
        "id": record_id,
        "question": "synthetic",
        "model_trace": original_trace,
        "model_answer": original,
        "gold_answer": gold,
        "task": "strategyqa",
        "candidate_pool": {
            "candidate_generation_mode": "suffix_repair",
            "risk_threshold": 0.2,
            "scoring": {
                "risk_threshold": 0.2,
                "improvement_threshold": 0.02,
                "max_iters": 1,
                "risk_formula": "weighted",
                "verifier_weight": 0.75,
                "contradiction_weight": 0.25,
            },
            "improvement_threshold": 0.02,
            "max_iters": 1,
            "original_score": score(old),
            "proposals": proposals,
            "generation_stop_reason": stop_reason,
        },
    }


PASS_PREFERRED = {
    "prefer_repaired": True,
    "original_answer_support": 0.5,
    "repaired_answer_support": 0.7,
    "regression_risk": 0.1,
}
PASS_NOT_PREFERRED = {
    "prefer_repaired": False,
    "original_answer_support": 0.5,
    "repaired_answer_support": 0.5,
    "regression_risk": 0.1,
}
BLOCK_REGRESSION = {
    "prefer_repaired": False,
    "original_answer_support": 0.8,
    "repaired_answer_support": 0.2,
    "regression_risk": 0.9,
}


class AcceptancePolicyAnalysisTests(unittest.TestCase):
    def test_policy_replay_and_paired_bootstrap_are_deterministic(self):
        records = [
            record("regression", "yes", "yes", "no", BLOCK_REGRESSION),
            record("correction", "no", "yes", "yes", PASS_PREFERRED),
            record("safe", "yes", "yes", "yes", PASS_NOT_PREFERRED),
            record("worsened", "no", "yes", "yes", PASS_PREFERRED, old=0.1, new=0.2),
            record("none", "no", "yes"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool_path = root / "pool.jsonl"
            pool_path.write_text(
                "".join(json.dumps(value) + "\n" for value in records),
                encoding="utf-8",
            )
            protocol = {
                "analysis_name": "test",
                "output_dir": str(root / "output"),
                "policies": [
                    "risk_only",
                    "strict_judge_guard",
                    "judge_guard",
                    "oracle_guard",
                ],
                "reference_policy": "judge_guard",
                "acceptance": {
                    "support_tolerance": 0.05,
                    "max_regression_risk": 0.35,
                    "improvement_threshold": 0.02,
                },
                "bootstrap": {
                    "method": "stratified_nonparametric_percentile_bootstrap",
                    "strata": "dataset_and_original_answer_correctness",
                    "paired_policy_comparisons": True,
                    "confidence_level": 0.95,
                    "resamples": 50,
                    "seed": 7,
                },
                "utility": {
                    "definition": "wrong_to_correct_minus_lambda_times_correct_to_wrong",
                    "regression_penalties": [1, 2, 4],
                },
                "datasets": {"synthetic": {"candidate_pools": [str(pool_path)]}},
            }
            first = build_report(protocol, project_root=root)
            second = build_report(protocol, project_root=root)

        metrics = first["datasets"]["synthetic"]["policies"]
        self.assertEqual(metrics["risk_only"]["wrong_to_correct"], 1)
        self.assertEqual(metrics["risk_only"]["correct_to_wrong"], 1)
        self.assertEqual(metrics["oracle_guard"]["correct_to_wrong"], 0)
        self.assertEqual(metrics["judge_guard"]["wrong_to_correct"], 1)
        self.assertEqual(metrics["judge_guard"]["correct_to_wrong"], 0)
        self.assertEqual(metrics["strict_judge_guard"]["accepted_records"], 1)
        self.assertEqual(metrics["judge_guard"]["accepted_records"], 2)
        self.assertEqual(first["datasets"], second["datasets"])

    def test_rejects_non_one_shot_pool(self):
        value = record("bad", "yes", "yes")
        value["candidate_pool"]["max_iters"] = 2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool_path = root / "pool.jsonl"
            pool_path.write_text(json.dumps(value) + "\n", encoding="utf-8")
            protocol = {
                "analysis_name": "test",
                "output_dir": str(root / "output"),
                "policies": ["judge_guard"],
                "reference_policy": "judge_guard",
                "acceptance": {
                    "support_tolerance": 0.05,
                    "max_regression_risk": 0.35,
                    "improvement_threshold": 0.02,
                },
                "bootstrap": {
                    "method": "stratified_nonparametric_percentile_bootstrap",
                    "strata": "dataset_and_original_answer_correctness",
                    "paired_policy_comparisons": True,
                    "confidence_level": 0.95,
                    "resamples": 5,
                    "seed": 7,
                },
                "utility": {
                    "definition": "wrong_to_correct_minus_lambda_times_correct_to_wrong",
                    "regression_penalties": [1],
                },
                "datasets": {"synthetic": {"candidate_pools": [str(pool_path)]}},
            }
            with self.assertRaisesRegex(ValueError, "not a one-shot pool"):
                build_report(protocol, project_root=root)


if __name__ == "__main__":
    unittest.main()
