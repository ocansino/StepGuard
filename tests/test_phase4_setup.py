import unittest
from pathlib import Path

from src.config import load_config
from src.risk import compute_risks, rescore_frozen_trace_score


class FrozenScoreReuseTests(unittest.TestCase):
    def test_risk_formulas_match_existing_definitions(self):
        verifier = [0.1, 0.3]
        contradiction = [0.0, 0.4]
        self.assertEqual(
            compute_risks(verifier, contradiction, {"risk_formula": "additive"}),
            [0.1, 0.7],
        )
        self.assertEqual(
            compute_risks(
                verifier,
                contradiction,
                {
                    "risk_formula": "weighted",
                    "verifier_weight": 0.75,
                    "contradiction_weight": 0.25,
                },
            ),
            [0.07500000000000001, 0.32499999999999996],
        )

    def test_rescore_preserves_frozen_verifier_and_nli_outputs(self):
        frozen = {
            "steps": ["Step 1: A", "Step 2: B"],
            "scores": {
                "verifier": [0.1, 0.3],
                "contradiction": [0.0, 0.4],
                "evidence_support": None,
            },
            "risks": [0.075, 0.325],
            "earliest_bad_step": 1,
            "risk_summary": {"avg_risk": 0.2, "max_risk": 0.325, "num_risky_steps": 1},
        }
        rescored = rescore_frozen_trace_score(
            frozen,
            scoring_cfg={"risk_formula": "additive"},
            tau=0.10,
        )
        self.assertEqual(rescored["scores"], frozen["scores"])
        self.assertEqual(rescored["risks"], [0.1, 0.7])
        self.assertEqual(rescored["earliest_bad_step"], 1)
        self.assertEqual(rescored["risk_summary"]["num_risky_steps"], 1)


class Phase4ConfigTests(unittest.TestCase):
    def test_phase4_configs_share_frozen_dataset_and_model(self):
        paths = sorted(Path("configs").glob("strategyqa_openai_100_*.yaml"))
        self.assertEqual(len(paths), 7)
        configs = [load_config(path) for path in paths]
        self.assertTrue(all(config.dataset_path == "data/prepared/strategyqa_100.jsonl" for config in configs))
        self.assertTrue(all(config.raw["task_profile"] == "strategyqa" for config in configs))
        self.assertEqual(len({json_model(config.raw["model"]) for config in configs}), 1)

    def test_calibrated_policies_have_identical_scoring(self):
        names = (
            "strategyqa_openai_100_calibrated_pool.yaml",
            "strategyqa_openai_100_calibrated_risk_only.yaml",
            "strategyqa_openai_100_oracle.yaml",
            "strategyqa_openai_100_strict_judge.yaml",
            "strategyqa_openai_100_updated_judge.yaml",
        )
        configs = [load_config(Path("configs") / name) for name in names]
        scoring = configs[0].raw["scoring"]
        self.assertTrue(all(config.raw["scoring"] == scoring for config in configs))
        modes = {
            config.raw["repair_acceptance"]["mode"]
            for config in configs[1:]
        }
        self.assertEqual(
            modes,
            {"risk_only", "oracle_guard", "strict_judge_guard", "judge_guard"},
        )

    def test_aggressive_pool_and_policy_configs_match(self):
        pool = load_config("configs/strategyqa_openai_100_aggressive_pool.yaml")
        policy = load_config("configs/strategyqa_openai_100_aggressive_risk_only.yaml")
        self.assertEqual(pool.raw["scoring"], policy.raw["scoring"])
        self.assertEqual(pool.raw["scoring"]["risk_formula"], "additive")
        self.assertEqual(pool.raw["scoring"]["risk_threshold"], 0.10)


def json_model(model):
    return tuple(sorted(model.items()))


if __name__ == "__main__":
    unittest.main()
