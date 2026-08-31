import json
import unittest
from pathlib import Path

from src.config import load_config


SMOKE_PATHS = (
    Path(
        "configs/strategyqa_gemini_3_5_flash_lite_"
        "smoke_5.template.yaml"
    ),
    Path(
        "configs/gsm8k_gemini_3_5_flash_lite_"
        "smoke_5.template.yaml"
    ),
)

RUNNABLE_SMOKE_PATHS = (
    Path(
        "configs/strategyqa_gemini_3_5_flash_lite_"
        "smoke_5.yaml"
    ),
    Path(
        "configs/gsm8k_gemini_3_5_flash_lite_"
        "smoke_5.yaml"
    ),
)

COMPARISON_PATHS = (
    Path(
        "configs/strategyqa_gemini_3_5_flash_lite_"
        "comparison_20.template.yaml"
    ),
    Path(
        "configs/gsm8k_gemini_3_5_flash_lite_"
        "comparison_20.template.yaml"
    ),
)

RUNNABLE_COMPARISON_PATHS = (
    Path(
        "configs/strategyqa_gemini_3_5_flash_lite_"
        "comparison_20.yaml"
    ),
    Path(
        "configs/gsm8k_gemini_3_5_flash_lite_"
        "comparison_20.yaml"
    ),
)

STRATEGYQA_POLICY_PATHS = (
    Path(
        "configs/strategyqa_gemini_3_5_flash_lite_"
        "comparison_20_strict_judge.yaml"
    ),
    Path(
        "configs/strategyqa_gemini_3_5_flash_lite_"
        "comparison_20_updated_judge.yaml"
    ),
    Path(
        "configs/strategyqa_gemini_3_5_flash_lite_"
        "comparison_20_oracle.yaml"
    ),
)

GSM8K_POLICY_PATHS = (
    Path(
        "configs/gsm8k_gemini_3_5_flash_lite_"
        "comparison_20_strict_judge.yaml"
    ),
    Path(
        "configs/gsm8k_gemini_3_5_flash_lite_"
        "comparison_20_updated_judge.yaml"
    ),
    Path(
        "configs/gsm8k_gemini_3_5_flash_lite_"
        "comparison_20_oracle.yaml"
    ),
)


class GeminiConfigTests(unittest.TestCase):
    def test_templates_use_stable_free_tier_model_and_sequential_execution(self):
        for path in (
            SMOKE_PATHS
            + RUNNABLE_SMOKE_PATHS
            + COMPARISON_PATHS
            + RUNNABLE_COMPARISON_PATHS
        ):
            config = load_config(path)

            self.assertEqual(config.raw["model"]["provider"], "gemini")
            self.assertEqual(
                config.raw["model"]["name"],
                "gemini-3.5-flash-lite",
            )
            self.assertEqual(config.raw["execution"]["max_concurrency"], 1)
            self.assertEqual(
                config.raw["execution"]["metrics_flush_interval_events"],
                1,
            )

    def test_templates_cannot_silently_invent_account_limits(self):
        for path in SMOKE_PATHS + COMPARISON_PATHS:
            execution = load_config(path).raw["execution"]

            self.assertIsNone(execution["gemini_requests_per_minute"])
            self.assertIsNone(execution["gemini_tokens_per_minute"])
            self.assertIsNone(execution["gemini_requests_per_day"])
            self.assertEqual(
                execution["rate_limit_headroom_fraction"],
                0.2,
            )

    def test_runnable_smoke_configs_pin_verified_account_limits(self):
        for path in (
            RUNNABLE_SMOKE_PATHS
            + RUNNABLE_COMPARISON_PATHS
        ):
            execution = load_config(path).raw["execution"]

            self.assertEqual(execution["gemini_requests_per_minute"], 15)
            self.assertEqual(execution["gemini_tokens_per_minute"], 250000)
            self.assertEqual(execution["gemini_requests_per_day"], 500)
            self.assertEqual(
                execution["rate_limit_headroom_fraction"],
                0.2,
            )

    def test_stepguard_scoring_and_acceptance_settings_are_preserved(self):
        expected_scoring = {
            "risk_threshold": 0.20,
            "improvement_threshold": 0.02,
            "max_iters": 2,
            "risk_formula": "weighted",
            "verifier_weight": 0.75,
            "contradiction_weight": 0.25,
        }
        expected_acceptance = {
            "mode": "risk_only",
            "support_tolerance": 0.05,
            "max_regression_risk": 0.35,
        }

        for path in (
            SMOKE_PATHS
            + RUNNABLE_SMOKE_PATHS
            + COMPARISON_PATHS
            + RUNNABLE_COMPARISON_PATHS
        ):
            config = load_config(path)

            self.assertEqual(config.raw["scoring"], expected_scoring)
            self.assertEqual(
                config.raw["repair_acceptance"],
                expected_acceptance,
            )

    def test_smoke_templates_use_existing_five_record_datasets(self):
        expected = {
            "strategyqa": Path("data/prepared/strategyqa_5.jsonl"),
            "math": Path("data/raw/gsm8k_test_5.jsonl"),
        }

        for path in SMOKE_PATHS + RUNNABLE_SMOKE_PATHS:
            config = load_config(path)
            dataset = Path(config.dataset_path)

            self.assertEqual(
                dataset,
                expected[config.raw["task_profile"]],
            )
            self.assertTrue(dataset.exists())
            self.assertEqual(
                len(dataset.read_text(encoding="utf-8").splitlines()),
                5,
            )

    def test_runnable_smoke_run_names_are_distinct_from_templates(self):
        template_names = {
            load_config(path).run_name for path in SMOKE_PATHS
        }
        runnable_names = {
            load_config(path).run_name for path in RUNNABLE_SMOKE_PATHS
        }

        self.assertTrue(template_names.isdisjoint(runnable_names))
        self.assertEqual(len(runnable_names), 2)

    def test_comparison_templates_reuse_exact_paired_twenty_record_subsets(self):
        expected = {
            "strategyqa": Path(
                "data/prepared/strategyqa_mistral_20_seed2603.jsonl"
            ),
            "math": Path(
                "data/prepared/gsm8k_mistral_20_seed2603.jsonl"
            ),
        }

        for path in COMPARISON_PATHS + RUNNABLE_COMPARISON_PATHS:
            config = load_config(path)
            dataset = Path(config.dataset_path)
            records = [
                json.loads(line)
                for line in dataset.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]

            self.assertEqual(
                dataset,
                expected[config.raw["task_profile"]],
            )
            self.assertEqual(len(records), 20)
            self.assertEqual(
                len({record["id"] for record in records}),
                20,
            )

    def test_runnable_comparison_names_are_unique_and_not_templates(self):
        template_names = {
            load_config(path).run_name for path in COMPARISON_PATHS
        }
        runnable_names = {
            load_config(path).run_name
            for path in RUNNABLE_COMPARISON_PATHS
        }

        self.assertTrue(template_names.isdisjoint(runnable_names))
        self.assertEqual(len(runnable_names), 2)

    def test_policy_replays_preserve_each_frozen_pool_contract(self):
        cases = (
            (
                Path(
                    "configs/strategyqa_gemini_3_5_flash_lite_"
                    "comparison_20.yaml"
                ),
                STRATEGYQA_POLICY_PATHS,
            ),
            (
                Path(
                    "configs/gsm8k_gemini_3_5_flash_lite_"
                    "comparison_20.yaml"
                ),
                GSM8K_POLICY_PATHS,
            ),
        )

        for base_path, policy_paths in cases:
            base = load_config(base_path)
            seen_modes = set()
            seen_names = {base.run_name}

            for path in policy_paths:
                config = load_config(path)
                seen_modes.add(
                    config.raw["repair_acceptance"]["mode"]
                )
                self.assertNotIn(config.run_name, seen_names)
                seen_names.add(config.run_name)
                self.assertEqual(config.dataset_path, base.dataset_path)
                self.assertEqual(config.raw["model"], base.raw["model"])
                self.assertEqual(
                    config.raw["execution"],
                    base.raw["execution"],
                )
                self.assertEqual(
                    config.raw["scoring"],
                    base.raw["scoring"],
                )
                self.assertEqual(
                    {
                        key: value
                        for key, value in config.raw[
                            "repair_acceptance"
                        ].items()
                        if key != "mode"
                    },
                    {
                        key: value
                        for key, value in base.raw[
                            "repair_acceptance"
                        ].items()
                        if key != "mode"
                    },
                )

            self.assertEqual(
                seen_modes,
                {
                    "strict_judge_guard",
                    "judge_guard",
                    "oracle_guard",
                },
            )


if __name__ == "__main__":
    unittest.main()
