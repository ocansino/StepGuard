import unittest
import hashlib
import json
import random
from pathlib import Path

from src.cli import create_mistral_rate_limiter
from src.config import load_config


TEMPLATE_PATHS = (
    Path("configs/gsm8k_mistral_small_2603_smoke_5.template.yaml"),
    Path("configs/strategyqa_mistral_small_2603_smoke_5.template.yaml"),
)

RUNNABLE_PATHS = (
    Path("configs/gsm8k_mistral_small_2603_smoke_5.yaml"),
    Path("configs/strategyqa_mistral_small_2603_smoke_5.yaml"),
)

EXPLORATORY_PATHS = (
    Path("configs/gsm8k_mistral_small_2603_exploratory_20.yaml"),
    Path("configs/strategyqa_mistral_small_2603_exploratory_20.yaml"),
)

STRATEGYQA_POLICY_PATHS = (
    Path(
        "configs/strategyqa_mistral_small_2603_"
        "exploratory_20_strict_judge.yaml"
    ),
    Path(
        "configs/strategyqa_mistral_small_2603_"
        "exploratory_20_updated_judge.yaml"
    ),
    Path(
        "configs/strategyqa_mistral_small_2603_"
        "exploratory_20_oracle.yaml"
    ),
)

GSM8K_POLICY_PATHS = (
    Path(
        "configs/gsm8k_mistral_small_2603_"
        "exploratory_20_strict_judge.yaml"
    ),
    Path(
        "configs/gsm8k_mistral_small_2603_"
        "exploratory_20_updated_judge.yaml"
    ),
    Path(
        "configs/gsm8k_mistral_small_2603_"
        "exploratory_20_oracle.yaml"
    ),
)

ALL_MISTRAL_CONFIG_PATHS = (
    TEMPLATE_PATHS
    + RUNNABLE_PATHS
    + EXPLORATORY_PATHS
)


class MistralSmokeConfigTests(unittest.TestCase):
    def test_smoke_templates_pin_model_and_force_sequential_execution(self):
        configs = [load_config(path) for path in ALL_MISTRAL_CONFIG_PATHS]

        for config in configs:
            self.assertEqual(config.raw["model"]["provider"], "mistral")
            self.assertEqual(
                config.raw["model"]["name"],
                "mistral-small-2603",
            )
            self.assertEqual(config.raw["execution"]["max_concurrency"], 1)
            self.assertEqual(
                config.raw["execution"]["metrics_flush_interval_events"],
                1,
            )

    def test_smoke_templates_preserve_calibrated_stepguard_settings(self):
        configs = [load_config(path) for path in ALL_MISTRAL_CONFIG_PATHS]
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

        for config in configs:
            self.assertEqual(config.raw["scoring"], expected_scoring)
            self.assertEqual(
                config.raw["repair_acceptance"],
                expected_acceptance,
            )

    def test_verified_account_specific_rate_limits_are_pinned(self):
        configs = [load_config(path) for path in ALL_MISTRAL_CONFIG_PATHS]

        for config in configs:
            execution = config.raw["execution"]
            self.assertEqual(
                execution["mistral_requests_per_second"],
                0.83,
            )
            self.assertEqual(
                execution["mistral_tokens_per_minute"],
                50000,
            )
            self.assertIsNone(
                execution["mistral_monthly_token_allowance"]
            )

    def test_rate_limiter_allows_unpublished_monthly_text_limit(self):
        limiter = create_mistral_rate_limiter(
            {
                "mistral_requests_per_second": 0.83,
                "mistral_tokens_per_minute": 50000,
                "mistral_monthly_token_allowance": None,
                "rate_limit_headroom_fraction": 0.8,
            },
            metrics=None,
        )

        self.assertAlmostEqual(limiter.requests_per_minute, 49.8)
        self.assertAlmostEqual(
            limiter.effective_requests_per_minute,
            39.84,
        )
        self.assertEqual(
            limiter.wait_name,
            "mistral.request_rate_limit",
        )

    def test_smoke_templates_use_distinct_existing_five_record_datasets(self):
        configs = [load_config(path) for path in RUNNABLE_PATHS]
        dataset_paths = {Path(config.dataset_path) for config in configs}

        self.assertEqual(
            dataset_paths,
            {
                Path("data/raw/gsm8k_test_5.jsonl"),
                Path("data/prepared/strategyqa_5.jsonl"),
            },
        )

        for dataset_path in dataset_paths:
            self.assertTrue(dataset_path.exists())
            records = dataset_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(records), 5)

    def test_runnable_configs_do_not_share_template_run_names(self):
        template_run_names = {
            load_config(path).run_name for path in TEMPLATE_PATHS
        }
        runnable_run_names = {
            load_config(path).run_name for path in RUNNABLE_PATHS
        }

        self.assertTrue(template_run_names.isdisjoint(runnable_run_names))
        self.assertEqual(len(runnable_run_names), 2)

    def test_runnable_configs_use_conservative_smoke_headroom(self):
        for path in RUNNABLE_PATHS + EXPLORATORY_PATHS:
            config = load_config(path)
            execution = config.raw["execution"]
            limiter = create_mistral_rate_limiter(
                execution,
                metrics=None,
            )

            self.assertEqual(
                execution["rate_limit_headroom_fraction"],
                0.2,
            )
            self.assertAlmostEqual(
                limiter.effective_requests_per_minute,
                9.96,
            )

    def test_exploratory_subsets_are_fresh_deterministic_twenty_record_sets(self):
        smoke_paths = {
            "math": Path("data/raw/gsm8k_test_5.jsonl"),
            "strategyqa": Path("data/prepared/strategyqa_5.jsonl"),
        }

        for config_path in EXPLORATORY_PATHS:
            config = load_config(config_path)
            task = config.raw["task_profile"]
            dataset_path = Path(config.dataset_path)
            manifest_path = dataset_path.with_name(
                f"manifest.{dataset_path.stem}.json"
            )
            records = [
                json.loads(line)
                for line in dataset_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            smoke_ids = {
                json.loads(line)["id"]
                for line in smoke_paths[task].read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            }
            selected_ids = [record["id"] for record in records]
            source_records = [
                json.loads(line)
                for line in Path(manifest["source_path"]).read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            eligible_records = [
                record
                for record in source_records
                if record["id"] not in smoke_ids
            ]
            sampled_ids = {
                record["id"]
                for record in random.Random(2603).sample(
                    eligible_records,
                    20,
                )
            }
            expected_ids = [
                record["id"]
                for record in source_records
                if record["id"] in sampled_ids
            ]

            self.assertEqual(len(records), 20)
            self.assertEqual(len(set(selected_ids)), 20)
            self.assertTrue(set(selected_ids).isdisjoint(smoke_ids))
            self.assertEqual(manifest["selection_seed"], 2603)
            self.assertEqual(manifest["selected_ids"], selected_ids)
            self.assertEqual(selected_ids, expected_ids)
            self.assertEqual(manifest["num_records"], 20)
            self.assertEqual(
                manifest["output_sha256"],
                hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            )

    def test_exploratory_runs_are_isolated_from_smoke_runs(self):
        smoke_run_names = {
            load_config(path).run_name for path in RUNNABLE_PATHS
        }
        exploratory_run_names = {
            load_config(path).run_name for path in EXPLORATORY_PATHS
        }

        self.assertTrue(smoke_run_names.isdisjoint(exploratory_run_names))
        self.assertEqual(len(exploratory_run_names), 2)

    def test_strategyqa_policy_configs_reuse_the_frozen_pool_contract(self):
        base = load_config(
            "configs/strategyqa_mistral_small_2603_exploratory_20.yaml"
        )
        expected_modes = {
            "strict_judge_guard",
            "judge_guard",
            "oracle_guard",
        }
        seen_modes = set()
        seen_run_names = {base.run_name}

        for path in STRATEGYQA_POLICY_PATHS:
            config = load_config(path)
            seen_modes.add(config.raw["repair_acceptance"]["mode"])

            self.assertNotIn(config.run_name, seen_run_names)
            seen_run_names.add(config.run_name)
            self.assertEqual(config.dataset_path, base.dataset_path)
            self.assertEqual(config.raw["model"], base.raw["model"])
            self.assertEqual(config.raw["execution"], base.raw["execution"])
            self.assertEqual(config.raw["scoring"], base.raw["scoring"])
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

        self.assertEqual(seen_modes, expected_modes)

    def test_gsm8k_policy_configs_reuse_the_frozen_pool_contract(self):
        base = load_config(
            "configs/gsm8k_mistral_small_2603_exploratory_20.yaml"
        )
        expected_modes = {
            "strict_judge_guard",
            "judge_guard",
            "oracle_guard",
        }
        seen_modes = set()
        seen_run_names = {base.run_name}

        for path in GSM8K_POLICY_PATHS:
            config = load_config(path)
            seen_modes.add(config.raw["repair_acceptance"]["mode"])

            self.assertNotIn(config.run_name, seen_run_names)
            seen_run_names.add(config.run_name)
            self.assertEqual(config.dataset_path, base.dataset_path)
            self.assertEqual(config.raw["model"], base.raw["model"])
            self.assertEqual(config.raw["execution"], base.raw["execution"])
            self.assertEqual(config.raw["scoring"], base.raw["scoring"])
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

        self.assertEqual(seen_modes, expected_modes)


if __name__ == "__main__":
    unittest.main()
