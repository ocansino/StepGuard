import json
from pathlib import Path
import unittest

from src.config import load_config


PAIRS = (
    (
        Path(
            "configs/strategyqa_openai_gpt_5_4_mini_"
            "comparison_20.yaml"
        ),
        Path(
            "configs/strategyqa_mistral_small_2603_"
            "exploratory_20.yaml"
        ),
    ),
    (
        Path(
            "configs/gsm8k_openai_gpt_5_4_mini_"
            "comparison_20.yaml"
        ),
        Path(
            "configs/gsm8k_mistral_small_2603_"
            "exploratory_20.yaml"
        ),
    ),
)

STRATEGYQA_OPENAI_POLICY_PATHS = (
    Path(
        "configs/strategyqa_openai_gpt_5_4_mini_"
        "comparison_20_strict_judge.yaml"
    ),
    Path(
        "configs/strategyqa_openai_gpt_5_4_mini_"
        "comparison_20_updated_judge.yaml"
    ),
    Path(
        "configs/strategyqa_openai_gpt_5_4_mini_"
        "comparison_20_oracle.yaml"
    ),
)

GSM8K_OPENAI_POLICY_PATHS = (
    Path(
        "configs/gsm8k_openai_gpt_5_4_mini_"
        "comparison_20_strict_judge.yaml"
    ),
    Path(
        "configs/gsm8k_openai_gpt_5_4_mini_"
        "comparison_20_updated_judge.yaml"
    ),
    Path(
        "configs/gsm8k_openai_gpt_5_4_mini_"
        "comparison_20_oracle.yaml"
    ),
)


class CrossModelComparisonConfigTests(unittest.TestCase):
    def test_openai_runs_use_pinned_snapshot_and_distinct_run_names(self):
        run_names = set()

        for openai_path, mistral_path in PAIRS:
            config = load_config(openai_path)
            mistral = load_config(mistral_path)

            self.assertEqual(config.raw["model"]["provider"], "openai")
            self.assertEqual(
                config.raw["model"]["name"],
                "gpt-5.4-mini-2026-03-17",
            )
            self.assertNotEqual(config.run_name, mistral.run_name)
            self.assertNotIn(config.run_name, run_names)
            run_names.add(config.run_name)

    def test_comparison_reuses_exact_mistral_dataset_and_pipeline(self):
        for openai_path, mistral_path in PAIRS:
            openai = load_config(openai_path)
            mistral = load_config(mistral_path)

            self.assertEqual(openai.dataset_path, mistral.dataset_path)
            self.assertEqual(
                Path(openai.dataset_path).read_bytes(),
                Path(mistral.dataset_path).read_bytes(),
            )
            self.assertEqual(
                openai.raw["task_profile"],
                mistral.raw["task_profile"],
            )
            self.assertEqual(
                openai.raw["model"]["temperature"],
                mistral.raw["model"]["temperature"],
            )
            self.assertEqual(
                openai.raw["model"]["max_output_tokens"],
                mistral.raw["model"]["max_output_tokens"],
            )
            self.assertEqual(openai.raw["scoring"], mistral.raw["scoring"])
            self.assertEqual(
                openai.raw["repair_acceptance"],
                mistral.raw["repair_acceptance"],
            )

            for key in (
                "max_concurrency",
                "generation_chunk_size",
                "candidate_chunk_size",
                "nli_batch_size",
                "metrics_flush_interval_events",
                "request_timeout_seconds",
                "max_retries",
                "retry_base_delay_seconds",
                "retry_jitter_fraction",
            ):
                self.assertEqual(
                    openai.raw["execution"][key],
                    mistral.raw["execution"][key],
                )

    def test_comparison_datasets_have_twenty_unique_records(self):
        for openai_path, _ in PAIRS:
            config = load_config(openai_path)
            rows = Path(config.dataset_path).read_text(
                encoding="utf-8"
            ).splitlines()
            ids = [json.loads(row)["id"] for row in rows]

            self.assertEqual(len(ids), 20)
            self.assertEqual(len(set(ids)), 20)

    def test_strategyqa_policy_replays_preserve_frozen_pool_contract(self):
        base = load_config(
            "configs/strategyqa_openai_gpt_5_4_mini_comparison_20.yaml"
        )
        seen_modes = set()
        seen_run_names = {base.run_name}

        for path in STRATEGYQA_OPENAI_POLICY_PATHS:
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

        self.assertEqual(
            seen_modes,
            {"strict_judge_guard", "judge_guard", "oracle_guard"},
        )

    def test_gsm8k_policy_replays_preserve_frozen_pool_contract(self):
        base = load_config(
            "configs/gsm8k_openai_gpt_5_4_mini_comparison_20.yaml"
        )
        seen_modes = set()
        seen_run_names = {base.run_name}

        for path in GSM8K_OPENAI_POLICY_PATHS:
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

        self.assertEqual(
            seen_modes,
            {"strict_judge_guard", "judge_guard", "oracle_guard"},
        )


if __name__ == "__main__":
    unittest.main()
