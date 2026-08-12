import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from src.candidate_pool_checkpoint import (
    save_candidate_pool_checkpoint as real_save_checkpoint,
)
from src.cli import build_candidate_pool
from tests.test_wave_candidate_pool import FakeNLIScorer, FakeWaveClient


class CandidatePoolResumeTests(unittest.TestCase):
    def test_resume_skips_completed_original_scoring_and_matches_clean_run(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = temp_path / "generated.jsonl"
            output_dir = temp_path / "runs"

            records = [
                {
                    "id": f"q{index}",
                    "question": f"Question {index}?",
                    "model_trace": (
                        f"Step 1: Original reasoning for question {index}.\n"
                        "Final answer: no"
                    ),
                    "model_answer": "no",
                    "gold_answer": "yes",
                    "task": "strategyqa",
                    "source": "offline-resume-test",
                }
                for index in range(1, 4)
            ]
            dataset_path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            def write_config(run_name):
                config_path = temp_path / f"{run_name}.yaml"
                config = {
                    "run_name": run_name,
                    "dataset_path": str(dataset_path),
                    "output_dir": str(output_dir),
                    "task_profile": "strategyqa",
                    "model": {
                        "provider": "openai",
                        "name": "fake-model",
                        "temperature": 0.2,
                        "max_output_tokens": 100,
                    },
                    "execution": {
                        "metrics_enabled": True,
                        "max_concurrency": 1,
                        "request_timeout_seconds": 45.0,
                        "max_retries": 1,
                        "retry_base_delay_seconds": 0.5,
                    },
                    "scoring": {
                        "risk_threshold": 0.2,
                        "improvement_threshold": 0.02,
                        "max_iters": 1,
                        "risk_formula": "weighted",
                        "verifier_weight": 0.75,
                        "contradiction_weight": 0.25,
                    },
                    "repair_acceptance": {
                        "mode": "risk_only",
                        "support_tolerance": 0.05,
                        "max_regression_risk": 0.35,
                    },
                }
                config_path.write_text(
                    yaml.safe_dump(config),
                    encoding="utf-8",
                )
                return config_path

            resumed_config = write_config("resume-run")
            clean_config = write_config("clean-run")
            call_log = []

            def make_client(*, model, metrics, **execution_options):
                return FakeWaveClient(
                    model=model,
                    metrics=metrics,
                    call_log=call_log,
                )

            interrupted = False

            def save_then_interrupt(path, **kwargs):
                nonlocal interrupted
                real_save_checkpoint(path, **kwargs)
                if (
                    kwargs["phase"] == "original_scoring_complete"
                    and not interrupted
                ):
                    interrupted = True
                    raise RuntimeError("simulated interruption")

            with (
                patch(
                    "src.cli.OpenAIClientWrapper",
                    side_effect=make_client,
                ),
                patch("src.cli.NLIScorer", FakeNLIScorer),
                patch(
                    "src.cli.save_candidate_pool_checkpoint",
                    side_effect=save_then_interrupt,
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "simulated interruption",
                ):
                    build_candidate_pool(
                        config=str(resumed_config),
                        input_path=str(dataset_path),
                        output_path=None,
                        reuse_original_scores=False,
                        resume=False,
                    )

            resumed_dir = output_dir / "resume-run"
            checkpoint_path = (
                resumed_dir / "candidate_pool.jsonl.checkpoint.json"
            )
            self.assertTrue(checkpoint_path.exists())
            self.assertFalse(
                (resumed_dir / "candidate_pool.jsonl").exists()
            )
            self.assertEqual(call_log.count("step_verification"), 3)

            with (
                patch(
                    "src.cli.OpenAIClientWrapper",
                    side_effect=make_client,
                ),
                patch("src.cli.NLIScorer", FakeNLIScorer),
            ):
                build_candidate_pool(
                    config=str(resumed_config),
                    input_path=str(dataset_path),
                    output_path=None,
                    reuse_original_scores=False,
                    resume=True,
                )

            self.assertEqual(call_log.count("step_verification"), 6)
            self.assertFalse(checkpoint_path.exists())

            resumed_metrics = json.loads(
                (
                    resumed_dir / "metrics.build_candidate_pool.json"
                ).read_text(encoding="utf-8")
            )
            self.assertNotIn(
                "original_verification_wave",
                resumed_metrics["stages"],
            )
            self.assertIn("checkpoint_loading", resumed_metrics["stages"])
            self.assertEqual(
                resumed_metrics["operations"]
                ["openai.step_verification"]["logical_calls"],
                3,
            )

            clean_call_log = []

            def make_clean_client(*, model, metrics, **execution_options):
                return FakeWaveClient(
                    model=model,
                    metrics=metrics,
                    call_log=clean_call_log,
                )

            with (
                patch(
                    "src.cli.OpenAIClientWrapper",
                    side_effect=make_clean_client,
                ),
                patch("src.cli.NLIScorer", FakeNLIScorer),
            ):
                build_candidate_pool(
                    config=str(clean_config),
                    input_path=str(dataset_path),
                    output_path=None,
                    reuse_original_scores=False,
                    resume=False,
                )

            def read_records(path):
                return [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                ]

            resumed_records = read_records(
                resumed_dir / "candidate_pool.jsonl"
            )
            clean_records = read_records(
                output_dir / "clean-run" / "candidate_pool.jsonl"
            )

            self.assertEqual(resumed_records, clean_records)


if __name__ == "__main__":
    unittest.main()
