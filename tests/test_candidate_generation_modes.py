import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from src.cli import build_candidate_pool
from tests.test_wave_candidate_pool import FakeNLIScorer, FakeWaveClient


class FullRegenerationClient(FakeWaveClient):
    def generate_trace(
        self,
        question,
        *,
        task="math",
        temperature=0.2,
        max_output_tokens=800,
    ):
        self._record_call("trace_generation")
        self.metrics.record_operation(
            provider="openai",
            operation="trace_generation",
            elapsed_seconds=0.01,
            success=True,
            usage={
                "input_tokens": 8,
                "output_tokens": 7,
                "total_tokens": 15,
            },
        )
        return (
            "Step 1: Repaired reasoning from a completely new trace.\n"
            "Final answer: yes",
            "yes",
        )

    def repair_suffix(self, *args, **kwargs):
        raise AssertionError(
            "full_regeneration must not call repair_suffix"
        )


class CandidateGenerationModeTests(unittest.TestCase):
    def test_full_regeneration_uses_complete_generated_trace(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = temp_path / "generated.jsonl"
            output_dir = temp_path / "runs"
            config_path = temp_path / "full-regeneration.yaml"

            dataset_path.write_text(
                json.dumps(
                    {
                        "id": "q1",
                        "question": "Is the test condition true?",
                        "model_trace": (
                            "Step 1: Original prefix that must be discarded.\n"
                            "Final answer: no"
                        ),
                        "model_answer": "no",
                        "gold_answer": "yes",
                        "task": "strategyqa",
                        "source": "offline-test",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            config = {
                "run_name": "full-regeneration-test",
                "dataset_path": str(dataset_path),
                "output_dir": str(output_dir),
                "task_profile": "strategyqa",
                "model": {
                    "provider": "openai",
                    "name": "fake-model",
                    "temperature": 0.2,
                    "max_output_tokens": 100,
                },
                "candidate_generation": {
                    "mode": "full_regeneration",
                },
                "execution": {
                    "metrics_enabled": True,
                    "max_concurrency": 1,
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
                    "mode": "judge_guard",
                    "support_tolerance": 0.05,
                    "max_regression_risk": 0.35,
                },
            }
            config_path.write_text(
                yaml.safe_dump(config),
                encoding="utf-8",
            )

            calls = []

            def make_client(*, model, metrics, **execution_options):
                return FullRegenerationClient(
                    model=model,
                    metrics=metrics,
                    call_log=calls,
                )

            with (
                patch(
                    "src.cli.OpenAIClientWrapper",
                    side_effect=make_client,
                ),
                patch("src.cli.NLIScorer", FakeNLIScorer),
            ):
                build_candidate_pool(
                    config=str(config_path),
                    input_path=str(dataset_path),
                    output_path=None,
                    reuse_original_scores=False,
                    resume=False,
                )

            pool_path = (
                output_dir
                / "full-regeneration-test"
                / "candidate_pool.jsonl"
            )
            record = json.loads(
                pool_path.read_text(encoding="utf-8").strip()
            )
            pool = record["candidate_pool"]
            proposal = pool["proposals"][0]

            self.assertEqual(
                pool["candidate_generation_mode"],
                "full_regeneration",
            )
            self.assertEqual(
                proposal["generation_mode"],
                "full_regeneration",
            )
            self.assertNotIn(
                "Original prefix that must be discarded",
                proposal["candidate_trace"],
            )
            self.assertTrue(
                proposal["candidate_trace"].startswith(
                    "Step 1: Repaired reasoning from a completely new trace."
                )
            )
            self.assertEqual(proposal["candidate_answer"], "yes")
            self.assertIn("trace_generation", calls)
            self.assertNotIn("suffix_repair", calls)

            metrics_path = (
                output_dir
                / "full-regeneration-test"
                / "metrics.build_candidate_pool.json"
            )
            metrics = json.loads(
                metrics_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                metrics["operations"]
                ["openai.trace_generation"]
                ["logical_calls"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
