import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from src.cli import generate_traces
from src.generation_checkpoint import (
    save_generation_checkpoint as real_save_checkpoint,
)


class FakeGenerationClient:
    def __init__(self, *, model, metrics, call_log):
        self.model = model
        self.metrics = metrics
        self.call_log = call_log

    def generate_trace(
        self,
        question,
        *,
        task="math",
        temperature=0.2,
        max_output_tokens=800,
    ):
        self.call_log.append(question)
        self.metrics.record_operation(
            provider="openai",
            operation="trace_generation",
            elapsed_seconds=0.01,
            success=True,
            usage={
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        )
        trace = (
            f"Step 1: Consider {question}\n"
            "Final answer: yes"
        )
        return trace, "yes"


class GenerationResumeTests(unittest.TestCase):
    def test_resume_skips_completed_chunks_and_matches_clean_run(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = temp_path / "dataset.jsonl"
            output_dir = temp_path / "runs"

            dataset = [
                {
                    "id": f"q{index}",
                    "question": f"Question {index}?",
                    "gold_answer": "yes",
                    "task": "strategyqa",
                }
                for index in range(1, 6)
            ]
            dataset_path.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in dataset
                )
                + "\n",
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
                        "metrics_flush_interval_events": 100,
                        "max_concurrency": 1,
                        "generation_chunk_size": 2,
                        "request_timeout_seconds": 45.0,
                        "max_retries": 1,
                        "retry_base_delay_seconds": 0.5,
                        "retry_jitter_fraction": 0.0,
                        "openai_requests_per_minute": None,
                        "rate_limit_headroom_fraction": 0.8,
                    },
                }
                config_path.write_text(
                    yaml.safe_dump(config),
                    encoding="utf-8",
                )
                return config_path

            resumed_config = write_config("generation-resume-run")
            clean_config = write_config("generation-clean-run")
            resumed_call_log = []

            def make_resumed_client(
                *,
                model,
                metrics,
                **execution_options,
            ):
                return FakeGenerationClient(
                    model=model,
                    metrics=metrics,
                    call_log=resumed_call_log,
                )

            interrupted = False

            def save_then_interrupt(path, **kwargs):
                nonlocal interrupted
                real_save_checkpoint(path, **kwargs)

                if (
                    kwargs["next_source_index"] == 2
                    and not interrupted
                ):
                    interrupted = True
                    raise RuntimeError("simulated interruption")

            with (
                patch(
                    "src.cli.OpenAIClientWrapper",
                    side_effect=make_resumed_client,
                ),
                patch(
                    "src.cli.save_generation_checkpoint",
                    side_effect=save_then_interrupt,
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "simulated interruption",
                ):
                    generate_traces(
                        config=str(resumed_config),
                        resume=False,
                    )

            resumed_dir = output_dir / "generation-resume-run"
            generated_path = resumed_dir / "generated.jsonl"
            checkpoint_path = generated_path.with_suffix(
                generated_path.suffix + ".checkpoint.json"
            )

            self.assertTrue(checkpoint_path.exists())
            self.assertFalse(generated_path.exists())
            self.assertEqual(
                resumed_call_log,
                ["Question 1?", "Question 2?"],
            )

            interrupted_metrics = json.loads(
                (
                    resumed_dir
                    / "metrics.generate_traces.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                interrupted_metrics["records_processed"],
                2,
            )
            self.assertEqual(
                interrupted_metrics["operations"]
                ["openai.trace_generation"]
                ["logical_calls"],
                2,
            )
            self.assertEqual(
                interrupted_metrics["stages"]
                ["checkpoint_write"]
                ["failures"],
                1,
            )

            with patch(
                "src.cli.OpenAIClientWrapper",
                side_effect=make_resumed_client,
            ):
                generate_traces(
                    config=str(resumed_config),
                    resume=True,
                )

            self.assertEqual(
                resumed_call_log,
                [
                    "Question 1?",
                    "Question 2?",
                    "Question 3?",
                    "Question 4?",
                    "Question 5?",
                ],
            )
            self.assertFalse(checkpoint_path.exists())

            resumed_metrics = json.loads(
                (
                    resumed_dir
                    / "metrics.generate_traces.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                resumed_metrics["records_processed"],
                5,
            )
            self.assertEqual(
                resumed_metrics["operations"]
                ["openai.trace_generation"]
                ["logical_calls"],
                3,
            )
            self.assertIn(
                "checkpoint_loading",
                resumed_metrics["stages"],
            )

            clean_call_log = []

            def make_clean_client(
                *,
                model,
                metrics,
                **execution_options,
            ):
                return FakeGenerationClient(
                    model=model,
                    metrics=metrics,
                    call_log=clean_call_log,
                )

            with patch(
                "src.cli.OpenAIClientWrapper",
                side_effect=make_clean_client,
            ):
                generate_traces(
                    config=str(clean_config),
                    resume=False,
                )

            def read_records(path):
                return [
                    json.loads(line)
                    for line in path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                ]

            resumed_records = read_records(generated_path)
            clean_records = read_records(
                output_dir
                / "generation-clean-run"
                / "generated.jsonl"
            )

            self.assertEqual(resumed_records, clean_records)
            self.assertEqual(len(clean_call_log), 5)


if __name__ == "__main__":
    unittest.main()
