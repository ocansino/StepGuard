import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from src.cli import generate_traces
from src.execution import BoundedExecutionError, map_bounded


class BoundedExecutionTests(unittest.TestCase):
    def test_concurrent_execution_preserves_input_order(self) -> None:
        barrier = threading.Barrier(3)

        def worker(value: int) -> int:
            barrier.wait(timeout=2.0)
            return value * 10

        results = map_bounded(
            [3, 1, 2],
            worker,
            max_workers=3,
        )

        self.assertEqual(results, [30, 10, 20])

    def test_single_worker_uses_calling_thread(self) -> None:
        calling_thread = threading.get_ident()
        observed_threads = []

        def worker(value: int) -> int:
            observed_threads.append(threading.get_ident())
            return value

        self.assertEqual(
            map_bounded([1, 2, 3], worker, max_workers=1),
            [1, 2, 3],
        )
        self.assertEqual(
            observed_threads,
            [calling_thread, calling_thread, calling_thread],
        )

    def test_failure_does_not_cancel_unrelated_items(self) -> None:
        completed = []
        completed_lock = threading.Lock()
        items = [
            {"id": "first"},
            {"id": "bad"},
            {"id": "last"},
        ]

        def worker(item):
            if item["id"] == "bad":
                raise ValueError("intentional failure")

            with completed_lock:
                completed.append(item["id"])
            return item["id"]

        with self.assertRaises(BoundedExecutionError) as caught:
            map_bounded(
                items,
                worker,
                max_workers=3,
                item_id=lambda item: item["id"],
            )

        self.assertCountEqual(completed, ["first", "last"])
        self.assertEqual(caught.exception.completed_items, 2)
        self.assertEqual(caught.exception.total_items, 3)
        self.assertEqual(len(caught.exception.failures), 1)
        self.assertEqual(caught.exception.failures[0].item_id, "bad")
        self.assertEqual(caught.exception.failures[0].error_type, "ValueError")

    def test_rejects_nonpositive_worker_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_workers"):
            map_bounded([1], lambda value: value, max_workers=0)


class FakeGenerationClient:
    def __init__(self, *, model, metrics, barrier) -> None:
        self.model = model
        self.metrics = metrics
        self.barrier = barrier

    def generate_trace(
        self,
        question,
        *,
        task="math",
        temperature=0.2,
        max_output_tokens=800,
    ):
        self.barrier.wait(timeout=2.0)
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
        trace = f"Step 1: Consider {question}\nFinal answer: yes"
        return trace, "yes"


class ConcurrentGenerationIntegrationTests(unittest.TestCase):
    def test_generate_traces_runs_concurrently_and_preserves_dataset_order(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = temp_path / "dataset.jsonl"
            config_path = temp_path / "config.yaml"
            output_dir = temp_path / "runs"

            dataset = [
                {
                    "id": "q1",
                    "question": "Question one?",
                    "gold_answer": "yes",
                    "task": "strategyqa",
                },
                {
                    "id": "q2",
                    "question": "Question two?",
                    "gold_answer": "yes",
                    "task": "strategyqa",
                },
                {
                    "id": "q3",
                    "question": "Question three?",
                    "gold_answer": "yes",
                    "task": "strategyqa",
                },
            ]
            dataset_path.write_text(
                "\n".join(json.dumps(row) for row in dataset) + "\n",
                encoding="utf-8",
            )

            config = {
                "run_name": "concurrent-generation-test",
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
                    "max_concurrency": 3,
                    "request_timeout_seconds": 45.0,
                    "max_retries": 1,
                    "retry_base_delay_seconds": 0.5,
                    "retry_jitter_fraction": 0.15,
                    "openai_requests_per_minute": 120.0,
                    "rate_limit_headroom_fraction": 0.75,
                },
            }
            config_path.write_text(
                yaml.safe_dump(config),
                encoding="utf-8",
            )

            barrier = threading.Barrier(3)
            client_options = []

            def make_client(*, model, metrics, **execution_options):
                client_options.append(execution_options)
                return FakeGenerationClient(
                    model=model,
                    metrics=metrics,
                    barrier=barrier,
                )

            with patch(
                "src.cli.OpenAIClientWrapper",
                side_effect=make_client,
            ):
                generate_traces(
                    config=str(config_path),
                    resume=False,
                )

            run_dir = output_dir / "concurrent-generation-test"
            generated = [
                json.loads(line)
                for line in (run_dir / "generated.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            metrics = json.loads(
                (run_dir / "metrics.generate_traces.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                [record["id"] for record in generated],
                ["q1", "q2", "q3"],
            )
            self.assertEqual(
                [record["question"] for record in generated],
                [
                    "Question one?",
                    "Question two?",
                    "Question three?",
                ],
            )
            self.assertEqual(metrics["records_processed"], 3)
            self.assertEqual(metrics["execution_config"]["max_concurrency"], 3)
            self.assertEqual(len(client_options), 3)

            rate_limiters = [
                options["rate_limiter"]
                for options in client_options
            ]
            self.assertTrue(
                all(
                    limiter is rate_limiters[0]
                    for limiter in rate_limiters
                )
            )
            self.assertEqual(
                rate_limiters[0].requests_per_minute,
                120.0,
            )
            self.assertEqual(
                rate_limiters[0].headroom_fraction,
                0.75,
            )
            self.assertIsNotNone(rate_limiters[0].metrics)
            self.assertTrue(
                all(
                    {
                        key: value
                        for key, value in options.items()
                        if key != "rate_limiter"
                    }
                    == {
                        "request_timeout_seconds": 45.0,
                        "max_retries": 1,
                        "retry_base_delay_seconds": 0.5,
                        "retry_jitter_fraction": 0.15,
                    }
                    for options in client_options
                )
            )
            self.assertEqual(
                metrics["operations"]["openai.trace_generation"]["logical_calls"],
                3,
            )
            self.assertEqual(
                set(metrics["stages"]),
                {
                    "dataset_loading",
                    "trace_generation_wave",
                    "checkpoint_write",
                    "artifact_write",
                    "checkpoint_cleanup",
                },
            )
            generation_event = next(
                event
                for event in metrics["stage_events"]
                if event["name"] == "trace_generation_wave"
            )
            self.assertEqual(
                generation_event["metadata"],
                {"items": 3, "max_concurrency": 3},
            )


if __name__ == "__main__":
    unittest.main()
