import json
import tempfile
import unittest
from pathlib import Path

from src.execution_metrics import ExecutionMetrics


class ExecutionMetricsTests(unittest.TestCase):
    def test_records_success_failure_timing_and_usage(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            output_path = Path(temp_dir) / "metrics.json"

            metrics = ExecutionMetrics(
                run_name="test-run",
                command="test-command",
                output_path=output_path,
                execution_config={
                    "metrics_enabled": True,
                    "max_concurrency": 1,
                },
                context={
                    "dataset_path": "example.jsonl",
                    "model": {"name": "fake-model"},
                },
            )

            metrics.record_operation(
                provider="openai",
                operation="trace_generation",
                elapsed_seconds=1.25,
                success=True,
                usage={
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 30,
                    "cached_input_tokens": 4,
                    "reasoning_output_tokens": 3,
                },
            )

            metrics.record_operation(
                provider="openai",
                operation="trace_generation",
                elapsed_seconds=0.5,
                success=False,
                error_type="TimeoutError",
            )

            metrics.set_records_processed(1)
            metrics.finish(records_processed=1)

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            operation = payload["operations"]["openai.trace_generation"]

            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["records_processed"], 1)
            self.assertIsNotNone(payload["completed_at_utc"])

            self.assertEqual(operation["logical_calls"], 2)
            self.assertEqual(operation["attempts"], 2)
            self.assertEqual(operation["successes"], 1)
            self.assertEqual(operation["failures"], 1)
            self.assertEqual(operation["input_tokens"], 10)
            self.assertEqual(operation["output_tokens"], 20)
            self.assertEqual(operation["total_tokens"], 30)
            self.assertEqual(operation["cached_input_tokens"], 4)
            self.assertEqual(operation["reasoning_output_tokens"], 3)
            self.assertEqual(
                operation["error_types"],
                {"TimeoutError": 1},
            )

            self.assertEqual(payload["totals"]["logical_calls"], 2)
            self.assertEqual(payload["totals"]["attempts"], 2)
            self.assertEqual(payload["totals"]["failures"], 1)
            self.assertGreaterEqual(payload["wall_time_seconds"], 0.0)

    def test_disabled_metrics_do_not_write_a_file(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            output_path = Path(temp_dir) / "disabled.json"

            metrics = ExecutionMetrics(
                run_name="disabled",
                command="disabled",
                output_path=output_path,
                enabled=False,
            )

            metrics.record_operation(
                provider="openai",
                operation="trace_generation",
                elapsed_seconds=1.0,
                success=True,
            )
            metrics.finish(records_processed=1)

            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()