import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.execution_metrics import ExecutionMetrics


class MetricsBufferingTests(unittest.TestCase):
    def test_snapshot_writes_are_buffered_until_interval(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            output_path = Path(temp_dir) / "metrics.json"
            metrics = ExecutionMetrics(
                run_name="buffer-test",
                command="test",
                output_path=output_path,
                flush_interval_events=3,
            )

            with patch.object(
                metrics,
                "_write_snapshot_locked",
                wraps=metrics._write_snapshot_locked,
            ) as write_mock:
                for _ in range(2):
                    metrics.record_operation(
                        provider="openai",
                        operation="test_call",
                        elapsed_seconds=0.01,
                        success=True,
                    )

                self.assertEqual(write_mock.call_count, 0)
                initial_payload = json.loads(
                    output_path.read_text(encoding="utf-8")
                )
                self.assertEqual(initial_payload["operations"], {})

                metrics.record_operation(
                    provider="openai",
                    operation="test_call",
                    elapsed_seconds=0.01,
                    success=True,
                )

                self.assertEqual(write_mock.call_count, 1)
                flushed_payload = json.loads(
                    output_path.read_text(encoding="utf-8")
                )
                self.assertEqual(
                    flushed_payload["operations"]
                    ["openai.test_call"]["logical_calls"],
                    3,
                )

                metrics.record_operation(
                    provider="openai",
                    operation="test_call",
                    elapsed_seconds=0.01,
                    success=True,
                )
                self.assertEqual(write_mock.call_count, 1)

                metrics.flush()
                self.assertEqual(write_mock.call_count, 2)
                manual_payload = json.loads(
                    output_path.read_text(encoding="utf-8")
                )
                self.assertEqual(
                    manual_payload["operations"]
                    ["openai.test_call"]["logical_calls"],
                    4,
                )

                # A clean flush with no new events should not rewrite.
                metrics.flush()
                self.assertEqual(write_mock.call_count, 2)

    def test_finish_forces_final_buffered_snapshot(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            output_path = Path(temp_dir) / "metrics.json"
            metrics = ExecutionMetrics(
                run_name="finish-test",
                command="test",
                output_path=output_path,
                flush_interval_events=100,
            )

            metrics.record_operation(
                provider="openai",
                operation="test_call",
                elapsed_seconds=0.01,
                success=True,
            )
            metrics.set_records_processed(7)

            before_finish = json.loads(
                output_path.read_text(encoding="utf-8")
            )
            self.assertEqual(before_finish["operations"], {})
            self.assertEqual(before_finish["records_processed"], 0)

            metrics.finish(records_processed=7)

            after_finish = json.loads(
                output_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                after_finish["operations"]
                ["openai.test_call"]["logical_calls"],
                1,
            )
            self.assertEqual(after_finish["records_processed"], 7)
            self.assertIsNotNone(
                after_finish["completed_at_utc"]
            )

    def test_default_interval_preserves_immediate_writes(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            output_path = Path(temp_dir) / "metrics.json"
            metrics = ExecutionMetrics(
                run_name="default-test",
                command="test",
                output_path=output_path,
            )

            metrics.record_operation(
                provider="openai",
                operation="test_call",
                elapsed_seconds=0.01,
                success=True,
            )

            payload = json.loads(
                output_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                payload["operations"]
                ["openai.test_call"]["logical_calls"],
                1,
            )

    def test_invalid_flush_interval_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            with self.assertRaises(ValueError):
                ExecutionMetrics(
                    run_name="invalid-test",
                    command="test",
                    output_path=(
                        Path(temp_dir) / "metrics.json"
                    ),
                    flush_interval_events=0,
                )


if __name__ == "__main__":
    unittest.main()
