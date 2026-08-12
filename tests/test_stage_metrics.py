import json
import tempfile
import unittest
from pathlib import Path

from src.execution_metrics import ExecutionMetrics


class StageMetricsTests(unittest.TestCase):
    def test_stage_context_records_success_metadata_and_aggregate(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            output_path = Path(temp_dir) / "metrics.json"
            metrics = ExecutionMetrics(
                run_name="stage-test",
                command="test-command",
                output_path=output_path,
            )

            with metrics.stage(
                "repair_wave",
                metadata={"iteration": 1, "items": 3},
            ):
                pass

            metrics.finish()
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            stage = payload["stages"]["repair_wave"]
            event = payload["stage_events"][0]

            self.assertEqual(stage["calls"], 1)
            self.assertEqual(stage["successes"], 1)
            self.assertEqual(stage["failures"], 0)
            self.assertGreaterEqual(stage["elapsed_seconds"], 0.0)
            self.assertEqual(event["name"], "repair_wave")
            self.assertTrue(event["success"])
            self.assertEqual(
                event["metadata"],
                {"iteration": 1, "items": 3},
            )

    def test_stage_context_records_failure_and_reraises(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            output_path = Path(temp_dir) / "metrics.json"
            metrics = ExecutionMetrics(
                run_name="stage-failure-test",
                command="test-command",
                output_path=output_path,
            )

            with self.assertRaisesRegex(RuntimeError, "stage failed"):
                with metrics.stage("candidate_verification_wave"):
                    raise RuntimeError("stage failed")

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            stage = payload["stages"]["candidate_verification_wave"]
            event = payload["stage_events"][0]

            self.assertEqual(stage["calls"], 1)
            self.assertEqual(stage["successes"], 0)
            self.assertEqual(stage["failures"], 1)
            self.assertEqual(stage["error_types"], {"RuntimeError": 1})
            self.assertFalse(event["success"])
            self.assertEqual(event["error_type"], "RuntimeError")

    def test_disabled_stage_metrics_do_not_write_a_file(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            output_path = Path(temp_dir) / "disabled.json"
            metrics = ExecutionMetrics(
                run_name="disabled-stage-test",
                command="test-command",
                output_path=output_path,
                enabled=False,
            )

            with metrics.stage("disabled_stage"):
                pass

            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
