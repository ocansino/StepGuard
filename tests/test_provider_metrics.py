import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.execution_metrics import ExecutionMetrics
from src.providers.openai_client import OpenAIClientWrapper


class FakeResponses:
    def create(self, **request):
        return SimpleNamespace(
            output_text="Step 1: Add the values.\nFinal answer: 4",
            usage=SimpleNamespace(
                input_tokens=12,
                output_tokens=8,
                total_tokens=20,
                input_tokens_details=SimpleNamespace(cached_tokens=3),
                output_tokens_details=SimpleNamespace(reasoning_tokens=2),
            ),
        )


class FakeOpenAIClient:
    def __init__(self):
        self.responses = FakeResponses()


class OpenAIProviderMetricsTests(unittest.TestCase):
    def test_generate_trace_records_usage_without_network_access(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            metrics_path = Path(temp_dir) / "metrics.json"
            metrics = ExecutionMetrics(
                run_name="provider-test",
                command="generate-traces",
                output_path=metrics_path,
            )

            with patch(
                "src.providers.openai_client.OpenAI",
                return_value=FakeOpenAIClient(),
            ):
                client = OpenAIClientWrapper(
                    model="fake-model",
                    metrics=metrics,
                )
                trace, answer = client.generate_trace("What is 2 + 2?")

            metrics.finish(records_processed=1)
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            operation = payload["operations"]["openai.trace_generation"]

            self.assertIn("Final answer: 4", trace)
            self.assertEqual(answer, "4")
            self.assertEqual(operation["logical_calls"], 1)
            self.assertEqual(operation["successes"], 1)
            self.assertEqual(operation["failures"], 0)
            self.assertEqual(operation["input_tokens"], 12)
            self.assertEqual(operation["output_tokens"], 8)
            self.assertEqual(operation["total_tokens"], 20)
            self.assertEqual(operation["cached_input_tokens"], 3)
            self.assertEqual(operation["reasoning_output_tokens"], 2)


if __name__ == "__main__":
    unittest.main()
