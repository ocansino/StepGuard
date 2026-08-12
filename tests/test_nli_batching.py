import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.execution_metrics import ExecutionMetrics
from src.scorers.nli import NLIScorer


class FakeTokenizer:
    def __init__(self):
        self.batch_sizes = []

    def __call__(
        self,
        premises,
        hypotheses,
        *,
        return_tensors,
        padding,
        truncation,
        max_length,
    ):
        if isinstance(premises, str):
            premises = [premises]
            hypotheses = [hypotheses]

        self.batch_sizes.append(len(premises))
        values = [
            float(premise.split("-")[-1])
            for premise in premises
        ]
        return {
            "input_ids": torch.tensor(
                [[value] for value in values],
                dtype=torch.float32,
            ),
            "attention_mask": torch.ones(
                (len(values), 1),
                dtype=torch.float32,
            ),
        }


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(
            id2label={
                0: "CONTRADICTION",
                1: "NEUTRAL",
                2: "ENTAILMENT",
            }
        )
        self.device = None
        self.eval_called = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, **inputs):
        values = inputs["input_ids"][:, 0]
        logits = torch.stack(
            (
                values,
                torch.zeros_like(values),
                -values,
            ),
            dim=1,
        )
        return SimpleNamespace(logits=logits)


class NLIBatchingTests(unittest.TestCase):
    def make_scorer(self, metrics=None, batch_size=2):
        tokenizer = FakeTokenizer()
        model = FakeModel()

        tokenizer_patch = patch(
            "src.scorers.nli.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        )
        model_patch = patch(
            "src.scorers.nli."
            "AutoModelForSequenceClassification.from_pretrained",
            return_value=model,
        )

        with tokenizer_patch, model_patch:
            scorer = NLIScorer(
                model_name="fake-nli",
                device="cpu",
                metrics=metrics,
                batch_size=batch_size,
            )

        return scorer, tokenizer, model

    def test_batching_preserves_order_and_uses_expected_batches(self):
        scorer, tokenizer, model = self.make_scorer(
            batch_size=2
        )
        pairs = [
            (f"premise-{index}", f"hypothesis-{index}")
            for index in range(1, 6)
        ]

        probabilities = scorer.contradiction_probs(pairs)

        expected = [
            float(
                torch.softmax(
                    torch.tensor(
                        [float(index), 0.0, -float(index)]
                    ),
                    dim=0,
                )[0]
            )
            for index in range(1, 6)
        ]
        self.assertEqual(tokenizer.batch_sizes, [2, 2, 1])
        self.assertEqual(len(probabilities), 5)

        for actual, wanted in zip(probabilities, expected):
            self.assertAlmostEqual(actual, wanted)

        self.assertEqual(model.device, "cpu")
        self.assertTrue(model.eval_called)

    def test_single_pair_method_uses_batch_path(self):
        scorer, tokenizer, _ = self.make_scorer(batch_size=8)

        probability = scorer.contradiction_prob(
            premise="premise-2",
            hypothesis="hypothesis-2",
        )

        expected = float(
            torch.softmax(
                torch.tensor([2.0, 0.0, -2.0]),
                dim=0,
            )[0]
        )
        self.assertAlmostEqual(probability, expected)
        self.assertEqual(tokenizer.batch_sizes, [1])

    def test_metrics_count_model_forward_batches(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            metrics_path = Path(temp_dir) / "metrics.json"
            metrics = ExecutionMetrics(
                run_name="nli-batch-test",
                command="test",
                output_path=metrics_path,
            )
            scorer, _, _ = self.make_scorer(
                metrics=metrics,
                batch_size=2,
            )

            scorer.contradiction_probs(
                [
                    (f"premise-{index}", f"hypothesis-{index}")
                    for index in range(1, 6)
                ]
            )
            metrics.finish()

            payload = json.loads(
                metrics_path.read_text(encoding="utf-8")
            )
            operation = payload["operations"][
                "local_roberta.contradiction_scoring"
            ]

            self.assertEqual(operation["logical_calls"], 3)
            self.assertEqual(operation["attempts"], 3)
            self.assertEqual(operation["successes"], 3)
            self.assertEqual(operation["failures"], 0)

    def test_empty_input_returns_without_model_work(self):
        scorer, tokenizer, _ = self.make_scorer()

        self.assertEqual(scorer.contradiction_probs([]), [])
        self.assertEqual(tokenizer.batch_sizes, [])

    def test_invalid_batch_size_is_rejected_before_model_load(self):
        with self.assertRaises(ValueError):
            NLIScorer(
                model_name="fake-nli",
                device="cpu",
                batch_size=0,
            )


if __name__ == "__main__":
    unittest.main()
