import unittest

from src.cli import assemble_trace_scores_batch


class RecordingBatchNLI:
    def __init__(self):
        self.calls = []

    def contradiction_probs(self, pairs):
        normalized = list(pairs)
        self.calls.append(normalized)
        return [
            0.1 * (index + 1)
            for index in range(len(normalized))
        ]

    def contradiction_prob(self, premise, hypothesis):
        raise AssertionError(
            "cross-trace batching must not use single-pair scoring"
        )


class NLITraceBatchingTests(unittest.TestCase):
    def test_multiple_traces_use_one_ordered_batch(self):
        nli = RecordingBatchNLI()
        items = [
            {
                "steps": [
                    "Step 1: Alpha.",
                    "Step 2: Beta.",
                    "Step 3: Gamma.",
                ],
                "judge_results": [
                    {"step_index": 0, "p_wrong": 0.1},
                    {"step_index": 1, "p_wrong": 0.2},
                    {"step_index": 2, "p_wrong": 0.3},
                ],
            },
            {
                "steps": [
                    "Step 1: Delta.",
                    "Step 2: Epsilon.",
                ],
                "judge_results": [
                    {"step_index": 0, "p_wrong": 0.4},
                    {"step_index": 1, "p_wrong": 0.5},
                ],
            },
            {
                "steps": ["Step 1: Zeta."],
                "judge_results": [
                    {"step_index": 0, "p_wrong": 0.6}
                ],
            },
        ]

        scores = assemble_trace_scores_batch(
            items=items,
            nli=nli,
            tau=0.8,
            scoring_cfg={"risk_formula": "verifier_only"},
        )

        self.assertEqual(
            nli.calls,
            [[
                ("Step 1: Alpha.", "Step 2: Beta."),
                (
                    "Step 1: Alpha.\nStep 2: Beta.",
                    "Step 3: Gamma.",
                ),
                ("Step 1: Delta.", "Step 2: Epsilon."),
            ]],
        )
        actual_contradictions = [
            score["scores"]["contradiction"]
            for score in scores
        ]
        expected_contradictions = [
            [0.0, 0.1, 0.2],
            [0.0, 0.3],
            [0.0],
        ]

        self.assertEqual(
            [len(values) for values in actual_contradictions],
            [len(values) for values in expected_contradictions],
        )

        for actual_trace, expected_trace in zip(
            actual_contradictions,
            expected_contradictions,
        ):
            for actual, expected in zip(
                actual_trace,
                expected_trace,
            ):
                self.assertAlmostEqual(actual, expected)
        self.assertEqual(
            [score["scores"]["verifier"] for score in scores],
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5],
                [0.6],
            ],
        )

    def test_empty_trace_collection_does_no_nli_work(self):
        nli = RecordingBatchNLI()

        self.assertEqual(
            assemble_trace_scores_batch(
                items=[],
                nli=nli,
                tau=0.8,
                scoring_cfg={},
            ),
            [],
        )
        self.assertEqual(nli.calls, [])


if __name__ == "__main__":
    unittest.main()
