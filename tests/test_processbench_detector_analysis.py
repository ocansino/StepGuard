import unittest

from scripts.analyze_processbench_detector import (
    analyze_development,
    average_precision,
    evaluate_detector,
    roc_auc,
)
from scripts.processbench_runner import PREDICTION_SCHEMA_VERSION


def prediction(record_id, gold_label, verifier, contradiction):
    steps = [f"Step {index}" for index in range(len(verifier))]
    judgments = [
        {
            "step_index": index,
            "verdict": "wrong" if score > 0.5 else "correct",
            "p_wrong": score,
            "explanation": f"Explanation {index}",
        }
        for index, score in enumerate(verifier)
    ]
    return {
        "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
        "id": record_id,
        "gold_label": gold_label,
        "predicted_label": -1,
        "stepguard_score": {
            "steps": steps,
            "scores": {
                "verifier": verifier,
                "contradiction": contradiction,
                "evidence_support": None,
            },
            "raw_verifier_judgments": judgments,
            "verifier_audit": {
                "expected_steps": len(steps),
                "received_judgments": len(steps),
                "complete": True,
            },
        },
    }


class ProcessBenchDetectorAnalysisTests(unittest.TestCase):
    def test_auc_and_average_precision_for_perfect_ranking(self):
        labels = [0, 1, 0, 1]
        scores = [0.1, 0.9, 0.2, 0.8]
        self.assertEqual(roc_auc(labels, scores), 1.0)
        self.assertEqual(average_precision(labels, scores), 1.0)

    def test_step_metrics_exclude_unlabeled_post_error_steps(self):
        records = [
            prediction("error", 1, [0.1, 0.9, 0.99], [0.0, 0.0, 0.0]),
            prediction("correct", -1, [0.1, 0.1], [0.0, 0.0]),
        ]
        result = evaluate_detector(
            records,
            scoring_cfg={"risk_formula": "verifier_only"},
            threshold=0.2,
        )
        step = result["risk_association"]["step_level"]

        self.assertEqual(step["examples"], 4)
        self.assertEqual(step["positive_examples"], 1)
        self.assertEqual(step["true_positive"], 1)
        self.assertEqual(step["false_positive"], 0)
        self.assertEqual(step["f1"], 1.0)

    def test_development_calibration_is_deterministic(self):
        records = [
            prediction("error-1", 1, [0.1, 0.9], [0.0, 0.0]),
            prediction("error-2", 0, [0.8, 0.2], [0.0, 0.0]),
            prediction("correct-1", -1, [0.1, 0.1], [0.0, 0.0]),
            prediction("correct-2", -1, [0.05, 0.1], [0.0, 0.0]),
        ]
        first = analyze_development(
            records,
            weight_grid=[0.5, 1.0],
            threshold_grid=[0.2, 0.5],
        )
        second = analyze_development(
            list(reversed(records)),
            weight_grid=[0.5, 1.0],
            threshold_grid=[0.2, 0.5],
        )

        self.assertEqual(
            first["calibration"]["selected"]["configuration"],
            second["calibration"]["selected"]["configuration"],
        )
        self.assertEqual(
            first["calibration"]["selected"]["localization"],
            second["calibration"]["selected"]["localization"],
        )
        self.assertEqual(first["calibration"]["evaluated_configurations"], 4)

    def test_analysis_requires_audited_prediction_schema(self):
        row = prediction("error", 0, [0.9], [0.0])
        row["prediction_schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "audit version"):
            evaluate_detector(
                [row],
                scoring_cfg={"risk_formula": "verifier_only"},
                threshold=0.2,
            )


if __name__ == "__main__":
    unittest.main()
