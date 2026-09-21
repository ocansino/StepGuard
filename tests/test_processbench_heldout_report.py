import copy
import tempfile
import unittest
from pathlib import Path

from src.config import load_config
from src.io_utils import write_json_atomic, write_jsonl

from scripts.processbench_runner import PREDICTION_SCHEMA_VERSION
from scripts.report_processbench_heldout import (
    build_heldout_report,
    freeze_prediction_receipt,
    join_blind_labels,
    report_from_files,
    sha256_file,
    validate_frozen_plan,
)


PRIMARY = {
    "scoring": {
        "risk_formula": "weighted",
        "verifier_weight": 0.75,
        "contradiction_weight": 0.25,
    },
    "risk_threshold": 0.2,
}

SELECTED = {
    "scoring": {
        "risk_formula": "weighted",
        "verifier_weight": 1.0,
        "contradiction_weight": 0.0,
    },
    "risk_threshold": 0.05,
}


def plan(records=4, labels_sha256="0" * 64, resamples=100):
    return {
        "schema_version": 1,
        "status": "frozen",
        "selection_source": "development_only",
        "detectors": {
            "primary_submitted": copy.deepcopy(PRIMARY),
            "development_selected": copy.deepcopy(SELECTED),
        },
        "heldout_partition": {
            "name": "fixture",
            "records": records,
            "labels_sha256": labels_sha256,
        },
        "bootstrap": {
            "method": "stratified_nonparametric_percentile_bootstrap",
            "strata": "gold_error_vs_correct_trace",
            "paired_detector_comparison": True,
            "resamples": resamples,
            "seed": 20260920,
            "confidence_level": 0.95,
        },
    }


def prediction(record_id, verifier, contradiction, gold_label=None):
    steps = [f"Step {index}" for index in range(len(verifier))]
    row = {
        "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
        "id": record_id,
        "predicted_label": -1,
        "detector_identity": {
            "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
            "risk_threshold": 0.2,
            "scoring": {
                "risk_threshold": 0.2,
                "risk_formula": "weighted",
                "verifier_weight": 0.75,
                "contradiction_weight": 0.25,
            },
        },
        "stepguard_score": {
            "steps": steps,
            "scores": {
                "verifier": verifier,
                "contradiction": contradiction,
                "evidence_support": None,
            },
            "raw_verifier_judgments": [
                {
                    "step_index": index,
                    "verdict": "invalid" if score > 0.5 else "valid",
                    "p_wrong": score,
                }
                for index, score in enumerate(verifier)
            ],
            "verifier_audit": {
                "expected_steps": len(steps),
                "received_judgments": len(steps),
                "complete": True,
            },
        },
    }
    if gold_label is not None:
        row["gold_label"] = gold_label
    return row


def evaluated_fixture():
    return [
        prediction("error-1", [0.01, 0.9], [0.0, 0.0], 1),
        prediction("error-2", [0.08, 0.01], [0.0, 0.0], 0),
        prediction("correct-1", [0.01, 0.01], [0.9, 0.0], -1),
        prediction("correct-2", [0.08, 0.0], [0.0, 0.0], -1),
    ]


class ProcessBenchHeldoutReportTests(unittest.TestCase):
    def test_frozen_plan_rejects_any_embedded_search_space(self):
        value = plan()
        value["threshold_grid"] = [0.05, 0.2]

        with self.assertRaisesRegex(ValueError, "forbidden search key"):
            validate_frozen_plan(value)

    def test_report_evaluates_only_the_two_frozen_detectors(self):
        value = build_heldout_report(
            records=evaluated_fixture(),
            plan=plan(),
            bootstrap_resamples=50,
        )

        self.assertFalse(value["selection_performed"])
        self.assertEqual(
            list(value["detectors"]),
            ["primary_submitted", "development_selected"],
        )
        self.assertEqual(
            value["detectors"]["development_selected"]["configuration"],
            SELECTED,
        )
        self.assertEqual(value["bootstrap"]["resamples"], 50)
        self.assertIn(
            "official_processbench_f1",
            value["paired_comparison"]["confidence_intervals"],
        )
        self.assertIn(
            "step_precision",
            value["paired_comparison"]["confidence_intervals"],
        )

    def test_paired_bootstrap_is_deterministic(self):
        first = build_heldout_report(
            records=evaluated_fixture(),
            plan=plan(),
            bootstrap_resamples=75,
        )
        second = build_heldout_report(
            records=evaluated_fixture(),
            plan=plan(),
            bootstrap_resamples=75,
        )

        self.assertEqual(first["bootstrap"], second["bootstrap"])
        self.assertEqual(
            first["paired_comparison"],
            second["paired_comparison"],
        )

    def test_prediction_receipt_refuses_unblinded_rows(self):
        rows = evaluated_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions_path = root / "predictions.jsonl"
            plan_path = root / "plan.json"
            write_jsonl(predictions_path, rows)
            write_json_atomic(plan_path, plan())

            with self.assertRaisesRegex(ValueError, "not blinded"):
                freeze_prediction_receipt(
                    predictions_path=predictions_path,
                    plan_path=plan_path,
                )

    def test_report_requires_pre_unblinding_prediction_hash(self):
        blind_rows = [
            {key: value for key, value in row.items() if key != "gold_label"}
            for row in evaluated_fixture()
        ]
        labels = [
            {"id": row["id"], "gold_label": row["gold_label"]}
            for row in evaluated_fixture()
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions_path = root / "predictions.jsonl"
            labels_path = root / "labels.jsonl"
            plan_path = root / "plan.json"
            receipt_path = root / "receipt.json"
            write_jsonl(predictions_path, blind_rows)
            write_jsonl(labels_path, labels)
            write_json_atomic(
                plan_path,
                plan(labels_sha256=sha256_file(labels_path), resamples=25),
            )
            receipt = freeze_prediction_receipt(
                predictions_path=predictions_path,
                plan_path=plan_path,
            )
            write_json_atomic(receipt_path, receipt)

            blind_rows[0]["predicted_label"] = 0
            write_jsonl(predictions_path, blind_rows)

            with self.assertRaisesRegex(ValueError, "hash differs"):
                report_from_files(
                    predictions_path=predictions_path,
                    labels_path=labels_path,
                    plan_path=plan_path,
                    receipt_path=receipt_path,
                )

    def test_label_join_requires_exact_id_order(self):
        rows = [
            {key: value for key, value in row.items() if key != "gold_label"}
            for row in evaluated_fixture()
        ]
        labels = [{"id": "wrong-id", "gold_label": 1}]

        with self.assertRaisesRegex(ValueError, "counts differ|id mismatch"):
            join_blind_labels(rows, labels)

    def test_repository_frozen_plan_validates_and_pins_development_files(self):
        root = Path(__file__).resolve().parents[1]
        plan_path = (
            root
            / "data"
            / "prepared"
            / "processbench"
            / "protocol_v1"
            / "gsm8k_heldout_300_detector_plan.json"
        )
        value = validate_frozen_plan(__import__("json").loads(plan_path.read_text()))
        evidence = value["development_evidence"]

        self.assertEqual(
            sha256_file(root / evidence["predictions_path"]),
            evidence["predictions_sha256"],
        )
        self.assertEqual(
            sha256_file(root / evidence["analysis_path"]),
            evidence["analysis_sha256"],
        )
        self.assertEqual(value["bootstrap"]["resamples"], 10000)
        heldout_config = load_config(
            root
            / "configs"
            / "processbench_gsm8k_openai_gpt_5_4_mini_heldout_300.yaml"
        )
        processbench = heldout_config.raw["processbench"]
        self.assertEqual(
            processbench["detector_plan_path"],
            str(plan_path.relative_to(root)).replace("\\", "/"),
        )
        self.assertEqual(
            processbench["detector_plan_sha256"],
            sha256_file(plan_path),
        )


if __name__ == "__main__":
    unittest.main()
