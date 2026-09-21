import json
import tempfile
import unittest
from pathlib import Path

from scripts.processbench import (
    PINNED_REVISION,
    convert_record,
    evaluate_labels,
    load_source,
    prepare_split,
    select_smoke_subset,
    validate_source_records,
)
from src.io_utils import read_jsonl
from src.parsing import split_steps


def source_record(**overrides):
    record = {
        "id": "gsm8k-0",
        "generator": "example-generator",
        "problem": "What is 2 + 2?",
        "steps": ["Add the two numbers.", "The result is 4."],
        "final_answer_correct": True,
        "label": -1,
    }
    record.update(overrides)
    return record


class ProcessBenchAdapterTests(unittest.TestCase):
    def test_correct_trace_preserves_steps_and_maps_minus_one_to_none(self):
        converted = convert_record(source_record(), "gsm8k")

        self.assertEqual(converted["gold_label"], -1)
        self.assertIsNone(converted["gold_earliest_bad_step"])
        self.assertEqual(converted["benchmark_steps"], source_record()["steps"])
        self.assertEqual(len(split_steps(converted["model_trace"])), 2)
        self.assertEqual(converted["benchmark"]["revision"], PINNED_REVISION)

    def test_error_label_remains_zero_based(self):
        converted = convert_record(source_record(label=1), "gsm8k")
        self.assertEqual(converted["gold_label"], 1)
        self.assertEqual(converted["gold_earliest_bad_step"], 1)

    def test_process_error_is_allowed_even_when_final_answer_is_correct(self):
        converted = convert_record(
            source_record(label=0, final_answer_correct=True),
            "gsm8k",
        )
        self.assertEqual(converted["gold_label"], 0)
        self.assertTrue(converted["final_answer_correct"])

    def test_out_of_range_label_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            convert_record(source_record(label=2), "gsm8k")

    def test_boolean_label_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "integer"):
            convert_record(source_record(label=True), "gsm8k")

    def test_duplicate_source_ids_are_rejected(self):
        rows = [source_record(), source_record()]
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            validate_source_records(rows, "gsm8k")

    def test_prepare_split_writes_jsonl_and_hash_manifest_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "gsm8k.json"
            output = root / "gsm8k.jsonl"
            source.write_text(json.dumps([source_record()]), encoding="utf-8")

            summary = prepare_split(
                source,
                output,
                "gsm8k",
                enforce_official_count=False,
            )
            prepared = list(read_jsonl(output))

            self.assertEqual(summary["records"], 1)
            self.assertEqual(len(summary["source_sha256"]), 64)
            self.assertEqual(len(summary["output_sha256"]), 64)
            self.assertEqual(prepared[0]["id"], "processbench:gsm8k:gsm8k-0")

    def test_prepare_rejects_truncated_official_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "gsm8k.json"
            source.write_text(json.dumps([source_record()]), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "expected 400"):
                prepare_split(source, root / "gsm8k.jsonl", "gsm8k")

    def test_source_file_must_contain_a_json_list(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gsm8k.json"
            path.write_text(json.dumps(source_record()), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON list"):
                load_source(path, "gsm8k")

    def test_smoke_subset_is_deterministic_and_stratified(self):
        rows = [
            {
                **convert_record(
                    source_record(id=f"gsm8k-{index}", label=label),
                    "gsm8k",
                )
            }
            for index, label in enumerate([-1, 0, -1, 1, 0, -1])
        ]
        first = select_smoke_subset(
            rows,
            seed=891,
            error_count=2,
            correct_count=2,
        )
        second = select_smoke_subset(
            list(reversed(rows)),
            seed=891,
            error_count=2,
            correct_count=2,
        )

        self.assertEqual(
            [record["id"] for record in first],
            [record["id"] for record in second],
        )
        self.assertEqual(sum(row["gold_label"] != -1 for row in first), 2)
        self.assertEqual(sum(row["gold_label"] == -1 for row in first), 2)


class ProcessBenchMetricTests(unittest.TestCase):
    def test_official_and_detection_metrics_are_distinct(self):
        metrics = evaluate_labels(
            gold_labels=[1, 0, -1, -1],
            predicted_labels=[1, 1, -1, 0],
        )

        official = metrics["official_processbench"]
        self.assertEqual(official["error_localization_accuracy"], 0.5)
        self.assertEqual(official["correct_trace_accuracy"], 0.5)
        self.assertEqual(official["f1"], 0.5)
        self.assertEqual(
            metrics["additional_metrics"]["overall_exact_label_accuracy"],
            0.5,
        )

        detection = metrics["additional_metrics"]["trace_error_detection"]
        self.assertEqual(detection["true_positive"], 2)
        self.assertEqual(detection["false_positive"], 1)
        self.assertEqual(detection["false_negative"], 0)
        self.assertEqual(detection["true_negative"], 1)
        self.assertAlmostEqual(detection["precision"], 2 / 3)
        self.assertEqual(detection["recall"], 1.0)
        self.assertAlmostEqual(detection["f1"], 0.8)

    def test_missing_official_class_is_reported_as_undefined(self):
        metrics = evaluate_labels([-1, -1], [-1, 0])
        official = metrics["official_processbench"]
        self.assertIsNone(official["error_localization_accuracy"])
        self.assertIsNone(official["f1"])

    def test_empty_evaluation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "No predictions"):
            evaluate_labels([], [])


if __name__ == "__main__":
    unittest.main()
