import tempfile
import unittest
from pathlib import Path

from scripts.freeze_processbench_protocol import (
    GOLD_FIELDS,
    blind_record,
    label_record,
    select_development_records,
)


def record(index, label):
    return {
        "id": f"processbench:gsm8k:gsm8k-{index}",
        "question": f"Question {index}",
        "benchmark_steps": ["Step A", "Step B"],
        "gold_label": label,
        "gold_earliest_bad_step": None if label == -1 else label,
        "final_answer_correct": label == -1,
        "benchmark": {"split": "gsm8k"},
    }


class ProcessBenchProtocolTests(unittest.TestCase):
    def test_development_selection_is_deterministic_and_includes_smoke(self):
        records = [
            record(index, 0 if index < 10 else -1)
            for index in range(20)
        ]
        required = {
            "processbench:gsm8k:gsm8k-0",
            "processbench:gsm8k:gsm8k-10",
        }
        first = select_development_records(
            records,
            required_ids=required,
            seed=123,
            error_count=4,
            correct_count=4,
        )
        second = select_development_records(
            list(reversed(records)),
            required_ids=required,
            seed=123,
            error_count=4,
            correct_count=4,
        )

        self.assertEqual(
            {row["id"] for row in first},
            {row["id"] for row in second},
        )
        self.assertTrue(required.issubset(row["id"] for row in first))
        self.assertEqual(sum(row["gold_label"] != -1 for row in first), 4)
        self.assertEqual(sum(row["gold_label"] == -1 for row in first), 4)

    def test_blinded_input_contains_no_gold_fields(self):
        source = record(1, 0)
        blinded = blind_record(source)
        self.assertFalse(GOLD_FIELDS.intersection(blinded))
        self.assertEqual(blinded["id"], source["id"])
        self.assertEqual(blinded["benchmark_steps"], source["benchmark_steps"])

    def test_label_record_contains_only_id_and_gold_fields(self):
        source = record(1, 0)
        labels = label_record(source)
        self.assertEqual(set(labels), {"id", *GOLD_FIELDS})
        self.assertEqual(labels["gold_label"], 0)
        self.assertNotIn("question", labels)

    def test_missing_required_smoke_id_is_rejected(self):
        records = [record(index, -1) for index in range(10)]
        with self.assertRaisesRegex(ValueError, "Required smoke ids are missing"):
            select_development_records(
                records,
                required_ids=["missing"],
                error_count=0,
                correct_count=4,
            )


if __name__ == "__main__":
    unittest.main()
