import json
import tempfile
import unittest
from pathlib import Path

from scripts.freeze_strategyqa_extension import freeze_extension


class StrategyQAExtensionTests(unittest.TestCase):
    def test_extension_is_balanced_disjoint_and_provenance_frozen(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            root = Path(temp_dir)
            source = root / "strategyqa_train.json"
            excluded = root / "manifest.strategyqa_prior.json"
            raw_output = root / "strategyqa_extension.json"
            prepared_output = root / "strategyqa_extension.jsonl"

            records = [
                {
                    "qid": f"q{index}",
                    "question": f"Question {index}?",
                    "answer": index % 2 == 1,
                }
                for index in range(1, 9)
            ]
            source.write_text(
                json.dumps(records),
                encoding="utf-8",
            )
            excluded.write_text(
                json.dumps({"qids": ["q1", "q2"]}),
                encoding="utf-8",
            )

            manifest = freeze_extension(
                source=source,
                exclude_manifest=excluded,
                raw_output=raw_output,
                prepared_output=prepared_output,
                count=4,
                seed=20260921,
                overwrite=False,
            )

            selected = json.loads(raw_output.read_text(encoding="utf-8"))
            selected_qids = {record["qid"] for record in selected}
            prepared = [
                json.loads(line)
                for line in prepared_output.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]

            self.assertEqual(len(selected), 4)
            self.assertTrue(selected_qids.isdisjoint({"q1", "q2"}))
            self.assertEqual(
                manifest["answer_counts"],
                {"yes": 2, "no": 2},
            )
            self.assertEqual(
                [record["id"] for record in prepared],
                manifest["qids"],
            )
            self.assertEqual(len(manifest["source_sha256"]), 64)
            self.assertEqual(len(manifest["prepared_output_sha256"]), 64)

            with self.assertRaises(FileExistsError):
                freeze_extension(
                    source=source,
                    exclude_manifest=excluded,
                    raw_output=raw_output,
                    prepared_output=prepared_output,
                    count=4,
                    seed=20260921,
                    overwrite=False,
                )


if __name__ == "__main__":
    unittest.main()
