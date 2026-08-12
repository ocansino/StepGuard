import json
import tempfile
import unittest
from pathlib import Path

from src.io_utils import read_json, read_jsonl, write_json_atomic, write_jsonl


class AtomicIOTests(unittest.TestCase):
    def test_jsonl_write_replaces_target_and_removes_temporary_file(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            path = Path(temp_dir) / "artifact.jsonl"
            path.write_text('{"id":"old"}\n', encoding="utf-8")

            write_jsonl(
                path,
                [
                    {"id": "q1", "answer": "yes"},
                    {"id": "q2", "answer": "no"},
                ],
            )

            self.assertEqual(
                list(read_jsonl(path)),
                [
                    {"id": "q1", "answer": "yes"},
                    {"id": "q2", "answer": "no"},
                ],
            )
            self.assertFalse(
                path.with_suffix(path.suffix + ".tmp").exists()
            )

    def test_failed_jsonl_write_preserves_existing_target(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            path = Path(temp_dir) / "artifact.jsonl"
            original_text = '{"id":"existing"}\n'
            path.write_text(original_text, encoding="utf-8")

            invalid_records = [
                {"id": "q1"},
                {"id": "q2", "not_json": {"set-values"}},
            ]

            with self.assertRaises(TypeError):
                write_jsonl(path, invalid_records)

            self.assertEqual(
                path.read_text(encoding="utf-8"),
                original_text,
            )
            self.assertFalse(
                path.with_suffix(path.suffix + ".tmp").exists()
            )

    def test_json_write_round_trips_atomically(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            path = Path(temp_dir) / "checkpoint.json"
            payload = {
                "schema_version": 1,
                "next_iteration": 2,
                "active_state_indexes": [0, 2],
            }

            write_json_atomic(path, payload)

            self.assertEqual(read_json(path), payload)
            self.assertFalse(
                path.with_suffix(path.suffix + ".tmp").exists()
            )

    def test_failed_json_write_preserves_existing_target(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            path = Path(temp_dir) / "checkpoint.json"
            original_payload = {"schema_version": 1, "status": "safe"}
            write_json_atomic(path, original_payload)

            with self.assertRaises(TypeError):
                write_json_atomic(path, {"invalid": {"set-values"}})

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                original_payload,
            )
            self.assertFalse(
                path.with_suffix(path.suffix + ".tmp").exists()
            )


if __name__ == "__main__":
    unittest.main()
