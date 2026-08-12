import tempfile
import unittest
from pathlib import Path

from src.generation_checkpoint import (
    CHECKPOINT_KIND,
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointCompatibilityError,
    CheckpointValidationError,
    delete_generation_checkpoint,
    load_generation_checkpoint,
    make_generation_checkpoint_identity,
    save_generation_checkpoint,
)
from src.io_utils import write_json_atomic


def sample_records():
    return [
        {
            "id": "q1",
            "question": "Question one?",
            "model_trace": (
                "Step 1: Offline reasoning.\nFinal answer: yes"
            ),
            "model_answer": "yes",
            "task": "strategyqa",
        },
        {
            "id": "q2",
            "question": "Question two?",
            "model_trace": (
                "Step 1: More offline reasoning.\nFinal answer: no"
            ),
            "model_answer": "no",
            "task": "strategyqa",
        },
    ]


class GenerationCheckpointTests(unittest.TestCase):
    def make_fixture(self, temp_path):
        input_path = temp_path / "dataset.jsonl"
        input_path.write_text(
            (
                '{"id":"q1","question":"Question one?"}\n'
                '{"id":"q2","question":"Question two?"}\n'
            ),
            encoding="utf-8",
        )
        config = {
            "run_name": "generation-checkpoint-test",
            "dataset_path": str(input_path),
            "output_dir": "runs",
            "task_profile": "strategyqa",
            "model": {
                "provider": "openai",
                "name": "fake-model",
            },
            "execution": {
                "generation_chunk_size": 25,
                "max_concurrency": 4,
            },
        }
        identity = make_generation_checkpoint_identity(
            run_name="generation-checkpoint-test",
            input_path=input_path,
            config_raw=config,
        )
        return input_path, config, identity

    def test_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint_path = temp_path / "generated.checkpoint.json"
            _, _, identity = self.make_fixture(temp_path)

            save_generation_checkpoint(
                checkpoint_path,
                identity=identity,
                next_source_index=2,
                records=sample_records(),
            )

            payload = load_generation_checkpoint(
                checkpoint_path,
                expected_identity=identity,
                expected_record_count=2,
            )

            self.assertEqual(payload["kind"], CHECKPOINT_KIND)
            self.assertEqual(
                payload["schema_version"],
                CHECKPOINT_SCHEMA_VERSION,
            )
            self.assertEqual(payload["next_source_index"], 2)
            self.assertEqual(payload["records"], sample_records())
            self.assertTrue(payload["saved_at_utc"])
            self.assertFalse(
                checkpoint_path.with_suffix(
                    checkpoint_path.suffix + ".tmp"
                ).exists()
            )

    def test_changed_input_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint_path = temp_path / "generated.checkpoint.json"
            input_path, config, identity = self.make_fixture(temp_path)

            save_generation_checkpoint(
                checkpoint_path,
                identity=identity,
                next_source_index=2,
                records=sample_records(),
            )

            input_path.write_text(
                '{"id":"q1","question":"Changed?"}\n',
                encoding="utf-8",
            )
            changed_identity = make_generation_checkpoint_identity(
                run_name="generation-checkpoint-test",
                input_path=input_path,
                config_raw=config,
            )

            with self.assertRaisesRegex(
                CheckpointCompatibilityError,
                "input_sha256",
            ):
                load_generation_checkpoint(
                    checkpoint_path,
                    expected_identity=changed_identity,
                    expected_record_count=2,
                )

    def test_changed_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint_path = temp_path / "generated.checkpoint.json"
            input_path, config, identity = self.make_fixture(temp_path)

            save_generation_checkpoint(
                checkpoint_path,
                identity=identity,
                next_source_index=2,
                records=sample_records(),
            )

            changed_config = {
                **config,
                "model": {
                    **config["model"],
                    "name": "different-model",
                },
            }
            changed_identity = make_generation_checkpoint_identity(
                run_name="generation-checkpoint-test",
                input_path=input_path,
                config_raw=changed_config,
            )

            with self.assertRaisesRegex(
                CheckpointCompatibilityError,
                "config_sha256",
            ):
                load_generation_checkpoint(
                    checkpoint_path,
                    expected_identity=changed_identity,
                    expected_record_count=2,
                )

    def test_record_count_must_match_next_source_index(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint_path = temp_path / "generated.checkpoint.json"
            _, _, identity = self.make_fixture(temp_path)

            write_json_atomic(
                checkpoint_path,
                {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "kind": CHECKPOINT_KIND,
                    "saved_at_utc": "2026-08-10T00:00:00+00:00",
                    "identity": identity,
                    "next_source_index": 2,
                    "records": sample_records()[:1],
                },
            )

            with self.assertRaisesRegex(
                CheckpointValidationError,
                "must equal",
            ):
                load_generation_checkpoint(
                    checkpoint_path,
                    expected_identity=identity,
                    expected_record_count=2,
                )

    def test_next_source_index_cannot_exceed_dataset(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint_path = temp_path / "generated.checkpoint.json"
            _, _, identity = self.make_fixture(temp_path)

            save_generation_checkpoint(
                checkpoint_path,
                identity=identity,
                next_source_index=2,
                records=sample_records(),
            )

            with self.assertRaisesRegex(
                CheckpointValidationError,
                "exceeds",
            ):
                load_generation_checkpoint(
                    checkpoint_path,
                    expected_identity=identity,
                    expected_record_count=1,
                )

    def test_checkpoint_delete_is_idempotent(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint_path = temp_path / "generated.checkpoint.json"
            _, _, identity = self.make_fixture(temp_path)

            save_generation_checkpoint(
                checkpoint_path,
                identity=identity,
                next_source_index=2,
                records=sample_records(),
            )

            delete_generation_checkpoint(checkpoint_path)
            delete_generation_checkpoint(checkpoint_path)
            self.assertFalse(checkpoint_path.exists())


if __name__ == "__main__":
    unittest.main()
