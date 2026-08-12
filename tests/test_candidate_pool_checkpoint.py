import tempfile
import unittest
from pathlib import Path

from src.candidate_pool_checkpoint import (
    CHECKPOINT_KIND,
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointCompatibilityError,
    CheckpointValidationError,
    delete_candidate_pool_checkpoint,
    load_candidate_pool_checkpoint,
    make_candidate_pool_checkpoint_identity,
    save_candidate_pool_checkpoint,
)
from src.io_utils import write_json_atomic


def sample_states():
    score = {
        "steps": ["Step 1: Offline reasoning."],
        "scores": {
            "verifier": [0.1],
            "contradiction": [0.0],
        },
        "risks": [0.075],
        "earliest_bad_step": None,
        "risk_summary": {"avg_risk": 0.075},
    }
    return [
        {
            "state_index": 0,
            "record": {"id": "q1", "question": "Question one?"},
            "task": "strategyqa",
            "current_trace": "Step 1: Offline reasoning.\nFinal answer: yes",
            "current_answer": "yes",
            "current_score": score,
            "original_score": score,
            "original_score_source": "newly_scored",
            "proposals": [],
            "generation_errors": [],
            "stop_reason": None,
            "scheduled_iteration": 1,
        }
    ]


class CandidatePoolCheckpointTests(unittest.TestCase):
    def make_fixture(self, temp_path):
        input_path = temp_path / "generated.jsonl"
        input_path.write_text(
            '{"id":"q1","question":"Question one?"}\n',
            encoding="utf-8",
        )
        config = {
            "run_name": "checkpoint-test",
            "dataset_path": "dataset.jsonl",
            "output_dir": "runs",
            "task_profile": "strategyqa",
            "model": {"provider": "openai", "name": "fake-model"},
            "scoring": {"max_iters": 2, "risk_threshold": 0.2},
        }
        identity = make_candidate_pool_checkpoint_identity(
            run_name="checkpoint-test",
            input_path=input_path,
            config_raw=config,
            reuse_original_scores=False,
        )
        return input_path, config, identity

    def test_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint_path = temp_path / "candidate_pool.checkpoint.json"
            _, _, identity = self.make_fixture(temp_path)

            save_candidate_pool_checkpoint(
                checkpoint_path,
                identity=identity,
                phase="original_scoring_complete",
                next_iteration=1,
                active_state_indexes=[0],
                states=sample_states(),
            )

            payload = load_candidate_pool_checkpoint(
                checkpoint_path,
                expected_identity=identity,
            )

            self.assertEqual(payload["kind"], CHECKPOINT_KIND)
            self.assertEqual(
                payload["schema_version"],
                CHECKPOINT_SCHEMA_VERSION,
            )
            self.assertEqual(payload["phase"], "original_scoring_complete")
            self.assertEqual(payload["next_iteration"], 1)
            self.assertEqual(payload["active_state_indexes"], [0])
            self.assertEqual(payload["states"], sample_states())
            self.assertTrue(payload["saved_at_utc"])
            self.assertFalse(
                checkpoint_path.with_suffix(
                    checkpoint_path.suffix + ".tmp"
                ).exists()
            )

    def test_changed_input_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint_path = temp_path / "candidate_pool.checkpoint.json"
            input_path, config, identity = self.make_fixture(temp_path)

            save_candidate_pool_checkpoint(
                checkpoint_path,
                identity=identity,
                phase="original_scoring_complete",
                next_iteration=1,
                active_state_indexes=[0],
                states=sample_states(),
            )

            input_path.write_text(
                '{"id":"q1","question":"Changed question?"}\n',
                encoding="utf-8",
            )
            changed_identity = make_candidate_pool_checkpoint_identity(
                run_name="checkpoint-test",
                input_path=input_path,
                config_raw=config,
                reuse_original_scores=False,
            )

            with self.assertRaisesRegex(
                CheckpointCompatibilityError,
                "input_sha256",
            ):
                load_candidate_pool_checkpoint(
                    checkpoint_path,
                    expected_identity=changed_identity,
                )

    def test_changed_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint_path = temp_path / "candidate_pool.checkpoint.json"
            input_path, config, identity = self.make_fixture(temp_path)

            save_candidate_pool_checkpoint(
                checkpoint_path,
                identity=identity,
                phase="iteration_complete",
                next_iteration=2,
                active_state_indexes=[],
                states=sample_states(),
            )

            changed_config = {
                **config,
                "scoring": {
                    **config["scoring"],
                    "risk_threshold": 0.9,
                },
            }
            changed_identity = make_candidate_pool_checkpoint_identity(
                run_name="checkpoint-test",
                input_path=input_path,
                config_raw=changed_config,
                reuse_original_scores=False,
            )

            with self.assertRaisesRegex(
                CheckpointCompatibilityError,
                "config_sha256",
            ):
                load_candidate_pool_checkpoint(
                    checkpoint_path,
                    expected_identity=changed_identity,
                )

    def test_invalid_active_state_index_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint_path = temp_path / "candidate_pool.checkpoint.json"
            _, _, identity = self.make_fixture(temp_path)

            write_json_atomic(
                checkpoint_path,
                {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "kind": CHECKPOINT_KIND,
                    "saved_at_utc": "2026-08-03T00:00:00+00:00",
                    "identity": identity,
                    "phase": "iteration_complete",
                    "next_iteration": 2,
                    "active_state_indexes": [9],
                    "states": sample_states(),
                },
            )

            with self.assertRaisesRegex(
                CheckpointValidationError,
                "active_state_indexes",
            ):
                load_candidate_pool_checkpoint(
                    checkpoint_path,
                    expected_identity=identity,
                )

    def test_checkpoint_delete_is_idempotent(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint_path = temp_path / "candidate_pool.checkpoint.json"
            _, _, identity = self.make_fixture(temp_path)

            save_candidate_pool_checkpoint(
                checkpoint_path,
                identity=identity,
                phase="iteration_complete",
                next_iteration=2,
                active_state_indexes=[],
                states=sample_states(),
            )

            delete_candidate_pool_checkpoint(checkpoint_path)
            delete_candidate_pool_checkpoint(checkpoint_path)
            self.assertFalse(checkpoint_path.exists())

    def test_iteration_chunk_phase_round_trips(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint_path = temp_path / "candidate_pool.checkpoint.json"
            _, _, identity = self.make_fixture(temp_path)

            save_candidate_pool_checkpoint(
                checkpoint_path,
                identity=identity,
                phase="iteration_chunk_complete",
                next_iteration=1,
                active_state_indexes=[0],
                states=sample_states(),
            )

            payload = load_candidate_pool_checkpoint(
                checkpoint_path,
                expected_identity=identity,
            )

            self.assertEqual(
                payload["phase"],
                "iteration_chunk_complete",
            )
            self.assertEqual(
                payload["states"][0]["scheduled_iteration"],
                1,
            )

    def test_invalid_scheduled_iteration_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint_path = temp_path / "candidate_pool.checkpoint.json"
            _, _, identity = self.make_fixture(temp_path)
            states = sample_states()
            states[0]["scheduled_iteration"] = 0

            write_json_atomic(
                checkpoint_path,
                {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "kind": CHECKPOINT_KIND,
                    "saved_at_utc": "2026-08-10T00:00:00+00:00",
                    "identity": identity,
                    "phase": "iteration_chunk_complete",
                    "next_iteration": 1,
                    "active_state_indexes": [0],
                    "states": states,
                },
            )

            with self.assertRaisesRegex(
                CheckpointValidationError,
                "scheduled_iteration",
            ):
                load_candidate_pool_checkpoint(
                    checkpoint_path,
                    expected_identity=identity,
                )


if __name__ == "__main__":
    unittest.main()
