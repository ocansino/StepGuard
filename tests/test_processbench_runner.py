import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.processbench_runner import (
    PREDICTION_SCHEMA_VERSION,
    VerifierJudgmentValidationError,
    aggregate_execution_metric_snapshots,
    archive_execution_metrics_for_resume,
    evaluate_blind_predictions,
    make_prediction_record,
    make_stepguard_score_batch,
    run_blind_detector,
    run_detector,
    validate_verifier_judgments,
    validate_runtime_config,
)
from scripts.freeze_processbench_protocol import GOLD_FIELDS, blind_record, label_record
from src.config import load_config
from src.config import RunConfig
from src.io_utils import write_jsonl
from src.io_utils import read_jsonl
from src.io_utils import write_json_atomic


SCORING = {
    "risk_formula": "weighted",
    "verifier_weight": 0.75,
    "contradiction_weight": 0.25,
}


def prepared_record(record_id, gold_label, steps=None):
    steps = steps or ["First step", "Second step"]
    return {
        "id": record_id,
        "question": "A test problem",
        "benchmark_steps": steps,
        "gold_label": gold_label,
        "gold_earliest_bad_step": None if gold_label == -1 else gold_label,
        "generator": "fixture-generator",
        "final_answer_correct": gold_label == -1,
        "benchmark": {
            "name": "ProcessBench",
            "revision": "fixture-revision",
            "split": "gsm8k",
        },
    }


class FakeJudgeClient:
    def __init__(self, probabilities):
        self.probabilities = list(probabilities)
        self.calls = []

    def judge_steps(self, *, question, steps, task, max_output_tokens):
        self.calls.append((question, list(steps), task, max_output_tokens))
        probabilities = self.probabilities.pop(0)
        return [
            {
                "step_index": index,
                "verdict": "wrong" if probability > 0.5 else "correct",
                "p_wrong": probability,
                "explanation": f"fixture explanation {index}",
            }
            for index, probability in enumerate(probabilities)
        ]


class FakeNLI:
    def __init__(self, probabilities):
        self.probabilities = list(probabilities)
        self.pairs = []

    def contradiction_probs(self, pairs):
        self.pairs.extend(pairs)
        result = self.probabilities[:len(pairs)]
        self.probabilities = self.probabilities[len(pairs):]
        return result


def fixed_score(record, earliest):
    steps = list(record["benchmark_steps"])
    judgments = [
        {
            "step_index": index,
            "verdict": "correct",
            "p_wrong": 0.1,
            "explanation": f"fixture explanation {index}",
        }
        for index in range(len(steps))
    ]
    return {
        "steps": steps,
        "scores": {
            "verifier": [0.1] * len(steps),
            "contradiction": [0.0] * len(steps),
            "evidence_support": None,
        },
        "risks": [0.075] * len(steps),
        "earliest_bad_step": earliest,
        "raw_verifier_judgments": judgments,
        "verifier_audit": {
            "expected_steps": len(steps),
            "received_judgments": len(judgments),
            "complete": True,
        },
        "risk_summary": {
            "avg_risk": 0.075,
            "max_risk": 0.075,
            "num_risky_steps": 0 if earliest is None else 1,
        },
    }


class ProcessBenchRunnerTests(unittest.TestCase):
    def test_frozen_openai_smoke_config_validates_offline(self):
        config = load_config(
            "configs/processbench_gsm8k_openai_gpt_5_4_mini_smoke5.yaml"
        )
        resolved = validate_runtime_config(config)

        self.assertEqual(resolved["provider"], "openai")
        self.assertEqual(resolved["model_name"], "gpt-5.4-mini-2026-03-17")
        self.assertEqual(resolved["records"], 5)
        self.assertEqual(resolved["chunk_size"], 1)
        self.assertEqual(resolved["risk_threshold"], 0.2)

    def test_protocol_v1_development_and_blind_configs_validate_offline(self):
        development = validate_runtime_config(
            load_config(
                "configs/processbench_gsm8k_openai_gpt_5_4_mini_development_100.yaml"
            )
        )
        heldout = validate_runtime_config(
            load_config(
                "configs/processbench_gsm8k_openai_gpt_5_4_mini_heldout_300.yaml"
            )
        )

        self.assertEqual(development["records"], 100)
        self.assertFalse(development["blinded"])
        self.assertEqual(heldout["records"], 300)
        self.assertTrue(heldout["blinded"])
        self.assertEqual(development["risk_threshold"], 0.2)
        self.assertEqual(heldout["risk_threshold"], 0.2)

    def test_provider_runtime_wiring_is_exercised_with_offline_fakes(self):
        records = [
            prepared_record("pb-1", 1),
            prepared_record("pb-2", -1),
        ]
        judge = FakeJudgeClient([[0.1, 0.9], [0.1, 0.1]])
        nli = FakeNLI([0.0, 0.0])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "smoke.jsonl"
            write_jsonl(dataset, records)
            config = RunConfig(
                run_name="offline-provider-wiring",
                dataset_path=str(dataset),
                output_dir=str(root / "runs"),
                raw={
                    "run_name": "offline-provider-wiring",
                    "dataset_path": str(dataset),
                    "output_dir": str(root / "runs"),
                    "task_profile": "math",
                    "model": {
                        "provider": "openai",
                        "name": "offline-model",
                    },
                    "execution": {
                        "metrics_enabled": True,
                        "max_concurrency": 1,
                        "processbench_chunk_size": 2,
                        "nli_batch_size": 32,
                        "metrics_flush_interval_events": 1,
                    },
                    "scoring": {
                        **SCORING,
                        "risk_threshold": 0.2,
                    },
                    "processbench": {
                        "expected_records": 2,
                        "verifier_max_output_tokens": 1200,
                    },
                },
            )

            with (
                patch(
                    "scripts.processbench_runner.load_config",
                    return_value=config,
                ),
                patch(
                    "scripts.processbench_runner.NLIScorer",
                    return_value=nli,
                ),
                patch(
                    "scripts.processbench_runner._make_judge_client",
                    return_value=judge,
                ),
            ):
                from scripts.processbench_runner import run_from_config

                metrics = run_from_config(root / "fixture.yaml")

            outdir = root / "runs" / "offline-provider-wiring"
            predictions = list(
                read_jsonl(outdir / "processbench_predictions.jsonl")
            )
            self.assertEqual(
                [row["predicted_label"] for row in predictions],
                [1, -1],
            )
            self.assertEqual(metrics["official_processbench"]["f1"], 1.0)
            self.assertTrue((outdir / "processbench_metrics.json").exists())
            self.assertTrue((outdir / "metrics.processbench_score.json").exists())
            self.assertTrue((outdir / "manifest.processbench_score.json").exists())

    def test_blind_scoring_keeps_gold_out_until_explicit_evaluation(self):
        labeled = [
            prepared_record("pb-1", 1),
            prepared_record("pb-2", -1),
        ]
        blinded = [blind_record(record) for record in labeled]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "blind_predictions.jsonl"
            summary = run_blind_detector(
                records=blinded,
                score_batch=lambda chunk: [
                    fixed_score(chunk[0], 1),
                    fixed_score(chunk[1], None),
                ],
                output_path=output,
                tau=0.2,
                scoring_cfg=SCORING,
                chunk_size=2,
            )
            predictions = list(read_jsonl(output))

        self.assertTrue(summary["blinded"])
        self.assertFalse(summary["metrics_computed"])
        self.assertTrue(all(not GOLD_FIELDS.intersection(row) for row in predictions))

        metrics, evaluated = evaluate_blind_predictions(
            predictions=predictions,
            labels=[label_record(record) for record in labeled],
        )
        self.assertEqual(metrics["official_processbench"]["f1"], 1.0)
        self.assertEqual([row["gold_label"] for row in evaluated], [1, -1])

    def test_blind_evaluation_rejects_id_order_mismatch(self):
        source = prepared_record("pb-1", -1)
        blind = blind_record(source)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.jsonl"
            run_blind_detector(
                records=[blind],
                score_batch=lambda chunk: [fixed_score(chunk[0], None)],
                output_path=output,
                tau=0.2,
                scoring_cfg=SCORING,
            )
            predictions = list(read_jsonl(output))

        mismatched = label_record({**source, "id": "pb-other"})
        with self.assertRaisesRegex(ValueError, "id mismatch"):
            evaluate_blind_predictions(
                predictions=predictions,
                labels=[mismatched],
            )

    def test_injected_runner_reuses_stepguard_scoring_and_raw_steps(self):
        records = [
            prepared_record("pb-1", 1),
            prepared_record("pb-2", -1),
        ]
        judge = FakeJudgeClient([[0.1, 0.9], [0.1, 0.1]])
        nli = FakeNLI([0.0, 0.0])
        scorer = make_stepguard_score_batch(
            judge_client=judge,
            nli=nli,
            tau=0.2,
            scoring_cfg=SCORING,
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.jsonl"
            metrics = run_detector(
                records=records,
                score_batch=scorer,
                output_path=output,
                tau=0.2,
                scoring_cfg=SCORING,
                chunk_size=2,
            )
            predictions = list(read_jsonl(output))

        self.assertEqual([row["predicted_label"] for row in predictions], [1, -1])
        self.assertEqual(judge.calls[0][1], records[0]["benchmark_steps"])
        self.assertEqual(judge.calls[0][2], "math")
        self.assertEqual(len(nli.pairs), 2)
        self.assertEqual(
            predictions[0]["prediction_schema_version"],
            PREDICTION_SCHEMA_VERSION,
        )
        self.assertEqual(
            predictions[0]["stepguard_score"]["raw_verifier_judgments"][1][
                "explanation"
            ],
            "fixture explanation 1",
        )
        self.assertTrue(
            predictions[0]["stepguard_score"]["verifier_audit"]["complete"]
        )
        self.assertEqual(
            metrics["official_processbench"]["f1"],
            1.0,
        )

    def test_failed_chunk_preserves_prior_chunks_and_can_resume(self):
        records = [
            prepared_record("pb-1", -1),
            prepared_record("pb-2", 0),
        ]
        calls = 0

        def fails_on_second_chunk(chunk):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("offline fixture failure")
            return [fixed_score(chunk[0], None)]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.jsonl"
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                run_detector(
                    records=records,
                    score_batch=fails_on_second_chunk,
                    output_path=output,
                    tau=0.2,
                    scoring_cfg=SCORING,
                    chunk_size=1,
                )
            self.assertEqual(len(list(read_jsonl(output))), 1)

            def resumed_scorer(chunk):
                return [fixed_score(chunk[0], 0)]

            metrics = run_detector(
                records=records,
                score_batch=resumed_scorer,
                output_path=output,
                tau=0.2,
                scoring_cfg=SCORING,
                chunk_size=1,
                resume=True,
            )
            predictions = list(read_jsonl(output))

        self.assertEqual([row["predicted_label"] for row in predictions], [-1, 0])
        self.assertEqual(metrics["official_processbench"]["f1"], 1.0)

    def test_resume_rejects_changed_detector_configuration(self):
        record = prepared_record("pb-1", -1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.jsonl"
            run_detector(
                records=[record],
                score_batch=lambda chunk: [fixed_score(chunk[0], None)],
                output_path=output,
                tau=0.2,
                scoring_cfg=SCORING,
            )
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                run_detector(
                    records=[record],
                    score_batch=lambda chunk: [fixed_score(chunk[0], None)],
                    output_path=output,
                    tau=0.3,
                    scoring_cfg=SCORING,
                    resume=True,
                )

    def test_existing_output_is_not_overwritten_without_resume(self):
        record = prepared_record("pb-1", -1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.jsonl"
            output.write_text("existing\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_detector(
                    records=[record],
                    score_batch=lambda chunk: [fixed_score(chunk[0], None)],
                    output_path=output,
                    tau=0.2,
                    scoring_cfg=SCORING,
                )
            self.assertEqual(output.read_text(encoding="utf-8"), "existing\n")

    def test_prediction_rejects_out_of_range_localization(self):
        record = prepared_record("pb-1", -1)
        score = fixed_score(record, 2)
        with self.assertRaisesRegex(ValueError, "invalid earliest_bad_step"):
            make_prediction_record(
                source=record,
                score=score,
                identity={"risk_threshold": 0.2},
            )

    def test_score_step_text_must_match_benchmark(self):
        record = prepared_record("pb-1", -1)
        score = fixed_score(record, None)
        score["steps"] = ["changed", "steps"]
        with self.assertRaisesRegex(ValueError, "differ"):
            make_prediction_record(
                source=record,
                score=score,
                identity={"risk_threshold": 0.2},
            )

    def test_incomplete_verifier_response_fails_before_scoring(self):
        record = prepared_record("pb-1", -1)
        judge = FakeJudgeClient([[0.1]])
        nli = FakeNLI([0.0])
        scorer = make_stepguard_score_batch(
            judge_client=judge,
            nli=nli,
            tau=0.2,
            scoring_cfg=SCORING,
            verifier_schema_max_retries=0,
        )

        with self.assertRaisesRegex(ValueError, "1 judgments for 2 steps"):
            scorer([record])
        self.assertEqual(nli.pairs, [])

    def test_verifier_schema_mismatch_retries_with_strict_contract(self):
        record = prepared_record("pb-1", -1)
        judge = FakeJudgeClient(
            [
                [0.1, 0.2, 0.3],
                [0.1, 0.9],
            ]
        )
        nli = FakeNLI([0.0, 0.0])
        scorer = make_stepguard_score_batch(
            judge_client=judge,
            nli=nli,
            tau=0.2,
            scoring_cfg=SCORING,
            verifier_schema_max_retries=2,
        )

        score = scorer([record])[0]
        audit = score["verifier_audit"]

        self.assertEqual(len(judge.calls), 2)
        self.assertEqual(judge.calls[0][0], record["question"])
        self.assertIn("exactly 2 result objects", judge.calls[1][0])
        self.assertEqual(judge.calls[1][1], record["benchmark_steps"])
        self.assertEqual(audit["schema_attempts"], 2)
        self.assertEqual(audit["schema_retries"], 1)
        self.assertEqual(audit["schema_retry_contract"], "expected_cardinality_v1")
        self.assertIn("3 judgments for 2 steps", audit["rejected_schema_errors"][0])
        self.assertEqual(score["scores"]["verifier"], [0.1, 0.9])

    def test_verifier_schema_retry_exhaustion_remains_a_hard_failure(self):
        record = prepared_record("pb-1", -1)
        judge = FakeJudgeClient(
            [
                [0.1, 0.2, 0.3],
                [0.1, 0.2, 0.3],
                [0.1, 0.2, 0.3],
            ]
        )
        nli = FakeNLI([])
        scorer = make_stepguard_score_batch(
            judge_client=judge,
            nli=nli,
            tau=0.2,
            scoring_cfg=SCORING,
            verifier_schema_max_retries=2,
        )

        with self.assertRaisesRegex(
            VerifierJudgmentValidationError,
            "remained invalid after 3 attempts",
        ):
            scorer([record])

        self.assertEqual(len(judge.calls), 3)
        self.assertEqual(nli.pairs, [])

    def test_execution_metric_sessions_are_archived_and_aggregated(self):
        first = {
            "schema_version": 1,
            "run_name": "fixture-run",
            "command": "processbench-score",
            "started_at_utc": "first-start",
            "completed_at_utc": "first-end",
            "wall_time_seconds": 10.0,
            "records_processed": 5,
            "execution_config": {},
            "context": {},
            "totals": {"logical_calls": 6, "total_tokens": 100},
            "wait_totals": {"events": 0, "elapsed_seconds": 0.0},
            "waits": {},
            "operations": {
                "openai.step_verification": {
                    "provider": "openai",
                    "operation": "step_verification",
                    "logical_calls": 6,
                    "attempts": 6,
                    "successes": 6,
                    "failures": 0,
                    "total_tokens": 100,
                    "error_types": {},
                }
            },
            "stages": {
                "scoring": {
                    "name": "scoring",
                    "calls": 1,
                    "successes": 0,
                    "failures": 1,
                    "elapsed_seconds": 10.0,
                    "error_types": {"ValueError": 1},
                }
            },
            "stage_events": [{"name": "scoring", "success": False}],
        }
        second = {
            **first,
            "started_at_utc": "second-start",
            "completed_at_utc": "second-end",
            "wall_time_seconds": 20.0,
            "records_processed": 10,
            "totals": {"logical_calls": 5, "total_tokens": 80},
            "operations": {
                "openai.step_verification": {
                    "provider": "openai",
                    "operation": "step_verification",
                    "logical_calls": 5,
                    "attempts": 5,
                    "successes": 5,
                    "failures": 0,
                    "total_tokens": 80,
                    "error_types": {},
                }
            },
            "stages": {
                "scoring": {
                    "name": "scoring",
                    "calls": 1,
                    "successes": 1,
                    "failures": 0,
                    "elapsed_seconds": 20.0,
                    "error_types": {},
                }
            },
            "stage_events": [{"name": "scoring", "success": True}],
        }

        aggregate = aggregate_execution_metric_snapshots([first, second])

        self.assertEqual(aggregate["sessions"], 2)
        self.assertEqual(aggregate["records_processed"], 10)
        self.assertEqual(aggregate["wall_time_seconds"], 30.0)
        self.assertEqual(aggregate["totals"]["logical_calls"], 11)
        self.assertEqual(aggregate["totals"]["total_tokens"], 180)
        operation = aggregate["operations"]["openai.step_verification"]
        self.assertEqual(operation["logical_calls"], 11)
        self.assertEqual(operation["total_tokens"], 180)
        self.assertEqual(aggregate["stages"]["scoring"]["failures"], 1)
        self.assertEqual(aggregate["stage_events"][1]["session_index"], 2)

        with tempfile.TemporaryDirectory() as directory:
            metrics_path = Path(directory) / "metrics.processbench_score.json"
            write_json_atomic(metrics_path, first)
            archived = archive_execution_metrics_for_resume(metrics_path)
            self.assertIsNotNone(archived)
            self.assertTrue(archived.exists())
            self.assertEqual(archived.name, "metrics.processbench_score.session_001.json")

    def test_duplicate_verifier_step_index_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate verifier step_index"):
            validate_verifier_judgments(
                [
                    {"step_index": 0, "p_wrong": 0.1},
                    {"step_index": 0, "p_wrong": 0.2},
                ],
                step_count=2,
                record_id="pb-1",
            )

    def test_invalid_verifier_probability_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid p_wrong"):
            validate_verifier_judgments(
                [
                    {"step_index": 0, "p_wrong": float("nan")},
                    {"step_index": 1, "p_wrong": 0.2},
                ],
                step_count=2,
                record_id="pb-1",
            )


if __name__ == "__main__":
    unittest.main()
