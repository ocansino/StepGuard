import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from src.cli import build_candidate_pool


class FakeWaveClient:
    def __init__(
        self,
        *,
        model,
        metrics,
        barriers=None,
        call_log=None,
    ) -> None:
        self.model = model
        self.metrics = metrics
        self.barriers = barriers or {}
        self.call_log = call_log

    def _record_call(self, operation: str) -> None:
        if self.call_log is not None:
            self.call_log.append(operation)

    def _synchronize(self, operation: str) -> None:
        barrier = self.barriers.get(operation)
        if barrier is not None:
            barrier.wait(timeout=3.0)

    def judge_steps(
        self,
        question,
        steps,
        *,
        task="math",
        max_output_tokens=1200,
    ):
        self._record_call("step_verification")
        self._synchronize("step_verification")
        repaired = any("Repaired reasoning" in step for step in steps)
        p_wrong = 0.05 if repaired else 0.80
        self.metrics.record_operation(
            provider="openai",
            operation="step_verification",
            elapsed_seconds=0.01,
            success=True,
            usage={
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        )
        return [
            {
                "step_index": index,
                "verdict": "valid" if repaired else "invalid",
                "p_wrong": p_wrong,
            }
            for index, _ in enumerate(steps)
        ]

    def repair_suffix(
        self,
        question,
        prefix_steps,
        next_step_number,
        *,
        task="math",
        temperature=0.2,
        max_output_tokens=800,
    ):
        self._record_call("suffix_repair")
        self._synchronize("suffix_repair")
        self.metrics.record_operation(
            provider="openai",
            operation="suffix_repair",
            elapsed_seconds=0.01,
            success=True,
            usage={
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        )
        return (
            f"Step {next_step_number}: Repaired reasoning for {question}\n"
            "Final answer: yes",
            "yes",
        )

    def judge_repair_candidate(
        self,
        *,
        question,
        original_trace,
        original_answer,
        repaired_trace,
        repaired_answer,
        task="math",
        max_output_tokens=800,
    ):
        self._record_call("acceptance_judging")
        self._synchronize("acceptance_judging")
        self.metrics.record_operation(
            provider="openai",
            operation="acceptance_judging",
            elapsed_seconds=0.01,
            success=True,
            usage={
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        )
        return {
            "prefer_repaired": True,
            "original_answer_support": 0.2,
            "repaired_answer_support": 0.9,
            "regression_risk": 0.05,
            "confidence": 0.9,
            "reason": "Deterministic test judgment.",
        }


class FakeNLIScorer:
    def __init__(
        self,
        *,
        model_name,
        metrics=None,
        batch_size=32,
    ) -> None:
        self.model_name = model_name
        self.metrics = metrics
        self.batch_size = batch_size

    def contradiction_prob(self, premise, hypothesis):
        if self.metrics is not None:
            self.metrics.record_operation(
                provider="local_roberta",
                operation="contradiction_scoring",
                elapsed_seconds=0.001,
                success=True,
            )
        return 0.0

    def contradiction_probs(self, pairs, *, batch_size=None):
        return [
            self.contradiction_prob(premise, hypothesis)
            for premise, hypothesis in pairs
        ]


class WaveCandidatePoolTests(unittest.TestCase):
    def test_wave_concurrency_matches_deterministic_sequential_pool(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = temp_path / "generated.jsonl"
            output_dir = temp_path / "runs"

            records = [
                {
                    "id": f"q{index}",
                    "question": f"Question {index}?",
                    "model_trace": (
                        f"Step 1: Original reasoning for question {index}.\n"
                        "Final answer: no"
                    ),
                    "model_answer": "no",
                    "gold_answer": "yes",
                    "task": "strategyqa",
                    "source": "offline-test",
                }
                for index in range(1, 4)
            ]
            dataset_path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            def write_config(run_name: str, max_concurrency: int) -> Path:
                config_path = temp_path / f"{run_name}.yaml"
                config = {
                    "run_name": run_name,
                    "dataset_path": str(dataset_path),
                    "output_dir": str(output_dir),
                    "task_profile": "strategyqa",
                    "model": {
                        "provider": "openai",
                        "name": "fake-model",
                        "temperature": 0.2,
                        "max_output_tokens": 100,
                    },
                    "execution": {
                        "metrics_enabled": True,
                        "max_concurrency": max_concurrency,
                        "request_timeout_seconds": 45.0,
                        "max_retries": 1,
                        "retry_base_delay_seconds": 0.5,
                        "retry_jitter_fraction": 0.15,
                        "openai_requests_per_minute": 120.0,
                        "rate_limit_headroom_fraction": 0.75,
                    },
                    "scoring": {
                        "risk_threshold": 0.2,
                        "improvement_threshold": 0.02,
                        "max_iters": 1,
                        "risk_formula": "weighted",
                        "verifier_weight": 0.75,
                        "contradiction_weight": 0.25,
                    },
                    "repair_acceptance": {
                        "mode": "risk_only",
                        "support_tolerance": 0.05,
                        "max_regression_risk": 0.35,
                    },
                }
                config_path.write_text(
                    yaml.safe_dump(config),
                    encoding="utf-8",
                )
                return config_path

            sequential_config = write_config("wave-sequential", 1)
            concurrent_config = write_config("wave-concurrent", 3)

            def run_pool(config_path, barriers=None):
                client_options = []

                def make_client(*, model, metrics, **execution_options):
                    client_options.append(execution_options)
                    return FakeWaveClient(
                        model=model,
                        metrics=metrics,
                        barriers=barriers,
                    )

                with (
                    patch(
                        "src.cli.OpenAIClientWrapper",
                        side_effect=make_client,
                    ),
                    patch("src.cli.NLIScorer", FakeNLIScorer),
                ):
                    build_candidate_pool(
                        config=str(config_path),
                        input_path=str(dataset_path),
                        output_path=None,
                        reuse_original_scores=False,
                        resume=False,
                    )

                return client_options

            sequential_client_options = run_pool(sequential_config)

            barriers = {
                "step_verification": threading.Barrier(3),
                "suffix_repair": threading.Barrier(3),
                "acceptance_judging": threading.Barrier(3),
            }
            concurrent_client_options = run_pool(
                concurrent_config,
                barriers=barriers,
            )

            sequential_dir = output_dir / "wave-sequential"
            concurrent_dir = output_dir / "wave-concurrent"

            def read_jsonl(path):
                return [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                ]

            sequential_records = read_jsonl(
                sequential_dir / "candidate_pool.jsonl"
            )
            concurrent_records = read_jsonl(
                concurrent_dir / "candidate_pool.jsonl"
            )

            self.assertEqual(concurrent_records, sequential_records)
            self.assertEqual(
                [record["id"] for record in concurrent_records],
                ["q1", "q2", "q3"],
            )
            self.assertTrue(
                all(
                    len(record["candidate_pool"]["proposals"]) == 1
                    for record in concurrent_records
                )
            )

            metrics = json.loads(
                (
                    concurrent_dir
                    / "metrics.build_candidate_pool.json"
                ).read_text(encoding="utf-8")
            )
            operations = metrics["operations"]
            expected_client_options = {
                "request_timeout_seconds": 45.0,
                "max_retries": 1,
                "retry_base_delay_seconds": 0.5,
                "retry_jitter_fraction": 0.15,
            }
            self.assertTrue(sequential_client_options)
            self.assertTrue(concurrent_client_options)

            sequential_limiters = [
                options["rate_limiter"]
                for options in sequential_client_options
            ]
            concurrent_limiters = [
                options["rate_limiter"]
                for options in concurrent_client_options
            ]
            self.assertTrue(
                all(
                    limiter is sequential_limiters[0]
                    for limiter in sequential_limiters
                )
            )
            self.assertTrue(
                all(
                    limiter is concurrent_limiters[0]
                    for limiter in concurrent_limiters
                )
            )
            self.assertIsNot(
                sequential_limiters[0],
                concurrent_limiters[0],
            )
            self.assertEqual(
                concurrent_limiters[0].requests_per_minute,
                120.0,
            )
            self.assertEqual(
                concurrent_limiters[0].headroom_fraction,
                0.75,
            )
            self.assertTrue(
                all(
                    {
                        key: value
                        for key, value in options.items()
                        if key != "rate_limiter"
                    }
                    == expected_client_options
                    for options in (
                        sequential_client_options
                        + concurrent_client_options
                    )
                )
            )
            self.assertEqual(
                operations["openai.step_verification"]["logical_calls"],
                6,
            )
            self.assertEqual(
                operations["openai.suffix_repair"]["logical_calls"],
                3,
            )
            self.assertEqual(
                operations["openai.acceptance_judging"]["logical_calls"],
                3,
            )
            self.assertEqual(
                set(metrics["stages"]),
                {
                    "dataset_loading",
                    "nli_initialization",
                    "original_verification_wave",
                    "repair_wave",
                    "candidate_verification_wave",
                    "acceptance_wave",
                    "artifact_write",
                    "checkpoint_write",
                    "checkpoint_cleanup",
                },
            )
            repair_event = next(
                event
                for event in metrics["stage_events"]
                if event["name"] == "repair_wave"
            )
            self.assertEqual(
                repair_event["metadata"],
                {
                    "iteration": 1,
                    "items": 3,
                    "max_concurrency": 3,
                },
            )


if __name__ == "__main__":
    unittest.main()
