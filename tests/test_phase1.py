import json
import tempfile
import unittest
from pathlib import Path

from src.datasets_prep import prepare_strategyqa_local
from src.parsing import split_steps, validate_score_alignment
from src.task_profiles import (
    answers_match,
    get_task_profile,
    normalize_math_answer,
    normalize_strategyqa_answer,
)


class StrategyQALoaderTests(unittest.TestCase):
    def test_local_loader_writes_standard_jsonl_and_manifest(self) -> None:
        examples = [
            {
                "qid": "q1",
                "question": "Is water wet?",
                "answer": True,
                "decomposition": ["What does wet mean?"],
                "evidence": [[{"title": "Water", "content": "Example"}]],
            },
            {
                "qid": "q2",
                "question": "Is the Moon a planet?",
                "answer": False,
                "facts": ["The Moon is a natural satellite."],
            },
        ]

        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "strategyqa_subset.json"
            output_path = temp_path / "strategyqa_subset.jsonl"
            input_path.write_text(json.dumps(examples), encoding="utf-8")

            count = prepare_strategyqa_local(
                input_path,
                output_path,
                source_path=input_path,
                selection_seed=42,
            )

            self.assertEqual(count, 2)
            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["id"] for row in rows], ["q1", "q2"])
            self.assertEqual([row["gold_answer"] for row in rows], ["yes", "no"])
            self.assertTrue(all(row["task"] == "strategyqa" for row in rows))
            self.assertNotIn("evidence", rows[0])
            self.assertIn("evidence", rows[0]["metadata"])

            manifest = json.loads(
                (temp_path / "manifest.strategyqa_subset.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["num_records"], 2)
            self.assertEqual(manifest["selection_seed"], 42)
            self.assertEqual(manifest["answer_counts"], {"yes": 1, "no": 1})
            self.assertEqual(manifest["qids"], ["q1", "q2"])
            self.assertEqual(len(manifest["source_sha256"]), 64)

    def test_loader_rejects_duplicate_qids(self) -> None:
        examples = [
            {"qid": "duplicate", "question": "First?", "answer": True},
            {"qid": "duplicate", "question": "Second?", "answer": False},
        ]
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "duplicate.json"
            input_path.write_text(json.dumps(examples), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate StrategyQA qid"):
                prepare_strategyqa_local(input_path, temp_path / "output.jsonl")

    def test_loader_accepts_utf8_bom(self) -> None:
        examples = [{"qid": "bom", "question": "Works?", "answer": True}]
        with tempfile.TemporaryDirectory(dir=Path("tests")) as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "bom.json"
            input_path.write_text(json.dumps(examples), encoding="utf-8-sig")
            output_path = temp_path / "bom.jsonl"
            self.assertEqual(prepare_strategyqa_local(input_path, output_path), 1)


class TaskProfileTests(unittest.TestCase):
    def test_math_normalization_preserves_existing_behavior(self) -> None:
        self.assertEqual(normalize_math_answer("The answer is $1,018.0"), "1018")
        self.assertEqual(normalize_math_answer("3.2500"), "3.25")
        self.assertTrue(answers_match("math", "18 dollars", "18"))

    def test_strategyqa_normalization_is_strict(self) -> None:
        for value in ("yes", "YES.", "true", "Final answer: **yes**"):
            self.assertEqual(normalize_strategyqa_answer(value), "yes")
        for value in ("no", "No!", "false", "Final answer: `no`"):
            self.assertEqual(normalize_strategyqa_answer(value), "no")
        self.assertIsNone(normalize_strategyqa_answer("probably yes"))
        self.assertIsNone(normalize_strategyqa_answer("yes or no"))

    def test_unknown_profile_fails_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown task profile"):
            get_task_profile("nonmath")

    def test_strategyqa_prompts_are_domain_specific(self) -> None:
        profile = get_task_profile("strategyqa")
        generation = profile.build_generation_prompt("Can penguins fly?")
        verifier = profile.build_verifier_prompt(
            "Can penguins fly?",
            ["Step 1: Penguins are flightless birds."],
        )
        repair = profile.build_repair_prompt(
            "Can penguins fly?",
            ["Step 1: Penguins are birds."],
            2,
        )
        acceptance = profile.build_acceptance_prompt(
            question="Can penguins fly?",
            original_trace="Step 1: No.",
            original_answer="no",
            repaired_trace="Step 1: Yes.",
            repaired_answer="yes",
        )

        self.assertIn("closed-book yes/no", generation)
        self.assertNotIn("math problem", generation.lower())
        self.assertIn("unsupported assumptions", verifier)
        self.assertIn("RETAINED LOW-RISK PREFIX", repair)
        self.assertIn("Final answer: no", repair)
        self.assertIn("unsupported factual assumptions", acceptance)
        self.assertNotIn("same math problem", acceptance.lower())

    def test_math_prompt_still_uses_existing_instruction(self) -> None:
        prompt = get_task_profile("math").build_generation_prompt("What is 2 + 2?")
        self.assertTrue(prompt.startswith("You are solving a problem."))
        self.assertIn("Final answer: <answer>", prompt)
        self.assertTrue(prompt.endswith("Problem:\nWhat is 2 + 2?"))


class ParsingTests(unittest.TestCase):
    def test_multiline_steps_are_preserved_and_final_answer_is_excluded(self) -> None:
        trace = """Step 1: First claim.
Supporting detail for the first claim.
Step 2: Second claim.
Another supporting line.
Final answer: yes
Trailing commentary that must not be parsed.
"""
        self.assertEqual(
            split_steps(trace),
            [
                "Step 1: First claim.\nSupporting detail for the first claim.",
                "Step 2: Second claim.\nAnother supporting line.",
            ],
        )

    def test_existing_single_line_math_steps_are_unchanged(self) -> None:
        trace = "Step 1: Add 2 and 2.\nStep 2: The result is 4.\nFinal answer: 4"
        self.assertEqual(
            split_steps(trace),
            ["Step 1: Add 2 and 2.", "Step 2: The result is 4."],
        )

    def test_score_alignment_validation(self) -> None:
        validate_score_alignment(["Step 1: A"], [0.1], [0.0])
        with self.assertRaisesRegex(ValueError, "alignment failure"):
            validate_score_alignment(["Step 1: A"], [0.1], [])


if __name__ == "__main__":
    unittest.main()
