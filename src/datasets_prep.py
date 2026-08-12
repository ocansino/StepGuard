from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .task_profiles import normalize_strategyqa_answer


@dataclass
class DatasetManifest:
    dataset_name: str
    config_name: Optional[str]
    split: str
    out_path: str
    created_at_utc: str


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def _write_manifest(path: Path, manifest: DatasetManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")


def prepare_gsm8k(split: str, out_path: Path) -> int:
    """
    GSM8K from HF typically has fields:
      - question: str
      - answer: str (contains explanation + '#### <final answer>')
    We store:
      - question
      - gold_answer (parsed final answer)
      - task = "math"
      - source
    """
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split=split)

    rows = []
    for i, ex in enumerate(ds):
        q = ex["question"]
        ans = ex["answer"]

        # GSM8K convention: final answer is after '####'
        gold = None
        if "####" in ans:
            gold = ans.split("####")[-1].strip()
        else:
            # fallback: keep raw answer if parsing fails
            gold = ans.strip()

        rid = f"gsm8k_{split}_{i:06d}"
        rows.append(
            {
                "id": rid,
                "task": "math",
                "source": f"gsm8k/main/{split}",
                "question": q,
                "gold_answer": gold,
            }
        )

    n = _write_jsonl(out_path, rows)

    manifest = DatasetManifest(
        dataset_name="gsm8k",
        config_name="main",
        split=split,
        out_path=str(out_path),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    _write_manifest(out_path.parent / f"manifest.gsm8k.{split}.json", manifest)
    return n


def prepare_strategyqa(split: str, out_path: Path) -> int:
    """
    StrategyQA fields vary by config, but generally include:
      - question
      - answer (bool or string)
    Evidence is often not included as a passage; you'll likely add retrieval later.
    For now we store:
      - question
      - gold_answer
      - task = "strategyqa"
      - source
    """
    from datasets import load_dataset

    ds = load_dataset("strategyqa", split=split)

    rows = []
    for i, ex in enumerate(ds):
        q = ex.get("question", "")
        a = ex.get("answer", None)

        # normalize gold_answer to a string ("yes"/"no" is common)
        if isinstance(a, bool):
            gold = "yes" if a else "no"
        elif a is None:
            gold = None
        else:
            gold = str(a).strip()

        rid = f"strategyqa_{split}_{i:06d}"
        row = {
            "id": rid,
            "task": "strategyqa",
            "source": f"strategyqa/{split}",
            "question": q,
        }
        if gold is not None:
            row["gold_answer"] = gold
        rows.append(row)

    n = _write_jsonl(out_path, rows)

    manifest = DatasetManifest(
        dataset_name="strategyqa",
        config_name=None,
        split=split,
        out_path=str(out_path),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    _write_manifest(out_path.parent / f"manifest.strategyqa.{split}.json", manifest)
    return n


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def prepare_strategyqa_local(
    input_path: Path,
    out_path: Path,
    *,
    source_path: Optional[Path] = None,
    selection_seed: int = 42,
) -> int:
    """Convert a local StrategyQA JSON array into StepGuard's JSONL schema."""
    input_path = Path(input_path)
    out_path = Path(out_path)
    source_path = Path(source_path) if source_path is not None else input_path

    # utf-8-sig accepts ordinary UTF-8 and files written with a BOM.
    with input_path.open("r", encoding="utf-8-sig") as file_obj:
        raw_records = json.load(file_obj)

    if not isinstance(raw_records, list):
        raise ValueError(
            f"StrategyQA input must be a JSON array, got {type(raw_records).__name__}"
        )

    rows = []
    qids = []
    seen_qids = set()
    answer_counts = {"yes": 0, "no": 0}

    metadata_fields = (
        "term",
        "description",
        "facts",
        "decomposition",
        "evidence",
    )

    for index, example in enumerate(raw_records):
        if not isinstance(example, dict):
            raise ValueError(f"StrategyQA record {index} is not a JSON object")

        qid = str(example.get("qid", "")).strip()
        question = str(example.get("question", "")).strip()
        if not qid:
            raise ValueError(f"StrategyQA record {index} is missing qid")
        if qid in seen_qids:
            raise ValueError(f"Duplicate StrategyQA qid: {qid}")
        if not question:
            raise ValueError(f"StrategyQA record {qid} is missing question")

        raw_answer = example.get("answer")
        if isinstance(raw_answer, bool):
            gold_answer = "yes" if raw_answer else "no"
        else:
            gold_answer = normalize_strategyqa_answer(raw_answer)
        if gold_answer is None:
            raise ValueError(
                f"StrategyQA record {qid} has an unsupported answer: {raw_answer!r}"
            )

        metadata = {
            field: example[field]
            for field in metadata_fields
            if field in example
        }

        rows.append(
            {
                "id": qid,
                "task": "strategyqa",
                "source": "strategyqa/public_train",
                "question": question,
                "gold_answer": gold_answer,
                "metadata": metadata,
            }
        )
        qids.append(qid)
        seen_qids.add(qid)
        answer_counts[gold_answer] += 1

    count = _write_jsonl(out_path, rows)
    manifest = {
        "dataset_name": "strategyqa",
        "split": "public_train_subset",
        "input_path": str(input_path),
        "input_sha256": _sha256(input_path),
        "source_path": str(source_path),
        "source_sha256": _sha256(source_path),
        "out_path": str(out_path),
        "selection_seed": selection_seed,
        "num_records": count,
        "answer_counts": answer_counts,
        "qids": qids,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = out_path.parent / f"manifest.{out_path.stem}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return count
