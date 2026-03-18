from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from datasets import load_dataset


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
      - task = "nonmath"
      - source
    """
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
            "task": "nonmath",
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