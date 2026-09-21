from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets_prep import prepare_strategyqa_local


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(record: Dict[str, Any], seed: int) -> str:
    qid = str(record["qid"]).strip()
    return hashlib.sha256(f"{seed}:{qid}".encode("utf-8")).hexdigest()


def normalized_answer(record: Dict[str, Any]) -> str:
    answer = record.get("answer")
    if isinstance(answer, bool):
        return "yes" if answer else "no"

    value = str(answer).strip().lower()
    if value in {"yes", "true"}:
        return "yes"
    if value in {"no", "false"}:
        return "no"
    raise ValueError(
        f"Unsupported answer for qid={record.get('qid')!r}: {answer!r}"
    )


def freeze_extension(
    *,
    source: Path,
    exclude_manifest: Path,
    raw_output: Path,
    prepared_output: Path,
    count: int,
    seed: int,
    overwrite: bool,
) -> Dict[str, Any]:
    if count < 2 or count % 2 != 0:
        raise ValueError("count must be an even integer of at least 2")

    outputs = (
        raw_output,
        prepared_output,
        prepared_output.parent / f"manifest.{prepared_output.stem}.json",
        raw_output.parent / f"manifest.{raw_output.stem}.selection.json",
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing extension artifacts: "
            + ", ".join(str(path) for path in existing)
        )

    source_records = json.loads(
        source.read_text(encoding="utf-8-sig")
    )
    excluded = json.loads(
        exclude_manifest.read_text(encoding="utf-8")
    )
    excluded_qids = {str(qid) for qid in excluded["qids"]}

    if not isinstance(source_records, list):
        raise ValueError("StrategyQA source must be a JSON array")

    seen = set()
    candidates: Dict[str, List[Dict[str, Any]]] = {
        "yes": [],
        "no": [],
    }
    for record in source_records:
        qid = str(record.get("qid", "")).strip()
        if not qid:
            raise ValueError("StrategyQA source contains a missing qid")
        if qid in seen:
            raise ValueError(f"Duplicate StrategyQA qid: {qid}")
        seen.add(qid)

        if qid in excluded_qids:
            continue
        candidates[normalized_answer(record)].append(record)

    per_answer = count // 2
    selected: List[Dict[str, Any]] = []
    for answer in ("yes", "no"):
        ranked = sorted(
            candidates[answer],
            key=lambda record: stable_key(record, seed),
        )
        if len(ranked) < per_answer:
            raise ValueError(
                f"Not enough {answer!r} candidates for count={count}"
            )
        selected.extend(ranked[:per_answer])

    selected.sort(key=lambda record: stable_key(record, seed))
    selected_qids = [str(record["qid"]) for record in selected]

    if excluded_qids.intersection(selected_qids):
        raise ValueError("Extension overlaps the excluded StrategyQA records")

    raw_output.parent.mkdir(parents=True, exist_ok=True)
    raw_output.write_text(
        json.dumps(selected, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    prepare_strategyqa_local(
        input_path=raw_output,
        out_path=prepared_output,
        source_path=source,
        selection_seed=seed,
    )

    selection_manifest = {
        "protocol": "strategyqa_disjoint_balanced_extension_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source),
        "source_sha256": sha256(source),
        "exclude_manifest": str(exclude_manifest),
        "exclude_manifest_sha256": sha256(exclude_manifest),
        "excluded_records": len(excluded_qids),
        "selection_seed": seed,
        "selection_method": (
            "exclude prior qids; rank yes/no strata independently by "
            "sha256(seed:qid); select equal counts; stable-rank combined set"
        ),
        "num_records": len(selected),
        "answer_counts": {
            "yes": sum(normalized_answer(record) == "yes" for record in selected),
            "no": sum(normalized_answer(record) == "no" for record in selected),
        },
        "qids": selected_qids,
        "raw_output": str(raw_output),
        "raw_output_sha256": sha256(raw_output),
        "prepared_output": str(prepared_output),
        "prepared_output_sha256": sha256(prepared_output),
    }
    selection_manifest_path = (
        raw_output.parent / f"manifest.{raw_output.stem}.selection.json"
    )
    selection_manifest_path.write_text(
        json.dumps(selection_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return selection_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--prepared-output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = freeze_extension(
        source=args.source,
        exclude_manifest=args.exclude_manifest,
        raw_output=args.raw_output,
        prepared_output=args.prepared_output,
        count=args.count,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
