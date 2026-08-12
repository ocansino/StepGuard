from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, TextIO


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def read_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected a JSON object in {path}, "
            f"found {type(payload).__name__}"
        )

    return payload


def _write_text_atomic(
    path: str | Path,
    writer: Callable[[TextIO], None],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as file:
            writer(file)
            file.flush()
            os.fsync(file.fileno())

        # The temporary file is in the same directory/filesystem as the
        # destination, allowing replacement to occur as one filesystem action.
        temporary_path.replace(path)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def write_jsonl(
    path: str | Path,
    records: Iterable[Dict[str, Any]],
) -> None:
    def write_records(file: TextIO) -> None:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    _write_text_atomic(path, write_records)


def write_json_atomic(
    path: str | Path,
    payload: Dict[str, Any],
) -> None:
    def write_payload(file: TextIO) -> None:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")

    _write_text_atomic(path, write_payload)