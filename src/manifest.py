import json
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _git_commit_hash() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


@dataclass
class Manifest:
    run_name: str
    command: str
    config_path: str
    config: Dict[str, Any]
    started_at_utc: str
    git_commit: Optional[str]


def write_manifest(path: str | Path, manifest: Manifest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=2, ensure_ascii=False)


def make_manifest(run_name: str, command: str, config_path: str, config: Dict[str, Any]) -> Manifest:
    return Manifest(
        run_name=run_name,
        command=command,
        config_path=config_path,
        config=config,
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=_git_commit_hash(),
    )