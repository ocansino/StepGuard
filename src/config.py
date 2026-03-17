from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml


@dataclass
class RunConfig:
    run_name: str
    dataset_path: str
    output_dir: str
    raw: Dict[str, Any]  # keep full config available


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return RunConfig(
        run_name=data["run_name"],
        dataset_path=data["dataset_path"],
        output_dir=data["output_dir"],
        raw=data,
    )