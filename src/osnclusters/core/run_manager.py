from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

from .config import save_run_config


@dataclass(frozen=True)
class RunContext:
    project_root: Path
    data_dir: Path
    processed_dir: Path
    run_dir: Path


def create_run_context(cfg: Dict[str, Any]) -> RunContext:
    project_root = Path(cfg["_resolved"]["project_root"]).resolve()
    data_dir = (project_root / cfg["project"]["data_dir"]).resolve()
    processed_dir = (project_root / cfg["project"]["processed_dir"]).resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = processed_dir / "_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if cfg.get("run", {}).get("save_run_config", True):
        save_run_config(cfg, run_dir / "run_config.json")

    return RunContext(
        project_root=project_root,
        data_dir=data_dir,
        processed_dir=processed_dir,
        run_dir=run_dir,
    )
