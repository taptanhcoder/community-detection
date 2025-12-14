from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

import yaml


def find_repo_root(start: Optional[Path] = None) -> Path:
    env_root = os.environ.get("CD_PROJECT_ROOT")
    if env_root:
        p = Path(env_root).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"CD_PROJECT_ROOT is set but does not exist: {env_root}")

    if start is None:
        start = Path.cwd()
    start = start.resolve()

    for p in [start] + list(start.parents):
        if (p / "pyproject.toml").exists():
            return p
        if (p / "configs" / "default.yaml").exists():
            return p
        if (p / "README.md").exists():
            return p

    return start


def default_cfg() -> Dict[str, Any]:
    # Mirror Phase 1 defaults (so no KeyError like CFG['run'])
    return {
        "project": {
            "name": "community-detection",
            "data_dir": "data",
            "processed_dir": "data/processed",
            "artifacts_format": "parquet",
        },
        "run": {
            "seed": 42,
            "log_level": "INFO",
            "save_run_config": True,
            "overwrite": False,
        },
        "preprocess": {
            "min_checkins": 10,
            "min_degree": 3,
            "iterative_filter": True,
            "drop_self_loops": True,
            "dedup_edges": True,
            "enforce_undirected": True,
            "lat_range": [-90.0, 90.0],
            "lon_range": [-180.0, 180.0],
        },
        "features": {
            "log1p_counts": True,
            "standardize": True,
        },
        "model": {
            "hidden_dim": 128,
            "embed_dim": 128,
            "neighbor_sampling": [25, 10],
        },
        "train": {
            "epochs": 10,
            "lr": 1e-3,
            "batch_size": 1024,
            "num_negative": 5,
            "train_edge_frac": 0.1,
        },
        "community": {
            "knn_k": 30,
            "mutual_knn": True,
            "leiden_resolution": 1.0,
        },
        "metrics": {
            "random_baseline_runs": 10,
        },
        "datasets": {
            "active": ["brightkite", "gowalla", "lbsn2vec"],
            "lbsn2vec_snapshot": "old",
            "lbsn2vec_tier": "curated",
        },
    }


def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (u or {}).items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def load_config(config_path: Optional[Path] = None, project_root: Optional[Path] = None) -> Dict[str, Any]:
    if project_root is None:
        project_root = find_repo_root()

    if config_path is None:
        config_path = project_root / "configs" / "default.yaml"

    cfg = default_cfg()

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        cfg = deep_update(cfg, y)

    # Derive absolute-ish paths from repo root, but keep strings for JSON friendliness
    cfg["_resolved"] = {
        "project_root": str(project_root),
        "config_path": str(config_path),
    }
    return cfg


def save_run_config(cfg: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
