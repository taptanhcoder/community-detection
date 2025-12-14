from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_npy(arr: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    np.save(str(path), arr)


def load_npy(path: Path) -> np.ndarray:
    return np.load(str(path), allow_pickle=False)


def _can_parquet() -> bool:
    try:
        import pyarrow  # noqa
        return True
    except Exception:
        try:
            import fastparquet  # noqa
            return True
        except Exception:
            return False


def save_df(df: pd.DataFrame, path: Path) -> None:
    """
    Save dataframe to parquet if available, else fallback to csv.
    If path suffix is '.parquet' but parquet not available, we write '.csv' instead.
    """
    ensure_dir(path.parent)
    if path.suffix.lower() == ".parquet":
        if _can_parquet():
            df.to_parquet(path, index=False)
            return
        # fallback
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return

    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
        return

    # default: parquet if possible
    if _can_parquet():
        df.to_parquet(path.with_suffix(".parquet"), index=False)
    else:
        df.to_csv(path.with_suffix(".csv"), index=False)


def load_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataframe format: {path}")


def make_artifact_paths(run_dir: Path, dataset: str) -> Dict[str, Path]:
    base = run_dir / dataset
    return {
        "edges_clean": base / "edges_clean.parquet",
        "checkins_clean": base / "checkins_clean.parquet",
        "users_final": base / "users_final.parquet",
        "edges_final": base / "edges_final.parquet",
        "checkins_final": base / "checkins_final.parquet",
        "feat_df": base / "feat_df.parquet",
        "X_users": base / "X_users.npy",
        "Z": base / "Z.npy",
        "comm_df": base / "comm_df.parquet",
        "comm_metrics": base / "comm_metrics.parquet",
        "metrics_global": base / "metrics_global.json",
    }
