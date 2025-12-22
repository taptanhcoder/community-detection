

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Paths & repo discovery
# ----------------------------
def find_repo_root(start: Optional[Path] = None) -> Path:
    start = start or Path.cwd()
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "pyproject.toml").exists() and (p / "data").exists():
            return p
    # fallback: find folder having "data"
    for p in [cur] + list(cur.parents):
        if (p / "data").exists():
            return p
    return cur


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    run_id: str  # where comm_df + Z.npy live (old run)
    # cleared_dir is always data/processed/data_cleared/<name>


def get_default_dataset_configs() -> Dict[str, DatasetConfig]:
    # Based on your current known run ids
    return {
        "brightkite": DatasetConfig(name="brightkite", run_id="20251214_192049"),
        "lbsn2vec": DatasetConfig(name="lbsn2vec", run_id="20251214_183903"),
    }


def get_paths() -> Dict[str, Path]:
    root = find_repo_root()
    data_dir = root / "data"
    processed = data_dir / "processed"
    return {
        "ROOT": root,
        "DATA_DIR": data_dir,
        "PROCESSED_DIR": processed,
        "RUNS_DIR": processed / "_runs",
        "CLEARED_DIR": processed / "data_cleared",
    }


# ----------------------------
# Safe readers
# ----------------------------
def read_json_safe(p: Path) -> Optional[dict]:
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def read_parquet_safe(p: Path, columns=None) -> Optional[pd.DataFrame]:
    try:
        return pd.read_parquet(p, columns=columns)
    except Exception:
        return None


def normalize_user_id(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.replace({"nan": np.nan, "None": np.nan})
    return s


def normalize_users(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "user_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "user_id"})
    df["user_id"] = normalize_user_id(df["user_id"])
    return df.dropna(subset=["user_id"])


def normalize_edges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "u" not in df.columns or "v" not in df.columns:
        df = df.rename(columns={df.columns[0]: "u", df.columns[1]: "v"})
    df["u"] = normalize_user_id(df["u"])
    df["v"] = normalize_user_id(df["v"])
    return df.dropna(subset=["u", "v"])


def normalize_checkins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["user_id"] = normalize_user_id(df["user_id"])
    return df.dropna(subset=["user_id"])


# ----------------------------
# Load dataset bundle
# ----------------------------
def dataset_paths(dataset: str, cfg: DatasetConfig) -> Dict[str, Path]:
    ps = get_paths()
    cleared = ps["CLEARED_DIR"] / dataset
    run_dir = ps["RUNS_DIR"] / cfg.run_id / dataset
    run_cfg = ps["RUNS_DIR"] / cfg.run_id / "run_config.json"
    return {
        "cleared_dir": cleared,
        "run_dir": run_dir,
        "run_config": run_cfg,
    }


def load_comm_df(run_dir: Path) -> pd.DataFrame:
    p_rep = run_dir / "comm_df.repaired.parquet"
    p_plain = run_dir / "comm_df.parquet"

    if p_rep.exists():
        df = pd.read_parquet(p_rep)
    elif p_plain.exists():
        df = pd.read_parquet(p_plain)
    else:
        raise FileNotFoundError(f"Missing comm_df(.repaired) in {run_dir}")

    if "user_id" not in df.columns:
        raise ValueError("comm_df missing user_id")

    if "community_id" not in df.columns:
        for alt in ["community", "comm_id", "cluster_id"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "community_id"})
                break

    if "community_id" not in df.columns:
        raise ValueError("comm_df missing community_id (or alt)")

    df = df[["user_id", "community_id"]].copy()
    df["user_id"] = normalize_user_id(df["user_id"])
    df = df.dropna(subset=["user_id"])
    df["community_id"] = pd.to_numeric(df["community_id"], errors="coerce").fillna(-1).astype(int)
    df = df[df["community_id"] >= 0].copy()
    return df


def load_Z_or_X(run_dir: Path, cleared_dir: Path) -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns: (name, Z, X)
    - Z from run_dir/Z.npy if exists
    - X from cleared_dir/X_users.npy if exists
    """
    Z = None
    X = None
    pZ = run_dir / "Z.npy"
    if pZ.exists():
        try:
            Z = np.load(pZ)
        except Exception:
            Z = None

    pX = cleared_dir / "X_users.npy"
    if pX.exists():
        try:
            X = np.load(pX)
        except Exception:
            X = None

    return ("Z", Z, X)


def compute_user_centroids(checkins_df: pd.DataFrame) -> pd.DataFrame:
    # mean lat/lon per user
    tmp = checkins_df[["user_id", "lat", "lon"]].copy()
    tmp["lat"] = pd.to_numeric(tmp["lat"], errors="coerce")
    tmp["lon"] = pd.to_numeric(tmp["lon"], errors="coerce")
    tmp = tmp.dropna(subset=["lat", "lon"])
    return tmp.groupby("user_id", as_index=False)[["lat", "lon"]].mean()


def load_cleared_tables(cleared_dir: Path) -> Dict[str, pd.DataFrame]:
    edges = read_parquet_safe(cleared_dir / "edges_final.parquet")
    users = read_parquet_safe(cleared_dir / "users_final.parquet")
    checkins = read_parquet_safe(cleared_dir / "checkins_final.parquet")

    if edges is None or users is None or checkins is None:
        raise RuntimeError(f"Failed to read cleared parquet(s) in {cleared_dir}")

    edges = normalize_edges(edges)
    users = normalize_users(users)
    checkins = normalize_checkins(checkins)
    return {"edges": edges, "users": users, "checkins": checkins}


def load_bundle(dataset: str, cfg: DatasetConfig) -> Dict[str, Any]:
    paths = dataset_paths(dataset, cfg)
    cleared_dir = paths["cleared_dir"]
    run_dir = paths["run_dir"]

    if not cleared_dir.exists():
        raise FileNotFoundError(cleared_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    tbls = load_cleared_tables(cleared_dir)
    comm_df = load_comm_df(run_dir)

    name, Z, X = load_Z_or_X(run_dir, cleared_dir)
    run_config = read_json_safe(paths["run_config"]) if paths["run_config"].exists() else None

    # For Z: user index mapping in old runs is often corrupted (users_final.parquet in run_dir).
    # A robust mapping: if len(comm_df) == Z rows, assume Z rows align with comm_df order.
    embed_user_ids = None
    if Z is not None:
        if len(comm_df) == Z.shape[0]:
            embed_user_ids = comm_df["user_id"].astype(str).to_numpy()
        else:
            # fallback to first n of cleared users
            u = tbls["users"]["user_id"].astype(str).to_numpy()
            n = min(len(u), Z.shape[0])
            embed_user_ids = u[:n]
            Z = Z[:n]

    return {
        "dataset": dataset,
        "paths": paths,
        "cleared_dir": cleared_dir,
        "run_dir": run_dir,

        "edges_final": tbls["edges"],
        "users_final": tbls["users"],
        "checkins_final": tbls["checkins"],

        "comm_df": comm_df,

        "Z": Z,
        "X_users": X,
        "embed_user_ids": embed_user_ids,  # for Z
        "run_config": run_config,
    }


# ----------------------------
# Step8 artifacts discovery
# ----------------------------
def find_latest_step8_out(runs_dir: Path) -> Optional[Path]:
    """
    Find newest .../step8_compare folder under data/processed/_runs/<timestamp>/step8_compare
    """
    if not runs_dir.exists():
        return None
    candidates = []
    for run in runs_dir.iterdir():
        if not run.is_dir():
            continue
        p = run / "step8_compare"
        if p.exists() and p.is_dir():
            candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.parent.name, reverse=True)
    return candidates[0]


def load_step8_tables(step8_dir: Path) -> Dict[str, Optional[pd.DataFrame]]:
    out = {}
    # You already have step8_structural_compare.csv
    for name, fn in [
        ("structural_compare", "step8_structural_compare.csv"),
        ("global_summary", "step8_global_summary.csv"),
        ("case_studies", "step8_case_studies.csv"),
    ]:
        p = step8_dir / fn
        out[name] = pd.read_csv(p) if p.exists() else None
    return out
