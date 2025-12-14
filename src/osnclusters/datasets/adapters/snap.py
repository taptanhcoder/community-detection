from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np


EDGES_COLUMNS = ["u", "v"]
CHECKINS_REQUIRED = ["user_id", "ts", "lat", "lon"]


def enforce_edges_schema(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in EDGES_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Edges missing columns: {missing}. Found: {list(df.columns)}")
    out = df[EDGES_COLUMNS].copy()
    out["u"] = out["u"].astype(str)
    out["v"] = out["v"].astype(str)
    return out


def enforce_checkins_schema(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in CHECKINS_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Checkins missing columns: {missing}. Found: {list(df.columns)}")

    out = df.copy()
    out["user_id"] = out["user_id"].astype(str)
    out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")

    if "venue_id" in out.columns:
        out["venue_id"] = out["venue_id"].astype(str)

    return out


def parse_snap_edges(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["u", "v"], dtype=str, engine="python")
    return enforce_edges_schema(df)


def parse_snap_checkins(path: Path, column_guess: Optional[List[str]] = None) -> pd.DataFrame:
    df0 = pd.read_csv(path, sep=r"\s+", header=None, dtype=str, engine="python")

    if column_guess is None:
        if df0.shape[1] >= 5:
            column_guess = ["user_id", "ts", "lat", "lon", "venue_id"] + [f"extra_{i}" for i in range(df0.shape[1] - 5)]
        else:
            column_guess = ["user_id", "ts", "lat", "lon"] + [f"extra_{i}" for i in range(df0.shape[1] - 4)]

    df0.columns = column_guess[:df0.shape[1]]
    keep = [c for c in ["user_id", "ts", "lat", "lon", "venue_id"] if c in df0.columns]
    df = df0[keep].copy()
    return enforce_checkins_schema(df)
