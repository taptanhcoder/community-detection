from __future__ import annotations

from pathlib import Path
from typing import Optional, Set
import numpy as np
import pandas as pd


def read_edges_two_cols(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, usecols=[0, 1], dtype=str, engine="python")
    df.columns = ["u", "v"]
    df["u"] = df["u"].astype(str)
    df["v"] = df["v"].astype(str)
    return df


def read_pois_minimal(path: Path) -> pd.DataFrame:
    poi = pd.read_csv(path, sep=r"\s+", header=None, usecols=[0, 1, 2], dtype=str, engine="python")
    poi.columns = ["venue_id", "lat", "lon"]
    poi["venue_id"] = poi["venue_id"].astype(str)
    poi["lat"] = pd.to_numeric(poi["lat"], errors="coerce")
    poi["lon"] = pd.to_numeric(poi["lon"], errors="coerce")
    return poi


def parse_lbsn_curated_checkins_9col_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    if chunk.shape[1] < 9:
        raise ValueError(f"Expected >=9 columns for curated checkins, got shape={chunk.shape}")

    df = chunk.iloc[:, :9].copy()
    df.columns = ["user_id", "venue_id", "dow", "mon", "day", "time", "tz", "year", "tz_offset_min"]

    ts_str = (
        df["dow"].astype(str) + " " +
        df["mon"].astype(str) + " " +
        df["day"].astype(str) + " " +
        df["time"].astype(str) + " " +
        df["tz"].astype(str) + " " +
        df["year"].astype(str)
    )
    ts = pd.to_datetime(ts_str, format="%a %b %d %H:%M:%S %z %Y", errors="coerce")

    out = pd.DataFrame({
        "user_id": df["user_id"].astype(str),
        "venue_id": df["venue_id"].astype(str),
        "ts": ts,
        "tz_offset_min": pd.to_numeric(df["tz_offset_min"], errors="coerce"),
    })
    return out


def sample_users_from_checkins_chunked(
    checkins_path: Path,
    sample_frac: float,
    seed: int,
    chunksize: int = 2_000_000,
) -> Set[str]:
    users = set()
    for chunk in pd.read_csv(checkins_path, sep=r"\s+", header=None, dtype=str, engine="python", chunksize=chunksize):
        users.update(chunk.iloc[:, 0].astype(str).unique().tolist())

    users = np.array(list(users), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    n = int(np.ceil(len(users) * float(sample_frac)))
    return set(users[:n].tolist())


def load_lbsn2vec_curated_with_pois(
    friendship_path: Path,
    checkins_path: Path,
    poi_path: Path,
    sample_frac: float,
    seed: int,
    chunksize: int = 2_000_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # sample users (like Phase 1)
    keep_users = None
    if sample_frac < 1.0:
        keep_users = sample_users_from_checkins_chunked(checkins_path, sample_frac, seed, chunksize=chunksize)

    edges_all = read_edges_two_cols(friendship_path)
    edges_raw = edges_all
    if keep_users is not None:
        edges_raw = edges_all[edges_all["u"].isin(keep_users) & edges_all["v"].isin(keep_users)].copy()

    chunks = []
    for chunk in pd.read_csv(checkins_path, sep=r"\s+", header=None, dtype=str, engine="python", chunksize=chunksize):
        if keep_users is not None:
            chunk = chunk[chunk.iloc[:, 0].astype(str).isin(keep_users)]
        if len(chunk) == 0:
            continue
        chunks.append(parse_lbsn_curated_checkins_9col_chunk(chunk))

    checkins_core = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
        columns=["user_id", "venue_id", "ts", "tz_offset_min"]
    )

    pois = read_pois_minimal(poi_path)
    if len(checkins_core) > 0:
        needed_venues = set(checkins_core["venue_id"].unique().tolist())
        pois = pois[pois["venue_id"].isin(needed_venues)].copy()

    checkins_raw = checkins_core.merge(pois, on="venue_id", how="left")
    checkins_raw["lat"] = pd.to_numeric(checkins_raw["lat"], errors="coerce")
    checkins_raw["lon"] = pd.to_numeric(checkins_raw["lon"], errors="coerce")
    return edges_raw, checkins_raw
