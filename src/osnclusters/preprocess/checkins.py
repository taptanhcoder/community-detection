from __future__ import annotations

import pandas as pd
import numpy as np


def clean_checkins(chk: pd.DataFrame, cfg: dict) -> pd.DataFrame:

    lat_lo, lat_hi = cfg["preprocess"]["lat_range"]
    lon_lo, lon_hi = cfg["preprocess"]["lon_range"]

    out = chk.copy()

    out["user_id"] = out["user_id"].astype(str)
    if "venue_id" in out.columns:
        out["venue_id"] = out["venue_id"].astype(str)
    else:
        out["venue_id"] = "NA"

    out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")

    out = out.dropna(subset=["user_id", "ts", "lat", "lon", "venue_id"])
    out = out[(out["lat"] >= lat_lo) & (out["lat"] <= lat_hi) & (out["lon"] >= lon_lo) & (out["lon"] <= lon_hi)]


    try:
        if hasattr(out["ts"].dt, "tz") and out["ts"].dt.tz is not None:
            out["ts"] = out["ts"].dt.tz_convert("UTC").dt.tz_localize(None)
        else:
            out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
    except Exception:
        out["ts"] = pd.to_datetime(out["ts"], errors="coerce")

    out = out.dropna(subset=["ts"]).reset_index(drop=True)
    return out[["user_id", "ts", "lat", "lon", "venue_id"]].copy()
