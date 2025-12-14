from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from osnclusters.core.types import DatasetName, DatasetSpec


def _human_size(nbytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(nbytes)
    for u in units:
        if size < 1024.0:
            return f"{size:.2f} {u}"
        size /= 1024.0
    return f"{size:.2f} PB"


def file_meta(path: Path) -> Dict[str, Any]:
    return {
        "exists": path.exists(),
        "size": _human_size(path.stat().st_size) if path.exists() else None,
        "path": str(path),
        "name": path.name,
    }


def safe_head_lines(path: Path, n: int = 3) -> List[str]:
    if path is None or (not path.exists()):
        return []
    lines = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for _ in range(n):
            line = f.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))
    return lines


def read_sample_whitespace(path: Path, nrows: int = 50) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, dtype=str, engine="python", nrows=nrows)


def validate_edges_df(df: pd.DataFrame) -> Dict[str, Any]:
    report = {"ok": True, "issues": []}
    if df.shape[1] < 2:
        report["ok"] = False
        report["issues"].append(f"Edges sample has <2 columns: shape={df.shape}")
        return report
    u = df.iloc[:, 0].astype(str)
    v = df.iloc[:, 1].astype(str)
    report["n_rows"] = len(df)
    report["n_unique_u"] = u.nunique()
    report["n_unique_v"] = v.nunique()
    report["self_loop_ratio"] = float((u == v).mean())
    pairs = pd.DataFrame({"u": u, "v": v})
    report["dup_ratio_sample"] = float(pairs.duplicated().mean())
    return report


def validate_checkins_df(df: pd.DataFrame) -> Dict[str, Any]:
    report = {"ok": True, "issues": [], "shape": df.shape}
    ncol = df.shape[1]
    if ncol < 3:
        report["ok"] = False
        report["issues"].append(f"Checkins sample has too few columns: {ncol}")
        return report

    candidates = []
    if ncol >= 5:
        candidates.append(("A5:user,ts,lat,lon,venue", [0, 1, 2, 3, 4]))
    if ncol >= 4:
        candidates.append(("B4:user,ts,lat,lon", [0, 1, 2, 3]))
    if ncol == 9:
        candidates.append(("L9:user,venue,wday,mon,day,time,tz,year,utc_offset", list(range(9))))
    candidates.append(("C3:user,venue,ts", [0, 1, 2]))

    best = None
    best_score = -1
    best_detail = None

    for name, idx in candidates:
        tmp = df.iloc[:, idx].copy()

        if name.startswith(("A5", "B4")):
            tmp.columns = ["user_id", "ts", "lat", "lon"] + (["venue_id"] if len(idx) == 5 else [])
            ts = pd.to_datetime(tmp["ts"], errors="coerce")
            lat = pd.to_numeric(tmp["lat"], errors="coerce")
            lon = pd.to_numeric(tmp["lon"], errors="coerce")

            ts_ok = ts.notna().mean()
            lat_ok = lat.notna().mean()
            lon_ok = lon.notna().mean()
            lat_in = ((lat >= -90) & (lat <= 90)).mean()
            lon_in = ((lon >= -180) & (lon <= 180)).mean()

            score = ts_ok + lat_ok + lon_ok + lat_in + lon_in
            best_detail_candidate = {
                "layout": name,
                "ts_parse_rate": float(ts_ok),
                "lat_parse_rate": float(lat_ok),
                "lon_parse_rate": float(lon_ok),
                "lat_in_range_rate": float(lat_in),
                "lon_in_range_rate": float(lon_in),
            }

        elif name.startswith("L9"):
            tmp.columns = ["user_id", "venue_id", "wday", "mon", "day", "time", "tz", "year", "utc_offset_min"]
            ts_raw = (
                tmp["wday"].astype(str) + " " +
                tmp["mon"].astype(str) + " " +
                tmp["day"].astype(str) + " " +
                tmp["time"].astype(str) + " " +
                tmp["tz"].astype(str) + " " +
                tmp["year"].astype(str)
            )
            ts = pd.to_datetime(ts_raw, errors="coerce", format="%a %b %d %H:%M:%S %z %Y")
            ts_ok = ts.notna().mean()
            score = ts_ok
            best_detail_candidate = {
                "layout": name,
                "ts_parse_rate": float(ts_ok),
                "note": "LBSN2Vec curated detected; lat/lon require POI join by venue_id.",
            }

        else:
            tmp.columns = ["user_id", "venue_id", "ts"]
            ts = pd.to_datetime(tmp["ts"], errors="coerce")
            ts_ok = ts.notna().mean()
            score = ts_ok
            best_detail_candidate = {
                "layout": name,
                "ts_parse_rate": float(ts_ok),
                "note": "lat/lon not present; requires POI join if needed.",
            }

        if score > best_score:
            best_score = score
            best = name
            best_detail = best_detail_candidate

    report["best_layout_guess"] = best
    report["best_layout_detail"] = best_detail

    if best is None:
        report["ok"] = False
        report["issues"].append("Cannot infer checkins layout from sample.")
        return report

    if best.startswith(("A5", "B4")):
        d = best_detail or {}
        if d.get("ts_parse_rate", 0) < 0.7 or d.get("lat_parse_rate", 0) < 0.7 or d.get("lon_parse_rate", 0) < 0.7:
            report["ok"] = False
            report["issues"].append("Low parse rate for ts/lat/lon in inferred layout.")
    else:
        if (best_detail or {}).get("ts_parse_rate", 0) < 0.7:
            report["ok"] = False
            report["issues"].append("Low timestamp parse rate.")
        report["needs_poi_join_for_latlon"] = True

    return report


def validate_dataset_registry_and_load(
    datasets: Dict[DatasetName, DatasetSpec],
    active: Optional[List[str]] = None,
    sample_rows: int = 100,
    preview_lines: int = 2,
) -> pd.DataFrame:
    rows = []
    active_set = set(active) if active else set(datasets.keys())

    for name, spec in datasets.items():
        if name not in active_set:
            continue

        row: Dict[str, Any] = {
            "dataset": name,
            "source": spec.source,
            "root_exists": spec.root.exists(),
            "root": str(spec.root),
        }

        paths: Dict[str, Optional[Path]] = {}
        if spec.source == "SNAP":
            paths["edges"] = spec.edges_path
            paths["checkins"] = spec.checkins_path
        else:
            paths["friendship_old"] = spec.friendship_old_path
            paths["friendship_new"] = spec.friendship_new_path
            paths["checkins_curated"] = spec.checkins_path
            paths["readme"] = spec.readme_path
            paths["poi"] = spec.poi_path

        for k, p in paths.items():
            meta = file_meta(p) if p else {"exists": False, "size": None}
            row[f"{k}_exists"] = meta["exists"]
            row[f"{k}_size"] = meta["size"]

        try:
            if spec.source == "SNAP":
                e_sample = read_sample_whitespace(spec.edges_path, nrows=sample_rows)
                e_val = validate_edges_df(e_sample)
                c_sample = read_sample_whitespace(spec.checkins_path, nrows=sample_rows)
                c_val = validate_checkins_df(c_sample)

                row["edges_ok"] = e_val["ok"]
                row["checkins_ok"] = c_val["ok"]
                row["checkins_best_layout"] = c_val.get("best_layout_guess")
                row["edges_head"] = " | ".join(safe_head_lines(spec.edges_path, n=preview_lines))
                row["checkins_head"] = " | ".join(safe_head_lines(spec.checkins_path, n=preview_lines))
                row["issues"] = "; ".join((e_val.get("issues") or []) + (c_val.get("issues") or []))
            else:
                old_sample = read_sample_whitespace(spec.friendship_old_path, nrows=sample_rows)
                new_sample = read_sample_whitespace(spec.friendship_new_path, nrows=sample_rows)
                old_val = validate_edges_df(old_sample)
                new_val = validate_edges_df(new_sample)

                chk_sample = read_sample_whitespace(spec.checkins_path, nrows=sample_rows)
                chk_val = validate_checkins_df(chk_sample)

                row["friendship_old_ok"] = old_val["ok"]
                row["friendship_new_ok"] = new_val["ok"]
                row["checkins_ok"] = chk_val["ok"]
                row["checkins_best_layout"] = chk_val.get("best_layout_guess")
                row["checkins_needs_poi_join"] = chk_val.get("needs_poi_join_for_latlon", False)
                row["issues"] = "; ".join((old_val.get("issues") or []) + (new_val.get("issues") or []) + (chk_val.get("issues") or []))

        except Exception as ex:
            row["issues"] = (row.get("issues", "") + f"; EXCEPTION: {type(ex).__name__}: {ex}").strip("; ")
            row.setdefault("edges_ok", False)
            row.setdefault("checkins_ok", False)

        rows.append(row)

    return pd.DataFrame(rows)
