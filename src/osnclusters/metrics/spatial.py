from __future__ import annotations

import numpy as np
import pandas as pd


def haversine_km_vec(lat, lon, lat2, lon2):
    R = 6371.0
    lat = np.radians(lat); lon = np.radians(lon)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat
    dlon = lon2 - lon
    a = np.sin(dlat/2.0)**2 + np.cos(lat)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c


def compute_user_centroids(checkins_final: pd.DataFrame) -> pd.DataFrame:
    chk = checkins_final.copy()
    chk["user_id"] = chk["user_id"].astype(str)
    cent = chk.groupby("user_id")[["lat", "lon"]].mean().reset_index()
    cent.columns = ["user_id", "user_lat", "user_lon"]
    return cent


def spatial_cohesion_metrics(comm_df: pd.DataFrame, user_centroids: pd.DataFrame) -> pd.DataFrame:
    """
    Per community: centroid of user centroids then median/mean distance user->community centroid.
    """
    df = comm_df.merge(user_centroids, on="user_id", how="left").dropna(subset=["user_lat", "user_lon"])
    g = df.groupby("community_id")

    rows = []
    for cid, sub in g:
        latc = sub["user_lat"].mean()
        lonc = sub["user_lon"].mean()
        d = haversine_km_vec(sub["user_lat"].to_numpy(), sub["user_lon"].to_numpy(), latc, lonc)
        rows.append({
            "community_id": int(cid),
            "comm_size": int(len(sub)),
            "spatial_median_km": float(np.median(d)) if len(d) else np.nan,
            "spatial_mean_km": float(np.mean(d)) if len(d) else np.nan,
        })
    return pd.DataFrame(rows)
