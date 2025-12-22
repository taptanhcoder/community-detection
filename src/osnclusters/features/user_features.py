from __future__ import annotations

import numpy as np
import pandas as pd


def _entropy_from_counts(counts: np.ndarray, eps: float = 1e-12) -> float:
    s = counts.sum()
    if s <= 0:
        return 0.0
    p = counts / (s + eps)
    p = p[p > 0]
    return float(-(p * np.log(p + eps)).sum())


def haversine_km_vec(lat, lon, lat2, lon2):
    R = 6371.0
    lat = np.radians(lat); lon = np.radians(lon)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat
    dlon = lon2 - lon
    a = np.sin(dlat/2.0)**2 + np.cos(lat)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c


def build_user_features_from_checkins(
    users_final: pd.DataFrame,
    checkins_final: pd.DataFrame,
    log1p_counts: bool = True,
    standardize: bool = True,
):

    chk = checkins_final.copy()
    chk["user_id"] = chk["user_id"].astype(str)
    chk["venue_id"] = chk["venue_id"].astype(str)
    chk["ts"] = pd.to_datetime(chk["ts"], errors="coerce")
    chk = chk.dropna(subset=["ts", "lat", "lon", "user_id"])

    chk["hour"] = chk["ts"].dt.hour.astype(int)
    chk["dow"] = chk["ts"].dt.dayofweek.astype(int)
    chk["date"] = chk["ts"].dt.date

    g = chk.groupby("user_id", sort=False)

    num_checkins = g.size().rename("num_checkins")
    num_active_days = g["date"].nunique().rename("num_active_days")
    num_unique_venues = g["venue_id"].nunique().rename("num_unique_venues")

    mean_lat = g["lat"].mean().rename("mean_lat")
    mean_lon = g["lon"].mean().rename("mean_lon")
    std_lat = g["lat"].std(ddof=0).fillna(0.0).rename("std_lat")
    std_lon = g["lon"].std(ddof=0).fillna(0.0).rename("std_lon")


    rog = {}
    med_dist = {}
    for uid, sub in g:
        latc = float(sub["lat"].mean())
        lonc = float(sub["lon"].mean())
        d = haversine_km_vec(sub["lat"].to_numpy(), sub["lon"].to_numpy(), latc, lonc)
        rog[uid] = float(np.sqrt(np.mean(d**2))) if len(d) else 0.0
        med_dist[uid] = float(np.median(d)) if len(d) else 0.0
    rog = pd.Series(rog, name="radius_of_gyration_km")
    med_dist = pd.Series(med_dist, name="median_dist_to_centroid_km")

    hour_counts = pd.crosstab(chk["user_id"], chk["hour"])
    for h in range(24):
        if h not in hour_counts.columns:
            hour_counts[h] = 0
    hour_counts = hour_counts[list(range(24))].sort_index(axis=1)

    dow_counts = pd.crosstab(chk["user_id"], chk["dow"])
    for d0 in range(7):
        if d0 not in dow_counts.columns:
            dow_counts[d0] = 0
    dow_counts = dow_counts[list(range(7))].sort_index(axis=1)

    hour_entropy = hour_counts.apply(lambda r: _entropy_from_counts(r.to_numpy()), axis=1).rename("hour_entropy")
    dow_entropy = dow_counts.apply(lambda r: _entropy_from_counts(r.to_numpy()), axis=1).rename("dow_entropy")
    venue_entropy = g["venue_id"].apply(lambda s: _entropy_from_counts(s.value_counts().to_numpy())).rename("venue_entropy")

    feat = pd.concat(
        [
            num_checkins, num_active_days, num_unique_venues,
            mean_lat, mean_lon, std_lat, std_lon,
            rog, med_dist,
            hour_entropy, dow_entropy, venue_entropy,
        ],
        axis=1,
    )

    hour_prop = hour_counts.div(hour_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    hour_prop.columns = [f"hour_{h:02d}_p" for h in hour_prop.columns]

    dow_prop = dow_counts.div(dow_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    dow_prop.columns = [f"dow_{d0}_p" for d0 in dow_prop.columns]

    feat = feat.join(hour_prop, how="left").join(dow_prop, how="left").fillna(0.0)

    if log1p_counts:
        for c in ["num_checkins", "num_active_days", "num_unique_venues"]:
            feat[c] = np.log1p(feat[c].astype(float))

    user_order = users_final["user_id"].astype(str).tolist()
    feat = feat.reindex(user_order).fillna(0.0)
    feat.index.name = "user_id"

    if standardize:
        mu = feat.mean(axis=0)
        sd = feat.std(axis=0, ddof=0).replace(0, 1.0)
        feat = (feat - mu) / sd

    X_users = feat.to_numpy(dtype=np.float32)
    feature_names = feat.columns.tolist()
    return X_users, feat, feature_names
