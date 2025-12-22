from __future__ import annotations

import numpy as np
import pandas as pd


def _shannon_entropy_from_counts(counts: np.ndarray) -> float:
    """Shannon entropy (entropy: độ đa dạng phân bố)."""
    counts = counts.astype(np.float64, copy=False)
    s = counts.sum()
    if s <= 0:
        return float("nan")
    p = counts / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _entropy_norm(ent: float, k: int) -> float:
    """Normalize entropy to [0,1] by log(k) when k>1."""
    if not np.isfinite(ent) or k <= 1:
        return float("nan")
    return float(ent / (np.log(k) + 1e-12))


def semantic_cohesion_metrics(
    comm_df: pd.DataFrame,
    checkins_final: pd.DataFrame,
    venue_col: str = "venue_id",
    category_col_candidates: tuple[str, ...] = ("category", "venue_type", "venue_category", "cat"),
    min_comm_size: int = 1,
) -> pd.DataFrame:
    """
    Per community semantic cohesion for LBSN-like checkins.
    - venue entropy (always if venue_id exists)
    - category entropy (optional if any category column exists)

    Output columns are safe even if category missing (filled with NaN).
    """
    if venue_col not in checkins_final.columns:
        # Không có venue => không thể làm semantic
        return pd.DataFrame(columns=[
            "community_id", "comm_size", "n_checkins",
            "venue_entropy", "venue_entropy_norm",
            "category_col", "category_entropy", "category_entropy_norm",
            "dominant_category", "dominant_category_share",
            "n_unique_categories", "n_unique_venues",
        ])

    chk = checkins_final.copy()
    chk["user_id"] = chk["user_id"].astype(str)
    chk[venue_col] = chk[venue_col].astype(str)

    # tìm category col nếu có
    cat_col = None
    for c in category_col_candidates:
        if c in chk.columns:
            cat_col = c
            break

    if cat_col is not None:
        chk[cat_col] = chk[cat_col].astype(str)

    df = comm_df.copy()
    df["user_id"] = df["user_id"].astype(str)

    # join community label vào checkins để đếm theo community
    dfc = chk.merge(df[["user_id", "community_id"]], on="user_id", how="inner")

    # comm_size theo membership user (không phải số checkins)
    comm_sizes = df.groupby("community_id")["user_id"].nunique().rename("comm_size")

    rows = []
    for cid, sub in dfc.groupby("community_id"):
        comm_size = int(comm_sizes.get(cid, 0))
        if comm_size < int(min_comm_size):
            continue

        n_checkins = int(len(sub))


        venue_counts = sub[venue_col].value_counts().to_numpy()
        venue_ent = _shannon_entropy_from_counts(venue_counts)
        venue_k = int(len(venue_counts))
        venue_ent_n = _entropy_norm(venue_ent, venue_k)


        if cat_col is not None:
            cat_vc = sub[cat_col].value_counts()
            cat_counts = cat_vc.to_numpy()
            cat_ent = _shannon_entropy_from_counts(cat_counts)
            cat_k = int(len(cat_counts))
            cat_ent_n = _entropy_norm(cat_ent, cat_k)

            dominant_cat = str(cat_vc.index[0]) if len(cat_vc) else None
            dominant_share = float(cat_vc.iloc[0] / max(1, n_checkins)) if len(cat_vc) else np.nan
            n_unique_cat = int(cat_k)
        else:
            cat_ent = np.nan
            cat_ent_n = np.nan
            dominant_cat = None
            dominant_share = np.nan
            n_unique_cat = 0

        rows.append({
            "community_id": int(cid),
            "comm_size": int(comm_size),
            "n_checkins": int(n_checkins),
            "n_unique_venues": int(venue_k),
            "venue_entropy": float(venue_ent),
            "venue_entropy_norm": float(venue_ent_n),
            "category_col": str(cat_col) if cat_col is not None else None,
            "n_unique_categories": int(n_unique_cat),
            "category_entropy": float(cat_ent),
            "category_entropy_norm": float(cat_ent_n),
            "dominant_category": dominant_cat,
            "dominant_category_share": float(dominant_share) if np.isfinite(dominant_share) else np.nan,
        })

    return pd.DataFrame(rows)
