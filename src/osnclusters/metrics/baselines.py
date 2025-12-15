from __future__ import annotations

import numpy as np
import pandas as pd

from .spatial import spatial_cohesion_metrics


def random_baseline_same_sizes(comm_df: pd.DataFrame, n_runs: int = 10, seed: int = 42):
    rng = np.random.default_rng(seed)
    labels = comm_df["community_id"].to_numpy()
    baselines = []
    for _ in range(n_runs):
        perm = labels.copy()
        rng.shuffle(perm)
        baselines.append(perm)
    return baselines


def spatial_baseline_zscore(
    comm_df: pd.DataFrame,
    user_centroids: pd.DataFrame,
    n_runs: int,
    seed: int,
    post_min_comm_size: int = 1,  # NEW
):
    """
    Global spatial baseline on median(spatial_median_km) across communities.

    post_min_comm_size:
      - ignore communities with comm_size < post_min_comm_size (giảm nhiễu singleton).
    """
    comm_spatial = spatial_cohesion_metrics(comm_df, user_centroids)

    if post_min_comm_size > 1 and len(comm_spatial):
        comm_spatial = comm_spatial[comm_spatial["comm_size"] >= int(post_min_comm_size)].copy()

    obs_global = float(comm_spatial["spatial_median_km"].median()) if len(comm_spatial) else np.nan

    perms = random_baseline_same_sizes(comm_df, n_runs=n_runs, seed=seed)
    baseline_vals = []
    for perm in perms:
        tmp = comm_df.copy()
        tmp["community_id"] = perm
        tmp_sp = spatial_cohesion_metrics(tmp, user_centroids)
        if post_min_comm_size > 1 and len(tmp_sp):
            tmp_sp = tmp_sp[tmp_sp["comm_size"] >= int(post_min_comm_size)].copy()
        baseline_vals.append(float(tmp_sp["spatial_median_km"].median()) if len(tmp_sp) else np.nan)

    baseline_vals = np.array(baseline_vals, dtype=float)
    rand_mean = float(np.nanmean(baseline_vals)) if baseline_vals.size else np.nan
    rand_std = float(np.nanstd(baseline_vals)) if baseline_vals.size else np.nan
    z = (obs_global - rand_mean) / (rand_std + 1e-12) if np.isfinite(obs_global) else np.nan
    return obs_global, rand_mean, rand_std, float(z), baseline_vals
