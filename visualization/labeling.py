from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class LabelingResult:
    labels: pd.DataFrame  # user_id, community_id, label_source
    coverage: dict


def build_original_labels(users_final: pd.DataFrame, comm_df: pd.DataFrame) -> LabelingResult:
    users = users_final[["user_id"]].copy()
    users["user_id"] = users["user_id"].astype(str)

    lab = comm_df[["user_id", "community_id"]].copy()
    lab["user_id"] = lab["user_id"].astype(str)
    lab = lab.drop_duplicates(subset=["user_id"], keep="first")

    merged = users.merge(lab, on="user_id", how="left")
    merged["label_source"] = np.where(merged["community_id"].notna(), "original", "unlabeled")
    merged["community_id"] = pd.to_numeric(merged["community_id"], errors="coerce")

    cov = {
        "users_total": int(len(users)),
        "users_labeled": int(merged["community_id"].notna().sum()),
        "label_coverage": float(merged["community_id"].notna().mean()) if len(users) else 0.0,
    }
    return LabelingResult(labels=merged, coverage=cov)


def knn_vote_augment_labels(
    users_final: pd.DataFrame,
    X_users: np.ndarray,
    comm_df: pd.DataFrame,
    k: int = 25,
    metric: str = "cosine",
    max_labeled_fit: int = 200000,
    seed: int = 42,
) -> LabelingResult:
    """
    Augment labels for unlabeled users in users_final using kNN voting in feature space X_users.
    Assumes X_users aligns with users_final row order (as produced by Step 4).
    """
    rng = np.random.default_rng(seed)

    users = users_final[["user_id"]].copy()
    users["user_id"] = users["user_id"].astype(str)

    if X_users is None or len(X_users) != len(users):
        # Cannot augment reliably
        base = build_original_labels(users_final, comm_df)
        base.labels["label_source"] = np.where(base.labels["community_id"].notna(), "original", "unlabeled")
        base.coverage["note"] = "X_users missing or not aligned with users_final; augmentation disabled."
        return base

    # seed labels
    seed_lab = comm_df[["user_id", "community_id"]].copy()
    seed_lab["user_id"] = seed_lab["user_id"].astype(str)
    seed_lab = seed_lab.drop_duplicates(subset=["user_id"], keep="first")

    merged = users.merge(seed_lab, on="user_id", how="left")
    y = merged["community_id"].to_numpy()
    is_labeled = ~pd.isna(y)

    if is_labeled.sum() == 0:
        merged["label_source"] = "unlabeled"
        return LabelingResult(
            labels=merged,
            coverage={"users_total": int(len(users)), "users_labeled": 0, "label_coverage": 0.0, "note": "No seed labels."},
        )

    X = X_users
    idx_l = np.where(is_labeled)[0]
    idx_u = np.where(~is_labeled)[0]

    # optional downsample labeled set for fitting if too big
    if len(idx_l) > max_labeled_fit:
        idx_l = rng.choice(idx_l, size=max_labeled_fit, replace=False)

    X_l = X[idx_l]
    y_l = merged.loc[idx_l, "community_id"].astype(int).to_numpy()

    # Fit kNN on labeled points only
    nn = NearestNeighbors(n_neighbors=min(k, len(X_l)), metric=metric, n_jobs=-1)
    nn.fit(X_l)

    # Query unlabeled points
    X_u = X[idx_u]
    dists, neigh = nn.kneighbors(X_u, return_distance=True)  # shape: (U, k)

    # Weighted vote: weight = 1/(dist+eps) for cosine distances
    eps = 1e-9
    weights = 1.0 / (dists + eps)

    pred = []
    for i in range(neigh.shape[0]):
        nbr_ids = neigh[i]
        nbr_labels = y_l[nbr_ids]
        nbr_w = weights[i]

        # aggregate weights per label
        wsum = {}
        for lab, w in zip(nbr_labels, nbr_w):
            wsum[int(lab)] = wsum.get(int(lab), 0.0) + float(w)
        # pick max weight
        pred.append(max(wsum.items(), key=lambda t: t[1])[0])

    merged.loc[idx_u, "community_id"] = np.array(pred, dtype=int)
    merged["label_source"] = np.where(is_labeled, "original", "augmented")

    cov = {
        "users_total": int(len(users)),
        "users_labeled": int(merged["community_id"].notna().sum()),
        "label_coverage": float(merged["community_id"].notna().mean()) if len(users) else 0.0,
        "seed_labeled": int(is_labeled.sum()),
        "augmented": int((merged["label_source"] == "augmented").sum()),
        "k": int(k),
        "metric": metric,
    }
    return LabelingResult(labels=merged, coverage=cov)
