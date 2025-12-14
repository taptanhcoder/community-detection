from __future__ import annotations

import numpy as np


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + eps)


def build_knn_edges_cosine(Z: np.ndarray, k: int = 30, mutual: bool = True):
    """
    Build kNN edges using cosine distance => convert to cosine similarity weights (sim=1-dist).
    Returns (src, dst, w). If mutual=True, keep only mutual edges.
    """
    Z = l2_normalize_rows(Z.astype(np.float32))

    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(k + 1, Z.shape[0]), metric="cosine", algorithm="auto")
        nn.fit(Z)
        dist, ind = nn.kneighbors(Z, return_distance=True)
        sim = 1.0 - dist
        ind = ind[:, 1:]
        sim = sim[:, 1:]
    except Exception:
        N = Z.shape[0]
        if N > 20000:
            raise RuntimeError("scikit-learn not available and N is large; install: pip install -e .[sklearn]")
        S = Z @ Z.T
        np.fill_diagonal(S, -np.inf)
        ind = np.argpartition(-S, kth=min(k, N - 1), axis=1)[:, :k]
        row = np.arange(N)[:, None]
        sims = S[row, ind]
        order = np.argsort(-sims, axis=1)
        ind = ind[row, order]
        sim = sims[row, order]

    src = np.repeat(np.arange(Z.shape[0]), ind.shape[1])
    dst = ind.reshape(-1)
    w = sim.reshape(-1)

    if not mutual:
        return src, dst, w

    pairs = set(zip(src.tolist(), dst.tolist()))
    keep = np.array([(j, i) in pairs for i, j in zip(src, dst)], dtype=bool)
    return src[keep], dst[keep], w[keep]
