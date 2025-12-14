from __future__ import annotations

import numpy as np


def leiden_partition_from_edges(n_nodes: int, src: np.ndarray, dst: np.ndarray, w: np.ndarray, resolution: float = 1.0):
    """
    Build igraph graph then Leiden partition (RBConfigurationVertexPartition).
    Fallback: igraph community_multilevel if leidenalg missing/fails.
    Returns: (labels, info)
    """
    try:
        import igraph as ig
    except Exception as e:
        raise RuntimeError("Leiden step requires python-igraph. Install extras: pip install -e .[igraph]") from e

    g = ig.Graph(n=n_nodes, edges=list(zip(src.tolist(), dst.tolist())), directed=False)
    g.es["weight"] = w.astype(float).tolist()

    try:
        import leidenalg
        part = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=float(resolution),
        )
        labels = np.array(part.membership, dtype=np.int64)
        method = "leiden"
        quality = float(part.quality())
    except Exception:
        part = g.community_multilevel(weights="weight")
        labels = np.array(part.membership, dtype=np.int64)
        method = "igraph_multilevel"
        quality = float(part.modularity)

    info = {
        "method": method,
        "quality": quality,
        "n_communities": int(labels.max() + 1) if labels.size else 0,
    }
    return labels, info
