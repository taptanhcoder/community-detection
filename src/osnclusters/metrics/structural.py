from __future__ import annotations

import numpy as np
import pandas as pd


def structural_metrics_igraph(edges_final: pd.DataFrame, comm_df: pd.DataFrame):
    """
    Graph structural metrics via igraph:
    - modularity
    - conductance mean/median (approx)
    - intra-community density mean/median
    """
    import igraph as ig

    users = comm_df["user_id"].astype(str).tolist()
    id2idx = {u: i for i, u in enumerate(users)}

    e = edges_final.copy()
    e["u"] = e["u"].astype(str); e["v"] = e["v"].astype(str)
    e = e[e["u"].isin(id2idx) & e["v"].isin(id2idx)]
    edges_idx = list(zip(e["u"].map(id2idx).tolist(), e["v"].map(id2idx).tolist()))
    g = ig.Graph(n=len(users), edges=edges_idx, directed=False)

    labels = comm_df.set_index("user_id").loc[users, "community_id"].to_numpy()
    mod = float(g.modularity(labels))

    deg = np.array(g.degree(), dtype=np.float64)
    total_vol = deg.sum()

    comm_to_nodes = {}
    for i, c in enumerate(labels):
        comm_to_nodes.setdefault(int(c), []).append(i)

    adj = [set(g.neighbors(i)) for i in range(g.vcount())]

    conductances = []
    densities = []
    for _, nodes in comm_to_nodes.items():
        nodes_set = set(nodes)
        cut = 0
        internal = 0
        for u in nodes:
            for v in adj[u]:
                if v in nodes_set:
                    internal += 1
                else:
                    cut += 1
        internal = internal / 2.0
        volS = deg[list(nodes)].sum()
        volT = total_vol - volS
        denom = min(volS, volT) if min(volS, volT) > 0 else np.nan
        phi = (cut / denom) if denom and not np.isnan(denom) else np.nan
        conductances.append(phi)

        n = len(nodes)
        possible = n * (n - 1) / 2.0
        dens = (internal / possible) if possible > 0 else np.nan
        densities.append(dens)

    return {
        "modularity": mod,
        "conductance_mean": float(np.nanmean(conductances)) if len(conductances) else np.nan,
        "conductance_median": float(np.nanmedian(conductances)) if len(conductances) else np.nan,
        "intra_density_mean": float(np.nanmean(densities)) if len(densities) else np.nan,
        "intra_density_median": float(np.nanmedian(densities)) if len(densities) else np.nan,
    }
