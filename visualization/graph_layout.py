from __future__ import annotations

from typing import Tuple, Dict

import numpy as np
import pandas as pd
import networkx as nx


def induced_edges(edges: pd.DataFrame, nodes_set: set[str]) -> pd.DataFrame:
    e = edges[["u", "v"]].copy()
    e["u"] = e["u"].astype(str)
    e["v"] = e["v"].astype(str)
    return e[e["u"].isin(nodes_set) & e["v"].isin(nodes_set)].copy()


def sample_nodes_by_communities(
    labels_df: pd.DataFrame,
    max_nodes: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Stratified sampling: keep more nodes from larger communities.
    labels_df columns: user_id, community_id (must exist)
    """
    rng = np.random.default_rng(seed)
    df = labels_df[["user_id", "community_id"]].dropna().copy()
    df["user_id"] = df["user_id"].astype(str)
    df["community_id"] = df["community_id"].astype(int)

    if len(df) <= max_nodes:
        return df

    # allocate samples per community proportional to size (with min 1)
    counts = df["community_id"].value_counts()
    comms = counts.index.to_numpy()
    sizes = counts.values.astype(float)
    probs = sizes / sizes.sum()

    alloc = np.floor(probs * max_nodes).astype(int)
    alloc = np.maximum(alloc, 1)

    # fix sum to max_nodes
    while alloc.sum() > max_nodes:
        i = int(rng.integers(0, len(alloc)))
        if alloc[i] > 1:
            alloc[i] -= 1
    while alloc.sum() < max_nodes:
        i = int(rng.integers(0, len(alloc)))
        alloc[i] += 1

    picked = []
    for c, k in zip(comms, alloc):
        members = df[df["community_id"] == c]["user_id"].to_numpy()
        if len(members) <= k:
            chosen = members
        else:
            chosen = rng.choice(members, size=k, replace=False)
        picked.append(pd.DataFrame({"user_id": chosen, "community_id": int(c)}))

    out = pd.concat(picked, ignore_index=True)
    return out.drop_duplicates(subset=["user_id"])


def compute_spring_layout(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    seed: int = 42,
    k: float | None = None,
) -> Dict[str, Tuple[float, float]]:
    """
    Returns positions dict: user_id -> (x, y)
    """
    G = nx.Graph()
    for _, r in nodes.iterrows():
        G.add_node(str(r["user_id"]))
    for _, r in edges.iterrows():
        u, v = str(r["u"]), str(r["v"])
        if u in G and v in G and u != v:
            G.add_edge(u, v)

    pos = nx.spring_layout(G, seed=seed, k=k)  # dict node->(x,y)
    return {n: (float(x), float(y)) for n, (x, y) in pos.items()}
