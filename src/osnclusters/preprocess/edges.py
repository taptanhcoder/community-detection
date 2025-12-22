from __future__ import annotations

import numpy as np
import pandas as pd


def make_undirected_dedup(edges: pd.DataFrame) -> pd.DataFrame:

    u = edges["u"].astype(str).to_numpy()
    v = edges["v"].astype(str).to_numpy()
    u2 = np.where(u <= v, u, v)
    v2 = np.where(u <= v, v, u)
    out = pd.DataFrame({"u": u2, "v": v2})
    out = out[out["u"] != out["v"]]
    out = out.drop_duplicates(["u", "v"]).reset_index(drop=True)
    return out


def degree_from_edges(edges: pd.DataFrame) -> pd.Series:
    u = edges["u"].astype(str)
    v = edges["v"].astype(str)
    return pd.concat([u, v]).value_counts()


def filter_induced_once(edges: pd.DataFrame, chk: pd.DataFrame, k: int, d: int):

    ccount = chk["user_id"].astype(str).value_counts()
    deg = degree_from_edges(edges)

    users_ok = ccount[ccount >= k].index
    deg_ok = deg[deg >= d].index
    v_keep = pd.Index(users_ok).intersection(pd.Index(deg_ok))

    edges2 = edges[edges["u"].isin(v_keep) & edges["v"].isin(v_keep)].copy().reset_index(drop=True)
    chk2 = chk[chk["user_id"].isin(v_keep)].copy().reset_index(drop=True)

    return v_keep, edges2, chk2


def iterative_filter(edges: pd.DataFrame, chk: pd.DataFrame, k: int, d: int, iterative: bool = True, max_rounds: int = 20):

    edges_tmp = edges.copy()
    chk_tmp = chk.copy()

    prev_users = -1
    history = []

    for r in range(1, max_rounds + 1):
        v_keep, edges_tmp, chk_tmp = filter_induced_once(edges_tmp, chk_tmp, k=k, d=d)
        n_users = len(v_keep)
        history.append((r, n_users, len(edges_tmp), len(chk_tmp)))

        if (not iterative) or (n_users == prev_users):
            break
        prev_users = n_users

    users_final = pd.DataFrame({"user_id": pd.Index(chk_tmp["user_id"].astype(str).unique()).sort_values()})
    return users_final, edges_tmp, chk_tmp, history
