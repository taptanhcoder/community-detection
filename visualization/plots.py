# visualization/plots.py
from __future__ import annotations

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go


# -----------------------------
# Numeric safety helpers
# -----------------------------
def sanitize_matrix(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (X_clean, keep_mask) after removing NaN/Inf rows and zero-norm rows.
    keep_mask is aligned with original X rows.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0:
        return X.reshape(0, X.shape[-1] if X.ndim == 2 else 0), np.zeros((0,), dtype=bool)

    finite = np.isfinite(X).all(axis=1)
    X1 = X[finite]
    if X1.shape[0] == 0:
        return X1, np.zeros((X.shape[0],), dtype=bool)

    norms = np.linalg.norm(X1, axis=1)
    nonzero = np.isfinite(norms) & (norms > 0)
    X2 = X1[nonzero]

    keep = np.zeros((X.shape[0],), dtype=bool)
    keep_idx = np.where(finite)[0]
    keep[keep_idx[nonzero]] = True
    return X2, keep


# -----------------------------
# Dimensionality reduction
# -----------------------------
def pca_2d(X: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    PCA to 2D with robust NaN handling:
    - drops NaN/Inf/zero-norm rows
    - if <2 rows remain -> returns zeros (same length as input)
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)

    X2, keep = sanitize_matrix(X)
    if X2.shape[0] < 2:
        return np.zeros((X.shape[0], 2), dtype=np.float32)

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=seed)
    xy2 = pca.fit_transform(X2).astype(np.float32)

    xy = np.zeros((X.shape[0], 2), dtype=np.float32)
    xy[keep] = xy2
    return xy


# -----------------------------
# Coloring (Top-K communities colored)
# -----------------------------
def _build_palette(n: int) -> list[str]:
    pal: list[str] = []
    pal += px.colors.qualitative.Dark24
    pal += px.colors.qualitative.Alphabet
    pal += px.colors.qualitative.Light24
    pal += px.colors.qualitative.Set3
    if len(pal) < n:
        pal = (pal * (n // max(1, len(pal)) + 1))[:n]
    return pal[:n]


def add_comm_color_group(
    df: pd.DataFrame,
    top_k: int = 120,
    other_label: str = "Other",
    comm_col: str = "community_id",
) -> tuple[pd.DataFrame, dict]:
    """
    Adds:
      - df['comm_group'] = community_id for Top-K, else 'Other'
    Returns:
      - df2, color_map
    """
    df2 = df.copy()
    if comm_col not in df2.columns:
        df2["comm_group"] = other_label
        return df2, {other_label: "#BDBDBD"}

    s = df2[comm_col].astype(str)
    top = s.value_counts().head(int(top_k)).index.tolist()
    df2["comm_group"] = np.where(s.isin(top), s, other_label)

    colors = _build_palette(len(top))
    cmap = {c: colors[i] for i, c in enumerate(top)}
    cmap[other_label] = "#BDBDBD"
    return df2, cmap


# -----------------------------
# DataFrame builders
# -----------------------------
def make_embedding_df(
    user_ids: np.ndarray,
    vecs: np.ndarray,
    labels: pd.DataFrame,
    seed: int = 42,
    max_points: int = 40000,
) -> pd.DataFrame:
    """
    Build df for PCA scatter.
    - Align by row index between user_ids and vecs (min length)
    - Merge community_id by user_id from labels (labels can be defined on users_final)
    - Sample max_points BEFORE PCA for performance
    - Remove NaN/Inf/zero-norm rows before PCA (drop those points)
    Returns columns: user_id, community_id, x, y
    """
    user_ids = np.asarray(user_ids).astype(str)
    vecs = np.asarray(vecs, dtype=np.float32)

    n = min(len(user_ids), vecs.shape[0])
    if n <= 0:
        return pd.DataFrame(columns=["user_id", "community_id", "x", "y"])

    user_ids = user_ids[:n]
    vecs = vecs[:n]

    df = pd.DataFrame({"user_id": user_ids})

    lab = labels.copy()
    lab["user_id"] = lab["user_id"].astype(str)
    if "community_id" not in lab.columns:
        df["community_id"] = -1
    else:
        df = df.merge(lab[["user_id", "community_id"]], on="user_id", how="left")
        df["community_id"] = pd.to_numeric(df["community_id"], errors="coerce").fillna(-1).astype(int)

    # sample for speed
    if max_points is not None and len(df) > int(max_points):
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(len(df), size=int(max_points), replace=False)
        df = df.iloc[idx].reset_index(drop=True)
        vecs = vecs[idx]

    # sanitize rows for PCA
    vecs_clean, keep_mask = sanitize_matrix(vecs)
    if vecs_clean.shape[0] < 2:
        # return empty or trivial
        df = df.iloc[0:0].copy()
        df["x"] = []
        df["y"] = []
        return df

    df = df.loc[keep_mask].reset_index(drop=True)
    xy = pca_2d(vecs[keep_mask], seed=int(seed))
    # xy returned same length as input -> safe
    df["x"] = xy[:, 0]
    df["y"] = xy[:, 1]
    return df


# -----------------------------
# Plotting
# -----------------------------
def plot_embedding_scatter(df: pd.DataFrame, title: str, top_k: int = 120) -> go.Figure:
    if df is None or len(df) == 0:
        fig = go.Figure()
        fig.update_layout(title=f"{title} (no data)", height=650)
        return fig

    df2, cmap = add_comm_color_group(df, top_k=int(top_k), comm_col="community_id")

    fig = px.scatter(
        df2,
        x="x",
        y="y",
        color="comm_group",
        color_discrete_map=cmap,
        hover_data=["user_id", "community_id"],
        title=title,
        opacity=0.95,
    )
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(height=650, legend_title_text="community")
    return fig


def plot_geo_scatter(
    centroids: pd.DataFrame,
    labels: pd.DataFrame,
    title: str,
    max_points: int = 40000,
    seed: int = 42,
    top_k: int = 120,
) -> go.Figure:
    """
    centroids: user_id, lat, lon
    labels: user_id, community_id
    """
    if centroids is None or len(centroids) == 0:
        fig = go.Figure()
        fig.update_layout(title=f"{title} (no centroids)", height=650)
        return fig

    df = centroids.copy()
    df["user_id"] = df["user_id"].astype(str)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])

    lab = labels.copy()
    lab["user_id"] = lab["user_id"].astype(str)
    df = df.merge(lab[["user_id", "community_id"]], on="user_id", how="left")
    df["community_id"] = pd.to_numeric(df["community_id"], errors="coerce").fillna(-1).astype(int)

    if max_points is not None and len(df) > int(max_points):
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(len(df), size=int(max_points), replace=False)
        df = df.iloc[idx].reset_index(drop=True)

    df2, cmap = add_comm_color_group(df, top_k=int(top_k), comm_col="community_id")

    fig = px.scatter_geo(
        df2,
        lat="lat",
        lon="lon",
        color="comm_group",
        color_discrete_map=cmap,
        hover_data=["user_id", "community_id"],
        title=title,
        opacity=0.9,
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(height=650, legend_title_text="community")
    return fig


def plot_graph_layout(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    pos,
    title: str,
    top_k: int = 120,
) -> go.Figure:
    """
    nodes_df: columns [user_id, community_id]
    edges_df: columns [u, v]
    pos: mapping user_id -> (x,y) OR DataFrame with columns user_id,x,y
    """
    if nodes_df is None or len(nodes_df) == 0:
        fig = go.Figure()
        fig.update_layout(title=f"{title} (no nodes)", height=750)
        return fig

    nodes = nodes_df.copy()
    nodes["user_id"] = nodes["user_id"].astype(str)
    nodes["community_id"] = pd.to_numeric(nodes["community_id"], errors="coerce").fillna(-1).astype(int)

    # attach x,y from pos
    if isinstance(pos, dict):
        nodes["x"] = nodes["user_id"].map(lambda u: float(pos.get(u, (0.0, 0.0))[0]))
        nodes["y"] = nodes["user_id"].map(lambda u: float(pos.get(u, (0.0, 0.0))[1]))
    elif isinstance(pos, pd.DataFrame) and {"user_id", "x", "y"}.issubset(set(pos.columns)):
        tmp = pos.copy()
        tmp["user_id"] = tmp["user_id"].astype(str)
        nodes = nodes.merge(tmp[["user_id", "x", "y"]], on="user_id", how="left")
        nodes["x"] = nodes["x"].fillna(0.0).astype(float)
        nodes["y"] = nodes["y"].fillna(0.0).astype(float)
    else:
        nodes["x"] = 0.0
        nodes["y"] = 0.0

    nodes2, cmap = add_comm_color_group(nodes, top_k=int(top_k), comm_col="community_id")

    # edges segments
    if edges_df is None or len(edges_df) == 0:
        edge_trace = go.Scattergl(x=[], y=[], mode="lines", hoverinfo="none", name="edges")
    else:
        e = edges_df[["u", "v"]].copy()
        e["u"] = e["u"].astype(str)
        e["v"] = e["v"].astype(str)

        pos_index = nodes2.set_index("user_id")[["x", "y"]]
        xs, ys = [], []
        for u, v in e.itertuples(index=False, name=None):
            if u in pos_index.index and v in pos_index.index:
                xs += [pos_index.at[u, "x"], pos_index.at[v, "x"], None]
                ys += [pos_index.at[u, "y"], pos_index.at[v, "y"], None]

        edge_trace = go.Scattergl(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(width=0.7, color="rgba(160,160,160,0.45)"),
            hoverinfo="none",
            name="edges",
        )

    # node scatter
    fig = px.scatter(
        nodes2,
        x="x",
        y="y",
        color="comm_group",
        color_discrete_map=cmap,
        hover_data=["user_id", "community_id"],
        title=title,
        opacity=0.98,
    )
    fig.update_traces(marker=dict(size=7), selector=dict(mode="markers"))
    fig.add_trace(edge_trace)

    fig.update_layout(
        height=750,
        showlegend=True,
        legend_title_text="community",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig
