import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path

from io_utils import (
    get_paths,
    get_default_dataset_configs,
    load_bundle,
    find_latest_step8_out,
    load_step8_tables,
    compute_user_centroids,
)
from labeling import build_original_labels, knn_vote_augment_labels
from graph_layout import (
    sample_nodes_by_communities,
    induced_edges,
    compute_spring_layout,
)
from plots import (
    make_embedding_df,
    plot_embedding_scatter,
    plot_geo_scatter,
    plot_graph_layout,
)


st.set_page_config(page_title="Community Detection – Visualization", layout="wide")


@st.cache_data(show_spinner=False)
def cached_load_bundle(dataset: str):
    cfgs = get_default_dataset_configs()
    return load_bundle(dataset, cfgs[dataset])


@st.cache_data(show_spinner=False)
def cached_user_centroids(checkins_final: pd.DataFrame) -> pd.DataFrame:
    return compute_user_centroids(checkins_final)


@st.cache_data(show_spinner=False)
def cached_original_labels(users_final: pd.DataFrame, comm_df: pd.DataFrame):
    res = build_original_labels(users_final, comm_df)
    return res.labels, res.coverage


@st.cache_data(show_spinner=False)
def cached_augmented_labels(users_final: pd.DataFrame, X_users: np.ndarray, comm_df: pd.DataFrame, k: int):
    res = knn_vote_augment_labels(users_final, X_users, comm_df, k=k)
    return res.labels, res.coverage


def build_knn_edges(
    user_ids: np.ndarray,
    X: np.ndarray,
    k: int = 10,
    mutual: bool = True,
    seed: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:

    from sklearn.neighbors import NearestNeighbors

    user_ids = np.asarray(user_ids).astype(str)
    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 2 or X.shape[0] == 0 or len(user_ids) == 0:
        return pd.DataFrame(columns=["u", "v"]), np.array([], dtype=str)

    # ensure same length
    n0 = min(len(user_ids), X.shape[0])
    user_ids = user_ids[:n0]
    X = X[:n0]

    # 1) remove NaN/Inf rows
    finite_mask = np.isfinite(X).all(axis=1)
    user_ids2 = user_ids[finite_mask]
    X2 = X[finite_mask]

    if X2.shape[0] < 2:
        return pd.DataFrame(columns=["u", "v"]), user_ids2

    # 2) remove zero-norm rows (cosine needs non-zero vectors)
    norms = np.linalg.norm(X2, axis=1)
    good = np.isfinite(norms) & (norms > 0)
    user_ids2 = user_ids2[good]
    X2 = X2[good]

    n = X2.shape[0]
    if n < 2:
        return pd.DataFrame(columns=["u", "v"]), user_ids2

    nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="cosine", n_jobs=-1)
    nn.fit(X2)
    _, neigh = nn.kneighbors(X2, return_distance=True)

    edges = []
    for i in range(n):
        u = user_ids2[i]
        for jpos in range(1, neigh.shape[1]):  # skip self at 0
            j = int(neigh[i, jpos])
            v = user_ids2[j]
            edges.append((u, v))

    edges_df = pd.DataFrame(edges, columns=["u", "v"]).drop_duplicates()

    if mutual and len(edges_df):
        s = set(map(tuple, edges_df[["u", "v"]].to_numpy()))
        keep = [(u, v) for (u, v) in s if (v, u) in s]
        edges_df = pd.DataFrame(keep, columns=["u", "v"])

    return edges_df, user_ids2


def main():
    ps = get_paths()
    runs_dir = ps["RUNS_DIR"]

    st.title("Visualization")
    st.caption("Diagrams user–community for Summary / Spatial / Structural-A / Structural-B")

    # Sidebar controls
    st.sidebar.header("Controls")
    dataset = st.sidebar.selectbox("Dataset", ["brightkite", "lbsn2vec"], index=0)
    label_mode = st.sidebar.selectbox("Label mode", ["original", "augmented"], index=1)
    aug_k = st.sidebar.slider("Augment k (kNN vote on X_users)", min_value=5, max_value=80, value=25, step=5)

    summary_vec_space = st.sidebar.selectbox("Summary vector space", ["Z (GraphSAGE)", "X_users (features)"], index=0)
    max_points = st.sidebar.slider("Max points (scatter)", 5000, 120000, 40000, step=5000)

    st.sidebar.subheader("Graph layout sampling")
    max_nodes = st.sidebar.slider("Max nodes (layout)", 300, 6000, 2000, step=100)
    seed = st.sidebar.number_input("Seed", value=42, step=1)

    st.sidebar.subheader("Community colors")
    top_k_colors = st.sidebar.slider("Top-K communities with unique colors", 10, 300, 120, step=10)

    # Step8 artifacts (optional)
    step8_dir = find_latest_step8_out(runs_dir)
    step8_tables = load_step8_tables(step8_dir) if step8_dir else {}
    if step8_dir:
        st.sidebar.success(f"Found  artifacts: {step8_dir}")
    else:
        st.sidebar.warning("No step_compare folder found in data/processed/_runs/*")

    # Load dataset bundle
    with st.spinner("Loading dataset bundle..."):
        bundle = cached_load_bundle(dataset)

    users_final = bundle["users_final"]
    edges_final = bundle["edges_final"]
    checkins_final = bundle["checkins_final"]
    comm_df = bundle["comm_df"]
    Z = bundle["Z"]
    X_users = bundle["X_users"]
    embed_user_ids = bundle["embed_user_ids"]

    # Build labels (original / augmented)
    labels_orig, cov_orig = cached_original_labels(users_final, comm_df)
    if label_mode == "augmented":
        labels_aug, cov_aug = cached_augmented_labels(users_final, X_users, comm_df, k=aug_k)
        labels = labels_aug
        cov = cov_aug
    else:
        labels = labels_orig
        cov = cov_orig

    # Quick stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users total", f"{cov['users_total']:,}")
    c2.metric("Users labeled", f"{cov['users_labeled']:,}")
    c3.metric("Label coverage", f"{cov['label_coverage']:.3f}")
    c4.metric("Edges total", f"{len(edges_final):,}")

    if "note" in cov and cov["note"]:
        st.info(cov["note"])

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Summary diagram (Embedding PCA)",
        "Spatial diagram (Lat/Lon centroids)",
        "Structural-A (Friendship layout)",
        "Structural-B (kNN layout)",
    ])

    # -----------------------
    # TAB 1: Summary diagram
    # -----------------------
    with tab1:
        st.subheader("Summary diagram – users in 2D embedding space, colored by community")

        if summary_vec_space.startswith("Z"):
            if Z is None or embed_user_ids is None:
                st.error("Z.npy not available (or cannot map user_ids). Switch to X_users.")
            else:
                df = make_embedding_df(
                    user_ids=embed_user_ids,
                    vecs=Z,
                    labels=labels,
                    seed=int(seed),
                    max_points=int(max_points),
                )
                fig = plot_embedding_scatter(
                    df,
                    f"{dataset} – PCA2D(Z) colored by community ({label_mode})",
                    top_k=int(top_k_colors),
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            if X_users is None or len(X_users) != len(users_final):
                st.error("X_users.npy missing or not aligned with users_final.")
            else:
                user_ids = users_final["user_id"].astype(str).to_numpy()
                df = make_embedding_df(
                    user_ids=user_ids,
                    vecs=X_users,
                    labels=labels,
                    seed=int(seed),
                    max_points=int(max_points),
                )
                fig = plot_embedding_scatter(
                    df,
                    f"{dataset} – PCA2D(X_users) colored by community ({label_mode})",
                    top_k=int(top_k_colors),
                )
                st.plotly_chart(fig, use_container_width=True)

        st.caption("Interpretation: nếu các community tạo thành cụm tách biệt trong PCA2D, điều đó gợi ý phân nhóm hành vi rõ ràng.")

    # -----------------------
    # TAB 2: Spatial diagram
    # -----------------------
    with tab2:
        st.subheader("Spatial compactness diagram – user geographic centroids colored by community")

        with st.spinner("Computing user centroids from checkins..."):
            centroids = cached_user_centroids(checkins_final)

        fig = plot_geo_scatter(
            centroids=centroids,
            labels=labels,
            title=f"{dataset} – user centroids (lat/lon) colored by community ({label_mode})",
            max_points=int(max_points),
            seed=int(seed),
            top_k=int(top_k_colors),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Interpretation: community ‘đậm địa lý’ sẽ tạo vùng cụm (clusters) rõ ràng theo lat/lon; so sánh với baseline trong Step 8.")

    # -----------------------
    # TAB 3: Structural-A
    # -----------------------
    with tab3:
        st.subheader("Structural-A — Social ties cohesion (layout on friendship edges)")

        nodes = labels.dropna(subset=["community_id"])[["user_id", "community_id"]].copy()
        nodes = sample_nodes_by_communities(nodes, max_nodes=int(max_nodes), seed=int(seed))
        nodes["user_id"] = nodes["user_id"].astype(str)
        nodes_set = set(nodes["user_id"].tolist())

        e_ind = induced_edges(edges_final, nodes_set)
        st.write(f"Sampled nodes: {len(nodes):,} | induced friendship edges: {len(e_ind):,}")

        if len(nodes) < 10:
            st.warning("Too few labeled nodes to draw layout. Try label_mode=augmented.")
        else:
            with st.spinner("Computing spring layout..."):
                pos = compute_spring_layout(e_ind, nodes, seed=int(seed))
            fig = plot_graph_layout(
                nodes_df=nodes,
                edges_df=e_ind,
                pos=pos,
                title=f"{dataset} – Structural-A (friendship graph layout) – colored by community ({label_mode})",
                top_k=int(top_k_colors),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.caption("Interpretation: nếu community thật sự ‘đậm social’, các node cùng màu thường tụ lại và có nhiều edges nội bộ trong layout.")

    # -----------------------
    # TAB 4: Structural-B
    # -----------------------
    with tab4:
        st.subheader("Structural-B — Behavioral similarity cohesion (layout on kNN graph built from vectors)")

        k_knn = st.slider("k for kNN graph (Structural-B)", 5, 50, 15, step=1)
        mutual_knn = st.checkbox("Mutual kNN (keep only mutual edges)", value=True)

        vec_src = st.selectbox("Vector source for kNN graph", ["Z (GraphSAGE)", "X_users (features)"], index=0)

        if vec_src.startswith("Z"):
            if Z is None or embed_user_ids is None:
                st.error("Z not available for this dataset.")
            else:
                embed_user_ids_str = np.asarray(embed_user_ids).astype(str)

                # keep only labels that have Z user_id
                labZ = labels.merge(
                    pd.DataFrame({"user_id": embed_user_ids_str}),
                    on="user_id",
                    how="inner",
                )

                nodes = labZ.dropna(subset=["community_id"])[["user_id", "community_id"]].copy()
                nodes = sample_nodes_by_communities(nodes, max_nodes=int(max_nodes), seed=int(seed))
                nodes["user_id"] = nodes["user_id"].astype(str)

                # align nodes -> Z rows
                idx_map = {u: i for i, u in enumerate(embed_user_ids_str.tolist())}
                node_ids = []
                idx_list = []
                for u in nodes["user_id"].tolist():
                    if u in idx_map:
                        node_ids.append(u)
                        idx_list.append(idx_map[u])

                if len(idx_list) < 10:
                    st.warning("Too few nodes have valid embeddings (after filtering NaN/Inf).")
                else:
                    X = np.asarray(Z, dtype=np.float32)[np.array(idx_list, dtype=int)]
                    knn_edges, kept_ids = build_knn_edges(
                        user_ids=np.array(node_ids, dtype=str),
                        X=X,
                        k=int(k_knn),
                        mutual=bool(mutual_knn),
                        seed=int(seed),
                    )

                    nodes2 = nodes[nodes["user_id"].isin(set(kept_ids.tolist()))].copy()
                    st.write(f"Nodes used (after sanitize): {len(nodes2):,} | kNN edges: {len(knn_edges):,}")

                    with st.spinner("Computing spring layout..."):
                        pos = compute_spring_layout(knn_edges, nodes2, seed=int(seed))

                    fig = plot_graph_layout(
                        nodes_df=nodes2,
                        edges_df=knn_edges,
                        pos=pos,
                        title=f"{dataset} – Structural-B (kNN on Z, k={k_knn}) – colored by community ({label_mode})",
                        top_k=int(top_k_colors),
                    )
                    st.plotly_chart(fig, use_container_width=True)

        else:
            if X_users is None or len(X_users) != len(users_final):
                st.error("X_users missing or not aligned with users_final.")
            else:
                nodes = labels.dropna(subset=["community_id"])[["user_id", "community_id"]].copy()
                nodes = sample_nodes_by_communities(nodes, max_nodes=int(max_nodes), seed=int(seed))
                nodes["user_id"] = nodes["user_id"].astype(str)

                uid_all = users_final["user_id"].astype(str).to_numpy()
                idx_map = {u: i for i, u in enumerate(uid_all.tolist())}

                node_ids = []
                idx_list = []
                for u in nodes["user_id"].tolist():
                    if u in idx_map:
                        node_ids.append(u)
                        idx_list.append(idx_map[u])

                if len(idx_list) < 10:
                    st.warning("Too few nodes map to users_final indices (after filtering).")
                else:
                    X = np.asarray(X_users, dtype=np.float32)[np.array(idx_list, dtype=int)]
                    knn_edges, kept_ids = build_knn_edges(
                        user_ids=np.array(node_ids, dtype=str),
                        X=X,
                        k=int(k_knn),
                        mutual=bool(mutual_knn),
                        seed=int(seed),
                    )

                    nodes2 = nodes[nodes["user_id"].isin(set(kept_ids.tolist()))].copy()
                    st.write(f"Nodes used (after sanitize): {len(nodes2):,} | kNN edges: {len(knn_edges):,}")

                    with st.spinner("Computing spring layout..."):
                        pos = compute_spring_layout(knn_edges, nodes2, seed=int(seed))

                    fig = plot_graph_layout(
                        nodes_df=nodes2,
                        edges_df=knn_edges,
                        pos=pos,
                        title=f"{dataset} – Structural-B (kNN on X_users, k={k_knn}) – colored by community ({label_mode})",
                        top_k=int(top_k_colors),
                    )
                    st.plotly_chart(fig, use_container_width=True)

        st.caption("Interpretation: Structural-B dùng đúng ‘graph theo hành vi’ (kNN), nên cohesion thường rõ hơn Structural-A.")

    # Optional: show Step 8 CSV if available
    st.divider()
    st.subheader("Step 8 artifacts (optional)")
    if step8_dir:
        st.write(f"Loaded from: `{step8_dir}`")
        if step8_tables.get("structural_compare") is not None:
            st.write("step8_structural_compare.csv")
            st.dataframe(step8_tables["structural_compare"], use_container_width=True)
        else:
            st.warning("Missing step8_structural_compare.csv in step8_compare folder.")
    else:
        st.info("No step8_compare artifacts discovered; diagrams still work from cleared + run dirs.")


if __name__ == "__main__":
    main()
