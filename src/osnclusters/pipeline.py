from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from osnclusters.core.types import DatasetName, DatasetSpec
from osnclusters.datasets.registry import build_dataset_registry
from osnclusters.datasets.adapters.snap import parse_snap_edges, parse_snap_checkins
from osnclusters.datasets.adapters.lbsn2vec import load_lbsn2vec_curated_with_pois
from osnclusters.preprocess.edges import make_undirected_dedup, iterative_filter
from osnclusters.preprocess.checkins import clean_checkins
from osnclusters.features.user_features import build_user_features_from_checkins
from osnclusters.models.graphsage_unsup import train_graphsage_unsup
from osnclusters.community.knn_graph import build_knn_edges_cosine
from osnclusters.community.leiden import leiden_partition_from_edges
from osnclusters.metrics.spatial import compute_user_centroids, spatial_cohesion_metrics
from osnclusters.metrics.baselines import spatial_baseline_zscore
from osnclusters.io.artifacts import make_artifact_paths, save_df, save_npy, save_json


@dataclass(frozen=True)
class RunResult:
    dataset: str
    comm_df: pd.DataFrame
    comm_metrics_df: pd.DataFrame
    metrics_global: Dict[str, Any]


def _resolve_data_dir(cfg: Dict[str, Any], project_root: Path) -> Path:
    return (project_root / cfg["project"]["data_dir"]).resolve()


def _resolve_snapshot_paths(spec: DatasetSpec, snapshot: str) -> Tuple[Path, Path, Path]:
    if spec.friendship_old_path is None or spec.friendship_new_path is None:
        raise ValueError("LBSN2Vec spec missing friendship paths.")
    edges_path = spec.friendship_old_path if snapshot == "old" else spec.friendship_new_path
    if spec.checkins_path is None or spec.poi_path is None:
        raise ValueError("LBSN2Vec spec missing checkins/poi paths.")
    return edges_path, spec.checkins_path, spec.poi_path


def run_one_dataset(
    dataset_name: DatasetName,
    cfg: Dict[str, Any],
    project_root: Path,
    run_dir: Path,
    sample_frac: float = 1.0,
    train_edge_frac: Optional[float] = None,
    logger=None,
) -> RunResult:
    data_dir = _resolve_data_dir(cfg, project_root)
    datasets = build_dataset_registry(cfg, data_dir=data_dir)
    spec = datasets[dataset_name]

    if train_edge_frac is None:
        train_edge_frac = float(cfg.get("train", {}).get("train_edge_frac", 0.1))

    if logger:
        logger.info(f"[RUN] dataset={dataset_name} | source={spec.source} | sample_frac={sample_frac} | train_edge_frac={train_edge_frac}")

    # Step 1: load raw -> parse
    if spec.source == "SNAP":
        if spec.edges_path is None or spec.checkins_path is None:
            raise FileNotFoundError(f"SNAP dataset spec missing files: {dataset_name}")
        if logger:
            logger.info("[C1] parsing SNAP edges/checkins ...")
        edges_raw = parse_snap_edges(spec.edges_path)
        checkins_raw = parse_snap_checkins(spec.checkins_path)

        if sample_frac < 1.0:
            rng = np.random.default_rng(int(cfg.get("run", {}).get("seed", 42)))
            users = checkins_raw["user_id"].astype(str).unique()
            rng.shuffle(users)
            n = max(1, int(np.ceil(len(users) * sample_frac)))
            keep = set(users[:n].tolist())
            checkins_raw = checkins_raw[checkins_raw["user_id"].astype(str).isin(keep)].copy()
            edges_raw = edges_raw[edges_raw["u"].astype(str).isin(keep) & edges_raw["v"].astype(str).isin(keep)].copy()
    else:
        snapshot = str(cfg["datasets"].get("lbsn2vec_snapshot", "old"))
        edges_path, checkins_path, poi_path = _resolve_snapshot_paths(spec, snapshot)
        if logger:
            logger.info(f"[C1] loading LBSN2Vec curated + POI join | snapshot={snapshot} ...")
        edges_raw, checkins_raw = load_lbsn2vec_curated_with_pois(
            friendship_path=edges_path,
            checkins_path=checkins_path,
            poi_path=poi_path,
            sample_frac=float(sample_frac),
            seed=int(cfg.get("run", {}).get("seed", 42)),
            chunksize=2_000_000,
        )

    if logger:
        ts_ok = float(pd.to_datetime(checkins_raw["ts"], errors="coerce").notna().mean()) if len(checkins_raw) else 0.0
        lat_ok = float(pd.to_numeric(checkins_raw["lat"], errors="coerce").notna().mean()) if "lat" in checkins_raw.columns and len(checkins_raw) else 0.0
        lon_ok = float(pd.to_numeric(checkins_raw["lon"], errors="coerce").notna().mean()) if "lon" in checkins_raw.columns and len(checkins_raw) else 0.0
        logger.info(f"[C1] edges_raw={edges_raw.shape} | checkins_raw={checkins_raw.shape} | ts_ok={ts_ok:.3f} lat_ok={lat_ok:.3f} lon_ok={lon_ok:.3f}")

    # Step 2: cleaning
    if logger:
        logger.info("[C2] cleaning edges/checkins ...")
    edges_clean = make_undirected_dedup(edges_raw)
    checkins_clean = clean_checkins(checkins_raw, cfg)

    # Step 3: induced filtering
    k = int(cfg["preprocess"].get("min_checkins", 10))
    d = int(cfg["preprocess"].get("min_degree", 3))
    iterative = bool(cfg["preprocess"].get("iterative_filter", True))
    if logger:
        logger.info(f"[C3] induced filter: k={k} d={d} iterative={iterative}")
    users_final, edges_final, checkins_final, history = iterative_filter(
        edges_clean, checkins_clean, k=k, d=d, iterative=iterative, max_rounds=20
    )
    if logger:
        logger.info(f"[C3] DONE users={len(users_final)} edges={len(edges_final)} checkins={len(checkins_final)} | last={history[-1] if history else None}")

    # Step 4: build X_users
    if logger:
        logger.info("[C4] building user features ...")
    X_users, feat_df, feature_names = build_user_features_from_checkins(
        users_final=users_final,
        checkins_final=checkins_final,
        log1p_counts=bool(cfg.get("features", {}).get("log1p_counts", True)),
        standardize=bool(cfg.get("features", {}).get("standardize", True)),
    )
    if logger:
        logger.info(f"[C4] X_users shape={X_users.shape} | n_features={len(feature_names)}")

    # Step 5: GraphSAGE
    if logger:
        logger.info("[C5] training GraphSAGE ...")
    m_cfg = cfg.get("model", {})
    t_cfg = cfg.get("train", {})
    Z, _ = train_graphsage_unsup(
        edges_final=edges_final,
        users_final=users_final,
        X_users=X_users,
        hidden_dim=int(m_cfg.get("hidden_dim", 128)),
        embed_dim=int(m_cfg.get("embed_dim", 128)),
        neighbor_sampling=tuple(m_cfg.get("neighbor_sampling", [25, 10])),
        epochs=int(t_cfg.get("epochs", 10)),
        batch_size=int(t_cfg.get("batch_size", 1024)),
        num_negative=int(t_cfg.get("num_negative", 5)),
        lr=float(t_cfg.get("lr", 1e-3)),
        seed=int(cfg.get("run", {}).get("seed", 42)),
        train_edge_frac=float(train_edge_frac),
        logger=logger,
    )
    if logger:
        logger.info(f"[C5] Z shape={Z.shape}")

    # Step 6: kNN + Leiden
    c_cfg = cfg.get("community", {})
    knn_k = int(c_cfg.get("knn_k", 30))
    mutual_knn = bool(c_cfg.get("mutual_knn", True))
    resolution = float(c_cfg.get("leiden_resolution", 1.0))
    if logger:
        logger.info(f"[C6] kNN graph k={knn_k} mutual={mutual_knn} resolution={resolution}")

    # (Nếu bạn đã áp dụng bản fix weight âm ở knn_graph/leiden thì giữ nguyên call này)
    src, dst, w = build_knn_edges_cosine(Z, k=knn_k, mutual=mutual_knn)
    labels, info = leiden_partition_from_edges(n_nodes=Z.shape[0], src=src, dst=dst, w=w, resolution=resolution)

    comm_df = pd.DataFrame({
        "user_id": users_final["user_id"].astype(str).tolist(),
        "community_id": labels.astype(int),
    })
    sizes = comm_df["community_id"].value_counts().sort_values(ascending=False)
    comm_stats = {
        "n_users": int(len(comm_df)),
        "n_communities": int(info["n_communities"]),
        "largest_comm_size": int(sizes.iloc[0]) if len(sizes) else 0,
        "median_comm_size": float(sizes.median()) if len(sizes) else 0.0,
        "method": info["method"],
        "quality": float(info["quality"]),
    }
    if logger:
        logger.info(f"[C6] stats: {comm_stats}")

    # Step 7: metrics
    if logger:
        logger.info("[C7] metrics: spatial + structural + random baseline ...")

    # 7.1 Spatial cohesion
    user_centroids = compute_user_centroids(checkins_final)
    comm_spatial = spatial_cohesion_metrics(comm_df, user_centroids)

    # 7.3 Semantic cohesion (ONLY for LBSN2Vec, optional if data has venue/category)
    comm_semantic = None
    if spec.source == "LBSN2Vec":
        try:
            from osnclusters.metrics.semantic import semantic_cohesion_metrics
            post_min = int(cfg.get("metrics", {}).get("post_min_comm_size", 1))
            comm_semantic = semantic_cohesion_metrics(
                comm_df=comm_df,
                checkins_final=checkins_final,
                venue_col="venue_id",
                min_comm_size=post_min,
            )
            if logger:
                cols = list(comm_semantic.columns) if comm_semantic is not None else []
                logger.info(f"[C7] semantic metrics enabled | rows={0 if comm_semantic is None else len(comm_semantic)} | cols={cols}")
        except Exception as e:
            comm_semantic = None
            if logger:
                logger.warning(f"[C7] semantic cohesion skipped. Reason: {type(e).__name__}: {e}")

    # merge per-community metrics into one table
    comm_metrics = comm_spatial.copy()
    if comm_semantic is not None and len(comm_semantic):
        comm_metrics = comm_metrics.merge(comm_semantic, on=["community_id", "comm_size"], how="left")

    comm_metrics = comm_metrics.sort_values(["comm_size"], ascending=False).reset_index(drop=True)

    # structural metrics (optional)
    try:
        import igraph  # noqa
        from osnclusters.metrics.structural import structural_metrics_igraph
        struct = structural_metrics_igraph(edges_final, comm_df)
    except Exception as e:
        struct = {
            "modularity": np.nan,
            "conductance_mean": np.nan,
            "conductance_median": np.nan,
            "intra_density_mean": np.nan,
            "intra_density_median": np.nan,
        }
        if logger:
            logger.warning(f"[C7] structural metrics skipped (igraph missing). Reason: {type(e).__name__}: {e}")

    # spatial baseline (with post_min_comm_size)
    seed = int(cfg.get("run", {}).get("seed", 42))
    n_rand = int(cfg.get("metrics", {}).get("random_baseline_runs", 10))
    post_min = int(cfg.get("metrics", {}).get("post_min_comm_size", 1))

    obs, rmean, rstd, z, _vals = spatial_baseline_zscore(
        comm_df,
        user_centroids,
        n_runs=n_rand,
        seed=seed,
        post_min_comm_size=post_min,
    )

    metrics_global = {
        "dataset": dataset_name,
        **comm_stats,
        **struct,
        "metrics_post_min_comm_size": int(post_min),
        "spatial_median_km_global": float(obs),
        "spatial_random_median_km_mean": float(rmean),
        "spatial_random_median_km_std": float(rstd),
        "spatial_zscore_vs_random": float(z),
        "sample_frac": float(sample_frac),
        "train_edge_frac": float(train_edge_frac),
        "preprocess_k_min_checkins": int(k),
        "preprocess_d_min_degree": int(d),
    }

    # add semantic summary (optional)
    if comm_semantic is not None and len(comm_semantic):
        # median entropy across communities (after post_min filter)
        metrics_global.update({
            "venue_entropy_median": float(pd.to_numeric(comm_semantic["venue_entropy"], errors="coerce").median()),
            "venue_entropy_norm_median": float(pd.to_numeric(comm_semantic["venue_entropy_norm"], errors="coerce").median()),
            "category_entropy_median": float(pd.to_numeric(comm_semantic["category_entropy"], errors="coerce").median()),
            "category_entropy_norm_median": float(pd.to_numeric(comm_semantic["category_entropy_norm"], errors="coerce").median()),
        })

    if logger:
        logger.info(f"[C7] global metrics: {metrics_global}")

    # Save artifacts
    paths = make_artifact_paths(run_dir, dataset_name)
    save_df(edges_clean, paths["edges_clean"])
    save_df(checkins_clean, paths["checkins_clean"])
    save_df(users_final, paths["users_final"])
    save_df(edges_final, paths["edges_final"])
    save_df(checkins_final, paths["checkins_final"])
    save_df(feat_df.reset_index(), paths["feat_df"])
    save_npy(X_users, paths["X_users"])
    save_npy(Z, paths["Z"])
    save_df(comm_df, paths["comm_df"])
    save_df(comm_metrics, paths["comm_metrics"])  # NOTE: now merged spatial+semantic
    save_json(metrics_global, paths["metrics_global"])

    return RunResult(dataset=dataset_name, comm_df=comm_df, comm_metrics_df=comm_metrics, metrics_global=metrics_global)
