from __future__ import annotations

import argparse
from pathlib import Path

from osnclusters.core.config import load_config, find_repo_root
from osnclusters.core.logging import setup_logging
from osnclusters.core.run_manager import create_run_context
from osnclusters.pipeline import run_one_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="Path to configs/default.yaml (optional)")
    ap.add_argument("--dataset", type=str, default=None, help="Dataset name (brightkite|gowalla|lbsn2vec). If omitted, run active list.")
    ap.add_argument("--sample-frac", type=float, default=None, help="Fraction of users to sample (default 1.0 or cfg).")
    ap.add_argument("--train-edge-frac", type=float, default=None, help="Fraction of edges for training (default cfg.train.train_edge_frac).")
    args = ap.parse_args()

    project_root = find_repo_root()
    cfg = load_config(Path(args.config) if args.config else None, project_root=project_root)
    logger = setup_logging(cfg.get("run", {}).get("log_level", "INFO"))

    # Resolve runtime overrides
    sample_frac = float(args.sample_frac) if args.sample_frac is not None else 1.0
    train_edge_frac = float(args.train_edge_frac) if args.train_edge_frac is not None else None

    ctx = create_run_context(cfg)
    logger.info(f"[RUN] project_root={ctx.project_root}")
    logger.info(f"[RUN] run_dir={ctx.run_dir}")

    if args.dataset:
        datasets_to_run = [args.dataset]
    else:
        datasets_to_run = list(cfg["datasets"].get("active", ["brightkite", "gowalla", "lbsn2vec"]))

    for ds in datasets_to_run:
        run_one_dataset(
            dataset_name=ds,  # type: ignore
            cfg=cfg,
            project_root=ctx.project_root,
            run_dir=ctx.run_dir,
            sample_frac=sample_frac,
            train_edge_frac=train_edge_frac,
            logger=logger,
        )

    logger.info("[DONE] All datasets finished.")
    logger.info(f"[DONE] Artifacts saved under: {ctx.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
