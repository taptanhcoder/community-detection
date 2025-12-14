from __future__ import annotations

import argparse
from pathlib import Path

from osnclusters.core.config import load_config, find_repo_root
from osnclusters.core.logging import setup_logging
from osnclusters.datasets.registry import build_dataset_registry
from osnclusters.datasets.validate import validate_dataset_registry_and_load


def _summarize(df_val, logger):
    logger.info("=== Validation Summary ===")
    for _, r in df_val.iterrows():
        name = r["dataset"]
        src = r["source"]
        ok_root = bool(r["root_exists"])
        if not ok_root:
            logger.info(f"[FAIL] {name} - missing root folder")
            continue
        if src == "SNAP":
            edges_ok = bool(r.get("edges_ok", False))
            chk_ok = bool(r.get("checkins_ok", False))
            logger.info(f"{'[OK] ' if (edges_ok and chk_ok) else '[FAIL]'} {name} (SNAP) | edges_ok={edges_ok} | checkins_ok={chk_ok} | layout={r.get('checkins_best_layout')}")
            if not (edges_ok and chk_ok):
                logger.info(f"  issues: {r.get('issues')}")
        else:
            old_ok = bool(r.get("friendship_old_ok", False))
            new_ok = bool(r.get("friendship_new_ok", False))
            chk_ok = bool(r.get("checkins_ok", False))
            need_join = bool(r.get("checkins_needs_poi_join", False))
            logger.info(f"{'[OK] ' if (old_ok and new_ok and chk_ok) else '[FAIL]'} {name} (LBSN2Vec) | old_ok={old_ok} new_ok={new_ok} checkins_ok={chk_ok} need_POI_join={need_join} | layout={r.get('checkins_best_layout')}")
            if not (old_ok and new_ok and chk_ok):
                logger.info(f"  issues: {r.get('issues')}")
    logger.info("==========================")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="Path to configs/default.yaml (optional)")
    ap.add_argument("--sample-rows", type=int, default=100)
    ap.add_argument("--preview-lines", type=int, default=2)
    args = ap.parse_args()

    project_root = find_repo_root()
    cfg = load_config(Path(args.config) if args.config else None, project_root=project_root)

    logger = setup_logging(cfg.get("run", {}).get("log_level", "INFO"))
    data_dir = (project_root / cfg["project"]["data_dir"]).resolve()

    datasets = build_dataset_registry(cfg, data_dir=data_dir)
    active = cfg["datasets"].get("active", list(datasets.keys()))

    df_val = validate_dataset_registry_and_load(
        datasets=datasets,
        active=active,
        sample_rows=int(args.sample_rows),
        preview_lines=int(args.preview_lines),
    )

    # print as table-like
    logger.info(f"Validation rows={len(df_val)}")
    _summarize(df_val, logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
