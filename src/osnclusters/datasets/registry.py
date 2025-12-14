from __future__ import annotations

from typing import Dict, Any
from pathlib import Path

from osnclusters.core.types import DatasetSpec, DatasetName, TierName, SnapshotName


def build_dataset_registry(cfg: Dict[str, Any], data_dir: Path) -> Dict[DatasetName, DatasetSpec]:
    brightkite_root = data_dir / "Brightkite"
    gowalla_root = data_dir / "Gowalla"
    lbsn_root = data_dir / "LBSN2Vec"

    tier: TierName = cfg["datasets"].get("lbsn2vec_tier", "curated")
    snapshot: SnapshotName = cfg["datasets"].get("lbsn2vec_snapshot", "old")

    return {
        "brightkite": DatasetSpec(
            name="brightkite",
            root=brightkite_root,
            edges_path=brightkite_root / "Brightkite_edges.txt",
            checkins_path=brightkite_root / "Brightkite_totalCheckins.txt",
            source="SNAP",
        ),
        "gowalla": DatasetSpec(
            name="gowalla",
            root=gowalla_root,
            edges_path=gowalla_root / "Gowalla_edges.txt",
            checkins_path=gowalla_root / "Gowalla_totalCheckins.txt",
            source="SNAP",
        ),
        "lbsn2vec": DatasetSpec(
            name="lbsn2vec",
            root=lbsn_root,
            friendship_old_path=lbsn_root / "dataset_WWW_friendship_old.txt",
            friendship_new_path=lbsn_root / "dataset_WWW_friendship_new.txt",
            readme_path=lbsn_root / "dataset_WWW_readme.txt",
            checkins_path=lbsn_root / "dataset_WWW_Checkins_anonymized.txt",
            raw_checkins_path=lbsn_root / "raw_Checkins_anonymized.txt",
            poi_path=lbsn_root / "raw_POIs.txt",
            source="LBSN2Vec",
            tier=tier,
            snapshot=snapshot,
        ),
    }
