from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Dict, Any

DatasetName = Literal["brightkite", "gowalla", "lbsn2vec"]
TierName = Literal["curated", "raw"]
SnapshotName = Literal["old", "new"]

@dataclass(frozen=True)
class DatasetSpec:
    name: DatasetName
    root: Path
    edges_path: Optional[Path] = None
    checkins_path: Optional[Path] = None

    friendship_old_path: Optional[Path] = None
    friendship_new_path: Optional[Path] = None
    readme_path: Optional[Path] = None
    poi_path: Optional[Path] = None
    raw_checkins_path: Optional[Path] = None

    source: Literal["SNAP", "LBSN2Vec"] = "SNAP"
    tier: Optional[TierName] = None
    snapshot: Optional[SnapshotName] = None


def relpath(p: Path, root: Path) -> str:
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except Exception:
        return str(p)
