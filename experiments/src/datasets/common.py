"""Common types + path resolution for dataset loaders."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..geometry import Camera

# experiments/src/datasets/common.py -> experiments/
EXPERIMENTS_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = EXPERIMENTS_ROOT.parent
DATASETS_ROOT = EXPERIMENTS_ROOT / "datasets"


def rel(path: Path) -> str:
    """Repository-relative posix path, for manifests and logs."""
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


@dataclass
class Sample:
    """One (frame, camera, hand) observation ready for reprojection.

    joints2d_gt is None when the dataset ships no independent 2D
    annotation; in that case only geometric sanity (depth, in-image)
    can be validated, never a reprojection error.
    """

    dataset: str
    subset: str
    sequence: str
    camera_name: str
    frame: str
    hand: str
    camera: Camera
    joints3d_world: np.ndarray
    joints2d_gt: np.ndarray | None = None
    valid: np.ndarray | None = None
    image_path: Path | None = None
    joints2d_source: str = "annotation"
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.joints3d_world = np.asarray(self.joints3d_world, dtype=float).reshape(-1, 3)
        n = len(self.joints3d_world)
        if self.joints2d_gt is not None:
            self.joints2d_gt = np.asarray(self.joints2d_gt, dtype=float).reshape(-1, 2)
        if self.valid is None:
            self.valid = np.ones(n, dtype=bool)
        else:
            self.valid = np.asarray(self.valid).reshape(-1).astype(bool)
