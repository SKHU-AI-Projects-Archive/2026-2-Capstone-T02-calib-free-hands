"""Shared paths and cluster-bootstrap machinery for CAM-EXP-004.1."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

RUN_DIR = Path(__file__).resolve().parents[1]
RUNS = RUN_DIR.parent
EXP = RUNS.parent
REPO = EXP.parent
MANIFESTS = EXP / "manifests"

CAM003 = RUNS / "CAM-EXP-003_single_frame_calibration_benchmark"
CAM0031 = RUNS / "CAM-EXP-003_1_distortion_aware_diagnostic"
CAM004 = RUNS / "CAM-EXP-004_static_camera_multiframe_aggregation"

FRAMES8_CSV = MANIFESTS / "gigahands_demo_cam_exp_003_frames_v1.csv.gz"
FRAMES64_CSV = MANIFESTS / "gigahands_demo_cam_exp_0041_64frames_v1.csv.gz"
E2_SPEC = MANIFESTS / "cam_exp_004_e2_frozen_spec_v1.json"

N_BOOT = 10000
SEED = 20260923

# The four resampling units, from the one CAM-EXP-003.1 actually used to the
# most conservative one the data structure allows.
CLUSTER_LEVELS = {
    "LEGACY_FRAME_BOOTSTRAP": None,          # what 003.1 reported; kept, not deleted
    "VIEW_CLUSTER": ("sequence", "camera"),  # 175 clusters - the correct default
    "PHYSICAL_CAMERA_CLUSTER": ("camera",),  # 40 clusters - same lens across takes
    "SEQUENCE_CLUSTER": ("sequence",),       # 5 clusters - sensitivity only
}


def read_csv(path: Path) -> list[dict]:
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(p)


def cluster_ids(rows: list[dict], level: str) -> np.ndarray:
    """Integer cluster label per row for the requested resampling unit."""
    spec = CLUSTER_LEVELS[level]
    if spec is None:
        return np.arange(len(rows))
    keys = [tuple(r[k] for k in spec) for r in rows]
    uniq = {k: i for i, k in enumerate(sorted(set(keys)))}
    return np.array([uniq[k] for k in keys])


def cluster_bootstrap(values: np.ndarray, clusters: np.ndarray,
                      stat=np.median, n_boot: int = N_BOOT, seed: int = SEED):
    """Resample whole clusters with replacement; every row of a drawn cluster
    enters the resample together.

    This is the fix for pseudo-replication: 8 frames of one static camera are
    not 8 independent observations of that camera's calibration error, so a
    frame-level bootstrap understates the uncertainty.
    """
    values = np.asarray(values, float)
    ok = np.isfinite(values)
    values, clusters = values[ok], np.asarray(clusters)[ok]
    uniq = np.unique(clusters)
    members = [np.flatnonzero(clusters == u) for u in uniq]
    rng = np.random.default_rng(seed)
    point = float(stat(values))
    draws = np.empty(n_boot)
    n_c = len(uniq)
    for b in range(n_boot):
        pick = rng.integers(0, n_c, size=n_c)
        idx = np.concatenate([members[p] for p in pick])
        draws[b] = stat(values[idx])
    return {
        "point_estimate": point,
        "ci_lo": float(np.percentile(draws, 2.5)),
        "ci_hi": float(np.percentile(draws, 97.5)),
        "n_rows": int(values.size),
        "n_clusters": int(n_c),
        "boot_sd": float(np.std(draws)),
    }
