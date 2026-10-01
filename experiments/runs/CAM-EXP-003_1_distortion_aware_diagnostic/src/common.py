"""Shared paths and GT-undistortion for CAM-EXP-003.1.

The frame list, the PNG cache and the metric definitions are all inherited from
CAM-EXP-003 unchanged — this experiment varies only the camera model, so the
frames must be identical for the paired comparison to mean anything.
"""
from __future__ import annotations

import csv
import gzip
import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parents[1]
EXPERIMENTS = RUN_DIR.parents[1]
REPO = EXPERIMENTS.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CAM003 = EXPERIMENTS / "runs" / "CAM-EXP-003_single_frame_calibration_benchmark"
CAM003_SRC = CAM003 / "src"
if str(CAM003_SRC) not in sys.path:
    sys.path.insert(0, str(CAM003_SRC))

MANIFESTS = EXPERIMENTS / "manifests"
FRAMES_CSV = MANIFESTS / "gigahands_demo_cam_exp_003_frames_v1.csv.gz"
RAW_CACHE = CAM003 / "cache" / "frames"                  # reused, never rewritten
UNDIST_CACHE = RUN_DIR / "cache" / "frames_gt_undistorted"

# Undistortion setting, fixed and recorded so the comparison is unambiguous.
# alpha = 0 keeps the output at the original 1280x720 with no invalid black
# border (a border would itself change what a calibration model sees), and
# cv2.getOptimalNewCameraMatrix returns the exact K_new that the undistorted
# image corresponds to. The GT focal for condition C is that K_new, never the
# original K - comparing against the original would manufacture an improvement.
UNDISTORT_ALPHA = 0.0


def rel(p) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return p.as_posix()


def read_csv(path: Path) -> list:
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list, gzipped: bool | None = None) -> None:
    if not rows:
        return
    if gzipped is None:
        gzipped = str(path).endswith(".gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen and not str(k).startswith("_"):
                seen.add(k)
                keys.append(k)
    opener = ((lambda: gzip.open(path, "wt", newline="", encoding="utf-8")) if gzipped
              else (lambda: open(path, "w", newline="", encoding="utf-8")))
    with opener() as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"  wrote {rel(path)} ({len(rows)} rows)")


def load_frames() -> list:
    return read_csv(FRAMES_CSV)


def raw_frame_path(r) -> Path:
    return RAW_CACHE / f"{r['sequence']}__{r['camera']}__{int(r['frame']):06d}.png"


def undist_frame_path(r) -> Path:
    return UNDIST_CACHE / f"{r['sequence']}__{r['camera']}__{int(r['frame']):06d}.png"


def gt_K_D(r):
    """GT intrinsics and OpenCV distortion for one frame row."""
    import numpy as np
    K = np.array([[float(r["gt_fx"]), 0.0, float(r["gt_cx"])],
                  [0.0, float(r["gt_fy"]), float(r["gt_cy"])],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.array([float(r.get("gt_k1") or 0.0), float(r.get("gt_k2") or 0.0),
                  float(r.get("gt_p1") or 0.0), float(r.get("gt_p2") or 0.0)],
                 dtype=np.float64)
    return K, D


def new_camera_matrix(r):
    """K_new for the GT-undistorted image, at the original resolution."""
    import cv2
    K, D = gt_K_D(r)
    W, H = int(r["image_width"]), int(r["image_height"])
    K_new, roi = cv2.getOptimalNewCameraMatrix(K, D, (W, H), UNDISTORT_ALPHA, (W, H))
    return K_new, roi
