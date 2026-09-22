"""Shared paths, frame manifest and GT access for CAM-EXP-003.

The benchmark frame list is frozen BEFORE any model runs and every model sees
exactly the same frames, so no model can be helped by a favourable selection.
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

MANIFESTS = EXPERIMENTS / "manifests"
CAMERA_CSV = MANIFESTS / "gigahands_demo_camera_benchmark_v1.csv.gz"
FRAMES_CSV = MANIFESTS / "gigahands_demo_cam_exp_003_frames_v1.csv.gz"
EXTERNAL = EXPERIMENTS / "cache" / "external_models"

N_FRAMES_PER_VIEW = 8
SEED = 20260923

# Baselines that are not learned models
DEMO_FIXED_FOCAL_CFG = {"FOCAL_LENGTH": 1000.0, "IMAGE_SIZE": 256.0}


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


def demo_fixed_focal(width: int, height: int) -> float:
    """The existing pipeline's assumed focal (CAM-EXP-002 audit)."""
    return (DEMO_FIXED_FOCAL_CFG["FOCAL_LENGTH"]
            / DEMO_FIXED_FOCAL_CFG["IMAGE_SIZE"] * max(width, height))


def build_frame_manifest() -> list:
    """Freeze the benchmark frames: every usable view, 8 evenly spaced frames.

    Selection uses only the camera-benchmark manifest and the video length -
    never any calibration model output, and never hand-annotation quality
    (this is a camera benchmark, so a view with poor hand annotation is still
    a perfectly good RGB + GT camera sample).
    """
    import numpy as np
    from experiments.src.datasets.gigahands import takes

    take_by_name = {t.name: t for t in takes()}
    rows = []
    for r in read_csv(CAMERA_CSV):
        if r["usable_for_camera_benchmark"] != "1":
            continue
        take = take_by_name.get(r["sequence"])
        if take is None:
            continue
        cam_name = r["camera"]
        cam = take.cameras.get(cam_name)
        vid = take.video_path(cam_name)
        if cam is None or vid is None:
            continue
        # valid frame range: frames that exist in the annotated take
        n_valid = len(take.union_sorted)
        if n_valid <= 0:
            continue
        frames = take.union_sorted
        k = min(N_FRAMES_PER_VIEW, len(frames))
        idx = np.unique(np.linspace(0, len(frames) - 1, num=k).astype(int))
        for pos, i in enumerate(idx):
            f = int(frames[i])
            rows.append({
                "sequence": r["sequence"], "take": r["take"], "camera": cam_name,
                "frame": f, "slot": pos,
                "video": rel(vid),
                "image_width": int(cam.width or 0), "image_height": int(cam.height or 0),
                "gt_fx": round(float(cam.K[0, 0]), 6),
                "gt_fy": round(float(cam.K[1, 1]), 6),
                "gt_cx": round(float(cam.K[0, 2]), 6),
                "gt_cy": round(float(cam.K[1, 2]), 6),
                "gt_k1": round(float(cam.dist[0]), 8) if cam.dist.size > 0 else "",
                "gt_k2": round(float(cam.dist[1]), 8) if cam.dist.size > 1 else "",
                "gt_p1": round(float(cam.dist[2]), 8) if cam.dist.size > 2 else "",
                "gt_p2": round(float(cam.dist[3]), 8) if cam.dist.size > 3 else "",
            })
    return rows


def load_frames() -> list:
    return read_csv(FRAMES_CSV)


if __name__ == "__main__":
    rows = build_frame_manifest()
    write_csv(FRAMES_CSV, rows)
    import collections
    print("views:", len({(r["sequence"], r["camera"]) for r in rows}))
    print("frames:", len(rows))
    print("per sequence:", dict(collections.Counter(r["sequence"] for r in rows)))
