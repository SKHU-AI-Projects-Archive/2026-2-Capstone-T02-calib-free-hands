"""Build the OTHER_CAMERA_ONLY_REFERENCE_3D cache for CAM-EXP-006.

For every (sequence, camera, frame, hand) on the frozen 16-frame grid, the hand
is reconstructed from the dataset-provided 2D of every OTHER calibrated camera
in that sequence. The camera under test is excluded BEFORE reconstruction, so
its own annotation cannot shape the reference hand it is later tested against.

This writes a cache only. It computes no focal, sees no GT focal, and runs no
solver.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CACHE, LOCO_MIN_INLIER_CAMERAS, LOCO_THRESHOLD_PX,  # noqa: E402
                    MANIFESTS, RAW, read_csv, write_csv, write_json)

FRAMES = MANIFESTS / "cam_exp_006_reference_hand_frames_v1.csv.gz"
MIN_OK_JOINTS = 12
NPZ = CACHE / "reference_3d_v1.npz"
AUDIT = RAW / "circularity_audit.csv"


def main() -> None:
    from common import loco_src_on_path
    loco_src_on_path()
    import loco
    from experiments.src.datasets.gigahands import is_zero_2d, takes

    rows = read_csv(FRAMES)
    by_view = defaultdict(list)
    for r in rows:
        by_view[(r["sequence"], r["camera"])].append(int(r["frame"]))

    take_by_name = {t.name: t for t in takes()}
    CACHE.mkdir(parents=True, exist_ok=True)

    store, audit = {}, []
    for vi, ((seq, cam), frames) in enumerate(sorted(by_view.items()), 1):
        take = take_by_name[seq]
        n_all = len(take.cameras)
        for frame in sorted(frames):
            for hand in ("left", "right"):
                hyp = loco.reconstruct(take, hand, frame, exclude_camera=cam,
                                       threshold_px=LOCO_THRESHOLD_PX,
                                       min_inlier_cameras=LOCO_MIN_INLIER_CAMERAS)
                g2d = take.joints2d(hand, cam, frame)
                if g2d is None:
                    uv = np.full((21, 2), np.nan)
                    conf = np.zeros(21)
                else:
                    g = np.asarray(g2d, float)
                    uv = g[:, :2].copy()
                    conf = g[:, 2] if g.shape[1] > 2 else np.ones(21)
                    uv[is_zero_2d(g)] = np.nan

                ok3 = hyp.ok_joints if hyp.ok else np.zeros(21, bool)
                pts = hyp.points if hyp.ok else np.full((21, 3), np.nan)
                usable = ok3 & np.isfinite(uv).all(1) & (conf >= 0.5)
                status = hyp.status
                if hyp.ok and usable.sum() < MIN_OK_JOINTS:
                    status = "too_few_paired_joints"

                key = f"{seq}|{cam}|{frame}|{hand}"
                store[key + "|xyz"] = pts.astype(np.float32)
                store[key + "|uv"] = uv.astype(np.float32)
                store[key + "|use"] = usable
                audit.append({
                    "sequence": seq, "camera": cam, "frame": frame,
                    "hand": hand,
                    "cameras_in_sequence": n_all,
                    "camera_under_test_excluded": 1,
                    "cameras_available_to_reconstruction": hyp.n_available_cameras,
                    "inlier_cameras": hyp.n_inlier_cameras,
                    "camera_under_test_in_inliers": int(
                        cam in (hyp.inlier_cameras or [])),
                    "loco_status": hyp.status,
                    "n_paired_joints": int(usable.sum()),
                    "usable_for_solver": int(status == "ok"),
                    "status": status,
                    "gt_focal_used": 0,
                    "gt_extrinsics_used_in_solver_input": 0,
                    "provided_3d_used": 0,
                })
        if vi % 25 == 0:
            print(f"  {vi}/{len(by_view)} views", flush=True)

    np.savez_compressed(NPZ, **store)
    write_csv(AUDIT, audit)

    n = len(audit)
    ok = sum(a["usable_for_solver"] for a in audit)
    leak = sum(a["camera_under_test_in_inliers"] for a in audit)
    joints = np.array([a["n_paired_joints"] for a in audit])
    write_json(RAW.parent / "summary" / "reference_3d_build.json", {
        "cases": n, "usable_for_solver": ok,
        "usable_share": round(ok / n, 4),
        "median_paired_joints_when_usable": float(
            np.median(joints[joints >= MIN_OK_JOINTS])) if ok else None,
        "camera_under_test_ever_an_inlier": leak,
        "non_circularity": "PASS" if leak == 0 else "VIOLATION",
        "min_ok_joints_required": MIN_OK_JOINTS,
        "cache": str(NPZ.name),
    })
    print(f"{n} cases, {ok} usable ({ok / n:.1%}); "
          f"camera-under-test leakage into inliers: {leak}")


if __name__ == "__main__":
    main()
