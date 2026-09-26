"""Build OTHER_CAMERA_ONLY_REFERENCE_3D for every candidate frame. TARGET-BLIND.

The camera under test is removed from the observation set BEFORE
reconstruction, so its own 2D cannot shape the reference hand it is later
tested against. Reuses CAM-EXP-001.3's frozen reconstruction unchanged.
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CACHE, LOCO_MIN_INLIER_CAMERAS, LOCO_THRESHOLD_PX,
                    MANIFESTS, MIN_PAIRED_JOINTS, RAW, loco_src_on_path,
                    read_csv, write_csv, write_json)

FRAMES = MANIFESTS / "cam_exp_008_candidate_frames_v1.csv.gz"
OUTDIR = CACHE / "reference_3d"


def main() -> None:
    loco_src_on_path()
    import loco
    from experiments.src.datasets.gigahands import is_zero_2d, takes

    rows = read_csv(FRAMES)
    by = defaultdict(list)
    for r in rows:
        by[(r["sequence"], r["camera"])].append(int(r["frame"]))
    views = sorted(by)
    take_by_name = {t.name: t for t in takes()}
    OUTDIR.mkdir(parents=True, exist_ok=True)

    audit = []
    for vi, (seq, cam) in enumerate(views, 1):
        out = OUTDIR / f"{seq}__{cam}.npz"
        if out.exists():
            continue
        take = take_by_name[seq]
        store, leak = {}, 0
        for fr in sorted(by[(seq, cam)]):
            for hand in ("left", "right"):
                hyp = loco.reconstruct(take, hand, fr, exclude_camera=cam,
                                       threshold_px=LOCO_THRESHOLD_PX,
                                       min_inlier_cameras=LOCO_MIN_INLIER_CAMERAS)
                g2d = take.joints2d(hand, cam, fr)
                if g2d is None:
                    continue
                g = np.asarray(g2d, float)
                uv = g[:, :2].copy()
                conf = g[:, 2] if g.shape[1] > 2 else np.ones(21)
                uv[is_zero_2d(g)] = np.nan
                ok3 = hyp.ok_joints if hyp.ok else np.zeros(21, bool)
                pts = hyp.points if hyp.ok else np.full((21, 3), np.nan)
                use = ok3 & np.isfinite(uv).all(1) & (conf >= 0.5)
                if cam in (hyp.inlier_cameras or []):
                    leak += 1
                if use.sum() < MIN_PAIRED_JOINTS:
                    continue
                k = f"{fr}|{hand}"
                store[k + "|xyz"] = pts.astype(np.float32)
                store[k + "|uv"] = uv.astype(np.float32)
                store[k + "|use"] = use
        np.savez_compressed(out, **store)
        audit.append({"sequence": seq, "camera": cam,
                      "hand_observations": len(store) // 3,
                      "camera_under_test_in_inliers": leak})
        if vi % 10 == 0:
            print(f"  {vi}/{len(views)} views", flush=True)

    if audit:
        write_csv(RAW / "reference_3d_build_audit.csv", audit)
        tot = sum(a["hand_observations"] for a in audit)
        leaks = sum(a["camera_under_test_in_inliers"] for a in audit)
        write_json(RAW.parent / "summary" / "reference_3d_meta.json", {
            "views": len(audit), "hand_observations": tot,
            "camera_under_test_ever_an_inlier": leaks,
            "non_circularity": "PASS" if leaks == 0 else "VIOLATION",
            "min_paired_joints": MIN_PAIRED_JOINTS, "target_blind": True,
        })
        print(f"views {len(audit)}, hand observations {tot}, leakage {leaks}")


if __name__ == "__main__":
    main()
