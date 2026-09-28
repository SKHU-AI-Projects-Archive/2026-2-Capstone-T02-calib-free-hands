"""PHASE B step 1 — per-frame bone vectors from OTHER_CAMERA_ONLY_REFERENCE_3D.

For each frozen (sequence, camera, hand, frame), the hand is reconstructed from
the dataset-provided 2D of every OTHER calibrated camera in that sequence. The
camera under test is excluded BEFORE reconstruction, so its own annotation
cannot shape the geometry attributed to it.

CAM-EXP-001.3's `loco.reconstruct` is imported unchanged. No focal is computed
and none is read.
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (LOCO_MIN_INLIER_CAMERAS, LOCO_THRESHOLD_PX,  # noqa: E402
                    MANIFESTS, MIN_OK_JOINTS, N_BONES, RAW, SUM, bone_lengths,
                    log_rep, loco_on_path, proportions, read_csv, write_csv,
                    write_json)


def main():
    loco_on_path()
    import loco
    from experiments.src.datasets.gigahands import takes

    split = read_csv(MANIFESTS / "cam_exp_0093_repeatability_split_v1.csv.gz")
    take_by_name = {t.name: t for t in takes()}

    # group so each (sequence, camera, hand) is handled together
    work = defaultdict(list)
    for r in split:
        work[(r["sequence"], r["camera"], r["hand"])].append(
            (int(r["frame"]), r["half"]))

    rows, audit = [], []
    t0 = time.time()
    done = 0
    for i, ((seq, cam, hand), frames) in enumerate(sorted(work.items()), 1):
        take = take_by_name[seq]
        for frame, half in sorted(frames):
            hyp = loco.reconstruct(take, hand, frame, exclude_camera=cam,
                                   threshold_px=LOCO_THRESHOLD_PX,
                                   min_inlier_cameras=LOCO_MIN_INLIER_CAMERAS)
            done += 1
            ok = bool(hyp.ok) and int(hyp.ok_joints.sum()) >= MIN_OK_JOINTS
            in_inliers = int(cam in (hyp.inlier_cameras or []))
            audit.append({
                "sequence": seq, "camera": cam, "hand": hand, "frame": frame,
                "reconstruction_ok": int(ok),
                "target_camera_in_reference": in_inliers,
                "inlier_cameras": hyp.n_inlier_cameras,
                "status": hyp.status,
            })
            if not ok:
                continue
            l = bone_lengths(hyp.points)
            if not np.isfinite(l).all() or l.min() <= 0:
                continue
            z = log_rep(proportions(l))
            rec = {"sequence": seq, "camera": cam, "hand": hand,
                   "frame": frame, "half": half,
                   "total_length_m": float(l.sum())}
            for b in range(N_BONES):
                rec["z%02d" % b] = float(z[b])
            rows.append(rec)
        if i % 25 == 0 or i == len(work):
            print("%3d/%d units  %d recon  %.1f min"
                  % (i, len(work), done, (time.time() - t0) / 60), flush=True)

    write_csv(RAW / "frame_bone_vectors.csv.gz", rows)
    write_csv(RAW / "reference_circularity_audit.csv.gz", audit)
    leak = sum(a["target_camera_in_reference"] for a in audit)
    write_json(SUM / "reference_geometry_audit.json", {
        "reconstructions_attempted": len(audit),
        "reconstructions_ok": sum(a["reconstruction_ok"] for a in audit),
        "frame_vectors_written": len(rows),
        "target_camera_in_reference_count": leak,
        "circularity_clean": leak == 0,
        "note": "target_camera_in_reference_count must be 0: the camera under "
                "test is excluded before reconstruction, so it can never "
                "appear among the inlier cameras of its own reference hand.",
    })
    print("frame vectors %d, circularity leaks %d" % (len(rows), leak))


if __name__ == "__main__":
    main()
