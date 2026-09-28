"""Per-frame translation errors under the ORACLE FOCAL, plus cached residuals.

For every frame of the CAM-EXP-009.4 common set:
  * T_pred  — WiLoR cam_t with tz rescaled to the reference focal
  * T_ref   — the reference root in target-camera coordinates
  * R_j     — root-relative pose residual, kp3_j - (Xc_j - T_ref), FIXED

`R` is cached so that any later oracle correction can be scored without
re-running the reconstruction: absolute MPJPE under a corrected translation `T`
is `mean_j ||R_j + (T - T_ref)||`.

The reference focal is used here as an ORACLE DIAGNOSTIC input, to remove focal
error so the remaining translation error can be studied. It is not a deployable
input.

Reference 3D is OTHER_CAMERA_ONLY: the target camera is excluded before
reconstruction, and its focal is never used for the world->camera transform.
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from common import (C013, LOCO_MIN_INLIER, LOCO_THRESHOLD_PX,  # noqa: E402
                    MANIFESTS, MIN_OK_JOINTS, RAW, REPO, SCENE_SRC,
                    SOURCE_ARTIFACTS, SUM, WILOR, fnum, pipeline_focal,
                    read_csv, scalar_focal, split_of, write_csv, write_json)


def reference_focal_map():
    """Reference focal per (sequence, camera) — ORACLE DIAGNOSTIC input."""
    out = {}
    for r in read_csv(SCENE_SRC):
        k = (r["sequence"], r["camera"])
        if k in out:
            continue
        fx, fy = fnum(r.get("gt_fx")), fnum(r.get("gt_fy"))
        if np.isfinite(fx) and np.isfinite(fy):
            out[k] = scalar_focal(fx, fy)
    return out


def main():
    for p in (str(REPO), str(C013 / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import loco
    from experiments.src.datasets.gigahands import takes

    common = {(r["sequence"], r["camera"])
              for r in read_csv(SOURCE_ARTIFACTS["cam0094_common_set"])}
    split_src = defaultdict(list)
    for r in read_csv(MANIFESTS / "cam_exp_0094_hand_split_v1.csv.gz"):
        k = (r["sequence"], r["camera"])
        if k in common:
            split_src[k].append(r)

    units = {(u["sequence"], u["camera"]): u
             for u in read_csv(MANIFESTS / "cam_exp_0094_units_v1.csv.gz")}
    ref_focal = reference_focal_map()
    take_by_name = {t.name: t for t in takes()}

    rows, Rs, keys = [], [], []
    leak = 0
    t0 = time.time()
    ordered = sorted(common)
    for i, (seq, cam) in enumerate(ordered, 1):
        take = take_by_name.get(seq)
        if take is None:
            continue
        camera = take.cameras.get(cam)
        f_ref = ref_focal.get((seq, cam))
        if camera is None or f_ref is None:
            continue
        Rmat = np.asarray(camera.R, float)
        tvec = np.asarray(camera.t, float).reshape(3)
        u = units.get((seq, cam), {})

        for r in split_src[(seq, cam)]:
            fr, hand = int(r["frame"]), r["hand"]
            p = WILOR / ("%s__%s__%06d.npz" % (seq, cam, fr))
            if not p.exists():
                continue
            d = np.load(p)
            W, H = int(d["image_width"]), int(d["image_height"])
            want_right = hand == "right"
            best, bs = None, -1.0
            for j in range(int(d["n_hands"])):
                if bool(int(d["h%d_is_right" % j])) != want_right:
                    continue
                s = float(d["h%d_score" % j])
                if s > bs:
                    best, bs = j, s
            if best is None:
                continue
            kp3 = np.asarray(d["h%d_keypoints_3d" % best], float)
            cam_t = np.asarray(d["h%d_cam_t" % best], float)
            bbox = np.asarray(d["h%d_bbox" % best], float)
            f_pipe = pipeline_focal(W, H)

            hyp = loco.reconstruct(take, hand, fr, exclude_camera=cam,
                                   threshold_px=LOCO_THRESHOLD_PX,
                                   min_inlier_cameras=LOCO_MIN_INLIER)
            if not hyp.ok:
                continue
            leak += int(cam in (hyp.inlier_cameras or []))
            ok3 = np.asarray(hyp.ok_joints, bool)
            if ok3.sum() < MIN_OK_JOINTS or not ok3[0]:
                continue
            Xw = np.asarray(hyp.points, float)
            Xc = (Rmat @ Xw.T).T + tvec           # extrinsics only

            T_ref = Xc[0].copy()
            tz = cam_t[2] * (f_ref / f_pipe)      # ORACLE FOCAL
            t = np.array([cam_t[0], cam_t[1], tz])
            # WiLoR's keypoints_3d are NOT root-centred: kp3[0] is about 96 mm
            # from the local origin (almost all in x, the left/right mirror
            # offset). The predicted absolute root is therefore kp3[0] + t,
            # which is what CAM-EXP-009.4 used. Verified by the reproduction
            # gate, which caught the earlier wrong assumption.
            T_pred = kp3[0] + t
            R = (kp3 - kp3[0]) - (Xc - Xc[0])     # fixed pose residual
            e = T_pred - T_ref

            bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            extent = float(np.max(np.linalg.norm(
                kp3[ok3] - kp3[0], axis=1))) if ok3.sum() > 1 else np.nan
            rows.append({
                "sequence": seq, "camera": cam, "hand": hand, "frame": fr,
                "participant": u.get("participant", ""),
                "outer_fold": u.get("outer_fold", ""),
                "split": split_of(seq, cam, hand, fr),
                "f_ref": f_ref, "f_pipeline": f_pipe,
                "x_pred": T_pred[0], "y_pred": T_pred[1], "z_pred": T_pred[2],
                "x_ref": T_ref[0], "y_ref": T_ref[1], "z_ref": T_ref[2],
                "dx_mm": e[0] * 1000.0, "dy_mm": e[1] * 1000.0,
                "dz_mm": e[2] * 1000.0,
                "root_error_mm": float(np.linalg.norm(e)) * 1000.0,
                "xy_error_mm": float(np.hypot(e[0], e[1])) * 1000.0,
                "depth_error_mm": float(abs(e[2])) * 1000.0,
                "log_z_ratio": (float(np.log(T_ref[2] / T_pred[2]))
                                if T_pred[2] > 0 and T_ref[2] > 0 else np.nan),
                "z_ratio": (float(T_ref[2] / T_pred[2])
                            if T_pred[2] > 0 else np.nan),
                "n_ok_joints": int(ok3.sum()),
                "bbox_w": bw, "bbox_h": bh,
                "bbox_frac": float(bw * bh) / float(W * H),
                "cam_t_z": float(cam_t[2]),
                "hand_extent_m": extent,
                "score": bs,
                "inlier_cameras": hyp.n_inlier_cameras,
            })
            Rs.append(np.where(ok3[:, None], R, np.nan).astype(np.float32))
            keys.append("%s|%s|%s|%d" % (seq, cam, hand, fr))
        if i % 20 == 0 or i == len(ordered):
            print("  %d/%d units  %d frames  %.1f min"
                  % (i, len(ordered), len(rows), (time.time() - t0) / 60),
                  flush=True)

    write_csv(RAW / "frame_translation_errors.csv.gz", rows)
    np.savez_compressed(RAW / "pose_residuals.npz",
                        R=np.stack(Rs) if Rs else np.zeros((0, 21, 3)),
                        keys=np.array(keys))
    write_json(SUM / "extraction_audit.json", {
        "frames": len(rows),
        "units": len(set((r["sequence"], r["camera"]) for r in rows)),
        "target_camera_in_reference_count": leak,
        "circularity_clean": leak == 0,
        "reference_focal_role": "ORACLE DIAGNOSTIC input, not deployable",
        "world_to_camera_transform": "extrinsics R,t only; target focal not "
                                     "used in the transform",
    })
    print("frames %d, circularity leaks %d, %.1f min"
          % (len(rows), leak, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
