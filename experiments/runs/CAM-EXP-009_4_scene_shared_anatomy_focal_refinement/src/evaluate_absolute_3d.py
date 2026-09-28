"""PHASE H — absolute 3D under each method's focal.

The camera translation equation is the one audited in CAM-EXP-002 and used by
the demo pipeline:

    tz = 2 f / (s * B)      ->      tz(f) = cam_t[2] * (f / f_pipeline)

so only tz scales with the focal; tx and ty do not. The WiLoR pose, shape and
2D prediction are FIXED: every method consumes the same cached network output
and differs only in the focal, which is what makes a downstream difference
attributable to the camera estimate.

Reference 3D is `OTHER_CAMERA_ONLY_REFERENCE_3D`: the target camera is excluded
BEFORE reconstruction, so its own annotation cannot shape the hand it is scored
against. World reference points are mapped into the target camera frame using
that camera's rotation and translation ONLY - its focal is never used for the
geometric transform.
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from common import (C013, CACHE, GATE, MANIFESTS, RAW, REPO, SUM,  # noqa: E402
                    TAB, fnum, med, paired_cluster_bootstrap, read_csv,
                    write_csv, write_json)

WILOR = CACHE / "wilor"
LOCO_THRESHOLD_PX = 8.0
LOCO_MIN_INLIER = 4
PIPELINE_FOCAL_RATIO = 1000.0 / 256.0      # FOCAL_LENGTH / IMAGE_SIZE


def pipeline_focal(W, H):
    return PIPELINE_FOCAL_RATIO * max(W, H)


def main():
    for p in (str(REPO), str(C013 / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import loco
    from experiments.src.datasets.gigahands import takes

    preds = read_csv(RAW / "frozen_test_predictions.csv")
    focal = defaultdict(dict)
    for r in preds:
        f = fnum(r["f_pred"])
        if np.isfinite(f):
            focal[(r["sequence"], r["camera"])][r["method"]] = f

    # oracle upper diagnostic
    from evaluate_focal import reference_focal_all
    ref_focal = reference_focal_all()
    for k, v in ref_focal.items():
        if k in focal:
            focal[k]["REF_FOCAL_ORACLE"] = v

    split = defaultdict(list)
    for r in read_csv(MANIFESTS / "cam_exp_0094_hand_split_v1.csv.gz"):
        split[(r["sequence"], r["camera"])].append(r)
    take_by_name = {t.name: t for t in takes()}

    rows, t0 = [], time.time()
    units = sorted(focal)
    leak = 0
    for i, (seq, cam) in enumerate(units, 1):
        take = take_by_name.get(seq)
        if take is None:
            continue
        cams = take.cameras
        camera = cams.get(cam) if isinstance(cams, dict) else None
        if camera is None:
            continue
        R = np.asarray(camera.R, float)
        T = np.asarray(camera.t, float).reshape(3)
        for r in split[(seq, cam)]:
            fr = int(r["frame"])
            hand = r["hand"]
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
            f_pipe = pipeline_focal(W, H)

            hyp = loco.reconstruct(take, hand, fr, exclude_camera=cam,
                                   threshold_px=LOCO_THRESHOLD_PX,
                                   min_inlier_cameras=LOCO_MIN_INLIER)
            if not hyp.ok:
                continue
            leak += int(cam in (hyp.inlier_cameras or []))
            ok3 = hyp.ok_joints
            if ok3.sum() < 12:
                continue
            Xw = np.asarray(hyp.points, float)
            Xc = (R @ Xw.T).T + T          # rotation + translation only

            for method, f in sorted(focal[(seq, cam)].items()):
                tz = cam_t[2] * (f / f_pipe)
                t = np.array([cam_t[0], cam_t[1], tz])
                Xp = kp3 + t
                dr = float(np.linalg.norm(Xp[0] - Xc[0]))
                dz = float(abs(Xp[0, 2] - Xc[0, 2]))
                m = ok3.copy()
                am = float(np.mean(np.linalg.norm(Xp[m] - Xc[m], axis=1)))
                ra = float(np.mean(np.linalg.norm(
                    (Xp[m] - Xp[0]) - (Xc[m] - Xc[0]), axis=1)))
                rows.append({
                    "sequence": seq, "camera": cam, "frame": fr,
                    "hand": hand, "method": method, "focal": f,
                    "wrist_root_error_mm": dr * 1000.0,
                    "root_depth_error_mm": dz * 1000.0,
                    "absolute_mpjpe_mm": am * 1000.0,
                    "root_aligned_mpjpe_mm": ra * 1000.0,
                })
        if i % 10 == 0 or i == len(units):
            print("  %d/%d units  %.1f min"
                  % (i, len(units), (time.time() - t0) / 60), flush=True)

    write_csv(RAW / "absolute3d_frame_results.csv.gz", rows)

    METRICS = ["wrist_root_error_mm", "root_depth_error_mm",
               "absolute_mpjpe_mm", "root_aligned_mpjpe_mm"]
    summ = []
    for method in sorted(set(r["method"] for r in rows)):
        for hand in ("left", "right", "BIMANUAL_COMBINED"):
            sub = [r for r in rows if r["method"] == method
                   and (hand == "BIMANUAL_COMBINED" or r["hand"] == hand)]
            rec = {"method": method, "hand": hand, "n_frames": len(sub)}
            for mt in METRICS:
                rec["median_" + mt] = med([r[mt] for r in sub])
            summ.append(rec)
    write_csv(SUM / "absolute3d_summary.csv", summ)
    write_csv(TAB / "absolute3d_main_table.csv", summ)

    # paired per-unit comparison, cluster = physical camera
    def unit_med(method, metric, hand=None):
        by = defaultdict(list)
        for r in rows:
            if r["method"] != method:
                continue
            if hand and r["hand"] != hand:
                continue
            by[(r["sequence"], r["camera"])].append(r[metric])
        return {k: float(np.median(v)) for k, v in by.items()}

    paired, tags = {}, {}
    for metric in METRICS:
        a = unit_med("M0_SCENE_ONLY", metric)
        b = unit_med("M3_SCENE_SHARED_GENERIC_FULL", metric)
        common = sorted(set(a) & set(b))
        gains = [a[k] - b[k] for k in common]
        pt, lo, hi = paired_cluster_bootstrap(gains, [k[1] for k in common])
        m0, m3 = med([a[k] for k in common]), med([b[k] for k in common])
        paired[metric] = {
            "n_units": len(common), "M0_median": m0, "M3_median": m3,
            "relative_reduction_pct": (100.0 * (m0 - m3) / m0
                                       if m0 else np.nan),
            "paired_median_gain_mm": pt, "ci95": [lo, hi]}
    write_json(SUM / "absolute3d_paired_summary.json", paired)

    root = paired["wrist_root_error_mm"]
    mpj = paired["absolute_mpjpe_mm"]
    ra = paired["root_aligned_mpjpe_mm"]
    abs3d_tag = bool(
        root["relative_reduction_pct"] >= GATE["abs3d_root_reduction_pct"]
        and mpj["relative_reduction_pct"] >= GATE["abs3d_mpjpe_reduction_pct"]
        and abs(ra["M3_median"] - ra["M0_median"])
        <= GATE["root_aligned_tolerance_mm"])
    review = bool(abs(ra["M3_median"] - ra["M0_median"])
                  > GATE["root_aligned_tolerance_mm"])

    write_json(SUM / "absolute3d_verdict.json", {
        "CAM0094_ABSOLUTE_3D_IMPROVED": abs3d_tag,
        "DOWNSTREAM_IMPLEMENTATION_REVIEW": review,
        "root_aligned_median_change_mm": ra["M3_median"] - ra["M0_median"],
        "paired": paired,
        "target_camera_in_reference_count": leak,
        "circularity_clean": leak == 0,
        "camera_equation": "tz(f) = cam_t[2] * f / f_pipeline; tx, ty "
                           "unchanged (CAM-EXP-002 audit)",
    })

    for s in summ:
        if s["hand"] == "BIMANUAL_COMBINED":
            print("%-32s root %7.1f  depth %7.1f  absMPJPE %7.1f  "
                  "rootMPJPE %6.2f"
                  % (s["method"], s["median_wrist_root_error_mm"],
                     s["median_root_depth_error_mm"],
                     s["median_absolute_mpjpe_mm"],
                     s["median_root_aligned_mpjpe_mm"]))
    print("circularity leaks:", leak)
    print("ABSOLUTE_3D_IMPROVED:", abs3d_tag, "| review:", review)


if __name__ == "__main__":
    main()
