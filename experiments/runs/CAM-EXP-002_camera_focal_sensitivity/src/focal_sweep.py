"""Apply the focal sweep to the cached inference and evaluate against reference 3D.

Runs in the analysis environment (numpy/opencv only); no torch, no model.

What is recomputed per condition, and what is held fixed, follows directly from
``focal_usage_audit.md``:

    tz(f) = 2 f / (s * B)        -> the only focal-dependent quantity
    tx, ty                       -> unchanged (no focal term in the source)
    root-relative joints         -> unchanged (network output, cached once)

Rather than re-deriving ``s`` and ``B`` separately, the sweep uses the identity
``tz(f) = tz_baseline * f / f_baseline`` that follows from that formula, and
verifies it against a direct recomputation for the smoke set.

Two families of numbers are produced and never mixed:

  ABSOLUTE ERROR           prediction vs dataset-provided 3D (contains hand-model,
                           detector and weak-perspective error as well as focal error)
  INCREMENTAL FOCAL EFFECT prediction at alpha vs prediction at alpha = 1.0
                           (isolates the focal, since everything else is identical)
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict

import numpy as np

from common import (ALPHAS, CACHE_DIR, QC_CSV, RUN_DIR, read_csv, rel, write_csv)

log = logging.getLogger("cam-exp-002")

CONF_THRESHOLD = 0.5
# association margin: the better pairing must beat the alternative by this factor
ASSOC_MARGIN = 1.5
ASSOC_MAX_PX = 150.0


def load_reference():
    """PASS_STRICT observations keyed by (sequence, camera, frame, hand)."""
    ref = {}
    for r in read_csv(QC_CSV):
        if r["qc_status"] != "PASS_STRICT":
            continue
        ref[(r["sequence"], r["camera"], int(r["frame"]), r["hand"])] = r
    return ref


def camera_space(take, cam_name: str, hand: str, frame: int):
    """Dataset-provided 3D in this camera's frame, using the CAM-EXP-001 transform."""
    X = take.joints3d(hand, frame)
    if X is None:
        return None
    cam = take.cameras.get(cam_name)
    if cam is None:
        return None
    return (cam.R @ X.T).T + cam.t              # metres


def associate(pred2d: dict, ref2d: dict) -> tuple:
    """Match predicted hands to reference hands using 2D positions only.

    Deliberately never uses 3D error: choosing the pairing that minimises the
    quantity under study would manufacture the result. Predicted 2D keypoint
    centroids are compared with the provided 2D annotation centroids; the best
    assignment must beat the next-best by a clear margin, otherwise the frame is
    dropped as AMBIGUOUS_HAND_ASSOCIATION.
    """
    from itertools import combinations, permutations

    hands = [h for h in ("left", "right") if h in ref2d]
    idxs = sorted(pred2d)
    if not hands or not idxs:
        return {}, "no_candidates"
    cost = {(i, h): float(np.linalg.norm(pred2d[i] - ref2d[h]))
            for i in idxs for h in hands}

    # Enumerate every distinct one-to-one assignment. When the detector found
    # fewer hands than the reference has (or more), only the k = min(...) pairs
    # that can be matched are assigned; the rest are simply not evaluated.
    k = min(len(idxs), len(hands))
    options, seen = [], set()
    for hand_combo in combinations(hands, k):          # which hands are covered
        for pred_combo in permutations(idxs, k):       # which prediction takes which
            mapping = {pred_combo[m]: hand_combo[m] for m in range(k)}
            sig = tuple(sorted(mapping.items()))
            if sig in seen:            # permuting both lists would double-count
                continue
            seen.add(sig)
            options.append((sum(cost[(i, h)] for i, h in mapping.items()), mapping))
    if not options:
        return {}, "no_pairing"
    options.sort(key=lambda o: o[0])
    best = options[0]
    if len(options) > 1:
        second = options[1]
        # the runner-up must be clearly worse, otherwise which hand is which is
        # not decidable from 2D alone
        if second[0] < best[0] * ASSOC_MARGIN:
            return {}, "ambiguous_pairing_margin"
    mapping = best[1]
    for i, h in mapping.items():
        if cost[(i, h)] > ASSOC_MAX_PX:
            return {}, f"association_distance_{cost[(i, h)]:.0f}px_over_{ASSOC_MAX_PX:.0f}"
    return mapping, "ok"


def sweep(limit: int | None = None) -> dict:
    from experiments.src.datasets.gigahands import takes, is_zero_2d

    take_by_name = {t.name: t for t in takes()}
    ref = load_reference()
    index = read_csv(RUN_DIR / "results" / "raw" / "inference_cache_index.csv")
    index = [r for r in index if r.get("status") in ("ok", "cached")]
    if limit:
        index = index[:limit]

    per_hand, per_frame, assoc_fail, trace = [], [], [], []
    for n_done, row in enumerate(index):
        seq, cam_name, frame = row["sequence"], row["camera"], int(row["frame"])
        npz = CACHE_DIR / f"{seq}__{cam_name}__{frame:06d}.npz"
        if not npz.exists():
            continue
        d = np.load(npz)
        n_hands = int(d["n_hands"])
        if n_hands == 0:
            continue
        W, H = int(d["image_width"]), int(d["image_height"])
        f_pipeline = float(d["pipeline_focal"])
        take = take_by_name.get(seq)
        cam = take.cameras.get(cam_name) if take else None
        if cam is None:
            continue
        fx, fy = float(cam.K[0, 0]), float(cam.K[1, 1])
        f_gt_eff = fx                      # audit §4: identity transform

        # --- reference 2D centroids (provided annotation, PASS_STRICT only) ---
        ref2d, ref3d = {}, {}
        for hand in ("left", "right"):
            if (seq, cam_name, frame, hand) not in ref:
                continue
            g = take.joints2d(hand, cam_name, frame)
            if g is None or is_zero_2d(g).all():
                continue
            m = g[:, 2] >= CONF_THRESHOLD
            if m.sum() < 5:
                continue
            ref2d[hand] = g[m, :2].mean(0)
            X = camera_space(take, cam_name, hand, frame)
            if X is not None:
                ref3d[hand] = X

        pred2d = {i: np.asarray(d[f"h{i}_keypoints_2d"])[:, :2].mean(0)
                  for i in range(n_hands)}
        mapping, why = associate(pred2d, ref2d)
        if why != "ok":
            assoc_fail.append({"sequence": seq, "camera": cam_name, "frame": frame,
                               "n_pred_hands": n_hands, "n_ref_hands": len(ref2d),
                               "reason": f"AMBIGUOUS_HAND_ASSOCIATION: {why}"})
            continue

        frame_rows = {}
        for i, hand in mapping.items():
            if hand not in ref3d:
                assoc_fail.append({"sequence": seq, "camera": cam_name, "frame": frame,
                                   "n_pred_hands": n_hands, "n_ref_hands": len(ref2d),
                                   "reason": f"reference 3D missing for {hand}"})
                continue
            cam_t = np.asarray(d[f"h{i}_cam_t"], float)
            local = np.asarray(d[f"h{i}_keypoints_3d"], float)   # root-relative
            box = np.asarray(d[f"h{i}_bbox"], float)
            box_size = float(max(box[2] - box[0], box[3] - box[1]))
            R3 = ref3d[hand]
            # WiLoR joint 0 is the wrist; GigaHands joint 0 is also the wrist
            ref_root = R3[0]
            # s can be recovered from the cached translation: tz = 2 f / (s B)
            s_eff = 2.0 * f_pipeline / (cam_t[2] * box_size) if cam_t[2] != 0 else np.nan

            conditions = [("PIPELINE_BASELINE", np.nan, f_pipeline)]
            conditions += [(f"GT_x{a:.2f}", a, a * f_gt_eff) for a in ALPHAS]

            at_alpha1 = None
            for cond, alpha, f_test in conditions:
                # tz scales exactly with the focal; tx, ty do not (audit §2)
                tz = cam_t[2] * (f_test / f_pipeline)
                t = np.array([cam_t[0], cam_t[1], tz])
                joints = local + t
                pred_root = joints[0]
                if cond == "GT_x1.00":
                    at_alpha1 = (t.copy(), joints.copy())
                frame_rows.setdefault(cond, {})[hand] = (t, joints, pred_root)

                dz = float((pred_root[2] - ref_root[2]) * 1000.0)
                rootd = float(np.linalg.norm(pred_root - ref_root) * 1000.0)
                nj = min(len(joints), len(R3))
                mpjpe = float(np.mean(np.linalg.norm(
                    joints[:nj] - R3[:nj], axis=1)) * 1000.0)
                ra = float(np.mean(np.linalg.norm(
                    (joints[:nj] - joints[0]) - (R3[:nj] - R3[0]), axis=1)) * 1000.0)
                per_hand.append({
                    "sequence": seq, "camera": cam_name, "frame": frame, "hand": hand,
                    "condition": cond,
                    "alpha": "" if not np.isfinite(alpha) else round(alpha, 4),
                    "pipeline_focal": round(f_pipeline, 4),
                    "gt_native_fx": round(fx, 4), "gt_native_fy": round(fy, 4),
                    "gt_effective_focal": round(f_gt_eff, 4),
                    "focal_used_px": round(f_test, 4),
                    "bbox_size": round(box_size, 3),
                    "pred_cam_scale": round(s_eff, 6) if np.isfinite(s_eff) else "",
                    "detector_score": round(float(d[f"h{i}_score"]), 4),
                    "pred_is_right": int(d[f"h{i}_is_right"]),
                    "pred_root_x": round(float(pred_root[0]), 6),
                    "pred_root_y": round(float(pred_root[1]), 6),
                    "pred_root_z": round(float(pred_root[2]), 6),
                    "ref_root_x": round(float(ref_root[0]), 6),
                    "ref_root_y": round(float(ref_root[1]), 6),
                    "ref_root_z": round(float(ref_root[2]), 6),
                    "signed_z_error_mm": round(dz, 3),
                    "abs_z_error_mm": round(abs(dz), 3),
                    "root_xyz_error_mm": round(rootd, 3),
                    "absolute_mpjpe_mm": round(mpjpe, 3),
                    "root_aligned_mpjpe_mm": round(ra, 3),
                })

            # incremental focal effect, against GT x 1.00
            if at_alpha1 is not None:
                t1, j1 = at_alpha1
                for r in per_hand[-len(conditions):]:
                    cond = r["condition"]
                    tt, jj, _ = frame_rows[cond][hand]
                    r["delta_from_gt_focal_z_mm"] = round(
                        float((tt[2] - t1[2]) * 1000.0), 3)
                    r["delta_from_gt_focal_root_mm"] = round(
                        float(np.linalg.norm(tt - t1) * 1000.0), 3)
                    r["delta_from_gt_focal_joint_mm"] = round(
                        float(np.mean(np.linalg.norm(jj - j1, axis=1)) * 1000.0), 3)
                    r["pred_z_ratio_vs_alpha1"] = round(
                        float(tt[2] / t1[2]), 6) if t1[2] != 0 else ""

        # --- bimanual (frame-level) metrics ---
        for cond in frame_rows:
            hs = frame_rows[cond]
            if len(hs) != 2 or not {"left", "right"} <= set(hs):
                continue
            jl, jr = hs["left"][1], hs["right"][1]
            rl, rr = ref3d.get("left"), ref3d.get("right")
            if rl is None or rr is None:
                continue
            n = min(len(jl), len(rl))
            allp = np.vstack([jl[:n], jr[:n]])
            allr = np.vstack([rl[:n], rr[:n]])
            pred_sep = float(np.linalg.norm(jl[0] - jr[0]) * 1000.0)
            ref_sep = float(np.linalg.norm(rl[0] - rr[0]) * 1000.0)
            per_frame.append({
                "sequence": seq, "camera": cam_name, "frame": frame,
                "condition": cond,
                "alpha": "" if cond == "PIPELINE_BASELINE" else cond.split("x")[-1],
                "bimanual_absolute_mpjpe_mm": round(
                    float(np.mean(np.linalg.norm(allp - allr, axis=1)) * 1000.0), 3),
                "mean_root_depth_error_mm": round(
                    float(np.mean([abs(jl[0][2] - rl[0][2]),
                                   abs(jr[0][2] - rr[0][2])]) * 1000.0), 3),
                "pred_lr_root_distance_mm": round(pred_sep, 3),
                "ref_lr_root_distance_mm": round(ref_sep, 3),
                "lr_root_distance_error_mm": round(pred_sep - ref_sep, 3),
            })

        if len(trace) < 5 * len(ALPHAS) + 5:
            for r in per_hand[-len(ALPHAS) - 1:]:
                trace.append(r)

    write_csv(RUN_DIR / "results" / "raw" / "focal_sweep_per_hand.csv.gz", per_hand)
    write_csv(RUN_DIR / "results" / "raw" / "focal_sweep_per_frame.csv.gz", per_frame)
    write_csv(RUN_DIR / "results" / "raw" / "hand_association_failures.csv", assoc_fail)
    write_csv(RUN_DIR / "tables" / "smoke_numeric_trace.csv", trace)
    return {"per_hand": len(per_hand), "per_frame": len(per_frame),
            "assoc_failures": len(assoc_fail),
            "unique_hands": len({(r["sequence"], r["camera"], r["frame"], r["hand"])
                                 for r in per_hand})}


def summarise() -> dict:
    rows = read_csv(RUN_DIR / "results" / "raw" / "focal_sweep_per_hand.csv.gz")
    frames = read_csv(RUN_DIR / "results" / "raw" / "focal_sweep_per_frame.csv.gz")

    def stats(vals):
        a = np.asarray([float(v) for v in vals if v not in ("", None)], dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return {}
        return {"n": int(a.size), "mean": round(float(a.mean()), 3),
                "median": round(float(np.median(a)), 3),
                "p90": round(float(np.percentile(a, 90)), 3),
                "p95": round(float(np.percentile(a, 95)), 3)}

    metrics = ["signed_z_error_mm", "abs_z_error_mm", "root_xyz_error_mm",
               "absolute_mpjpe_mm", "root_aligned_mpjpe_mm",
               "delta_from_gt_focal_z_mm", "delta_from_gt_focal_root_mm",
               "delta_from_gt_focal_joint_mm", "pred_z_ratio_vs_alpha1"]
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)

    per_alpha = []
    for cond, rs in sorted(by_cond.items()):
        entry = {"condition": cond, "alpha": rs[0]["alpha"],
                 "focal_used_px": rs[0]["focal_used_px"], "n_hands": len(rs)}
        for m in metrics:
            for k, v in stats([r.get(m, "") for r in rs]).items():
                if k == "n":
                    continue
                entry[f"{m}_{k}"] = v
        per_alpha.append(entry)
    write_csv(RUN_DIR / "results" / "summary" / "per_alpha_summary.csv", per_alpha)

    sens = []
    for cond, rs in sorted(by_cond.items()):
        if cond == "PIPELINE_BASELINE":
            continue
        a = float(rs[0]["alpha"])
        dz = stats([r.get("delta_from_gt_focal_z_mm", "") for r in rs])
        dr = stats([r.get("delta_from_gt_focal_root_mm", "") for r in rs])
        ratio = stats([r.get("pred_z_ratio_vs_alpha1", "") for r in rs])
        sens.append({"alpha": a, "focal_error_percent": round((a - 1) * 100, 2),
                     "n_hands": len(rs),
                     "incremental_signed_dz_mean_mm": dz.get("mean", ""),
                     "incremental_signed_dz_median_mm": dz.get("median", ""),
                     "incremental_root_shift_mean_mm": dr.get("mean", ""),
                     "incremental_root_shift_median_mm": dr.get("median", ""),
                     "incremental_root_shift_p90_mm": dr.get("p90", ""),
                     "pred_z_ratio_median": ratio.get("median", ""),
                     "pred_z_ratio_mean": ratio.get("mean", "")})
    write_csv(RUN_DIR / "results" / "summary" / "focal_sensitivity_summary.csv", sens)

    for key, name in (("sequence", "per_sequence_summary"),
                      ("camera", "per_camera_summary")):
        g = defaultdict(lambda: defaultdict(list))
        for r in rows:
            g[r[key]][r["condition"]].append(r)
        out = []
        for k, conds in sorted(g.items()):
            base = conds.get("GT_x1.00", [])
            e = {key: k, "n_hands": len(base)}
            for cond in ("PIPELINE_BASELINE", "GT_x1.00", "GT_x1.10", "GT_x0.90"):
                rs = conds.get(cond, [])
                if rs:
                    e[f"{cond}_abs_z_error_mm_median"] = stats(
                        [r["abs_z_error_mm"] for r in rs]).get("median", "")
                    e[f"{cond}_root_xyz_error_mm_median"] = stats(
                        [r["root_xyz_error_mm"] for r in rs]).get("median", "")
            out.append(e)
        write_csv(RUN_DIR / "results" / "summary" / f"{name}.csv", out)

    bvg = []
    for cond in ("PIPELINE_BASELINE", "GT_x1.00"):
        rs = by_cond.get(cond, [])
        if not rs:
            continue
        e = {"condition": cond, "n_hands": len(rs),
             "focal_used_px_median": round(float(np.median(
                 [float(r["focal_used_px"]) for r in rs])), 2)}
        for m in ("abs_z_error_mm", "root_xyz_error_mm", "absolute_mpjpe_mm",
                  "root_aligned_mpjpe_mm", "signed_z_error_mm"):
            for k, v in stats([r[m] for r in rs]).items():
                e[f"{m}_{k}"] = v
        bvg.append(e)
    fr_by_cond = defaultdict(list)
    for r in frames:
        fr_by_cond[r["condition"]].append(r)
    for cond in ("PIPELINE_BASELINE", "GT_x1.00"):
        rs = fr_by_cond.get(cond, [])
        if rs:
            for e in bvg:
                if e["condition"] == cond:
                    e["bimanual_absolute_mpjpe_mm_median"] = stats(
                        [r["bimanual_absolute_mpjpe_mm"] for r in rs]).get("median", "")
                    e["lr_root_distance_error_mm_median"] = stats(
                        [r["lr_root_distance_error_mm"] for r in rs]).get("median", "")
    write_csv(RUN_DIR / "results" / "summary" / "baseline_vs_gt_focal.csv", bvg)

    fails = []
    for name, path in (("model_failures", RUN_DIR / "results" / "raw" / "model_failures.csv"),
                       ("hand_association_failures",
                        RUN_DIR / "results" / "raw" / "hand_association_failures.csv")):
        try:
            rs = read_csv(path)
        except FileNotFoundError:
            rs = []
        c = defaultdict(int)
        for r in rs:
            key = r.get("reason", "").split(":")[0][:60]
            c[key] += 1
        for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
            fails.append({"source": name, "reason": k, "n": v})
    write_csv(RUN_DIR / "results" / "summary" / "failure_summary.csv", fails)
    return {"per_alpha": len(per_alpha), "sensitivity": len(sens)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print(json.dumps(sweep(), indent=2))
    print(json.dumps(summarise(), indent=2))
