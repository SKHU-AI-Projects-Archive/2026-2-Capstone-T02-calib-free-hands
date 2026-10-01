"""PHASE C — label-free per-unit candidate score curves. REFERENCE FOCAL CLOSED.

Produces, for every eligible sequence-camera unit:
  * the scene-only focal estimate and a robust scene scale
  * the hand score curve over the candidate grid, for each method variant

Nothing here reads the reference focal: scene predictions arrive through
`read_scene_predictions()`, which drops every `gt_*` column at load.

Method variants (the hand branch differs only in how anatomy is parameterised):
  M1_SCENE_GENERIC_INDEPENDENT  generic prior, LEFT/RIGHT independent, no p_seq
  M2_SCENE_SHARED_NO_GENERIC    shared p_seq, uniform base instead of MANO
  M3_SCENE_SHARED_GENERIC_FULL  shared p_seq + generic prior + side deviations
M0_SCENE_ONLY needs no hand pass.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from common import (CACHE, EPS, MANIFESTS, MIN_FRAMES_PER_SIDE,  # noqa: E402
                    MIN_OK_JOINTS, N_SCENE_FRAMES, RAW, SIGMA_SCENE_FLOOR,
                    bone_unit_directions, fnum, finger_permutation, q_grid,
                    read_csv, read_json, read_scene_predictions, scalar_focal,
                    write_csv)
from joint_anatomy_solver import fit_unit, score_frames  # noqa: E402

WILOR = CACHE / "wilor"

VARIANTS = {
    "M1_SCENE_GENERIC_INDEPENDENT": dict(shared=False, use_generic=True),
    "M2_SCENE_SHARED_NO_GENERIC": dict(shared=True, use_generic=False),
    "M3_SCENE_SHARED_GENERIC_FULL": dict(shared=True, use_generic=True),
}


def scene_estimates(n_set=N_SCENE_FRAMES):
    """Per-unit scene focal and robust log-scale. No reference focal read."""
    frames = read_csv(MANIFESTS / "cam_exp_0094_scene_frames_v1.csv.gz")
    want = defaultdict(set)
    for r in frames:
        if int(r["n_set"]) == n_set:
            want[(r["sequence"], r["camera"])].add(int(r["frame"]))
    preds = read_scene_predictions()
    by = defaultdict(list)
    for r in preds:
        k = (r["sequence"], r["camera"])
        if k in want and int(r["frame"]) in want[k] \
                and r.get("success") in ("1", "True", "true"):
            f = scalar_focal(fnum(r["pred_fx"]), fnum(r["pred_fy"]))
            if np.isfinite(f) and f > 0:
                by[k].append((f, fnum(r["pred_cx"]), fnum(r["pred_cy"])))
    out = {}
    for k, v in by.items():
        f = np.array([x[0] for x in v])
        lf = np.log(f)
        med = float(np.median(lf))
        mad = float(np.median(np.abs(lf - med))) * 1.4826
        out[k] = {"f_scene": float(np.exp(med)),
                  "sigma_scene": max(mad, SIGMA_SCENE_FLOOR),
                  "n_scene_frames": len(v),
                  "cx": float(np.median([x[1] for x in v])),
                  "cy": float(np.median([x[2] for x in v]))}
    return out


def load_hand_frames(seq, cam, split_rows, mapping=None):
    """Unit bone directions + 2D observations from the cached WiLoR output.

    Bone LENGTHS from the network are discarded; only unit directions are kept.
    """
    out = {("left", "FIT"): [], ("left", "EVAL"): [],
           ("right", "FIT"): [], ("right", "EVAL"): []}
    W = H = None
    for r in split_rows:
        f = int(r["frame"])
        p = WILOR / ("%s__%s__%06d.npz" % (seq, cam, f))
        if not p.exists():
            continue
        d = np.load(p)
        W, H = int(d["image_width"]), int(d["image_height"])
        want_right = r["hand"] == "right"
        best, best_score = None, -1.0
        for j in range(int(d["n_hands"])):
            if bool(int(d["h%d_is_right" % j])) != want_right:
                continue
            s = float(d["h%d_score" % j])
            if s > best_score:
                best, best_score = j, s
        if best is None:
            continue
        kp3 = np.asarray(d["h%d_keypoints_3d" % best], float)
        uv = np.asarray(d["h%d_keypoints_2d" % best], float)
        dirs, ok = bone_unit_directions(kp3)
        use = np.isfinite(uv).all(1)
        use[0] = use[0] and True
        if not ok.all() or use.sum() < MIN_OK_JOINTS:
            continue
        if mapping is not None and want_right:
            dirs = dirs[np.asarray(mapping, int)]
        out[(r["hand"], r["split"])].append((dirs, uv, use))
    return out, W, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control-wrong-bone", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--methods", default="",
                    help="comma-separated subset of VARIANTS to run")
    args = ap.parse_args()
    variants = (VARIANTS if not args.methods
                else {k: v for k, v in VARIANTS.items()
                      if k in args.methods.split(",")})

    prior = read_json(MANIFESTS / "cam_exp_0094_generic_mano_prior_v1.json")
    p0 = np.array(prior["p0_generic_prior"], float)
    spec = read_json(MANIFESTS / "cam_exp_0094_synthetic_spec_v1.json")
    lam_g, lam_s = spec["lambda_generic"], spec["lambda_side"]
    mapping = finger_permutation() if args.control_wrong_bone else None

    units = [u for u in read_csv(MANIFESTS / "cam_exp_0094_units_v1.csv.gz")
             if u["eligible"] == "1"]
    if args.limit:
        units = units[:args.limit]
    split = defaultdict(list)
    for r in read_csv(MANIFESTS / "cam_exp_0094_hand_split_v1.csv.gz"):
        split[(r["sequence"], r["camera"])].append(r)

    scene = scene_estimates()
    qs = q_grid()
    rows, t0 = [], time.time()
    for i, u in enumerate(units, 1):
        seq, cam = u["sequence"], u["camera"]
        sc = scene.get((seq, cam))
        if sc is None:
            continue
        frames, W, H = load_hand_frames(seq, cam, split[(seq, cam)], mapping)
        nL = len(frames[("left", "FIT")]) + len(frames[("left", "EVAL")])
        nR = len(frames[("right", "FIT")]) + len(frames[("right", "EVAL")])
        if (len(frames[("left", "FIT")]) < 2
                or len(frames[("right", "FIT")]) < 2
                or len(frames[("left", "EVAL")]) < 2
                or len(frames[("right", "EVAL")]) < 2):
            rows.append({"sequence": seq, "camera": cam,
                         "participant": u["participant"],
                         "outer_fold": u["outer_fold"],
                         "status": "insufficient_wilor_frames",
                         "n_left": nL, "n_right": nR,
                         "f_scene": sc["f_scene"],
                         "sigma_scene": sc["sigma_scene"]})
            continue
        diag = float(np.hypot(W, H))
        dist = None          # deployment-compatible: no GT distortion
        for name, cfg in variants.items():
            curve = np.full(len(qs), np.nan)
            anat = {}
            for k, q in enumerate(qs):
                f = sc["f_scene"] * q
                K = np.array([[f, 0, sc["cx"]], [0, f, sc["cy"]],
                              [0, 0, 1.0]])
                fit = fit_unit(frames[("left", "FIT")],
                               frames[("right", "FIT")], K, dist, p0,
                               lam_g, lam_s, **cfg)
                if fit is None:
                    continue
                sL = score_frames(frames[("left", "EVAL")], K, dist,
                                  fit["p_L"], diag)
                sR = score_frames(frames[("right", "EVAL")], K, dist,
                                  fit["p_R"], diag)
                if np.isfinite(sL) and np.isfinite(sR):
                    curve[k] = 0.5 * (sL + sR)
                    anat[k] = fit
            ok = np.isfinite(curve)
            if ok.sum() < 5:
                rows.append({"sequence": seq, "camera": cam,
                             "participant": u["participant"],
                             "outer_fold": u["outer_fold"], "method": name,
                             "status": "hand_score_failed"})
                continue
            idx = np.flatnonzero(ok)
            j = idx[int(np.argmin(curve[idx]))]
            fit = anat.get(j, {})
            rec = {"sequence": seq, "camera": cam,
                   "participant": u["participant"],
                   "outer_fold": u["outer_fold"], "method": name,
                   "status": "ok", "n_left": nL, "n_right": nR,
                   "f_scene": sc["f_scene"], "sigma_scene": sc["sigma_scene"],
                   "n_scene_frames": sc["n_scene_frames"],
                   "q_hand_min": float(qs[j]),
                   "f_hand_only": float(sc["f_scene"] * qs[j]),
                   "hand_boundary": int(j == idx[0] or j == idx[-1]),
                   "hand_score_min": float(curve[j]),
                   "hand_score_range": float(np.nanmax(curve)
                                             - np.nanmin(curve)),
                   "norm_a": float(np.linalg.norm(fit.get("a", np.zeros(1)))),
                   "norm_delta_L": float(np.linalg.norm(
                       fit.get("delta_L", np.zeros(1)))),
                   "norm_delta_R": float(np.linalg.norm(
                       fit.get("delta_R", np.zeros(1))))}
            for k in range(len(qs)):
                rec["c%02d" % k] = float(curve[k]) if np.isfinite(curve[k]) \
                    else ""
            rows.append(rec)
        if i % 10 == 0 or i == len(units):
            print("  %d/%d units  %.1f min"
                  % (i, len(units), (time.time() - t0) / 60), flush=True)

    name = ("control_wrong_bone_curves.csv.gz" if args.control_wrong_bone
            else "candidate_score_curves.csv.gz")
    write_csv(RAW / name, rows)
    okn = sum(1 for r in rows if r.get("status") == "ok")
    print("wrote %d rows (%d ok) in %.1f min"
          % (len(rows), okn, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
