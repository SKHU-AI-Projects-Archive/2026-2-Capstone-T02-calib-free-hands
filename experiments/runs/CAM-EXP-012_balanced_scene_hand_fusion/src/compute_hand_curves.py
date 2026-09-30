"""PHASE D/E — cross-fitted hand score curves on the GLOBAL absolute focal grid.

Computed once per video and shared by all three scene models, so the three
`+ Hand` conditions read identical hand evidence and differ only in the scene
term.

Cross-fitting (spec section 20):
    PASS 1   fit on fold A observations, score held-out fold B
    PASS 2   fit on fold B observations, score held-out fold A
    H(f)     held-out-count-weighted mean of the two passes

Every usable observation therefore contributes exactly once as fitting data and
exactly once as held-out evaluation data. Folds are temporal blocks of 8
frames, so near-duplicate adjacent poses cannot straddle the boundary.

`--variant wrong_bone` permutes the RIGHT hand's bone order using
CAM-EXP-009.4's frozen within-finger permutation. Everything else — RGB,
observations, folds, prior, solver, grid — is identical to the correct variant.

No reference focal is read.
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
from c012_common import (C011_RAW, C0094_SRC, CACHE, MANIFESTS,  # noqa: E402
                         MIN_OBS_PER_FOLD_PER_SIDE, RAW, SUM,
                         finger_permutation, read_csv, read_json,
                         temporal_fold, write_csv, write_json)
from hand_io import load_video_observations  # noqa: E402

sys.path.insert(0, str(C0094_SRC))

VARIANTS = {"correct": None, "wrong_bone": "perm"}


def split_folds(items, frames):
    A = [x for x, f in zip(items, frames) if temporal_fold(f) == "A"]
    B = [x for x, f in zip(items, frames) if temporal_fold(f) == "B"]
    return A, B


def cross_fitted_curve(hands, idx, W, H, p0, lam_g, lam_s, focals):
    """-> (curve, diagnostics) or (None, reason)."""
    from joint_anatomy_solver import fit_unit, score_frames
    AL, BL = split_folds(hands["left"], idx["left"])
    AR, BR = split_folds(hands["right"], idx["right"])
    if min(len(AL), len(BL), len(AR), len(BR)) < MIN_OBS_PER_FOLD_PER_SIDE:
        return None, "insufficient_fold_coverage"
    diag = float(np.hypot(W, H))
    out = np.full(len(focals), np.nan)
    for i, f in enumerate(focals):
        K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])
        vals, wts = [], []
        for fit_L, fit_R, ev_L, ev_R in ((AL, AR, BL, BR), (BL, BR, AL, AR)):
            fit = fit_unit(fit_L, fit_R, K, None, p0, lam_g, lam_s,
                           shared=False, use_generic=True)
            if fit is None:
                continue
            sL = score_frames(ev_L, K, None, fit["p_L"], diag)
            sR = score_frames(ev_R, K, None, fit["p_R"], diag)
            if np.isfinite(sL):
                vals.append(sL)
                wts.append(len(ev_L))
            if np.isfinite(sR):
                vals.append(sR)
                wts.append(len(ev_R))
        if vals:
            out[i] = float(np.average(vals, weights=wts))
    return out, {"n_A_left": len(AL), "n_B_left": len(BL),
                 "n_A_right": len(AR), "n_B_right": len(BR)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), default="correct")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    out_dir = CACHE / ("%s_hand_curves" % args.variant)
    out_dir.mkdir(parents=True, exist_ok=True)

    p0 = np.array(read_json(MANIFESTS
                            / "cam_exp_0094_generic_mano_prior_v1.json")
                  ["p0_generic_prior"], float)
    spec = read_json(MANIFESTS / "cam_exp_0094_synthetic_spec_v1.json")
    lam_g, lam_s = spec["lambda_generic"], spec["lambda_side"]
    grid = np.array(read_json(MANIFESTS
                              / "cam_exp_012_global_focal_grid_v1.json")
                    ["grid"], float)
    mapping = finger_permutation() if args.variant == "wrong_bone" else None

    prim = read_csv(MANIFESTS / "cam_exp_012_primary_video_set_v1.csv")
    obs = read_csv(C011_RAW / "hand_observations.csv.gz")
    frames_by = defaultdict(list)
    for r in obs:
        if r["hand_any_available"] == "1":
            frames_by[(r["sequence"], r["camera"])].append(int(r["frame"]))

    videos = [(r["sequence"], r["camera"]) for r in prim]
    if args.nshards > 1:
        videos = [v for i, v in enumerate(videos)
                  if i % args.nshards == args.shard]

    rows, t0 = [], time.time()
    for vi, (seq, cam) in enumerate(videos, 1):
        outp = out_dir / ("%s__%s.npz" % (seq, cam))
        if outp.exists():
            continue
        frames = sorted(frames_by[(seq, cam)])
        hands, idx, W, H = load_video_observations(seq, cam, frames,
                                                   bone_mapping=mapping)
        curve, diag = cross_fitted_curve(hands, idx, W, H, p0, lam_g, lam_s,
                                         grid)
        if curve is None:
            rows.append({"sequence": seq, "camera": cam, "ok": 0,
                         "reason": diag})
            continue
        np.savez_compressed(outp, f=grid, score=curve, **{
            k: np.int32(v) for k, v in diag.items()})
        rows.append({"sequence": seq, "camera": cam, "ok": 1, "reason": "",
                     "n_valid": int(np.isfinite(curve).sum()), **diag})
        if vi % 2 == 0 or vi == len(videos):
            el = (time.time() - t0) / 60
            print("  %3d/%d  %.1f min  (%.1f min/video)"
                  % (vi, len(videos), el, el / max(vi, 1)), flush=True)

    idxp = (RAW / ("%s_curve_index_shard%02d.csv.gz"
                   % (args.variant, args.shard)) if args.nshards > 1
            else RAW / ("%s_curve_index.csv.gz" % args.variant))
    write_csv(idxp, rows)
    print("variant %s: %d videos written" % (args.variant, len(rows)))


if __name__ == "__main__":
    main()
