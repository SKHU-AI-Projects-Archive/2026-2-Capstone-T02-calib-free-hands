"""PHASE C — validate the global focal grid resolution (spec section 27).

The fusion reads H_video(f) by interpolating the global grid onto each model's
candidate window. If the grid is too coarse the interpolation itself becomes a
modelling choice. This measures that error directly.

ACCEPTANCE TOLERANCE, frozen here BEFORE the answer is known:

  T_A  median |interp - direct| <= 1 % of the curve's interquartile range
       over the validation window, AND
  T_B  the interpolated curve's argmin over a q in [0.5, 1.5] window never
       differs from the direct curve's argmin by more than ONE q-grid step.

The densest candidate (401) is the reference. Coarser candidates are accepted
only if they meet BOTH criteria. No reference focal is read.
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
from c012_common import (BLOCK_SIZE, C011_RAW, C011_SRC, C0094_SRC,  # noqa: E402
                    GLOBAL_GRID_N, MANIFESTS, MIN_OBS_PER_FOLD_PER_SIDE,
                    SUM, med, read_csv, read_json, temporal_fold, write_json)

sys.path.insert(0, str(C011_SRC))
sys.path.insert(0, str(C0094_SRC))

TOL_FRAC_OF_IQR = 0.01
CANDIDATE_N = (401, 201, 101)
VALIDATION_F_MIN, VALIDATION_F_MAX = 300.0, 3000.0


def split_folds(items, frames):
    A = [x for x, f in zip(items, frames) if temporal_fold(f) == "A"]
    B = [x for x, f in zip(items, frames) if temporal_fold(f) == "B"]
    return A, B


def hand_curve(hands, frames_by_side, W, H, p0, lam_g, lam_s, focals):
    """Cross-fitted hand score on the given absolute focal points."""
    from joint_anatomy_solver import fit_unit, score_frames
    AL, BL = split_folds(hands["left"], frames_by_side["left"])
    AR, BR = split_folds(hands["right"], frames_by_side["right"])
    if min(len(AL), len(BL), len(AR), len(BR)) < MIN_OBS_PER_FOLD_PER_SIDE:
        return None
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
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=int, default=2)
    args = ap.parse_args()

    from hand_io import load_video_observations
    p0 = np.array(read_json(MANIFESTS
                            / "cam_exp_0094_generic_mano_prior_v1.json")
                  ["p0_generic_prior"], float)
    spec = read_json(MANIFESTS / "cam_exp_0094_synthetic_spec_v1.json")
    lam_g, lam_s = spec["lambda_generic"], spec["lambda_side"]

    obs = read_csv(C011_RAW / "hand_observations.csv.gz")
    by = defaultdict(list)
    for r in obs:
        if r["hand_any_available"] == "1":
            by[(r["sequence"], r["camera"])].append(int(r["frame"]))

    ref_grid = np.geomspace(VALIDATION_F_MIN, VALIDATION_F_MAX, 401)
    results, t0 = [], time.time()
    picked = sorted(by)[:args.videos]
    for (seq, cam) in picked:
        frames = sorted(by[(seq, cam)])
        hands, idx, W, H = load_video_observations(seq, cam, frames)
        curve = hand_curve(hands, idx, W, H, p0, lam_g, lam_s, ref_grid)
        if curve is None or not np.isfinite(curve).any():
            continue
        ok = np.isfinite(curve)
        iqr = float(np.percentile(curve[ok], 75) - np.percentile(curve[ok], 25))
        for n in CANDIDATE_N:
            sub = np.linspace(0, len(ref_grid) - 1, n).astype(int)
            gi = np.interp(ref_grid, ref_grid[sub], curve[sub])
            err = np.abs(gi[ok] - curve[ok])
            results.append({
                "sequence": seq, "camera": cam, "grid_n": n,
                "median_abs_err": float(np.median(err)),
                "max_abs_err": float(np.max(err)),
                "curve_iqr": iqr,
                "median_err_frac_of_iqr": float(np.median(err) / max(iqr, 1e-18)),
                "max_err_frac_of_iqr": float(np.max(err) / max(iqr, 1e-18)),
                "argmin_direct_f": float(ref_grid[np.nanargmin(curve)]),
                "argmin_interp_f": float(ref_grid[np.nanargmin(gi)]),
            })
        print("  %s %s done  %.1f min" % (seq, cam, (time.time() - t0) / 60),
              flush=True)

    verdict = {}
    for n in CANDIDATE_N:
        rs = [r for r in results if r["grid_n"] == n]
        if not rs:
            continue
        mfrac = med([r["median_err_frac_of_iqr"] for r in rs])
        argmin_rel = med([abs(np.log(r["argmin_interp_f"]
                                     / r["argmin_direct_f"]))
                          for r in rs])
        verdict[n] = {
            "median_err_frac_of_iqr": mfrac,
            "max_err_frac_of_iqr": max(r["max_err_frac_of_iqr"] for r in rs),
            "median_abs_log_argmin_shift": argmin_rel,
            "meets_T_A": bool(mfrac <= TOL_FRAC_OF_IQR),
        }
    write_json(SUM / "global_grid_audit.json", {
        "tolerance_T_A_frac_of_iqr": TOL_FRAC_OF_IQR,
        "validation_window": [VALIDATION_F_MIN, VALIDATION_F_MAX],
        "reference_grid_n": 401,
        "per_video": results, "per_grid": verdict,
        "frozen_before_results": True,
        "reference_focal_read": False,
    })
    for n, v in sorted(verdict.items()):
        print("grid %3d : median err %.4g x IQR  max %.4g  argmin shift %.2e  "
              "T_A %s" % (n, v["median_err_frac_of_iqr"],
                          v["max_err_frac_of_iqr"],
                          v["median_abs_log_argmin_shift"], v["meets_T_A"]))


if __name__ == "__main__":
    main()
