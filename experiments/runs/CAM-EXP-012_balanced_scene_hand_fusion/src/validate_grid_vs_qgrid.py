"""PHASE C — validate the global grid against the operation the fusion uses.

The fusion never reads the global grid directly: it interpolates H onto each
model's q-grid mapped to absolute focals. So the honest test is to compute H
DIRECTLY at those q-grid focals and compare against the interpolated value.

The production global grid spans the union of every model's candidate window,
which is wide (GeoCalib produces extreme estimates), so its local step is
COARSER than the q-grid step. This measures whether that matters.

ACCEPTANCE TOLERANCE, frozen before the answer is known:
  median |interp - direct| <= 1 % of the curve's interquartile range over the
  q-window, and the interpolated argmin within the q-window must not differ
  from the direct argmin by more than one q-grid step.

No reference focal is read.
"""
from __future__ import annotations

import argparse, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from c012_common import (C011_RAW, C0094_SRC, MANIFESTS, MIN_OBS_PER_FOLD_PER_SIDE,
                         SUM, med, read_csv, read_json, temporal_fold, write_json,
                         q_grid, fnum)
from hand_io import load_video_observations
sys.path.insert(0, str(C0094_SRC))

TOL_FRAC_OF_IQR = 0.01


def split_folds(items, frames):
    A = [x for x, f in zip(items, frames) if temporal_fold(f) == "A"]
    B = [x for x, f in zip(items, frames) if temporal_fold(f) == "B"]
    return A, B


def curve_at(hands, idx, W, H, p0, lg, ls, focals):
    from joint_anatomy_solver import fit_unit, score_frames
    AL, BL = split_folds(hands["left"], idx["left"])
    AR, BR = split_folds(hands["right"], idx["right"])
    if min(len(AL), len(BL), len(AR), len(BR)) < MIN_OBS_PER_FOLD_PER_SIDE:
        return None
    diag = float(np.hypot(W, H))
    out = np.full(len(focals), np.nan)
    for i, f in enumerate(focals):
        K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])
        vals, wts = [], []
        for fL, fR, eL, eR in ((AL, AR, BL, BR), (BL, BR, AL, AR)):
            fit = fit_unit(fL, fR, K, None, p0, lg, ls, shared=False,
                           use_generic=True)
            if fit is None:
                continue
            sL = score_frames(eL, K, None, fit["p_L"], diag)
            sR = score_frames(eR, K, None, fit["p_R"], diag)
            if np.isfinite(sL): vals.append(sL); wts.append(len(eL))
            if np.isfinite(sR): vals.append(sR); wts.append(len(eR))
        if vals:
            out[i] = float(np.average(vals, weights=wts))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--videos", type=int, default=1)
    args = ap.parse_args()
    p0 = np.array(read_json(MANIFESTS / "cam_exp_0094_generic_mano_prior_v1.json")
                  ["p0_generic_prior"], float)
    spec = read_json(MANIFESTS / "cam_exp_0094_synthetic_spec_v1.json")
    lg, ls = spec["lambda_generic"], spec["lambda_side"]
    G = np.array(read_json(MANIFESTS / "cam_exp_012_global_focal_grid_v1.json")["grid"])
    qs = q_grid()
    prim = read_csv(MANIFESTS / "cam_exp_012_primary_video_set_v1.csv")
    pv = read_csv(Path(SRC).parents[0] / "tables" / "primary_video_set.csv")
    fs_by = {(r["sequence"], r["camera"]): fnum(r["f_scene_ANYCALIB"]) for r in pv}

    obs = read_csv(C011_RAW / "hand_observations.csv.gz")
    by = defaultdict(list)
    for r in obs:
        if r["hand_any_available"] == "1":
            by[(r["sequence"], r["camera"])].append(int(r["frame"]))

    res, t0 = [], time.time()
    for r in prim[:args.videos]:
        k = (r["sequence"], r["camera"])
        f_scene = fs_by.get(k)
        if not f_scene or not np.isfinite(f_scene):
            continue
        frames = sorted(by[k])
        hands, idx, W, H = load_video_observations(k[0], k[1], frames)
        q_focals = f_scene * qs
        direct = curve_at(hands, idx, W, H, p0, lg, ls, q_focals)
        if direct is None: continue
        print("  direct q-grid done %.1f min" % ((time.time()-t0)/60), flush=True)
        # only the global-grid points inside a margin of the q window are needed
        m = (G >= q_focals.min()*0.98) & (G <= q_focals.max()*1.02)
        gsub = G[m]
        gcurve = curve_at(hands, idx, W, H, p0, lg, ls, gsub)
        print("  global-grid subset done %.1f min" % ((time.time()-t0)/60), flush=True)
        ok = np.isfinite(direct)
        interp = np.interp(q_focals, gsub[np.isfinite(gcurve)],
                           gcurve[np.isfinite(gcurve)])
        err = np.abs(interp[ok] - direct[ok])
        iqr = float(np.percentile(direct[ok],75)-np.percentile(direct[ok],25))
        jd, ji = int(np.nanargmin(direct)), int(np.nanargmin(interp))
        res.append({"sequence": k[0], "camera": k[1], "f_scene": f_scene,
            "n_global_points_in_window": int(m.sum()),
            "median_abs_err": float(np.median(err)), "max_abs_err": float(np.max(err)),
            "curve_iqr": iqr,
            "median_err_frac_of_iqr": float(np.median(err)/max(iqr,1e-18)),
            "max_err_frac_of_iqr": float(np.max(err)/max(iqr,1e-18)),
            "argmin_q_direct": float(qs[jd]), "argmin_q_interp": float(qs[ji]),
            "argmin_step_diff": abs(jd-ji)})
    verdict = {"tolerance_frac_of_iqr": TOL_FRAC_OF_IQR, "per_video": res,
        "median_err_frac_of_iqr": med([x["median_err_frac_of_iqr"] for x in res]),
        "max_argmin_step_diff": max([x["argmin_step_diff"] for x in res]) if res else None,
        "meets_tolerance": bool(res and med([x["median_err_frac_of_iqr"] for x in res]) <= TOL_FRAC_OF_IQR
                                and max(x["argmin_step_diff"] for x in res) <= 1),
        "frozen_before_results": True, "reference_focal_read": False}
    write_json(SUM / "global_grid_audit.json", verdict)
    for x in res:
        print("%s %s  median err %.3g x IQR  max %.3g  argmin step diff %d  (%d global pts in window)"
              % (x["sequence"], x["camera"], x["median_err_frac_of_iqr"],
                 x["max_err_frac_of_iqr"], x["argmin_step_diff"], x["n_global_points_in_window"]))
    print("MEETS TOLERANCE:", verdict["meets_tolerance"])


if __name__ == "__main__":
    main()
