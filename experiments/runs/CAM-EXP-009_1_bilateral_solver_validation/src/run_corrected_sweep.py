"""Corrected synthetic sweep, DOF ablation and controls.

Runs only after G0-G4 pass. The solver configuration is frozen before this
executes; nothing here changes it.

Per candidate focal and side:
    l(q) fitted on FIT frames only
    FIT reprojection          - on the frames it was fitted to
    HELD-OUT reprojection     - the SAME l(q) scored on EVAL frames
    bilateral FIT distance    - ||l_L(q) - l_R(q)|| from the FIT fits
    bilateral EVAL distance   - from an independent refit on EVAL frames

The held-out score is the primary identifiability diagnostic, because 20 free
bone proportions can absorb a wrong focal on the frames they were fitted to.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DIST_REAL, F_TRUE, GATE, RAW, SUM, TAB, make_K, q_grid,  # noqa: E402
                    stable_seed, write_csv, write_json)
from corrected_bone_fitter import (bilateral_distance, fit_bones,  # noqa: E402
                                   score_fixed_shape)
from synthetic import make_sequence

GRID = q_grid()
# Reduced from CAM-EXP-009's 20 to 8 on MEASURED cost (38.4 s per trial ->
# 4.3 h at 20 trials, 1.7 h at 8). Decided before any sweep result was seen
# and applied uniformly to every condition, so no condition is given more
# power than another. The cost is lower precision on effect sizes; the
# boundary-rate and argmin-q findings are coarse enough to survive it.
N_TRIALS = 8
N_FRAMES = 12


def split(frames):
    """Deterministic alternating FIT / EVAL split, target-independent."""
    return frames[0::2], frames[1::2]


def profiles(seq, dist, dof="D20", mapping=None, swap_right=None):
    """Every profile for one trial, over the candidate grid."""
    Lf, Le = split(seq["left"])
    Rsrc = swap_right if swap_right is not None else seq["right"]
    Rf, Re = split(Rsrc)
    if min(len(Lf), len(Le), len(Rf), len(Re)) < 2:
        return None
    lt_L = seq["lL_true"] / seq["lL_true"].sum()
    out = {k: [] for k in ("fit", "held", "bilat_fit", "bilat_eval",
                           "oracle_shape", "bilat_fixed")}
    lfix_L = seq["lL_true"] / seq["lL_true"].sum()
    lfix_R = seq["lR_true"] / seq["lR_true"].sum()
    for q in GRID:
        K = make_K(q * F_TRUE)
        lL, iL, _ = fit_bones(Lf, K, dist, dof=dof)
        lR, iR, _ = fit_bones(Rf, K, dist, dof=dof)
        if lL is None or lR is None:
            for k in out:
                out[k].append(np.nan)
            continue
        hL, _ = score_fixed_shape(Le, K, dist, lL)
        hR, _ = score_fixed_shape(Re, K, dist, lR)
        lLe, _, _ = fit_bones(Le, K, dist, dof=dof)
        lRe, _, _ = fit_bones(Re, K, dist, dof=dof)
        oL, _ = score_fixed_shape(Le, K, dist, lt_L)
        b = lR if mapping is None else lR[mapping]
        be = None if lRe is None else (lRe if mapping is None
                                       else lRe[mapping])
        out["fit"].append(np.nanmean([iL, iR]))
        out["held"].append(np.nanmean([hL, hR]))
        out["bilat_fit"].append(bilateral_distance(lL, b))
        out["bilat_eval"].append(bilateral_distance(lLe, be))
        out["oracle_shape"].append(oL)
        out["bilat_fixed"].append(bilateral_distance(
            lfix_L, lfix_R if mapping is None else lfix_R[mapping]))
    return out


def argmin_q(v):
    v = np.asarray(v, float)
    if not np.isfinite(v).any():
        return np.nan, False
    i = int(np.nanargmin(v))
    return float(GRID[i]), (i in (0, len(v) - 1))


def summarise(trials, key):
    qs, bnd = [], []
    for t in trials:
        q, b = argmin_q(t[key])
        qs.append(q)
        bnd.append(b)
    qs = np.array(qs, float)
    e = 100.0 * np.abs(qs - 1.0)
    rng = [np.nanmax(t[key]) - np.nanmin(t[key]) for t in trials
           if np.isfinite(t[key]).any()]
    return {
        "median_recovered_q": round(float(np.nanmedian(qs)), 4),
        "median_focal_err_pct": round(float(np.nanmedian(e)), 3),
        "within_5pct": round(float(np.nanmean(e <= 5)), 3),
        "boundary_rate": round(float(np.mean(bnd)), 3),
        "profile_dynamic_range": round(float(np.nanmedian(rng)), 6)
        if rng else np.nan,
    }


def run_condition(name, n_trials=N_TRIALS, dof="D20", dist=None,
                  mapping=None, swap=False, **kw):
    dist = DIST_REAL if dist is None else dist
    trials = []
    for t in range(n_trials):
        rng = np.random.default_rng(stable_seed(name, t))
        seq = make_sequence(rng, n_frames=N_FRAMES, dist=dist, **kw)
        if len(seq["left"]) < 6 or len(seq["right"]) < 6:
            continue
        sw = None
        if swap:
            rng2 = np.random.default_rng(stable_seed(name + "_swap", t))
            s2 = make_sequence(rng2, n_frames=N_FRAMES, dist=dist, **kw)
            sw = s2["right"] if len(s2["right"]) >= 6 else None
            if sw is None:
                continue
        p = profiles(seq, dist, dof=dof, mapping=mapping, swap_right=sw)
        if p is not None:
            trials.append(p)
    return trials


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "sweep", "dof", "controls"])
    args = ap.parse_args()

    KEYS = ["fit", "held", "bilat_fit", "bilat_eval", "oracle_shape"]
    rows, curves = [], {}

    def record(tag, trials, **extra):
        if not trials:
            return
        r = {"condition": tag, "n_trials": len(trials), **extra}
        for k in KEYS:
            s = summarise(trials, k)
            for kk, vv in s.items():
                r[f"{k}_{kk}"] = vv
        fx = [np.nanmax(t["bilat_fixed"]) - np.nanmin(t["bilat_fixed"])
              for t in trials if np.isfinite(t["bilat_fixed"]).any()]
        r["bilat_fixed_range"] = float(np.nanmedian(fx)) if fx else np.nan
        rows.append(r)
        curves[tag + "__held"] = np.nanmedian(
            np.array([t["held"] for t in trials], float), axis=0)
        curves[tag + "__bilat"] = np.nanmedian(
            np.array([t["bilat_fit"] for t in trials], float), axis=0)
        curves[tag + "__oracle"] = np.nanmedian(
            np.array([t["oracle_shape"] for t in trials], float), axis=0)
        print(f"  {tag:22s} held q={r['held_median_recovered_q']:.3f} "
              f"({r['held_median_focal_err_pct']:5.2f}%) | "
              f"bilat q={r['bilat_fit_median_recovered_q']:.3f} "
              f"({r['bilat_fit_median_focal_err_pct']:5.2f}%) | "
              f"oracle q={r['oracle_shape_median_recovered_q']:.3f} "
              f"({r['oracle_shape_median_focal_err_pct']:5.2f}%)",
              flush=True)

    if args.stage in ("all", "sweep"):
        # STEP 1: no distortion first
        record("NODIST_d4", run_condition("NODIST_d4", dist=np.zeros(4),
                                          dist_over_diam=4), distortion="none")
        # STEP 2: realistic distortion, same everything else
        record("DIST_d4", run_condition("DIST_d4", dist_over_diam=4),
               distortion="realistic")
        for d in (2, 8, 16, 32):
            record(f"DIST_d{d}", run_condition(f"DIST_d{d}",
                                               dist_over_diam=d),
                   distortion="realistic", distance=d)
        for a in (0.0, 0.01, 0.02, 0.05):
            record(f"ASYM_{a}", run_condition(f"ASYM_{a}", dist_over_diam=4,
                                              asymmetry=a), asymmetry=a)
        for nz in (0.0, 0.5, 1.0, 2.0, 4.0):
            record(f"NOISE_{nz}", run_condition(f"NOISE_{nz}",
                                                dist_over_diam=4,
                                                noise_px=nz), noise_px=nz)
        for sp, nm in ((0.35, "LOW"), (1.0, "MED"), (1.8, "HIGH")):
            record(f"POSE_{nm}", run_condition(f"POSE_{nm}",
                                               dist_over_diam=4,
                                               pose_spread=sp),
                   pose_diversity=nm)

    if args.stage in ("all", "dof"):
        for dof in ("D20", "D10", "D5"):
            record(f"DOF_{dof}", run_condition(f"DOF_{dof}", dof=dof,
                                               dist_over_diam=4), dof=dof)

    if args.stage in ("all", "controls"):
        perm = np.random.default_rng(987654321).permutation(20)
        record("CTRL_correct", run_condition("CTRL_correct",
                                             dist_over_diam=4),
               control="C1_correct_mapping")
        record("CTRL_wrongbone", run_condition("CTRL_wrongbone",
                                               dist_over_diam=4,
                                               mapping=perm),
               control="C2_wrong_bone_mapping")
        record("CTRL_swap", run_condition("CTRL_swap", dist_over_diam=4,
                                          swap=True),
               control="C3_right_subject_swap")

    write_csv(SUM / "corrected_synthetic_summary.csv", rows)
    np.savez_compressed(RAW / "focal_profiles.npz", grid=GRID, **curves)
    write_json(SUM / "sweep_meta.json", {
        "n_trials": N_TRIALS, "n_frames": N_FRAMES,
        "grid": {"q_min": float(GRID[0]), "q_max": float(GRID[-1]),
                 "n": len(GRID)},
        "conditions": [r["condition"] for r in rows],
        "primary_identifiability_metric": "HELD-OUT reprojection",
    })
    print(f"\n{len(rows)} conditions written")


if __name__ == "__main__":
    main()
