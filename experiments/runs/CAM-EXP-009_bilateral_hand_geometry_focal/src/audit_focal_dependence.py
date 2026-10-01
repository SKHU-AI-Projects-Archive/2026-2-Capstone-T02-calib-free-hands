"""FOCAL_DEPENDENCE_AUDIT — run before anything else.

Two claims are checked numerically on controlled synthetic data, where the true
focal is known by construction:

  CLAIM 1  Comparing the FIXED reference 3D bone lengths of the two hands is
           focal-INVARIANT. Its value does not change as the candidate focal
           sweeps, so d/df = 0 and it cannot estimate a focal. This is the
           naive formulation, and it must be shown to be useless.

  CLAIM 2  The CANDIDATE-CONDITIONED bilateral loss - refit l_L(f) and l_R(f)
           from each side's 2D at every candidate focal, then compare - DOES
           vary with the candidate focal, by an amount far above numerical
           noise.

If CLAIM 2 fails, the whole idea is unidentifiable in this formulation and the
run stops with BILATERAL_OBJECTIVE_FOCAL_INVARIANT.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BONE_CORRESPONDENCE, N_BONES, SEED, SUM, q_grid,  # noqa: E402
                    write_csv, write_json)
from fit_candidate_conditioned_bones import bilateral_distance, fit_bones
from generate_synthetic_bimanual import make_sequence


def main() -> None:
    rng = np.random.default_rng(SEED)
    grid = q_grid()

    seq = make_sequence(rng, n_frames=14, asymmetry=0.0, noise_px=0.0,
                        dist_over_diam=4.0)
    f_true = seq["f_true"]
    K0 = seq["K"]
    dist = seq["dist"]

    rows = []
    fixed_vals, cand_vals = [], []
    for q in grid:
        f = q * f_true
        K = np.array([[f, 0, K0[0, 2]], [0, f, K0[1, 2]], [0, 0, 1.0]])

        # --- CLAIM 1: fixed reference lengths, compared directly -----------
        lL_fixed = seq["lL_true"] / seq["lL_true"].sum()
        lR_fixed = seq["lR_true"] / seq["lR_true"].sum()
        d_fixed = bilateral_distance(lL_fixed, lR_fixed, BONE_CORRESPONDENCE)

        # --- CLAIM 2: refit both sides at this candidate focal -------------
        lL, eL = fit_bones(seq["left"], K, dist)
        lR, eR = fit_bones(seq["right"], K, dist)
        d_cand = bilateral_distance(lL, lR, BONE_CORRESPONDENCE)

        fixed_vals.append(d_fixed)
        cand_vals.append(d_cand)
        rows.append({"q": round(float(q), 5), "f": round(float(f), 3),
                     "fixed_length_bilateral": d_fixed,
                     "candidate_conditioned_bilateral": d_cand,
                     "left_reproj_px": eL, "right_reproj_px": eR})

    write_csv(SUM.parent / "raw" / "focal_dependence_curve.csv", rows)

    fv = np.array(fixed_vals, float)
    cv = np.array(cand_vals, float)
    fv_ok = fv[np.isfinite(fv)]
    cv_ok = cv[np.isfinite(cv)]

    fixed_range = float(fv_ok.max() - fv_ok.min()) if fv_ok.size else np.nan
    cand_range = float(cv_ok.max() - cv_ok.min()) if cv_ok.size else np.nan
    cand_min_q = (float(grid[int(np.nanargmin(cv))])
                  if cv_ok.size else float("nan"))
    # relative dynamic range: how much the loss moves compared to its own level
    cand_rel = (cand_range / float(np.nanmedian(cv))
                if cv_ok.size and np.nanmedian(cv) > 0 else np.nan)

    fixed_invariant = bool(np.isfinite(fixed_range) and fixed_range < 1e-12)
    cand_varies = bool(np.isfinite(cand_rel) and cand_rel > 0.05)

    verdict = ("FOCAL_DEPENDENCE_CONFIRMED" if (fixed_invariant and cand_varies)
               else "BILATERAL_OBJECTIVE_FOCAL_INVARIANT")

    out = {
        "claim_1_fixed_length_bilateral_is_focal_invariant": {
            "value_range_over_+-50pct_focal_sweep": fixed_range,
            "is_invariant": fixed_invariant,
            "meaning": "the fixed reference lengths do not change when the "
                       "candidate focal changes, so this naive loss has zero "
                       "derivative with respect to f and cannot estimate a "
                       "focal. This is why CAM-EXP-009 does NOT use it.",
        },
        "claim_2_candidate_conditioned_bilateral_varies_with_focal": {
            "value_range_over_+-50pct_focal_sweep": cand_range,
            "relative_dynamic_range": cand_rel,
            "argmin_q": cand_min_q,
            "varies": cand_varies,
            "meaning": "refitting the bone proportions at every candidate "
                       "focal makes the bilateral agreement depend on f, so it "
                       "can in principle carry focal information.",
        },
        "synthetic_setting": {
            "f_true": float(f_true), "n_frames": 14,
            "asymmetry": 0.0, "noise_px": 0.0,
            "distance_over_diameter": 4.0,
            "note": "noiseless, perfectly symmetric, moderate perspective - "
                    "the most favourable case. If the objective is flat here "
                    "it will be flat everywhere.",
        },
        "verdict": verdict,
    }
    write_json(SUM / "focal_dependence_audit.json", out)

    print(f"fixed-length bilateral range over +-50 % sweep : {fixed_range:.3e}"
          f"   invariant={fixed_invariant}")
    print(f"candidate-conditioned range                    : {cand_range:.3e}"
          f"   relative={cand_rel:.3f}  argmin q={cand_min_q:.3f}")
    print(f"\n{verdict}")
    return 0 if verdict == "FOCAL_DEPENDENCE_CONFIRMED" else 1


if __name__ == "__main__":
    sys.exit(main())
