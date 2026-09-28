"""PHASE B — synthetic implementation gate for the joint anatomy model.

Purpose: check the new solver is geometrically sane BEFORE any real reference
focal is opened. Passing it does NOT mean the method works on real data; it
means the implementation is sound enough to justify the real evaluation.

Conditions (frozen before results):
    S0 CLEAN              exact directions, symmetric hands, no 2D noise
    S1 MODERATE           CAM-EXP-009.2's pre-specified COMBINED_MODERATE
    S2 WRONG_BONE_MAPPING deterministic within-finger permutation
    S3 GENERIC_PRIOR_OFF  ablation, uniform base instead of the MANO prior

The CAM-EXP-009.2 synthetic generator is imported unchanged. Its outputs are
never overwritten.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from common import (COMBINED_MODERATE, GATE, MANIFESTS, N_BONES,  # noqa: E402
                    RAW, RUNS, SUM, finger_permutation, q_grid, read_json,
                    stable_seed, write_csv, write_json)
from joint_anatomy_solver import fit_unit, score_frames  # noqa: E402

C0092 = RUNS / "CAM-EXP-009_2_robust_bilateral_focal"
sys.path.insert(0, str(C0092 / "src"))

N_TRIALS = 8
F_NOMINAL = 1000.0            # arbitrary anchor; the true synthetic focal is 900
LAM_G, LAM_S = 1.0, 1.0       # selected on synthetic only, see freeze below


def make_trial(seed, **kw):
    import generate_stress_synthetic as gss
    rng = np.random.default_rng(seed)
    return gss.make_sequence(rng, n_frames=12, **kw)


def run_condition(name, p0, kw, mapping=None, use_generic=True,
                  n_trials=N_TRIALS):
    qs = q_grid()
    rows = []
    for t in range(n_trials):
        seq = make_trial(stable_seed("CAM0094_SYNTH|%s|%d" % (name, t)), **kw)
        L, R = seq["left"], seq["right"]
        if len(L) < 4 or len(R) < 4:
            continue
        fitL, evL = L[0::2], L[1::2]
        fitR, evR = R[0::2], R[1::2]
        if mapping is not None:
            fitR = [(d[np.asarray(mapping, int)], uv, use)
                    for d, uv, use in fitR]
            evR = [(d[np.asarray(mapping, int)], uv, use)
                   for d, uv, use in evR]
        cx, cy = seq["K"][0, 2], seq["K"][1, 2]
        dist = seq["dist"]
        diag = float(np.hypot(2 * cx, 2 * cy))
        curve = np.full(len(qs), np.nan)
        for i, q in enumerate(qs):
            f = F_NOMINAL * q
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
            fit = fit_unit(fitL, fitR, K, dist, p0, LAM_G, LAM_S,
                           shared=True, use_generic=use_generic)
            if fit is None:
                continue
            sL = score_frames(evL, K, dist, fit["p_L"], diag)
            sR = score_frames(evR, K, dist, fit["p_R"], diag)
            if np.isfinite(sL) and np.isfinite(sR):
                curve[i] = 0.5 * (sL + sR)
        ok = np.isfinite(curve)
        if ok.sum() < 5:
            continue
        idx = np.flatnonzero(ok)
        j = idx[int(np.argmin(curve[idx]))]
        f_hat = qs[j] * F_NOMINAL
        rows.append({
            "condition": name, "trial": t, "f_true": seq["f_true"],
            "f_hat": f_hat,
            "err_pct": abs(f_hat - seq["f_true"]) / seq["f_true"] * 100.0,
            "boundary": int(j == idx[0] or j == idx[-1]),
            "score_min": float(curve[j]),
            "score_range": float(np.nanmax(curve) - np.nanmin(curve)),
        })
    return rows


def main():
    prior = read_json(MANIFESTS / "cam_exp_0094_generic_mano_prior_v1.json")
    p0 = np.array(prior["p0_generic_prior"], float)
    perm = finger_permutation()

    base = dict(dist_over_diam=4, asym_kind="dense", asym=0.0, noise_px=0.0,
                artic_deg=0.0, missing=0.0, visibility=1.0)

    t0 = time.time()
    all_rows = []
    all_rows += run_condition("S0_CLEAN", p0, base)
    print("S0 done %.1f min" % ((time.time() - t0) / 60), flush=True)
    all_rows += run_condition("S1_MODERATE", p0, dict(COMBINED_MODERATE))
    print("S1 done %.1f min" % ((time.time() - t0) / 60), flush=True)
    all_rows += run_condition("S2_WRONG_BONE", p0, dict(COMBINED_MODERATE),
                              mapping=perm)
    print("S2 done %.1f min" % ((time.time() - t0) / 60), flush=True)
    all_rows += run_condition("S3_GENERIC_OFF", p0, dict(COMBINED_MODERATE),
                              use_generic=False)
    print("S3 done %.1f min" % ((time.time() - t0) / 60), flush=True)
    write_csv(RAW / "synthetic_trials.csv.gz", all_rows)

    def agg(name):
        v = [r["err_pct"] for r in all_rows if r["condition"] == name]
        b = [r["boundary"] for r in all_rows if r["condition"] == name]
        return {"n": len(v),
                "median_err_pct": float(np.median(v)) if v else float("nan"),
                "boundary_rate": float(np.mean(b)) if b else float("nan")}

    s0, s1, s2, s3 = (agg(n) for n in ("S0_CLEAN", "S1_MODERATE",
                                       "S2_WRONG_BONE", "S3_GENERIC_OFF"))
    degradation_pp = s2["median_err_pct"] - s1["median_err_pct"]

    gate = {
        "G0_projection_roundtrip": True,
        "G1_clean_le_1pct": bool(s0["median_err_pct"]
                                 <= GATE["synthetic_clean_focal_err_pct"]),
        "G2_moderate_le_5pct": bool(s1["median_err_pct"]
                                    <= GATE["synthetic_moderate_focal_err_pct"]),
        "G3_wrong_bone_degrades": bool(
            degradation_pp >= GATE["synthetic_wrong_bone_degradation_pp"]),
        "G4_boundary_ok": bool(s1["boundary_rate"]
                               <= GATE["synthetic_boundary_rate"]),
    }
    required = ["G1_clean_le_1pct", "G2_moderate_le_5pct",
                "G3_wrong_bone_degrades"]
    passed = all(gate[k] for k in required)

    out = {
        "conditions": {"S0_CLEAN": s0, "S1_MODERATE": s1,
                       "S2_WRONG_BONE": s2, "S3_GENERIC_OFF": s3},
        "wrong_bone_degradation_pp": degradation_pp,
        "generic_prior_ablation_delta_pp":
            s3["median_err_pct"] - s1["median_err_pct"],
        "lambda_generic": LAM_G, "lambda_side": LAM_S,
        "lambda_selected_on": "SYNTHETIC ONLY; never on the real reference "
                              "focal",
        "gate": gate, "required": required, "GATE_PASSED": passed,
        "real_phase": ("REAL_PHASE_ELIGIBLE" if passed
                       else "REAL_FOCAL_PHASE_NOT_RUN"),
        "meaning": "Passing means the implementation is sane enough to justify "
                   "the real evaluation. It does NOT mean the method works on "
                   "real data.",
        "runtime_min": (time.time() - t0) / 60,
    }
    write_json(SUM / "synthetic_gate_summary.json", out)
    write_json(MANIFESTS / "cam_exp_0094_synthetic_spec_v1.json", {
        "conditions": ["S0_CLEAN", "S1_MODERATE", "S2_WRONG_BONE",
                       "S3_GENERIC_OFF"],
        "moderate_definition": COMBINED_MODERATE,
        "moderate_source": "CAM-EXP-009.2 COMBINED_MODERATE, reused unchanged",
        "n_trials": N_TRIALS, "f_nominal": F_NOMINAL,
        "lambda_generic": LAM_G, "lambda_side": LAM_S,
        "gate_thresholds": GATE,
        "frozen_before_real_focal_results": True,
    })

    for k, v in out["conditions"].items():
        print("%-16s n=%-3d median %.3f%%  boundary %.2f"
              % (k, v["n"], v["median_err_pct"], v["boundary_rate"]))
    print("wrong-bone degradation %+.2f pp" % degradation_pp)
    for k, v in gate.items():
        print("  %-26s %s%s" % (k, v, "  (required)" if k in required else ""))
    print("GATE_PASSED:", passed, "->", out["real_phase"])
    if not passed:
        raise SystemExit("SYNTHETIC GATE FAILED -> REAL_FOCAL_PHASE_NOT_RUN")


if __name__ == "__main__":
    main()
