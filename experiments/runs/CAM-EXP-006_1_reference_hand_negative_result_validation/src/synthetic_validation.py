"""Synthetic validation of the CAM-EXP-006 focal-profile solver.

This answers one question only: does this solver recover a KNOWN focal length
when the focal is genuinely identifiable? If it does not, CAM-EXP-006's
real-data negative result cannot be read as a geometric limitation.

Synthetic evidence validates the implementation and the theory. It is never
substituted for real-data evidence.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import camera_model_solver as S  # noqa: E402
import synthetic_geometry as G  # noqa: E402
from common import (MANIFESTS, RAW, SEED, SUM, write_csv, write_json)  # noqa: E402

N_TRIALS = 100
GRID = S.gamma_grid()
TRUE_GAMMA = G.F_TRUE / max(G.W, G.H)


def run_condition(name, shapes, *, dist_over_diam=2.5, depth_scale=1.0,
                  noise_px=0.0, proj_dist=None, solver_dist=None,
                  proj_cx=None, proj_cy=None, solver_cx=None, solver_cy=None,
                  fy_over_fx=1.0, solver_fy_over_fx=1.0, n_trials=N_TRIALS,
                  seed_offset=0, keep_profiles=False):
    rng = np.random.default_rng(SEED + seed_offset)
    rows, profiles = [], []
    tries = 0
    while len(rows) < n_trials and tries < n_trials * 20:
        tries += 1
        shape = shapes[len(rows) % len(shapes)]
        t = G.make_trial(shape, rng, dist_over_diam, depth_scale, noise_px,
                         proj_dist, proj_cx, proj_cy, fy_over_fx)
        if t is None:
            continue
        obj, uv = t
        curve = S.profile_hand(obj, uv, G.W, G.H, solver_cx, solver_cy,
                               solver_dist, solver_fy_over_fx, GRID)
        if not np.isfinite(curve).any():
            continue
        r = S.pick(curve, GRID)
        f_hat = r["gamma"] * max(G.W, G.H)
        rows.append({
            "condition": name, "trial": len(rows),
            "distance_over_diameter": dist_over_diam,
            "depth_scale": depth_scale, "noise_px": noise_px,
            "n_points": len(obj),
            "f_true_px": G.F_TRUE, "f_hat_px": f_hat,
            "rel_focal_err_pct": 100.0 * abs(f_hat - G.F_TRUE) / G.F_TRUE,
            "signed_log_err": float(np.log(f_hat / G.F_TRUE))
            if f_hat > 0 else np.nan,
            "flat_profile": r["flat_profile"],
            "boundary_solution": r["boundary_solution"],
            "profile_width": r["profile_width"],
            "curvature": r["curvature"],
            "depth_of_minimum": r["depth_of_minimum"],
            "status": r["status"],
        })
        if keep_profiles and len(profiles) < 40:
            profiles.append({"condition": name, "trial": len(rows) - 1,
                             **{f"g{i}": float(v)
                                for i, v in enumerate(curve)}})
    return rows, profiles


def summarise(rows):
    e = np.array([r["rel_focal_err_pct"] for r in rows], float)
    if e.size == 0:
        return {}
    return {
        "n_trials": len(rows),
        "median_rel_focal_err_pct": round(float(np.median(e)), 4),
        "mean_rel_focal_err_pct": round(float(np.mean(e)), 4),
        "p90_rel_focal_err_pct": round(float(np.percentile(e, 90)), 4),
        "share_within_1pct": round(float(np.mean(e <= 1)), 4),
        "share_within_5pct": round(float(np.mean(e <= 5)), 4),
        "share_within_10pct": round(float(np.mean(e <= 10)), 4),
        "share_flat_profile": round(
            float(np.mean([r["flat_profile"] for r in rows])), 4),
        "share_boundary": round(
            float(np.mean([r["boundary_solution"] for r in rows])), 4),
        "share_identifiable": round(
            float(np.mean([r["status"] == "ok" for r in rows])), 4),
        "median_profile_width": round(float(np.nanmedian(
            [r["profile_width"] for r in rows])), 4),
        "median_curvature": round(float(np.nanmedian(
            [r["curvature"] for r in rows])), 6),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    generic = [G.generic_cloud(rng) for _ in range(N_TRIALS)]
    hands = G.hand_shapes(limit=N_TRIALS)
    print(f"generic clouds: {len(generic)}   real hand shapes: {len(hands)}")

    all_rows, all_profiles, summary = [], [], []

    def add(tag, rows, profiles=None, **extra):
        all_rows.extend(rows)
        if profiles:
            all_profiles.extend(profiles)
        s = summarise(rows)
        if s:
            summary.append({"suite": tag, **extra, **s})
        return s

    # ---- 1. the two positive controls -----------------------------------
    r, p = run_condition("SYNTH_GENERIC_STRONG_PERSPECTIVE", generic,
                         keep_profiles=True)
    s_gen = add("SYNTH_GENERIC_STRONG_PERSPECTIVE", r, p, family="positive")
    print("  generic strong perspective:", s_gen)

    r, p = run_condition("SYNTH_HAND_STRONG_PERSPECTIVE", hands,
                         seed_offset=1, keep_profiles=True)
    s_hand = add("SYNTH_HAND_STRONG_PERSPECTIVE", r, p, family="positive")
    print("  hand strong perspective:   ", s_hand)

    # ---- 2. distance sweep ----------------------------------------------
    dist_rows = []
    for d in (2, 3, 5, 10, 20, 40):
        r, _ = run_condition(f"DISTANCE_{d}x", hands, dist_over_diam=d,
                             seed_offset=10 + d)
        s = add(f"DISTANCE_{d}x", r, family="distance_sweep",
                distance_over_diameter=d)
        dist_rows.append({"distance_over_diameter": d, **s})
        print(f"  distance {d:>2}x diameter: "
              f"err {s['median_rel_focal_err_pct']:8.2f} %  "
              f"flat {s['share_flat_profile']:.2f}  "
              f"width {s['median_profile_width']:.3f}")

    # ---- 3. planarity sweep ---------------------------------------------
    plan_rows = []
    for ds in (1.0, 0.75, 0.5, 0.25, 0.1, 0.0):
        r, _ = run_condition(f"DEPTH_SCALE_{ds}", hands, depth_scale=ds,
                             seed_offset=int(100 + ds * 100))
        s = add(f"DEPTH_SCALE_{ds}", r, family="planarity_sweep",
                depth_scale=ds)
        plan_rows.append({"depth_scale": ds, **s})
        print(f"  depth scale {ds:<5}: err {s['median_rel_focal_err_pct']:8.2f} %"
              f"  flat {s['share_flat_profile']:.2f}  "
              f"width {s['median_profile_width']:.3f}")

    # ---- 3b. distance sweep WITH realistic 2D noise ----------------------
    # Noiseless data pins the minimum of even a very flat curve exactly, so
    # the distance sweep above shows the loss of identifiability only through
    # profile width. Real observations carry noise, and a flat curve plus
    # noise is what turns into a large focal error. This sweep makes that
    # link explicit.
    dist_noise_rows = []
    for d in (2, 3, 5, 10, 20, 40):
        r, _ = run_condition(f"DISTANCE_{d}x_NOISE1px", hands,
                             dist_over_diam=d, noise_px=1.0,
                             seed_offset=500 + d)
        s = add(f"DISTANCE_{d}x_NOISE1px", r,
                family="distance_sweep_noise1px", distance_over_diameter=d,
                noise_px=1.0)
        dist_noise_rows.append({"distance_over_diameter": d, "noise_px": 1.0,
                                **s})
        print(f"  distance {d:>2}x + 1px noise: "
              f"err {s['median_rel_focal_err_pct']:8.2f} %  "
              f"within5 {s['share_within_5pct']:.2f}  "
              f"flat {s['share_flat_profile']:.2f}")

    # ---- 4. noise sweep --------------------------------------------------
    noise_rows = []
    for nz in (0.0, 0.5, 1.0, 2.0):
        r, _ = run_condition(f"NOISE_{nz}px", hands, noise_px=nz,
                             seed_offset=int(200 + nz * 10))
        s = add(f"NOISE_{nz}px", r, family="noise_sweep", noise_px=nz)
        noise_rows.append({"noise_px": nz, **s})
        print(f"  noise {nz:<4} px: err {s['median_rel_focal_err_pct']:8.2f} %"
              f"  within5 {s['share_within_5pct']:.2f}")

    # ---- 5. principal point ---------------------------------------------
    cm_rows = []
    dx, dy = G.PP_OFFSET
    for tag, kw in [
        ("CENTER_PP", {}),
        ("OFFCENTER_PP_SOLVER_ASSUMES_CENTRE",
         {"proj_cx": G.W / 2 + dx, "proj_cy": G.H / 2 + dy}),
        ("OFFCENTER_PP_SOLVER_GIVEN_TRUE_PP",
         {"proj_cx": G.W / 2 + dx, "proj_cy": G.H / 2 + dy,
          "solver_cx": G.W / 2 + dx, "solver_cy": G.H / 2 + dy}),
    ]:
        r, _ = run_condition(tag, hands, seed_offset=300 + len(cm_rows), **kw)
        s = add(tag, r, family="principal_point")
        cm_rows.append({"family": "principal_point", "condition": tag, **s})
        print(f"  {tag:38s} err {s['median_rel_focal_err_pct']:8.2f} %  "
              f"flat {s['share_flat_profile']:.2f}")

    # ---- 6. distortion ---------------------------------------------------
    for tag, kw in [
        ("D0_none_none", {}),
        ("D1_distorted_solver_assumes_none", {"proj_dist": G.DIST_RIG}),
        ("D2_distorted_solver_given_true",
         {"proj_dist": G.DIST_RIG, "solver_dist": G.DIST_RIG}),
    ]:
        r, _ = run_condition(tag, hands, seed_offset=400 + len(cm_rows), **kw)
        s = add(tag, r, family="distortion")
        cm_rows.append({"family": "distortion", "condition": tag, **s})
        print(f"  {tag:38s} err {s['median_rel_focal_err_pct']:8.2f} %  "
              f"flat {s['share_flat_profile']:.2f}")

    # ---- write ------------------------------------------------------------
    write_csv(RAW / "synthetic_trials.csv.gz", all_rows)
    write_csv(RAW / "synthetic_profiles.csv.gz", all_profiles)
    write_csv(SUM / "synthetic_validation_summary.csv", summary)
    write_csv(SUM / "distance_sweep_summary.csv",
              dist_rows + dist_noise_rows)
    write_csv(SUM / "planarity_sweep_summary.csv", plan_rows)
    write_csv(SUM / "noise_sweep_summary.csv", noise_rows)
    write_csv(SUM / "camera_model_synthetic_summary.csv", cm_rows)
    write_csv(MANIFESTS / "cam_exp_0061_synthetic_trials_v1.csv.gz", all_rows)

    # ---- pass conditions, frozen before this ran -------------------------
    pc = {
        "median_rel_focal_err_pct <= 1.0":
            s_gen["median_rel_focal_err_pct"] <= 1.0,
        "share_within_5pct >= 0.95": s_gen["share_within_5pct"] >= 0.95,
        "share_flat_profile <= 0.05": s_gen["share_flat_profile"] <= 0.05,
        "share_boundary <= 0.05": s_gen["share_boundary"] <= 0.05,
    }
    passed = all(pc.values())
    hand_ok = (s_hand["median_rel_focal_err_pct"] <= 1.0
               and s_hand["share_within_5pct"] >= 0.95)

    # does identifiability degrade with distance and with flattening?
    d_err = [r["median_rel_focal_err_pct"] for r in dist_rows]
    d_width = [r["median_profile_width"] for r in dist_rows]
    p_width = [r["median_profile_width"] for r in plan_rows]
    # Identifiability is measured by how WIDE the near-optimal region is and
    # how often the profile is flagged flat, not by the noiseless error: with
    # exact 2D even a nearly flat curve still has its minimum at the truth.
    d_flat = [r["share_flat_profile"] for r in dist_rows]
    dn_err = [r["median_rel_focal_err_pct"] for r in dist_noise_rows]
    degrades_with_distance = bool(d_width[-1] > 10 * d_width[0]
                                  and d_flat[-1] > d_flat[0])
    degrades_with_distance_under_noise = bool(dn_err[-1] > dn_err[0])
    degrades_with_flatness = bool(
        plan_rows[-2]["median_rel_focal_err_pct"]
        > plan_rows[0]["median_rel_focal_err_pct"])

    verdict = ("SOLVER_VALIDATED" if passed and hand_ok
               else "SOLVER_PARTIALLY_VALIDATED" if passed
               else "SOLVER_VALIDATION_FAILED")
    out = {
        "generic_positive_control": s_gen,
        "hand_positive_control": s_hand,
        "pass_conditions": pc,
        "generic_positive_control_passed": passed,
        "hand_positive_control_passed": hand_ok,
        "identifiability_degrades_with_distance": degrades_with_distance,
        "identifiability_measure": "profile width and flat-profile share; the "
                                   "noiseless focal error stays low at every "
                                   "distance because an exact 2D observation "
                                   "pins the minimum of even a very flat "
                                   "curve. Identifiability is about how "
                                   "sharply the objective distinguishes "
                                   "focals, which is what width measures.",
        "distance_sweep_noise1px_median_err_pct": dn_err,
        "focal_error_degrades_with_distance_under_1px_noise":
            degrades_with_distance_under_noise,
        "identifiability_degrades_with_flattening": degrades_with_flatness,
        "distance_profile_width_first_last": [d_width[0], d_width[-1]],
        "planarity_profile_width_first_last": [p_width[0], p_width[-2]],
        "verdict": verdict,
        "scope": "synthetic evidence validates the implementation and the "
                 "theory. It says nothing on its own about whether focal is "
                 "recoverable from real GigaHands hands.",
    }
    write_json(SUM / "synthetic_validation_verdict.json", out)
    print(f"\nSYNTHETIC VERDICT: {verdict}")
    print(f"  degrades with distance: {degrades_with_distance}")
    print(f"  degrades with flattening: {degrades_with_flatness}")


if __name__ == "__main__":
    main()
