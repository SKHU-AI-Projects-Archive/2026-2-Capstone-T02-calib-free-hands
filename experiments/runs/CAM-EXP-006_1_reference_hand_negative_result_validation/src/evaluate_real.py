"""Evaluate the R0-R5 real-data reanalysis against the frozen thresholds.

This is the first file in CAM-EXP-006.1 that compares a focal estimate to the
dataset-provided reference focal. Nothing it reads fed back into any solver.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (COUNTS, R006_SUMMARY, RAW, SUM, TAB, VIEW_TARGETS,  # noqa: E402
                    cluster_bootstrap, corr_cluster_ci, fnum, read_csv,
                    write_csv, write_json)

SOL = RAW / "real_validation_solutions.csv.gz"
CAM006_N16_ERR, CAM006_N16_FLAT, CAM006_N16_IDENT = 202.1629, 0.8686, 0.04


def rel_err_pct(hat, ref):
    return 100.0 * np.abs(hat - ref) / ref


def summarise(sub, const):
    e = np.array([r["rel_focal_err_pct"] for r in sub], float)
    cl = [r["physical_camera_id"] for r in sub]
    med, lo, hi = cluster_bootstrap(e, cl)
    sl = np.array([r["signed_log_err"] for r in sub], float)
    return {
        "n_views": len(sub),
        "median_rel_focal_err_pct": round(med, 4),
        "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
        "p90_rel_focal_err_pct": round(float(np.nanpercentile(e, 90)), 4),
        "median_signed_log_err": round(float(np.nanmedian(sl)), 5),
        "mean_signed_log_err": round(float(np.nanmean(sl)), 5),
        "share_within_5pct": round(float(np.nanmean(e <= 5)), 4),
        "share_within_10pct": round(float(np.nanmean(e <= 10)), 4),
        "share_within_20pct": round(float(np.nanmean(e <= 20)), 4),
        "share_flat_profile": round(
            float(np.mean([r["flat_profile"] for r in sub])), 4),
        "share_boundary": round(
            float(np.mean([r["boundary_solution"] for r in sub])), 4),
        "share_cleanly_identifiable": round(
            float(np.mean([r["solver_status"] == "ok" for r in sub])), 4),
        "median_profile_width": round(float(np.nanmedian(
            [r["profile_width"] for r in sub])), 4),
        "median_curvature": round(float(np.nanmedian(
            [r["curvature"] for r in sub])), 6),
    }


def main() -> None:
    tgt = {(r["sequence"], r["camera"]): r for r in read_csv(VIEW_TARGETS)}
    refs = np.array([fnum(t["gt_reference_focal_px"]) for t in tgt.values()])
    const = float(np.median(refs))

    rows = []
    for s in read_csv(SOL):
        t = tgt.get((s["sequence"], s["camera"]))
        if t is None:
            continue
        ref = fnum(t["gt_reference_focal_px"])
        hat = fnum(s["focal_hat_px"])
        rows.append({
            "sequence": s["sequence"], "camera": s["camera"],
            "physical_camera_id": t["physical_camera_id"],
            "condition": s["condition"],
            "condition_name": s["condition_name"],
            "n_frames": int(s["n_frames"]),
            "focal_hat_px": hat, "gt_reference_focal_px": ref,
            "rel_focal_err_pct": rel_err_pct(hat, ref),
            "signed_log_err": np.log(hat / ref) if hat > 0 else np.nan,
            "flat_profile": int(s["flat_profile"]),
            "boundary_solution": int(s["boundary_solution"]),
            "solver_status": s["solver_status"],
            "profile_width": fnum(s["profile_width"]),
            "curvature": fnum(s["curvature"]),
            "n_frames_used": int(s["n_frames_used"]),
            "n_hand_observations": int(s["n_hand_observations"]),
        })
    write_csv(RAW / "real_validation_view_estimates.csv.gz", rows)

    conds = ["R0", "R1", "R2", "R3", "R4", "R5"]
    names = {r["condition"]: r["condition_name"] for r in rows}

    # ---- per-condition, per-N ------------------------------------------
    summary = []
    for c in conds:
        for n in COUNTS:
            sub = [r for r in rows if r["condition"] == c
                   and r["n_frames"] == n]
            if sub:
                summary.append({"condition": c, "condition_name": names[c],
                                "n_frames": n, **summarise(sub, const)})
    write_csv(SUM / "real_condition_summary.csv", summary)

    def get(c, n):
        return next(s for s in summary
                    if s["condition"] == c and s["n_frames"] == n)

    # ---- tracking -------------------------------------------------------
    tracking = []
    for c in conds:
        for n in COUNTS:
            sub = [r for r in rows if r["condition"] == c
                   and r["n_frames"] == n]
            hat = [r["focal_hat_px"] for r in sub]
            ref = [r["gt_reference_focal_px"] for r in sub]
            cl = [r["physical_camera_id"] for r in sub]
            sp, spp, splo, sphi, nn = corr_cluster_ci(hat, ref, cl, "spearman")
            pe, pep, pelo, pehi, _ = corr_cluster_ci(hat, ref, cl, "pearson")
            dh = [h - const for h in hat]
            dr = [r - const for r in ref]
            dsp, _, dlo, dhi, _ = corr_cluster_ci(dh, dr, cl, "spearman")
            hv = np.array(hat, float)
            tracking.append({
                "condition": c, "condition_name": names[c], "n_frames": n,
                "n_views": nn,
                "spearman_hat_vs_ref": round(sp, 4),
                "spearman_ci95_lo": round(splo, 4),
                "spearman_ci95_hi": round(sphi, 4),
                "spearman_p": round(spp, 6),
                "pearson_hat_vs_ref": round(pe, 4),
                "pearson_ci95_lo": round(pelo, 4),
                "pearson_ci95_hi": round(pehi, 4),
                "spearman_deviation_from_rig_median": round(dsp, 4),
                "deviation_ci95_lo": round(dlo, 4),
                "deviation_ci95_hi": round(dhi, 4),
                "tracking_positive_ci_excludes_zero": bool(
                    np.isfinite(splo) and splo > 0),
                "estimate_cv_pct": round(float(
                    100 * np.nanstd(hv) / np.nanmean(hv)), 3),
                "reference_cv_pct": round(float(
                    100 * np.std(ref) / np.mean(ref)), 3),
            })
    write_csv(SUM / "real_tracking_summary.csv", tracking)

    # ---- stress subsets -------------------------------------------------
    stress = []
    for tag, thr in (("S2", 0.02), ("S5", 0.05)):
        keys = {k for k, t in tgt.items()
                if abs(fnum(t["gt_reference_focal_px"]) - const) / const >= thr}
        for c in conds:
            sub = [r for r in rows if r["condition"] == c
                   and r["n_frames"] == 16
                   and (r["sequence"], r["camera"]) in keys]
            if not sub:
                continue
            ncam = len({r["physical_camera_id"] for r in sub})
            e = np.array([r["rel_focal_err_pct"] for r in sub], float)
            ce = rel_err_pct(const, np.array(
                [r["gt_reference_focal_px"] for r in sub], float))
            stress.append({
                "subset": tag, "threshold_pct": thr * 100,
                "condition": c, "n_views": len(sub),
                "n_physical_cameras": ncam,
                "underpowered": int(len(sub) < 20 or ncam < 8),
                "median_rel_focal_err_pct": round(float(np.nanmedian(e)), 4),
                "median_signed_log_err": round(float(np.nanmedian(
                    [r["signed_log_err"] for r in sub])), 5),
                "constant_oracle_median_err_pct": round(
                    float(np.median(ce)), 4),
            })
    write_csv(SUM / "nondefault_focal_stress_summary.csv", stress)

    # ---- constant oracle -------------------------------------------------
    ce = rel_err_pct(const, refs)
    cl = [t["physical_camera_id"] for t in tgt.values()]
    cmed, clo, chi = cluster_bootstrap(ce, cl)
    const_row = {
        "comparator": "CONSTANT_RIG_FOCAL_ORACLE",
        "status": "ORACLE_SANITY_CONTROL",
        "registration": "PRE_REGISTERED_IN_CAM006_1_ORACLE_SANITY_CONTROL",
        "motivation": "MOTIVATED_BY_CAM005_POST_HOC_FINDING",
        "n_views": len(ce),
        "median_rel_focal_err_pct": round(cmed, 4),
        "ci95_lo": round(clo, 4), "ci95_hi": round(chi, 4),
    }
    write_csv(TAB / "constant_rig_focal_oracle.csv", [const_row])

    # ---- frozen threshold evaluations ------------------------------------
    r0, r1, r5 = get("R0", 16), get("R1", 16), get("R5", 16)
    d_err = abs(r1["median_rel_focal_err_pct"]
                - r0["median_rel_focal_err_pct"])
    d_flat = abs(r1["share_flat_profile"] - r0["share_flat_profile"]) * 100
    d_id = abs(r1["share_cleanly_identifiable"]
               - r0["share_cleanly_identifiable"]) * 100
    weighting_material = bool(d_err >= 10 or d_flat >= 10 or d_id >= 10)

    rel_red = (100.0 * (r1["median_rel_focal_err_pct"]
                        - r5["median_rel_focal_err_pct"])
               / r1["median_rel_focal_err_pct"])
    id_gain = (r5["share_cleanly_identifiable"]
               - r1["share_cleanly_identifiable"]) * 100
    flat_drop = (r1["share_flat_profile"] - r5["share_flat_profile"]) * 100
    t5 = next(t for t in tracking if t["condition"] == "R5"
              and t["n_frames"] == 16)
    model_material = bool(rel_red >= 25 or id_gain >= 20 or flat_drop >= 20
                          or t5["tracking_positive_ci_excludes_zero"])

    repro_err = abs(r0["median_rel_focal_err_pct"] - CAM006_N16_ERR)
    repro_flat = abs(r0["share_flat_profile"] - CAM006_N16_FLAT) * 100
    reproduction_ok = bool(repro_err <= 5 and repro_flat <= 5)

    write_csv(TAB / "original_vs_frame_balanced.csv", [
        {"n_frames": n,
         "R0_median_err_pct": get("R0", n)["median_rel_focal_err_pct"],
         "R1_median_err_pct": get("R1", n)["median_rel_focal_err_pct"],
         "R0_flat": get("R0", n)["share_flat_profile"],
         "R1_flat": get("R1", n)["share_flat_profile"],
         "R0_identifiable": get("R0", n)["share_cleanly_identifiable"],
         "R1_identifiable": get("R1", n)["share_cleanly_identifiable"]}
        for n in COUNTS])
    write_csv(TAB / "original_vs_full_oracle_camera_model.csv", [
        {"condition": c, "condition_name": names[c],
         "median_err_pct_N16": get(c, 16)["median_rel_focal_err_pct"],
         "flat_N16": get(c, 16)["share_flat_profile"],
         "identifiable_N16": get(c, 16)["share_cleanly_identifiable"],
         "profile_width_N16": get(c, 16)["median_profile_width"]}
        for c in conds])

    out = {
        "constant_rig_focal_px": round(const, 3),
        "constant_rig_focal_oracle": const_row,
        "R0_N16": r0, "R1_N16": r1, "R5_N16": r5,
        "R0_N1": get("R0", 1), "R1_N1": get("R1", 1),
        "reproduction": {
            "cam006_n16_median_err_pct": CAM006_N16_ERR,
            "r0_n16_median_err_pct": r0["median_rel_focal_err_pct"],
            "abs_diff_pp": round(repro_err, 4),
            "cam006_n16_flat": CAM006_N16_FLAT,
            "r0_n16_flat": r0["share_flat_profile"],
            "flat_diff_pp": round(repro_flat, 4),
            "verdict": "REPRODUCTION_OK" if reproduction_ok
                       else "REPRODUCTION_MISMATCH"},
        "frame_weighting": {
            "delta_median_err_pp": round(d_err, 4),
            "delta_flat_pp": round(d_flat, 4),
            "delta_identifiable_pp": round(d_id, 4),
            "verdict": "FRAME_WEIGHTING_MATERIAL" if weighting_material
                       else "FRAME_WEIGHTING_NON_MATERIAL"},
        "camera_model": {
            "relative_error_reduction_pct": round(rel_red, 4),
            "identifiable_gain_pp": round(id_gain, 4),
            "flat_drop_pp": round(flat_drop, 4),
            "r5_tracking_positive": t5["tracking_positive_ci_excludes_zero"],
            "verdict": "CAMERA_MODEL_MISMATCH_MATERIAL" if model_material
                       else "CAMERA_MODEL_MISMATCH_NON_MATERIAL"},
        "tracking_R1_N16": next(t for t in tracking if t["condition"] == "R1"
                                and t["n_frames"] == 16),
        "tracking_R5_N16": t5,
    }
    write_json(SUM / "real_evaluation.json", out)

    print(f"constant-rig oracle: {cmed:.3f} %")
    print(f"\nreproduction: {out['reproduction']['verdict']}  "
          f"(R0 N16 {r0['median_rel_focal_err_pct']:.2f} % vs CAM-006 "
          f"{CAM006_N16_ERR:.2f} %)")
    print(f"frame weighting: {out['frame_weighting']['verdict']}  "
          f"(d_err {d_err:.2f} pp, d_flat {d_flat:.2f} pp, "
          f"d_ident {d_id:.2f} pp)")
    print(f"camera model:    {out['camera_model']['verdict']}  "
          f"(rel reduction {rel_red:.2f} %, ident +{id_gain:.2f} pp, "
          f"flat {-flat_drop:+.2f} pp)")
    print("\ncond  N16 median_err%   flat%  ident%  width")
    for c in conds:
        s = get(c, 16)
        print(f"{c:4s} {s['median_rel_focal_err_pct']:12.2f} "
              f"{100*s['share_flat_profile']:8.1f} "
              f"{100*s['share_cleanly_identifiable']:7.1f} "
              f"{s['median_profile_width']:7.3f}   {s['condition_name']}")
    print("\ntracking (estimate vs provided reference focal):")
    for c in conds:
        t = next(t for t in tracking if t["condition"] == c
                 and t["n_frames"] == 16)
        print(f"  {c}: spearman {t['spearman_hat_vs_ref']:+.3f} "
              f"CI [{t['spearman_ci95_lo']:+.3f}, "
              f"{t['spearman_ci95_hi']:+.3f}]  "
              f"est CV {t['estimate_cv_pct']:.1f} % vs ref CV "
              f"{t['reference_cv_pct']:.2f} %")
    print("\nstress subsets:")
    for s in stress:
        if s["condition"] in ("R1", "R5"):
            print(f"  {s['subset']} {s['condition']}: {s['n_views']} views, "
                  f"{s['n_physical_cameras']} cameras, "
                  f"underpowered={s['underpowered']}, "
                  f"err {s['median_rel_focal_err_pct']:.2f} % vs constant "
                  f"{s['constant_oracle_median_err_pct']:.2f} %")


if __name__ == "__main__":
    main()
