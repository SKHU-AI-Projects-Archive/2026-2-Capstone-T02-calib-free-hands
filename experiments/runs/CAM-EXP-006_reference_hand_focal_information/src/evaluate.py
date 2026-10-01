"""Evaluate the frozen solver output against the frozen evaluation spec.

This is the first file in CAM-EXP-006 that reads GT_EFFECTIVE_FOCAL. Nothing it
reads here fed back into the solver, the frame grid or the controls.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (R005, RAW, SUM, TAB, cluster_bootstrap, fnum, read_csv,  # noqa: E402
                    write_csv, write_json)

SOL = RAW / "focal_profile_solutions.csv.gz"
TARGETS = R005 / "results" / "raw" / "view_targets.csv.gz"
COUNTS = (1, 2, 4, 8, 16)


def rel_err_pct(hat, ref):
    return 100.0 * np.abs(hat - ref) / ref


def main() -> None:
    tgt = {(r["sequence"], r["camera"]): r for r in read_csv(TARGETS)}
    sol = read_csv(SOL)

    rows = []
    for s in sol:
        key = (s["sequence"], s["camera"])
        t = tgt.get(key)
        if t is None:
            continue
        ref = fnum(t["gt_reference_focal_px"])
        hat = fnum(s["focal_hat_px"])
        rows.append({
            "sequence": s["sequence"], "camera": s["camera"],
            "physical_camera_id": t["physical_camera_id"],
            "condition": s["condition"], "n_frames": int(s["n_frames"]),
            "focal_hat_px": hat,
            "gt_reference_focal_px": ref,
            "rel_focal_err_pct": rel_err_pct(hat, ref),
            "signed_log_err": np.log(hat / ref) if hat > 0 else np.nan,
            "flat_profile": int(s["flat_profile"]),
            "boundary_solution": int(s["boundary_solution"]),
            "solver_status": s["solver_status"],
            "n_hand_observations": int(s["n_hand_observations"]),
            "min_objective_px": fnum(s["min_objective_px"]),
        })
    write_csv(RAW / "view_focal_estimates.csv.gz", rows)

    refs = np.array([fnum(t["gt_reference_focal_px"]) for t in tgt.values()])
    const = float(np.median(refs))

    # ---- per-condition, per-N summary -------------------------------------
    summary = []
    for cond in sorted({r["condition"] for r in rows}):
        for n in COUNTS:
            sub = [r for r in rows if r["condition"] == cond
                   and r["n_frames"] == n]
            if not sub:
                continue
            e = np.array([r["rel_focal_err_pct"] for r in sub])
            cl = [r["physical_camera_id"] for r in sub]
            med, lo, hi = cluster_bootstrap(e, cl)
            b = np.array([r["signed_log_err"] for r in sub])
            bias = np.nanmean(b)
            summary.append({
                "condition": cond, "n_frames": n, "n_views": len(sub),
                "median_rel_focal_err_pct": round(med, 4),
                "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
                "mean_rel_focal_err_pct": round(float(np.nanmean(e)), 4),
                "mean_signed_log_err": round(float(bias), 5),
                "share_flat_profile": round(
                    float(np.mean([r["flat_profile"] for r in sub])), 4),
                "share_boundary": round(
                    float(np.mean([r["boundary_solution"] for r in sub])), 4),
                "share_identifiable": round(float(np.mean(
                    [r["solver_status"] == "ok" for r in sub])), 4),
            })

    # ---- comparators -------------------------------------------------------
    comp = []
    for name, col in [("ANYCALIB_N8", "anycalib_view_focal_px"),
                      ("GEOCALIB_N8", "geocalib_view_focal_px")]:
        e, cl = [], []
        for k, t in tgt.items():
            ref = fnum(t["gt_reference_focal_px"])
            e.append(rel_err_pct(fnum(t[col]), ref))
            cl.append(t["physical_camera_id"])
        med, lo, hi = cluster_bootstrap(np.array(e), cl)
        comp.append({"comparator": name, "role": "SINGLE_IMAGE_BASELINE",
                     "n_views": len(e),
                     "median_rel_focal_err_pct": round(med, 4),
                     "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4)})
    # frozen E2 = geometric mean of the two, per view
    e, cl = [], []
    for k, t in tgt.items():
        ref = fnum(t["gt_reference_focal_px"])
        f2 = np.sqrt(fnum(t["anycalib_view_focal_px"])
                     * fnum(t["geocalib_view_focal_px"]))
        e.append(rel_err_pct(f2, ref))
        cl.append(t["physical_camera_id"])
    med, lo, hi = cluster_bootstrap(np.array(e), cl)
    comp.append({"comparator": "E2_N8",
                 "role": "FROZEN_EXPLORATORY_ENSEMBLE_not_a_proposed_method",
                 "n_views": len(e), "median_rel_focal_err_pct": round(med, 4),
                 "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4)})
    # pre-registered constant-rig oracle
    e = rel_err_pct(const, refs)
    cl = [t["physical_camera_id"] for t in tgt.values()]
    med, lo, hi = cluster_bootstrap(e, cl)
    comp.append({"comparator": "CONSTANT_RIG_FOCAL_ORACLE",
                 "role": "PRE_REGISTERED_POST_HOC_ORACLE_SANITY_CONTROL",
                 "n_views": len(e), "median_rel_focal_err_pct": round(med, 4),
                 "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4)})

    TAB.mkdir(parents=True, exist_ok=True)
    write_csv(TAB / "solver_summary.csv", summary)
    write_csv(TAB / "comparators.csv", comp)

    # ---- pre-registered hypothesis tests -----------------------------------
    def get(cond, n):
        return next(s for s in summary
                    if s["condition"] == cond and s["n_frames"] == n)

    main16 = get("MAIN", 16)
    main1 = get("MAIN", 1)
    oracle = next(c for c in comp
                  if c["comparator"] == "CONSTANT_RIG_FOCAL_ORACLE")

    # paired difference main(N=16) - constant oracle, bootstrapped by camera
    paired, cl = [], []
    for r in rows:
        if r["condition"] != "MAIN" or r["n_frames"] != 16:
            continue
        paired.append(r["rel_focal_err_pct"]
                      - rel_err_pct(const, r["gt_reference_focal_px"]))
        cl.append(r["physical_camera_id"])
    dmed, dlo, dhi = cluster_bootstrap(np.array(paired), cl)

    h1 = bool(main16["median_rel_focal_err_pct"]
              < oracle["median_rel_focal_err_pct"] and dhi < 0)
    ctrl16 = {c: get(c, 16)["median_rel_focal_err_pct"]
              for c in ("JOINT_PERMUTATION", "WRONG_POSE", "PLANARIZED")}
    h2 = bool(all(v >= 2.0 * main16["median_rel_focal_err_pct"]
                  for v in ctrl16.values()))
    h3 = bool(main1["median_rel_focal_err_pct"]
              - main16["median_rel_focal_err_pct"] >= 2.0)

    tags = []
    if not h1:
        tags += ["D_SIGNAL_EXPLAINED_BY_SINGLE_FOCAL_RIG_CONFOUND",
                 "B_REFERENCE_HAND_FOCAL_INFORMATION_ABSENT"]
    else:
        tags.append("A_REFERENCE_HAND_FOCAL_INFORMATION_CONFIRMED")
    if h1 and not h2:
        tags.append("C_SIGNAL_PRESENT_BUT_NOT_GEOMETRIC")
    if not h3:
        tags.append("F_SATURATED_BY_FEW_FRAMES")
    if main16["share_identifiable"] < 0.5:
        tags.append("E_WEAKLY_IDENTIFIABLE")

    verdict = {
        "constant_rig_focal_px": round(const, 3),
        "main_n16": main16, "main_n1": main1,
        "controls_n16_median_rel_focal_err_pct": ctrl16,
        "comparators": comp,
        "paired_main16_minus_constant_oracle": {
            "median_pp": round(dmed, 4), "ci95_lo": round(dlo, 4),
            "ci95_hi": round(dhi, 4),
            "interpretation": "negative means the reference-hand solver beats "
                              "the constant oracle"},
        "H1_reference_hand_carries_focal_information": h1,
        "H2_the_information_is_geometric": h2,
        "H3_more_frames_help": h3,
        "decision_tags": tags,
        "scope_reminder": "this is an INTERNAL_REFERENCE_HAND_DIAGNOSTIC and an "
                          "UPPER BOUND under an oracle reference hand; it is "
                          "not a deployable calibration method and its numbers "
                          "must never be quoted as achievable performance",
    }
    write_json(SUM / "cam006_verdict.json", verdict)
    print(json.dumps({k: verdict[k] for k in
                      ("constant_rig_focal_px",
                       "controls_n16_median_rel_focal_err_pct",
                       "H1_reference_hand_carries_focal_information",
                       "H2_the_information_is_geometric",
                       "H3_more_frames_help", "decision_tags")}, indent=1))
    print("MAIN  N=16:", main16)
    print("MAIN  N=1 :", main1)
    for c in comp:
        print(f"  {c['comparator']:32s} {c['median_rel_focal_err_pct']:8.3f} %")


if __name__ == "__main__":
    import json
    main()
