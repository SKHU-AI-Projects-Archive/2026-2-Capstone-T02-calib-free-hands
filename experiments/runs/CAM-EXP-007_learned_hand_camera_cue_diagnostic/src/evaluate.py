"""Evaluate CAM-EXP-007 against the frozen success criteria."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (MANIFESTS, RAW, SUM, TAB, cluster_bootstrap,  # noqa: E402
                    corr_cluster_ci, fnum, read_csv, read_json, spearman,
                    write_csv, write_json)

PROBE_SPEC = MANIFESTS / "cam_exp_007_probe_spec_v1.json"


def metrics(rows):
    y = np.array([fnum(r["y_true"]) for r in rows])
    p = np.array([fnum(r["y_pred"]) for r in rows])
    fc = np.array([fnum(r["f_corrected"]) for r in rows])
    fr = np.array([fnum(r["f_reference"]) for r in rows])
    cam = [r["physical_camera_id"] for r in rows]
    err = 100.0 * np.abs(fc - fr) / fr
    med, lo, hi = cluster_bootstrap(err, cam)
    mae, mlo, mhi = cluster_bootstrap(np.abs(p - y), cam, stat=np.mean)
    sp, splo, sphi, n = corr_cluster_ci(p, y, cam, "spearman")
    return {
        "n_views": len(rows),
        "bias_mae": round(mae, 6), "bias_mae_lo": round(mlo, 6),
        "bias_mae_hi": round(mhi, 6),
        "bias_median_abs_err": round(float(np.median(np.abs(p - y))), 6),
        "spearman_pred_vs_actual": round(sp, 4),
        "spearman_ci_lo": round(splo, 4), "spearman_ci_hi": round(sphi, 4),
        "median_rel_focal_err_pct": round(med, 4),
        "focal_ci_lo": round(lo, 4), "focal_ci_hi": round(hi, 4),
        "p90_rel_focal_err_pct": round(float(np.percentile(err, 90)), 4),
        "signed_median_focal_err_pct": round(
            float(np.median(100.0 * (fc - fr) / fr)), 4),
        "within_5pct": round(float(np.mean(err <= 5)), 4),
        "within_10pct": round(float(np.mean(err <= 10)), 4),
        "within_20pct": round(float(np.mean(err <= 20)), 4),
        "corrected_focal_mean": round(float(np.mean(fc)), 2),
        "corrected_focal_std": round(float(np.std(fc)), 3),
        "corrected_focal_cv_pct": round(
            float(100 * np.std(fc) / np.mean(fc)), 3),
        "reference_focal_cv_pct": round(
            float(100 * np.std(fr) / np.mean(fr)), 3),
        "focal_tracking_spearman": round(spearman(fc, fr), 4),
    }


def main() -> None:
    preds = read_csv(RAW / "probe_fold_predictions.csv.gz")
    ctrl = read_csv(RAW / "shuffle_control_predictions.csv.gz")
    spec = read_json(PROBE_SPEC)

    by = defaultdict(list)
    for r in preds:
        by[(r["protocol"], r["probe"])].append(r)

    names = {r["probe"]: r["probe_name"] for r in preds}
    summ = defaultdict(list)
    for (proto, pid), rows in sorted(by.items()):
        summ[proto].append({"probe": pid, "probe_name": names[pid],
                            "n_features": rows[0]["n_features"],
                            **metrics(rows)})
    loco = sorted(summ["LOCO_PHYSICAL_CAMERA"],
                  key=lambda r: r["median_rel_focal_err_pct"])
    write_csv(SUM / "loco_probe_summary.csv", loco)
    write_csv(SUM / "loso_probe_summary.csv", summ["LOSO_SEQUENCE"])

    L = {r["probe"]: r for r in summ["LOCO_PHYSICAL_CAMERA"]}
    S = {r["probe"]: r for r in summ["LOSO_SEQUENCE"]}
    B1, B2 = L["B1"], L["B2"]

    # ---- controls -------------------------------------------------------
    cby = defaultdict(list)
    for r in ctrl:
        cby[(r["control"], r["probe"], r["seed"])].append(r)
    crows = []
    for (cname, pid, seed), rows in sorted(cby.items()):
        crows.append({"control": cname, "probe": pid, "seed": seed,
                      **metrics(rows)})
    write_csv(SUM / "shuffle_control_summary.csv", crows)

    shuffle_by_probe = defaultdict(list)
    for r in crows:
        if r["control"] == "C2_HAND_FEATURE_VIEW_SHUFFLE":
            shuffle_by_probe[r["probe"]].append(r)

    # ---- frozen success criteria ---------------------------------------
    def beats_b1(r):
        rel = (100.0 * (B1["median_rel_focal_err_pct"]
                        - r["median_rel_focal_err_pct"])
               / B1["median_rel_focal_err_pct"])
        w5 = (r["within_5pct"] - B1["within_5pct"]) * 100
        return rel, w5, bool(rel >= 10 or w5 >= 5)

    def beats_b2(r):
        rel = (100.0 * (B2["median_rel_focal_err_pct"]
                        - r["median_rel_focal_err_pct"])
               / B2["median_rel_focal_err_pct"])
        w5 = (r["within_5pct"] - B2["within_5pct"]) * 100
        return rel, w5, bool(rel >= 10 or w5 >= 5)

    decisions = []
    hand_probes = ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]
    scene_hand = ["P8", "P9", "P10"]
    for pid in hand_probes + scene_hand:
        r = L[pid]
        rel1, w51, c1 = beats_b1(r)
        rel2, w52, c2 = beats_b2(r)
        sh = shuffle_by_probe.get(pid, [])
        sh_med = (float(np.median([s["median_rel_focal_err_pct"]
                                   for s in sh])) if sh else np.nan)
        sh_sp = (float(np.median([s["spearman_pred_vs_actual"]
                                  for s in sh])) if sh else np.nan)
        beats_shuffle = (bool(np.isfinite(sh_med)
                              and r["median_rel_focal_err_pct"] < sh_med
                              and r["spearman_pred_vs_actual"] > sh_sp)
                         if sh else None)
        collapse = bool(r["corrected_focal_cv_pct"] < 0.5
                        and abs(r["focal_tracking_spearman"]) < 0.2)
        cond2 = bool(r["spearman_pred_vs_actual"] > 0
                     and r["spearman_ci_lo"] > 0)
        promising = bool(c1 and cond2 and (beats_shuffle is True)
                         and not collapse)
        decisions.append({
            "probe": pid, "probe_name": r["probe_name"],
            "median_rel_focal_err_pct": r["median_rel_focal_err_pct"],
            "within_5pct": r["within_5pct"],
            "bias_mae": r["bias_mae"],
            "spearman": r["spearman_pred_vs_actual"],
            "spearman_ci_lo": r["spearman_ci_lo"],
            "vs_B1_rel_reduction_pct": round(rel1, 3),
            "vs_B1_within5_gain_pp": round(w51, 3),
            "cond1_beats_B1": int(c1),
            "cond2_positive_spearman_ci": int(cond2),
            "cond3_beats_shuffle": ("" if beats_shuffle is None
                                    else int(beats_shuffle)),
            "cond4_not_constant_collapse": int(not collapse),
            "HAND_DERIVED_SIGNAL_PROMISING": int(promising),
            "vs_B2_scene_rel_reduction_pct": round(rel2, 3),
            "vs_B2_within5_gain_pp": round(w52, 3),
            "SCENE_PLUS_HAND_PROMISING": int(c2 and cond2)
            if pid in scene_hand else "",
            "loso_median_rel_focal_err_pct":
                S[pid]["median_rel_focal_err_pct"],
            "loso_spearman": S[pid]["spearman_pred_vs_actual"],
        })
    write_csv(TAB / "decision_summary.csv", decisions)

    # ---- focal tracking --------------------------------------------------
    track = []
    for pid in ["B0", "B1", "B2", "B3", "C1"] + hand_probes + scene_hand:
        rows = by[("LOCO_PHYSICAL_CAMERA", pid)]
        fc = np.array([fnum(r["f_corrected"]) for r in rows])
        fr = np.array([fnum(r["f_reference"]) for r in rows])
        cam = [r["physical_camera_id"] for r in rows]
        sp, lo, hi, n = corr_cluster_ci(fc, fr, cam, "spearman")
        track.append({
            "probe": pid, "probe_name": names.get(pid, pid), "n_views": n,
            "spearman_corrected_vs_reference": round(sp, 4),
            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "corrected_focal_cv_pct": L[pid]["corrected_focal_cv_pct"],
            "reference_focal_cv_pct": L[pid]["reference_focal_cv_pct"],
            "constant_collapse": int(L[pid]["corrected_focal_cv_pct"] < 0.5),
        })
    write_csv(SUM / "focal_tracking_summary.csv", track)

    # ---- stress subsets --------------------------------------------------
    stress = []
    allrows = by[("LOCO_PHYSICAL_CAMERA", "B1")]
    med_ref = float(np.median([fnum(r["f_reference"]) for r in allrows]))
    for tag, thr in (("S2", 0.02), ("S5", 0.05)):
        keys = {(r["sequence"], r["camera"]) for r in allrows
                if abs(fnum(r["f_reference"]) - med_ref) / med_ref >= thr}
        for pid in ["B1", "B2"] + hand_probes + scene_hand:
            rows = [r for r in by[("LOCO_PHYSICAL_CAMERA", pid)]
                    if (r["sequence"], r["camera"]) in keys]
            if not rows:
                continue
            ncam = len({r["physical_camera_id"] for r in rows})
            fc = np.array([fnum(r["f_corrected"]) for r in rows])
            fr = np.array([fnum(r["f_reference"]) for r in rows])
            e = 100.0 * np.abs(fc - fr) / fr
            stress.append({
                "subset": tag, "threshold_pct": thr * 100, "probe": pid,
                "n_views": len(rows), "n_physical_cameras": ncam,
                "underpowered": int(len(rows) < 20 or ncam < 8),
                "median_rel_focal_err_pct": round(float(np.median(e)), 4),
            })
    write_csv(SUM / "nondefault_focal_stress_summary.csv", stress)

    # ---- scene incremental ----------------------------------------------
    inc = []
    for pid in scene_hand:
        r = L[pid]
        inc.append({
            "probe": pid, "probe_name": r["probe_name"],
            "scene_only_median_err": B2["median_rel_focal_err_pct"],
            "with_hand_median_err": r["median_rel_focal_err_pct"],
            "delta_median_err_pp": round(
                r["median_rel_focal_err_pct"]
                - B2["median_rel_focal_err_pct"], 4),
            "scene_only_within5": B2["within_5pct"],
            "with_hand_within5": r["within_5pct"],
            "delta_within5_pp": round(
                (r["within_5pct"] - B2["within_5pct"]) * 100, 3),
            "scene_only_bias_mae": B2["bias_mae"],
            "with_hand_bias_mae": r["bias_mae"],
            "scene_only_spearman": B2["spearman_pred_vs_actual"],
            "with_hand_spearman": r["spearman_pred_vs_actual"],
        })
    write_csv(SUM / "scene_incremental_summary.csv", inc)

    # ---- verdict ---------------------------------------------------------
    promising = [d["probe"] for d in decisions
                 if d["HAND_DERIVED_SIGNAL_PROMISING"] == 1]
    explicit_ok = any(p in promising for p in ("P1", "P2", "P3", "P4", "P5"))
    latent_ok = any(p in promising for p in ("P6", "P7"))
    scene_inc = [d["probe"] for d in decisions
                 if d.get("SCENE_PLUS_HAND_PROMISING") == 1]
    loso_better = [d["probe"] for d in decisions
                   if d["loso_median_rel_focal_err_pct"]
                   < B1["median_rel_focal_err_pct"] * 0.9
                   and d["cond1_beats_B1"] == 0]

    tags = []
    if explicit_ok:
        tags.append("A_EXPLICIT_HAND_CAMERA_CUE_PROMISING")
    if latent_ok:
        tags.append("B_LATENT_CAMERA_SIGNAL_PROMISING")
    if latent_ok and not explicit_ok:
        tags.append("C_LATENT_ONLY_CAMERA_SIGNAL")
    if scene_inc:
        tags.append("D_HAND_SIGNAL_INCREMENTAL_TO_SCENE")
    if loso_better and not promising:
        tags.append("E_RIG_REPETITION_SIGNAL_ONLY")
    c1r = L["C1"]
    _, _, c1_beats = beats_b1(c1r)
    if c1_beats and not promising:
        tags.append("F_SINGLE_FOCAL_PRIOR_CONFOUND")
    notdeg = [d["probe"] for d in decisions if d["cond3_beats_shuffle"] == 0]
    if notdeg:
        tags.append("G_SHUFFLE_CONTROL_NOT_DEGRADED")
    if not promising:
        tags.append("H_NO_GENERALIZABLE_HAND_DERIVED_SIGNAL")
    cov = read_json(SUM / "feature_extraction_meta.json")
    if cov["views_hand_feature_eligible"] < 0.9 * cov["views_total"]:
        tags.append("I_HAND_FEATURE_COVERAGE_LIMITED")
    if cov.get("views_with_latent", 0) == 0:
        tags.append("J_LATENT_EXTRACTION_UNAVAILABLE")

    write_json(SUM / "cam007_verdict.json", {
        "primary_protocol": "LOCO_PHYSICAL_CAMERA",
        "decision_tags": tags,
        "promising_probes": promising,
        "baselines": {k: L[k] for k in ("B0", "B1", "B2", "B3")},
        "anycalib_focal_prior_control": c1r,
        "decisions": decisions,
        "scene_incremental": inc,
        "shuffle_control_medians": {
            p: {"median_rel_focal_err_pct": float(np.median(
                [s["median_rel_focal_err_pct"] for s in v])),
                "spearman": float(np.median(
                    [s["spearman_pred_vs_actual"] for s in v]))}
            for p, v in shuffle_by_probe.items()},
        "coverage": cov,
        "scope": "LEARNED_HAND_CAMERA_INFORMATION_DIAGNOSTIC. No network was "
                 "trained and no calibration method is proposed. A probe that "
                 "works shows linearly decodable predictive information, not "
                 "that the model 'knows' the focal.",
    })

    print(f"{'probe':5s} {'name':34s} {'medErr%':>8s} {'w5%':>6s} "
          f"{'biasMAE':>8s} {'rho':>7s} {'rhoLo':>7s}")
    for r in loco:
        print(f"{r['probe']:5s} {r['probe_name'][:34]:34s} "
              f"{r['median_rel_focal_err_pct']:8.3f} "
              f"{100*r['within_5pct']:6.1f} {r['bias_mae']:8.5f} "
              f"{r['spearman_pred_vs_actual']:7.3f} "
              f"{r['spearman_ci_lo']:7.3f}")
    print("\nTAGS:", tags)
    print("promising:", promising or "(none)")


if __name__ == "__main__":
    main()
