"""PHASE B: nested lambda selection, paired scoring and the frozen verdict.

This is the first CAM-EXP-008 file that reads the dataset-provided reference
focal. Everything it consumes was frozen before it ran.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (LAMBDAS, MANIFESTS, RAW, SUM, TAB, VIEW_TARGETS,  # noqa: E402
                    cluster_bootstrap, corr_cluster_ci, fnum, read_csv,
                    read_json, spearman, write_csv, write_json)

SPLITS = MANIFESTS / "cam_exp_008_outer_splits_v1.json"
# The frozen spec (section 36) pre-registers ALL_COMMON as the preferred
# primary and N64 only as a fallback approximation. Under
# FULL_DURATION_64_APPROXIMATION, "ALL_COMMON" means every QC-passing frame
# WITHIN the 64-frame grid (median 51 per view), not every frame of the video.
# N64 turned out to be reachable by only 23 of 154 views, because most views
# have fewer than 64 QC-passing frames on the grid, so it is reported as a
# secondary row. The verdict is identical under either choice.
PRIMARY_SUBSET = "ALL_COMMON"


def relerr(hat, ref):
    return 100.0 * np.abs(np.asarray(hat, float) - ref) / ref


def summarise(hat, ref, cams):
    hat, ref = np.asarray(hat, float), np.asarray(ref, float)
    e = relerr(hat, ref)
    med, lo, hi = cluster_bootstrap(e, cams)
    return {
        "n_views": len(hat),
        "median_rel_focal_err_pct": round(med, 4),
        "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
        "p90_rel_focal_err_pct": round(float(np.nanpercentile(e, 90)), 4),
        "signed_median_focal_err_pct": round(
            float(np.nanmedian(100.0 * (hat - ref) / ref)), 4),
        "within_5pct": round(float(np.nanmean(e <= 5)), 4),
        "within_10pct": round(float(np.nanmean(e <= 10)), 4),
        "within_20pct": round(float(np.nanmean(e <= 20)), 4),
        "estimate_cv_pct": round(float(100 * np.nanstd(hat)
                                       / np.nanmean(hat)), 3),
        "reference_cv_pct": round(float(100 * np.std(ref) / np.mean(ref)), 3),
        "tracking_spearman": round(spearman(hat, ref), 4),
    }


def main() -> None:
    rows = read_csv(RAW / "fusion_fold_predictions.csv.gz")
    tgt = {(r["sequence"], r["camera"]): r for r in read_csv(VIEW_TARGETS)}
    splits = read_json(SPLITS)

    # index: (view, subset, condition, lambda) -> f_hat
    idx = {}
    for r in rows:
        key = (r["sequence"], r["camera"], r["subset"], r["condition"],
               float(r["lambda"]) if r["lambda"] != "inf" else np.inf)
        idx[key] = fnum(r["f_hat"])
    views_of = defaultdict(set)
    for r in rows:
        views_of[r["subset"]].add((r["sequence"], r["camera"]))

    def ref_of(v):
        return fnum(tgt[v]["gt_reference_focal_px"])

    def cam_of(v):
        return tgt[v]["physical_camera_id"]

    # ---------- nested lambda selection on TRAINING cameras only ----------
    lam_rows, fold_rows = [], []
    for proto in ("LOCO_PHYSICAL_CAMERA", "LOSO_SEQUENCE"):
        key = ("held_out_physical_camera" if proto.startswith("LOCO")
               else "held_out_sequence")
        for subset in sorted(views_of):
            vs = [v for v in sorted(views_of[subset]) if v in tgt]
            if not vs:
                continue
            for fold in splits[proto]["folds"]:
                ho = fold[key]
                test = [v for v in vs
                        if (cam_of(v) if key.startswith("held_out_phys")
                            else v[0]) == ho]
                train = [v for v in vs if v not in test]
                if not test or len(train) < 5:
                    continue
                for cond in ("S1", "C1", "C2"):
                    best, best_err = None, np.inf
                    for lam in LAMBDAS:
                        if lam == 0:
                            continue
                        h = [idx.get((v[0], v[1], subset, cond, lam))
                             for v in train]
                        r = [ref_of(v) for v in train]
                        ok = [i for i, x in enumerate(h)
                              if x is not None and np.isfinite(x)]
                        if len(ok) < 5:
                            continue
                        e = float(np.median(relerr(
                            [h[i] for i in ok], np.array([r[i] for i in ok]))))
                        if e < best_err:
                            best, best_err = lam, e
                    if best is None:
                        continue
                    lam_rows.append({"protocol": proto, "subset": subset,
                                     "condition": cond, "fold": fold["fold"],
                                     "held_out": ho, "lambda": best,
                                     "train_median_err_pct": round(best_err,
                                                                   4),
                                     "n_train_views": len(train)})
                    for v in test:
                        f1 = idx.get((v[0], v[1], subset, cond, best))
                        f0 = idx.get((v[0], v[1], subset, "S0", 0.0))
                        if f1 is None or f0 is None:
                            continue
                        fold_rows.append({
                            "protocol": proto, "subset": subset,
                            "condition": cond, "fold": fold["fold"],
                            "held_out": ho, "sequence": v[0], "camera": v[1],
                            "physical_camera_id": cam_of(v),
                            "lambda": best, "f_scene_only": f0,
                            "f_fused": f1, "f_reference": ref_of(v)})
    write_csv(RAW / "control_fold_predictions.csv.gz", fold_rows)
    write_csv(TAB / "lambda_selection_summary.csv", lam_rows)

    # ---------- paired summaries -----------------------------------------
    def paired(proto, subset, cond):
        sel = [r for r in fold_rows if r["protocol"] == proto
               and r["subset"] == subset and r["condition"] == cond]
        if not sel:
            return None
        ref = np.array([fnum(r["f_reference"]) for r in sel])
        f0 = np.array([fnum(r["f_scene_only"]) for r in sel])
        f1 = np.array([fnum(r["f_fused"]) for r in sel])
        cams = [r["physical_camera_id"] for r in sel]
        s0, s1 = summarise(f0, ref, cams), summarise(f1, ref, cams)
        d = relerr(f0, ref) - relerr(f1, ref)      # >0 means fusion helped
        gmed, glo, ghi = cluster_bootstrap(d, cams)
        rel_red = (100.0 * (s0["median_rel_focal_err_pct"]
                            - s1["median_rel_focal_err_pct"])
                   / s0["median_rel_focal_err_pct"])
        return {
            "protocol": proto, "subset": subset, "condition": cond,
            "n_views": len(sel),
            "S0_median_err_pct": s0["median_rel_focal_err_pct"],
            "S1_median_err_pct": s1["median_rel_focal_err_pct"],
            "S0_within5": s0["within_5pct"], "S1_within5": s1["within_5pct"],
            "median_paired_gain_pp": round(gmed, 4),
            "gain_ci_lo": round(glo, 4), "gain_ci_hi": round(ghi, 4),
            "relative_reduction_pct": round(rel_red, 4),
            "within5_gain_pp": round(
                (s1["within_5pct"] - s0["within_5pct"]) * 100, 3),
            "share_views_improved": round(float(np.mean(d > 0)), 4),
            "S1_estimate_cv_pct": s1["estimate_cv_pct"],
            "reference_cv_pct": s1["reference_cv_pct"],
            "S1_tracking_spearman": s1["tracking_spearman"],
            "S0_tracking_spearman": s0["tracking_spearman"],
            "median_lambda": float(np.median(
                [fnum(r["lambda"]) for r in sel])),
        }

    pg = [p for proto in ("LOCO_PHYSICAL_CAMERA", "LOSO_SEQUENCE")
          for subset in sorted(views_of)
          for cond in ("S1", "C1", "C2")
          if (p := paired(proto, subset, cond)) is not None]
    write_csv(SUM / "paired_gain_summary.csv", pg)

    # scene-only and H0 across subsets (not fold dependent)
    so, h0 = [], []
    for subset in sorted(views_of):
        vs = [v for v in sorted(views_of[subset]) if v in tgt]
        ref = np.array([ref_of(v) for v in vs])
        cams = [cam_of(v) for v in vs]
        f0 = [idx.get((v[0], v[1], subset, "S0", 0.0)) for v in vs]
        ok = [i for i, x in enumerate(f0) if x is not None and np.isfinite(x)]
        if len(ok) > 5:
            so.append({"subset": subset,
                       **summarise([f0[i] for i in ok], ref[ok],
                                   [cams[i] for i in ok])})
        fh = [idx.get((v[0], v[1], subset, "H0", np.inf)) for v in vs]
        ok = [i for i, x in enumerate(fh) if x is not None and np.isfinite(x)]
        if len(ok) > 5:
            h0.append({"subset": subset,
                       **summarise([fh[i] for i in ok], ref[ok],
                                   [cams[i] for i in ok])})
    write_csv(SUM / "scene_only_summary.csv", so)
    write_csv(SUM / "hand_only_summary.csv", h0)

    # constant-rig oracle sanity control
    vs = [v for v in sorted(views_of.get(PRIMARY_SUBSET, [])) if v in tgt]
    ref = np.array([ref_of(v) for v in vs])
    cams = [cam_of(v) for v in vs]
    const = float(np.median(ref))
    corc = {"control": "CONSTANT_RIG_FOCAL_ORACLE",
            "status": "ORACLE_SANITY_CONTROL, not deployable",
            **summarise(np.full(len(ref), const), ref, cams)}
    write_csv(SUM / "constant_rig_oracle.csv", [corc])

    # frame-count and pose-diversity views of the same table
    write_csv(SUM / "frame_count_summary.csv",
              [p for p in pg if p["protocol"] == "LOCO_PHYSICAL_CAMERA"
               and p["condition"] == "S1"
               and p["subset"].startswith("N")])
    write_csv(SUM / "pose_diversity_summary.csv",
              [p for p in pg if p["protocol"] == "LOCO_PHYSICAL_CAMERA"
               and p["condition"] == "S1"
               and (p["subset"].startswith("DIVERSE")
                    or p["subset"].startswith("LOW"))])

    # tracking
    tr = []
    for subset in sorted(views_of):
        for cond, lamkey in (("S0", 0.0), ("H0", np.inf)):
            vs = [v for v in sorted(views_of[subset]) if v in tgt]
            h = [idx.get((v[0], v[1], subset, cond, lamkey)) for v in vs]
            ok = [i for i, x in enumerate(h) if x is not None
                  and np.isfinite(x)]
            if len(ok) < 8:
                continue
            r = np.array([ref_of(vs[i]) for i in ok])
            c = [cam_of(vs[i]) for i in ok]
            sp, lo, hi, n = corr_cluster_ci([h[i] for i in ok], r, c)
            tr.append({"subset": subset, "condition": cond, "n_views": n,
                       "spearman": round(sp, 4), "ci_lo": round(lo, 4),
                       "ci_hi": round(hi, 4)})
    for p in pg:
        if p["protocol"] == "LOCO_PHYSICAL_CAMERA" and p["condition"] == "S1":
            tr.append({"subset": p["subset"], "condition": "S1",
                       "n_views": p["n_views"],
                       "spearman": p["S1_tracking_spearman"],
                       "ci_lo": "", "ci_hi": ""})
    write_csv(SUM / "focal_tracking_summary.csv", tr)

    # stress subsets
    stress = []
    for tag, thr in (("S2", 0.02), ("S5", 0.05)):
        keys = {v for v in vs if abs(ref_of(v) - const) / const >= thr}
        for cond in ("S1", "C1", "C2"):
            sel = [r for r in fold_rows
                   if r["protocol"] == "LOCO_PHYSICAL_CAMERA"
                   and r["subset"] == PRIMARY_SUBSET
                   and r["condition"] == cond
                   and (r["sequence"], r["camera"]) in keys]
            if not sel:
                continue
            rr = np.array([fnum(r["f_reference"]) for r in sel])
            e0 = relerr([fnum(r["f_scene_only"]) for r in sel], rr)
            e1 = relerr([fnum(r["f_fused"]) for r in sel], rr)
            ncam = len({r["physical_camera_id"] for r in sel})
            stress.append({
                "subset": tag, "threshold_pct": thr * 100, "condition": cond,
                "n_views": len(sel), "n_physical_cameras": ncam,
                "underpowered": int(len(sel) < 20 or ncam < 8),
                "S0_median_err_pct": round(float(np.median(e0)), 4),
                "S1_median_err_pct": round(float(np.median(e1)), 4),
                "median_gain_pp": round(float(np.median(e0 - e1)), 4)})
    write_csv(SUM / "nondefault_focal_stress_summary.csv", stress)

    # ---------- frozen success criteria ----------------------------------
    prim = next((p for p in pg if p["protocol"] == "LOCO_PHYSICAL_CAMERA"
                 and p["subset"] == PRIMARY_SUBSET
                 and p["condition"] == "S1"), None)
    c1 = next((p for p in pg if p["protocol"] == "LOCO_PHYSICAL_CAMERA"
               and p["subset"] == PRIMARY_SUBSET and p["condition"] == "C1"),
              None)
    c2 = next((p for p in pg if p["protocol"] == "LOCO_PHYSICAL_CAMERA"
               and p["subset"] == PRIMARY_SUBSET and p["condition"] == "C2"),
              None)

    cond1 = bool(prim and (prim["relative_reduction_pct"] >= 10
                           or prim["within5_gain_pp"] >= 5))
    cond2 = bool(prim and prim["gain_ci_lo"] > 0)
    cond3 = bool(prim and c1 and c2
                 and prim["median_paired_gain_pp"]
                 > max(c1["median_paired_gain_pp"],
                       c2["median_paired_gain_pp"]))
    cond4 = bool(prim and prim["S1_estimate_cv_pct"] > 0.5)
    promising = bool(cond1 and cond2 and cond3 and cond4)

    tags = []
    if promising:
        tags.append("SCENE_HAND_FUSION_PROMISING")
    else:
        tags.append("NO_INCREMENTAL_HAND_GEOMETRY_SIGNAL")
    if prim and c1 and c2 and not cond3:
        tags.append("CONTROLS_NOT_DEGRADED")
    loso = next((p for p in pg if p["protocol"] == "LOSO_SEQUENCE"
                 and p["subset"] == PRIMARY_SUBSET
                 and p["condition"] == "S1"), None)
    if loso and prim and loso["relative_reduction_pct"] >= 10 > \
            prim["relative_reduction_pct"]:
        tags.append("HAND_SIGNAL_LOCAL_BUT_NOT_GENERALIZABLE")

    write_json(SUM / "cam008_verdict.json", {
        "mode": "FULL_DURATION_64_APPROXIMATION",
        "primary_subset": PRIMARY_SUBSET,
        "primary_protocol": "LOCO_PHYSICAL_CAMERA",
        "decision_tags": tags,
        "primary_paired": prim,
        "control_C1_wrong_frame": c1,
        "control_C2_view_shuffled": c2,
        "criteria": {
            "cond1_beats_S0_by_threshold": cond1,
            "cond2_gain_ci_excludes_zero": cond2,
            "cond3_beats_controls": cond3,
            "cond4_not_constant_collapse": cond4,
        },
        "scene_only": so, "hand_only": h0,
        "constant_rig_focal_oracle": corc,
        "scope": "INTERNAL INFORMATION CEILING on the GigaHands development "
                 "rig. The hand geometry is OTHER_CAMERA_ONLY_REFERENCE_3D "
                 "paired with dataset-provided 2D observations, which is not "
                 "available at deployment. No deployable method is proposed.",
    })

    print(f"=== PRIMARY {PRIMARY_SUBSET} / LOCO physical camera ===")
    for p in (prim, c1, c2):
        if p:
            print(f"  {p['condition']:3s} S0 {p['S0_median_err_pct']:7.3f}%  "
                  f"S1 {p['S1_median_err_pct']:7.3f}%  "
                  f"gain {p['median_paired_gain_pp']:+7.4f} pp "
                  f"CI [{p['gain_ci_lo']:+.4f}, {p['gain_ci_hi']:+.4f}]  "
                  f"relred {p['relative_reduction_pct']:+6.2f}%  "
                  f"improved {100*p['share_views_improved']:.1f}%")
    print(f"\ncriteria: beats_S0={cond1} ci_excl_0={cond2} "
          f"beats_controls={cond3} not_collapse={cond4}")
    print("TAGS:", tags)


if __name__ == "__main__":
    main()
