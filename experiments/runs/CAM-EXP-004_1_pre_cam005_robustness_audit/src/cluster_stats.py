"""Re-run the CAM-EXP-003.1 and CAM-EXP-004 headline statistics with the
correct resampling unit.

Nothing here changes a point estimate. Only the uncertainty around it changes,
because 8 frames of one static camera are not 8 independent samples of that
camera's calibration error.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CAM0031, CAM004, CLUSTER_LEVELS, RUN_DIR,  # noqa: E402
                    cluster_bootstrap, cluster_ids, read_csv, write_csv)

RAW = RUN_DIR / "results" / "raw"
SUM = RUN_DIR / "results" / "summary"


def frac_improved(vals):
    v = np.asarray(vals, float)
    return float(np.mean(v > 0) * 100), float(np.mean(v < 0) * 100)


# ------------------------------------------------------------------ 003.1
def reanalyse_0031():
    rows = read_csv(CAM0031 / "results" / "raw" / "paired_comparisons.csv.gz")
    by_run = defaultdict(list)
    for r in rows:
        by_run[(r["model_run"], r["baseline"], r["condition"])].append(r)

    raw_rows, summary = [], []
    for (run, base, cond), rs in sorted(by_run.items()):
        delta = np.array([float(r["improvement_pct_points"]) for r in rs])
        up, down = frac_improved(delta)
        for level in CLUSTER_LEVELS:
            cid = cluster_ids(rs, level)
            res = cluster_bootstrap(delta, cid)
            rec = {"source": "CAM-EXP-003.1", "model_run": run, "baseline": base,
                   "condition": cond, "resampling_unit": level,
                   "median_improvement_pp": res["point_estimate"],
                   "ci_lo": res["ci_lo"], "ci_hi": res["ci_hi"],
                   "ci_width": res["ci_hi"] - res["ci_lo"],
                   "excludes_zero": int(res["ci_lo"] > 0 or res["ci_hi"] < 0),
                   "n_rows": res["n_rows"], "n_clusters": res["n_clusters"],
                   "pct_frames_improved": up, "pct_frames_worsened": down,
                   "interpretation": ("sensitivity analysis only - too few "
                                      "clusters for a trustworthy CI"
                                      if level == "SEQUENCE_CLUSTER" else
                                      "as reported in CAM-EXP-003.1 - "
                                      "pseudo-replicated"
                                      if level == "LEGACY_FRAME_BOOTSTRAP"
                                      else "primary")}
            summary.append(rec)
            raw_rows.append(rec)
    write_csv(RAW / "cam0031_cluster_bootstrap.csv.gz", raw_rows)
    write_csv(SUM / "cam0031_statistical_reanalysis.csv", summary)
    return summary


# -------------------------------------------------------------------- 004
def reanalyse_004():
    """AnyCalib-gen N=8 vs the frozen E2 ensemble N=8, per view."""
    pv = read_csv(CAM004 / "results" / "raw" / "per_view_aggregation.csv.gz")
    ens = read_csv(CAM004 / "results" / "raw" / "cross_model_ensemble.csv.gz")

    prim = {(r["sequence"], r["camera"]): r for r in pv
            if r["model"] == "anycalib_gen" and r["method"] == "median"
            and r["N"] == "8"}
    e2 = {(r["sequence"], r["camera"]): r for r in ens
          if r["ensemble"] == "E2_perframe_geomean" and r["N"] == "8"}
    geo = {(r["sequence"], r["camera"]): r for r in pv
           if r["model"] == "geocalib_distorted" and r["method"] == "median"
           and r["N"] == "8"}

    keys = sorted(set(prim) & set(e2))
    base = [{"sequence": k[0], "camera": k[1],
             "primary_err": float(prim[k]["mean_rel_err_pct"]),
             "e2_err": float(e2[k]["mean_rel_err_pct"]),
             "geo_err": float(geo[k]["mean_rel_err_pct"]),
             "primary_signed": float(prim[k]["mean_signed_err_pct"]),
             "e2_signed": float(e2[k]["mean_signed_err_pct"])} for k in keys]

    stats = {
        "median_rel_err_improvement_pp":
            lambda r: r["primary_err"] - r["e2_err"],
        "within_5_delta_indicator":
            lambda r: (r["e2_err"] <= 5) - (r["primary_err"] <= 5),
        "within_10_delta_indicator":
            lambda r: (r["e2_err"] <= 10) - (r["primary_err"] <= 10),
        # signed bias of each estimator separately. (A "reduction in |signed
        # bias|" statistic was dropped: where a view's error has a consistent
        # sign it is algebraically the same number as the error improvement
        # above, so reporting both would double-count one piece of evidence.)
        "signed_bias_primary_pct": lambda r: r["primary_signed"],
        "signed_bias_e2_pct": lambda r: r["e2_signed"],
        "e2_vs_geocalib_improvement_pp":
            lambda r: r["geo_err"] - r["e2_err"],
    }
    raw_rows, summary = [], []
    for name, fn in stats.items():
        vals = np.array([fn(r) for r in base], float)
        stat = np.mean if name.endswith("_indicator") else np.median
        for level in ("VIEW_CLUSTER", "PHYSICAL_CAMERA_CLUSTER",
                      "SEQUENCE_CLUSTER"):
            cid = cluster_ids(base, level)
            res = cluster_bootstrap(vals, cid, stat=stat)
            scale = 100.0 if name.endswith("_indicator") else 1.0
            rec = {"source": "CAM-EXP-004", "comparison":
                   "E2_perframe_geomean vs anycalib_gen (N=8)"
                   if "e2_vs_geocalib" not in name else
                   "E2_perframe_geomean vs geocalib_distorted (N=8)",
                   "statistic": name, "aggregate": stat.__name__,
                   "resampling_unit": level,
                   "value": res["point_estimate"] * scale,
                   "ci_lo": res["ci_lo"] * scale, "ci_hi": res["ci_hi"] * scale,
                   "excludes_zero": int(res["ci_lo"] > 0 or res["ci_hi"] < 0),
                   "n_views": res["n_rows"], "n_clusters": res["n_clusters"],
                   "interpretation": ("sensitivity analysis only - 5 clusters"
                                      if level == "SEQUENCE_CLUSTER"
                                      else "primary")}
            summary.append(rec)
            raw_rows.append(rec)
    write_csv(RAW / "cam004_cluster_bootstrap.csv.gz", raw_rows)
    write_csv(SUM / "cam004_statistical_reanalysis.csv", summary)
    return summary


def statistical_units_table():
    rows = [
        {"analysis": "CAM-EXP-003 single-frame benchmark",
         "unit_used_originally": "frame (1400)",
         "correct_unit": "view (175)",
         "status": "point estimates unaffected; CIs were not the headline there"},
        {"analysis": "CAM-EXP-003.1 paired distortion comparisons",
         "unit_used_originally": "frame (1400)",
         "correct_unit": "view (175)",
         "status": "re-analysed here; see cam0031_statistical_reanalysis.csv"},
        {"analysis": "CAM-EXP-004 aggregation / ensemble",
         "unit_used_originally": "view (175)",
         "correct_unit": "view (175)",
         "status": "already correct; extended here to camera and sequence clusters"},
        {"analysis": "cluster counts",
         "unit_used_originally": "-", "correct_unit": "-",
         "status": "1400 frames, 175 (sequence,camera) views, 40 unique physical "
                   "camera ids, 5 sequences"},
    ]
    write_csv(RUN_DIR / "tables" / "statistical_units.csv", rows)


def main() -> None:
    a = reanalyse_0031()
    print("=== CAM-EXP-003.1 paired improvement, by resampling unit ===")
    for r in a:
        if r["model_run"] in ("anycalib_gen_radial", "geocalib_distorted_radial"):
            print(f"  {r['model_run']:26s} {r['resampling_unit']:24s} "
                  f"{r['median_improvement_pp']:+6.2f} pp "
                  f"[{r['ci_lo']:+6.2f},{r['ci_hi']:+6.2f}] "
                  f"n_clusters={r['n_clusters']:4d} excl0={r['excludes_zero']}")
    b = reanalyse_004()
    print("=== CAM-EXP-004 E2 vs AnyCalib-gen, by resampling unit ===")
    for r in b:
        print(f"  {r['statistic']:34s} {r['resampling_unit']:24s} "
              f"{r['value']:+7.2f} [{r['ci_lo']:+7.2f},{r['ci_hi']:+7.2f}] "
              f"excl0={r['excludes_zero']}")
    statistical_units_table()


if __name__ == "__main__":
    main()
