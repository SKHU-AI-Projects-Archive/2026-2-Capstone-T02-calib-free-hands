"""Build ONE fair 1 -> 64 frame curve from a single inference run.

The problem this solves: CAM-EXP-004 measured N = 1, 2, 4, 8 and CAM-EXP-004.1
measured N = 8, 16, 32, 64, and because GeoCalib is not run-to-run deterministic
the two runs disagree slightly at N = 8 (GeoCalib 11.18 vs 10.86 %, E2 6.14 vs
6.46 %). Splicing them into one line would invent a step that is a run boundary,
not a result.

Instead every N is recomputed from the CAM-EXP-004.1 extended predictions, which
cover all 64 frames of all 175 views in ONE pass. The 8-frame subset is flagged
in the frozen manifest (`in_n8`), so N = 1, 2, 4 are obtained exactly as
CAM-EXP-004 obtained them - all C(8,N) subsets of those same 8 frames, averaged
per view - but from this run's predictions.

This is an aggregation of existing predictions. No inference is run, no
aggregation rule is changed, and the CAM-EXP-004 numbers are left untouched and
reported alongside for comparison.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PKG, R004, R0041, read_csv, write_csv, write_json  # noqa: E402

NS = (1, 2, 4, 8, 16, 32, 64)
ESTIMATORS = ("anycalib_gen", "geocalib_distorted", "E2_frozen")


def load_views():
    per = defaultdict(dict)
    for r in read_csv(R0041 / "results" / "raw" / "extended_frame_predictions.csv.gz"):
        if r.get("success") != "1" or not r.get("model_key"):
            continue
        key = (r["sequence"], r["camera"])
        per[key].setdefault(r["model_key"], {})[int(r["grid_pos"])] = (
            float(r["pred_fx"]), r["in_n8"], r["in_n16"], r["in_n32"],
            float(r["gt_fx"]))
    return per


def estimate(fa, fg, name):
    """The frozen CAM-EXP-004 aggregation rules, unchanged."""
    if name == "anycalib_gen":
        return float(np.median(fa))
    if name == "geocalib_distorted":
        return float(np.median(fg))
    if name == "E2_frozen":
        return float(np.median(np.sqrt(fa * fg)))
    raise ValueError(name)


def main() -> None:
    per = load_views()
    rows = []
    for key, d in sorted(per.items()):
        a = d.get("anycalib_gen_radial", {})
        g = d.get("geocalib_distorted_radial", {})
        pos = sorted(set(a) & set(g))
        if not pos:
            continue
        gt = a[pos[0]][4]
        n8 = [p for p in pos if a[p][1] == "1"]
        for n in NS:
            if n <= 8:
                # every C(8,n) subset of the frozen 8, exactly as CAM-EXP-004 did
                subsets = list(combinations(n8, n))
            else:
                col = {16: 2, 32: 3}.get(n)
                sel = pos if n == 64 else [p for p in pos if a[p][col] == "1"]
                subsets = [tuple(sel)]
            for est in ESTIMATORS:
                errs = []
                for sub in subsets:
                    fa = np.array([a[p][0] for p in sub])
                    fg = np.array([g[p][0] for p in sub])
                    fv = estimate(fa, fg, est)
                    errs.append(abs(fv - gt) / gt * 100)
                rows.append({
                    "sequence": key[0], "camera": key[1], "estimator": est,
                    "N": n, "n_subsets": len(subsets),
                    "n_frames_used": len(subsets[0]),
                    "expected_rel_err_pct": float(np.mean(errs)),
                })
    write_csv(PKG / "figures" / "main" / "data" / "Fig06_per_view.csv.gz", rows)

    idx = defaultdict(dict)
    for r in rows:
        idx[(r["estimator"], r["N"])][(r["sequence"], r["camera"])] = \
            r["expected_rel_err_pct"]

    out = []
    for est in ESTIMATORS:
        for n in NS:
            v = np.array(list(idx[(est, n)].values()))
            out.append({
                "estimator": est, "N": n, "n_views": int(v.size),
                "median_rel_err_pct": round(float(np.median(v)), 3),
                "mean_rel_err_pct": round(float(v.mean()), 3),
                "p90_rel_err_pct": round(float(np.percentile(v, 90)), 3),
                "within_5_pct": round(float((v <= 5).mean() * 100), 2),
                "within_10_pct": round(float((v <= 10).mean() * 100), 2),
                "source_run": "CAM-EXP-004.1 extended predictions (single run)",
            })
    write_csv(PKG / "figures" / "main" / "data" / "Fig06.csv", out)

    # the CAM-EXP-004 originals, kept beside it rather than spliced into it
    pv = read_csv(R004 / "results" / "raw" / "per_view_aggregation.csv.gz")
    ens = read_csv(R004 / "results" / "raw" / "cross_model_ensemble.csv.gz")
    orig = []
    for est, rs, keyname, val in (
            ("anycalib_gen", pv, "model", "anycalib_gen"),
            ("geocalib_distorted", pv, "model", "geocalib_distorted"),
            ("E2_frozen", ens, "ensemble", "E2_perframe_geomean")):
        for n in (1, 2, 4, 8):
            sel = [float(r["mean_rel_err_pct"]) for r in rs
                   if r[keyname] == val and int(r["N"]) == n
                   and (keyname == "ensemble" or r["method"] == "median")]
            v = np.array(sel)
            orig.append({"estimator": est, "N": n, "n_views": int(v.size),
                         "median_rel_err_pct": round(float(np.median(v)), 3),
                         "within_5_pct": round(float((v <= 5).mean() * 100), 2),
                         "source_run": "CAM-EXP-004 original run"})
    write_csv(PKG / "figures" / "main" / "data" / "Fig06_cam004_original.csv", orig)

    # how far apart are the two runs where they overlap?
    cmp_rows = []
    for est in ESTIMATORS:
        for n in (1, 2, 4, 8):
            a = next(r for r in out if r["estimator"] == est and r["N"] == n)
            b = next(r for r in orig if r["estimator"] == est and r["N"] == n)
            cmp_rows.append({
                "estimator": est, "N": n,
                "cam004_original_median_pct": b["median_rel_err_pct"],
                "cam0041_rerun_median_pct": a["median_rel_err_pct"],
                "difference_pp": round(a["median_rel_err_pct"]
                                       - b["median_rel_err_pct"], 3)})
    write_csv(PKG / "tables" / "multiframe_run_agreement.csv", cmp_rows)

    write_json(PKG / "figures" / "main" / "data" / "Fig06_provenance.json", {
        "what": "median relative focal error vs number of aggregated frames, "
                "N = 1..64, recomputed for every N from ONE inference run",
        "why": "CAM-EXP-004 (N<=8) and CAM-EXP-004.1 (N>=8) are separate runs and "
               "GeoCalib is not run-to-run deterministic, so splicing them would "
               "show a step that is a run boundary rather than a result",
        "source_predictions": "experiments/runs/CAM-EXP-004_1_pre_cam005_"
                              "robustness_audit/results/raw/extended_frame_"
                              "predictions.csv.gz",
        "aggregation": "frozen CAM-EXP-004 rules, unchanged: AnyCalib median, "
                       "GeoCalib median, E2 = per-frame geometric mean then median",
        "subsets": "N<=8: all C(8,N) subsets of the frozen 8-frame set, averaged "
                   "per view. N>8: the single nested set from the frozen manifest.",
        "statistical_unit": "view (175)",
        "cam004_original_kept_separately":
            "figures/main/data/Fig06_cam004_original.csv and "
            "tables/multiframe_run_agreement.csv",
        "no_new_inference": True,
    })

    print(f"{len(rows)} per-view rows, {len(out)} curve points")
    for est in ESTIMATORS:
        s = " ".join(f"N{r['N']}={r['median_rel_err_pct']:.2f}"
                     for r in out if r["estimator"] == est)
        print(f"  {est:20s} {s}")
    print("run agreement (rerun - original, pp):")
    for r in cmp_rows:
        print(f"  {r['estimator']:20s} N={r['N']:<2} {r['difference_pp']:+.3f}")


if __name__ == "__main__":
    main()
