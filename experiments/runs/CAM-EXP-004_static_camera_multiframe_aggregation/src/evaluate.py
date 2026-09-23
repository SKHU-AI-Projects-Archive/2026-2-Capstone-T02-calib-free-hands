"""Summarise PHASE A: error vs frame count, methods, ensembles, oracle, agreement.

Every comparison is paired at the VIEW level. Where a view contributes many
frame subsets, the subsets are first collapsed to that view's expected error;
the 175 views, not the ~19k subsets, are the sample.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (AGG_METHODS, MODELS, N_VALUES, RUN_DIR, paired_bootstrap,  # noqa: E402
                    read_csv, spearman, write_csv, write_json)

RAW = RUN_DIR / "results" / "raw"
SUM = RUN_DIR / "results" / "summary"
KEY = ("sequence", "camera")


def view_key(r):
    return (r["sequence"], r["camera"])


def summarize(vals: np.ndarray, signed: np.ndarray | None = None) -> dict:
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    d = {
        "n_views": int(v.size),
        "median_rel_err_pct": float(np.median(v)),
        "mean_rel_err_pct": float(np.mean(v)),
        "p90_rel_err_pct": float(np.percentile(v, 90)),
        "p95_rel_err_pct": float(np.percentile(v, 95)),
        "within_5_pct": float(np.mean(v <= 5) * 100),
        "within_10_pct": float(np.mean(v <= 10) * 100),
        "within_20_pct": float(np.mean(v <= 20) * 100),
    }
    if signed is not None:
        s = np.asarray(signed, float)
        s = s[np.isfinite(s)]
        d["signed_median_err_pct"] = float(np.median(s))
    return d


def main() -> None:
    pv = read_csv(RAW / "per_view_aggregation.csv.gz")
    for r in pv:
        r["N"] = int(r["N"])
        for c in ("mean_rel_err_pct", "median_rel_err_pct", "mean_signed_err_pct",
                  "mean_log_err", "sd_over_subsets_pct"):
            r[c] = float(r[c])

    # index: (model, method, N) -> {view: expected rel err}
    idx = defaultdict(dict)
    sgn = defaultdict(dict)
    for r in pv:
        idx[(r["model"], r["method"], r["N"])][view_key(r)] = r["mean_rel_err_pct"]
        sgn[(r["model"], r["method"], r["N"])][view_key(r)] = r["mean_signed_err_pct"]

    # ---------------------------------------------------- error vs frame count
    rows, thr_rows = [], []
    for mk in MODELS:
        for method in AGG_METHODS:
            base = None
            for n in N_VALUES:
                d = idx[(mk, method, n)]
                views = sorted(d)
                vals = np.array([d[k] for k in views])
                s = np.array([sgn[(mk, method, n)][k] for k in views])
                row = {"model": mk, "model_label": MODELS[mk]["label"],
                       "role": MODELS[mk]["role"], "method": method, "N": n}
                row.update(summarize(vals, s))
                if base is None:
                    base = {k: v for k, v in zip(views, vals)}
                    row["gain_vs_N1_pp"] = 0.0
                    row["gain_vs_N1_ci_lo"] = row["gain_vs_N1_ci_hi"] = 0.0
                    row["frac_views_improved_vs_N1"] = float("nan")
                else:
                    delta = np.array([base[k] - d[k] for k in views])
                    med, lo, hi = paired_bootstrap(delta)
                    row["gain_vs_N1_pp"] = med
                    row["gain_vs_N1_ci_lo"] = lo
                    row["gain_vs_N1_ci_hi"] = hi
                    row["frac_views_improved_vs_N1"] = float(np.mean(delta > 0) * 100)
                    row["frac_views_worsened_vs_N1"] = float(np.mean(delta < 0) * 100)
                rows.append(row)
                thr_rows.append({"model": mk, "method": method, "N": n,
                                 "within_5_pct": row["within_5_pct"],
                                 "within_10_pct": row["within_10_pct"],
                                 "within_20_pct": row["within_20_pct"]})
    write_csv(SUM / "error_vs_frame_count.csv", rows)
    write_csv(SUM / "within_threshold_vs_frame_count.csv", thr_rows)

    # ------------------------------------------------- aggregation method comp
    meth_rows = []
    for mk in MODELS:
        ref = idx[(mk, "median", 8)]
        for method in AGG_METHODS:
            d = idx[(mk, method, 8)]
            views = sorted(set(d) & set(ref))
            vals = np.array([d[k] for k in views])
            delta = np.array([ref[k] - d[k] for k in views])
            med, lo, hi = paired_bootstrap(delta)
            r = {"model": mk, "method": method, "N": 8}
            r.update(summarize(vals))
            r.update({"delta_vs_median_pp": med, "ci_lo": lo, "ci_hi": hi,
                      "frac_improved_vs_median": float(np.mean(delta > 0) * 100)})
            meth_rows.append(r)
    write_csv(SUM / "aggregation_method_summary.csv", meth_rows)

    # ------------------------------------------------------------- ensembles
    ens = read_csv(RAW / "cross_model_ensemble.csv.gz")
    for r in ens:
        r["N"] = int(r["N"])
        r["mean_rel_err_pct"] = float(r["mean_rel_err_pct"])
        r["mean_signed_err_pct"] = float(r["mean_signed_err_pct"])
    eidx, esgn = defaultdict(dict), defaultdict(dict)
    for r in ens:
        eidx[(r["ensemble"], r["N"])][view_key(r)] = r["mean_rel_err_pct"]
        esgn[(r["ensemble"], r["N"])][view_key(r)] = r["mean_signed_err_pct"]

    primary8 = idx[("anycalib_gen", "median", 8)]
    erows = []
    for (ek, n), d in sorted(eidx.items()):
        views = sorted(d)
        vals = np.array([d[k] for k in views])
        s = np.array([esgn[(ek, n)][k] for k in views])
        r = {"ensemble": ek, "N": n}
        r.update(summarize(vals, s))
        delta = np.array([primary8[k] - d[k] for k in views if k in primary8])
        med, lo, hi = paired_bootstrap(delta)
        r.update({"gain_vs_primary_median8_pp": med, "ci_lo": lo, "ci_hi": hi,
                  "frac_views_improved": float(np.mean(delta > 0) * 100),
                  "frac_views_worsened": float(np.mean(delta < 0) * 100)})
        erows.append(r)
    # single models at N=8 with median, on the same table for comparison
    for mk in MODELS:
        d = idx[(mk, "median", 8)]
        views = sorted(d)
        vals = np.array([d[k] for k in views])
        s = np.array([sgn[(mk, "median", 8)][k] for k in views])
        r = {"ensemble": f"SINGLE:{mk}", "N": 8}
        r.update(summarize(vals, s))
        delta = np.array([primary8[k] - d[k] for k in views])
        med, lo, hi = paired_bootstrap(delta)
        r.update({"gain_vs_primary_median8_pp": med, "ci_lo": lo, "ci_hi": hi,
                  "frac_views_improved": float(np.mean(delta > 0) * 100),
                  "frac_views_worsened": float(np.mean(delta < 0) * 100)})
        erows.append(r)
    write_csv(SUM / "ensemble_summary.csv", erows)

    # ---------------------------------------------------------------- oracle
    orc = read_csv(RAW / "oracle_best_frame.csv.gz")
    oidx = defaultdict(dict)
    for r in orc:
        oidx[(r["model"], int(r["N"]))][view_key(r)] = float(
            r["oracle_best_of_n_mean_rel_err_pct"])
    orows = []
    for mk in MODELS:
        for n in N_VALUES:
            d = oidx[(mk, n)]
            views = sorted(d)
            vals = np.array([d[k] for k in views])
            dep = idx[(mk, "median", n)]
            depv = np.array([dep[k] for k in views])
            r = {"model": mk, "N": n, "estimator": "ORACLE_BEST_OF_N",
                 "deployable": 0}
            r.update(summarize(vals))
            r["deployable_median_for_reference_pct"] = float(np.median(depv))
            r["gap_deployable_minus_oracle_pp"] = float(
                np.median(depv) - np.median(vals))
            orows.append(r)
    write_csv(SUM / "oracle_summary.csv", orows)

    # ------------------------------------------------------------- agreement
    ag = read_csv(SUM / "model_agreement_vs_error.csv")
    dis3 = np.array([float(r["disagreement_3model"]) for r in ag])
    dis2 = np.array([float(r["disagreement_2model"]) for r in ag])
    stats = []
    for name, x in (("disagreement_3model", dis3), ("disagreement_2model", dis2)):
        for tgt in ("primary_rel_err_pct", "ensemble_e4_rel_err_pct",
                    "ensemble_e5_rel_err_pct"):
            y = np.array([float(r[tgt]) for r in ag])
            rho, p, n = spearman(x, y)
            # quartile bins of the GT-blind agreement signal
            q = np.quantile(x, [0.25, 0.5, 0.75])
            bins = np.digitize(x, q)
            binned = {f"bin{b}_median_err_pct": float(np.median(y[bins == b]))
                      for b in range(4)}
            row = {"agreement_metric": name, "target": tgt, "n_views": n,
                   "spearman_rho": rho, "spearman_p": p}
            row.update(binned)
            stats.append(row)
    write_csv(SUM / "agreement_correlation.csv", stats)

    # --------------------------------------------------- per camera / sequence
    for level, col in (("camera", "camera"), ("sequence", "sequence")):
        agg = defaultdict(lambda: defaultdict(list))
        for r in pv:
            if r["method"] != "median":
                continue
            agg[(r[col], r["model"], r["N"])]["e"].append(r["mean_rel_err_pct"])
        out = [{level: k[0], "model": k[1], "N": k[2], "n_views": len(v["e"]),
                "median_rel_err_pct": float(np.median(v["e"]))}
               for k, v in sorted(agg.items())]
        write_csv(SUM / f"per_{level}_summary.csv", out)

    # --------------------------------------------------- PHASE B stopping rule
    rule = read_csv(RUN_DIR / "tables" / "stopping_rule.csv")
    prim4 = idx[("anycalib_gen", "median", 4)]
    prim8 = idx[("anycalib_gen", "median", 8)]
    v = sorted(prim8)
    m4 = float(np.median([prim4[k] for k in v]))
    m8 = float(np.median([prim8[k] for k in v]))
    w4 = float(np.mean(np.array([prim4[k] for k in v]) <= 5) * 100)
    w8 = float(np.mean(np.array([prim8[k] for k in v]) <= 5) * 100)
    rel_drop = (m4 - m8) / m4 * 100
    abs_gain = w8 - w4
    crit_a = rel_drop >= 10.0
    crit_b = abs_gain >= 3.0
    decision = {
        "rule_source": "config.json, fixed before the numbers were inspected",
        "primary_model": "anycalib_gen", "aggregation": "median",
        "median_rel_err_pct_N4": m4, "median_rel_err_pct_N8": m8,
        "relative_median_drop_4_to_8_pct": rel_drop,
        "criterion_A_relative_drop_ge_10pct": bool(crit_a),
        "within_5_pct_N4": w4, "within_5_pct_N8": w8,
        "within_5_absolute_gain_pp": abs_gain,
        "criterion_B_within5_gain_ge_3pp": bool(crit_b),
        "run_phase_b": bool(crit_a or crit_b),
        "status": "RUN_PHASE_B" if (crit_a or crit_b) else "SATURATED_BY_8_FRAMES",
    }
    # The pre-registered rule is evaluated on the PRIMARY model alone. The same
    # two thresholds are additionally applied to the secondary model and to the
    # best ensemble as a supplementary check, so the decision to skip PHASE B
    # does not rest on one model saturating while others were still climbing.
    supp = {}
    for name, (d4, d8) in {
        "geocalib_distorted+median": (idx[("geocalib_distorted", "median", 4)],
                                      idx[("geocalib_distorted", "median", 8)]),
        "E4_perview_geomean": (eidx[("E4_perview_geomean", 4)],
                               eidx[("E4_perview_geomean", 8)]),
        "E2_perframe_geomean": (eidx[("E2_perframe_geomean", 4)],
                                eidx[("E2_perframe_geomean", 8)]),
    }.items():
        kk = sorted(set(d4) & set(d8))
        a4 = np.array([d4[k] for k in kk])
        a8 = np.array([d8[k] for k in kk])
        rd = (np.median(a4) - np.median(a8)) / np.median(a4) * 100
        wg = (np.mean(a8 <= 5) - np.mean(a4 <= 5)) * 100
        supp[name] = {"median_N4": float(np.median(a4)),
                      "median_N8": float(np.median(a8)),
                      "relative_drop_pct": float(rd),
                      "within_5_gain_pp": float(wg),
                      "would_trigger_phase_b": bool(rd >= 10.0 or wg >= 3.0)}
    decision["supplementary_check_not_part_of_the_rule"] = supp
    write_json(SUM / "phase_b_decision.json", decision)
    print(f"PHASE B decision: {decision['status']} "
          f"(median {m4:.3f}->{m8:.3f} = {rel_drop:.2f}% rel drop; "
          f"within5 {w4:.1f}->{w8:.1f} = {abs_gain:+.2f} pp)")

    for mk in MODELS:
        for n in N_VALUES:
            d = idx[(mk, "median", n)]
            vals = np.array([d[k] for k in sorted(d)])
            print(f"  {mk:20s} N={n}  median {np.median(vals):6.2f}%  "
                  f"w5 {np.mean(vals <= 5) * 100:5.1f}%  "
                  f"w10 {np.mean(vals <= 10) * 100:5.1f}%  "
                  f"w20 {np.mean(vals <= 20) * 100:5.1f}%")


if __name__ == "__main__":
    main()
