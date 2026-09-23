"""PHASE A: multi-frame aggregation from the existing 8 frames per view.

Zero model inference. Everything here is computed from the per-frame predictions
already produced by CAM-EXP-003 / 003.1 on the frozen 1400-frame manifest.

Statistical unit
----------------
For N < 8 there are many possible frame subsets per view (C(8,2)=28,
C(8,4)=70). Every subset is evaluated so the answer does not depend on which
frames a single random draw happened to pick. The per-view result is then the
*mean over subsets* - the expected error of a randomly chosen N-frame subset -
and the VIEW is the statistical unit for every comparison. Treating the 70
subsets of a view as independent samples would inflate the sample size ~40x, so
it is never done.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (AGG_METHODS, MODELS, N_VALUES, RUN_DIR, aggregate,  # noqa: E402
                    aggregate_signed, gt_distortion, load_views, log_err,
                    paired_bootstrap, rel_err_pct, signed_err_pct,
                    summarize_errors, write_csv, write_json)

RAW = RUN_DIR / "results" / "raw"
SUM = RUN_DIR / "results" / "summary"


def subsets(n_avail: int, n: int):
    if n > n_avail:
        return []
    return list(combinations(range(n_avail), n))


# --------------------------------------------------------------- single models
def run_single_models():
    views = {k: load_views(k) for k in MODELS}
    gtk = gt_distortion()
    subset_rows, per_view_rows, oracle_rows = [], [], []

    for mk, spec in MODELS.items():
        for key in sorted(views[mk]):
            v = views[mk][key]
            fx, gt = v["fx"], v["gt_fx"]
            n_avail = fx.size
            for n in N_VALUES:
                combos = subsets(n_avail, n)
                if not combos:
                    continue
                # ---- oracle: the best single frame inside the subset (GT used)
                best = [min(rel_err_pct(fx[i], gt) for i in c) for c in combos]
                oracle_rows.append({
                    "model": mk, "sequence": key[0], "camera": key[1], "N": n,
                    "n_subsets": len(combos),
                    "oracle_best_of_n_mean_rel_err_pct": float(np.mean(best)),
                    "oracle_best_of_n_median_rel_err_pct": float(np.median(best)),
                    "note": "ORACLE_BEST_OF_N - uses GT, NOT a deployable method",
                })
                for method in AGG_METHODS:
                    errs, signs, logs = [], [], []
                    for ci, c in enumerate(combos):
                        f = aggregate(fx[list(c)], method)
                        e = rel_err_pct(f, gt)
                        s = signed_err_pct(f, gt)
                        lg = log_err(f, gt)
                        errs.append(e)
                        signs.append(s)
                        logs.append(lg)
                        # full subset table only where it is small enough to be
                        # useful: every combination for N<=4, the single one for 8
                        subset_rows.append({
                            "model": mk, "sequence": key[0], "camera": key[1],
                            "N": n, "subset": "-".join(str(i) for i in c),
                            "method": method, "agg_fx": round(f, 4),
                            "gt_fx": round(gt, 4),
                            "rel_err_pct": round(e, 6),
                            "signed_err_pct": round(s, 6),
                            "log_err": round(lg, 8),
                        })
                    row = {
                        "model": mk, "sequence": key[0], "camera": key[1],
                        "N": n, "method": method, "n_subsets": len(combos),
                        "gt_fx": round(gt, 4),
                        "mean_rel_err_pct": float(np.mean(errs)),
                        "median_rel_err_pct": float(np.median(errs)),
                        "p90_subset_rel_err_pct": float(np.percentile(errs, 90)),
                        "sd_over_subsets_pct": float(np.std(errs)),
                        "mean_signed_err_pct": float(np.mean(signs)),
                        "mean_log_err": float(np.mean(logs)),
                    }
                    # static-camera parameters other than focal, aggregated the
                    # same way, but only where the model actually predicts them
                    if n == max(N_VALUES) and method in ("mean", "median"):
                        if spec["has_pp"] and np.isfinite(v["cx"]).any():
                            row["agg_cx"] = aggregate_signed(v["cx"], method)
                            row["agg_cy"] = aggregate_signed(v["cy"], method)
                            row["gt_cx"] = v["gt_cx"]
                            row["gt_cy"] = v["gt_cy"]
                            row["pp_err_px"] = float(np.hypot(
                                row["agg_cx"] - v["gt_cx"],
                                row["agg_cy"] - v["gt_cy"]))
                        if spec["has_k"] and np.isfinite(v["k1"]).any():
                            row["agg_k1"] = aggregate_signed(v["k1"], method)
                            row["agg_k2"] = aggregate_signed(v["k2"], method)
                            row["gt_k1"], row["gt_k2"] = gtk.get(key, (np.nan, np.nan))
                            row["k1_abs_err"] = abs(row["agg_k1"] - row["gt_k1"])
                    per_view_rows.append(row)

    write_csv(RAW / "phase_a_subset_results.csv.gz", subset_rows)
    write_csv(RAW / "per_view_aggregation.csv.gz", per_view_rows)
    write_csv(RAW / "oracle_best_frame.csv.gz", oracle_rows)
    return views, per_view_rows, oracle_rows


# ------------------------------------------------------------------- ensembles
ENSEMBLES = {
    "E1_perframe_mean": ("per-frame arithmetic mean of AnyCalib-gen and "
                         "GeoCalib-distorted, then median over frames"),
    "E2_perframe_geomean": ("per-frame geometric mean of the two, then median "
                            "over frames"),
    "E3_perview_mean": ("median over frames per model, then arithmetic mean of "
                        "the two model focals"),
    "E4_perview_geomean": ("median over frames per model, then geometric mean "
                           "of the two model focals"),
    "E5_three_model_median": ("median over frames per model, then median of "
                              "AnyCalib-gen, GeoCalib-distorted and PF-uncentered"),
}


def run_ensembles(views):
    """Parameter-free, GT-blind ensembles only. No learned or tuned weights."""
    common = (set(views["anycalib_gen"]) & set(views["geocalib_distorted"])
              & set(views["pf_uncentered"]))
    rows = []
    for key in sorted(common):
        a = views["anycalib_gen"][key]
        g = views["geocalib_distorted"][key]
        p = views["pf_uncentered"][key]
        gt = a["gt_fx"]
        # align by frame id; all three ran the identical frozen manifest
        fa = {int(f): x for f, x in zip(a["frame"], a["fx"])}
        fg = {int(f): x for f, x in zip(g["frame"], g["fx"])}
        fp = {int(f): x for f, x in zip(p["frame"], p["fx"])}
        frames = sorted(set(fa) & set(fg) & set(fp))
        va = np.array([fa[f] for f in frames])
        vg = np.array([fg[f] for f in frames])
        vp = np.array([fp[f] for f in frames])
        for n in N_VALUES:
            combos = subsets(len(frames), n)
            if not combos:
                continue
            acc = {e: [] for e in ENSEMBLES}
            for c in combos:
                c = list(c)
                sa, sg, sp = va[c], vg[c], vp[c]
                acc["E1_perframe_mean"].append(
                    aggregate((sa + sg) / 2.0, "median"))
                acc["E2_perframe_geomean"].append(
                    aggregate(np.sqrt(sa * sg), "median"))
                ma, mg, mp = (aggregate(sa, "median"), aggregate(sg, "median"),
                              aggregate(sp, "median"))
                acc["E3_perview_mean"].append((ma + mg) / 2.0)
                acc["E4_perview_geomean"].append(float(np.sqrt(ma * mg)))
                acc["E5_three_model_median"].append(float(np.median([ma, mg, mp])))
            for ek, fs in acc.items():
                errs = np.array([rel_err_pct(f, gt) for f in fs])
                signs = np.array([signed_err_pct(f, gt) for f in fs])
                logs = np.array([log_err(f, gt) for f in fs])
                rows.append({
                    "ensemble": ek, "sequence": key[0], "camera": key[1],
                    "N": n, "n_subsets": len(combos), "gt_fx": round(gt, 4),
                    "mean_rel_err_pct": float(errs.mean()),
                    "median_rel_err_pct": float(np.median(errs)),
                    "mean_signed_err_pct": float(signs.mean()),
                    "mean_log_err": float(logs.mean()),
                })
    write_csv(RAW / "cross_model_ensemble.csv.gz", rows)
    return rows


# -------------------------------------------------------- bias / noise split
def bias_noise(views):
    """Decompose squared log focal error into between-view bias and within-view
    variation.

    For view v and frame t:    e_vt = log(f_pred_vt / f_gt_v)
    view bias                  b_v  = mean_t e_vt
    within-view residual       r_vt = e_vt - b_v
    Then                mean(e^2) = mean(b^2) + mean(r^2)
    exactly, because the residuals have zero mean inside each view. The two
    terms are reported as fractions of the total.
    """
    rows, summary = [], []
    for mk in MODELS:
        all_e, all_b, all_r, same_sign = [], [], [], []
        for key, v in sorted(views[mk].items()):
            e = np.log(v["fx"] / v["gt_fx"])
            b = float(np.mean(e))
            r = e - b
            # If every frame of a view errs in the same direction, then the mean
            # of the absolute errors equals the absolute error of the mean: mean
            # aggregation cannot reduce the expected error at all, for any N.
            consistent = bool(np.all(np.sign(e) == np.sign(e[0])))
            same_sign.append(consistent)
            rows.append({
                "model": mk, "sequence": key[0], "camera": key[1],
                "n_frames": int(e.size),
                "view_bias_log": b,
                "view_bias_pct": float((np.exp(b) - 1) * 100),
                "abs_view_bias_pct": float(abs(np.exp(b) - 1) * 100),
                "within_view_std_log": float(np.std(e, ddof=0)),
                "within_view_std_pct": float(np.std(e, ddof=0) * 100),
                "within_view_cv_fx": float(np.std(v["fx"]) / np.mean(v["fx"])),
                "mean_sq_log_err": float(np.mean(e ** 2)),
                "all_frames_same_error_sign": int(consistent),
            })
            all_e.append(e ** 2)
            all_b.append(np.full(e.size, b ** 2))
            all_r.append(r ** 2)
        e2 = np.concatenate(all_e)
        b2 = np.concatenate(all_b)
        r2 = np.concatenate(all_r)
        vb = np.array([x["abs_view_bias_pct"] for x in rows if x["model"] == mk])
        wn = np.array([x["within_view_std_pct"] for x in rows if x["model"] == mk])
        summary.append({
            "model": mk,
            "n_views": int(vb.size),
            "pct_views_all_frames_same_error_sign": float(np.mean(same_sign) * 100),
            "total_mean_sq_log_err": float(e2.mean()),
            "bias_component": float(b2.mean()),
            "within_component": float(r2.mean()),
            "bias_fraction_pct": float(b2.mean() / e2.mean() * 100),
            "within_fraction_pct": float(r2.mean() / e2.mean() * 100),
            "median_abs_view_bias_pct": float(np.median(vb)),
            "p90_abs_view_bias_pct": float(np.percentile(vb, 90)),
            "median_within_view_std_pct": float(np.median(wn)),
            "p90_within_view_std_pct": float(np.percentile(wn, 90)),
        })
    write_csv(RAW / "bias_noise_decomposition.csv.gz", rows)
    write_csv(SUM / "bias_noise_summary.csv", summary)
    return rows, summary


# ------------------------------------------------------- agreement diagnostic
def agreement(views):
    """GT-blind spread between the three models vs the error actually made.

    If the spread predicts the error it could later become a deployable
    confidence signal. Measuring that is not the same as proposing it.
    """
    common = (set(views["anycalib_gen"]) & set(views["geocalib_distorted"])
              & set(views["pf_uncentered"]))
    rows = []
    for key in sorted(common):
        vals = {}
        for mk in ("anycalib_gen", "geocalib_distorted", "pf_uncentered"):
            v = views[mk][key]
            vals[mk] = aggregate(v["fx"], "median")
        gt = views["anycalib_gen"][key]["gt_fx"]
        f = np.array(list(vals.values()))
        f2 = np.array([vals["anycalib_gen"], vals["geocalib_distorted"]])
        rows.append({
            "sequence": key[0], "camera": key[1],
            "disagreement_3model": float((f.max() - f.min()) / np.median(f)),
            "log_spread_3model": float(np.log(f.max() / f.min())),
            "disagreement_2model": float(abs(f2[0] - f2[1]) / np.mean(f2)),
            "primary_rel_err_pct": rel_err_pct(vals["anycalib_gen"], gt),
            "ensemble_e4_rel_err_pct": rel_err_pct(float(np.sqrt(f2[0] * f2[1])), gt),
            "ensemble_e5_rel_err_pct": rel_err_pct(float(np.median(f)), gt),
        })
    write_csv(SUM / "model_agreement_vs_error.csv", rows)
    return rows


def main() -> None:
    views, per_view, oracle = run_single_models()
    print(f"single models: {len(per_view)} per-view/N/method rows")
    ens = run_ensembles(views)
    print(f"ensembles: {len(ens)} rows")
    bn, bns = bias_noise(views)
    for s in bns:
        print(f"  {s['model']:20s} bias {s['bias_fraction_pct']:5.1f}% "
              f"within {s['within_fraction_pct']:5.1f}%  "
              f"|bias| med {s['median_abs_view_bias_pct']:.2f}%  "
              f"noise sd {s['median_within_view_std_pct']:.2f}%")
    ag = agreement(views)
    print(f"agreement rows: {len(ag)}")


if __name__ == "__main__":
    main()
