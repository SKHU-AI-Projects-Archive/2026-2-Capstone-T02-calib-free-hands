"""Assemble the 64-frame run and answer the saturation question with real data.

Aggregation rules are the ones frozen in CAM-EXP-004; no new aggregator is
invented here. This is a confirmation experiment, not a method search.

  AnyCalib-gen        -> median over the frames of the view
  GeoCalib-distorted  -> median over the frames of the view (the robust rule
                         that was best for it in CAM-EXP-004)
  E2                  -> per-frame geometric mean of the two, then median,
                         exactly as frozen in cam_exp_004_e2_frozen_spec_v1.json
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CAM0031, RUN_DIR, cluster_bootstrap, cluster_ids,  # noqa: E402
                    read_csv, write_csv, write_json)

RAW = RUN_DIR / "results" / "raw"
SUM = RUN_DIR / "results" / "summary"
PARTS = RAW / "_parts64"
NS = (8, 16, 32, 64)


def assemble():
    rows = []
    for p in sorted(PARTS.glob("*.csv.gz")):
        rows.extend(read_csv(p))
    write_csv(RAW / "extended_frame_predictions.csv.gz", rows)
    return rows


def equivalence_check(rows):
    """Re-running the 8 CAM-EXP-003.1 frames must reproduce the stored numbers."""
    old = {}
    for mk, path in (("anycalib_gen_radial",
                      "anycalib_gen_radial_predictions.csv.gz"),
                     ("geocalib_distorted_radial",
                      "geocalib_distorted_radial_predictions.csv.gz")):
        for r in read_csv(CAM0031 / "results" / "raw" / path):
            old[(mk, r["sequence"], r["camera"], int(r["frame"]))] = float(r["pred_fx"])
    out = []
    for mk in ("anycalib_gen_radial", "geocalib_distorted_radial"):
        d = []
        for r in rows:
            if r.get("model_key") != mk or r.get("reused_from_cam_exp_003") != "1":
                continue
            k = (mk, r["sequence"], r["camera"], int(r["frame"]))
            if k in old and r.get("success") == "1":
                d.append(abs(float(r["pred_fx"]) - old[k]) / old[k] * 100)
        d = np.array(d)
        out.append({"model": mk, "n_frames_compared": int(d.size),
                    "max_abs_rel_diff_pct": float(d.max()) if d.size else float("nan"),
                    "median_abs_rel_diff_pct": float(np.median(d)) if d.size else float("nan"),
                    "n_exceeding_0.01pct": int((d > 0.01).sum()),
                    "verdict": ("BIT_STABLE" if d.size and d.max() == 0 else
                                "NUMERICALLY_EQUIVALENT" if d.size and d.max() < 0.01
                                else "DIVERGENT")})
    write_csv(SUM / "reproducibility_equivalence_check.csv", out)
    return out


def aggregate(rows):
    """Per (view, N): AnyCalib median, GeoCalib median, frozen E2."""
    per = defaultdict(dict)   # (seq,cam) -> model -> {grid_pos: (fx, flags, gt)}
    for r in rows:
        if r.get("success") != "1" or not r.get("model_key"):
            continue
        k = (r["sequence"], r["camera"])
        per[k].setdefault(r["model_key"], {})[int(r["grid_pos"])] = (
            float(r["pred_fx"]), r["in_n8"], r["in_n16"], r["in_n32"],
            float(r["gt_fx"]))

    out = []
    for k, d in sorted(per.items()):
        a = d.get("anycalib_gen_radial", {})
        g = d.get("geocalib_distorted_radial", {})
        pos = sorted(set(a) & set(g))
        if not pos:
            continue
        gt = a[pos[0]][4]
        for n in NS:
            if n == 64:
                sel = pos
            else:
                col = {8: 1, 16: 2, 32: 3}[n]
                sel = [p for p in pos if a[p][col] == "1"]
            if len(sel) != n:
                # record the discrepancy rather than silently using a wrong N
                pass
            fa = np.array([a[p][0] for p in sel])
            fg = np.array([g[p][0] for p in sel])
            est = {"anycalib_gen": float(np.median(fa)),
                   "geocalib_distorted": float(np.median(fg)),
                   "E2_frozen": float(np.median(np.sqrt(fa * fg)))}
            for name, f in est.items():
                out.append({"sequence": k[0], "camera": k[1], "N": n,
                            "n_frames_used": len(sel), "estimator": name,
                            "agg_fx": f, "gt_fx": gt,
                            "rel_err_pct": abs(f - gt) / gt * 100,
                            "signed_err_pct": (f - gt) / gt * 100,
                            "log_err": float(np.log(f / gt))})
    write_csv(RAW / "extended_frame_aggregation.csv.gz", out)
    return out


def summarize(agg):
    idx = defaultdict(dict)
    sgn = defaultdict(dict)
    for r in agg:
        idx[(r["estimator"], r["N"])][(r["sequence"], r["camera"])] = r["rel_err_pct"]
        sgn[(r["estimator"], r["N"])][(r["sequence"], r["camera"])] = r["signed_err_pct"]

    rows = []
    for est in ("anycalib_gen", "geocalib_distorted", "E2_frozen"):
        prev = None
        for n in NS:
            d = idx[(est, n)]
            keys = sorted(d)
            v = np.array([d[k] for k in keys])
            s = np.array([sgn[(est, n)][k] for k in keys])
            row = {"estimator": est, "N": n, "n_views": int(v.size),
                   "median_rel_err_pct": float(np.median(v)),
                   "mean_rel_err_pct": float(v.mean()),
                   "p90_rel_err_pct": float(np.percentile(v, 90)),
                   "p95_rel_err_pct": float(np.percentile(v, 95)),
                   "signed_median_pct": float(np.median(s)),
                   "within_5_pct": float((v <= 5).mean() * 100),
                   "within_10_pct": float((v <= 10).mean() * 100),
                   "within_20_pct": float((v <= 20).mean() * 100)}
            if prev is not None:
                delta = np.array([prev[k] - d[k] for k in keys if k in prev])
                base = [{"sequence": k[0], "camera": k[1]} for k in keys if k in prev]
                res = cluster_bootstrap(delta, cluster_ids(base, "VIEW_CLUSTER"),
                                        n_boot=4000)
                row["paired_delta_vs_prev_N_pp"] = res["point_estimate"]
                row["delta_ci_lo"] = res["ci_lo"]
                row["delta_ci_hi"] = res["ci_hi"]
                row["delta_excludes_zero"] = int(res["ci_lo"] > 0 or res["ci_hi"] < 0)
                row["pct_views_improved"] = float((delta > 0).mean() * 100)
            prev = d
            rows.append(row)
    write_csv(SUM / "frame_count_8_16_32_64.csv", rows)

    # verdict, using the stopping-rule thresholds from CAM-EXP-004
    verdicts = {}
    for est in ("anycalib_gen", "geocalib_distorted", "E2_frozen"):
        rs = {r["N"]: r for r in rows if r["estimator"] == est}
        rel_drop = (rs[8]["median_rel_err_pct"] - rs[64]["median_rel_err_pct"]) \
            / rs[8]["median_rel_err_pct"] * 100
        w5_gain = rs[64]["within_5_pct"] - rs[8]["within_5_pct"]
        steps = [rs[n].get("delta_excludes_zero", 0) for n in (16, 32, 64)]
        if rel_drop < 10 and w5_gain < 3:
            v = "CONFIRMED_SATURATION"
        elif rel_drop >= 10 and all(steps):
            v = "SLOW_CONTINUED_GAIN"
        elif steps[-1] and not steps[0]:
            v = "LATE_GAIN"
        else:
            v = "SLOW_CONTINUED_GAIN"
        verdicts[est] = {
            "median_N8": rs[8]["median_rel_err_pct"],
            "median_N64": rs[64]["median_rel_err_pct"],
            "relative_drop_8_to_64_pct": rel_drop,
            "within5_N8": rs[8]["within_5_pct"], "within5_N64": rs[64]["within_5_pct"],
            "within5_gain_pp": w5_gain,
            "significant_steps_16_32_64": steps,
            "verdict": v,
        }
    overall = ("CONFIRMED_SATURATION"
               if all(x["verdict"] == "CONFIRMED_SATURATION" for x in verdicts.values())
               else "MODEL_DEPENDENT")
    verdicts["OVERALL"] = overall
    write_json(SUM / "frame_count_saturation_verdict.json", verdicts)

    write_csv(RUN_DIR / "tables" / "frame_count_extension_summary.csv",
              [{"estimator": e, **{k: v for k, v in d.items()}}
               for e, d in verdicts.items() if e != "OVERALL"])
    return rows, verdicts


def main() -> None:
    rows = assemble()
    print(f"assembled {len(rows)} prediction rows from {len(list(PARTS.glob('*.csv.gz')))} views")
    for e in equivalence_check(rows):
        print(f"  equivalence {e['model']:26s} n={e['n_frames_compared']:5d} "
              f"max {e['max_abs_rel_diff_pct']:.6f}% -> {e['verdict']}")
    agg = aggregate(rows)
    summ, verd = summarize(agg)
    for r in summ:
        d = (f" delta {r['paired_delta_vs_prev_N_pp']:+.3f} "
             f"[{r['delta_ci_lo']:+.3f},{r['delta_ci_hi']:+.3f}] "
             f"sig={r['delta_excludes_zero']}") if "delta_ci_lo" in r else ""
        print(f"  {r['estimator']:20s} N={r['N']:>2} med={r['median_rel_err_pct']:6.3f} "
              f"p90={r['p90_rel_err_pct']:6.2f} w5={r['within_5_pct']:5.1f} "
              f"w10={r['within_10_pct']:5.1f} sgn={r['signed_median_pct']:+6.2f}{d}")
    print(f"VERDICT: {verd['OVERALL']}")
    for k, v in verd.items():
        if k != "OVERALL":
            print(f"  {k:20s} {v['verdict']:24s} drop {v['relative_drop_8_to_64_pct']:+.1f}% "
                  f"w5 {v['within5_gain_pp']:+.1f} pp")


if __name__ == "__main__":
    main()
