"""Summarise the benchmark: accuracy, thresholds, stability, failures, runtime.

Every number here is derived from ``results/raw/all_predictions.csv.gz`` alone,
so any table can be regenerated without re-running a model.

Two reporting modes are kept side by side on purpose:
  * success-only  - statistics over the frames a model actually answered;
  * coverage-aware - a failed frame counts as "not within threshold", so a model
    cannot look good by answering only the easy frames.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict

import numpy as np

from common import RUN_DIR, read_csv, rel, write_csv

log = logging.getLogger("cam-exp-003")

THRESHOLDS = (5.0, 10.0, 20.0)
# Reference-only models: reported, never ranked as estimators.
REFERENCE_ONLY = {"ORACLE_GT"}


def num(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def load():
    return read_csv(RUN_DIR / "results" / "raw" / "all_predictions.csv.gz")


def stats(vals) -> dict:
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return {}
    return {"n": int(a.size), "mean": round(float(a.mean()), 4),
            "median": round(float(np.median(a)), 4),
            "p90": round(float(np.percentile(a, 90)), 4),
            "p95": round(float(np.percentile(a, 95)), 4),
            "max": round(float(a.max()), 4)}


def model_summary(rows) -> list:
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    out = []
    for model, rs in sorted(by_model.items()):
        errs = [num(r["relative_focal_error_pct"]) for r in rs if r["success"] == "1"]
        errs = [e for e in errs if e is not None]
        n_att = len(rs)
        n_suc = sum(1 for r in rs if r["success"] == "1")
        e = {"model": model,
             "is_reference_only": int(model in REFERENCE_ONLY),
             "n_attempted": n_att, "n_success": n_suc,
             "failure_rate": round(1 - n_suc / max(n_att, 1), 4)}
        st = stats(errs)
        for k in ("mean", "median", "p90", "p95", "max"):
            e[f"focal_error_pct_{k}"] = st.get(k, "")
        # success-only and coverage-aware threshold rates
        for t in THRESHOLDS:
            within = sum(1 for x in errs if x <= t)
            e[f"within_{int(t)}pct_rate"] = round(within / max(len(errs), 1), 4)
            e[f"within_{int(t)}pct_rate_coverage_aware"] = round(within / max(n_att, 1), 4)
        e["signed_focal_error_pct_median"] = stats(
            [num(r["signed_focal_error_pct"]) for r in rs]).get("median", "")
        e["log_focal_error_median"] = stats(
            [num(r["log_focal_error"]) for r in rs]).get("median", "")
        e["hfov_error_deg_median"] = stats(
            [num(r["hfov_error_deg"]) for r in rs]).get("median", "")
        e["vfov_error_deg_median"] = stats(
            [num(r["vfov_error_deg"]) for r in rs]).get("median", "")
        e["fx_relative_error_pct_median"] = stats(
            [num(r["fx_relative_error_pct"]) for r in rs]).get("median", "")
        e["fy_relative_error_pct_median"] = stats(
            [num(r["fy_relative_error_pct"]) for r in rs]).get("median", "")
        pp = [num(r["principal_point_error_px"]) for r in rs
              if r.get("principal_point_error_px") not in ("NOT_PREDICTED", "", None)]
        pp = [x for x in pp if x is not None]
        if pp:
            e["principal_point_error_px_median"] = round(float(np.median(pp)), 3)
            ppn = [num(r["principal_point_error_norm"]) for r in rs
                   if r.get("principal_point_error_norm") not in ("NOT_PREDICTED", "", None)]
            ppn = [x for x in ppn if x is not None]
            e["principal_point_error_norm_median"] = round(float(np.median(ppn)), 5)
        else:
            e["principal_point_error_px_median"] = "NOT_PREDICTED"
            e["principal_point_error_norm_median"] = "NOT_PREDICTED"
        rt = stats([num(r["runtime_ms"]) for r in rs])
        e["runtime_ms_median"] = rt.get("median", "")
        e["runtime_ms_p90"] = rt.get("p90", "")
        out.append(e)
    return out


def group_summary(rows, key: str) -> list:
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["success"] != "1":
            continue
        v = num(r["relative_focal_error_pct"])
        if v is not None:
            g[r[key]][r["model"]].append(v)
    out = []
    for k, per_model in sorted(g.items()):
        for model, vals in sorted(per_model.items()):
            st = stats(vals)
            out.append({key: k, "model": model, "n": st.get("n", 0),
                        "focal_error_pct_median": st.get("median", ""),
                        "focal_error_pct_p90": st.get("p90", ""),
                        "within_5pct_rate": round(
                            sum(1 for x in vals if x <= 5) / max(len(vals), 1), 4)})
    return out


def stability(rows) -> list:
    """Frame-to-frame spread of the prediction for one static camera.

    The camera does not move and its intrinsics do not change, so any spread
    here is the model's own instability, not a property of the scene geometry.
    This is a diagnostic only - no aggregation across frames is performed.
    """
    g = defaultdict(list)
    for r in rows:
        if r["success"] != "1":
            continue
        f = num(r["pred_fx"])
        if f is not None:
            g[(r["model"], r["sequence"], r["camera"])].append(f)
    out = []
    for (model, seq, cam), vals in sorted(g.items()):
        a = np.asarray(vals, dtype=float)
        if a.size < 2:
            continue
        mean = float(a.mean())
        out.append({"model": model, "sequence": seq, "camera": cam,
                    "n_frames": int(a.size),
                    "pred_focal_mean_px": round(mean, 3),
                    "pred_focal_median_px": round(float(np.median(a)), 3),
                    "pred_focal_std_px": round(float(a.std(ddof=1)), 3),
                    "coefficient_of_variation": round(float(a.std(ddof=1) / mean), 5)
                    if mean else "",
                    "pred_focal_min_px": round(float(a.min()), 3),
                    "pred_focal_max_px": round(float(a.max()), 3),
                    "range_px": round(float(a.max() - a.min()), 3),
                    "range_pct_of_mean": round(float((a.max() - a.min()) / mean * 100), 3)
                    if mean else ""})
    return out


def representative(rows, n: int = 6) -> list:
    """Best / typical / worst frames per model, chosen by focal error only."""
    g = defaultdict(list)
    for r in rows:
        if r["success"] == "1" and num(r["relative_focal_error_pct"]) is not None:
            g[r["model"]].append(r)
    out = []
    for model, rs in sorted(g.items()):
        rs.sort(key=lambda r: num(r["relative_focal_error_pct"]))
        mid = len(rs) // 2
        picks = ([("best", r) for r in rs[:n]]
                 + [("typical", r) for r in rs[max(mid - n // 2, 0):mid + (n + 1) // 2]]
                 + [("worst", r) for r in rs[-n:]])
        for kind, r in picks:
            out.append({"model": model, "kind": kind, "sequence": r["sequence"],
                        "camera": r["camera"], "frame": r["frame"],
                        "gt_fx": r["gt_fx"], "pred_fx": r["pred_fx"],
                        "relative_focal_error_pct": r["relative_focal_error_pct"],
                        "gt_hfov": r["gt_hfov"], "pred_hfov": r["pred_hfov"]})
    return out


def main() -> dict:
    rows = load()
    ms = model_summary(rows)
    write_csv(RUN_DIR / "results" / "summary" / "model_summary.csv", ms)
    write_csv(RUN_DIR / "results" / "summary" / "per_sequence_summary.csv",
              group_summary(rows, "sequence"))
    write_csv(RUN_DIR / "results" / "summary" / "per_camera_summary.csv",
              group_summary(rows, "camera"))
    st = stability(rows)
    write_csv(RUN_DIR / "results" / "summary" / "single_frame_stability.csv", st)

    # stability rolled up per model
    g = defaultdict(list)
    for r in st:
        if r["coefficient_of_variation"] != "":
            g[r["model"]].append(float(r["coefficient_of_variation"]))
    roll = []
    for model, vals in sorted(g.items()):
        s = stats(vals)
        rng = stats([float(r["range_pct_of_mean"]) for r in st
                     if r["model"] == model and r["range_pct_of_mean"] != ""])
        roll.append({"model": model, "n_views": s.get("n", 0),
                     "cv_median": s.get("median", ""), "cv_p90": s.get("p90", ""),
                     "range_pct_of_mean_median": rng.get("median", ""),
                     "range_pct_of_mean_p90": rng.get("p90", "")})
    write_csv(RUN_DIR / "tables" / "stability_by_model.csv", roll)
    write_csv(RUN_DIR / "tables" / "representative_frames.csv", representative(rows))

    ranked = [e for e in ms if not e["is_reference_only"]]
    ranked.sort(key=lambda e: (e["focal_error_pct_median"]
                               if e["focal_error_pct_median"] != "" else 1e9))
    write_csv(RUN_DIR / "tables" / "model_ranking.csv", ranked)
    return {"models": len(ms), "rows": len(rows), "stability_views": len(st)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print(json.dumps(main(), indent=2))
