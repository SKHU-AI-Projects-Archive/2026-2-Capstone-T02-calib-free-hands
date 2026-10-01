"""Paired evaluation of the three conditions.

Everything is computed frame-by-frame against CAM-EXP-003's condition-A rows,
so an improvement is always a *paired* difference on the same image rather than
a difference of two medians over possibly different subsets.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict

import numpy as np

from common import CAM003, RUN_DIR, load_frames, read_csv, rel, write_csv

log = logging.getLogger("cam-exp-003.1")

THRESHOLDS = (5.0, 10.0, 20.0)
PARTS = RUN_DIR / "results" / "raw" / "_parts"

# Which condition-A (CAM-EXP-003) model each new run is paired against.
PAIRING = {
    "anycalib_dist_radial": "anycalib",
    "anycalib_gen_radial": "anycalib",
    "geocalib_distorted_radial": "geocalib",
    "undist_anycalib_pinhole": "anycalib",
    "undist_geocalib_pinhole": "geocalib",
    "undist_pf_centered": "pf_centered",
    "undist_pf_uncentered": "pf_uncentered",
}
CONDITION_A_LABEL = "A_RAW_PINHOLE"


def num(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def key_of(r):
    return (r["sequence"], r["camera"], str(int(float(r["frame"]))))


def stats(vals) -> dict:
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return {}
    return {"n": int(a.size), "mean": round(float(a.mean()), 4),
            "median": round(float(np.median(a)), 4),
            "p90": round(float(np.percentile(a, 90)), 4),
            "p95": round(float(np.percentile(a, 95)), 4)}


def assemble() -> dict:
    """Join chunk part files into one predictions file per model run."""
    by_model = defaultdict(list)
    for p in sorted(PARTS.glob("*.csv.gz")):
        by_model[p.stem.rsplit("_", 1)[0]].extend(read_csv(p))
    n_expected = len(load_frames())
    out = {}
    for key, rows in sorted(by_model.items()):
        seen, uniq = set(), []
        for r in rows:
            k = key_of(r)
            if k in seen:
                continue
            seen.add(k)
            uniq.append(r)
        write_csv(RUN_DIR / "results" / "raw" / f"{key}_predictions.csv.gz", uniq)
        out[key] = {"rows": len(uniq), "complete": len(uniq) >= n_expected}
    return out


def load_condition_a() -> dict:
    """CAM-EXP-003 raw + pinhole predictions, reused verbatim."""
    a = {}
    for key in ("anycalib", "geocalib", "pf_centered", "pf_uncentered"):
        p = CAM003 / "results" / "raw" / f"{key}_predictions.csv.gz"
        if p.exists():
            a[key] = {key_of(r): r for r in read_csv(p)}
    return a


def load_new() -> dict:
    out = {}
    for key in PAIRING:
        p = RUN_DIR / "results" / "raw" / f"{key}_predictions.csv.gz"
        if p.exists():
            out[key] = {key_of(r): r for r in read_csv(p)}
    return out


def bootstrap_ci(deltas, n_boot=2000, seed=0):
    """Nonparametric paired bootstrap CI for the median improvement."""
    a = np.asarray([d for d in deltas if d is not None], dtype=float)
    if a.size < 10:
        return "", ""
    rng = np.random.default_rng(seed)
    meds = [float(np.median(rng.choice(a, a.size, replace=True))) for _ in range(n_boot)]
    return round(float(np.percentile(meds, 2.5)), 4), round(float(np.percentile(meds, 97.5)), 4)


def model_summary(cond_a, new) -> list:
    rows = []
    for key, m in sorted(cond_a.items()):
        errs = [num(r["relative_focal_error_pct"]) for r in m.values()
                if r["success"] == "1"]
        rows.append(_summary_row(f"{key} (CAM-EXP-003)", CONDITION_A_LABEL,
                                 list(m.values()), errs))
    for key, m in sorted(new.items()):
        errs = [num(r.get("relative_focal_error_pct")) for r in m.values()
                if str(r.get("success")) == "1"]
        cond = next(iter(m.values())).get("condition", "")
        rows.append(_summary_row(key, cond, list(m.values()), errs))
    return rows


def _summary_row(name, condition, all_rows, errs) -> dict:
    errs = [e for e in errs if e is not None]
    n_att = len(all_rows)
    n_suc = sum(1 for r in all_rows if str(r.get("success")) == "1")
    e = {"model_run": name, "condition": condition,
         "n_attempted": n_att, "n_success": n_suc,
         "failure_rate": round(1 - n_suc / max(n_att, 1), 4)}
    st = stats(errs)
    for k in ("mean", "median", "p90", "p95"):
        e[f"focal_error_pct_{k}"] = st.get(k, "")
    for t in THRESHOLDS:
        e[f"within_{int(t)}pct_rate"] = round(
            sum(1 for x in errs if x <= t) / max(len(errs), 1), 4)
    e["signed_focal_error_pct_median"] = stats(
        [num(r.get("signed_focal_error_pct")) for r in all_rows]).get("median", "")
    e["hfov_error_deg_median"] = stats(
        [num(r.get("hfov_error_deg")) for r in all_rows]).get("median", "")
    pp = [num(r.get("principal_point_error_px")) for r in all_rows
          if r.get("principal_point_error_px") not in ("NOT_PREDICTED", "", None)]
    pp = [x for x in pp if x is not None]
    e["principal_point_error_px_median"] = (round(float(np.median(pp)), 3) if pp
                                            else "NOT_PREDICTED")
    e["runtime_ms_median"] = stats(
        [num(r.get("runtime_ms")) for r in all_rows]).get("median", "")
    # distortion coefficients, where the model predicts them
    k1 = [num(r.get("extra_k1")) for r in all_rows]
    k1 = [x for x in k1 if x is not None]
    e["pred_k1_median"] = round(float(np.median(k1)), 5) if k1 else "NOT_PREDICTED"
    return e


def paired(cond_a, new) -> tuple:
    """Per-frame paired improvement of each new run over its condition-A model."""
    k1map = gt_k1_lookup()
    rows, summary = [], []
    for key, m in sorted(new.items()):
        base_key = PAIRING[key]
        base = cond_a.get(base_key, {})
        deltas, per_cam = [], defaultdict(list)
        for k, r in m.items():
            b = base.get(k)
            if b is None:
                continue
            e_new = num(r.get("relative_focal_error_pct"))
            e_old = num(b.get("relative_focal_error_pct"))
            if e_new is None or e_old is None:
                continue
            d = e_old - e_new                     # positive = new run is better
            deltas.append(d)
            per_cam[r["camera"]].append(d)
            rows.append({
                "model_run": key, "baseline": base_key,
                "condition": r.get("condition", ""),
                "sequence": r["sequence"], "camera": r["camera"], "frame": r["frame"],
                "pinhole_error_pct": round(e_old, 4),
                "new_error_pct": round(e_new, 4),
                "improvement_pct_points": round(d, 4),
                "pinhole_signed_pct": b.get("signed_focal_error_pct", ""),
                "new_signed_pct": r.get("signed_focal_error_pct", ""),
                "gt_abs_k1": k1map.get((r["sequence"], r["camera"]), ""),
            })
        if not deltas:
            continue
        a = np.asarray(deltas)
        lo, hi = bootstrap_ci(deltas)
        summary.append({
            "model_run": key, "baseline": f"{base_key} (raw pinhole)",
            "condition": next(iter(m.values())).get("condition", ""),
            "n_paired": int(a.size),
            "median_improvement_pct_points": round(float(np.median(a)), 4),
            "mean_improvement_pct_points": round(float(a.mean()), 4),
            "ci95_low": lo, "ci95_high": hi,
            "fraction_improved": round(float((a > 0).mean()), 4),
            "fraction_worsened": round(float((a < 0).mean()), 4),
            "median_improvement_by_camera_min": round(
                float(min(np.median(v) for v in per_cam.values())), 4),
            "median_improvement_by_camera_max": round(
                float(max(np.median(v) for v in per_cam.values())), 4),
        })
    return rows, summary


def gt_k1_lookup() -> dict:
    """GT |k1| per (sequence, camera), taken from the frozen frame manifest.

    The prediction rows do not carry it, so it is joined here rather than
    re-derived, keeping one source of truth for the ground truth.
    """
    out = {}
    for r in load_frames():
        out[(r["sequence"], r["camera"])] = abs(float(r.get("gt_k1") or 0.0))
    return out


def per_group(cond_a, new, key_name: str) -> list:
    k1map = gt_k1_lookup()
    out = []
    for key, m in sorted(new.items()):
        base = cond_a.get(PAIRING[key], {})
        g = defaultdict(lambda: {"old": [], "new": [], "k1": []})
        for k, r in m.items():
            b = base.get(k)
            if b is None:
                continue
            e_new, e_old = num(r.get("relative_focal_error_pct")), num(
                b.get("relative_focal_error_pct"))
            if e_new is None or e_old is None:
                continue
            gk = g[r[key_name]]
            gk["old"].append(e_old)
            gk["new"].append(e_new)
            k1 = k1map.get((r["sequence"], r["camera"]))
            if k1 is not None:
                gk["k1"].append(k1)
        for name, d in sorted(g.items()):
            out.append({key_name: name, "model_run": key,
                        "condition": next(iter(m.values())).get("condition", ""),
                        "n": len(d["old"]),
                        "pinhole_median_pct": round(float(np.median(d["old"])), 4),
                        "new_median_pct": round(float(np.median(d["new"])), 4),
                        "improvement_pct_points": round(
                            float(np.median(d["old"]) - np.median(d["new"])), 4),
                        "gt_abs_k1_median": round(float(np.median(d["k1"])), 5)
                        if d["k1"] else ""})
    return out


def distortion_vs_improvement(per_camera) -> list:
    """Does a camera with stronger distortion gain more? Correlation only."""
    from scipy import stats as sps
    out = []
    by_run = defaultdict(list)
    for r in per_camera:
        if r["gt_abs_k1_median"] == "":
            continue
        by_run[r["model_run"]].append((float(r["gt_abs_k1_median"]),
                                       float(r["improvement_pct_points"])))
    for run, pts in sorted(by_run.items()):
        a = np.asarray(pts)
        if a.shape[0] < 5:
            continue
        pr, pp = sps.pearsonr(a[:, 0], a[:, 1])
        sr, sp = sps.spearmanr(a[:, 0], a[:, 1])
        out.append({"model_run": run, "n_cameras": int(a.shape[0]),
                    "pearson_r": round(float(pr), 4), "pearson_p": round(float(pp), 6),
                    "spearman_r": round(float(sr), 4), "spearman_p": round(float(sp), 6),
                    "abs_k1_min": round(float(a[:, 0].min()), 5),
                    "abs_k1_max": round(float(a[:, 0].max()), 5),
                    "note": "correlation only - not evidence of causation"})
    return out


def stability(cond_a, new) -> list:
    """Per-view signed bias and spread, before and after."""
    out = []
    runs = {f"A:{k}": v for k, v in cond_a.items()}
    runs.update(new)
    for name, m in sorted(runs.items()):
        g = defaultdict(list)
        for r in m.values():
            if str(r.get("success")) != "1":
                continue
            s = num(r.get("signed_focal_error_pct"))
            f = num(r.get("pred_fx"))
            if s is not None and f is not None:
                g[(r["sequence"], r["camera"])].append((s, f))
        biases, cvs = [], []
        for (seq, cam), vals in g.items():
            a = np.asarray(vals)
            biases.append(float(np.median(a[:, 0])))
            mean = float(a[:, 1].mean())
            if mean:
                cvs.append(float(a[:, 1].std(ddof=1) / mean))
        if not biases:
            continue
        out.append({"model_run": name, "n_views": len(biases),
                    "median_abs_view_bias_pct": round(
                        float(np.median(np.abs(biases))), 4),
                    "median_signed_view_bias_pct": round(float(np.median(biases)), 4),
                    "view_bias_p90_abs_pct": round(
                        float(np.percentile(np.abs(biases), 90)), 4),
                    "within_view_cv_median": round(float(np.median(cvs)), 5) if cvs else "",
                    "within_view_cv_p90": round(float(np.percentile(cvs, 90)), 5)
                    if cvs else ""})
    return out


def main() -> dict:
    asm = assemble()
    cond_a, new = load_condition_a(), load_new()
    log.info("condition A models: %s | new runs: %s", list(cond_a), list(new))

    write_csv(RUN_DIR / "results" / "summary" / "model_summary.csv",
              model_summary(cond_a, new))
    prows, psum = paired(cond_a, new)
    write_csv(RUN_DIR / "results" / "raw" / "paired_comparisons.csv.gz", prows)
    write_csv(RUN_DIR / "results" / "summary" / "paired_improvement_summary.csv", psum)

    per_cam = per_group(cond_a, new, "camera")
    write_csv(RUN_DIR / "results" / "summary" / "per_camera_summary.csv", per_cam)
    write_csv(RUN_DIR / "results" / "summary" / "per_sequence_summary.csv",
              per_group(cond_a, new, "sequence"))
    write_csv(RUN_DIR / "results" / "summary" / "distortion_vs_improvement.csv",
              distortion_vs_improvement(per_cam))
    write_csv(RUN_DIR / "results" / "summary" / "stability_summary.csv",
              stability(cond_a, new))
    return {"assembled": asm, "n_paired_rows": len(prows)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print(json.dumps(main(), indent=2))
