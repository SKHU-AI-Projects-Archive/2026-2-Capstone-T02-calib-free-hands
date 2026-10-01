"""Assemble frame features into view features, add the model-self and
model-disagreement groups, and audit feature quality.

The quality audit uses TARGET-INDEPENDENT criteria only: missingness, constant
or near-zero variance, and non-finite values. No feature is dropped because of
how it correlates with the bias.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (MANIFESTS, PRED64, PRIMARY_MODEL, RAW, SECONDARY_MODEL,  # noqa: E402
                    SUM, TAB, fnum, read_csv, read_json, write_csv, write_json)

IQR_FEATURES = {"self_pred_focal_log_iqr", "self_pred_k1_iqr",
                "self_pp_offset_iqr", "disagree_log_iqr"}
NAN_RATE_MAX = 0.20          # fixed before the target was examined
MIN_REL_VARIANCE = 1e-10


def assemble_frames():
    parts = sorted((RAW / "_feature_parts").glob("*.csv.gz"))
    rows = []
    for p in parts:
        rows.extend(read_csv(p))
    write_csv(RAW / "frame_scene_features.csv.gz", rows)
    return rows


def model_self_features():
    """From the frozen CAM-EXP-004.1 predictions - no new inference."""
    per = defaultdict(lambda: defaultdict(list))
    for r in read_csv(PRED64):
        if r.get("success") != "1":
            continue
        key = (r["sequence"], r["camera"])
        mk = r["model_key"]
        w = fnum(r.get("image_width")), fnum(r.get("image_height"))
        fx, fy = fnum(r.get("pred_fx")), fnum(r.get("pred_fy"))
        d = per[key][mk]
        d.append({
            "frame": int(r["frame"]), "fx": fx, "fy": fy,
            "k1": fnum(r.get("extra_k1")), "k2": fnum(r.get("extra_k2")),
            "cx": fnum(r.get("pred_cx")), "cy": fnum(r.get("pred_cy")),
            "W": w[0], "H": w[1],
        })
    out = {}
    for key, models in per.items():
        a = models.get(PRIMARY_MODEL, [])
        g = models.get(SECONDARY_MODEL, [])
        if not a:
            continue
        fx = np.array([x["fx"] for x in a], float)
        fy = np.array([x["fy"] for x in a], float)
        k1 = np.array([x["k1"] for x in a], float)
        k2 = np.array([x["k2"] for x in a], float)
        cx = np.array([x["cx"] for x in a], float)
        cy = np.array([x["cy"] for x in a], float)
        W, H = a[0]["W"], a[0]["H"]
        diag = float(np.hypot(W, H))
        logfx = np.log(fx[fx > 0]) if np.any(fx > 0) else np.array([np.nan])
        pp = np.hypot(cx - W / 2.0, cy - H / 2.0) / diag
        f = {
            "self_pred_focal_px": float(np.median(fx)),
            "self_pred_focal_log_iqr": float(np.subtract(*np.percentile(logfx, [75, 25]))),
            "self_pred_k1": float(np.nanmedian(k1)),
            "self_pred_k2": float(np.nanmedian(k2)),
            "self_pred_k1_iqr": float(np.subtract(*np.nanpercentile(k1, [75, 25]))),
            "self_pp_offset_norm": float(np.nanmedian(pp)),
            "self_pp_offset_iqr": float(np.subtract(*np.nanpercentile(pp, [75, 25]))),
            "self_fx_over_fy": float(np.nanmedian(fx / np.where(fy > 0, fy, np.nan))),
        }
        if g:
            gfx = {x["frame"]: x["fx"] for x in g}
            pairs = [(x["fx"], gfx[x["frame"]]) for x in a
                     if x["frame"] in gfx and x["fx"] > 0 and gfx[x["frame"]] > 0]
            if pairs:
                lr = np.array([np.log(p[0] / p[1]) for p in pairs])
                f["disagree_signed_log_any_vs_geo"] = float(np.median(lr))
                f["disagree_abs_log_any_vs_geo"] = float(np.median(np.abs(lr)))
                f["disagree_log_iqr"] = float(np.subtract(*np.percentile(lr, [75, 25])))
        out[key] = f
    return out


def main() -> None:
    frames = assemble_frames()
    print(f"frame feature rows: {len(frames)}")
    feat_cols = [c for c in frames[0] if c.startswith("feature_")]

    by_view = defaultdict(list)
    for r in frames:
        by_view[(r["sequence"], r["camera_id"])].append(r)

    self_feats = model_self_features()
    targets = {(r["sequence"], r["camera"]): r
               for r in read_csv(RAW / "view_targets.csv.gz")}

    views = []
    for key, rs in sorted(by_view.items()):
        row = {"sequence": key[0], "camera": key[1],
               "physical_camera_id": key[1], "n_frames": len(rs)}
        for c in feat_cols:
            v = np.array([fnum(r[c]) for r in rs], float)
            v = v[np.isfinite(v)]
            name = c[len("feature_"):]
            row[name] = float(np.median(v)) if v.size else float("nan")
            row[f"{name}__iqr"] = (float(np.subtract(*np.percentile(v, [75, 25])))
                                   if v.size else float("nan"))
        row.update(self_feats.get(key, {}))
        t = targets.get(key, {})
        for k in ("anycalib_signed_log_bias", "anycalib_abs_log_bias",
                  "anycalib_rel_focal_err_pct", "anycalib_view_focal_px",
                  "geocalib_signed_log_bias", "geocalib_rel_focal_err_pct",
                  "gt_reference_focal_px"):
            if k in t:
                row[k] = fnum(t[k])
        views.append(row)
    write_csv(SUM / "view_scene_features.csv.gz", views)
    print(f"view feature rows: {len(views)}")

    # ---- quality audit (target-independent) --------------------------------
    spec = read_json(MANIFESTS / "cam_exp_005_scene_feature_spec_v1.json")
    declared = {f["name"]: g for g, fs in spec["groups"].items() for f in fs}
    audit = []
    keep = []
    for name, group in declared.items():
        col = name if name in views[0] else None
        if col is None:
            audit.append({"feature": name, "group": group, "status": "NOT_PRODUCED",
                          "reason": "feature was declared but no column was "
                                    "produced by the extractor",
                          "kept_for_probe": 0})
            continue
        v = np.array([r.get(col, np.nan) for r in views], float)
        nan_rate = float(np.mean(~np.isfinite(v)))
        finite = v[np.isfinite(v)]
        var = float(np.var(finite)) if finite.size else 0.0
        scale = float(np.mean(np.abs(finite))) if finite.size else 0.0
        rel_var = var / (scale ** 2 + 1e-12)
        status, reason = "KEPT", ""
        if nan_rate > NAN_RATE_MAX:
            status, reason = "DROPPED_HIGH_MISSINGNESS", f"nan rate {nan_rate:.2f}"
        elif finite.size == 0:
            status, reason = "DROPPED_ALL_NAN", "no finite values"
        elif var == 0:
            status, reason = "DROPPED_CONSTANT", "zero variance"
        elif rel_var < MIN_REL_VARIANCE:
            status, reason = "DROPPED_NEAR_ZERO_VARIANCE", f"relative var {rel_var:.2e}"
        audit.append({
            "feature": name, "group": group, "status": status, "reason": reason,
            "nan_rate": round(nan_rate, 4),
            "min": float(finite.min()) if finite.size else "",
            "median": float(np.median(finite)) if finite.size else "",
            "max": float(finite.max()) if finite.size else "",
            "relative_variance": rel_var,
            "criterion": "TARGET-INDEPENDENT quality only: missingness, constant "
                         "or near-zero variance. No feature was dropped for how "
                         "it correlates with the bias.",
            "kept_for_probe": int(status == "KEPT"),
        })
        if status == "KEPT":
            keep.append(name)
    write_csv(TAB / "scene_feature_quality_audit.csv", audit)
    dropped = [a for a in audit if a["kept_for_probe"] == 0]
    print(f"features declared {len(declared)}, kept {len(keep)}, "
          f"dropped {len(dropped)}")
    for d in dropped:
        print(f"  {d['feature']:38s} {d['status']}  {d['reason']}")
    write_json(SUM / "_kept_features.json", {"kept": keep,
                                             "dropped": [d["feature"] for d in dropped]})


if __name__ == "__main__":
    main()
