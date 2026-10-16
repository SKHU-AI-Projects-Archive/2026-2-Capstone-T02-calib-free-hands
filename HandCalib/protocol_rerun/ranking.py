"""Internal detector quality ranking (not accuracy): category, then the yaml keys in order; exact ties share the mean position."""
from __future__ import annotations

import math

import numpy as np


def category(d: dict) -> str:
    if "fixed_template" not in d:
        return "FAILED"
    if d["fixed_template"]["at_grid_edge"]:
        return "EDGE"
    return d["verdict"]


def key_values(d: dict) -> dict:
    """The ranking inputs of one detector result (16's summary dict). Missing / not computable -> None."""
    if "fixed_template" not in d:
        return {"sharpness": None, "f_rel_uncertainty": None, "reproj_rms_rel": None, "detect_rate": None}
    sh = d.get("profiles", {}).get("flat17", {}).get("official", {}).get("sharpness")
    unc = None if d["fixed_template"]["at_grid_edge"] else d.get("f_rel_uncertainty")
    return {"sharpness": sh, "f_rel_uncertainty": unc, "reproj_rms_rel": d.get("reproj_rms_rel"), "detect_rate": d.get("detect_rate")}


def _sort_key(cat, vals, rcfg):
    k = [rcfg["category_order"].index(cat)]
    for spec in rcfg["keys"]:
        v = vals.get(spec["name"])
        if v is None or not math.isfinite(v):
            k.append(math.inf)                                   # worst
        else:
            k.append(-v if spec["better"] == "higher" else v)
    return tuple(k)


def rank(dets: dict, rcfg: dict) -> dict:
    """dets: {detector: summary}. Returns {detector: {"rank", "category", "keys"}} with 1 = best; exact ties get the mean position."""
    items = []
    for name, d in dets.items():
        cat = category(d) if "skipped" not in d else "FAILED"
        vals = key_values(d) if "skipped" not in d else key_values({})
        items.append((name, cat, vals, _sort_key(cat, vals, rcfg)))
    items.sort(key=lambda x: x[3])
    out = {}
    i = 0
    while i < len(items):
        j = i
        while j + 1 < len(items) and items[j + 1][3] == items[i][3]:
            j += 1
        r = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            out[items[k][0]] = {"rank": r, "category": items[k][1], "keys": items[k][2], "verdict": dets[items[k][0]].get("verdict")}
        i = j + 1
    return out


def summary(per_video: dict, guide: dict) -> dict:
    """per_video: {video: {detector: rank dict}}; guide: {video: bool}. Mean rank over guide-satisfying videos and category counts."""
    names = sorted({d for v in per_video.values() for d in v})
    out = {}
    for n in names:
        rs = [per_video[v][n]["rank"] for v in per_video if guide.get(v) and n in per_video[v]]
        cats = [per_video[v][n]["category"] for v in per_video if n in per_video[v]]
        verd = [per_video[v][n]["verdict"] for v in per_video if n in per_video[v]]
        out[n] = {"mean_rank_guide": float(np.mean(rs)) if rs else None, "n_guide_videos": len(rs),
                  "verdict_counts": {c: verd.count(c) for c in ("GOOD", "ROUGH", "REJECT")}, "n_edge": cats.count("EDGE"),
                  "n_failed": cats.count("FAILED"), "category_counts": {c: cats.count(c) for c in ("GOOD", "ROUGH", "REJECT", "EDGE", "FAILED")}}
    return out
