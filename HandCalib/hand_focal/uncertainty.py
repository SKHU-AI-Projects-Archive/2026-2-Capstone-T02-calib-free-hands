"""Heuristic uncertainty of the hand f: joint-set range, block bootstrap, consistency splits, estimator difference.

f_rel_uncertainty = max(2 x bootstrap relative std, joint-set range / 2, consistency range / 2, estimator difference / 2).
This combination is a heuristic, not a calibrated interval.
"""
from __future__ import annotations

import time

import numpy as np

from . import estimate as E


def rel_range(values, ref):
    v = [x for x in values if np.isfinite(x)]
    return float((max(v) - min(v)) / ref) if len(v) >= 2 and ref else float("nan")


def block_bootstrap(pb: E.Problem, front_ids, tilt_ids, method, grid, ecfg, sigma2, sol, n, block, seed, budget_s):
    """Resample blocks of `block` consecutive tilt frames with replacement (front frames always kept)."""
    rng = np.random.default_rng(seed)
    tilt = list(tilt_ids)
    blocks = [tilt[i:i + block] for i in range(0, len(tilt), block)] or [[]]
    fs, t0, reduced = [], time.time(), False
    start = (np.log(sol["f"]), sol["delta_deg"], sol["phi_deg"])
    for b in range(n):
        if b >= 2 and (time.time() - t0) / b * (b + 1) > budget_s:     # the next repetition would exceed the budget
            reduced = True
            break
        pick = [blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]
        ids = list(front_ids) + [i for blk in pick for i in blk]
        sub = pb.subset(ids)
        r = E.run(sub, method, grid, ecfg, sigma2, start=start if method == "joint_delta" else None)
        fs.append(r["f"])
    fs = np.array(fs)
    return {"f": fs.tolist(), "n_done": int(len(fs)), "n_requested": int(n), "reduced": reduced,
            "rel_std": float(np.std(fs, ddof=1) / np.mean(fs)) if len(fs) > 1 else float("nan")}


def consistency(pb, front_ids, tilt_ids, axis_deg, method, grid, ecfg, sigma2, sol):
    tilt = list(tilt_ids)
    h = len(tilt) // 2
    start = (np.log(sol["f"]), sol["delta_deg"], sol["phi_deg"])
    def f_of(ids):
        if len(ids) < 4:
            return float("nan")
        return E.run(pb.subset(list(front_ids) + list(ids)), method, grid, ecfg, sigma2, start=start if method == "joint_delta" else None)["f"]
    ax = np.asarray([axis_deg[i] for i in tilt])
    horiz = [i for i, a in zip(tilt, ax) if (a < 45 or a >= 135)]
    vert = [i for i, a in zip(tilt, ax) if 45 <= a < 135]
    out = {"first_half": f_of(tilt[:h]), "second_half": f_of(tilt[h:]), "axis_horizontal": f_of(horiz), "axis_vertical": f_of(vert),
           "n_horizontal": len(horiz), "n_vertical": len(vert)}
    ref = sol["f"]
    out["halves_range"] = rel_range([out["first_half"], out["second_half"]], ref)
    out["axis_range"] = rel_range([out["axis_horizontal"], out["axis_vertical"]], ref)
    out["range"] = float(np.nanmax([out["halves_range"], out["axis_range"]])) if np.isfinite([out["halves_range"], out["axis_range"]]).any() else float("nan")
    return out


def combine(boot_rel_std, jointset_range, consistency_range, method_diff) -> float:
    parts = [2 * boot_rel_std, jointset_range / 2, consistency_range / 2, method_diff / 2]
    parts = [p for p in parts if np.isfinite(p)]
    return float(max(parts)) if parts else float("nan")
