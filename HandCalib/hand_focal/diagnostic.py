"""Experiment 14 diagnostics (not pre-registered; never reported as f). Reads the official runs' detection cache and frame list.

D1  cost profiles on frame subsets (theta2d <= 30 / 45 / 60 deg, boxes not touching the image edge) x joint sets
D2  per-frame contribution: slope of each frame's cost in log f, against theta2d, edge contact and frame RMS
D3  same-pose synthetic replay: per-frame PnP poses at an assumed f (0.7 / 1.0 / 1.3 x long side) -> planar template projected,
    real masks kept, front-jitter noise -> fixed_template profile
D4  per-joint residuals by theta2d bin (at the reference f = 1.0 x long side)

Classification rule (fixed before the run):
  - D3 synthetic replays also end at the upper edge of the official grid for every assumed f  -> pose distribution (shooting)
  - D3 synthetic minima are interior and within 10 % of the assumed f for every assumed f, while the real data end at the edge
    -> detector / finger planarity
  - otherwise -> undetermined
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import analysis as A
from . import estimate as E
from . import template as TP
from . import track as TR
from .detectors import adapters as DA

THETA_SUBSETS = (30.0, 45.0, 60.0)
ASSUMED = (0.7, 1.0, 1.3)
THETA_BINS = (0, 15, 30, 45, 60, 91)


def load_official(run_dir: Path, cfg: dict) -> dict:
    """Detections (cached npz copy) + frame list from an official run; tracking and the analysis core are re-run unchanged."""
    res = json.loads((run_dir / "results.json").read_text())
    det = DA._load(run_dir / "detections.npz")
    frames = res["frames"]["list"]
    DA.check_alignment(frames, det)
    W, H = res["normalization"]["output"]["display_w"], res["normalization"]["output"]["display_h"]
    times = np.array([r["time_s"] for r in frames])
    tr = TR.track(det, times, (W, H), cfg["tracking"], front_start_s=float(cfg["phases"]["front_skip_s"]))
    a = A.analyze(tr["kp"], tr["mask"], tr["present"], tr["box"], tr["cls"], times, (W, H), cfg, methods_full=())
    return {"res": res, "tr": tr, "a": a, "W": W, "H": H, "times": times}


def edge_flags(box, present, W, H):
    e = np.zeros(len(box), bool)
    for i in np.flatnonzero(present):
        b = box[i]
        e[i] = bool(b[0] <= 2 or b[1] <= 2 or b[2] >= W - 2 or b[3] >= H - 2)
    return e


def frame_cost_matrix(T, kp, mask, ids_frames, set_pos, set_ids, grid, cx, cy, huber):
    """C[i, k] = Huber mean cost of frame ids_frames[i] at grid[k] (mean over frames = the estimator's cost)."""
    C = np.full((len(ids_frames), len(grid)), np.nan)
    R = np.full((len(ids_frames), len(grid)), np.nan)
    for k, f in enumerate(grid):
        K = E.K_of(f, cx, cy)
        for r, i in enumerate(ids_frames):
            v = mask[i][set_ids]
            if v.sum() < 4:
                continue
            res, _, _ = E.frame_fit(T[set_pos], kp[i][set_ids], v, K)
            C[r, k] = float(E._huber(res, huber[i]).mean())
            R[r, k] = float(np.sqrt(np.mean(res ** 2)))
    return C, R


def profile_stats(grid, c, official_hi):
    """Minimum on the official range and on the extended grid; sharpness min(c(1.15 f), c(f / 1.15)) / c(f) by log interpolation."""
    lg = np.log(grid)
    out = {}
    for name, sel in (("official", grid <= official_hi * (1 + 1e-9)), ("extended", np.ones(len(grid), bool))):
        g, cc, l = grid[sel], c[sel], lg[sel]
        if not np.isfinite(cc).any():
            out[name] = None
            continue
        k = int(np.nanargmin(cc))
        interior = 0 < k < len(g) - 1
        fmin = float(g[k])
        side = [np.interp(np.log(fmin) + d, l, cc, left=np.nan, right=np.nan) for d in (np.log(1.15), -np.log(1.15))]
        side = [s for s in side if np.isfinite(s)]
        out[name] = {"interior": bool(interior), "min_at_px": fmin, "at": "interior" if interior else ("upper" if k == len(g) - 1 else "lower"),
                     "sharpness": float(min(side) / cc[k]) if side and cc[k] > 0 else float("nan")}
    return out


def same_pose_replay(T, kp, mask, used, front, grid, cx, cy, huber, f_assumed, sigma_xy, seed, iters):
    """Poses from PnP at f_assumed on the real points -> template projected -> real mask -> Gaussian noise -> fixed_template profile."""
    rng = np.random.default_rng(seed)
    K = E.K_of(f_assumed, cx, cy)
    obj = np.c_[T, np.zeros(len(T))]
    syn = np.full((max(used) + 1, len(T), 2), np.nan)
    for i in used:
        v = mask[i]
        if v.sum() < 4:
            continue
        _, R, t = E.frame_fit(T, kp[i], v, K)
        X = obj @ R.T + t
        syn[i] = (K[:2, :2] @ (X[:, :2] / X[:, 2:3]).T + K[:2, 2:3]).T + rng.normal(0, sigma_xy, (len(T), 2))
    fr = [i for i in front if mask[i].all()]
    T2 = TP.build(syn[fr], iters)["template"]                      # template rebuilt from the synthetic front frames, as in the pipeline
    frames = [(syn[i], mask[i]) if i < len(syn) else (np.zeros((len(T), 2)), np.zeros(len(T), bool)) for i in range(len(syn))]
    pb = E.Problem(T2, frames, cx, cy, huber[:len(syn)], 4).subset(used)
    c = np.array([E.cost(pb, f) for f in grid])
    return c
