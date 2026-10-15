"""Synthetic study: a planar 17-point hand (wrist, 4 x MCP/PIP/DIP/tip; palm ~80 x 100 mm, phalanges 16-44 mm) seen by a pinhole
camera, analysed by the same analysis core as the real videos (thumb joints 1-4 are masked; front / tilt frames are given).

Error = f_est / f_true - 1. This is the only place in experiment 14 where an error against a known f is computed.
"""
from __future__ import annotations

import copy
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import analysis as A

# hand layout in mm (x right, y towards the fingertips = -y in the image when upright); index .. pinky
_MCP = {"index": (-30.0, -92.0), "middle": (-9.0, -100.0), "ring": (12.0, -95.0), "pinky": (31.0, -84.0)}
_BONES = {"index": (40.0, 24.0, 18.0), "middle": (44.0, 28.0, 19.0), "ring": (41.0, 27.0, 18.0), "pinky": (32.0, 20.0, 16.0)}
_FAN = {"index": -6.0, "middle": -1.0, "ring": 4.0, "pinky": 10.0}
_IDS = {"index": [5, 6, 7, 8], "middle": [9, 10, 11, 12], "ring": [13, 14, 15, 16], "pinky": [17, 18, 19, 20]}


def hand_points(arch_mm: float = 0.0) -> np.ndarray:
    """(21, 3) hand-frame points (thumb rows NaN). z = transverse arch offset along the palm normal (MCP row raised in the middle)."""
    P = np.full((21, 3), np.nan)
    P[0] = (0.0, 0.0, 0.0)
    for name, (x, y) in _MCP.items():
        a = np.radians(_FAN[name])
        d = np.array([np.sin(a), -np.cos(a)])
        pts = [np.array([x, y])]
        for L in _BONES[name]:
            pts.append(pts[-1] + L * d)
        z = -arch_mm * np.cos(np.pi * x / 80.0)           # towards the camera (negative z) at the middle of the MCP row
        for j, q in zip(_IDS[name], pts):
            P[j] = (q[0], q[1], z)
    P[[5, 9, 13, 17], 2] = [-arch_mm * np.cos(np.pi * _MCP[k][0] / 80.0) for k in ("index", "middle", "ring", "pinky")]
    return P


def _rot(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def pose(theta_deg, azim_deg, inplane_deg):
    """Hand normal tilted by theta towards azimuth (image plane), after an in-plane rotation."""
    a = np.radians(azim_deg)
    Rt = _rot([-np.sin(a), np.cos(a), 0.0], np.radians(theta_deg)) if theta_deg else np.eye(3)
    return Rt @ _rot([0, 0, 1.0], np.radians(inplane_deg))


def project(P, R, t, f, cx, cy):
    X = P @ R.T + t
    return np.c_[f * X[:, 0] / X[:, 2] + cx, f * X[:, 1] / X[:, 2] + cy]


def generate(cond: dict, scfg: dict, rng) -> dict:
    f = float(scfg["f_px"])
    W, H = scfg["image"]
    cx, cy = W / 2.0, H / 2.0
    D = float(scfg["distance_mm"])
    P = hand_points(cond.get("arch_mm", 0.0))
    P[:, 1] += 95.0                                         # centre the hand (wrist .. tips ~ 190 mm) on the optical axis
    vis_ids = [0] + list(range(5, 21))
    nF, nT = int(cond.get("front_frames", scfg["front_frames"])), int(scfg["tilt_frames"])
    phi = rng.uniform(0, 360)
    g0 = rng.uniform(-10, 10)
    c0 = np.array([rng.uniform(-30, 30), rng.uniform(-8, 8), D * rng.uniform(0.97, 1.05)])
    poses = []
    for _ in range(nF):
        v = cond["delta_deg"] * np.array([np.cos(np.radians(phi)), np.sin(np.radians(phi))]) + rng.normal(0, 1.0, 2)
        th, az = float(np.hypot(*v)), float(np.degrees(np.arctan2(v[1], v[0])))
        poses.append((pose(th, az, g0 + rng.uniform(-5, 5)), c0 + rng.normal(0, 2.0, 3), th))
    lo, hi = cond["tilt_range_deg"]
    for _ in range(nT):
        th = rng.uniform(lo, hi)
        poses.append((pose(th, rng.uniform(0, 360), rng.uniform(-30, 30)),
                      np.array([rng.uniform(-40, 40), rng.uniform(-10, 10), D * rng.uniform(0.95, 1.1)]), th))
    n = len(poses)
    kp = np.full((n, 21, 2), np.nan)
    bias_dir = rng.normal(size=(21, 2))
    bias_dir /= np.linalg.norm(bias_dir, axis=1, keepdims=True)
    noise = cond["noise"]
    for i, (R, t, th) in enumerate(poses):
        q = project(P[vis_ids], R, t, f, cx, cy)
        bl = float(np.ptp(q, axis=0).max() * 1.1)
        sig = noise["px"] if "px" in noise else noise["rel"] * bl
        e = rng.normal(0, sig, q.shape)
        if cond.get("heavy_tail"):
            m = rng.random(len(q)) < 0.10
            e[m] *= 3.0
        q = q + e + cond.get("bias_rel", 0.0) * bl * (th / 45.0) * bias_dir[vis_ids]
        kp[i, vis_ids] = q
    mask = np.isfinite(kp).all(-1) & (kp[..., 0] >= 0) & (kp[..., 0] < W) & (kp[..., 1] >= 0) & (kp[..., 1] < H)
    box = np.full((n, 4), np.nan)
    for i in range(n):
        q = kp[i][mask[i]]
        lo2, hi2 = q.min(0), q.max(0)
        pad = 0.05 * (hi2 - lo2).max()
        box[i] = [lo2[0] - pad, lo2[1] - pad, hi2[0] + pad, hi2[1] + pad]
    times = np.r_[np.arange(nF) / 15.0 + 0.5, 3.0 + np.arange(nT) / 15.0]
    return {"kp": kp, "mask": mask, "present": np.ones(n, bool), "box": box, "cls": np.ones(n, int), "times": times,
            "size": (W, H), "front_idx": np.arange(nF), "tilt_start": 3.0, "f_true": f, "theta_true": [p[2] for p in poses]}


def conditions(scfg: dict) -> list:
    b = copy.deepcopy(scfg["base"])
    def c(name, **kw):
        d = copy.deepcopy(b)
        d.update(kw)
        return {"name": name, **d}
    out = [c("base")]
    out += [c(f"delta{d}", delta_deg=d) for d in (5, 10, 15)]
    out += [c("noise_px3", noise={"px": 3.0})]
    out += [c(f"noise_rel{r:g}", noise={"rel": r / 100}) for r in (0.5, 1, 2, 4, 6)]
    out += [c("heavy_tail", heavy_tail=True), c("tilt10_25", tilt_range_deg=[10, 25])]
    out += [c(f"arch{a}", arch_mm=a) for a in (3, 6)]
    out += [c(f"bias{b_}", bias_rel=b_ / 100) for b_ in (1, 3)]
    out += [c("front30", front_frames=30)]
    out += [c("delta5_rel2", delta_deg=5, noise={"rel": 0.02}), c("delta10_rel2", delta_deg=10, noise={"rel": 0.02}),
            c("rel2_tilt10_25", noise={"rel": 0.02}, tilt_range_deg=[10, 25]),
            c("realistic", delta_deg=5, noise={"rel": 0.02}, heavy_tail=True, arch_mm=3, bias_rel=0.01)]
    return out


def _one(args):
    cond, cfg, rep = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import cv2
    cv2.setNumThreads(1)
    rng = np.random.default_rng([int(cfg["experiment"]["seed"]), rep, abs(hash(cond["name"])) % (2 ** 31)])
    g = generate(cond, cfg["synthetic"], rng)
    t = time.time()
    r = A.analyze(g["kp"], g["mask"], g["present"], g["box"], g["cls"], g["times"], g["size"], cfg, front_idx=g["front_idx"],
                  tilt_start=g["tilt_start"], methods_full=A.METHODS, n_boot=cfg["synthetic"]["n_bootstrap"], seed=rep, budget_s=1e9)
    row = {"cond": cond["name"], "rep": rep, "time_s": time.time() - t, "front_jitter_rel": r.get("front_jitter_rel"), "front_jitter_px": r.get("front_jitter_px")}
    if "per_method" not in r:
        row["failed"] = r.get("failed")
        return row
    for m in A.METHODS:
        pm = r["per_method"][m]
        row[m] = {"f": pm["f"], "err": pm["f"] / g["f_true"] - 1, "verdict": pm["verdict"]["verdict"], "failed_good": pm["verdict"]["failed_good"],
                  "f_rel_uncertainty": pm["f_rel_uncertainty"], "delta_deg": r["solutions"][m]["delta_deg"], "metrics": pm["metrics"]}
    return row


METRIC_KEYS = ("theta_p90", "axis_bins6", "reproj_rms_rel", "jointset_range", "bootstrap_rel_std", "consistency_range", "delta_deg", "sharpness",
               "front_jitter_rel", "method_diff")


def _stats(e):
    e = np.asarray(e, float)
    a = np.abs(e)
    return {"median_abs": float(np.median(a)), "p90_abs": float(np.percentile(a, 90)), "signed_mean": float(e.mean()), "n": int(len(e))}


def summarize(rows: list, conds: list, scfg: dict) -> dict:
    res = {}
    for c in conds:
        rs = [r for r in rows if r["cond"] == c["name"] and "failed" not in r]
        d = {"n_ok": len(rs), "n_failed": sum(1 for r in rows if r["cond"] == c["name"] and "failed" in r), "condition": c,
             "front_jitter_rel_median": float(np.median([r["front_jitter_rel"] for r in rs])) if rs else None}
        for m in A.METHODS:
            if rs:
                d[m] = _stats([r[m]["err"] for r in rs])
                vs = [r[m]["verdict"] for r in rs]
                d[m]["verdicts"] = {v: vs.count(v) for v in ("GOOD", "ROUGH", "REJECT")}
                fails = {}
                for r in rs:
                    for k in r[m]["failed_good"]:
                        fails[k] = fails.get(k, 0) + 1
                d[m]["failed_good_counts"] = fails
                d[m]["f_rel_uncertainty_median"] = float(np.median([r[m]["f_rel_uncertainty"] for r in rs]))
                d[m]["delta_est_median"] = float(np.median([r[m]["delta_deg"] for r in rs]))
                d[m]["metric_medians"] = {k: float(np.nanmedian([np.nan if r[m]["metrics"][k] is None else float(r[m]["metrics"][k]) for r in rs]))
                                          for k in METRIC_KEYS}
                good = [r[m]["err"] for r in rs if r[m]["verdict"] == "GOOD"]
                d[m]["good_only"] = _stats(good) if good else None
        res[c["name"]] = d
    return res


def adoption(summary: dict, conds: list, rule: dict) -> dict:
    """joint_delta is adopted only if, in every condition with delta >= delta_min, its median |error| is at least min_rel_gain (relative) lower
    than fixed_template's, and in every condition with delta = 0 it is not higher (strict). Fixed before the official study."""
    rows, ok = [], True
    for c in conds:
        s = summary[c["name"]]
        if "fixed_template" not in s:
            continue
        a, b = s["fixed_template"]["median_abs"], s["joint_delta"]["median_abs"]
        if c["delta_deg"] >= rule["delta_min_deg"]:
            gain = (a - b) / a if a > 0 else 0.0
            passed = gain >= rule["min_rel_gain"]
            rows.append({"cond": c["name"], "delta": c["delta_deg"], "fixed": a, "joint": b, "rel_gain": gain, "pass": passed, "kind": "delta>=min"})
            ok &= passed
        elif c["delta_deg"] == 0:
            passed = b <= a
            rows.append({"cond": c["name"], "delta": 0, "fixed": a, "joint": b, "pass": passed, "kind": "delta0_not_worse"})
            ok &= passed
    return {"adopt_joint_delta": bool(ok), "estimator": "joint_delta" if ok else "fixed_template", "rows": rows, "rule": rule}


REGRESSION = [  # (condition, method, expected median |error| range) from the specification
    ("base", "fixed_template", (0.004, 0.012)), ("noise_px3", "fixed_template", (0.004, 0.012)),
    ("delta5", "fixed_template", (0.01, 0.05)), ("delta10", "fixed_template", (0.03, 0.10)), ("delta15", "fixed_template", (0.06, 0.12)),
]


def regression(summary: dict) -> list:
    out = []
    for c, m, (lo, hi) in REGRESSION:
        v = summary[c][m]["median_abs"]
        out.append({"cond": c, "method": m, "median_abs": v, "expected": [lo, hi], "within": bool(lo <= v <= hi)})
    return out


def run_study(cfg: dict, log=print) -> dict:
    scfg = cfg["synthetic"]
    conds = conditions(scfg)
    jobs = [(c, cfg, r) for c in conds for r in range(int(scfg["reps"]))]
    t = time.time()
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"                                   # inherited by the spawned workers before numpy loads
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=int(scfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        rows = list(ex.map(_one, jobs, chunksize=1))
    elapsed = time.time() - t
    summary = summarize(rows, conds, scfg)
    return {"rows": rows, "summary": summary, "adoption": adoption(summary, conds, scfg["joint_delta_adopt"]), "regression": regression(summary),
            "elapsed_s": elapsed, "n_jobs": len(jobs), "conditions": conds}


def expected_error(study: dict, jitter_rel: float, method: str) -> dict:
    """Noise-only condition (base, noise_rel*, noise_px3) whose median front jitter (relative) is closest to the video's."""
    cands = [k for k in study["summary"] if k == "base" or k.startswith("noise_")]
    best = min(cands, key=lambda k: abs(np.log(max(study["summary"][k]["front_jitter_rel_median"], 1e-9)) - np.log(max(jitter_rel, 1e-9))))
    s = study["summary"][best]
    return {"condition": best, "front_jitter_rel_condition": s["front_jitter_rel_median"], "front_jitter_rel_video": jitter_rel,
            "median_abs": s[method]["median_abs"], "p90_abs": s[method]["p90_abs"], "method": method,
            "note": "합성 연구에서 정면 지터가 가장 가까운 잡음 조건의 오차(δ=0, 휨·편향 없음 가정). 이 영상의 오차가 아니다."}
