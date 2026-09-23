"""Shared paths, IO and aggregation primitives for CAM-EXP-004.

PHASE A performs zero model inference: it reuses the per-frame predictions
already produced by CAM-EXP-003 and CAM-EXP-003.1 on the frozen 1400-frame
manifest. Those runs are read-only here.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np

RUN_DIR = Path(__file__).resolve().parents[1]
RUNS = RUN_DIR.parent
REPO = RUNS.parents[1]
EXP = RUNS.parent

CAM003 = RUNS / "CAM-EXP-003_single_frame_calibration_benchmark"
CAM0031 = RUNS / "CAM-EXP-003_1_distortion_aware_diagnostic"
FRAMES_CSV = EXP / "manifests" / "gigahands_demo_cam_exp_003_frames_v1.csv.gz"
BENCH_CSV = EXP / "manifests" / "gigahands_demo_camera_benchmark_v1.csv.gz"

# Deployable models only. Condition-C (GT-undistorted) runs from CAM-EXP-003.1
# are oracle diagnostics and are deliberately absent from this table.
MODELS = {
    "anycalib_gen": {
        "label": "AnyCalib-gen + radial:2",
        "role": "PRIMARY",
        "path": CAM0031 / "results" / "raw" / "anycalib_gen_radial_predictions.csv.gz",
        "has_k": True, "has_pp": True,
    },
    "geocalib_distorted": {
        "label": "GeoCalib-distorted + radial",
        "role": "SECONDARY",
        "path": CAM0031 / "results" / "raw" / "geocalib_distorted_radial_predictions.csv.gz",
        "has_k": True, "has_pp": False,   # principal point fixed at image centre
    },
    "pf_uncentered": {
        "label": "PerspectiveFields uncentered (raw)",
        "role": "DIVERSITY_BASELINE",
        "path": CAM003 / "results" / "raw" / "pf_uncentered_predictions.csv.gz",
        "has_k": False, "has_pp": True,
    },
}

AGG_METHODS = ("mean", "median", "geomean", "log_huber")
N_VALUES = (1, 2, 4, 8)

# Huber tuning constant: the standard robust-statistics default giving 95 %
# asymptotic efficiency at the Gaussian. Not tuned on GT.
HUBER_C = 1.345
HUBER_ITERS = 25


# --------------------------------------------------------------------------- IO
def read_csv(path: Path) -> list[dict]:
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(p)


def fnum(s, default=float("nan")) -> float:
    try:
        v = float(s)
    except (TypeError, ValueError):
        return default
    return v


# ------------------------------------------------------------------- view model
def load_views(model_key: str) -> dict[tuple[str, str], dict]:
    """Per-(sequence, camera) arrays of the per-frame predictions.

    The aggregation unit is (sequence, camera): one static camera inside one
    take. Frames are never pooled across sequences even when the same physical
    camera id reappears, because a deployment sees one video at a time.
    """
    spec = MODELS[model_key]
    out: dict[tuple[str, str], dict] = {}
    for r in read_csv(spec["path"]):
        if r.get("success") != "1":
            continue
        key = (r["sequence"], r["camera"])
        v = out.setdefault(key, {"sequence": key[0], "camera": key[1],
                                 "frame": [], "fx": [], "fy": [], "cx": [],
                                 "cy": [], "k1": [], "k2": [], "gt_fx": None,
                                 "gt_fy": None, "gt_cx": None, "gt_cy": None})
        v["frame"].append(int(r["frame"]))
        v["fx"].append(fnum(r["pred_fx"]))
        v["fy"].append(fnum(r["pred_fy"]))
        v["cx"].append(fnum(r.get("pred_cx")))
        v["cy"].append(fnum(r.get("pred_cy")))
        v["k1"].append(fnum(r.get("extra_k1")))
        v["k2"].append(fnum(r.get("extra_k2")))
        for g in ("gt_fx", "gt_fy", "gt_cx", "gt_cy"):
            v[g] = fnum(r[g])
    for v in out.values():
        order = np.argsort(np.asarray(v["frame"]))
        for k in ("frame", "fx", "fy", "cx", "cy", "k1", "k2"):
            v[k] = np.asarray(v[k], dtype=float)[order]
    return out


def gt_distortion() -> dict[tuple[str, str], tuple[float, float]]:
    out = {}
    for r in read_csv(FRAMES_CSV):
        out[(r["sequence"], r["camera"])] = (fnum(r["gt_k1"]), fnum(r["gt_k2"]))
    return out


# ------------------------------------------------------------------ aggregation
def log_huber_location(x: np.ndarray) -> float:
    """Huber M-estimator of location in log space, returned in linear space.

    Iteratively reweighted with scale = MAD / 0.6745 and the standard c = 1.345.
    Degenerates gracefully: with n <= 2, or when the MAD is zero, every residual
    is inside the quadratic region and the result is the log-domain mean
    (= geometric mean), which is the correct Huber answer for those inputs.
    """
    lx = np.log(x)
    mu = float(np.median(lx))
    for _ in range(HUBER_ITERS):
        resid = lx - mu
        mad = float(np.median(np.abs(resid)))
        scale = mad / 0.6745
        if scale <= 1e-12:
            return float(np.exp(np.mean(lx)) if lx.size <= 2 else np.exp(mu))
        u = resid / scale
        au = np.abs(u)
        w = np.where(au <= HUBER_C, 1.0, HUBER_C / np.maximum(au, 1e-12))
        new = float(np.sum(w * lx) / np.sum(w))
        if abs(new - mu) < 1e-12:
            mu = new
            break
        mu = new
    return float(np.exp(mu))


def aggregate(x: np.ndarray, method: str) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    if method == "mean":
        return float(np.mean(x))
    if method == "median":
        return float(np.median(x))
    if method == "geomean":
        if np.any(x <= 0):
            return float("nan")
        return float(np.exp(np.mean(np.log(x))))
    if method == "log_huber":
        if np.any(x <= 0):
            return float("nan")
        return log_huber_location(x)
    raise ValueError(method)


def aggregate_signed(x: np.ndarray, method: str) -> float:
    """For quantities that can be negative or zero (k1, k2, cx, cy).

    Only the two scale-free methods are meaningful there; the log-domain
    methods are not defined, so they are not silently substituted.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.mean(x)) if method == "mean" else float(np.median(x))


# ---------------------------------------------------------------------- metrics
def rel_err_pct(pred: float, gt: float) -> float:
    return abs(pred - gt) / gt * 100.0


def signed_err_pct(pred: float, gt: float) -> float:
    return (pred - gt) / gt * 100.0


def log_err(pred: float, gt: float) -> float:
    return math.log(pred / gt)


def summarize_errors(errs: np.ndarray, signed: np.ndarray | None = None,
                     logs: np.ndarray | None = None) -> dict:
    errs = np.asarray(errs, dtype=float)
    errs = errs[np.isfinite(errs)]
    if errs.size == 0:
        return {"n": 0}
    d = {
        "n": int(errs.size),
        "median_rel_err_pct": float(np.median(errs)),
        "mean_rel_err_pct": float(np.mean(errs)),
        "p90_rel_err_pct": float(np.percentile(errs, 90)),
        "p95_rel_err_pct": float(np.percentile(errs, 95)),
        "within_5_pct": float(np.mean(errs <= 5) * 100),
        "within_10_pct": float(np.mean(errs <= 10) * 100),
        "within_20_pct": float(np.mean(errs <= 20) * 100),
    }
    if signed is not None:
        s = np.asarray(signed, float)
        s = s[np.isfinite(s)]
        d["signed_median_err_pct"] = float(np.median(s)) if s.size else float("nan")
    if logs is not None:
        lg = np.asarray(logs, float)
        lg = lg[np.isfinite(lg)]
        if lg.size:
            d["median_abs_log_err"] = float(np.median(np.abs(lg)))
            d["rms_log_err"] = float(np.sqrt(np.mean(lg ** 2)))
    return d


# ------------------------------------------------------------------- statistics
def paired_bootstrap(delta: np.ndarray, n_boot: int = 2000, seed: int = 20260923):
    """Bootstrap CI for the median paired delta. Views are the resample unit."""
    d = np.asarray(delta, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    meds = np.median(d[idx], axis=1)
    return (float(np.median(d)), float(np.percentile(meds, 2.5)),
            float(np.percentile(meds, 97.5)))


def spearman(a: np.ndarray, b: np.ndarray):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return float("nan"), float("nan"), int(m.sum())
    from scipy import stats
    r = stats.spearmanr(a[m], b[m])
    return float(r.statistic), float(r.pvalue), int(m.sum())
