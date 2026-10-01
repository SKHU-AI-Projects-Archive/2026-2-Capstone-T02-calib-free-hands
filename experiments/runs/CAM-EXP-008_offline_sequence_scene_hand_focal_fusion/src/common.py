"""Shared paths and helpers for CAM-EXP-008.

Offline sequence-level scene + hand geometry focal fusion.

The whole experiment turns on ONE paired comparison:

    S0  L_total(f) = L_scene(f)
    S1  L_total(f) = L_scene(f) + lambda * L_hand(f)

on the identical sequence, identical frame IDs and identical scene
predictions. If that pairing breaks, the primary result is invalid.

Vocabulary:
  * target 2D are "dataset-provided 2D observations", never "GT2D"
  * the evaluation target is the "dataset-provided reference focal", never
    "true physical focal"
  * the hand condition is an INTERNAL INFORMATION CEILING, not a deployable
    method
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

RUN_DIR = Path(__file__).resolve().parents[1]
RUNS = RUN_DIR.parent
EXP = RUNS.parent
REPO = EXP.parent
MANIFESTS = EXP / "manifests"

R0013 = RUNS / "CAM-EXP-001_3_gigahands_multiview_triangulation"
R0041 = RUNS / "CAM-EXP-004_1_pre_cam005_robustness_audit"
R005 = RUNS / "CAM-EXP-005_static_view_bias_attribution"
R006 = RUNS / "CAM-EXP-006_reference_hand_focal_information"

QC_MANIFEST = MANIFESTS / "gigahands_demo_qc_v1.csv.gz"
CAMERA_MANIFEST = MANIFESTS / "gigahands_demo_camera_benchmark_v1.csv.gz"
FRAMES64 = MANIFESTS / "gigahands_demo_cam_exp_0041_64frames_v1.csv.gz"
PRED64 = R0041 / "results" / "raw" / "extended_frame_predictions.csv.gz"

# PHASE B ONLY - the reference focal. Never imported by Phase A code.
VIEW_TARGETS = R005 / "results" / "raw" / "view_targets.csv.gz"

RAW = RUN_DIR / "results" / "raw"
SUM = RUN_DIR / "results" / "summary"
TAB = RUN_DIR / "tables"
FIG = RUN_DIR / "figures"
CACHE = RUN_DIR / "cache"

SEED = 20260929
N_BOOT = 10000

# CAM-EXP-001.3 reconstruction settings, reused unchanged
LOCO_THRESHOLD_PX = 8.0
LOCO_MIN_INLIER_CAMERAS = 4
MIN_PAIRED_JOINTS = 12

# frozen focal grid: f = q * f_scene
Q_MIN, Q_MAX, Q_N = 0.5, 1.5, 161
LAMBDAS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
FRAME_COUNTS = (8, 16, 32, 64)
SIGMA_FLOOR = 0.02
HUBER_DELTA = 1.345

ANYCALIB_MODEL_ID = "anycalib_gen"
ANYCALIB_CAM_ID = "radial:2"


def q_grid() -> np.ndarray:
    return np.geomspace(Q_MIN, Q_MAX, Q_N)


def read_csv(path) -> list[dict]:
    path = Path(path)
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames, seen = [], set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hash_obj(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def fnum(x, default=float("nan")) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def loco_src_on_path() -> None:
    """Make CAM-EXP-001.3's frozen reconstruction code importable."""
    import sys
    for p in (str(R0013 / "src"), str(REPO)):
        if p not in sys.path:
            sys.path.insert(0, p)


# ------------------------------------------------------------------ robust
def mad_sigma(logs) -> float:
    a = np.asarray(logs, float)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return SIGMA_FLOOR
    s = 1.4826 * np.median(np.abs(a - np.median(a)))
    return float(max(s, SIGMA_FLOOR))


def huber(z, delta=HUBER_DELTA):
    z = np.abs(np.asarray(z, float))
    return np.where(z <= delta, 0.5 * z ** 2, delta * (z - 0.5 * delta))


# ------------------------------------------------------------- statistics
def cluster_bootstrap(values, clusters, stat=np.median, n_boot=N_BOOT,
                      seed=SEED):
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    v, c = v[ok], np.asarray(clusters)[ok]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    uniq = np.unique(c)
    members = [np.flatnonzero(c == u) for u in uniq]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([members[p] for p in pick])
        draws[b] = stat(v[idx])
    return (float(stat(v)), float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)))


def _rank(a):
    a = np.asarray(a, float)
    o = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), float)
    r[o] = np.arange(1, len(a) + 1)
    s = a[o]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return r


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return float("nan")
    return pearson(_rank(x), _rank(y))


def corr_cluster_ci(x, y, clusters, method="spearman", n_boot=N_BOOT,
                    seed=SEED):
    f = spearman if method == "spearman" else pearson
    x, y = np.asarray(x, float), np.asarray(y, float)
    c = np.asarray(clusters)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, c = x[ok], y[ok], c[ok]
    if x.size < 8:
        return float("nan"), float("nan"), float("nan"), int(x.size)
    s0 = f(x, y)
    uniq = np.unique(c)
    members = [np.flatnonzero(c == u) for u in uniq]
    rng = np.random.default_rng(seed)
    draws = np.full(n_boot, np.nan)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([members[p] for p in pick])
        draws[b] = f(x[idx], y[idx])
    d = draws[np.isfinite(draws)]
    lo, hi = ((float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
              if d.size else (float("nan"), float("nan")))
    return float(s0), lo, hi, int(x.size)
