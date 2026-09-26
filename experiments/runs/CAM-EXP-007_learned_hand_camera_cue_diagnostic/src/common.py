"""Shared paths and helpers for CAM-EXP-007.

LEARNED_HAND_CAMERA_INFORMATION_DIAGNOSTIC. This run trains no network and
proposes no calibration method.

Vocabulary rules:
  * the released 2D/3D are "dataset-provided annotations", never "GT";
  * the target is AnyCalib's view-level signed log focal residual, reused
    unchanged from CAM-EXP-005;
  * a probe that works is evidence of "linearly decodable predictive
    information", never that a network "knows the focal".

PHASE SEPARATION: this module must never import or expose the target. Feature
extraction code imports only from here and must stay target-blind.
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

R0041 = RUNS / "CAM-EXP-004_1_pre_cam005_robustness_audit"
R005 = RUNS / "CAM-EXP-005_static_view_bias_attribution"
R006 = RUNS / "CAM-EXP-006_reference_hand_focal_information"

# frozen, target-independent frame grid reused from CAM-EXP-006
FRAMES16 = MANIFESTS / "cam_exp_006_reference_hand_frames_v1.csv.gz"
CAMERA_MANIFEST = MANIFESTS / "gigahands_demo_camera_benchmark_v1.csv.gz"

# PHASE B ONLY. Never imported by feature extraction.
VIEW_TARGETS = R005 / "results" / "raw" / "view_targets.csv.gz"
SCENE_FEATURES = R005 / "results" / "summary" / "view_scene_features.csv.gz"
CAM005_KEPT = R005 / "results" / "summary" / "_kept_features.json"

RAW = RUN_DIR / "results" / "raw"
SUM = RUN_DIR / "results" / "summary"
TAB = RUN_DIR / "tables"
FIG = RUN_DIR / "figures"
CACHE = RUN_DIR / "cache"

SEED = 20260928
N_BOOT = 10000
N_FRAMES = 16
MIN_FRAMES_WITH_HAND = 4          # frozen eligibility rule
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
PCA_DIM = 32

MODELS = {"wilor": REPO / "models" / "anyhand_wilor.ckpt",
          "detector": REPO / "models" / "detector.pt"}


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


def fnum(x, default=float("nan")) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------- statistics
def cluster_bootstrap(values, clusters, stat=np.median, n_boot=N_BOOT,
                      seed=SEED):
    """Resample whole physical-camera clusters with replacement."""
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


def _rankdata(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1)
    # average ties
    sa = a[order]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


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
    return pearson(_rankdata(x), _rankdata(y))


def corr_cluster_ci(x, y, clusters, method="spearman", n_boot=N_BOOT,
                    seed=SEED):
    f = spearman if method == "spearman" else pearson
    x, y = np.asarray(x, float), np.asarray(y, float)
    c = np.asarray(clusters)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, c = x[ok], y[ok], c[ok]
    if x.size < 8:
        return float("nan"), float("nan"), float("nan"), int(x.size)
    stat0 = f(x, y)
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
    return float(stat0), lo, hi, int(x.size)


# -------------------------------------------------- ridge / PCA, numpy only
def ridge_fit(X, y, alpha):
    """Closed-form ridge on standardised X with an unpenalised intercept."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, d = X.shape
    xm, ym = X.mean(0), y.mean()
    Xc, yc = X - xm, y - ym
    A = Xc.T @ Xc + alpha * np.eye(d)
    w = np.linalg.solve(A, Xc.T @ yc)
    # the intercept is the training mean of y; centring makes it unpenalised
    return w, float(ym), xm


def ridge_predict(X, w, y_mean, x_mean):
    return (np.asarray(X, float) - x_mean) @ w + y_mean


def pca_fit(X, n_comp):
    """PCA by SVD on training data only. Returns (mean, components)."""
    X = np.asarray(X, float)
    mu = X.mean(0)
    Xc = X - mu
    n_comp = int(min(n_comp, Xc.shape[0] - 1, Xc.shape[1]))
    n_comp = max(n_comp, 1)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return mu, vt[:n_comp]


def pca_transform(X, mu, comps):
    return (np.asarray(X, float) - mu) @ comps.T
