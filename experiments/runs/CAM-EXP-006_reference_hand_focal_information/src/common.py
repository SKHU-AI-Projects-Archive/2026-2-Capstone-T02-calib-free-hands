"""Shared paths and helpers for CAM-EXP-006.

Vocabulary rules that this run holds to everywhere:

* the released 2D are "dataset-provided 2D observations", never "GT 2D";
* the leave-one-camera-out 3D is "OTHER_CAMERA_ONLY_REFERENCE_3D", never
  "ground truth 3D" and never "independent physical ground truth";
* the whole experiment is an INTERNAL_REFERENCE_HAND_DIAGNOSTIC - an information
  ceiling, not a deployable calibration method.
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

QC_MANIFEST = MANIFESTS / "gigahands_demo_qc_v1.csv.gz"
CAMERA_MANIFEST = MANIFESTS / "gigahands_demo_camera_benchmark_v1.csv.gz"
FRAMES64 = MANIFESTS / "gigahands_demo_cam_exp_0041_64frames_v1.csv.gz"
PRED64 = R0041 / "results" / "raw" / "extended_frame_predictions.csv.gz"

RAW = RUN_DIR / "results" / "raw"
SUM = RUN_DIR / "results" / "summary"
TAB = RUN_DIR / "tables"
FIG = RUN_DIR / "figures"
QCDIR = RUN_DIR / "manual_qc"
CACHE = RUN_DIR / "cache"

SEED = 20260926
N_BOOT = 10000

# CAM-EXP-001.3 reconstruction settings, reused unchanged
LOCO_THRESHOLD_PX = 8.0
LOCO_MIN_INLIER_CAMERAS = 4


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
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(p)


def fnum(x, default=float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def loco_src_on_path() -> None:
    """Make CAM-EXP-001.3's frozen reconstruction code importable.

    That run's code is REUSED unchanged; none of its outputs are rewritten.
    """
    import sys
    p = str(R0013 / "src")
    if p not in sys.path:
        sys.path.insert(0, p)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def cluster_bootstrap(values, clusters, stat=np.median, n_boot=N_BOOT, seed=SEED):
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


def spearman_cluster_ci(x, y, clusters, n_boot=N_BOOT, seed=SEED):
    from scipy import stats
    x, y = np.asarray(x, float), np.asarray(y, float)
    c = np.asarray(clusters)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, c = x[ok], y[ok], c[ok]
    if x.size < 8 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), int(x.size)
    r = stats.spearmanr(x, y)
    uniq = np.unique(c)
    members = [np.flatnonzero(c == u) for u in uniq]
    rng = np.random.default_rng(seed)
    draws = np.full(n_boot, np.nan)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([members[p] for p in pick])
        xs, ys = x[idx], y[idx]
        if np.std(xs) == 0 or np.std(ys) == 0:
            continue
        draws[b] = stats.spearmanr(xs, ys).statistic
    d = draws[np.isfinite(draws)]
    lo, hi = ((float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
              if d.size else (float("nan"), float("nan")))
    return float(r.statistic), float(r.pvalue), lo, hi, int(x.size)
