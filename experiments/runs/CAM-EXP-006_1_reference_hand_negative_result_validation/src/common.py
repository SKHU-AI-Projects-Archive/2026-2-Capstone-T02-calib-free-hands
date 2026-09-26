"""Shared paths and helpers for CAM-EXP-006.1.

This run VALIDATES CAM-EXP-006's negative result. It builds no new calibration
method.

Vocabulary rules carried over unchanged:
  * the released 2D are "dataset-provided 2D observations", never "GT 2D";
  * the leave-one-camera-out 3D is "OTHER_CAMERA_ONLY_REFERENCE_3D", never
    "ground truth 3D" and never "independent physical ground truth";
  * conditions R2-R5 are ORACLE_DIAGNOSTIC_ONLY - they consume provided camera
    metadata and are not deployable.
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
R005 = RUNS / "CAM-EXP-005_static_view_bias_attribution"
R006 = RUNS / "CAM-EXP-006_reference_hand_focal_information"

# CAM-EXP-006 artefacts, read-only. Never written by this run.
R006_CACHE = R006 / "cache" / "reference_3d_v1.npz"
R006_FRAMES = MANIFESTS / "cam_exp_006_reference_hand_frames_v1.csv.gz"
R006_SUMMARY = R006 / "tables" / "solver_summary.csv"
R006_VERDICT = R006 / "results" / "summary" / "cam006_verdict.json"
VIEW_TARGETS = R005 / "results" / "raw" / "view_targets.csv.gz"

RAW = RUN_DIR / "results" / "raw"
SUM = RUN_DIR / "results" / "summary"
TAB = RUN_DIR / "tables"
FIG = RUN_DIR / "figures"
CACHE = RUN_DIR / "cache"

SEED = 20260927
N_BOOT = 10000
COUNTS = (1, 2, 4, 8, 16)


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
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fnum(x, default=float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def cam006_src_on_path() -> None:
    """Make CAM-EXP-006's frozen solver importable, for exact R0 reproduction."""
    import sys
    for p in (str(R006 / "src"), str(REPO)):
        if p not in sys.path:
            sys.path.insert(0, p)


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


def corr_cluster_ci(x, y, clusters, method="spearman", n_boot=N_BOOT,
                    seed=SEED):
    """Correlation with a physical-camera cluster bootstrap interval."""
    from scipy import stats
    f = stats.spearmanr if method == "spearman" else stats.pearsonr
    x, y = np.asarray(x, float), np.asarray(y, float)
    c = np.asarray(clusters)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, c = x[ok], y[ok], c[ok]
    if x.size < 8 or np.std(x) == 0 or np.std(y) == 0:
        return (float("nan"),) * 4 + (int(x.size),)
    r = f(x, y)
    stat0 = r.statistic if hasattr(r, "statistic") else r[0]
    pval = r.pvalue if hasattr(r, "pvalue") else r[1]
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
        rr = f(xs, ys)
        draws[b] = rr.statistic if hasattr(rr, "statistic") else rr[0]
    d = draws[np.isfinite(draws)]
    lo, hi = ((float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
              if d.size else (float("nan"), float("nan")))
    return float(stat0), float(pval), lo, hi, int(x.size)
