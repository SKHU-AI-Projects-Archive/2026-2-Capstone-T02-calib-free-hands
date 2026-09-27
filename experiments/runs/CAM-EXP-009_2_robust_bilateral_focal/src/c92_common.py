"""CAM-EXP-009.2 — asymmetry- and noise-robust bilateral hand-geometry focal.

CAM-EXP-009.1 showed the bilateral cue works on clean, perfectly symmetric
synthetic data but decays fast with bilateral asymmetry and 2D noise. This run
asks whether a ROBUST comparison of corresponding bones keeps the signal when
the two hands are only *related*, not identical.

The CAM-EXP-009.1 corrected solver is imported UNCHANGED:
  * UNDISTORT_ONCE_INTERNAL - undistort once per candidate K, then solve with
    distCoeffs=None
  * N_ITER = 32, the validated configuration
  * D20 bone fitter only. CAM-EXP-009.1's D10/D5 groupings forced equal bone
    lengths inside a group, which no real hand satisfies, so they are NOT
    reused here.

Nothing in this module re-implements the solver.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

RUN_DIR = Path(__file__).resolve().parents[1]
RUNS = RUN_DIR.parent
EXP = RUNS.parent
REPO = EXP.parent
MANIFESTS = EXP / "manifests"

C91 = RUNS / "CAM-EXP-009_1_bilateral_solver_validation"
C8 = RUNS / "CAM-EXP-008_offline_sequence_scene_hand_focal_fusion"
R005 = RUNS / "CAM-EXP-005_static_view_bias_attribution"

# The validated solver is imported, not copied. CAM-EXP-009.1's own module is
# also called `common`, so this run's module is named `c92_common` to avoid
# shadowing it in sys.modules; that shadowing would silently hand the solver
# the wrong constants.
sys.path.insert(0, str(C91 / "src"))

RAW = RUN_DIR / "results" / "raw"
SUM = RUN_DIR / "results" / "summary"
TAB = RUN_DIR / "tables"
FIG = RUN_DIR / "figures"
CACHE = RUN_DIR / "cache"

BASE_SEED = 20261002
EPS = 1e-8
Q_MIN, Q_MAX, Q_N = 0.5, 1.5, 161
N_FRAMES = 12
MIN_FIT_FRAMES = 3

# subject-level stress definitions
DEV_SUBJECTS = 8
TEST_SUBJECTS_PRIMARY = 16
TEST_SUBJECTS_SECONDARY = 8

LAMBDAS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)

# frozen success thresholds
GATE = {
    "clean_focal_err_pct": 1.0,
    "moderate_focal_err_pct": 5.0,
    "relative_reduction_vs_M0_pct": 30.0,
    "weak_relative_reduction_pct": 20.0,
    "boundary_rate": 0.10,
    "fit_eval_diff_pct": 5.0,
    "control_degradation_pp": 10.0,
    "control_within5_drop_pp": 20.0,
    "real_relative_reduction_pct": 10.0,
    "real_within5_gain_pp": 5.0,
}

COMBINED_MODERATE = dict(dist_over_diam=4, asym_kind="dense", asym=0.02,
                         noise_px=1.0, artic_deg=1.0, missing=0.10,
                         visibility=0.75)
COMBINED_STRONG = dict(dist_over_diam=8, asym_kind="dense", asym=0.05,
                       noise_px=2.0, artic_deg=2.0, missing=0.20,
                       visibility=0.50)


def q_grid() -> np.ndarray:
    return np.geomspace(Q_MIN, Q_MAX, Q_N)


def stable_seed(name: str, salt: int = 0) -> int:
    """SHA-256 based. Python's hash() is salted per process and is never used."""
    h = hashlib.sha256(f"{name}|{salt}".encode()).digest()
    return (BASE_SEED + int.from_bytes(h[:8], "big")) % (2 ** 63 - 1)


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


def cluster_bootstrap(values, clusters, stat=np.median, n_boot=10000,
                      seed=BASE_SEED):
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
    return r


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])
