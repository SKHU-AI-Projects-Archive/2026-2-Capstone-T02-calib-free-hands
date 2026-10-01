"""CAM-EXP-009.1 — bilateral bone-fitting solver validation.

Purpose: CAM-EXP-009 concluded the bilateral objective was not identifiable.
Before that can be read as a GEOMETRIC fact, two implementation faults found in
its solver must be excluded:

  DOUBLE_DISTORTION_CONVENTION_BUG
      fit_bones undistorted the observations once, then handed the undistorted
      coordinates to solvePnP together with the ORIGINAL non-zero distortion
      coefficients - undistorting a second time.

  RAW_VS_PINHOLE_REPROJECTION_MISMATCH
      run_synthetic.eval_reproj ran a distortion-aware PnP but then projected
      with a plain pinhole model and compared against RAW DISTORTED pixels.

This run fixes both under one explicit convention and re-tests.

CONVENTION: UNDISTORT_ONCE_INTERNAL
    per candidate focal, undistort the raw pixels ONCE using that candidate's
    own K and the known coefficients, then work in ideal pinhole space with
    distCoeffs = None everywhere inside the solver. Scoring is reported in both
    spaces: ideal (pinhole vs undistorted) and raw (projectPoints with
    distortion vs raw pixels).

Seeds are derived from SHA-256 of a condition name, never from Python's hash(),
which is salted per process.
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
C9 = RUNS / "CAM-EXP-009_bilateral_hand_geometry_focal"

RAW = RUN_DIR / "results" / "raw"
SUM = RUN_DIR / "results" / "summary"
TAB = RUN_DIR / "tables"
FIG = RUN_DIR / "figures"

BASE_SEED = 20261001

BONES = [(0, 1), (1, 2), (2, 3), (3, 4),
         (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16),
         (0, 17), (17, 18), (18, 19), (19, 20)]
PARENT = {c: p for p, c in BONES}
ORDER = [c for _, c in BONES]
N_BONES = len(BONES)

W, H = 1280, 720
F_TRUE = 900.0
DIST_REAL = np.array([-0.39218, 0.13298, 0.0, 0.0])
Q_MIN, Q_MAX, Q_N = 0.5, 1.5, 161
# Raised from 8 to 32 DURING SOLVER VALIDATION, judged only against the
# positive control (reprojection at the TRUE focal with a uniform start) and
# before any focal sweep was run. CAM-EXP-009 used 6 iterations, which left a
# ~0.4 px residual at the true focal - a third implementation fault,
# INSUFFICIENT_ALTERNATION_ITERATIONS. Convergence measured: 8 -> 0.425 px,
# 16 -> 0.021 px, 32 -> 0.00067 px, 64 -> 0.00061 px, 128 -> 0.00054 px.
N_ITER = 32
N_MULTISTART = 8

# frozen PASS thresholds, fixed before the corrected execution
GATE = {
    "raw_roundtrip_median_px": 1e-6, "raw_roundtrip_max_px": 1e-4,
    "ideal_roundtrip_median_px": 1e-5, "ideal_roundtrip_max_px": 1e-3,
    "true_geometry_median_px": 0.05,
    "focal_err_pct": 5.0, "boundary_rate": 0.10,
    "control_degradation_pp": 10.0, "control_within5_drop_pp": 20.0,
}


def q_grid() -> np.ndarray:
    return np.geomspace(Q_MIN, Q_MAX, Q_N)


def stable_seed(name: str, salt: int = 0) -> int:
    """Process-independent seed. Python's hash() is salted; this is not."""
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


def arr_sha(a) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(a, np.float64)).tobytes()).hexdigest()


def fnum(x, default=float("nan")) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------ hand assembly
def joint_bone_matrix():
    idx = {bc: i for i, bc in enumerate(BONES)}
    M = np.zeros((21, N_BONES))
    for j in range(21):
        k = j
        while k != 0:
            p = PARENT[k]
            M[j, idx[(p, k)]] = 1.0
            k = p
    return M


MJB = joint_bone_matrix()
BIDX = {bc: i for i, bc in enumerate(BONES)}


def assemble(dirs, lengths):
    X = np.zeros((21, 3))
    for j in ORDER:
        p = PARENT[j]
        i = BIDX[(p, j)]
        X[j] = X[p] + dirs[i] * lengths[i]
    return X


def make_K(f, cx=None, cy=None, fy=None):
    cx = W / 2.0 if cx is None else cx
    cy = H / 2.0 if cy is None else cy
    fy = f if fy is None else fy
    return np.array([[f, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


# --------------------------------------------- shape degrees of freedom
def dof_groups(dof: str):
    """Target-independent groupings, fixed before any corrected result.

    D20 : every bone free
    D10 : adjacent bones paired along each finger chain -> 10 parameters
    D5  : one parameter per finger chain -> 5 parameters
    D0  : no free shape (handled separately by the true-shape oracle)
    """
    if dof == "D20":
        return [[i] for i in range(N_BONES)]
    if dof == "D10":
        return [[2 * i, 2 * i + 1] for i in range(N_BONES // 2)]
    if dof == "D5":
        return [[4 * f + k for k in range(4)] for f in range(5)]
    raise ValueError(dof)


def spearman_safe(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return float("nan")

    def rank(a):
        o = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), float)
        r[o] = np.arange(1, len(a) + 1)
        return r

    rx, ry = rank(x), rank(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])
