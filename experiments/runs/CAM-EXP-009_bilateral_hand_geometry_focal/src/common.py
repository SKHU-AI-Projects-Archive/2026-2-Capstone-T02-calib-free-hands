"""Shared code for CAM-EXP-009: bilateral sequence hand-geometry focal signal.

The idea under test: for the same person, anatomically corresponding bones of
the LEFT and RIGHT hand have related proportions. If the candidate focal is
right, the bone proportions needed to explain the LEFT images and those needed
to explain the RIGHT images should agree better.

THE CRITICAL DESIGN POINT. Comparing the FIXED reference 3D bone lengths of the
two hands is useless: that number does not depend on the candidate focal at
all, so its derivative with respect to f is zero and it carries no focal
information. This module therefore implements a CANDIDATE-CONDITIONED fit: at
every candidate focal the sequence-shared bone proportions are re-estimated per
side from that side's 2D observations, and only then compared.

`audit_focal_dependence.py` proves both halves of that claim numerically before
anything else runs.
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

C8 = RUNS / "CAM-EXP-008_offline_sequence_scene_hand_focal_fusion"
R0013 = RUNS / "CAM-EXP-001_3_gigahands_multiview_triangulation"
R005 = RUNS / "CAM-EXP-005_static_view_bias_attribution"

RAW = RUN_DIR / "results" / "raw"
SUM = RUN_DIR / "results" / "summary"
TAB = RUN_DIR / "tables"
FIG = RUN_DIR / "figures"
CACHE = RUN_DIR / "cache"

SEED = 20260930
N_BOOT = 10000

# 21-joint hand kinematic tree, the dataset convention verified by the
# CAM-EXP-006 manual-QC overlays. Only true parent-child bones; no arbitrary
# joint pairs such as 4-8 or 8-20.
BONES = [(0, 1), (1, 2), (2, 3), (3, 4),
         (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16),
         (0, 17), (17, 18), (18, 19), (19, 20)]
PARENT = {c: p for p, c in BONES}
ORDER = [c for _, c in BONES]
N_BONES = len(BONES)

# anatomically corresponding bones: bone b of the left hand <-> bone b of the
# right hand. The tree is identical for both sides, so correspondence is the
# identity on bone index. This is asserted, not assumed, in freeze_spec.py.
BONE_CORRESPONDENCE = [(i, i) for i in range(N_BONES)]

Q_MIN, Q_MAX, Q_N = 0.5, 1.5, 161
LAMBDAS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
MIN_BIMANUAL_FRAMES = 8
MIN_PAIRED_JOINTS = 12
FIT_EVAL_SPLIT = 0.5          # deterministic alternating split


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


def fnum(x, default=float("nan")) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------- hand assembly
def bone_dirs(xyz):
    """Unit direction of every bone, from a per-frame 3D hand."""
    d = np.zeros((N_BONES, 3))
    for i, (p, c) in enumerate(BONES):
        v = xyz[c] - xyz[p]
        n = np.linalg.norm(v)
        if n < 1e-9 or not np.isfinite(n):
            return None
        d[i] = v / n
    return d


def assemble(dirs, lengths):
    """Build a hand from bone directions and a bone-length vector."""
    X = np.zeros((21, 3))
    idx = {bc: i for i, bc in enumerate(BONES)}
    for j in ORDER:
        p = PARENT[j]
        X[j] = X[p] + dirs[idx[(p, j)]] * lengths[idx[(p, j)]]
    return X


def joint_bone_matrix():
    """M[j] = indicator of which bones lie on the path root -> j.

    Then X[j] = sum_b M[j,b] * l_b * dir_b, i.e. the hand is LINEAR in the
    bone-length vector once the directions are fixed. That linearity is what
    makes the candidate-conditioned fit tractable.
    """
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


# --------------------------------------------------------------- statistics
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
