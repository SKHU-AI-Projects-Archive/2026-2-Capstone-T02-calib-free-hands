"""CAM-EXP-009.3 — real reference-geometry bilateral separability.

This run estimates NO focal. It loads no reference focal, runs no AnyCalib or
GeoCalib, and builds no candidate focal grid. It asks one question about the
GigaHands reference geometry alone:

    is a given sequence's LEFT hand structure closer to its own RIGHT hand
    structure than to a RIGHT hand from a different sequence?

What is measured is REFERENCE_GEOMETRY_BILATERAL_SEPARABILITY. The reference 3D
mixes annotation error, triangulation error, calibration error, association
error and actual anatomy, so nothing here is called true anatomy, ground-truth
asymmetry or physical bone truth.

Reused unchanged from earlier runs:
  * CAM-EXP-001.3 `loco.reconstruct` — OTHER_CAMERA_ONLY_REFERENCE_3D, the
    target camera excluded BEFORE reconstruction (threshold 8.0 px, minimum 4
    inlier cameras), exactly as CAM-EXP-006 used it.
  * `gigahands_demo_qc_v1.csv.gz` — the frozen CAM-EXP-001.3 QC. It carries no
    focal column, and eligibility comes only from it.
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

RAW = RUN_DIR / "results" / "raw"
SUM = RUN_DIR / "results" / "summary"
TAB = RUN_DIR / "tables"
FIG = RUN_DIR / "figures"
CACHE = RUN_DIR / "cache"

C013 = RUNS / "CAM-EXP-001_3_gigahands_multiview_triangulation"
QC = MANIFESTS / "gigahands_demo_qc_v1.csv.gz"

# Manifests that carry a reference-focal column. Never opened by this run;
# `self_audit.py` greps the source for these names.
FOCAL_BEARING_MANIFESTS = (
    "gigahands_demo_cam_exp_0041_64frames_v1.csv.gz",
    "gigahands_demo_camera_benchmark_v1.csv.gz",
    "gigahands_demo_cam_exp_003_frames_v1.csv.gz",
)

BASE_SEED = 20260928
EPS = 1e-8

# ---------------------------------------------------------------- topology
# The 20 CONNECTED bones of the verified 21-joint hand. Arbitrary joint pairs
# such as 4-8, 4-12 or 8-20 span no bone and are never used.
BONES = [(0, 1), (1, 2), (2, 3), (3, 4),
         (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16),
         (0, 17), (17, 18), (18, 19), (19, 20)]
N_BONES = len(BONES)
FINGERS = [list(range(4 * f, 4 * f + 4)) for f in range(5)]

# ---------------------------------------------------------------- eligibility
QC_PASS_STATUSES = ("PASS_STRICT", "PASS_SINGLE_HAND")
LOCO_THRESHOLD_PX = 8.0
LOCO_MIN_INLIER_CAMERAS = 4
MIN_OK_JOINTS = 21           # every bone endpoint must reconstruct

# Deterministic, target-independent frame cap. Chosen from measured runtime
# (0.122 s per reconstruction) BEFORE any distance was computed, not from any
# result. Evenly spaced over each unit's eligible frames.
MAX_FRAMES_PER_UNIT = 48
MIN_FRAMES_PER_SIDE = 8
MIN_FRAMES_PER_HALF = 8

# ---------------------------------------------------------------- thresholds
# Frozen before any distance result (see freeze_spec.py).
GATE = {
    "same_over_cross_ratio": 0.80,       # D_same <= 0.80 * D_cross
    "nearest_margin_fraction": 0.75,     # >=75 % of units with M_nearest > 0
    "s_sep_min": 1.0,
    "control_degradation_pct": 20.0,
    "control_top1_drop_pp": 20.0,
    "strong_ratio": 0.60,
    "strong_nearest_fraction": 0.90,
    "min_participants_for_power": 8,
    "alpha": 0.05,
}
N_PERMUTATIONS = 10000
N_BOOTSTRAP = 10000


def q_unused():  # pragma: no cover - placeholder kept out of the API
    raise NotImplementedError


def stable_seed(name: str, salt: int = 0) -> int:
    """SHA-256 based. Python's hash() is salted per process and is never used."""
    h = hashlib.sha256(f"{name}|{salt}".encode()).digest()
    return (BASE_SEED + int.from_bytes(h[:8], "big")) % (2 ** 63 - 1)


def frame_half(sequence: str, camera: str, hand: str, frame: int) -> int:
    """Deterministic A/B split for the repeatability diagnostic.

    A stable hash of the frame identity, never the frame's fit quality.
    """
    h = hashlib.sha256(f"{sequence}|{camera}|{hand}|{frame}".encode()).digest()
    return h[0] & 1


def loco_on_path() -> None:
    """Make CAM-EXP-001.3's frozen reconstruction code importable, unchanged."""
    for p in (str(REPO), str(C013 / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------- geometry
def bone_lengths(xyz: np.ndarray) -> np.ndarray:
    """20 connected bone lengths from a (21, 3) reconstruction."""
    x = np.asarray(xyz, float)
    return np.array([np.linalg.norm(x[b] - x[a]) for a, b in BONES])


def proportions(lengths: np.ndarray) -> np.ndarray:
    """Remove absolute hand size: p sums to 1."""
    l = np.asarray(lengths, float)
    s = l.sum()
    if not np.isfinite(s) or s <= 0:
        return np.full(N_BONES, np.nan)
    return l / s


def log_rep(p: np.ndarray) -> np.ndarray:
    """PRIMARY representation z_b = log(p_b + 1e-8). EPS frozen for continuity
    with CAM-EXP-009.2."""
    return np.log(np.asarray(p, float) + EPS)


# ---------------------------------------------------------------- distances
def d_primary(zA: np.ndarray, zB: np.ndarray, mapping=None) -> float:
    """PRIMARY distance: median absolute log-proportion difference."""
    a = np.asarray(zA, float)
    b = np.asarray(zB, float)
    if mapping is not None:
        b = b[np.asarray(mapping, int)]
    if a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("nan")
    return float(np.median(np.abs(a - b)))


def d_mean_log(zA, zB, mapping=None) -> float:
    a, b = np.asarray(zA, float), np.asarray(zB, float)
    if mapping is not None:
        b = b[np.asarray(mapping, int)]
    return float(np.mean(np.abs(a - b)))


def d_raw_l1(zA, zB, mapping=None) -> float:
    a, b = np.exp(np.asarray(zA, float)), np.exp(np.asarray(zB, float))
    if mapping is not None:
        b = b[np.asarray(mapping, int)]
    return float(np.mean(np.abs(a - b)))


def d_cosine(zA, zB, mapping=None) -> float:
    a, b = np.exp(np.asarray(zA, float)), np.exp(np.asarray(zB, float))
    if mapping is not None:
        b = b[np.asarray(mapping, int)]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return float("nan")
    return float(1.0 - np.dot(a, b) / (na * nb))


def d_finger_internal(zA, zB, mapping=None) -> float:
    def internal(p):
        out = np.zeros_like(p)
        for f in FINGERS:
            s = p[f].sum()
            out[f] = p[f] / s if s > 0 else 0.0
        return out
    a, b = np.exp(np.asarray(zA, float)), np.exp(np.asarray(zB, float))
    if mapping is not None:
        b = b[np.asarray(mapping, int)]
    ra, rb = internal(a), internal(b)
    return float(np.median(np.abs(np.log(ra + EPS) - np.log(rb + EPS))))


SECONDARY = {
    "mean_abs_log": d_mean_log,
    "raw_proportion_l1": d_raw_l1,
    "cosine": d_cosine,
    "finger_internal": d_finger_internal,
}


def finger_permutation(seed_name="CAM0093_BONE_PERM") -> np.ndarray:
    """Deterministic within-finger bone permutation for the negative control.

    Within-finger rather than global: a global shuffle would also destroy the
    coarse size ordering of the bones, so its degradation could be explained
    without any anatomical correspondence at all.
    """
    rng = np.random.default_rng(stable_seed(seed_name))
    idx = np.arange(N_BONES)
    for f in range(5):
        blk = idx[4 * f:4 * f + 4].copy()
        while True:
            q = rng.permutation(blk)
            if not np.array_equal(q, blk):
                break
        idx[4 * f:4 * f + 4] = q
    return idx


# ---------------------------------------------------------------- io
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


def cluster_bootstrap(values, clusters, stat=np.median, n_boot=N_BOOTSTRAP,
                      seed=BASE_SEED):
    """Bootstrap resampling whole clusters, so camera views inside one
    sequence are never treated as independent samples."""
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
        draws[b] = stat(v[np.concatenate([members[p] for p in pick])])
    return (float(stat(v)), float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)))
