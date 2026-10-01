"""CAM-EXP-012 — balanced scene-hand fusion with full-video cross-fitted hand geometry.

One question: with the scene and hand losses put on a comparable numerical
scale WITHOUT using any reference focal, and with a strictly positive hand
weight selected on TRAIN cameras only, does adding whole-video hand-joint
geometry reduce focal error on UNSEEN physical cameras?

    S0  SCENE_ONLY
    S1  SCENE_PLUS_CORRECT_HAND      <- the only difference is the hand term

CAM-EXP-011 is not treated as a failed experiment. It measured what the
inherited fusion scaling does in the all-frame regime, and found the scene and
hand local magnitudes differed by roughly 13x, so the frozen lambda grid could
not let the hand term compete. CAM-EXP-012 addresses that by normalising the
two losses label-free, not by enlarging lambda.

Three design changes from CAM-EXP-011, all frozen before any test focal:

  1. TEMPORAL-BLOCK 2-FOLD CROSS-FITTING (block = 8 frames) instead of a
     per-frame hash split, so near-duplicate adjacent poses cannot straddle the
     fit/evaluation boundary.
  2. A GLOBAL ABSOLUTE FOCAL GRID shared by every video, so shuffled-donor
     curves always cover the recipient's candidate window and every condition
     lives on one focal axis.
  3. LOSS-SCALE NORMALISATION alpha measured at a FIXED +/- 5 % log-focal
     offset (not at an adjacent grid point, which would make alpha depend on
     grid resolution).
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

RAW = RUN_DIR / "results" / "raw"
SUM = RUN_DIR / "results" / "summary"
TAB = RUN_DIR / "tables"
FIG = RUN_DIR / "figures"
CACHE = RUN_DIR / "cache"

# ------------------------------------------------------------------ sources
C011 = RUNS / "CAM-EXP-011_allframe_three_model_hand_paired_benchmark"
C011_SRC = C011 / "src"
C011_RAW = C011 / "results" / "raw"
C011_CACHE = C011 / "cache"
WILOR = C011_CACHE / "wilor"
C0094_SRC = RUNS / "CAM-EXP-009_4_scene_shared_anatomy_focal_refinement" / "src"
SCENE_SRC_64 = (RUNS / "CAM-EXP-004_1_pre_cam005_robustness_audit"
                / "results" / "raw" / "extended_frame_predictions.csv.gz")

SCENE_PRED = {
    "ANYCALIB": C011_RAW / "anycalib_allframe_predictions.csv.gz",
    "GEOCALIB": C011_RAW / "geocalib_allframe_predictions.csv.gz",
    "PERSPECTIVE_FIELDS": C011_RAW
    / "perspective_fields_allframe_predictions.csv.gz",
}
MODELS = {
    "ANYCALIB": {"display": "AnyCalib"},
    "GEOCALIB": {"display": "GeoCalib"},
    "PERSPECTIVE_FIELDS": {"display": "Perspective Fields"},
}
CONDITIONS = ["SCENE_ONLY", "SCENE_PLUS_CORRECT_HAND",
              "SCENE_PLUS_SHUFFLED_HAND", "SCENE_PLUS_WRONG_BONE"]

# ------------------------------------------------------- frozen parameters
BASE_SEED = 20260930
EPS = 1e-12

# temporal-block cross-fit
BLOCK_SIZE = 8                      # frames per temporal block
MIN_OBS_PER_FOLD_PER_SIDE = 4       # so each side needs >= 8 in total

# loss-scale normalisation: a FIXED log-focal offset, deliberately not an
# adjacent grid point, so alpha does not depend on grid resolution
DELTA_LOG_F = 0.05
Q_MINUS = float(np.exp(-DELTA_LOG_F))      # ~0.951229
Q_PLUS = float(np.exp(+DELTA_LOG_F))       # ~1.051271
ALPHA_DEGENERATE_THRESHOLD = 1e-8

# candidate ratio grid for fusion
Q_N_HALF = 101                      # per half, 1.0 shared -> 201 total
Q_MIN, Q_MAX = 0.5, 1.5

# global absolute focal grid; N fixed by the interpolation validation below
GLOBAL_GRID_N = 401

# positive-only hand weights. lambda = 0 is NOT a candidate; it exists only as
# an implementation sanity test.
LAMBDA_GRID = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

N_OUTER_FOLDS = 5
N_INNER_FOLDS = 4
N_BOOTSTRAP = 10000
SIGMA_SCENE_FLOOR = 0.02            # unchanged from CAM-EXP-011
TIE_TOLERANCE_PP = 1e-9

BONES = [(0, 1), (1, 2), (2, 3), (3, 4),
         (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16),
         (0, 17), (17, 18), (18, 19), (19, 20)]
N_BONES = len(BONES)

PARTICIPANT_OF = {
    "p36-tea-0010": "p36", "p41-boxing-0021": "p41",
    "p41-plant-0004": "p41", "p44-dog-0004": "p44",
    "p52-instrument-0034": "p52",
}
DISPLAY_SEQUENCE = {
    "p36-tea-0010": "Tea", "p41-boxing-0021": "Boxing",
    "p41-plant-0004": "Plant", "p44-dog-0004": "Dog",
    "p52-instrument-0034": "Instrument",
}

FOCAL_FORBIDDEN_PREFIXES = ("gt_", "signed_focal_error",
                            "relative_focal_error", "fx_relative_error",
                            "fy_relative_error", "log_focal_error",
                            "hfov_error", "vfov_error",
                            "principal_point_error")


# ------------------------------------------------------------------ grids
def q_grid() -> np.ndarray:
    """201 candidate ratios containing q = 1.0 EXACTLY.

    Built as two log-spaced halves that meet at 1.0, so the identity point is
    exact rather than the nearest grid sample.
    """
    lo = np.geomspace(Q_MIN, 1.0, Q_N_HALF)
    hi = np.geomspace(1.0, Q_MAX, Q_N_HALF)
    q = np.concatenate([lo, hi[1:]])
    q[Q_N_HALF - 1] = 1.0
    return q


def global_focal_grid(f_min, f_max, n=GLOBAL_GRID_N) -> np.ndarray:
    return np.geomspace(f_min, f_max, n)


def temporal_fold(frame_idx: int) -> str:
    """Temporal-block 2-fold split on the 0-based video frame index."""
    return "A" if (int(frame_idx) // BLOCK_SIZE) % 2 == 0 else "B"


def stable_seed(name: str) -> int:
    h = hashlib.sha256(name.encode()).digest()
    return (BASE_SEED + int.from_bytes(h[:8], "big")) % (2 ** 63 - 1)


def hash_bucket(name: str, n: int) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big") % n


def scalar_focal(fx, fy) -> float:
    return float(np.sqrt(float(fx) * float(fy)))


def aggregate_video_focal(focals) -> float:
    f = np.asarray([x for x in focals if np.isfinite(x) and x > 0], float)
    if f.size == 0:
        return float("nan")
    return float(np.exp(np.median(np.log(f))))


def scene_loss(q, sigma):
    """CAM-EXP-011's scene loss, unchanged. Centred so L(1) = 0."""
    return (np.log(np.asarray(q, float)) / sigma) ** 2


def finger_permutation(seed_name="CAM0094_BONE_PERM") -> np.ndarray:
    """CAM-EXP-009.4's frozen within-finger bone permutation, reproduced."""
    h = hashlib.sha256(f"{seed_name}|0".encode()).digest()
    seed = (20260928 + int.from_bytes(h[:8], "big")) % (2 ** 63 - 1)
    rng = np.random.default_rng(seed)
    idx = np.arange(N_BONES)
    for f in range(5):
        blk = idx[4 * f:4 * f + 4].copy()
        while True:
            q = rng.permutation(blk)
            if not np.array_equal(q, blk):
                break
        idx[4 * f:4 * f + 4] = q
    return idx


# ------------------------------------------------------------------ io
def read_csv(path) -> list[dict]:
    path = Path(path)
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_focal_blind(path) -> list[dict]:
    return [{k: v for k, v in r.items()
             if not any(k.startswith(p) for p in FOCAL_FORBIDDEN_PREFIXES)}
            for r in read_csv(path)]


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


def med(vals) -> float:
    v = np.asarray(vals, float).ravel()
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def pct(vals, q) -> float:
    v = np.asarray(vals, float).ravel()
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if v.size else float("nan")


def paired_cluster_bootstrap(gains, clusters, n_boot=N_BOOTSTRAP,
                             seed=BASE_SEED):
    """Paired bootstrap resampling whole PHYSICAL CAMERAS."""
    g = np.asarray(gains, float)
    ok = np.isfinite(g)
    g, c = g[ok], np.asarray(clusters)[ok]
    if g.size == 0:
        return float("nan"), float("nan"), float("nan")
    uniq = np.unique(c)
    members = [np.flatnonzero(c == u) for u in uniq]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        draws[b] = np.median(g[np.concatenate([members[p] for p in pick])])
    return (float(np.median(g)), float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)))
