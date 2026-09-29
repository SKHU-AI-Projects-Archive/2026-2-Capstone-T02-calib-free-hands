"""CAM-EXP-011 — all-frame, three-model, scene-vs-hand paired focal benchmark.

One question: on the same GigaHands videos, does adding whole-video hand
structure to an existing scene focal estimator reduce focal error?

Three scene estimators, each in two conditions:

    AnyCalib            / AnyCalib + Hand
    GeoCalib            / GeoCalib + Hand
    Perspective Fields  / Perspective Fields + Hand

Frame policy. CAM-EXP-003 used 8 uniformly spaced frames per view and
CAM-EXP-004.1 extended that to 64 and found scene aggregation largely
saturated. Neither was wrong. This run instead processes **every usable RGB
frame**, so the presentation can answer "does it hold on the whole video?"
directly rather than by extrapolation.

Counting discipline — these are different quantities and are never merged:

    SEQUENCE_TOTAL_FRAMES          frames that exist in the video
    SCENE_INPUT_FRAMES             frames fed to the scene models
    THREE_MODEL_COMMON_SCENE_FRAMES frames where all three models succeeded
    HAND_AVAILABLE_FRAMES          frames with usable hand geometry
    HAND_CORRECTION_FRAMES         frames entering the hand objective
    FINAL_PAIRED_VIDEO_UNITS       videos comparable OFF vs ON

Evaluation unit is the **sequence-camera video**: one video, one focal
estimate, one vote. Frames are repeated observations inside a video and are
never treated as independent samples.
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
CAMERA_BENCHMARK = MANIFESTS / "gigahands_demo_camera_benchmark_v1.csv.gz"
QC = MANIFESTS / "gigahands_demo_qc_v1.csv.gz"
C0094 = RUNS / "CAM-EXP-009_4_scene_shared_anatomy_focal_refinement"
C0094_WILOR = C0094 / "cache" / "wilor"
C003 = RUNS / "CAM-EXP-003_single_frame_calibration_benchmark"
C0041 = RUNS / "CAM-EXP-004_1_pre_cam005_robustness_audit"
SCENE_SRC_64 = C0041 / "results" / "raw" / "extended_frame_predictions.csv.gz"

# ---------------------------------------------------- scene model variants
# The CAM-EXP-003 historical primary variants, confirmed against that run's
# report (spreads 4.6 / 17.7 / 18.8 %). Not re-chosen for this run.
MODELS = {
    "ANYCALIB": {"adapter": "anycalib",
                 "historical_key": "AnyCalib[anycalib_pinhole/pinhole]",
                 "display": "AnyCalib"},
    "GEOCALIB": {"adapter": "geocalib",
                 "historical_key": "GeoCalib[pinhole]",
                 "display": "GeoCalib"},
    "PERSPECTIVE_FIELDS": {"adapter": "pf_centered",
                           "historical_key": "PerspectiveFields[centered]",
                           "display": "Perspective Fields"},
}
CONDITIONS = ["SCENE_ONLY", "SCENE_PLUS_HAND"]

# ------------------------------------------------------------------ rules
BASE_SEED = 20260929
EPS = 1e-12

# Frozen before any focal result.
MIN_SCENE_COVERAGE = 0.80          # of SCENE_INPUT_FRAMES, per video
MIN_HAND_FRAMES_PER_SIDE = 8       # inherited from CAM-EXP-009.4
LAMBDA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
N_OUTER_FOLDS = 5
N_BOOTSTRAP = 10000
Q_MIN, Q_MAX, Q_N = 0.5, 1.5, 81
SIGMA_SCENE_FLOOR = 0.02

# Reference-focal columns. Phase A/B/C must never see them.
FOCAL_FORBIDDEN_PREFIXES = ("gt_", "signed_focal_error", "relative_focal_error",
                            "fx_relative_error", "fy_relative_error",
                            "log_focal_error", "hfov_error", "vfov_error",
                            "principal_point_error")

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


def stable_seed(name: str) -> int:
    h = hashlib.sha256(name.encode()).digest()
    return (BASE_SEED + int.from_bytes(h[:8], "big")) % (2 ** 63 - 1)


def hash_bucket(name: str, n: int) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big") % n


def q_grid() -> np.ndarray:
    """Candidate ratios around a video's scene focal.

    The element nearest 1.0 is snapped to exactly 1.0 so that lambda = 0
    reproduces the scene-only estimate bit-for-bit. Without this the grid does
    not contain q = 1 and every `+ Hand` estimate would inherit a spurious
    quantisation offset (measured at -0.65 % before the fix), and the lambda=0
    sanity point would not actually be a sanity point.
    """
    q = np.geomspace(Q_MIN, Q_MAX, Q_N)
    q[int(np.argmin(np.abs(np.log(q))))] = 1.0
    return q


def scalar_focal(fx, fy) -> float:
    """Scalar focal convention, unchanged from CAM-EXP-003/005/008."""
    return float(np.sqrt(float(fx) * float(fy)))


def aggregate_video_focal(focals) -> float:
    """Log-domain median (geometric median) over a video's frames.

    One video -> one focal. Long sequences do not get more votes.
    """
    f = np.asarray([x for x in focals if np.isfinite(x) and x > 0], float)
    if f.size == 0:
        return float("nan")
    return float(np.exp(np.median(np.log(f))))


def usable_cameras():
    """The verified CAM-EXP-003 benchmark camera set. Never re-derived."""
    rows = read_csv(CAMERA_BENCHMARK)
    out = {}
    for r in rows:
        ok = r["usable_for_camera_benchmark"] in ("1", "True", "true")
        out[(r["sequence"], r["camera"])] = {
            "usable": ok,
            "rgb_video_available": r["rgb_video_available"],
            "video_segment_matches_annotation":
                r["video_segment_matches_annotation"],
            "note": r["note"],
        }
    return out


# ------------------------------------------------------------------ io
def read_csv(path) -> list[dict]:
    path = Path(path)
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_scene_predictions_focal_blind(path):
    """Loader that DROPS every reference-focal column at read time."""
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
