"""CAM-EXP-010 — monocular depth/scale error decomposition and identifiability.

CAM-EXP-009.4 found that even with a perfect (reference) focal, wrist/root error
stays around 78 mm. This run decomposes that residual. It builds no deployable
depth estimator.

Every correction here is fitted using the REFERENCE 3D and is therefore an
**ORACLE DIAGNOSTIC**, never an inference method. The reference focal is an
oracle diagnostic input too: the point is to remove focal error so the remaining
translation error can be studied on its own.

Key algebra used throughout. For a frame with WiLoR root-relative joints `kp3`
and a camera translation `T`, the predicted absolute joints are `kp3_j + T`.
Writing the reference joints in camera coordinates as `Xc_j` and the reference
root as `T_ref = Xc_0`:

    (kp3_j + T) - Xc_j  =  R_j + e ,
    R_j = kp3_j - (Xc_j - T_ref)        root-relative pose residual, FIXED
    e   = T - T_ref                     translation error

So absolute MPJPE under ANY corrected translation is
`mean_j ||R_j + e||`, computable from cached `R_j` without re-running the
reconstruction. That is why `R` is cached per frame.
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
C0094 = RUNS / "CAM-EXP-009_4_scene_shared_anatomy_focal_refinement"
C0094_COMMIT = "1c36d92"
WILOR = C0094 / "cache" / "wilor"
C013 = RUNS / "CAM-EXP-001_3_gigahands_multiview_triangulation"
SCENE_SRC = (RUNS / "CAM-EXP-004_1_pre_cam005_robustness_audit" /
             "results" / "raw" / "extended_frame_predictions.csv.gz")

SOURCE_ARTIFACTS = {
    "cam0094_hand_split": MANIFESTS / "cam_exp_0094_hand_split_v1.csv.gz",
    "cam0094_units": MANIFESTS / "cam_exp_0094_units_v1.csv.gz",
    "cam0094_folds": MANIFESTS / "cam_exp_0094_outer_camera_folds_v1.csv",
    "cam0094_common_set": C0094 / "tables" / "common_set.csv",
    "cam0094_abs3d": C0094 / "results" / "raw" /
                     "absolute3d_frame_results.csv.gz",
    "cam0094_abs3d_summary": C0094 / "results" / "summary" /
                             "absolute3d_summary.csv",
    "loco": C013 / "src" / "loco.py",
}

# ------------------------------------------------------------------ constants
EPS = 1e-12
BASE_SEED = 20260929
LOCO_THRESHOLD_PX = 8.0
LOCO_MIN_INLIER = 4
MIN_OK_JOINTS = 12
PIPELINE_FOCAL_RATIO = 1000.0 / 256.0     # FOCAL_LENGTH / IMAGE_SIZE
N_BOOTSTRAP = 10000
MIN_FIT_FRAMES = 6
MIN_EVAL_FRAMES = 4

ORACLES = ["O0_ORACLE_FOCAL_ONLY",
           "O1_SEQUENCE_MULTIPLICATIVE_Z_SCALE",
           "O2_SEQUENCE_AFFINE_Z",
           "O3_SEQUENCE_CONSTANT_3D_BIAS",
           "O4_SEQUENCE_SCALE_PLUS_BIAS",
           "O5_PER_FRAME_DEPTH_ORACLE",
           "O6_PER_FRAME_ROOT_TRANSLATION_ORACLE"]

# Diagnostic branching thresholds, frozen before any result.
GATE = {
    "seq_scale_depth_reduction_pct": 25.0,
    "seq_scale_root_reduction_pct": 15.0,
    "affine_adds_value_pct": 10.0,
    "root_oracle_mpjpe_tolerance_mm": 2.0,
    "scale_invariance_tol_px": 1e-6,
}


def pipeline_focal(W, H):
    return PIPELINE_FOCAL_RATIO * max(W, H)


def stable_seed(name: str) -> int:
    h = hashlib.sha256(name.encode()).digest()
    return (BASE_SEED + int.from_bytes(h[:8], "big")) % (2 ** 63 - 1)


def split_of(sequence, camera, hand, frame) -> str:
    """Deterministic FIT/EVAL split, independent of any error or outcome.

    Oracle corrections are fitted on FIT frames and frozen before being applied
    to EVAL frames, so an EVAL frame's own reference never fits its own
    correction.
    """
    h = hashlib.sha256(
        f"CAM010|{sequence}|{camera}|{hand}|{frame}".encode()).digest()
    return "FIT" if (h[0] & 1) == 0 else "EVAL"


def scalar_focal(fx, fy) -> float:
    return float(np.sqrt(float(fx) * float(fy)))


# ------------------------------------------------------------------ io
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


def med(vals) -> float:
    v = np.asarray(vals, float).ravel()
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def pct(vals, q) -> float:
    v = np.asarray(vals, float).ravel()
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if v.size else float("nan")


def mad(vals) -> float:
    v = np.asarray(vals, float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    m = np.median(v)
    return float(np.median(np.abs(v - m)) * 1.4826)


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


def spearman(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 5:
        return float("nan")

    def rank(a):
        o = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), float)
        r[o] = np.arange(len(a))
        # average ties
        _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, r)
        return (sums / cnt)[inv]
    rx, ry = rank(x), rank(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])
