"""CAM-EXP-009.4 — sequence-level scene + shared-hand-anatomy focal refinement.

The first experiment in the CAM-009 line that asks whether hand geometry ADDS
anything to a scene-based focal estimate, rather than whether hand geometry
contains focal information at all.

Model:
    p_seq = softmax(log(p0 + eps) + a)              sequence-shared anatomy
    p_L   = softmax(log(p_seq + eps) + delta_L)     side deviations
    p_R   = softmax(log(p_seq + eps) + delta_R)

`p0` is the neutral MANO hand's bone proportions. `a`, `delta_L`, `delta_R` are
re-estimated from scratch for EVERY sequence-camera unit: no participant
template is carried over from another recording, per CAM-EXP-009.3.1's finding
that a person's geometry did not transfer between sessions.

LEAKAGE BARRIER. The reference focal is closed during PHASE A/B/C. The scene
prediction cache physically contains `gt_*` columns, so this module provides
`read_scene_predictions()`, which DROPS every `gt_*` and every error column at
load time. Phase A/B/C code must use only that reader.
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
QC = MANIFESTS / "gigahands_demo_qc_v1.csv.gz"
SCENE_SRC = (RUNS / "CAM-EXP-004_1_pre_cam005_robustness_audit" /
             "results" / "raw" / "extended_frame_predictions.csv.gz")
SCENE_MODEL_KEY = "anycalib_gen_radial"          # AnyCalib anycalib_gen/radial:2
C0093 = RUNS / "CAM-EXP-009_3_real_bilateral_geometry_separability"
C0091 = RUNS / "CAM-EXP-009_1_bilateral_solver_validation"
C013 = RUNS / "CAM-EXP-001_3_gigahands_multiview_triangulation"

# Columns that carry the reference focal or an error derived from it. Dropped
# at load time so a Phase A/B/C script cannot read them even by accident.
FOCAL_FORBIDDEN_PREFIXES = ("gt_", "signed_focal_error", "relative_focal_error",
                            "fx_relative_error", "fy_relative_error",
                            "log_focal_error", "hfov_error", "vfov_error",
                            "principal_point_error")

# ------------------------------------------------------------------ topology
BONES = [(0, 1), (1, 2), (2, 3), (3, 4),
         (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16),
         (0, 17), (17, 18), (18, 19), (19, 20)]
N_BONES = len(BONES)
PARENT = {c: p for p, c in BONES}
ORDER = [c for _, c in BONES]
BIDX = {bc: i for i, bc in enumerate(BONES)}
FINGERS = [list(range(4 * f, 4 * f + 4)) for f in range(5)]

# ------------------------------------------------------------------ identity
PARTICIPANT_OF = {
    "p36-tea-0010": "p36", "p41-boxing-0021": "p41",
    "p41-plant-0004": "p41", "p44-dog-0004": "p44",
    "p52-instrument-0034": "p52",
}

# ------------------------------------------------------------------ constants
EPS = 1e-8
BASE_SEED = 20260928

N_SCENE_FRAMES = 32                  # PRIMARY
N_HAND_FRAMES = 32                   # PRIMARY
FRAME_COUNTS = (8, 16, 32, 64)
MIN_FRAMES_PER_SIDE = 8

Q_MIN, Q_MAX, Q_N = 0.5, 1.5, 81     # candidate grid around the scene estimate
N_ITER = 32                          # CAM-EXP-009.1 validated alternation count
MIN_OK_JOINTS = 12

LAMBDA_GENERIC_GRID = (0.1, 1.0, 10.0)
LAMBDA_SIDE_GRID = (0.1, 1.0, 10.0)
FUSION_LAMBDA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
N_OUTER_FOLDS = 5
N_BOOTSTRAP = 10000

SIGMA_SCENE_FLOOR = 0.02             # log-focal units; robust-scale floor

METHODS = ["M0_SCENE_ONLY", "M1_SCENE_GENERIC_INDEPENDENT",
           "M2_SCENE_SHARED_NO_GENERIC", "M3_SCENE_SHARED_GENERIC_FULL"]

GATE = {
    "synthetic_clean_focal_err_pct": 1.0,
    "synthetic_moderate_focal_err_pct": 5.0,
    "synthetic_wrong_bone_degradation_pp": 10.0,
    "synthetic_boundary_rate": 0.25,
    "focal_relative_reduction_pct": 10.0,
    "abs3d_root_reduction_pct": 10.0,
    "abs3d_mpjpe_reduction_pct": 5.0,
    "root_aligned_tolerance_mm": 2.0,
}

# CAM-EXP-009.2's pre-specified moderate synthetic perturbation, reused
COMBINED_MODERATE = dict(dist_over_diam=4, asym_kind="dense", asym=0.02,
                         noise_px=1.0, artic_deg=1.0, missing=0.10,
                         visibility=0.75)


def stable_seed(name: str, salt: int = 0) -> int:
    h = hashlib.sha256(f"{name}|{salt}".encode()).digest()
    return (BASE_SEED + int.from_bytes(h[:8], "big")) % (2 ** 63 - 1)


def hash_bucket(name: str, n: int) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big") % n


def q_grid() -> np.ndarray:
    return np.geomspace(Q_MIN, Q_MAX, Q_N)


# ------------------------------------------------------------------ geometry
def assemble(dirs, lengths):
    """Root-relative skeleton from unit bone directions and bone lengths."""
    X = np.zeros((21, 3))
    for j in ORDER:
        p = PARENT[j]
        i = BIDX[(p, j)]
        X[j] = X[p] + dirs[i] * lengths[i]
    return X


def bone_unit_directions(kp3d):
    """Unit direction per connected bone. Lengths are DISCARDED on purpose.

    WiLoR's predicted bone lengths encode its own training shape prior; using
    them as the anatomy target would smuggle that prior into this model. Only
    articulation direction is taken from the network.
    """
    X = np.asarray(kp3d, float)
    d = np.zeros((N_BONES, 3))
    ok = np.zeros(N_BONES, bool)
    for i, (a, b) in enumerate(BONES):
        v = X[b] - X[a]
        n = np.linalg.norm(v)
        if np.isfinite(n) and n > 1e-9:
            d[i] = v / n
            ok[i] = True
    return d, ok


def softmax_from_log(log_p):
    m = np.max(log_p)
    e = np.exp(log_p - m)
    return e / e.sum()


def anatomy_from_params(p0, a, delta):
    """p_seq = softmax(log p0 + a); p_side = softmax(log p_seq + delta)."""
    p_seq = softmax_from_log(np.log(np.asarray(p0, float) + EPS) + a)
    return softmax_from_log(np.log(p_seq + EPS) + delta), p_seq


def finger_permutation(seed_name="CAM0094_BONE_PERM") -> np.ndarray:
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


# ------------------------------------------------------------------ readers
def read_csv(path) -> list[dict]:
    path = Path(path)
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_scene_predictions(path=None, model_key=SCENE_MODEL_KEY) -> list[dict]:
    """Scene focal predictions with EVERY reference-focal column removed.

    The cache on disk contains `gt_fx` and error columns. This reader drops
    them, so a PHASE A/B/C script that uses it cannot see the reference focal
    even by accident. `self_audit.py` fails if any phase-A/B/C script reads the
    cache by any other route.
    """
    rows = read_csv(SCENE_SRC if path is None else path)
    out = []
    for r in rows:
        if model_key is not None and r.get("model_key") != model_key:
            continue
        out.append({k: v for k, v in r.items()
                    if not any(k.startswith(p)
                               for p in FOCAL_FORBIDDEN_PREFIXES)})
    return out


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
    v = [x for x in np.asarray(vals, float).ravel() if np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def scalar_focal(fx, fy) -> float:
    """Scalar focal convention, unchanged from CAM-EXP-003/005/008."""
    return float(np.sqrt(float(fx) * float(fy)))


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
