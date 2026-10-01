"""CAM-EXP-009.3.1 — participant-aware reanalysis of CAM-EXP-009.3.

This is a POST-RESULT PROVENANCE CORRECTION, not a preregistered confirmatory
experiment. CAM-EXP-009.3's results were already known when this analysis was
designed, and the provenance says so: `created_after_cam0093_results = true`.

No reference 3D is recomputed. CAM-EXP-009.3's frame bone vectors and geometry
templates are reused byte-identically as the source of truth. No focal is
estimated or read.

What changes is the GROUPING. CAM-EXP-009.3 could not find a participant field
in the local files and therefore analysed WITHIN_SEQUENCE vs CROSS_SEQUENCE.
The official GigaHands README documents the directory layout as
`p<participant id>-<scene>-<squence id>/`, so `p41-boxing-0021` and
`p41-plant-0004` are two sessions of participant p41. That makes three distinct
quantities separable, and they are never merged here:

    D_WITHIN_SESSION                 same participant AND same session
    D_SAME_PARTICIPANT_CROSS_SESSION same participant, different session (p41)
    D_CROSS_PARTICIPANT              different participants
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

# ---------------------------------------------------------------- source run
C093 = RUNS / "CAM-EXP-009_3_real_bilateral_geometry_separability"
C093_RAW = C093 / "results" / "raw"
C093_SUM = C093 / "results" / "summary"
C093_COMMIT = "1b92dc1"

SOURCE_ARTIFACTS = {
    "frame_bone_vectors.csv.gz": C093_RAW / "frame_bone_vectors.csv.gz",
    "geometry_templates.csv.gz": C093_RAW / "geometry_templates.csv.gz",
    "repeatability_distances.csv.gz": C093_RAW / "repeatability_distances.csv.gz",
    "view_repeatability_distances.csv.gz":
        C093_RAW / "view_repeatability_distances.csv.gz",
    "same_subject_distances.csv.gz": C093_RAW / "same_subject_distances.csv.gz",
    "cross_subject_distances.csv.gz":
        C093_RAW / "cross_subject_distances.csv.gz",
    "margins.csv.gz": C093_RAW / "margins.csv.gz",
    "identification.csv.gz": C093_RAW / "identification.csv.gz",
    "reference_circularity_audit.csv.gz":
        C093_RAW / "reference_circularity_audit.csv.gz",
    "control_summary.csv": C093_SUM / "control_summary.csv",
    "cam0093_verdict.json": C093_SUM / "cam0093_verdict.json",
    "reference_geometry_audit.json": C093_SUM / "reference_geometry_audit.json",
}

# facts CAM-EXP-009.3 recorded; re-checked against the artefacts, never forced
EXPECTED = {
    "reconstructions_attempted": 14015,
    "frame_vectors": 13903,
    "target_camera_in_reference_count": 0,
    "n_bones": 20,
    "max_frames_per_unit": 48,
}

# ---------------------------------------------------------------- identity
# Source: the official GigaHands README directory listing, which labels the
# field `p<participant id>`. NOT inferred from MANO shape, and not from
# filename intuition. See official_participant_provenance.md.
PARTICIPANT_OF = {
    "p36-tea-0010": "p36",
    "p41-boxing-0021": "p41",
    "p41-plant-0004": "p41",
    "p44-dog-0004": "p44",
    "p52-instrument-0034": "p52",
}
PARTICIPANTS = ["p36", "p41", "p44", "p52"]
MULTI_SESSION = {"p41": ["p41-boxing-0021", "p41-plant-0004"]}

ADEQUACY_BAR_PARTICIPANTS = 8      # unchanged from CAM-EXP-009.3
N_BOOTSTRAP = 10000
EPS = 1e-8
N_BONES = 20
BASE_SEED = 20260928


def zof(r):
    return np.array([float(r["z%02d" % b]) for b in range(N_BONES)], float)


def d_primary(zA, zB, mapping=None) -> float:
    """PRIMARY distance, identical to CAM-EXP-009.3: median absolute
    log-proportion difference. No new metric is introduced."""
    a = np.asarray(zA, float)
    b = np.asarray(zB, float)
    if mapping is not None:
        b = b[np.asarray(mapping, int)]
    if a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("nan")
    return float(np.median(np.abs(a - b)))


def finger_permutation(seed_name="CAM0093_BONE_PERM") -> np.ndarray:
    """Reproduces CAM-EXP-009.3's frozen within-finger permutation exactly."""
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


def med(vals) -> float:
    v = [x for x in np.asarray(vals, float).ravel() if np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


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


def participant_bootstrap(values, participants, stat=np.median,
                          n_boot=N_BOOTSTRAP, seed=BASE_SEED):
    """Cluster bootstrap with the PARTICIPANT as the cluster.

    p41's two sessions live in one cluster and are resampled together. With 4
    clusters the interval is unstable; it is reported as supplemental and
    never as strong evidence.
    """
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    v, c = v[ok], np.asarray(participants)[ok]
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
