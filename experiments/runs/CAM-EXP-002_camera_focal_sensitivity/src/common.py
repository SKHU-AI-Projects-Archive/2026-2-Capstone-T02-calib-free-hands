"""Shared paths, sample selection and constants for CAM-EXP-002.

Only the CAM-EXP-001.3 manifests decide which observations may be used:
bimanual work needs both hands PASS_STRICT, and a camera must have the RGB
segment that actually matches the annotation (CAM-EXP-001.2 M6). REVIEW and
EXCLUDE records are never used.
"""
from __future__ import annotations

import csv
import gzip
import sys
from collections import defaultdict
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parents[1]
EXPERIMENTS = RUN_DIR.parents[1]
REPO = EXPERIMENTS.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MANIFESTS = EXPERIMENTS / "manifests"
QC_CSV = MANIFESTS / "gigahands_demo_qc_v1.csv.gz"
BIMANUAL_CSV = MANIFESTS / "gigahands_demo_bimanual_clean_v1.csv.gz"
CAMERA_CSV = MANIFESTS / "gigahands_demo_camera_benchmark_v1.csv.gz"

CACHE_DIR = RUN_DIR / "cache" / "hand_inference"

# From models/model_config_wilor.yaml, confirmed in focal_usage_audit.md
CFG_FOCAL_LENGTH = 1000.0
CFG_IMAGE_SIZE = 256.0

ALPHAS = [0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20]
SEED = 20260922


def rel(p) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return p.as_posix()


def read_csv(path: Path) -> list:
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list, gzipped: bool | None = None) -> None:
    if not rows:
        return
    if gzipped is None:
        gzipped = str(path).endswith(".gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen and not str(k).startswith("_"):
                seen.add(k)
                keys.append(k)
    opener = ((lambda: gzip.open(path, "wt", newline="", encoding="utf-8")) if gzipped
              else (lambda: open(path, "w", newline="", encoding="utf-8")))
    with opener() as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"  wrote {rel(path)} ({len(rows)} rows)")


def pipeline_focal(width: int, height: int) -> float:
    """The focal the existing pipeline uses (rgb_predictor.py:407-411)."""
    return CFG_FOCAL_LENGTH / CFG_IMAGE_SIZE * max(width, height)


def usable_cameras() -> set:
    """(sequence, camera) whose downloaded RGB matches the annotated take."""
    return {(r["sequence"], r["camera"]) for r in read_csv(CAMERA_CSV)
            if r["usable_for_camera_benchmark"] == "1"}


def select_frames(n_target: int, seed: int = SEED, require_rgb: bool = True) -> list:
    """Stratified pick of bimanual-clean frames, balanced over sequence x camera.

    Sampling rule (recorded in config.json):
      1. start from gigahands_demo_bimanual_clean_v1 (both hands PASS_STRICT);
      2. keep only cameras whose RGB segment matches the annotation;
      3. group by (sequence, camera) and take frames round-robin from shuffled
         groups, so no sequence or camera can dominate;
      4. within a group, frames are taken evenly spaced in time.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    rows = read_csv(BIMANUAL_CSV)
    ok = usable_cameras()
    if require_rgb:
        rows = [r for r in rows if (r["sequence"], r["camera"]) in ok]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["sequence"], r["camera"])].append(r)
    for k in groups:
        groups[k].sort(key=lambda r: int(r["frame"]))
        g = groups[k]
        if len(g) > 1:                       # even temporal spread inside a group
            idx = np.linspace(0, len(g) - 1, num=len(g)).astype(int)
            groups[k] = [g[i] for i in idx]
    keys = sorted(groups)
    rng.shuffle(keys)
    out, cursor = [], defaultdict(int)
    while len(out) < n_target:
        progressed = False
        for k in keys:
            i = cursor[k]
            if i < len(groups[k]):
                out.append(groups[k][i])
                cursor[k] += 1
                progressed = True
                if len(out) >= n_target:
                    break
        if not progressed:
            break
    return out
