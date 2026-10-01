"""Shared paths and helpers for the qualitative research demo.

This is a QUALITATIVE demonstration, not a benchmark. The sample clips have no
dataset-provided camera calibration, so nothing here may be called ground truth,
correct, or accurate.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1]
SAMPLES = DEMO.parent
REPO = SAMPLES.parent

CACHE = DEMO / "cache"
CALIB = DEMO / "calibration"
META = DEMO / "metadata"
OUT_LEGACY = DEMO / "legacy"
OUT_ANYCALIB = DEMO / "anycalib"
OUT_E2 = DEMO / "e2"
OUT_COMPARISON = DEMO / "comparison"
OUT_FRAMES = DEMO / "comparison_frames"

# Pre-registered: 8 frames spread uniformly over the clip, matching the frame
# budget CAM-EXP-004/004.1 used. CAM-EXP-004.1 measured no further improvement
# from 8 to 64 frames on GigaHands; that is a compute policy here, not a claim
# that 8 is optimal for these clips.
N_CALIB_FRAMES = 8


def videos() -> list[Path]:
    return sorted(p for p in SAMPLES.glob("*.mp4") if p.is_file())


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_json(p, obj) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(p, rows, fieldnames=None) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames, seen = [], set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def read_csv(p) -> list[dict]:
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def pipeline_focal(width: int, height: int, focal_length: float = 1000.0,
                   image_size: float = 256.0) -> float:
    """The pipeline's own focal convention, as a formula rather than a constant.

    rgb_predictor.py:407-411 computes
        scaled_focal = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE
                       * max(W, H)
    with FOCAL_LENGTH = 1000 and IMAGE_SIZE = 256 in
    models/model_config_wilor.yaml. That is a TRAINING-CONVENTION VIRTUAL FOCAL,
    never read from any calibration (CAM-EXP-002 focal_usage_audit.md).

    The value actually used is read back from the cached per-hand
    `focal_length` field; this helper exists so the number is never hard-coded.
    """
    return float(focal_length) / float(image_size) * float(max(width, height))
