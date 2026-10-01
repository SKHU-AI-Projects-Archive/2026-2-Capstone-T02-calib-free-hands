"""Write config.json and environment.json for CAM-EXP-003.1."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from common import (CAM003, FRAMES_CSV, RUN_DIR, UNDISTORT_ALPHA, load_frames, rel)


def git(*a, default=""):
    try:
        return subprocess.check_output(["git", *a], cwd=RUN_DIR, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return default


def sha256(p) -> str:
    p = Path(p)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:32]


def main() -> None:
    frames = load_frames()
    views = {(r["sequence"], r["camera"]) for r in frames}
    cfg = {
        "experiment": "CAM-EXP-003.1_distortion_aware_diagnostic",
        "question": ("how much of the single-frame focal error measured in "
                     "CAM-EXP-003 was caused by using a pinhole camera model on "
                     "lenses that have real radial distortion"),
        "not_in_scope": ["new calibration method", "fine-tuning",
                         "multi-frame aggregation (CAM-EXP-004)",
                         "hand geometry as a calibration cue"],
        "frames": {
            "manifest": rel(FRAMES_CSV),
            "reused_unchanged_from": "CAM-EXP-003",
            "n_frames": len(frames), "n_views": len(views),
            "note": "identical frames to CAM-EXP-003 so every comparison is paired",
        },
        "conditions": {
            "A_RAW_PINHOLE": {
                "description": "raw image + pinhole camera model",
                "source": "CAM-EXP-003 predictions reused verbatim, not re-run",
                "runs": ["anycalib", "geocalib", "pf_centered", "pf_uncentered"],
            },
            "B_RAW_DISTORTION_AWARE": {
                "description": "raw image + official distortion-aware variant",
                "deployable": True,
                "runs": ["anycalib_dist_radial (anycalib_dist, cam_id=radial:2)",
                         "anycalib_gen_radial (anycalib_gen, cam_id=radial:2)",
                         "geocalib_distorted_radial (weights=distorted, radial)"],
            },
            "C_GT_UNDISTORTED_PINHOLE": {
                "description": "image undistorted with the GT distortion, then the "
                               "unchanged pinhole models",
                "deployable": False,
                "why": "needs the ground-truth distortion, so it is an oracle "
                       "diagnostic only and is never reported as deployment performance",
                "undistortion": {
                    "function": "cv2.undistort with K_new from "
                                "cv2.getOptimalNewCameraMatrix",
                    "alpha": UNDISTORT_ALPHA,
                    "output_size": "original 1280x720 (no crop, no black border)",
                    "ground_truth_used": "K_new, NOT the original K",
                    "median_K_new_fx_over_orig_fx": 0.7734,
                },
                "runs": ["undist_anycalib_pinhole", "undist_geocalib_pinhole",
                         "undist_pf_centered", "undist_pf_uncentered"],
            },
        },
        "metrics": ("identical to CAM-EXP-003 - run_benchmark.to_row is imported "
                    "and reused so 'relative focal error' means the same thing"),
        "paired_statistics": {
            "unit": "per frame, same image in both conditions",
            "reported": ["median improvement in percentage points",
                         "fraction improved / worsened",
                         "nonparametric paired bootstrap 95% CI (2000 resamples)",
                         "per-camera median improvement"],
        },
        "distortion_comparison": "see distortion_model_audit.md; AnyCalib and "
                                 "GeoCalib radial k1/k2 share the OpenCV convention "
                                 "so coefficients are compared directly",
        "thresholds_pct": [5, 10, 20],
        "why_5_pct": ("imported from CAM-EXP-002: 5 % focal error ~ 32.8 mm of depth "
                      "displacement on this data. Nothing here measures depth."),
    }
    (RUN_DIR / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    import numpy as np
    env = {
        "experiment": "CAM-EXP-003.1_distortion_aware_diagnostic",
        "date": datetime.now(timezone.utc).isoformat(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "gpu": "NVIDIA GeForce RTX 4060 Ti (16 GB)",
        "analysis_python": sys.version.split()[0],
        "analysis_numpy": np.__version__,
        "model_environment": {
            "venv": "experiments/.venv-calib (git-ignored, shared with CAM-EXP-003)",
            "python": "3.11", "torch": "2.1.2+cu121", "cuda": "12.1",
            "numpy": "1.26.4", "opencv": "4.10.0.84", "kornia": "0.7.2",
        },
        "checkpoints": {
            "anycalib_dist": "anycalib_dist.pt, official GitHub release v1.0.0",
            "anycalib_gen": "anycalib_gen.pt, official GitHub release v1.0.0",
            "anycalib_pinhole": "anycalib_pinhole.pt, official GitHub release v1.0.0",
            "geocalib_distorted": "geocalib-distorted.tar, official release v1.0",
            "geocalib_pinhole": "geocalib-pinhole.tar, official release v1.0",
            "perspective_fields": "Paramnet-360Cities-edina-{centered,uncentered}",
        },
        "repos": {
            "AnyCalib": "https://github.com/javrtg/AnyCalib (main, pip install from git)",
            "GeoCalib": "https://github.com/cvg/GeoCalib (main, pip install from git)",
            "PerspectiveFields": "https://github.com/jinlinyi/PerspectiveFields "
                                 "@ d54be737d6eacfb9d39a2b7079a494924b45bb6c",
        },
        "frame_manifest_sha256_32": sha256(FRAMES_CSV),
        "condition_a_source": rel(CAM003 / "results" / "raw"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "reproduce": [
            "python src/undistort_frames.py                       # condition C cache",
            "python src/run_conditions.py --model <key> [--start N --count M]",
            "python src/evaluate.py",
            "python src/figures.py",
        ],
        "note": ("the CAM-EXP-003 lossless PNG frame cache is reused; no video is "
                 "re-decoded, so conditions A/B see bit-identical pixels"),
    }
    (RUN_DIR / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print("wrote config.json and environment.json")


if __name__ == "__main__":
    main()
