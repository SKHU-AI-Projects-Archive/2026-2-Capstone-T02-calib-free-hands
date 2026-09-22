"""Write config.json and environment.json for CAM-EXP-003."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from common import (CAMERA_CSV, FRAMES_CSV, N_FRAMES_PER_VIEW, RUN_DIR, SEED,
                    load_frames, rel)


def git(*a, default=""):
    try:
        return subprocess.check_output(["git", *a], cwd=RUN_DIR, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return default


def sha256(p: Path) -> str:
    if not Path(p).exists():
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
        "experiment": "CAM-EXP-003_single_frame_calibration_benchmark",
        "question": ("how accurately can existing single-image calibration models "
                     "recover the physical focal of a static workspace camera from "
                     "one RGB frame"),
        "not_in_scope": ["developing a calibration model", "fine-tuning",
                         "multi-frame aggregation (CAM-EXP-004)",
                         "using hand geometry as a calibration cue"],
        "frame_manifest": {
            "path": rel(FRAMES_CSV),
            "frozen_before_any_model_ran": True,
            "source": rel(CAMERA_CSV) + " (usable_for_camera_benchmark == 1)",
            "rule": (f"every usable sequence x camera view, {N_FRAMES_PER_VIEW} frames "
                     "evenly spaced over the annotated frame range; no duplicates; "
                     "selection never uses any calibration model output and never uses "
                     "hand-annotation quality"),
            "seed": SEED,
            "n_views": len(views), "n_frames": len(frames),
        },
        "fairness": [
            "identical frames for every model, read from one lossless PNG cache",
            "failures are recorded as rows with success=0, never dropped",
            "success-only AND coverage-aware threshold rates are both reported",
            "ORACLE_GT is a reference line and is excluded from rankings",
        ],
        "thresholds_pct": [5, 10, 20],
        "why_5_pct": ("CAM-EXP-002 measured on this same data that a 5 % focal error "
                      "displaces the reconstructed hand by ~32.8 mm in depth; below "
                      "that the focal stops dominating the pipeline's own residual "
                      "error. The 5 % figure is a target taken from that experiment, "
                      "not a depth error measured here."),
        "models": [
            {"key": "anycalib", "repo": "https://github.com/javrtg/AnyCalib",
             "variant": "anycalib_pinhole / cam_id=pinhole"},
            {"key": "geocalib", "repo": "https://github.com/cvg/GeoCalib",
             "variant": "pinhole"},
            {"key": "pf_uncentered",
             "repo": "https://github.com/jinlinyi/PerspectiveFields",
             "variant": "Paramnet-360Cities-edina-uncentered"},
            {"key": "pf_centered",
             "repo": "https://github.com/jinlinyi/PerspectiveFields",
             "variant": "Paramnet-360Cities-edina-centered"},
            {"key": "demo_fixed", "repo": "this repository",
             "variant": "f = 1000/256*max(W,H) = 5000 px (CAM-EXP-002 audit)"},
            {"key": "image_center_pp", "repo": "n/a",
             "variant": "cx=W/2, cy=H/2 (principal-point reference)"},
            {"key": "oracle_gt", "repo": "n/a",
             "variant": "ground truth (reference only)"},
        ],
        "not_run": {"anycam": "OUT_OF_SCOPE_SINGLE_FRAME - video method; see "
                              "model_output_convention_audit.md"},
        "focal_conventions": ("see model_output_convention_audit.md and "
                              "tables/model_output_conversion.csv"),
        "gt_convention": ("GigaHands optim_params.txt intrinsics in original 1280x720 "
                          "pixels, validated in CAM-EXP-001; primary metric against "
                          "gt_fx with the gt_fy error stored per row"),
        "distortion": ("images are fed RAW; no model variant used predicts distortion, "
                       "so no distortion comparison is made"),
        "runtime_note": ("frames are decoded once into a lossless PNG cache; reported "
                         "runtimes are model inference plus cache read, measured on the "
                         "same machine for every model"),
    }
    (RUN_DIR / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    import numpy as np
    env = {
        "experiment": "CAM-EXP-003_single_frame_calibration_benchmark",
        "date": datetime.now(timezone.utc).isoformat(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "gpu": "NVIDIA GeForce RTX 4060 Ti (16 GB)",
        "analysis_python": sys.version.split()[0],
        "analysis_numpy": np.__version__,
        "model_environment": {
            "venv": "experiments/.venv-calib (git-ignored)",
            "python": "3.11", "torch": "2.1.2+cu121", "cuda": "12.1",
            "numpy": "1.26.4", "opencv": "4.10.0.84", "kornia": "0.7.2",
            "install_notes": [
                "kornia pinned to 0.7.2: newer kornia uses .any(dim=(-2,-1)) which the "
                "TorchScript in torch 2.1.2 rejects, breaking the GeoCalib import",
                "PerspectiveFields installed with --no-deps (+ yacs, timm, einops, "
                "scipy, scikit-learn, pyequilib, imageio, omegaconf, opencv-contrib): "
                "albumentations pulls stringzilla, which has no Windows wheel and is "
                "not needed for inference",
                "PYTHONUTF8=1 needed on this Korean-locale Windows to read UTF-8 "
                "package metadata",
                "PYOPENGL_PLATFORM=win32 so pyrender-dependent imports do not try EGL",
            ],
            "kept_separate_from": ["experiments/.venv (analysis)",
                                   "experiments/.venv-anyhand (CAM-EXP-002 pipeline)"],
        },
        "checkpoints": {
            "anycalib": ("anycalib_pinhole.pt, official GitHub release v1.0.0, 1.19 GB, "
                         "torch hub cache"),
            "geocalib": ("geocalib-pinhole.tar, official GitHub release v1.0, 111 MB, "
                         "torch hub cache"),
            "perspective_fields": ("Paramnet-360Cities-edina-{centered,uncentered}, "
                                   "official Hugging Face release"),
        },
        "frame_manifest_sha256_32": sha256(FRAMES_CSV),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "reproduce": [
            "python src/common.py                      # freeze the frame manifest",
            "python src/cache_frames.py                # decode frames once (PNG cache)",
            "python src/run_benchmark.py --models ...  # per model, optionally --start/--count",
            "python src/assemble.py                    # join chunk part files",
            "python src/evaluate.py && python src/figures.py",
        ],
    }
    (RUN_DIR / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print("wrote config.json and environment.json")


if __name__ == "__main__":
    main()
