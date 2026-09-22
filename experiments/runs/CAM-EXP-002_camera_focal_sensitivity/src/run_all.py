"""CAM-EXP-002 orchestrator.

Two environments are involved on purpose:

  * inference  -> experiments/.venv-anyhand  (torch 2.1.2+cu121, numpy 1.26.4)
                  because the professor's pipeline pins numpy<2; keeping it
                  separate leaves the analysis env - and every CAM-EXP-001.x
                  result produced with it - untouched.
  * analysis   -> experiments/.venv          (numpy 2.x, matplotlib)

    # step 1, in .venv-anyhand
    PYTHONPATH=<repo> .venv-anyhand/Scripts/python src/run_inference.py --n 1200
    # steps 2-4, in .venv
    PYTHONPATH=<repo> .venv/Scripts/python src/run_all.py
"""
from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

from common import (ALPHAS, BIMANUAL_CSV, CACHE_DIR, CAMERA_CSV, CFG_FOCAL_LENGTH,
                    CFG_IMAGE_SIZE, QC_CSV, RUN_DIR, SEED, read_csv, rel)
import focal_sweep
import figures

log = logging.getLogger("cam-exp-002")


def git(*args, default=""):
    try:
        return subprocess.check_output(["git", *args], cwd=RUN_DIR, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return default


def setup_logging() -> None:
    (RUN_DIR / "logs").mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(RUN_DIR / "logs" / "run.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)


def write_config() -> None:
    idx = read_csv(RUN_DIR / "results" / "raw" / "inference_cache_index.csv")
    seqs = sorted({r["sequence"] for r in idx})
    cams = sorted({r["camera"] for r in idx})
    per_seq = {s: sum(1 for r in idx if r["sequence"] == s) for s in seqs}
    cfg = {
        "experiment": "CAM-EXP-002_camera_focal_sensitivity",
        "question": ("how far off is the absolute 3D hand position and depth of the "
                     "existing pipeline when the camera focal length is wrong"),
        "not_in_scope": ["developing any calibration model",
                         "changing the professor's pipeline"],
        "read_only_inputs": [rel(QC_CSV), rel(BIMANUAL_CSV), rel(CAMERA_CSV)],
        "data_selection": {
            "source": "gigahands_demo_bimanual_clean_v1 (LEFT and RIGHT both PASS_STRICT)",
            "rgb_filter": "camera must have the RGB segment matching the annotation "
                          "(gigahands_demo_camera_benchmark_v1.usable_for_camera_benchmark)",
            "review_or_exclude_used": False,
            "sampling_rule": ("group by (sequence, camera); shuffle groups with the seed; "
                              "take frames round-robin across groups, evenly spaced in "
                              "time within a group"),
            "seed": SEED,
            "target_frames": 1200,
            "frames_selected": len(idx),
            "sequences": per_seq,
            "n_distinct_cameras": len(cams),
        },
        "focal_definitions": {
            "PIPELINE_BASELINE_FOCAL": {
                "formula": "EXTRA.FOCAL_LENGTH / MODEL.IMAGE_SIZE * max(W, H)",
                "constants": {"FOCAL_LENGTH": CFG_FOCAL_LENGTH,
                              "IMAGE_SIZE": CFG_IMAGE_SIZE},
                "value_for_1280x720_px": 5000.0,
                "kind": "training-convention virtual focal, in original image pixels",
                "source": "rgb_predictor.py:407-411",
            },
            "GT_NATIVE_FX_FY": {
                "source": "GigaHands optim_params.txt per camera",
                "coordinate_system": "original 1280x720 image pixels",
            },
            "GT_EFFECTIVE_FOCAL": {
                "value": "= GT_NATIVE_FX (identity transform)",
                "why": ("the pipeline focal is already in original full-image pixels and "
                        "there is no image resize, so no conversion is required; "
                        "derived in focal_usage_audit.md section 4"),
                "scalar_choice": ("fx is used because the pipeline takes a single scalar; "
                                  "fy differs by 0.1-0.3 % and both are stored per row"),
            },
        },
        "conditions": ["PIPELINE_BASELINE"] + [f"GT_x{a:.2f}" for a in ALPHAS],
        "alphas": ALPHAS,
        "held_fixed_across_conditions": [
            "RGB frame", "detector output", "bounding box", "crop",
            "pred_cam", "root-relative joints", "MANO result",
            "hand association", "camera rotation", "reference 3D"],
        "recomputed_per_condition": ["camera translation tz (the only focal-dependent term)"],
        "hand_association_rule": {
            "signal": "2D only: predicted keypoint centroid vs provided 2D annotation centroid",
            "never_uses": "3D error (that would select the quantity under study)",
            "margin": focal_sweep.ASSOC_MARGIN,
            "max_distance_px": focal_sweep.ASSOC_MAX_PX,
            "on_failure": "record AMBIGUOUS_HAND_ASSOCIATION and drop the frame",
        },
        "reference_3d": {
            "source": "GigaHands provided 3D (not external ground truth)",
            "transform": "world -> camera with the CAM-EXP-001 validated R, t",
            "units": "metres in, millimetres reported",
        },
        "principal_point": ("not swept: the pipeline uses the image centre and never reads "
                            "the calibrated cx, cy, so it is structurally insensitive to it "
                            "(focal_usage_audit.md section 6)"),
        "metric_families": {
            "ABSOLUTE_ERROR": "prediction vs provided 3D; contains hand-model, detector and "
                              "weak-perspective error as well as focal error",
            "INCREMENTAL_FOCAL_EFFECT": "prediction at alpha vs prediction at alpha = 1.0; "
                                        "isolates the focal exactly",
        },
    }
    (RUN_DIR / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def write_environment(extra: dict) -> None:
    import matplotlib
    env = {
        "experiment": "CAM-EXP-002_camera_focal_sensitivity",
        "date": datetime.now(timezone.utc).isoformat(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "analysis_python": sys.version.split()[0],
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "inference_environment": {
            "venv": "experiments/.venv-anyhand (git-ignored)",
            "torch": "2.1.2+cu121", "numpy": "1.26.4",
            "gpu": "NVIDIA GeForce RTX 4060 Ti (16 GB), CUDA available",
            "note": "kept separate because the pipeline pins numpy<2; the analysis "
                    "environment and all CAM-EXP-001.x results are unaffected",
            "extra_fixes": ["PYOPENGL_PLATFORM=win32 so WiLoR's renderer import does not "
                            "try EGL on Windows",
                            "working directory set to the repo root because "
                            "model_config_wilor.yaml uses './mano_data/...'"],
        },
        "model": {"checkpoint": "models/anyhand_wilor.ckpt", "backend": "wilor",
                  "detector": "models/detector.pt"},
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        **extra,
    }
    (RUN_DIR / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")


def main() -> None:
    setup_logging()
    log.info("=== CAM-EXP-002 focal sensitivity ===")
    if not any(CACHE_DIR.glob("*.npz")):
        raise SystemExit("no cached inference; run src/run_inference.py in .venv-anyhand first")

    log.info("step 1: focal sweep on cached inference")
    sw = focal_sweep.sweep()
    log.info("  %s", sw)
    log.info("step 2: summaries")
    su = focal_sweep.summarise()
    log.info("  %s", su)
    log.info("step 3: figures")
    fg = figures.main()
    log.info("  %s", fg)

    write_config()
    write_environment({"sweep": sw, "summaries": su,
                       "depth_ratio_max_deviation": fg["max_depth_ratio_deviation"]})
    log.info("done -> %s", rel(RUN_DIR))


if __name__ == "__main__":
    main()
