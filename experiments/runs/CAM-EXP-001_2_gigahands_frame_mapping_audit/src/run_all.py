"""CAM-EXP-001.2 orchestrator - GigaHands frame / annotation mapping audit.

    PYTHONPATH=<repo root> python src/run_all.py

CAM-EXP-001 and CAM-EXP-001.1 are read-only inputs; all output stays in this
run directory. The dataset is never modified.
"""
from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

from audit_common import (CAM_EXP_001, CAM_EXP_0011, HAND_POSE_ROOT, RUN_DIR,
                          rel, sequences)
import step1_structure_audit
import step2_census_reaudit
import step3_figures

log = logging.getLogger("cam-exp-001.2")


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
    fh = logging.FileHandler(RUN_DIR / "logs" / "run.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)


def write_config() -> None:
    cfg = {
        "experiment": "CAM-EXP-001.2_gigahands_frame_mapping_audit",
        "purpose": ("decide whether GigaHands bad views come from the dataset's own "
                    "annotation or from our loader/indexing of RGB, 2D, 3D, "
                    "chosen_frames and cameras"),
        "read_only_inputs": [rel(CAM_EXP_001), rel(CAM_EXP_0011)],
        "dataset_root": rel(HAND_POSE_ROOT),
        "official_source_consulted": {
            "repository": "https://github.com/brown-ivl/GigaHands",
            "file": "render_mesh_video.py",
            "function": "hand_pose_loader(keypoints3d_path)",
            "what_it_shows": (
                "reads chosen_frames_left.json / chosen_frames_right.json as SETS of "
                "frame ids, returns their union and intersection, then indexes MANO "
                "parameters by [chosen_hand_union_frames.index(f) for f in "
                "chosen_video_frames] - i.e. params are indexed by POSITION IN THE "
                "UNION, not by frame id"),
            "annotation_pipeline": "2D keypoints from HaMeR; left/right assignment via ViTPose",
        },
        "mapping_rules_verified": {
            "M1": "keypoints_2d row index == RGB video frame index",
            "M2": "keypoints_3d row index == RGB video frame index (rows 0..max(chosen))",
            "M3": "params / repro_2d_vid / repro_3d_vid / mano_vid use union position",
            "M4": "chosen_frames_<hand> = frame ids with a valid 3D pose for that hand",
        },
        "diagnostic_bands": {
            "good_max_px": step2_census_reaudit.GOOD_MAX_PX,
            "bad_min_px": step2_census_reaudit.BAD_MIN_PX,
            "note": "same bands as CAM-EXP-001.1 for comparability; NOT the final "
                    "quality gate, which is decided in CAM-EXP-001.3",
        },
        "census_frame_stride": step2_census_reaudit.FRAME_STRIDE,
        "confidence_threshold": 0.5,
        "hand_swap_criterion": ("annotation centroid closer to the other hand's projection "
                                "AND within 60 px of it; centroids are used because left "
                                "and right skeletons have mirrored joint ordering"),
        "loader_change": {
            "file": "experiments/src/datasets/gigahands.py",
            "change": "added drop_zero_2d (default True) and is_zero_2d(); "
                      "documented index conventions and the video-matching rule",
            "note": "CAM-EXP-001 raw outputs are NOT regenerated or overwritten",
        },
        "not_done_here": ["leave-one-camera-out triangulation at scale",
                          "final PASS/EXCLUDE quality gate", "CAM-EXP-002"],
    }
    (RUN_DIR / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def write_environment(extra: dict) -> dict:
    import cv2
    import matplotlib
    env = {
        "experiment": "CAM-EXP-001.2_gigahands_frame_mapping_audit",
        "date": datetime.now(timezone.utc).isoformat(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "matplotlib_version": matplotlib.__version__,
        "gpu_used": False,
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "git_remote": git("remote", "get-url", "origin"),
        "preflight_checks": {
            "cam_exp_001_present": (CAM_EXP_001 / "results" / "raw" /
                                    "reprojection_per_joint.csv.gz").exists(),
            "cam_exp_001_1_present": (CAM_EXP_0011 / "results" / "raw" /
                                      "representative_cases.csv").exists(),
            "gigahands_root": rel(HAND_POSE_ROOT),
            "gigahands_root_exists": HAND_POSE_ROOT.exists(),
            "sequences_found": [d.name for d in sequences()],
        },
        **extra,
    }
    (RUN_DIR / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    return env


def main() -> None:
    setup_logging()
    log.info("=== CAM-EXP-001.2 GigaHands frame/annotation mapping audit ===")
    log.info("branch=%s commit=%s", git("rev-parse", "--abbrev-ref", "HEAD"),
             git("rev-parse", "HEAD")[:8])
    write_config()

    log.info("step 1: frame structure, timestamps, chosen_frames semantics")
    structure = step1_structure_audit.main()
    log.info("  2D rows == RGB frames: %s | 3D rows == max(chosen)+1: %s | "
             "repro == |union|: %s | params == |union|: %s",
             structure["kp2d_rows_eq_rgb_frames"],
             structure["kp3d_rows_eq_max_chosen_plus1"],
             structure["repro_videos_eq_union"], structure["params_eq_union"])

    log.info("step 2: QC census and re-audit of CAM-EXP-001.1 cases")
    census = step2_census_reaudit.main()
    log.info("  census: %s", census["status_counts"])
    log.info("  systematic-swap cameras: %s", census["systematic_swap_cameras"])

    log.info("step 3: bimanual figures and official repro comparison")
    step3_figures.main()

    write_environment({"structure_verdicts": {k: v for k, v in structure.items()
                                              if isinstance(v, bool)},
                       "census_status_counts": census["status_counts"]})
    log.info("done -> %s", rel(RUN_DIR))


if __name__ == "__main__":
    main()
