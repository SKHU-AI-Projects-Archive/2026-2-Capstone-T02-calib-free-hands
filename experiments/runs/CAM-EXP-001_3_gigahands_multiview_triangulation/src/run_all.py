"""CAM-EXP-001.3 orchestrator - independent multi-view triangulation and QC.

    PYTHONPATH=<repo root> python src/run_all.py [--stride N] [--no-resume]

Order is deliberate and is the non-circularity guarantee in practice:
tests -> clean-control reconstruction -> threshold derivation -> suspected
cases -> full pass -> manifests -> figures. The dataset-provided 3D is read
only after a hypothesis is frozen.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

from common import (CAM_EXP_001, CAM_EXP_0011, CAM_EXP_0012, CENSUS_CSV,
                    MANIFESTS, RUN_DIR, TAKES, rel)
import step1_smoke
import step2_full_qc
import step3_manifests
import step4_figures

log = logging.getLogger("cam-exp-001.3")


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
    for name in ("cam-exp-001.3",):
        logging.getLogger(name).setLevel(logging.INFO)


def run_tests() -> dict:
    """Synthetic geometry tests and the non-circularity tests must pass first."""
    out = {}
    for mod, label in (("tests.test_triangulation_synthetic", "synthetic_geometry"),
                       ("tests.test_no_circularity", "non_circularity")):
        r = subprocess.run([sys.executable, "-m", mod], cwd=RUN_DIR / "src",
                           capture_output=True, text=True,
                           env={**__import__("os").environ,
                                "PYTHONPATH": str(RUN_DIR.parents[2])})
        ok = r.returncode == 0
        out[label] = {"passed": ok,
                      "tail": (r.stdout or r.stderr).strip().splitlines()[-1:]}
        log.info("tests[%s]: %s", label, "PASSED" if ok else "FAILED")
        if not ok:
            log.error("%s", (r.stdout or "")[-2000:])
    return out


def write_config(thr: dict, args) -> None:
    cfg = {
        "experiment": "CAM-EXP-001.3_gigahands_multiview_triangulation",
        "purpose": ("decide whether the GigaHands hand-identity mismatches are a "
                    "per-camera 2D annotation problem, a dataset 3D problem, or "
                    "undecidable, using a reconstruction that never sees the "
                    "dataset's 3D"),
        "non_circularity_rules": [
            "reconstruction inputs are only keypoints_2d and camera calibration",
            "the camera under test is removed from the observation set before "
            "reconstruction",
            "dataset-provided 3D is read only after the hypothesis is frozen, and "
            "never feeds back into inlier selection",
            "CAM-EXP-001.2 status is used only to choose which cases to inspect, "
            "never as truth and never as an inlier criterion",
            "all-(0,0) records are excluded as observations (M5)",
        ],
        "mapping_rules_from_001_2": {
            "M1": "RGB frame i == timestamp line i == keypoints_2d row i == keypoints_3d row i",
            "M2": "chosen_frames_<hand> = frame ids with a valid 3D pose for that hand",
            "M3": "rows outside chosen_frames are placeholders and are not used",
            "M4": "params / repro_*_vid use the position within sorted(union)",
            "M5": "all-(0,0) 2D is an observed invalid pattern",
            "M6": "some p52-instrument-0034 cameras ship a different video segment",
        },
        "read_only_inputs": [rel(CAM_EXP_001), rel(CAM_EXP_0011), rel(CAM_EXP_0012),
                             rel(CENSUS_CSV)],
        "triangulation": {
            "method": "RANSAC over camera pairs, consensus refit, per joint",
            "distortion": "observations undistorted to normalised coordinates before "
                          "geometry; residuals evaluated in pixels with distortion "
                          "re-applied",
            "min_joints_per_view": 8,
            "confidence_threshold": 0.5,
        },
        "full_pass": {
            "stride": args.stride,
            "scope": "every chosen frame x every annotated camera x both hands",
            "loco_variant": "consensus refit excluding the held-out camera "
                            "(see loco.py); validated against the strict variant",
            "rgb_decoded": False,
        },
        "thresholds": thr,
        "threshold_derivation": "see results/summary/_thresholds.json",
        "qc_statuses": ["PASS_STRICT", "PASS_SINGLE_HAND", "REVIEW", "EXCLUDE"],
        "dataset_mutation": "none - the dataset is immutable; manifests carry the QC",
    }
    (RUN_DIR / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def write_environment(extra: dict) -> None:
    import cv2
    import matplotlib
    env = {
        "experiment": "CAM-EXP-001.3_gigahands_multiview_triangulation",
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
            "cam_exp_001_2_census_present": CENSUS_CSV.exists(),
            "sequences": [t.name for t in TAKES],
            "n_cameras_per_sequence": {t.name: len(t.cameras_2d("left")) for t in TAKES},
            "n_chosen_frames": {t.name: len(t.union_sorted) for t in TAKES},
        },
        **extra,
    }
    (RUN_DIR / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=1,
                    help="frame stride for the full pass (1 = every chosen frame)")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--skip-full", action="store_true",
                    help="reuse an existing full pass")
    args = ap.parse_args()

    setup_logging()
    t_start = time.time()
    log.info("=== CAM-EXP-001.3 independent multi-view triangulation ===")
    log.info("branch=%s commit=%s", git("rev-parse", "--abbrev-ref", "HEAD"),
             git("rev-parse", "HEAD")[:8])

    log.info("step 0: regression tests")
    tests = run_tests()

    log.info("step 1: clean-control validation, threshold derivation, targeted cases")
    smoke = step1_smoke.main()
    thr = smoke["thresholds"]["thresholds"]
    write_config(thr, args)
    if not smoke["control_sanity"]["passed"]:
        log.error("CONTROL SANITY FAILED - stopping before the full pass: %s",
                  smoke["control_sanity"])
        write_environment({"tests": tests, "smoke": smoke})
        raise SystemExit("clean-control validation failed; fix triangulation first")
    log.info("control sanity passed: %s", smoke["control_sanity"])

    log.info("step 1b: fast-vs-strict leave-one-out agreement")
    agree = step2_full_qc.validate_fast_vs_strict(thr)
    log.info("  %s", agree)

    if not args.skip_full:
        log.info("step 2: full-demo QC (stride=%d)", args.stride)
        full = step2_full_qc.main(thr, stride=args.stride, resume=not args.no_resume)
        log.info("  %d leave-one-out rows", full["n_rows"])

    log.info("step 3: QC manifests")
    summary = step3_manifests.build(thr, git_commit=git("rev-parse", "HEAD"))
    step3_manifests.before_after(summary)
    log.info("  qc status: %s", summary["qc_status_counts"])
    log.info("  bimanual clean frames: %d", summary["bimanual_clean_frames"])

    log.info("step 4: figures")
    step4_figures.main(thr)

    write_environment({"tests": tests,
                       "control_sanity": smoke["control_sanity"],
                       "fast_vs_strict": agree,
                       "qc_status_counts": summary["qc_status_counts"],
                       "case_counts": summary["case_counts"],
                       "bimanual_clean_frames": summary["bimanual_clean_frames"],
                       "runtime_seconds": round(time.time() - t_start, 1)})
    log.info("done in %.1f s -> %s", time.time() - t_start, rel(RUN_DIR))


if __name__ == "__main__":
    main()
