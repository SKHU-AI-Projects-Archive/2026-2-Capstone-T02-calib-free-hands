"""CAM-EXP-001.1 orchestrator - run the whole GigaHands bad-view diagnosis.

    python src/run_all.py

Reads CAM-EXP-001 as read-only input and writes everything under this run
directory only.
"""
from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

from bad_view_common import (CAM_EXP_001, RAW_CSV, RUN_DIR, VIEW_KEYS, rel,
                             write_csv)
import step1_view_stats
import step2_hypotheses
import step3_figures
from experiments.src.datasets.common import DATASETS_ROOT

log = logging.getLogger("cam-exp-001.1")

# Ordered by how much of the observed failure each explains; support level is
# taken from the measurements, not asserted in advance.
RANK_ORDER = ["H1", "H7", "H6", "H3", "H5", "H2", "H4"]


def git(*args, default=""):
    try:
        return subprocess.check_output(["git", *args], cwd=RUN_DIR,
                                       text=True, stderr=subprocess.DEVNULL).strip()
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
    for name in ("cam-exp-001.1",):
        logging.getLogger(name).setLevel(logging.INFO)


def write_environment(cls: dict) -> dict:
    import cv2
    import matplotlib
    gigahands_root = DATASETS_ROOT / "gigahands" / "demo_all" / "raw" / "hand_pose"
    env = {
        "experiment": "CAM-EXP-001.1_gigahands_bad_view_diagnosis",
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
            "cam_exp_001_run_dir": rel(CAM_EXP_001),
            "cam_exp_001_raw_csv": rel(RAW_CSV),
            "cam_exp_001_raw_csv_exists": RAW_CSV.exists(),
            "gigahands_root": rel(gigahands_root),
            "gigahands_root_exists": gigahands_root.exists(),
            "gigahands_sequences": sorted(p.name for p in gigahands_root.iterdir()
                                          if p.is_dir()) if gigahands_root.exists() else [],
            "shared_src_modules_present": {
                m: (DATASETS_ROOT.parent / "src" / m).exists()
                for m in ("datasets", "geometry", "metrics", "visualization")},
        },
        "view_classification": cls,
    }
    (RUN_DIR / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    return env


def write_config(cls: dict) -> None:
    cfg = {
        "experiment": "CAM-EXP-001.1_gigahands_bad_view_diagnosis",
        "purpose": ("identify, as far as the available evidence allows, why some "
                    "GigaHands views reproject badly in CAM-EXP-001"),
        "input": {"cam_exp_001_raw_csv": rel(RAW_CSV),
                  "reused_read_only": True,
                  "dataset_re_read": "only for lag, distortion, flip, hand-identity and figures"},
        "view_classification_rule": cls,
        "confidence_filter": "keypoints_2d confidence >= 0.5 (as in CAM-EXP-001)",
        "lag_sweep_range": [-5, 5],
        "max_views_per_class_for_dataset_re_read": step2_hypotheses.MAX_PER_CLASS,
        "hypotheses": {
            "H1": "plain 2D detection failure",
            "H2": "frame synchronisation / temporal lag",
            "H3": "camera mapping / camera parameter mis-assignment",
            "H4": "distortion model misuse",
            "H5": "coordinate system / axis / flip misinterpretation",
            "H6": "sequence-specific or hand-specific concentration",
            "H7": "hand identity swap in the 2D annotation (added after inspecting figures)",
        },
        "support_scale": ["supported", "partially supported",
                          "insufficient evidence", "close to rejected"],
        "note": ("Thresholds are derived from the observed distribution (Otsu on "
                 "log per-view median error), not fixed in code."),
    }
    (RUN_DIR / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def main() -> None:
    setup_logging()
    log.info("=== CAM-EXP-001.1 GigaHands bad-view diagnosis ===")
    log.info("branch=%s commit=%s", git("rev-parse", "--abbrev-ref", "HEAD"),
             git("rev-parse", "HEAD")[:8])
    log.info("input (read-only): %s", rel(RAW_CSV))

    log.info("step 1: per-view statistics and case selection")
    cls = step1_view_stats.main()
    write_config(cls)
    write_environment(cls)
    log.info("view classes: %s (otsu=%.2f px)", cls["counts"], cls["otsu_threshold_px"])

    log.info("step 2: hypothesis tests H1..H7")
    rows = step2_hypotheses.read_rows(RUN_DIR / "results" / "raw" / "per_view_error_summary.csv")
    class_of = {tuple(r[k] for k in VIEW_KEYS): r["view_class"] for r in rows}
    findings = step2_hypotheses.main(class_of)

    hyp_rows = []
    for hid in sorted(findings):
        f = findings[hid]
        hyp_rows.append({"hypothesis_id": hid, "hypothesis": f["hypothesis"],
                         "verdict": f["verdict"], "evidence": f["evidence"],
                         "metrics": json.dumps({k: v for k, v in f.items()
                                                if k not in ("hypothesis", "verdict", "evidence")})})
    write_csv(RUN_DIR / "results" / "raw" / "hypothesis_tests.csv", hyp_rows)

    ranking = []
    for i, hid in enumerate([h for h in RANK_ORDER if h in findings], start=1):
        f = findings[hid]
        ranking.append({"rank": i, "hypothesis_id": hid, "hypothesis": f["hypothesis"],
                        "support_level": f["verdict"],
                        "evidence_summary": f["evidence"]})
    write_csv(RUN_DIR / "results" / "summary" / "root_cause_ranking.csv", ranking)

    log.info("step 3: figures")
    step3_figures.main()

    _human_summary(findings, cls)
    for hid in sorted(findings):
        log.info("%s %-55s -> %s", hid, findings[hid]["hypothesis"], findings[hid]["verdict"])
    log.info("done -> %s", rel(RUN_DIR))


def _human_summary(findings: dict, cls: dict) -> None:
    import csv as _csv
    census_p = RUN_DIR / "results" / "summary" / "failure_mode_census.csv"
    census = {}
    if census_p.exists():
        with open(census_p, newline="", encoding="utf-8") as f:
            census = {r["failure_mode"]: r["share"] for r in _csv.DictReader(f)}
    counts = cls["counts"]
    total = sum(counts.values())
    rows = [
        {"question": "How many GigaHands views were examined?",
         "answer": f"{total} views (sequence x camera x frame x hand) from CAM-EXP-001"},
        {"question": "How many views have no usable 2D annotation at all?",
         "answer": (f"{counts.get('zero_sentinel', 0)} ({counts.get('zero_sentinel', 0) / total:.1%}) - "
                    "every GT 2D joint is exactly (0,0) with confidence 1.0")},
        {"question": "How many views agree well?",
         "answer": f"{counts.get('good', 0)} ({counts.get('good', 0) / total:.1%}) at <= "
                   f"{cls['good_max_median_px']} px median"},
        {"question": "How many views genuinely disagree?",
         "answer": f"{counts.get('bad', 0)} ({counts.get('bad', 0) / total:.1%}) at >= "
                   f"{cls['bad_min_median_px']} px median"},
        {"question": "Is our camera model wrong?",
         "answer": ("No evidence of that. No flip, axis swap, lag or distortion setting "
                    "repairs the bad views, and the projection lands on a real hand.")},
        {"question": "What do the bad views actually show?",
         "answer": ("Mostly the annotation describing the other hand: in "
                    f"{findings['H7'].get('frac_annotation_lands_on_other_hand', '')} of bad "
                    "views with both hands present the annotation sits within one hand-width "
                    "of the hand we did NOT project.")},
        {"question": "Most likely root cause",
         "answer": ("Annotation-side failure in GigaHands keypoints_2d: undetected hands "
                    "written as (0,0) sentinels, plus left/right identity swaps.")},
        {"question": "What should downstream experiments do?",
         "answer": ("Drop (0,0) sentinel rows, and filter or re-associate per-view 2D by "
                    "agreement with the multi-view 3D before using it as a reference.")},
    ]
    for mode, share in census.items():
        rows.append({"question": f"Failure-mode share: {mode}", "answer": share})
    write_csv(RUN_DIR / "results" / "summary" / "human_readable_summary.csv", rows)


if __name__ == "__main__":
    main()
