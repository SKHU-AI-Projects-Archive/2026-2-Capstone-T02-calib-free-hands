"""Collect summary.csv and pick the representative BEST_COMPARISON.mp4.

There is no ground-truth calibration for these clips, so there is no error
column and the "best" clip is chosen on recording properties only, never on how
favourable its numbers look.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from common import (CACHE, CALIB, DEMO, OUT_COMPARISON, pipeline_focal,  # noqa: E402
                    read_csv, read_json, videos, write_csv, write_json)


def main() -> None:
    probe = {r["video"]: r for r in read_csv(DEMO / "sample_probe.csv")}
    rows = []
    for p in videos():
        pr = probe.get(p.name, {})
        est_path = CALIB / f"{p.stem}_video_estimate.json"
        est = read_json(est_path) if est_path.exists() else {}
        hands_json = CACHE / f"{p.stem}_hands.json"
        hands = read_json(hands_json) if hands_json.exists() else {}
        render_json = DEMO / "metadata" / f"{p.stem}_render.json"
        render = read_json(render_json) if render_json.exists() else {}

        w = int(pr.get("width", 0)), int(pr.get("height", 0))
        rows.append({
            "source_video": p.name,
            "resolution": f"{w[0]}x{w[1]}",
            "fps": pr.get("fps", ""),
            "duration_sec": pr.get("duration_sec", ""),
            "camera_motion_status": pr.get("camera_motion_status", ""),
            "pipeline_focal_px": round(hands.get("pipeline_focal_px")
                                       or pipeline_focal(w[0], w[1]), 2),
            "anycalib_video_focal_px": _r(est.get("anycalib", {}).get("video_focal_px")),
            "geocalib_video_focal_px": _r(est.get("geocalib", {}).get("video_focal_px")),
            "e2_video_focal_px": _r(est.get("e2_frozen", {}).get("video_focal_px")),
            "current_demo_method": render.get("current_demo_method", "not rendered"),
            "n_calibration_frames": len(est.get("calibration_frames", [])),
            "n_processed_frames": hands.get("frames_processed", 0),
            "n_frames_with_left": hands.get("frames_with_left", 0),
            "n_frames_with_right": hands.get("frames_with_right", 0),
            "n_frames_bimanual": hands.get("frames_bimanual", 0),
            "notes": _notes(pr, hands, render),
        })
    write_csv(DEMO / "summary.csv", rows)

    rendered = [r for r in rows if r["n_processed_frames"]]
    best = None
    if rendered:
        # recording properties only: a genuinely static camera first, then the
        # most continuous bimanual coverage, then the shorter clip
        def key(r):
            static = 0 if r["camera_motion_status"] == "STATIC_CAMERA" else 1
            return (static, -r["n_frames_bimanual"], float(r["duration_sec"] or 1e9))
        best = sorted(rendered, key=key)[0]
        src = OUT_COMPARISON / f"{Path(best['source_video']).stem}_comparison.mp4"
        if src.exists():
            shutil.copy2(src, DEMO / "BEST_COMPARISON.mp4")
        write_json(DEMO / "metadata" / "best_comparison_choice.json", {
            "chosen": best["source_video"],
            "copied_to": "BEST_COMPARISON.mp4",
            "criteria_in_order": [
                "camera_motion_status == STATIC_CAMERA (this is a static-camera "
                "study)",
                "most frames with both hands detected (longest continuous "
                "bimanual coverage)",
                "shorter clip, so the whole thing can be watched",
            ],
            "explicitly_not_used": [
                "how large or small the focal estimate came out",
                "how favourable the depth change looks",
                "any ground truth - these clips have none",
            ],
            "values": {k: best[k] for k in
                       ("camera_motion_status", "n_frames_bimanual",
                        "duration_sec", "n_processed_frames")},
        })
    for r in rows:
        print(f"{r['source_video']:16s} {r['camera_motion_status']:14s} "
              f"pipeline={r['pipeline_focal_px']:>8} anycalib={r['anycalib_video_focal_px']:>8} "
              f"geocalib={r['geocalib_video_focal_px']:>8} e2={r['e2_video_focal_px']:>8} "
              f"frames={r['n_processed_frames']}")
    if best:
        print(f"\nBEST_COMPARISON.mp4 <- {best['source_video']}")


def _r(v):
    return round(float(v), 2) if isinstance(v, (int, float)) else ""


def _notes(pr, hands, render):
    bits = []
    if pr.get("camera_motion_status") not in ("STATIC_CAMERA", "MOSTLY_STATIC"):
        bits.append("OUT_OF_SCOPE_FOR_STATIC_AGGREGATION: camera moves, so a "
                    "single video-level focal is not the setting CAM-EXP-004 "
                    "studied; calibration frames were still recorded")
    if not hands:
        bits.append("hand inference not run for this clip")
    if render:
        bits.append(f"rendered with {render.get('codec')}")
    return "; ".join(bits)


if __name__ == "__main__":
    main()
