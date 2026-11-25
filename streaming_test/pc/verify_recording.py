"""Verify frame timing of a recorded video using ffprobe.

Usage: python verify_recording.py [config.yaml]
Settings (input_path, nominal_fps, gap_threshold, ...) come from the YAML config.
Requires: pyyaml, and ffprobe (ffmpeg) installed.
"""

import argparse
import csv
import statistics
import subprocess
import sys
from pathlib import Path

import yaml

REQUIRED_KEYS = (
    "input_path", "nominal_fps", "gap_threshold",
    "ffprobe_path", "video_stream", "output_csv", "max_gaps_printed",
)


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        sys.exit(f"config {path}: missing keys: {', '.join(missing)}")
    return cfg


def read_frame_pts(cfg):
    """Presentation timestamps (seconds) of all video frames, sorted ascending."""
    cmd = [
        cfg["ffprobe_path"], "-v", "error",
        "-select_streams", str(cfg["video_stream"]),
        "-show_entries", "frame=best_effort_timestamp_time",
        "-of", "csv=p=0",
        str(cfg["input_path"]),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        sys.exit(f"ffprobe not found: {cfg['ffprobe_path']}")
    except subprocess.CalledProcessError as e:
        sys.exit(f"ffprobe failed:\n{e.stderr}")
    pts = []
    for line in proc.stdout.splitlines():
        value = line.strip().rstrip(",")
        if value and value != "N/A":
            pts.append(float(value))
    # ffprobe lists frames in decode order; B-frames make pts non-monotonic there.
    pts.sort()
    return pts


def analyze(pts, nominal_fps, gap_threshold):
    nominal = 1.0 / nominal_fps
    intervals = [b - a for a, b in zip(pts, pts[1:])]
    gap_limit = gap_threshold * nominal
    rows = []
    gaps = []
    for i, dt in enumerate(intervals, start=1):
        is_gap = dt > gap_limit
        dropped = max(1, round(dt / nominal) - 1) if is_gap else 0
        rows.append((i, pts[i], dt, int(is_gap), dropped))
        if is_gap:
            gaps.append((i, pts[i - 1], dt, dropped))
    duration = pts[-1] - pts[0]
    stats = {
        "frames": len(pts),
        "duration_s": duration,
        "mean_fps": (len(pts) - 1) / duration if duration > 0 else float("nan"),
        "interval_mean_ms": statistics.fmean(intervals) * 1000,
        "interval_std_ms": statistics.pstdev(intervals) * 1000,
        "interval_min_ms": min(intervals) * 1000,
        "interval_max_ms": max(intervals) * 1000,
        "gap_limit_ms": gap_limit * 1000,
        "gaps": len(gaps),
        "dropped_est": sum(g[3] for g in gaps),
        "expected_frames": round(duration / nominal) + 1,
    }
    return rows, gaps, stats


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "pts_s", "interval_s", "is_gap", "dropped_est"])
        w.writerows(rows)


def print_summary(cfg, gaps, s):
    print(f"input            : {cfg['input_path']}")
    print(f"frames           : {s['frames']} (expected ~{s['expected_frames']} at {cfg['nominal_fps']} fps)")
    print(f"duration         : {s['duration_s']:.3f} s (first to last frame pts)")
    print(f"mean fps         : {s['mean_fps']:.3f} (nominal {cfg['nominal_fps']})")
    print(f"interval mean    : {s['interval_mean_ms']:.3f} ms")
    print(f"interval std     : {s['interval_std_ms']:.3f} ms")
    print(f"interval min/max : {s['interval_min_ms']:.3f} / {s['interval_max_ms']:.3f} ms")
    print(f"gaps (> {cfg['gap_threshold']} x nominal = {s['gap_limit_ms']:.3f} ms): "
          f"{s['gaps']}, estimated dropped frames: {s['dropped_est']}")
    for idx, t, dt, dropped in gaps[: cfg["max_gaps_printed"]]:
        print(f"  frame {idx}: after t={t:.3f} s, interval {dt * 1000:.1f} ms, ~{dropped} dropped")
    if len(gaps) > cfg["max_gaps_printed"]:
        print(f"  ... {len(gaps) - cfg['max_gaps_printed']} more (see CSV)")
    print(f"intervals CSV    : {cfg['output_csv']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", nargs="?",
                        default=str(Path(__file__).with_name("verify_config.yaml")),
                        help="YAML config path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pts = read_frame_pts(cfg)
    if len(pts) < 2:
        sys.exit(f"need at least 2 frames, got {len(pts)}")
    rows, gaps, stats = analyze(pts, cfg["nominal_fps"], cfg["gap_threshold"])
    write_csv(cfg["output_csv"], rows)
    print_summary(cfg, gaps, stats)


if __name__ == "__main__":
    main()
