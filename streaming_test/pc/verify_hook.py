"""MediaMTX runOnRecordSegmentComplete hook: verify one finished recording.

Usage: python verify_hook.py <segment_path> [--config verify_config.yaml]
Loads verify_config.yaml, replaces input_path with the segment and output_csv with a CSV next
to the segment, then runs the same analysis as verify_recording.py (which is imported unchanged).
"""

import argparse
import sys
from pathlib import Path

import verify_recording as vr


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("segment", help="path of the finished recording")
    parser.add_argument("--config", default=str(Path(__file__).with_name("verify_config.yaml")),
                        help="YAML config path (input_path and output_csv are overridden)")
    args = parser.parse_args()

    segment = Path(args.segment)
    cfg = vr.load_config(args.config)
    cfg["input_path"] = str(segment)
    cfg["output_csv"] = str(segment.with_suffix(".intervals.csv"))

    print(f"[verify] {segment}", flush=True)
    pts = vr.read_frame_pts(cfg)
    if len(pts) < 2:
        sys.exit(f"[verify] need at least 2 frames, got {len(pts)}")
    rows, gaps, stats = vr.analyze(pts, cfg["nominal_fps"], cfg["gap_threshold"])
    vr.write_csv(cfg["output_csv"], rows)
    vr.print_summary(cfg, gaps, stats)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
