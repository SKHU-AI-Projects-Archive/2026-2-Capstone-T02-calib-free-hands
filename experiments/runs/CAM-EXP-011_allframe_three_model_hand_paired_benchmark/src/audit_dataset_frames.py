"""PHASE A — measure the real frame inventory. No number is assumed.

The expected historical counts (Tea 381, Boxing 367, Plant 174, Dog 337,
Instrument 139; 175 usable views) are treated as a hypothesis to check, not as
values to hard-code. Where the videos disagree, the videos win.

Three quantities are kept apart from the start, because conflating them is the
easiest way to mislead a reader:

    SEQUENCE_TOTAL_FRAMES   frames that exist in the video
    SCENE_INPUT_FRAMES      frames this benchmark feeds to the scene models
    HAND_AVAILABLE_FRAMES   frames where hand geometry is usable

Reference focal is not read here.
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from c011_common import (DISPLAY_SEQUENCE, MANIFESTS, PARTICIPANT_OF,  # noqa: E402
                    REPO, SUM, TAB, read_csv, usable_cameras, write_csv,
                    write_json)


def main():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import cv2
    from experiments.src.datasets.gigahands import takes

    cams = usable_cameras()
    take_by_name = {t.name: t for t in takes()}
    rows, t0 = [], time.time()

    for (seq, cam), info in sorted(cams.items()):
        take = take_by_name.get(seq)
        rec = {"sequence": seq, "display": DISPLAY_SEQUENCE.get(seq, seq),
               "participant": PARTICIPANT_OF.get(seq, ""), "camera": cam,
               "usable_for_camera_benchmark": int(info["usable"]),
               "rgb_video_available": info["rgb_video_available"],
               "video_segment_matches_annotation":
                   info["video_segment_matches_annotation"],
               "benchmark_note": info["note"]}
        vid = take.video_path(cam) if take else None
        rec["video_path"] = (str(Path(vid).relative_to(REPO))
                             if vid else "")
        n = 0
        if vid and Path(vid).exists():
            cap = cv2.VideoCapture(str(vid))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        rec["video_frame_count"] = n
        rows.append(rec)
        if len(rows) % 50 == 0:
            print("  %d/%d videos probed  %.1f s"
                  % (len(rows), len(cams), time.time() - t0), flush=True)

    write_csv(SUM / "video_frame_counts.csv", rows)

    # ---------------- hand availability, from the frozen CAM-EXP-001.3 QC
    qc = read_csv(MANIFESTS / "gigahands_demo_qc_v1.csv.gz")
    hand_ok = defaultdict(lambda: defaultdict(set))
    for r in qc:
        if r["qc_status"] in ("PASS_STRICT", "PASS_SINGLE_HAND") \
                and r["triangulation_success"] == "1":
            hand_ok[(r["sequence"], r["camera"])][r["hand"]].add(int(r["frame"]))

    # ---------------- per-sequence inventory
    inv = []
    for seq in sorted(set(r["sequence"] for r in rows)):
        srows = [r for r in rows if r["sequence"] == seq]
        us = [r for r in srows if r["usable_for_camera_benchmark"]]
        ex = [r for r in srows if not r["usable_for_camera_benchmark"]]
        counts = [r["video_frame_count"] for r in us if r["video_frame_count"]]
        total_per_video = int(np.median(counts)) if counts else 0
        varies = len(set(counts)) > 1
        hl = [len(hand_ok[(seq, r["camera"])].get("left", ())) for r in us]
        hr = [len(hand_ok[(seq, r["camera"])].get("right", ())) for r in us]
        inv.append({
            "sequence": seq, "display": DISPLAY_SEQUENCE.get(seq, seq),
            "participant": PARTICIPANT_OF.get(seq, ""),
            "sequence_total_frames_per_video": total_per_video,
            "frame_count_varies_across_cameras": int(varies),
            "frame_count_min": int(min(counts)) if counts else 0,
            "frame_count_max": int(max(counts)) if counts else 0,
            "benchmark_candidate_cameras": len(srows),
            "usable_rgb_cameras": len(us),
            "excluded_rgb_cameras": len(ex),
            "excluded_camera_reason": (ex[0]["benchmark_note"] if ex else ""),
            "total_rgb_frames_across_usable_cameras": int(sum(counts)),
            "scene_input_frames": int(sum(counts)),
            "hand_available_left_frames": int(sum(hl)),
            "hand_available_right_frames": int(sum(hr)),
            "median_hand_left_per_camera": float(np.median(hl)) if hl else 0.0,
            "median_hand_right_per_camera": float(np.median(hr)) if hr else 0.0,
            "notes": "scene_input_frames = every frame of every usable video; "
                     "hand counts are a SUBSET and are not the sequence length",
        })
    write_csv(TAB / "sequence_frame_inventory.csv", inv)

    total_scene = sum(r["scene_input_frames"] for r in inv)
    expected = {"p36-tea-0010": (381, 40), "p41-boxing-0021": (367, 40),
                "p41-plant-0004": (174, 40), "p44-dog-0004": (337, 40),
                "p52-instrument-0034": (139, 15)}
    checks = []
    for r in inv:
        ef, ec = expected.get(r["sequence"], (None, None))
        checks.append({
            "sequence": r["sequence"],
            "expected_frames_per_video": ef,
            "measured_frames_per_video": r["sequence_total_frames_per_video"],
            "frames_match": int(ef == r["sequence_total_frames_per_video"]),
            "expected_usable_cameras": ec,
            "measured_usable_cameras": r["usable_rgb_cameras"],
            "cameras_match": int(ec == r["usable_rgb_cameras"]),
        })
    write_csv(SUM / "expected_vs_measured.csv", checks)

    write_json(SUM / "dataset_frame_summary.json", {
        "sequences": len(inv),
        "usable_videos": sum(r["usable_rgb_cameras"] for r in inv),
        "excluded_videos": sum(r["excluded_rgb_cameras"] for r in inv),
        "total_scene_input_frames": total_scene,
        "expected_total_historical": 52445,
        "matches_expected_total": total_scene == 52445,
        "per_sequence": inv,
        "expected_vs_measured": checks,
        "reference_focal_read": False,
    })
    write_csv(SUM / "dataset_frame_summary.csv", inv)

    print("\n%-12s %8s %8s %8s %12s %12s"
          % ("sequence", "frames", "usable", "excl", "scene_frames", "handL/R"))
    for r in inv:
        print("%-12s %8d %8d %8d %12d %6d/%d"
              % (r["display"], r["sequence_total_frames_per_video"],
                 r["usable_rgb_cameras"], r["excluded_rgb_cameras"],
                 r["scene_input_frames"], r["hand_available_left_frames"],
                 r["hand_available_right_frames"]))
    print("\nTOTAL scene input frames: %d  (historical expectation 52,445 -> %s)"
          % (total_scene, "MATCH" if total_scene == 52445 else "DIFFERS"))
    for c in checks:
        if not (c["frames_match"] and c["cameras_match"]):
            print("  MISMATCH %s: frames %s vs %s, cameras %s vs %s"
                  % (c["sequence"], c["expected_frames_per_video"],
                     c["measured_frames_per_video"],
                     c["expected_usable_cameras"],
                     c["measured_usable_cameras"]))


if __name__ == "__main__":
    main()
