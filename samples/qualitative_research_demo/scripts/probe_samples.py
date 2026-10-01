"""Probe the sample clips: format, and whether the camera is static.

Static-camera aggregation is what CAM-EXP-004 studied, so clips whose camera
moves are recorded as out of scope for the video-level calibration estimate
rather than deleted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEMO, META, videos, write_csv, write_json  # noqa: E402

N_PROBE = 25          # sampled frames used for the motion diagnostic
STATIC_PX = 1.0       # median background shift below this -> static
MOSTLY_PX = 4.0       # ... below this -> mostly static


def fourcc_str(cap) -> str:
    fc = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((fc >> 8 * i) & 0xFF) for i in range(4)).strip()


def motion_diagnostic(path: Path, n_frames: int):
    """Median background feature displacement between uniformly sampled frames.

    Sparse Lucas-Kanade on good features over the whole frame. Hands move in
    every clip, so a small median with a larger p90 means a fixed camera with
    moving content; a large median means the camera itself moved.
    """
    cap = cv2.VideoCapture(str(path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    idxs = np.unique(np.linspace(0, max(n_frames - 1, 0), N_PROBE).astype(int))
    prev, shifts = None, []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        small = cv2.resize(frame, (480, max(int(480 * h / max(w, 1)), 1)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            pts = cv2.goodFeaturesToTrack(prev, 300, 0.01, 8)
            if pts is not None:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None)
                if st is not None and int(st.sum()) > 20:
                    d = (nxt[st == 1] - pts[st == 1]).reshape(-1, 2)
                    shifts.append(float(np.median(np.linalg.norm(d, axis=1))))
        prev = gray
    cap.release()
    if not shifts:
        return float("nan"), float("nan"), "UNCERTAIN"
    med = float(np.median(shifts))
    p90 = float(np.percentile(shifts, 90))
    if med < STATIC_PX:
        status = "STATIC_CAMERA" if p90 < 3 * MOSTLY_PX else "MOSTLY_STATIC"
    elif med < MOSTLY_PX:
        status = "MOSTLY_STATIC"
    else:
        status = "MOVING_CAMERA"
    return med, p90, status


def main() -> None:
    rows = []
    for p in videos():
        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = fourcc_str(cap)
        cap.release()
        med, p90, status = motion_diagnostic(p, n)
        row = {
            "video": p.name, "width": w, "height": h, "fps": round(fps, 3),
            "frames": n, "duration_sec": round(n / max(fps, 1e-9), 2),
            "codec": codec,
            "median_bg_shift_px_between_sampled_frames": round(med, 2),
            "p90_bg_shift_px": round(p90, 2),
            "camera_motion_status": status,
            "scope": ("primary target for the static-camera demo"
                      if status in ("STATIC_CAMERA", "MOSTLY_STATIC")
                      else "OUT_OF_SCOPE_FOR_STATIC_AGGREGATION"),
        }
        rows.append(row)
        write_json(META / f"{p.stem}.json", row)
        print(f"{p.name:18s} {w}x{h} {fps:.2f}fps {n:5d}f {codec:5s} "
              f"median_shift={med:6.2f}px -> {status}")
    write_csv(DEMO / "sample_probe.csv", rows)
    print(f"\nwrote {DEMO / 'sample_probe.csv'}")


if __name__ == "__main__":
    main()
