"""Draw the calibrator.py hand points on every frame of a session's hand videos.

    python overlay_points.py /public/data/handcalib_data/<session> [--out ../runs/overlay] [--width 960]

For each hand_*.mov/mp4 the rtmlib Wholebody model of calibrator.py (Calibrator("hand").model,
unchanged) is run on every frame and the right-hand points (indices 112-132) are drawn:
    green points + finger lines   frame used by calibrator.py (all 21 scores > 3)
    red points                    frame rejected by calibrator.py (some score <= 3)
    top-left text                 frame number, used/rejected, minimum score
Output: <out>/<session>/<video>_points.mp4 (H.264, no audio, no metadata). The session folder
and its videos are only read.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from calibrator import RIGHT_HAND, Calibrator
from session_rules import HAND_STEMS, find_video

BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]


def overlay(model, src: Path, dst: Path, width: int):
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tmp = dst.with_suffix(".tmp.mp4")
    writer, used, n = None, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        kps, scores = model(frame)
        pts, conf = kps[0, RIGHT_HAND], scores[0, RIGHT_HAND]
        good = conf.min() > 3                   # the same rule as Calibrator.detect
        used += int(good)
        s = width / frame.shape[1]
        img = cv2.resize(frame, (width, int(round(frame.shape[0] * s))), interpolation=cv2.INTER_AREA)
        p = (pts * s).astype(int)
        color = (0, 220, 0) if good else (0, 0, 255)
        if good:
            for a, b in BONES:
                cv2.line(img, tuple(p[a]), tuple(p[b]), (0, 180, 0), 1, cv2.LINE_AA)
        for q in p:
            cv2.circle(img, tuple(q), 3, color, -1, cv2.LINE_AA)
        txt = f"{n:4d}  {'used' if good else 'rejected'}  min score {conf.min():.2f}"
        cv2.putText(img, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        if writer is None:
            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, img.shape[1::-1])
        writer.write(img)
        n += 1
    cap.release()
    writer.release()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                    "-map_metadata", "-1", "-an", str(dst)], check=True)
    tmp.unlink()
    return used, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "runs" / "overlay"))
    ap.add_argument("--width", type=int, default=960)
    args = ap.parse_args()
    session = Path(args.session)
    out = Path(args.out) / session.name
    out.mkdir(parents=True, exist_ok=True)
    model = Calibrator("hand").model
    for stem in HAND_STEMS:
        src = find_video(session, stem)
        if src is None:
            continue
        dst = out / f"{stem}_points.mp4"
        used, n = overlay(model, src, dst, args.width)
        print(f"{dst}  used {used} / {n} frames", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
