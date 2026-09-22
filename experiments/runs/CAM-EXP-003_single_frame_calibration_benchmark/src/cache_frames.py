"""Decode the benchmark frames once into a lossless PNG cache.

Two reasons this exists rather than decoding inside each model run:
  * random seeking into 175 videos dominated the runtime and had to be repeated
    for every model;
  * caching guarantees every model sees bit-identical pixels, which is the
    fairness requirement of this benchmark. PNG is used so nothing is re-encoded
    lossily.

The cache is git-ignored and regenerable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

from common import RUN_DIR, load_frames, rel

CACHE = RUN_DIR / "cache" / "frames"


def frame_path(r) -> Path:
    return CACHE / f"{r['sequence']}__{r['camera']}__{int(r['frame']):06d}.png"


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    frames = load_frames()
    from collections import defaultdict
    by_video = defaultdict(list)
    for r in frames:
        by_video[r["video"]].append(r)
    repo = Path(__file__).resolve().parents[4]
    n_new = n_have = 0
    for video, rows in by_video.items():
        todo = [r for r in rows if not frame_path(r).exists()]
        n_have += len(rows) - len(todo)
        if not todo:
            continue
        cap = cv2.VideoCapture(str(repo / video))
        for r in sorted(todo, key=lambda x: int(x["frame"])):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["frame"]))
            ok, img = cap.read()
            if ok and img is not None:
                cv2.imwrite(str(frame_path(r)), img,
                            [cv2.IMWRITE_PNG_COMPRESSION, 1])
                n_new += 1
        cap.release()
        print(f"  {n_new + n_have}/{len(frames)}", flush=True)
    print(f"cached {n_new} new, {n_have} already present -> {rel(CACHE)}")


if __name__ == "__main__":
    main()
