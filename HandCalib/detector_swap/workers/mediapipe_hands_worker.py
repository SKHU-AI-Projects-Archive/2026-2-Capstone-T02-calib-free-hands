"""Standalone MediaPipe Hand Landmarker worker (run by /home/juson/venvs/det_mediapipe). IMAGE mode: every frame independently,
no tracking or smoothing. Only the 21 normalised image landmarks (x, y) and handedness are used; z and world landmarks are ignored.

    <python> mediapipe_hands_worker.py --model hand_landmarker.task --list frames.tsv --out det.npz [--num-hands 2] [--box-margin 0.1]

Output: 14's detection npz. Points: x * W, y * H (pixels of the PNG). The landmarker has no per-point confidence, so every point gets
the hand's handedness score (the hand already passed min_hand_detection / presence confidence). Box: the landmarks' bounding box
enlarged by box_margin x its size on every side (the landmarker does not return its crop box). cls: handedness label as returned.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-hands", type=int, default=2)
    ap.add_argument("--min-detection", type=float, default=0.5)
    ap.add_argument("--min-presence", type=float, default=0.5)
    ap.add_argument("--box-margin", type=float, default=0.1)
    ap.add_argument("--remap", default="")
    a = ap.parse_args()
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    rows = common.read_list(a.list)
    common.verify(rows)
    opts = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=a.model, delegate=BaseOptions.Delegate.CPU),
                                        running_mode=vision.RunningMode.IMAGE, num_hands=a.num_hands,
                                        min_hand_detection_confidence=a.min_detection, min_hand_presence_confidence=a.min_presence)
    order = [int(x) for x in a.remap.split(",")] if a.remap else None
    per_frame = []
    with vision.HandLandmarker.create_from_options(opts) as lm:
        for i, png, _ in rows:
            img = mp.Image.create_from_file(png)
            W, H = img.width, img.height
            res = lm.detect(img)
            cands = []
            for lms, hd in zip(res.hand_landmarks, res.handedness):
                xy = common.to_pixels([[p.x, p.y] for p in lms], W, H)
                score = float(hd[0].score)
                k, kc = common.remap(xy[None], np.full((1, 21), score), order)
                lo, hi = k[0].min(0), k[0].max(0)
                pad = a.box_margin * (hi - lo)
                cands.append({"box": [lo[0] - pad[0], lo[1] - pad[1], hi[0] + pad[0], hi[1] + pad[1]], "box_conf": score,
                              "cls": 0 if hd[0].category_name.lower() == "left" else 1, "kxy": k[0], "kconf": kc[0]})
            per_frame.append(cands)
    info = {"detector": "mediapipe_hands", "mediapipe": mp.__version__, "model_path": a.model, "running_mode": "IMAGE", "num_hands": a.num_hands,
            "min_hand_detection_confidence": a.min_detection, "min_hand_presence_confidence": a.min_presence, "box": f"landmark bbox + {a.box_margin} x size",
            "point_confidence": "hand handedness score copied to all 21 points", "cls": "handedness category_name (as returned; mirroring not applied)",
            "remap": order, "python": sys.version.split()[0], "names": {"0": "left", "1": "right"}}
    common.save(a.out, rows, per_frame, info)


if __name__ == "__main__":
    main()
