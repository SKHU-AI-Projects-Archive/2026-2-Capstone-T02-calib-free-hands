"""PHASE A — cache the monocular WiLoR output once per frame.

Every method (M0-M3) consumes these SAME cached predictions, so any downstream
difference is attributable to the focal estimate and to nothing else. The
network is never re-run per focal condition.

Reference focal is not touched here.

Run with the anyhand environment from the repo root:
    PYOPENGL_PLATFORM=win32 experiments/.venv-anyhand/Scripts/python \
        experiments/runs/CAM-EXP-009_4_.../src/cache_wilor_predictions.py
"""
from __future__ import annotations

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "win32")

import argparse
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from common import CACHE, MANIFESTS, RAW, read_csv, write_csv  # noqa: E402

REPO = SRC.parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = CACHE / "wilor"


def load_predictor():
    from rgb_predictor import AnyHandPredictor
    return AnyHandPredictor(
        backend="wilor",
        wilor_ckpt=str(REPO / "models" / "anyhand_wilor.ckpt"),
        wilor_cfg=str(REPO / "models" / "model_config_wilor.yaml"),
        detector_pt=str(REPO / "models" / "detector.pt"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import cv2
    from experiments.src.datasets.gigahands import takes

    rows = read_csv(MANIFESTS / "cam_exp_0094_hand_frames_v1.csv.gz")
    want = defaultdict(set)
    for r in rows:
        want[(r["sequence"], r["camera"])].add(int(r["frame"]))
    take_by_name = {t.name: t for t in takes()}
    OUT.mkdir(parents=True, exist_ok=True)

    todo = []
    for (seq, cam), frames in sorted(want.items()):
        for f in sorted(frames):
            p = OUT / ("%s__%s__%06d.npz" % (seq, cam, f))
            if not p.exists():
                todo.append((seq, cam, f, p))
    if args.limit:
        todo = todo[:args.limit]
    print("frames to infer:", len(todo), flush=True)
    if not todo:
        return

    predictor = load_predictor()
    cap, cur = None, None
    index, failures = [], []
    t0 = time.time()
    for i, (seq, cam, fr, outp) in enumerate(todo):
        take = take_by_name.get(seq)
        vid = take.video_path(cam) if take else None
        if vid is None:
            failures.append({"sequence": seq, "camera": cam, "frame": fr,
                             "stage": "video", "reason": "no annotated video"})
            continue
        if cur != (seq, cam):
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(str(vid))
            cur = (seq, cam)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, img = cap.read()
        if not ok or img is None:
            failures.append({"sequence": seq, "camera": cam, "frame": fr,
                             "stage": "decode", "reason": "decode failed"})
            continue
        H, W = img.shape[:2]
        try:
            preds = predictor.predict(img)
        except Exception as exc:
            failures.append({"sequence": seq, "camera": cam, "frame": fr,
                             "stage": "inference",
                             "reason": "%s: %s" % (type(exc).__name__, exc)})
            traceback.print_exc()
            continue
        n_left = sum(1 for p in preds if not bool(p.is_right))
        n_right = sum(1 for p in preds if bool(p.is_right))
        data = {"image_width": np.int32(W), "image_height": np.int32(H),
                "n_hands": np.int32(len(preds))}
        for j, p in enumerate(preds):
            data["h%d_bbox" % j] = np.asarray(p.bbox, float).reshape(4)
            data["h%d_cam_t" % j] = np.asarray(p.cam_t, float)
            data["h%d_keypoints_3d" % j] = np.asarray(p.keypoints_3d, float)
            data["h%d_keypoints_2d" % j] = np.asarray(p.keypoints_2d, float)
            data["h%d_is_right" % j] = np.int32(int(p.is_right))
            data["h%d_score" % j] = np.float64(p.score)
            data["h%d_focal_length" % j] = np.float64(p.focal_length)
        np.savez_compressed(outp, **data)
        index.append({"sequence": seq, "camera": cam, "frame": fr,
                      "n_hands": len(preds), "n_left": n_left,
                      "n_right": n_right,
                      "multi_hand_ambiguity": int(n_left > 1 or n_right > 1),
                      "image_width": W, "image_height": H})
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print("  %d/%d  %.1f min  (%.2f s/frame)"
                  % (i + 1, len(todo), el / 60, el / (i + 1)), flush=True)
    if cap is not None:
        cap.release()

    prev = RAW / "wilor_cache_index.csv.gz"
    old = read_csv(prev) if prev.exists() else []
    write_csv(prev, old + index)
    write_csv(RAW / "wilor_failures.csv.gz", failures)
    print("cached %d, failures %d, %.1f min"
          % (len(index), len(failures), (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
