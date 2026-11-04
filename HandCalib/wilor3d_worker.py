"""Experiment 71 worker: WiLoR (AnyHand checkpoint, /public/hand-demo's predictor, read-only) 3-D hand per frame.

Runs in /public/hand-demo/.venv (torch + WiLoR); the main code calls it as a subprocess, one video per call, on one GPU (CUDA_VISIBLE_DEVICES).
Input: a TSV of frame index and PNG path. Output (npz): per frame the highest-score hand's keypoints_3d (21, 3) and vertices (778, 3) in camera metres (left hands
mirrored to the right-hand frame by the predictor), is_right, score, bbox; frames without a hand get NaN.
GPU thermal guard (user rule): before every frame the GPU temperature is read; at >= pause_c the worker waits until it is below resume_c.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def gpu_temp(dev: int) -> int:
    out = subprocess.run(["nvidia-smi", "-i", str(dev), "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(out.stdout.strip().splitlines()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, required=True, help="physical GPU index (for the temperature)")
    ap.add_argument("--pause_c", type=int, default=70)
    ap.add_argument("--resume_c", type=int, default=65)
    a = ap.parse_args()
    sys.path.insert(0, a.repo)
    import cv2
    from rgb_predictor import AnyHandPredictor
    pred = AnyHandPredictor(backend="wilor", device="cuda")
    rows = [ln.split("\t") for ln in Path(a.list).read_text().splitlines() if ln.strip()]
    n = len(rows)
    kp3 = np.full((n, 21, 3), np.nan, np.float32)
    ver = np.full((n, 778, 3), np.nan, np.float32)
    right = np.zeros(n, bool)
    score = np.full(n, np.nan, np.float32)
    bbox = np.full((n, 4), np.nan, np.float32)
    waited = 0.0
    for k, (idx, png) in enumerate(rows):
        if gpu_temp(a.gpu) >= a.pause_c:
            t0 = time.time()
            while gpu_temp(a.gpu) >= a.resume_c:
                time.sleep(5)
            waited += time.time() - t0
        hands = pred.predict(cv2.imread(png.strip()))
        if not hands:
            continue
        h = max(hands, key=lambda x: x.score)
        kp3[k], ver[k], right[k], score[k], bbox[k] = h.keypoints_3d, h.vertices, bool(h.is_right), float(h.score), h.bbox
    np.savez_compressed(a.out, frame_index=np.array([int(r[0]) for r in rows]), keypoints_3d=kp3, vertices=ver, is_right=right, score=score, bbox=bbox,
                        thermal_wait_s=np.array(waited))


if __name__ == "__main__":
    main()
