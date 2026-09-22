"""Run the existing AnyHand/WiLoR pipeline ONCE per frame and cache everything.

This is the only step that needs torch, and it runs in the separate
``experiments/.venv-anyhand`` environment so the analysis environment (and the
CAM-EXP-001.x results produced with it) is left untouched.

Caching is what makes the focal sweep clean: the detector, the crop and the
network regression all happen once, so any later difference between focal
conditions is attributable to the focal alone and to nothing else.

    PYTHONPATH=<repo root> .venv-anyhand/Scripts/python src/run_inference.py \
        --n 1000 [--limit N] [--no-resume]
"""
from __future__ import annotations

import os

# WiLoR's renderer defaults PYOPENGL_PLATFORM to 'egl', which has no library on
# Windows and kills the import. Rendering is not used here (the predictor loads
# with init_renderer=False), so the platform is pinned before any import.
os.environ.setdefault("PYOPENGL_PLATFORM", "win32")

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from common import (CACHE_DIR, RUN_DIR, SEED, pipeline_focal, rel,  # noqa: E402
                    select_frames, write_csv)

REPO = SRC.parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def frame_key(seq: str, cam: str, frame: str) -> str:
    return f"{seq}__{cam}__{int(frame):06d}"


def load_predictor():
    from rgb_predictor import AnyHandPredictor
    return AnyHandPredictor(
        backend="wilor",
        wilor_ckpt=str(REPO / "models" / "anyhand_wilor.ckpt"),
        wilor_cfg=str(REPO / "models" / "model_config_wilor.yaml"),
        detector_pt=str(REPO / "models" / "detector.pt"),
    )


def video_for(take, camera: str):
    return take.video_path(camera)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="target number of frames")
    ap.add_argument("--limit", type=int, default=0, help="stop after N frames (smoke)")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    import cv2
    from experiments.src.datasets.gigahands import takes

    # model_config_wilor.yaml refers to './mano_data/...' relative to the repo
    # root, so the pipeline is run from there exactly as the demo would be.
    os.chdir(REPO)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    take_by_name = {t.name: t for t in takes()}

    frames = select_frames(args.n, seed=SEED)
    if args.limit:
        frames = frames[:args.limit]
    print(f"selected {len(frames)} frames")

    predictor = load_predictor()
    index_rows, failures = [], []
    # group by (sequence, camera) so each video is opened once
    frames.sort(key=lambda r: (r["sequence"], r["camera"], int(r["frame"])))

    cap = None
    cur_key = None
    t0 = time.time()
    for i, r in enumerate(frames):
        seq, cam, fr = r["sequence"], r["camera"], int(r["frame"])
        key = frame_key(seq, cam, fr)
        out_npz = CACHE_DIR / f"{key}.npz"
        if not args.no_resume and out_npz.exists():
            index_rows.append({"key": key, "sequence": seq, "camera": cam,
                               "frame": fr, "status": "cached"})
            continue
        take = take_by_name.get(seq)
        vid = video_for(take, cam) if take else None
        if vid is None:
            failures.append({"sequence": seq, "camera": cam, "frame": fr,
                             "stage": "video", "reason": "annotated RGB segment not available"})
            continue
        if cur_key != (seq, cam):
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(str(vid))
            cur_key = (seq, cam)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, img_bgr = cap.read()
        if not ok or img_bgr is None:
            failures.append({"sequence": seq, "camera": cam, "frame": fr,
                             "stage": "decode", "reason": "frame decode failed"})
            continue
        H, W = img_bgr.shape[:2]
        try:
            preds = predictor.predict(img_bgr)
        except Exception as exc:
            failures.append({"sequence": seq, "camera": cam, "frame": fr,
                             "stage": "inference", "reason": f"{type(exc).__name__}: {exc}"})
            traceback.print_exc()
            continue
        if not preds:
            failures.append({"sequence": seq, "camera": cam, "frame": fr,
                             "stage": "detector", "reason": "no hand detected"})
            index_rows.append({"key": key, "sequence": seq, "camera": cam,
                               "frame": fr, "status": "no_detection", "n_hands": 0,
                               "image_width": W, "image_height": H})
            continue

        # Everything needed to redo the focal-dependent part analytically.
        data = {
            "image_width": np.int32(W), "image_height": np.int32(H),
            "pipeline_focal": np.float64(pipeline_focal(W, H)),
            "n_hands": np.int32(len(preds)),
        }
        for j, p in enumerate(preds):
            box = np.asarray(p.bbox, dtype=np.float64).reshape(4)
            # ViTDetDataset geometry: square crop around the padded box
            # (rgb_predictor.py:414-419, rescale_factor default 2.0)
            cx = 0.5 * (box[0] + box[2])
            cy = 0.5 * (box[1] + box[3])
            data[f"h{j}_bbox"] = box
            data[f"h{j}_box_center"] = np.array([cx, cy])
            data[f"h{j}_cam_t"] = np.asarray(p.cam_t, dtype=np.float64)
            data[f"h{j}_keypoints_3d"] = np.asarray(p.keypoints_3d, dtype=np.float64)
            data[f"h{j}_keypoints_2d"] = np.asarray(p.keypoints_2d, dtype=np.float64)
            data[f"h{j}_is_right"] = np.int32(int(p.is_right))
            data[f"h{j}_score"] = np.float64(p.score)
            data[f"h{j}_focal_length"] = np.float64(p.focal_length)
            data[f"h{j}_mano_shape"] = np.asarray(p.mano_shape, dtype=np.float64)
        np.savez_compressed(out_npz, **data)
        index_rows.append({"key": key, "sequence": seq, "camera": cam, "frame": fr,
                           "status": "ok", "n_hands": len(preds),
                           "image_width": W, "image_height": H,
                           "pipeline_focal": pipeline_focal(W, H),
                           "cache_file": rel(out_npz)})
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(frames)} frames ({el:.1f}s, {el / (i + 1):.2f}s/frame)",
                  flush=True)
    if cap is not None:
        cap.release()

    write_csv(RUN_DIR / "results" / "raw" / "inference_cache_index.csv", index_rows)
    write_csv(RUN_DIR / "results" / "raw" / "model_failures.csv", failures)
    (RUN_DIR / "results" / "summary" / "_inference_meta.json").write_text(
        json.dumps({"n_selected": len(frames), "n_cached": len(index_rows),
                    "n_failures": len(failures), "seed": SEED,
                    "target_n": args.n}, indent=2), encoding="utf-8")
    print(f"done: {len(index_rows)} cached, {len(failures)} failures")


if __name__ == "__main__":
    main()
