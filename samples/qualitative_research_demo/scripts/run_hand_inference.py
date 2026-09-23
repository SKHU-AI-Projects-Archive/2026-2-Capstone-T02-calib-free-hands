"""Run the existing hand pipeline ONCE per clip and cache everything per frame.

Reuses demo/demo_hand_mano.py verbatim — `load_predictor`, `stage1_detect`,
`stage2_fit_mano`, `process_frame` — so the detections, crops and MANO results
are exactly what the existing demo produces. Nothing about the camera focal is
decided here.

What is cached per hand is deliberately the focal-INDEPENDENT part plus the
focal that was used, so any other focal can be applied afterwards without
re-running the model:

    keypoints_3d   root-relative joints, metres      (focal-independent)
    cam_t          [tx, ty, tz] under the pipeline focal
    focal_length   the pipeline focal actually used

rgb_predictor._cam_crop_to_full defines
    tz = 2 f / (s * box_size),  tx = tx_crop + 2 (cx_box - cx_img) / (s * box_size)
so tx and ty carry no focal at all and tz is exactly proportional to it. That is
what makes the LEGACY vs CURRENT comparison an isolated focal intervention
rather than two separate inferences.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from common import CACHE, REPO, videos, write_json  # noqa: E402

# demo_hand_mano chdir()s to the service root and sets PYOPENGL_PLATFORM itself
sys.path.insert(0, str(REPO / "demo"))
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import demo_hand_mano as D  # noqa: E402
from hand_topology import HAND_BASE, JOINTS_PER_HAND, LEFT, RIGHT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="substring filter on the filename")
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--checkpoint", default="anyhand", choices=list(D.CHECKPOINTS))
    ap.add_argument("--det_conf", type=float, default=0.3)
    ap.add_argument("--det_iou", type=float, default=0.3)
    ap.add_argument("--rescale_factor", type=float, default=2.0)
    ap.add_argument("--tag", default="", help="suffix for the cache file")
    args = ap.parse_args()

    todo = [p for p in videos() if not args.only or args.only in p.name]
    if not todo:
        raise SystemExit("no matching sample video")

    t0 = time.time()
    print(f"[load] checkpoint={args.checkpoint}", flush=True)
    predictor = D.load_predictor(args.checkpoint, args.det_conf, args.det_iou,
                                 args.rescale_factor)
    print(f"[load] done in {time.time() - t0:.1f}s", flush=True)

    for path in todo:
        out = CACHE / f"{path.stem}{args.tag}_hands.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        t1 = time.time()
        records = []
        # per-hand focal-independent bookkeeping, kept beside the 42-point arrays
        kp3d_rel, focal_used = [], []
        for idx, frame in D.iter_frames(path, args.stride, args.max_frames):
            rec = D.empty_record(idx)
            rel = np.zeros((42, 3), np.float32)
            fl = np.zeros(2, np.float64)
            boxes, is_right, scores = D.stage1_detect(predictor, frame)
            hands = D.stage2_fit_mano(predictor, frame, boxes, is_right, scores)
            for hand in hands:
                side = RIGHT if hand.is_right else LEFT
                if rec["detected"][side] and rec["scores"][side] >= hand.score:
                    continue
                b = HAND_BASE[side]
                rel[b:b + JOINTS_PER_HAND] = hand.keypoints_3d
                fl[side] = float(hand.focal_length)
                rec["kp3d_abs"][b:b + JOINTS_PER_HAND] = (hand.keypoints_3d
                                                          + hand.cam_t[None, :])
                rec["kp2d"][b:b + JOINTS_PER_HAND] = hand.keypoints_2d
                rec["cam_t"][side] = hand.cam_t
                rec["mano_pose"][side] = hand.mano_pose
                rec["mano_shape"][side] = hand.mano_shape
                rec["bbox"][side] = hand.bbox
                rec["scores"][side] = hand.score
                rec["detected"][side] = True
            records.append(rec)
            kp3d_rel.append(rel)
            focal_used.append(fl)
            if len(records) % 60 == 0:
                print(f"  [{path.name}] {len(records)} frames "
                      f"({time.time() - t1:.0f}s)", flush=True)

        fps, w, h, n_total = D.iter_frames.meta
        np.savez_compressed(
            out,
            kp3d_rel=np.stack(kp3d_rel),
            kp3d_abs_legacy=np.stack([r["kp3d_abs"] for r in records]),
            kp2d=np.stack([r["kp2d"] for r in records]),
            cam_t_legacy=np.stack([r["cam_t"] for r in records]),
            focal_used_legacy=np.stack(focal_used),
            bbox=np.stack([r["bbox"] for r in records]),
            scores=np.stack([r["scores"] for r in records]),
            detected=np.stack([r["detected"] for r in records]),
            frame_indices=np.array([r["frame_idx"] for r in records]),
            fps=fps, width=w, height=h, n_frames_total=n_total,
            source_path=str(path.resolve()),
            checkpoint=args.checkpoint,
            det_conf=args.det_conf, det_iou=args.det_iou,
            rescale_factor=args.rescale_factor, stride=args.stride,
            coordinate_system="camera space: X image-right, Y image-down, "
                              "Z forward from the camera, metres, origin at the "
                              "camera. NOT a world or robot frame.",
            layout="42 = 0-20 left, 21-41 right; OpenPose 21 order",
            focal_note="cam_t_legacy and kp3d_abs_legacy use the pipeline focal "
                       "convention in focal_used_legacy. kp3d_rel is "
                       "focal-independent; tz scales linearly with the focal and "
                       "tx, ty do not depend on it at all "
                       "(rgb_predictor._cam_crop_to_full).",
        )
        n_det = int(sum(r["detected"].any() for r in records))
        write_json(CACHE / f"{path.stem}{args.tag}_hands.json", {
            "video": path.name, "frames_processed": len(records),
            "frames_with_any_hand": n_det,
            "frames_with_left": int(sum(r["detected"][LEFT] for r in records)),
            "frames_with_right": int(sum(r["detected"][RIGHT] for r in records)),
            "frames_bimanual": int(sum(bool(r["detected"][LEFT] and r["detected"][RIGHT])
                                       for r in records)),
            "pipeline_focal_px": float(np.max(np.stack(focal_used))),
            "checkpoint": args.checkpoint, "stride": args.stride,
            "runtime_sec": round(time.time() - t1, 1),
        })
        print(f"[{path.name}] {len(records)} frames, hands in {n_det} "
              f"-> {out.name}  ({time.time() - t1:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
