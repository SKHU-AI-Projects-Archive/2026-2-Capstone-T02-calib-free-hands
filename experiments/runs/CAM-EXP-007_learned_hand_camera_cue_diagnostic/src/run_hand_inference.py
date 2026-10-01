"""Run the frozen hand model over the frozen frame grid and cache raw outputs.

TARGET-BLIND. This file never imports the target, the reference focal or any
AnyCalib prediction.

Three read-only taps, none of which modifies model source:
  * forward-pre-hook on the model      -> batch box_center, box_size, right
  * forward hook on model.backbone     -> vit_out latent (pooled)
  * forward hook on model.refine_net   -> final pred_cam (crop space)

pred_cam is stored as the model emits it, in CROP space. It is never converted
to a metric translation, because that conversion multiplies in the pipeline
focal and would re-inject the camera focal into the predictor.

Run with the hand-model environment:
  experiments/.venv-anyhand/Scripts/python.exe src/run_hand_inference.py
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, MANIFESTS, RAW, REPO, read_csv, write_csv, write_json  # noqa: E402

FRAMES = MANIFESTS / "cam_exp_007_frame_manifest_v1.csv.gz"
OUT_NPZ = CACHE / "hand_outputs"
INDEX = RAW / "hand_inference_index.csv.gz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO))
    import cv2
    from rgb_predictor import AnyHandPredictor
    from experiments.src.datasets.gigahands import takes

    rows = read_csv(FRAMES)
    by_view = defaultdict(list)
    for r in rows:
        by_view[(r["sequence"], r["camera"])].append(int(r["frame"]))
    views = sorted(by_view)
    if args.count:
        views = views[args.start:args.start + args.count]

    pred = AnyHandPredictor(backend="wilor")
    # force the lazy loader to build the model
    _ = pred.predict(np.zeros((256, 256, 3), np.uint8))
    model = pred._wilor_model
    if model is None:
        raise RuntimeError("WiLoR model not loaded")

    cap = defaultdict(list)

    def pre_hook(_m, inputs):
        b = inputs[0]
        for k in ("box_center", "box_size", "right"):
            if k in b:
                cap[k].append(b[k].detach().float().cpu().numpy())

    def backbone_hook(_m, _i, out):
        vit = out[3] if isinstance(out, (tuple, list)) else out
        cap["latent"].append(
            vit.detach().float().mean(dim=(-2, -1)).cpu().numpy())

    def refine_hook(_m, _i, out):
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            cap["pred_cam"].append(out[1].detach().float().cpu().numpy())

    hs = [model.register_forward_pre_hook(pre_hook),
          model.backbone.register_forward_hook(backbone_hook)]
    if hasattr(model, "refine_net"):
        hs.append(model.refine_net.register_forward_hook(refine_hook))
    else:
        print("WARNING: refine_net absent; pred_cam tap unavailable")

    take_by_name = {t.name: t for t in takes()}
    OUT_NPZ.mkdir(parents=True, exist_ok=True)
    index = []

    for vi, (seq, cam) in enumerate(views, 1):
        take = take_by_name[seq]
        want = sorted(by_view[(seq, cam)])
        vc = cv2.VideoCapture(str(take.video_path(cam)))
        wanted, store = set(want), {}
        i, hi = 0, max(want)
        while i <= hi:
            ok, frame = vc.read()
            if not ok:
                break
            if i in wanted:
                cap.clear()
                try:
                    hands = pred.predict(frame)
                    if isinstance(hands, dict):
                        hands = hands.get("wilor", [])
                    status = "ok" if hands else "no_hand"
                except Exception as e:                      # noqa: BLE001
                    hands, status = [], f"model_failure:{type(e).__name__}"
                H, W = frame.shape[:2]
                got = {k: (np.concatenate(v, axis=0) if v else None)
                       for k, v in cap.items()}
                for k, hd in enumerate(hands):
                    p = f"{i}_{k}"
                    store[p + "_bbox"] = np.asarray(hd.bbox, np.float32)
                    store[p + "_kp3d_rootrel"] = np.asarray(
                        hd.keypoints_3d, np.float32)
                    store[p + "_mano_pose"] = np.asarray(
                        hd.mano_pose, np.float32)
                    store[p + "_mano_shape"] = np.asarray(
                        hd.mano_shape, np.float32)
                    store[p + "_is_right"] = np.float32(hd.is_right)
                    store[p + "_score"] = np.float32(hd.score)
                    for key in ("pred_cam", "box_center", "box_size",
                                "latent"):
                        arr = got.get(key)
                        if arr is not None and k < len(arr):
                            store[p + "_" + key] = np.asarray(
                                arr[k], np.float32)
                index.append({
                    "sequence": seq, "camera": cam, "frame": i,
                    "status": status, "n_hands": len(hands),
                    "image_width": W, "image_height": H,
                    "has_pred_cam": int(got.get("pred_cam") is not None),
                    "latent_dim": int(got["latent"].shape[1])
                    if got.get("latent") is not None else 0,
                })
            i += 1
        vc.release()
        np.savez_compressed(OUT_NPZ / f"{seq}__{cam}.npz", **store)
        if args.selftest:
            break
        if vi % 10 == 0:
            print(f"  {vi}/{len(views)} views", flush=True)

    for h in hs:
        h.remove()

    if args.selftest:
        k0 = sorted(store)[:14]
        print("stored keys sample:", k0)
        print("index sample:", index[:2])
        return

    write_csv(INDEX, index)
    nz = sum(1 for r in index if r["status"] == "ok")
    write_json(RAW.parent / "summary" / "inference_meta.json", {
        "frames_attempted": len(index), "frames_with_hand": nz,
        "views": len(views),
        "latent_dim": max((r["latent_dim"] for r in index), default=0),
        "frames_with_pred_cam": sum(r["has_pred_cam"] for r in index),
        "target_blind": True,
    })
    print(f"frames {len(index)}, with hand {nz} "
          f"({nz / max(len(index), 1):.1%})")


if __name__ == "__main__":
    main()
