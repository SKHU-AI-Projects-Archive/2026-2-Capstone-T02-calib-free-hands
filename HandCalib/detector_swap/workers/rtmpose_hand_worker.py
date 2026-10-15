"""Standalone RTMPose hand worker (run by /home/juson/venvs/det_rtmpose, rtmlib + onnxruntime). Top-down on given hand boxes.

    <python> rtmpose_hand_worker.py --onnx end2end.onnx --list frames.tsv --boxes boxes.npz --out det.npz [--box-scale 1.25]

boxes.npz: offsets (n + 1,), boxes (M, 4) xyxy, box_conf (M,), cls (M,) -- only the boxes of the box source (no keypoints are read).
rtmlib crops bbox_xyxy2cs(box, padding=1.25); to crop at box_scale the box is first scaled about its centre by box_scale / 1.25.
The model was trained on RGB input (deploy pipeline to_rgb: true), so the BGR PNG is converted to RGB before inference.
Point confidence: the SimCC maximum returned by rtmlib. Output: 14's detection npz with the source's box, box_conf and cls.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

RTMLIB_PADDING = 1.25


def scale_box(b, s):
    b = np.asarray(b, dtype=np.float64)
    c = (b[:2] + b[2:]) / 2
    h = (b[2:] - b[:2]) / 2 * s
    return np.r_[c - h, c + h]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--input-size", default="256,256")
    ap.add_argument("--box-scale", type=float, default=1.25)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--remap", default="")
    a = ap.parse_args()
    import cv2
    import onnxruntime
    from rtmlib import RTMPose
    rows = common.read_list(a.list)
    common.verify(rows)
    bx = np.load(a.boxes)
    if len(bx["offsets"]) != len(rows) + 1:
        raise RuntimeError("box file and frame list differ in length")
    size = tuple(int(x) for x in a.input_size.split(","))
    model = RTMPose(a.onnx, model_input_size=size, backend="onnxruntime", device=a.device)
    order = [int(x) for x in a.remap.split(",")] if a.remap else None
    per_frame = []
    for r, (i, png, _) in enumerate(rows):
        lo, hi = int(bx["offsets"][r]), int(bx["offsets"][r + 1])
        cands = []
        if hi > lo:
            img = cv2.cvtColor(cv2.imread(png, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            boxes = [scale_box(bx["boxes"][k], a.box_scale / RTMLIB_PADDING) for k in range(lo, hi)]
            kxy, sc = model(img, bboxes=boxes)
            kxy, sc = common.remap(np.asarray(kxy), np.asarray(sc), order)
            for j, k in enumerate(range(lo, hi)):
                cands.append({"box": bx["boxes"][k], "box_conf": float(bx["box_conf"][k]), "cls": int(bx["cls"][k]), "kxy": kxy[j], "kconf": sc[j]})
        per_frame.append(cands)
    from rtmlib.version import __version__ as rtmlib_version
    info = {"detector": "rtmpose_hand", "rtmlib": rtmlib_version, "onnxruntime": onnxruntime.__version__, "onnx": a.onnx,
            "input_size": list(size), "box_scale": a.box_scale, "rtmlib_padding": RTMLIB_PADDING, "color": "RGB", "device": a.device,
            "point_confidence": "SimCC maximum (rtmlib get_simcc_maximum)", "box_cls_from": "box source", "remap": order,
            "python": sys.version.split()[0], "names": {"0": "left", "1": "right"}}
    common.save(a.out, rows, per_frame, info)


if __name__ == "__main__":
    main()
