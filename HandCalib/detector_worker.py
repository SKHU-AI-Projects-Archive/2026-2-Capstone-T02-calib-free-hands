"""Standalone hand-keypoint detector workers (one file, three detectors). Each runs in its own virtual environment and imports nothing
from this repository; all write the same detection npz (frame_index, sha256, offsets, boxes, box_conf, cls, kxy, kconf, info).

    <MonoDETR venv python>         detector_worker.py wilor_yolo      --weights detector.pt --list frames.tsv --out det.npz [--conf --iou --imgsz --device]
    <det_mediapipe venv python>    detector_worker.py mediapipe_hands --model hand_landmarker.task --list frames.tsv --out det.npz [...]
    <det_rtmpose venv python>      detector_worker.py rtmpose_hand    --onnx end2end.onnx --list frames.tsv --boxes boxes.npz --out det.npz [...]

frames.tsv: index \t png \t sha256 per line (hashes are checked). Must stay Python 3.8 compatible (MonoDETR venv).

wilor_yolo:
    Standalone worker run by a separate Python that has ultralytics (MonoDETR's venv, read only).
    
    Reads a frame list (index, png path, sha256), runs the YOLO pose hand detector on every PNG and writes one npz with every
    hand candidate: box, box confidence, class (0 left, 1 right), 21 x [u, v, confidence]. Imports nothing from this repository.
        <python> wilor_yolo_worker.py --weights W --list frames.tsv --out det.npz --conf 0.3 --iou 0.3 [--imgsz 512] --device 0

mediapipe_hands:
    Standalone MediaPipe Hand Landmarker worker (run by /home/juson/venvs/det_mediapipe). IMAGE mode: every frame independently,
    no tracking or smoothing. Only the 21 normalised image landmarks (x, y) and handedness are used; z and world landmarks are ignored.
    
        <python> mediapipe_hands_worker.py --model hand_landmarker.task --list frames.tsv --out det.npz [--num-hands 2] [--box-margin 0.1]
    
    Output: 14's detection npz. Points: x * W, y * H (pixels of the PNG). The landmarker has no per-point confidence, so every point gets
    the hand's handedness score (the hand already passed min_hand_detection / presence confidence). Box: the landmarks' bounding box
    enlarged by box_margin x its size on every side (the landmarker does not return its crop box). cls: handedness label as returned.

rtmpose_hand:
    Standalone RTMPose hand worker (run by /home/juson/venvs/det_rtmpose, rtmlib + onnxruntime). Top-down on given hand boxes.
    
        <python> rtmpose_hand_worker.py --onnx end2end.onnx --list frames.tsv --boxes boxes.npz --out det.npz [--box-scale 1.25]
    
    boxes.npz: offsets (n + 1,), boxes (M, 4) xyxy, box_conf (M,), cls (M,) -- only the boxes of the box source (no keypoints are read).
    rtmlib crops bbox_xyxy2cs(box, padding=1.25); to crop at box_scale the box is first scaled about its centre by box_scale / 1.25.
    The model was trained on RGB input (deploy pipeline to_rgb: true), so the BGR PNG is converted to RGB before inference.
    Point confidence: the SimCC maximum returned by rtmlib. Output: 14's detection npz with the source's box, box_conf and cls.
"""
import argparse
import hashlib
import json
import sys

import numpy as np

# ---------------------------------------------------------------- shared helpers (numpy only)
def sha256(path) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def read_list(path):
    """14's frame list: index \\t png \\t sha256 per line."""
    rows = [line.rstrip("\n").split("\t") for line in open(path) if line.strip()]
    return [(int(i), p, h) for i, p, h in rows]


def verify(rows):
    for i, p, h in rows:
        if sha256(p) != h:
            raise RuntimeError(f"hash mismatch for frame {i}: {p}")


def to_pixels(xy_norm, width, height):
    """Normalised image coordinates (x / W, y / H, origin at the top-left image corner) -> pixels."""
    xy = np.asarray(xy_norm, dtype=np.float64)
    return np.stack([xy[..., 0] * width, xy[..., 1] * height], axis=-1)


def remap(kxy, kconf, order):
    if order is None:
        return kxy, kconf
    order = list(order)
    return kxy[:, order], kconf[:, order]


def save(out, rows, per_frame, info):
    """per_frame: list (one per row) of lists of dicts {box (4,), box_conf, cls, kxy (21, 2), kconf (21,)}.
    Writes 14's detection npz: frame_index, sha256, offsets, boxes, box_conf, cls, kxy, kconf, info."""
    offsets, boxes, bconf, cls, kxy, kconf = [0], [], [], [], [], []
    for cands in per_frame:
        for c in cands:
            boxes.append(c["box"])
            bconf.append(c["box_conf"])
            cls.append(c["cls"])
            kxy.append(c["kxy"])
            kconf.append(c["kconf"])
        offsets.append(offsets[-1] + len(cands))
    arr = lambda xs, shape, dt: np.asarray(xs, dtype=dt).reshape(shape) if xs else np.zeros((0,) + shape[1:], dtype=dt)   # noqa: E731
    np.savez_compressed(out, frame_index=np.array([r[0] for r in rows], dtype=np.int64), sha256=np.array([r[2] for r in rows]),
                        offsets=np.asarray(offsets, dtype=np.int64), boxes=arr(boxes, (-1, 4), np.float64),
                        box_conf=arr(bconf, (-1,), np.float64), cls=arr(cls, (-1,), np.int64), kxy=arr(kxy, (-1, 21, 2), np.float64),
                        kconf=arr(kconf, (-1, 21), np.float64), info=json.dumps(info))


# ---------------------------------------------------------------- wilor_yolo (ultralytics, MonoDETR venv)
def wilor_yolo(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=0)
    ap.add_argument("--device", default="0")
    a = ap.parse_args(argv)
    import ultralytics
    import torch
    from ultralytics import YOLO
    model = YOLO(a.weights)
    rows = [line.rstrip("\n").split("\t") for line in open(a.list) if line.strip()]
    idx, sha, offsets = [], [], [0]
    boxes, bconf, cls, kxy, kconf = [], [], [], [], []
    kw = {"conf": a.conf, "iou": a.iou, "device": a.device, "verbose": False}
    if a.imgsz:
        kw["imgsz"] = a.imgsz
    for i, path, h in rows:
        data = open(path, "rb").read()
        got = hashlib.sha256(data).hexdigest()
        if got != h:
            raise SystemExit(f"hash mismatch for frame {i}: {path}")
        r = model.predict(path, **kw)[0]
        n = 0 if r.boxes is None else len(r.boxes)
        if n:
            boxes.append(r.boxes.xyxy.cpu().numpy())
            bconf.append(r.boxes.conf.cpu().numpy())
            cls.append(r.boxes.cls.cpu().numpy().astype(int))
            kxy.append(r.keypoints.xy.cpu().numpy())
            kc = r.keypoints.conf
            kconf.append(kc.cpu().numpy() if kc is not None else np.ones(r.keypoints.xy.shape[:2]))
        offsets.append(offsets[-1] + n)
        idx.append(int(i))
        sha.append(got)
    cat = lambda xs, shape: np.concatenate(xs).astype(np.float64) if xs else np.zeros(shape)   # noqa: E731
    info = {"weights": a.weights, "ultralytics": ultralytics.__version__, "torch": torch.__version__, "cuda": torch.cuda.is_available(),
            "device": a.device, "conf": a.conf, "iou": a.iou, "imgsz": a.imgsz or model.overrides.get("imgsz"), "task": model.task,
            "names": {int(k): v for k, v in model.names.items()}, "kpt_shape": list(model.model.yaml.get("kpt_shape", [])), "python": sys.version.split()[0]}
    np.savez_compressed(a.out, frame_index=np.asarray(idx), sha256=np.asarray(sha), offsets=np.asarray(offsets), boxes=cat(boxes, (0, 4)),
                        box_conf=cat(bconf, (0,)), cls=np.concatenate(cls) if cls else np.zeros(0, int), kxy=cat(kxy, (0, 21, 2)), kconf=cat(kconf, (0, 21)),
                        info=json.dumps(info))


# ---------------------------------------------------------------- mediapipe_hands (det_mediapipe venv)
def mediapipe_hands(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-hands", type=int, default=2)
    ap.add_argument("--min-detection", type=float, default=0.5)
    ap.add_argument("--min-presence", type=float, default=0.5)
    ap.add_argument("--box-margin", type=float, default=0.1)
    ap.add_argument("--remap", default="")
    a = ap.parse_args(argv)
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    rows = read_list(a.list)
    verify(rows)
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
                xy = to_pixels([[p.x, p.y] for p in lms], W, H)
                score = float(hd[0].score)
                k, kc = remap(xy[None], np.full((1, 21), score), order)
                lo, hi = k[0].min(0), k[0].max(0)
                pad = a.box_margin * (hi - lo)
                cands.append({"box": [lo[0] - pad[0], lo[1] - pad[1], hi[0] + pad[0], hi[1] + pad[1]], "box_conf": score,
                              "cls": 0 if hd[0].category_name.lower() == "left" else 1, "kxy": k[0], "kconf": kc[0]})
            per_frame.append(cands)
    info = {"detector": "mediapipe_hands", "mediapipe": mp.__version__, "model_path": a.model, "running_mode": "IMAGE", "num_hands": a.num_hands,
            "min_hand_detection_confidence": a.min_detection, "min_hand_presence_confidence": a.min_presence, "box": f"landmark bbox + {a.box_margin} x size",
            "point_confidence": "hand handedness score copied to all 21 points", "cls": "handedness category_name (as returned; mirroring not applied)",
            "remap": order, "python": sys.version.split()[0], "names": {"0": "left", "1": "right"}}
    save(a.out, rows, per_frame, info)


# ---------------------------------------------------------------- rtmpose_hand (det_rtmpose venv)
RTMLIB_PADDING = 1.25


def scale_box(b, s):
    b = np.asarray(b, dtype=np.float64)
    c = (b[:2] + b[2:]) / 2
    h = (b[2:] - b[:2]) / 2 * s
    return np.r_[c - h, c + h]


def rtmpose_hand(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--input-size", default="256,256")
    ap.add_argument("--box-scale", type=float, default=1.25)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--remap", default="")
    a = ap.parse_args(argv)
    import cv2
    import onnxruntime
    from rtmlib import RTMPose
    rows = read_list(a.list)
    verify(rows)
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
            kxy, sc = remap(np.asarray(kxy), np.asarray(sc), order)
            for j, k in enumerate(range(lo, hi)):
                cands.append({"box": bx["boxes"][k], "box_conf": float(bx["box_conf"][k]), "cls": int(bx["cls"][k]), "kxy": kxy[j], "kconf": sc[j]})
        per_frame.append(cands)
    from rtmlib.version import __version__ as rtmlib_version
    info = {"detector": "rtmpose_hand", "rtmlib": rtmlib_version, "onnxruntime": onnxruntime.__version__, "onnx": a.onnx,
            "input_size": list(size), "box_scale": a.box_scale, "rtmlib_padding": RTMLIB_PADDING, "color": "RGB", "device": a.device,
            "point_confidence": "SimCC maximum (rtmlib get_simcc_maximum)", "box_cls_from": "box source", "remap": order,
            "python": sys.version.split()[0], "names": {"0": "left", "1": "right"}}
    save(a.out, rows, per_frame, info)


WORKERS = {"wilor_yolo": wilor_yolo, "mediapipe_hands": mediapipe_hands, "rtmpose_hand": rtmpose_hand}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in WORKERS:
        raise SystemExit("usage: detector_worker.py {wilor_yolo,mediapipe_hands,rtmpose_hand} ...")
    WORKERS[sys.argv[1]](sys.argv[2:])
