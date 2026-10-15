"""Standalone worker run by a separate Python that has ultralytics (MonoDETR's venv, read only).

Reads a frame list (index, png path, sha256), runs the YOLO pose hand detector on every PNG and writes one npz with every
hand candidate: box, box confidence, class (0 left, 1 right), 21 x [u, v, confidence]. Imports nothing from this repository.
    <python> wilor_yolo_worker.py --weights W --list frames.tsv --out det.npz --conf 0.3 --iou 0.3 [--imgsz 512] --device 0
"""
import argparse
import hashlib
import json
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=0)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
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


if __name__ == "__main__":
    main()
