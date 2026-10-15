"""Run every detector on the same PNG list and return 14-format detection npz files (cached by frame hashes + settings)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from hand_focal.detectors import adapters as DA
from hand_focal.paths import resolve

HERE = Path(__file__).resolve().parent


class SkipDetector(RuntimeError):
    pass


def frames_key(frames, settings: dict, name: str) -> str:
    h = hashlib.sha256()
    for r in frames:
        h.update(f"{r['index']}:{r['sha256']};".encode())
    h.update(json.dumps({"detector": name, **settings}, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def write_list(frames, path: Path):
    path.write_text("".join(f"{r['index']}\t{r['png']}\t{r['sha256']}\n" for r in frames))


def _run(cmd, log):
    env = dict(os.environ, PYTHONNOUSERSITE="1")
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise SkipDetector(f"worker failed (exit {p.returncode}): {p.stderr.strip()[-600:]}")


def wilor_yolo(frames, s, cache_dir: Path, cfg16, log):
    dcfg = {"python": s["python"], "weights": s["weights"], "conf": s["conf"], "iou": s["iou"], "imgsz": s.get("imgsz"), "device": s["device"], "cache": True}
    key = DA.cache_key(frames, dcfg)
    src14 = resolve(cfg16["reuse_14"]["detection_cache"]) / f"detections_{key}.npz"
    dst = cache_dir / f"detections_{key}.npz"
    reused = False
    if not dst.exists() and src14.exists():
        shutil.copy2(src14, dst)                     # 14's cache is only read; the copy lives in 16's cache
        reused = True
    d = DA.wilor_yolo(frames, dcfg, cache_dir, log)
    return Path(d["cache_file"]), {"reused_14_cache": reused, "cache_key": key}


def mediapipe_hands(frames, s, cache_dir: Path, cfg16, log):
    settings = {k: s[k] for k in ("weights", "num_hands", "min_hand_detection_confidence", "min_hand_presence_confidence", "box_margin", "remap")}
    out = cache_dir / f"mediapipe_hands_{frames_key(frames, settings, 'mediapipe_hands')}.npz"
    if not out.exists():
        lst = cache_dir / (out.stem + ".tsv")
        write_list(frames, lst)
        cmd = [s["python"], str(HERE / "workers" / "mediapipe_hands_worker.py"), "--model", s["weights"], "--list", str(lst), "--out", str(out),
               "--num-hands", str(s["num_hands"]), "--min-detection", str(s["min_hand_detection_confidence"]),
               "--min-presence", str(s["min_hand_presence_confidence"]), "--box-margin", str(s["box_margin"])]
        if s.get("remap"):
            cmd += ["--remap", ",".join(map(str, s["remap"]))]
        _run(cmd, log)
        lst.unlink()
    return out, {}


def rtmpose_hand(frames, s, cache_dir: Path, cfg16, log, box_npz: Path):
    settings = {k: s[k] for k in ("weights", "input_size", "box_source", "box_scale", "remap", "device")}
    settings["boxes_sha256"] = hashlib.sha256(box_npz.read_bytes()).hexdigest()
    out = cache_dir / f"rtmpose_hand_{frames_key(frames, settings, 'rtmpose_hand')}.npz"
    if not out.exists():
        z = np.load(box_npz)
        bx = cache_dir / (out.stem + "_boxes.npz")
        np.savez(bx, offsets=z["offsets"], boxes=z["boxes"], box_conf=z["box_conf"], cls=z["cls"])      # boxes only, no keypoints
        lst = cache_dir / (out.stem + ".tsv")
        write_list(frames, lst)
        cmd = [s["python"], str(HERE / "workers" / "rtmpose_hand_worker.py"), "--onnx", s["weights"], "--list", str(lst), "--boxes", str(bx),
               "--out", str(out), "--input-size", ",".join(map(str, s["input_size"])), "--box-scale", str(s["box_scale"]), "--device", s["device"]]
        if s.get("remap"):
            cmd += ["--remap", ",".join(map(str, s["remap"]))]
        _run(cmd, log)
        lst.unlink()
        bx.unlink()
    return out, {"box_source": s["box_source"]}


def run_all(frames, cfg16, cache_dir: Path, log) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    order = list(cfg16["detectors"])
    if "rtmpose_hand" in order:                      # the box source must run first
        src = cfg16["detector_settings"]["rtmpose_hand"]["box_source"]
        src = {"mediapipe": "mediapipe_hands"}.get(src, src)
        order = [d for d in order if d == src] + [d for d in order if d != src]
        if src not in order:
            order.insert(0, src)
    for name in order:
        s = cfg16["detector_settings"][name]
        try:
            if name == "wilor_yolo":
                p, extra = wilor_yolo(frames, s, cache_dir, cfg16, log)
            elif name == "mediapipe_hands":
                p, extra = mediapipe_hands(frames, s, cache_dir, cfg16, log)
            elif name == "rtmpose_hand":
                src = {"mediapipe": "mediapipe_hands"}.get(s["box_source"], s["box_source"])
                if src not in out or "npz" not in out[src]:
                    raise SkipDetector(f"box source {src} unavailable")
                p, extra = rtmpose_hand(frames, s, cache_dir, cfg16, log, out[src]["npz"])
            else:
                raise SkipDetector(f"unknown detector {name}")
            d = DA._load(p)
            DA.check_alignment(frames, d)
            out[name] = {"npz": p, **extra, "info": d["info"]}
            log(f"  검출 {name}: 후보 {len(d['boxes'])}개 ({p.name})")
        except (SkipDetector, DA.DetectorError, FileNotFoundError) as e:
            out[name] = {"skipped": str(e)}
            log(f"  검출 {name}: 건너뜀 — {e}")
    return {k: out[k] for k in order}
