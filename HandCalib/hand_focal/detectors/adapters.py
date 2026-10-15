"""Detector adapters. Both return the same dict: frame_index, offsets, boxes, box_conf, cls, kxy, kconf, info."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


class DetectorError(RuntimeError):
    pass


def _load(npz: Path) -> dict:
    z = np.load(npz, allow_pickle=False)
    d = {k: z[k] for k in z.files if k != "info"}
    d["info"] = json.loads(str(z["info"])) if "info" in z.files else {}
    return d


def cache_key(frames: list, dcfg: dict) -> str:
    h = hashlib.sha256()
    for r in frames:
        h.update(f"{r['index']}:{r['sha256']};".encode())
    h.update(json.dumps({k: dcfg.get(k) for k in ("weights", "conf", "iou", "imgsz")}, sort_keys=True).encode())
    return h.hexdigest()[:16]


def wilor_yolo(frames: list, dcfg: dict, cache_dir: Path, log=print) -> dict:
    """Run the worker in the configured (read-only) venv; cache by frame hashes + settings."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key(frames, dcfg)
    out = cache_dir / f"detections_{key}.npz"
    if dcfg.get("cache", True) and out.exists():
        log(f"  detection cache hit: {out.name}")
        d = _load(out)
    else:
        lst = cache_dir / f"frames_{key}.tsv"
        lst.write_text("".join(f"{r['index']}\t{r['png']}\t{r['sha256']}\n" for r in frames))
        cmd = [dcfg["python"], str(HERE / "wilor_yolo_worker.py"), "--weights", dcfg["weights"], "--list", str(lst), "--out", str(out),
               "--conf", str(dcfg["conf"]), "--iou", str(dcfg["iou"]), "--device", str(dcfg["device"])]
        if dcfg.get("imgsz"):
            cmd += ["--imgsz", str(dcfg["imgsz"])]
        env = dict(os.environ, PYTHONNOUSERSITE="1")
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if p.returncode != 0 or not out.exists():
            raise DetectorError(f"YOLO worker failed (exit {p.returncode}): {p.stderr.strip()[-800:]}")
        lst.unlink()
        d = _load(out)
    check_alignment(frames, d)
    d["cache_file"] = str(out)
    return d


def mock(frames: list, dcfg: dict, cache_dir: Path, log=print) -> dict:
    d = _load(Path(dcfg["mock_npz"]))
    if "sha256" not in d:
        d["sha256"] = np.array([r["sha256"] for r in frames])
    check_alignment(frames, d)
    d["cache_file"] = str(dcfg["mock_npz"])
    return d


def check_alignment(frames: list, d: dict) -> None:
    """The worker's frame numbers and PNG hashes must equal the frame list of the main code."""
    idx = [int(x) for x in d["frame_index"]]
    if idx != [r["index"] for r in frames]:
        raise DetectorError("detector frame indices differ from the frame list")
    if [str(x) for x in d["sha256"]] != [r["sha256"] for r in frames]:
        raise DetectorError("detector PNG hashes differ from the frame list")


def run(frames, dcfg, cache_dir, log=print) -> dict:
    if dcfg["name"] == "wilor_yolo":
        return wilor_yolo(frames, dcfg, cache_dir, log)
    if dcfg["name"] == "mock":
        return mock(frames, dcfg, cache_dir, log)
    raise DetectorError(f"unknown detector {dcfg['name']}")
