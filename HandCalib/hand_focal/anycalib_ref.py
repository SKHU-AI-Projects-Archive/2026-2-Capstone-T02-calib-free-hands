"""Separate AnyCalib f on the same PNG frames (01's model_id / cam_id / predict path). Never an input to the hand method.

predict() resizes / centre-crops internally and returns intrinsics already mapped back to the input image (reverse_scale_and_shift),
so f = (fx + fy) / 2 is in the normalised resolution of the PNGs. Offline: HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are set and
torch.hub downloads are blocked; the hub directory is listed before and after.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
from pathlib import Path

import numpy as np

from .paths import resolve


class DownloadBlocked(RuntimeError):
    pass


def hub_snapshot(hub_dir) -> dict:
    root = resolve(hub_dir)
    out = {}
    if root.exists():
        for p in root.rglob("*"):
            if p.is_file():
                out[str(p.relative_to(root))] = p.stat().st_size
    return out


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@contextlib.contextmanager
def offline(env: dict):
    """Offline environment variables and a torch.hub download guard (raises instead of downloading)."""
    import torch.hub
    old = {k: os.environ.get(k) for k in env}
    os.environ.update({k: str(v) for k, v in env.items()})
    orig = torch.hub.download_url_to_file

    def _blocked(url, *a, **k):
        raise DownloadBlocked(f"download blocked: {url}")
    torch.hub.download_url_to_file = _blocked
    try:
        yield
    finally:
        torch.hub.download_url_to_file = orig
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def available_models(acfg: dict) -> list:
    hub = resolve(acfg["torch_hub_dir"]) / "anycalib"
    ids = [acfg["model_id"]] + list(acfg.get("extra_models") or [])
    return [(m, hub / f"{m}.pt") for m in ids if (hub / f"{m}.pt").exists() or m == acfg["model_id"]]


def load_rgb(png):
    import cv2
    im = cv2.imread(str(png), cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"cannot read {png}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def predict_frames(model_id: str, cam_id: str, images, device: str, batch: int = 8) -> np.ndarray:
    """images: list of (H, W, 3) uint8 RGB arrays (same size). Returns (N, 4) fx, fy, cx, cy in the input resolution."""
    import torch
    from models.anycalib import AnyCalibAdapter        # 01's adapter (main, unchanged)
    ad = AnyCalibAdapter(model_id=model_id, cam_id=cam_id, device=device).build()
    out = []
    for i in range(0, len(images), batch):
        x = torch.from_numpy(np.stack(images[i:i + batch])).permute(0, 3, 1, 2).float() / 255.0
        p = ad.predict(x)
        intr = p["intrinsics"]
        intr = torch.stack([torch.as_tensor(v) for v in intr]) if isinstance(intr, (list, tuple)) else torch.as_tensor(intr)
        out.append(intr.reshape(-1, intr.shape[-1])[:, :4].detach().cpu().double().numpy())
    del ad
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return np.concatenate(out, 0)


def summarize(f: np.ndarray, times: np.ndarray, t_split: float) -> dict:
    f = np.asarray(f, float)
    ok = np.isfinite(f)
    g = f[ok]
    if not len(g):
        return {"n": 0}
    q10, q25, q50, q75, q90 = np.percentile(g, [10, 25, 50, 75, 90])
    t = np.asarray(times)[ok]
    return {"n": int(len(g)), "median": float(q50), "iqr": [float(q25), float(q75)], "p10_p90": [float(q10), float(q90)],
            "front_median": float(np.median(g[t < t_split])) if (t < t_split).any() else None,
            "tilt_median": float(np.median(g[t >= t_split])) if (t >= t_split).any() else None,
            "cv": float(np.std(g) / np.mean(g))}


def run(frames: list, acfg: dict, t_split: float, log=print) -> dict:
    import sys
    from .paths import PROJECT_ROOT
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    sel = frames[::max(1, int(acfg.get("stride", 1)))]
    before = hub_snapshot(acfg["torch_hub_dir"])
    images = [load_rgb(r["png"]) for r in sel]
    times = np.array([r["time_s"] for r in sel])
    res = {"frames": [int(r["index"]) for r in sel], "times": times.tolist(), "models": {}}
    with offline(acfg["offline_env"]):
        for m, wpath in available_models(acfg):
            if not wpath.exists():
                raise FileNotFoundError(f"AnyCalib weight missing: {wpath} (name {m}.pt, source github.com/javrtg/AnyCalib release v1.0.0)")
            intr = predict_frames(m, acfg["cam_id"], images, acfg["device"])
            f = (intr[:, 0] + intr[:, 1]) / 2
            res["models"][m] = {"weight": str(wpath), "weight_sha256": sha256_file(wpath), "fx": intr[:, 0].tolist(), "fy": intr[:, 1].tolist(),
                                "cx": intr[:, 2].tolist(), "cy": intr[:, 3].tolist(), "f": f.tolist(), "summary": summarize(f, times, t_split)}
            log(f"  AnyCalib {m}: 중앙값 f {res['models'][m]['summary']['median']:.1f} px ({len(f)}장)")
    skipped = [m for m in (acfg.get("extra_models") or []) if m not in res["models"]]
    after = hub_snapshot(acfg["torch_hub_dir"])
    new = sorted(set(after) - set(before))
    changed = sorted(k for k in before if k in after and after[k] != before[k])
    res["hub_check"] = {"dir": str(resolve(acfg["torch_hub_dir"])), "n_before": len(before), "n_after": len(after), "new_files": new, "changed_files": changed}
    res["skipped_extra_models"] = skipped
    if new or changed:
        raise RuntimeError(f"torch hub cache changed during AnyCalib: new {new}, changed {changed}")
    return res
