"""AssemblyHands loader (annotation-only download: demo / test splits).

Camera storage (``*_calib_*.json``):
    calibration[seq]['intrinsics'][cam]            3x3 K
    calibration[seq]['extrinsics'][frame][cam]     3x4 [R | t], world -> camera
    X_cam = R @ X_world + t                        units: millimetres
Images in the release are ``ego_images_rectified``, i.e. already undistorted.

Camera naming differs between files: ``images[].camera`` is e.g.
``HMC_84358933`` while the calibration key is ``HMC_84358933_mono10bit``,
so the loader resolves by prefix instead of assuming either spelling.

3D joints (``*_joint_3d_*.json``)['annotations'][seq][frame]['world_coord'],
21 joints in mm with ``joint_valid``.

Unlike InterHand2.6M, the ``annotations[].keypoints`` field holds a per-view
2D annotation [u, v, conf], so a true reprojection error can be computed even
though no RGB images were downloaded.

Caveat found during preparation: in the ``test-eccv2024`` split every
``world_coord`` and every ``keypoints`` entry is the constant 1.0 - it is the
held-out HANDS2024 benchmark split whose GT is withheld for submission. The
loader therefore refuses degenerate records instead of emitting numbers that
would look like a huge reprojection error.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..geometry import Camera
from .common import DATASETS_ROOT, Sample


def annotations_root(root: Path | None = None) -> Path:
    return root or DATASETS_ROOT / "assemblyhands" / "annotations"


def discover_splits(root: Path | None = None) -> dict[str, dict[str, str]]:
    """Find (calib, data, joint_3d) triples on disk, keyed by split dir.

    File names carry version suffixes that change between releases, so the
    triple is discovered by role keyword rather than hard-coded names.
    """
    base = annotations_root(root)
    out: dict[str, dict[str, str]] = {}
    for sub in sorted(p for p in base.iterdir() if p.is_dir()):
        found: dict[str, str] = {}
        for f in sorted(sub.glob("*.json")):
            n = f.name.lower()
            if "template" in n or "pred" in n:
                continue
            for role, keys in (("calib", ("calib",)), ("joint_3d", ("joint_3d", "joint3d")),
                               ("data", ("data",))):
                if any(k in n for k in keys) and role not in found:
                    found[role] = str(f)
        if {"calib", "data", "joint_3d"} <= set(found):
            out[sub.name] = found
    return out


def _resolve(d: dict, cam: str):
    if cam in d:
        return d[cam]
    for k in d:
        if k.startswith(cam) or cam.startswith(k.split("_mono")[0]):
            return d[k]
    return None


def _is_real_gt(X: np.ndarray) -> bool:
    """Reject placeholder GT: all-constant world coordinates carry no pose."""
    return len(np.unique(np.round(X, 6), axis=0)) > 1


def iter_samples(split: str = "demo", root: Path | None = None,
                 max_images: int | None = None):
    splits = discover_splits(root)
    if split not in splits:
        return
    paths = splits[split]
    calib = json.load(open(paths["calib"]))["calibration"]
    data = json.load(open(paths["data"]))
    j3d = json.load(open(paths["joint_3d"]))
    j3d = j3d.get("annotations", j3d)
    ann_by_img = {a["image_id"]: a for a in data["annotations"]}

    images = data["images"]
    if max_images:
        step = max(1, len(images) // max_images)
        images = images[::step][:max_images]

    for img in images:
        seq, cam_name = img["seq_name"], img["camera"]
        frame = f"{int(img['frame_idx']):06d}"
        rec = j3d.get(seq, {}).get(frame)
        cal = calib.get(seq)
        ann = ann_by_img.get(img["id"])
        if rec is None or cal is None or ann is None:
            continue
        K = _resolve(cal["intrinsics"], cam_name)
        ext_frame = cal["extrinsics"].get(frame)
        if K is None or ext_frame is None:
            continue
        M = _resolve(ext_frame, cam_name)
        if M is None:
            continue
        M = np.asarray(M, dtype=float)
        cam = Camera(K=np.asarray(K, dtype=float), R=M[:3, :3], t=M[:3, 3],
                     width=img["width"], height=img["height"], name=cam_name)
        X = np.asarray(rec["world_coord"], dtype=float)
        if not _is_real_gt(X):
            continue  # withheld/placeholder GT (e.g. the test-eccv2024 split)
        jv = np.asarray(rec.get("joint_valid", np.ones(len(X)))).reshape(-1).astype(bool)
        kp = ann.get("keypoints")
        # 42 joints = right(0..20) + left(21..41); single-hand records give 21.
        hands = {"right": slice(0, 21), "left": slice(21, 42)} if len(X) >= 42 else {"right": slice(0, len(X))}
        kp_arr = np.asarray(kp, dtype=float).reshape(-1, 3) if kp else None
        for hand, sl in hands.items():
            v = jv[sl]
            if not v.any():
                continue
            g = None
            if kp_arr is not None and len(kp_arr) >= sl.stop:
                g = kp_arr[sl][:, :2]
                v = v & (kp_arr[sl][:, 2] > 0)
            yield Sample(
                dataset="assemblyhands", subset=split, sequence=seq,
                camera_name=cam_name, frame=frame, hand=hand, camera=cam,
                joints3d_world=X[sl], joints2d_gt=g, valid=v,
                joints2d_source="annotation keypoints [u,v,conf]" if g is not None else "none",
                notes="annotation-only download; no RGB images on disk",
                extra={"bbox": ann.get("bbox"), "file_name": img["file_name"]},
            )
