"""InterHand2.6M (5 fps annotations) loader.

Camera storage (``*_camera.json``), official convention:
    X_cam = camrot @ (X_world - campos)        units: millimetres
    (u, v) = focal * X_cam[:2] / X_cam[2] + princpt,  no distortion
    (images in the release are already undistorted)

3D joints: ``*_joint_3d.json``[capture][frame_idx]['world_coord'], 42 joints
(right hand 0..20, left hand 21..41) in mm, with per-joint ``joint_valid``.

The release ships **no independent 2D keypoint annotation**; the official
2D is defined as the projection itself. The independent 2D signal available
for validation is therefore the annotated ``bbox`` per image, so this loader
exposes it via ``extra['bbox']`` and leaves ``joints2d_gt`` as None.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..geometry import Camera
from .common import DATASETS_ROOT, Sample

JOINTS_PER_HAND = 21
HAND_SLICES = {"right": slice(0, 21), "left": slice(21, 42)}


def annotations_root(root: Path | None = None) -> Path:
    return root or DATASETS_ROOT / "interhand26m" / "annotations"


def available_sets(root: Path | None = None) -> dict[str, list[str]]:
    """Map annotation set -> available splits, from what is on disk."""
    base = annotations_root(root)
    out: dict[str, list[str]] = {}
    for sub in sorted(p for p in base.iterdir() if p.is_dir()):
        splits = sorted({f.stem.split("_")[1] for f in sub.glob("InterHand2.6M_*_data.json")})
        out[sub.name] = splits
    return out


def iter_samples(
    annot_set: str = "human_annot",
    split: str = "test",
    root: Path | None = None,
    max_images: int | None = None,
):
    base = annotations_root(root) / annot_set
    cams = json.load(open(base / f"InterHand2.6M_{split}_camera.json"))
    data = json.load(open(base / f"InterHand2.6M_{split}_data.json"))
    j3d = json.load(open(base / f"InterHand2.6M_{split}_joint_3d.json"))
    ann_by_img = {a["image_id"]: a for a in data["annotations"]}

    images = data["images"]
    if max_images:
        step = max(1, len(images) // max_images)
        images = images[::step][:max_images]

    for img in images:
        cap, cid, fidx = img["capture"], img["camera"], img["frame_idx"]
        rec = j3d.get(cap, {}).get(fidx)
        if rec is None:
            continue
        c = cams[cap]
        cam = Camera.from_camera_pose(
            K=np.array([[c["focal"][cid][0], 0, c["princpt"][cid][0]],
                        [0, c["focal"][cid][1], c["princpt"][cid][1]],
                        [0, 0, 1]]),
            camrot=c["camrot"][cid], campos=c["campos"][cid],
            width=img["width"], height=img["height"], name=cid,
        )
        X = np.asarray(rec["world_coord"], dtype=float)
        jv = np.asarray(rec["joint_valid"]).reshape(-1).astype(bool)
        ann = ann_by_img.get(img["id"], {})
        av = np.asarray(ann.get("joint_valid", jv)).reshape(-1).astype(bool)
        valid_all = jv & av
        for hand, sl in HAND_SLICES.items():
            if not valid_all[sl].any():
                continue
            yield Sample(
                dataset="interhand26m", subset=f"{annot_set}/{split}",
                sequence=f"Capture{cap}/{img['seq_name']}", camera_name=cid,
                frame=str(fidx), hand=hand, camera=cam,
                joints3d_world=X[sl], joints2d_gt=None, valid=valid_all[sl],
                joints2d_source="none (official 2D is the projection itself)",
                notes="validated against annotated bbox",
                extra={"bbox": ann.get("bbox"), "hand_type": ann.get("hand_type"),
                       "file_name": img["file_name"]},
            )
