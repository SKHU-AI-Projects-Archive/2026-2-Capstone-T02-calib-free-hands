"""HanCo loader (tester subset + full calibration/meta).

Camera storage (``calib/<seq>/<frame>.json``): per frame, ``K`` is a list of
8 intrinsic matrices and ``M`` a list of 8 4x4 world->camera matrices, one
per camera cam0..cam7. Images in the release are already undistorted, so no
distortion coefficients are provided.

3D joints (``xyz/<seq>/<frame>.json``): 21 joints in metres, expressed in the
world frame that ``M`` maps from (cam0 is the identity, i.e. world == cam0).

HanCo ships no 2D keypoint annotation, so ``joints2d_gt`` is None and only
geometric sanity plus multi-view consistency can be validated numerically.
RGB images *are* available, which makes HanCo the best overlay target.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..geometry import Camera
from .common import DATASETS_ROOT, Sample

N_CAMS = 8


def tester_root(root: Path | None = None) -> Path:
    return root or DATASETS_ROOT / "hanco" / "HanCo_tester"


def calib_meta_root(root: Path | None = None) -> Path:
    return root or DATASETS_ROOT / "hanco" / "HanCo_calib_meta"


def load_cameras(calib_json: Path, image_size=(224, 224)) -> list[Camera]:
    c = json.load(open(calib_json))
    cams = []
    for i in range(len(c["K"])):
        M = np.asarray(c["M"][i], dtype=float)
        cams.append(Camera(K=np.asarray(c["K"][i], dtype=float),
                           R=M[:3, :3], t=M[:3, 3],
                           width=image_size[0], height=image_size[1],
                           name=f"cam{i}"))
    return cams


def iter_samples(root: Path | None = None, max_sequences: int | None = None,
                 max_frames: int | None = None):
    base = tester_root(root)
    xyz_root = base / "xyz"
    if not xyz_root.is_dir():
        return
    for seq_dir in sorted(p for p in xyz_root.iterdir() if p.is_dir())[:max_sequences]:
        seq = seq_dir.name
        frames = sorted(seq_dir.glob("*.json"))
        if max_frames:
            step = max(1, len(frames) // max_frames)
            frames = frames[::step][:max_frames]
        for fp in frames:
            calib_p = base / "calib" / seq / fp.name
            if not calib_p.exists():
                continue
            X = np.asarray(json.load(open(fp)), dtype=float)
            for i, cam in enumerate(load_cameras(calib_p)):
                img = base / "rgb" / seq / f"cam{i}" / f"{fp.stem}.jpg"
                if img.exists():
                    cam.width, cam.height = _image_size(img)
                yield Sample(
                    dataset="hanco", subset="tester", sequence=seq,
                    camera_name=f"cam{i}", frame=fp.stem, hand="right",
                    camera=cam, joints3d_world=X, joints2d_gt=None,
                    image_path=img if img.exists() else None,
                    joints2d_source="none (HanCo ships no 2D annotation)",
                    extra={},
                )


_SIZE_CACHE: dict[Path, tuple[int, int]] = {}


def _image_size(path: Path) -> tuple[int, int]:
    if path not in _SIZE_CACHE:
        import cv2
        im = cv2.imread(str(path))
        _SIZE_CACHE[path] = (im.shape[1], im.shape[0]) if im is not None else (224, 224)
    return _SIZE_CACHE[path]
