"""GigaHands demo_all loader.

Camera storage (``optim_params.txt``, COLMAP convention):
    cam_id width height fx fy cx cy k1 k2 p1 p2 cam_name qw qx qy qz tx ty tz
    X_cam = R(qvec) @ X_world + tvec,  distortion = OpenCV [k1, k2, p1, p2]

3D joints (``keypoints_3d/<take>/<hand>.jsonl``) are one JSON list per frame,
21 joints of [x, y, z, conf] in metres, row index == video frame index.
``chosen_frames_<hand>.json`` lists the frame indices with a valid 3D fit.

2D keypoints (``keypoints_2d/<hand>/<take>/<cam>_<ts>.jsonl``) are *per-view
detections* of 21 joints flattened as [x, y, conf] * 21 - not curated GT, so
occluded / mis-assigned views legitimately produce large outliers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..geometry import Camera, quat_to_rotmat
from .common import DATASETS_ROOT, Sample

HANDS = ("left", "right")


def _read_jsonl(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_cameras(seq_dir: Path) -> dict[str, Camera]:
    cams: dict[str, Camera] = {}
    for line in open(seq_dir / "optim_params.txt"):
        if line.startswith("#") or not line.strip():
            continue
        tok = line.split()
        w, h = int(tok[1]), int(tok[2])
        fx, fy, cx, cy = (float(v) for v in tok[3:7])
        dist = np.array([float(v) for v in tok[7:11]])
        name = tok[11]
        R = quat_to_rotmat([float(v) for v in tok[12:16]])
        t = np.array([float(v) for v in tok[16:19]])
        cams[name] = Camera(
            K=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
            R=R, t=t, dist=dist, width=w, height=h, name=name,
        )
    return cams


def sequences(root: Path | None = None) -> list[Path]:
    root = root or DATASETS_ROOT / "gigahands" / "demo_all" / "raw" / "hand_pose"
    return sorted(p for p in root.iterdir() if p.is_dir())


def iter_samples(
    root: Path | None = None,
    max_sequences: int | None = None,
    max_cameras: int | None = None,
    max_frames: int | None = None,
    conf_threshold: float = 0.5,
):
    """Yield Samples pairing GT 3D joints with per-view 2D detections."""
    for seq_dir in sequences(root)[:max_sequences]:
        cams = load_cameras(seq_dir)
        for take_dir in sorted((seq_dir / "keypoints_3d").iterdir()):
            take = take_dir.name
            for hand in HANDS:
                p3 = take_dir / f"{hand}.jsonl"
                chosen_p = take_dir / f"chosen_frames_{hand}.json"
                if not p3.exists():
                    continue
                kp3 = _read_jsonl(p3)
                chosen = json.load(open(chosen_p)) if chosen_p.exists() else range(len(kp3))
                frames = [f for f in sorted(set(chosen)) if f < len(kp3)]
                if max_frames:
                    step = max(1, len(frames) // max_frames)
                    frames = frames[::step][:max_frames]
                d2 = seq_dir / "keypoints_2d" / hand / take
                if not d2.is_dir():
                    continue
                for f2 in sorted(d2.glob("*.jsonl"))[:max_cameras]:
                    cam_name = "_".join(f2.stem.split("_")[:2])
                    cam = cams.get(cam_name)
                    if cam is None:
                        continue
                    kp2 = _read_jsonl(f2)
                    vid = (seq_dir / "rgb_vid" / cam_name / f"{f2.stem}.mp4")
                    for fi in frames:
                        if fi >= len(kp2):
                            continue
                        X = np.asarray(kp3[fi], dtype=float)
                        g = np.asarray(kp2[fi], dtype=float).reshape(-1, 3)
                        conf3 = X[:, 3] if X.shape[1] > 3 else np.ones(len(X))
                        valid = (g[:, 2] >= conf_threshold) & (conf3 > 0)
                        yield Sample(
                            dataset="gigahands", subset="demo_all",
                            sequence=f"{seq_dir.name}/{take}", camera_name=cam_name,
                            frame=str(fi), hand=hand, camera=cam,
                            joints3d_world=X[:, :3], joints2d_gt=g[:, :2], valid=valid,
                            image_path=vid if vid.exists() else None,
                            joints2d_source="per-view 2D detection (keypoints_2d)",
                            extra={"video_frame": fi, "conf": g[:, 2]},
                        )
