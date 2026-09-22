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

Two data properties, both established in CAM-EXP-001.2 (see that run's report):

* When a hand is not annotated in a view, all 21 joints are written as exactly
  ``(0, 0)`` *with confidence 1.0*, so a confidence filter alone does not
  remove them. ``drop_zero_2d`` marks those joints invalid. GigaHands does not
  document this value, so it is treated as an observed invalid pattern rather
  than a documented sentinel.
* ``rgb_vid/<cam>/`` may contain a recording whose filename timestamp differs
  from the annotated take (25 of 40 cameras in p52-instrument-0034). The video
  is therefore matched by exact filename stem and left unset otherwise, rather
  than pairing the annotation with a different segment.

Index conventions verified against all five demo sequences in CAM-EXP-001.2:
2D row index, 3D row index and RGB frame index are the same 0-based index;
``params`` and the official ``repro_*_vid`` montages instead use the position
within ``sorted(chosen_left | chosen_right)``.
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


def is_zero_2d(g2d: np.ndarray) -> np.ndarray:
    """Per-joint mask of the observed 'not annotated' pattern: exactly (0, 0).

    Confidence is reported as 1.0 for these rows, so they survive any
    confidence threshold and must be excluded explicitly.
    """
    g = np.asarray(g2d, dtype=float)
    return (g[:, 0] == 0.0) & (g[:, 1] == 0.0)


def sequences(root: Path | None = None) -> list[Path]:
    root = root or DATASETS_ROOT / "gigahands" / "demo_all" / "raw" / "hand_pose"
    return sorted(p for p in root.iterdir() if p.is_dir())


def iter_samples(
    root: Path | None = None,
    max_sequences: int | None = None,
    max_cameras: int | None = None,
    max_frames: int | None = None,
    conf_threshold: float = 0.5,
    drop_zero_2d: bool = True,
):
    """Yield Samples pairing GT 3D joints with per-view 2D detections.

    ``drop_zero_2d`` (default on) marks joints annotated as exactly (0, 0) as
    invalid. Pass False only to reproduce the pre-CAM-EXP-001.2 behaviour.
    """
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
                        zero2d = is_zero_2d(g)
                        valid = (g[:, 2] >= conf_threshold) & (conf3 > 0)
                        if drop_zero_2d:
                            valid &= ~zero2d
                        yield Sample(
                            dataset="gigahands", subset="demo_all",
                            sequence=f"{seq_dir.name}/{take}", camera_name=cam_name,
                            frame=str(fi), hand=hand, camera=cam,
                            joints3d_world=X[:, :3], joints2d_gt=g[:, :2], valid=valid,
                            image_path=vid if vid.exists() else None,
                            joints2d_source="per-view 2D detection (keypoints_2d)",
                            extra={"video_frame": fi, "conf": g[:, 2],
                                   "zero_2d_joints": int(zero2d.sum()),
                                   "zero_2d_view": bool(zero2d.all())},
                        )


class GigaHandsTake:
    """Random access to one GigaHands take, following the CAM-EXP-001.2 rules.

    M1  RGB frame i == timestamp line i == keypoints_2d row i == keypoints_3d row i
    M2  chosen_frames_<hand> = frame ids with a valid 3D pose for that hand
    M3  rows outside chosen_frames are all-zero placeholders - never use them
    M4  params / repro_*_vid / mano_vid are indexed by position within
        sorted(chosen_left | chosen_right), not by frame id
    M5  a 2D record whose 21 joints are all exactly (0,0) is an observed
        invalid pattern and is not a usable observation
    M6  rgb_vid may hold a different segment; the video is matched by exact
        filename stem and left unset otherwise

    ``joints3d`` deliberately sits behind an explicit call so that an
    experiment can keep dataset-provided 3D out of a reconstruction path.
    """

    def __init__(self, seq_dir: Path):
        self.dir = Path(seq_dir)
        self.name = self.dir.name
        self.take = sorted(p.name for p in (self.dir / "keypoints_3d").iterdir()
                           if p.is_dir())[0]
        self.k3dir = self.dir / "keypoints_3d" / self.take
        self.cameras = load_cameras(self.dir)
        self.chosen = {h: set(json.load(open(self.k3dir / f"chosen_frames_{h}.json")))
                       for h in HANDS}
        self.union_sorted = sorted(self.chosen["left"] | self.chosen["right"])
        self.intersect_sorted = sorted(self.chosen["left"] & self.chosen["right"])
        self._kp3: dict = {}
        self._kp2: dict = {}
        self._files2d: dict = {}
        for h in HANDS:
            d = self.dir / "keypoints_2d" / h / self.take
            self._files2d[h] = {"_".join(p.stem.split("_")[:2]): p
                                for p in d.glob("*.jsonl")} if d.is_dir() else {}

    # -- 2D (reconstruction input) ----------------------------------------
    def cameras_2d(self, hand: str = "left") -> list:
        return sorted(self._files2d[hand])

    def kp2(self, hand: str, camera: str) -> list:
        key = (hand, camera)
        if key not in self._kp2:
            p = self._files2d[hand].get(camera)
            self._kp2[key] = _read_jsonl(p) if p else []
        return self._kp2[key]

    def joints2d(self, hand: str, camera: str, frame: int):
        rows = self.kp2(hand, camera)
        if frame < 0 or frame >= len(rows):
            return None
        return np.asarray(rows[frame], dtype=float).reshape(-1, 3)

    def is_usable_2d(self, hand: str, camera: str, frame: int,
                     conf_threshold: float = 0.5, min_joints: int = 8) -> bool:
        g = self.joints2d(hand, camera, frame)
        if g is None or is_zero_2d(g).all():
            return False
        return int((g[:, 2] >= conf_threshold).sum()) >= min_joints

    # -- 3D (evaluation only) ---------------------------------------------
    def joints3d(self, hand: str, frame: int):
        """Dataset-provided 3D. Keep out of any reconstruction input."""
        if frame not in self.chosen[hand]:
            return None                      # M3: placeholder row
        if hand not in self._kp3:
            self._kp3[hand] = _read_jsonl(self.k3dir / f"{hand}.jsonl")
        rows = self._kp3[hand]
        if frame < 0 or frame >= len(rows):
            return None
        return np.asarray(rows[frame], dtype=float)[:, :3]

    # -- media -------------------------------------------------------------
    def video_path(self, camera: str):
        p = self._files2d["left"].get(camera) or self._files2d["right"].get(camera)
        if p is None:
            return None
        v = self.dir / "rgb_vid" / camera / f"{p.stem}.mp4"
        return v if v.exists() else None

    def union_index(self, frame: int) -> int:
        try:
            return self.union_sorted.index(frame)
        except ValueError:
            return -1


def takes(root: Path | None = None) -> list:
    """All demo takes as GigaHandsTake objects."""
    return [GigaHandsTake(d) for d in sequences(root)]
