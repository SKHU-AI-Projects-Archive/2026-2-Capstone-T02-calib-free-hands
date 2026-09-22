"""Shared helpers for CAM-EXP-001.2 (GigaHands frame/annotation mapping audit).

Mapping facts established in this run (see report.md for the evidence):

  M1  keypoints_2d/<hand>/<take>/<cam>.jsonl  row index == RGB video frame index
  M2  keypoints_3d/<take>/<hand>.jsonl        row index == RGB video frame index,
                                              but only rows 0..max(chosen) exist
  M3  params/<take>.json and repro_*_vid/mano_vid are indexed by POSITION in the
      sorted union of chosen_frames_left | chosen_frames_right
  M4  chosen_frames_<hand>.json lists the frame ids with a valid hand pose

M3 follows the official loader in GigaHands/render_mesh_video.py, which does
``[chosen_hand_union_frames.index(f) for f in chosen_video_frames]`` and uses
those positions to index the MANO parameters.
"""
from __future__ import annotations

import csv
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

RUN_DIR = Path(__file__).resolve().parents[1]
EXPERIMENTS = RUN_DIR.parents[1]
REPO = EXPERIMENTS.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.src.datasets import gigahands            # noqa: E402
from experiments.src.datasets.common import DATASETS_ROOT  # noqa: E402

HAND_POSE_ROOT = DATASETS_ROOT / "gigahands" / "demo_all" / "raw" / "hand_pose"
CAM_EXP_001 = EXPERIMENTS / "runs" / "CAM-EXP-001_gt_projection_validation"
CAM_EXP_0011 = EXPERIMENTS / "runs" / "CAM-EXP-001_1_gigahands_bad_view_diagnosis"

HANDS = ("left", "right")
# four-colour scheme used by every bimanual figure in this run
COLORS = {("left", "2d"): "#22c55e",    # green  - LEFT 2D annotation
          ("left", "3d"): "#ef4444",    # red    - LEFT 3D reprojection
          ("right", "2d"): "#3b82f6",   # blue   - RIGHT 2D annotation
          ("right", "3d"): "#f97316"}   # orange - RIGHT 3D reprojection
COLOR_LEGEND = ("green = LEFT 2D annotation   red = LEFT 3D reprojection   "
                "blue = RIGHT 2D annotation   orange = RIGHT 3D reprojection")


def rel(p) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return p.as_posix()


def write_csv(path: Path, rows: list, gzipped: bool = False) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    opener = ((lambda: __import__("gzip").open(path, "wt", newline="", encoding="utf-8"))
              if gzipped else (lambda: open(path, "w", newline="", encoding="utf-8")))
    with opener() as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"  wrote {rel(path)} ({len(rows)} rows)")


def read_csv(path: Path) -> list:
    import gzip
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sequences() -> list:
    return sorted(p for p in HAND_POSE_ROOT.iterdir() if p.is_dir())


def take_of(seq_dir: Path) -> str:
    return sorted(p.name for p in (seq_dir / "keypoints_3d").iterdir() if p.is_dir())[0]


class Sequence:
    """Everything needed to reason about one GigaHands take, loaded once."""

    def __init__(self, seq_dir: Path):
        self.dir = Path(seq_dir)
        self.name = self.dir.name
        self.take = take_of(self.dir)
        self.k3dir = self.dir / "keypoints_3d" / self.take
        self.cameras = gigahands.load_cameras(self.dir)
        self.chosen = {h: set(json.load(open(self.k3dir / f"chosen_frames_{h}.json")))
                       for h in HANDS}
        self.union_sorted = sorted(self.chosen["left"] | self.chosen["right"])
        self.intersect_sorted = sorted(self.chosen["left"] & self.chosen["right"])
        self._kp3 = {}
        self._kp2 = {}
        self._files2d = {}
        for h in HANDS:
            d = self.dir / "keypoints_2d" / h / self.take
            self._files2d[h] = {"_".join(p.stem.split("_")[:2]): p
                                for p in d.glob("*.jsonl")} if d.is_dir() else {}

    # -- 3D ---------------------------------------------------------------
    def kp3(self, hand: str) -> list:
        if hand not in self._kp3:
            self._kp3[hand] = gigahands._read_jsonl(self.k3dir / f"{hand}.jsonl")
        return self._kp3[hand]

    def joints3d(self, hand: str, frame: int):
        rows = self.kp3(hand)
        if frame < 0 or frame >= len(rows):
            return None
        return np.asarray(rows[frame], dtype=float)[:, :3]

    # -- 2D ---------------------------------------------------------------
    def cameras_2d(self, hand: str = "left") -> list:
        return sorted(self._files2d[hand])

    def kp2(self, hand: str, camera: str):
        key = (hand, camera)
        if key not in self._kp2:
            p = self._files2d[hand].get(camera)
            self._kp2[key] = gigahands._read_jsonl(p) if p else []
        return self._kp2[key]

    def joints2d(self, hand: str, camera: str, frame: int):
        rows = self.kp2(hand, camera)
        if frame < 0 or frame >= len(rows):
            return None
        return np.asarray(rows[frame], dtype=float).reshape(-1, 3)

    def video_path(self, camera: str) -> Path | None:
        p = self._files2d["left"].get(camera) or self._files2d["right"].get(camera)
        if p is None:
            return None
        v = self.dir / "rgb_vid" / camera / f"{p.stem}.mp4"
        return v if v.exists() else None

    def timestamps_path(self, camera: str) -> Path | None:
        v = self.video_path(camera)
        if v is None:
            return None
        t = v.with_suffix(".txt")
        return t if t.exists() else None

    # -- official union-position mapping (M3) ------------------------------
    def union_index(self, frame: int) -> int:
        """Position of a frame id inside the sorted chosen-frame union.

        This is the index the official renderer uses for MANO params and for
        the repro_2d_vid / repro_3d_vid / mano_vid montages.
        """
        try:
            return self.union_sorted.index(frame)
        except ValueError:
            return -1

    def repro_video(self, kind: str) -> Path | None:
        d = self.dir / f"repro_{kind}_vid"
        if not d.is_dir():
            return None
        f = sorted(d.glob("*.mp4"))
        return f[0] if f else None


def is_zero_pattern(g2d: np.ndarray) -> bool:
    """All 21 joints exactly at the origin - the observed 'no annotation' form."""
    return bool(g2d is not None and np.all(g2d[:, :2] == 0.0))


def project(cam, X, distortion: bool = True):
    uv, z = cam.project(X, apply_distortion=distortion)
    return uv, z


def median_err(cam, X, g2d, conf_thr: float = 0.5):
    """Median per-joint reprojection error, ignoring low-confidence joints."""
    if X is None or g2d is None:
        return float("nan"), 0
    uv, _ = project(cam, X)
    m = (g2d[:, 2] >= conf_thr) & np.isfinite(uv[:, 0])
    if m.sum() < 5:
        return float("nan"), int(m.sum())
    return float(np.median(np.linalg.norm(uv - g2d[:, :2], axis=1)[m])), int(m.sum())


def centroid_dist(cam, X, g2d, conf_thr: float = 0.5) -> float:
    """Distance between annotation centroid and projected centroid.

    Centroids rather than per-joint distances, because a left and a right hand
    skeleton use mirrored joint ordering: a genuine identity swap still leaves a
    large per-joint residual, which would mask the very effect we are testing.
    """
    if X is None or g2d is None:
        return float("nan")
    uv, _ = project(cam, X)
    m = (g2d[:, 2] >= conf_thr) & np.isfinite(uv[:, 0])
    if m.sum() < 5:
        return float("nan")
    return float(np.linalg.norm(g2d[m, :2].mean(0) - uv[m].mean(0)))


@lru_cache(maxsize=64)
def _cap(path: str):
    import cv2
    return cv2.VideoCapture(path)


def video_frame(path: Path, index: int):
    """Decode one frame as RGB. Captures are cached to avoid re-opening files."""
    import cv2
    if path is None:
        return None
    cap = _cap(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, im = cap.read()
    if not ok or im is None:
        return None
    return im[:, :, ::-1]


def repro_tile(seq: "Sequence", kind: str, frame_id: int, cam_index: int,
               n_cols: int = 7, n_rows: int = 6):
    """Crop one camera's tile out of an official repro montage.

    The montage carries exactly |union| frames, so the frame id must first be
    converted to its union position (M3).
    """
    v = seq.repro_video(kind)
    if v is None:
        return None
    ui = seq.union_index(frame_id)
    if ui < 0:
        return None
    im = video_frame(v, ui)
    if im is None:
        return None
    H, W = im.shape[:2]
    th, tw = H // n_rows, W // n_cols
    r, c = divmod(cam_index, n_cols)
    if r >= n_rows:
        return None
    return im[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
