from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CanonicalResult:
    frame_indices: np.ndarray
    timestamps: np.ndarray
    keypoints_2d: np.ndarray
    keypoints_3d: np.ndarray
    detected: np.ndarray
    focal_px: np.ndarray
    image_size: tuple[int, int]
    mano_vertices: np.ndarray | None = None
    mano_faces: np.ndarray | None = None
    confidence: np.ndarray | None = None
    bbox: np.ndarray | None = None
    mano_pose: np.ndarray | None = None
    mano_shape: np.ndarray | None = None
    camera_translation: np.ndarray | None = None
    schema_version: int = 1
    video_focal_px: float | None = None
    media_metadata: dict | None = None

    def __post_init__(self) -> None:
        frames = len(self.frame_indices)
        if self.timestamps.shape != (frames,):
            raise ValueError("timestamps must have one value per frame")
        if self.detected.shape[0] != frames or self.keypoints_2d.shape[0] != frames or self.keypoints_3d.shape[0] != frames:
            raise ValueError("result arrays must agree on frame count")
        if self.focal_px.shape != (frames,):
            raise ValueError("focal_px must have one value per frame")
        if self.mano_faces is not None and (self.mano_faces.ndim not in (2, 3) or self.mano_faces.shape[-1] != 3):
            raise ValueError("mano_faces must have shape (F, 3) or (2, F, 3)")
        validate_result(self)


def validate_result(result: CanonicalResult) -> None:
    """Reject ambiguous missing-value encodings at the canonical boundary."""
    for frame, hand in np.argwhere(~result.detected):
        if not np.isnan(result.keypoints_2d[frame, hand]).all() or not np.isnan(result.keypoints_3d[frame, hand]).all():
            raise ValueError("undetected hands must use NaN coordinates")
    for frame, hand in np.argwhere(result.detected):
        if np.isnan(result.keypoints_2d[frame, hand]).all() or np.isnan(result.keypoints_3d[frame, hand]).all():
            raise ValueError("detected hands must contain coordinates")


def save_result(result: CanonicalResult, output_dir: Path, provenance: dict) -> tuple[Path, Path]:
    validate_result(result)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "frame_indices": result.frame_indices,
        "timestamps": result.timestamps,
        "keypoints_2d": result.keypoints_2d,
        "keypoints_3d": result.keypoints_3d,
        "detected": result.detected,
        "focal_px": result.focal_px,
        "image_size": np.asarray(result.image_size, dtype=np.int32),
        "schema_version": np.asarray(result.schema_version, dtype=np.int32),
    }
    if result.mano_vertices is not None:
        arrays["mano_vertices"] = result.mano_vertices
    if result.mano_faces is not None:
        arrays["mano_faces"] = result.mano_faces.astype(np.int32, copy=False)
    for name in ("confidence", "bbox", "mano_pose", "mano_shape", "camera_translation"):
        value = getattr(result, name)
        if value is not None:
            arrays[name] = value
    if result.video_focal_px is not None:
        arrays["video_focal_px"] = np.asarray(result.video_focal_px, dtype=np.float32)
    if result.media_metadata is not None:
        provenance = {**provenance, "media_metadata": result.media_metadata}
    npz_path = output_dir / "results.npz"
    np.savez_compressed(npz_path, **arrays)
    run_path = output_dir / "run.json"
    run_path.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    return npz_path, run_path
