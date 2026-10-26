from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from calibfreehands.calibration import aggregate_focal, sample_indices
from calibfreehands.media.ffmpeg import DecodedFrame
from calibfreehands.export.preview import encode_result_preview
from .result import CanonicalResult, save_result
from .provenance import runtime_metadata, sha256_file


@dataclass(frozen=True)
class HandEstimate:
    is_right: bool
    keypoints_2d: np.ndarray
    keypoints_3d_relative: np.ndarray
    camera_translation: np.ndarray
    vertices_relative: np.ndarray | None
    bbox: np.ndarray
    confidence: float
    mano_pose: np.ndarray | None = None
    mano_shape: np.ndarray | None = None
    mano_faces: np.ndarray | None = None


class CalibrationBackend(Protocol):
    """One focal estimator selected for a single processing run."""

    def estimate_focal(self, image_rgb: np.ndarray) -> float: ...


class PoseBackend(Protocol):
    """One hand reconstruction backend paired with the run's focal estimator."""

    def predict(self, image_rgb: np.ndarray, focal_px: float) -> list[HandEstimate]: ...


def process_frames(frames: list[DecodedFrame], calibrator: CalibrationBackend, pose: PoseBackend,
                   sampling_mode: str = "uniform", sample_count: int = 8, aggregation: str = "median",
                   progress=None, cancelled=None) -> CanonicalResult:
    """Process one calibration/reconstruction pair and preserve the canonical result contract."""
    if not frames:
        raise ValueError("at least one decoded frame is required")
    sample_ids = sample_indices(len(frames), sampling_mode, sample_count)
    sampled_focals = []
    for position, index in enumerate(sample_ids, 1):
        if cancelled and cancelled(): raise InterruptedError("Processing cancelled during calibration")
        sampled_focals.append(calibrator.estimate_focal(frames[int(index)].rgb))
        if progress: progress("Estimating focal", position, len(sample_ids))
    video_focal = aggregate_focal(sampled_focals, aggregation)
    frame_count = len(frames)
    keypoints_2d = np.full((frame_count, 2, 21, 2), np.nan, np.float32)
    keypoints_3d = np.full((frame_count, 2, 21, 3), np.nan, np.float32)
    detected = np.zeros((frame_count, 2), bool)
    confidence = np.full((frame_count, 2), np.nan, np.float32)
    bbox = np.full((frame_count, 2, 4), np.nan, np.float32)
    camera_translation = np.full((frame_count, 2, 3), np.nan, np.float32)
    mano_vertices = None
    mano_faces = None
    mano_pose = None
    mano_shape = None
    frame_focal = np.full(frame_count, video_focal, np.float32)
    for frame_pos, frame in enumerate(frames):
        if cancelled and cancelled(): raise InterruptedError("Processing cancelled during reconstruction")
        for estimate in pose.predict(frame.rgb, video_focal):
            hand = 1 if estimate.is_right else 0
            keypoints_2d[frame_pos, hand] = estimate.keypoints_2d
            keypoints_3d[frame_pos, hand] = estimate.keypoints_3d_relative + estimate.camera_translation
            detected[frame_pos, hand] = True
            confidence[frame_pos, hand] = estimate.confidence
            bbox[frame_pos, hand] = estimate.bbox
            camera_translation[frame_pos, hand] = estimate.camera_translation
            if estimate.vertices_relative is not None:
                if mano_vertices is None:
                    mano_vertices = np.full((frame_count, 2, *estimate.vertices_relative.shape), np.nan, np.float32)
                mano_vertices[frame_pos, hand] = estimate.vertices_relative + estimate.camera_translation
            for field_name in ("mano_pose", "mano_shape"):
                value = getattr(estimate, field_name)
                if value is None:
                    continue
                value = np.asarray(value, dtype=np.float32)
                stored = mano_pose if field_name == "mano_pose" else mano_shape
                if stored is None:
                    stored = np.full((frame_count, 2, *value.shape), np.nan, np.float32)
                elif stored.shape[2:] != value.shape:
                    raise ValueError(f"{field_name} shape changed from {stored.shape[2:]} to {value.shape}")
                stored[frame_pos, hand] = value
                if field_name == "mano_pose":
                    mano_pose = stored
                else:
                    mano_shape = stored
            if estimate.mano_faces is not None and mano_faces is None:
                faces = np.asarray(estimate.mano_faces, dtype=np.int32)
                mano_faces = np.stack((faces[:, [0, 2, 1]], faces), axis=0) if faces.ndim == 2 else faces
        if progress: progress("Reconstructing hands", frame_pos + 1, frame_count)
    def json_float(value):
        value = float(value)
        return value if np.isfinite(value) else None
    calibration_metadata = {
        "sampling_mode": sampling_mode, "requested_sample_count": int(sample_count),
        "actual_sample_count": int(len(sample_ids)), "aggregation": aggregation,
        "frame_indices": [int(frames[int(index)].index) for index in sample_ids],
        "timestamps_s": [json_float(frames[int(index)].timestamp_s) for index in sample_ids],
        "focal_px": [json_float(value) for value in sampled_focals],
        "aggregated_focal_px": json_float(video_focal),
    }
    return CanonicalResult(np.array([frame.index for frame in frames]), np.array([frame.timestamp_s for frame in frames]),
                           keypoints_2d, keypoints_3d, detected, frame_focal,
                           (frames[0].rgb.shape[1], frames[0].rgb.shape[0]), mano_vertices, mano_faces,
                           confidence, bbox, mano_pose=mano_pose, mano_shape=mano_shape,
                           camera_translation=camera_translation, video_focal_px=video_focal,
                           media_metadata={"frame_count": frame_count}, calibration_metadata=calibration_metadata)


def process_and_save(frames: list[DecodedFrame], calibrator: CalibrationBackend, pose: PoseBackend,
                     output_dir: Path, settings: dict, progress=None, cancelled=None) -> tuple[Path, Path]:
    started = time.monotonic()
    result = process_frames(frames, calibrator, pose, settings.get("sampling_mode", "uniform"),
                             int(settings.get("sample_count", 8)), settings.get("aggregation", "median"), progress, cancelled)
    provenance = {"schema_version": result.schema_version, "application_version": "0.1.0",
                  "calibration_backend": getattr(calibrator, "name", type(calibrator).__name__),
                  "pose_backend": getattr(pose, "name", type(pose).__name__),
                  "inference_settings": settings, "coordinate_system": "camera_space",
                  "unit": "metre", "elapsed_s": time.monotonic() - started,
                  "applied_camera_parameters": {"focal_px": result.video_focal_px,
                                                 "principal_point_applied": False,
                                                 "distortion_applied": False},
                  "runtime": runtime_metadata(), "outputs": ["results.npz", "run.json"]}
    if settings.get("model_selection") is not None:
        provenance["model_selection"] = settings["model_selection"]
    if result.calibration_metadata is not None:
        provenance["calibration"] = result.calibration_metadata
    input_path = settings.get("input_path")
    if input_path and Path(input_path).is_file():
        provenance["input_path"] = str(Path(input_path).resolve())
        provenance["input_sha256"] = sha256_file(Path(input_path))
    checkpoint_paths = settings.get("checkpoint_paths", {})
    provenance["checkpoint_sha256"] = {name: sha256_file(Path(path)) for name, path in checkpoint_paths.items() if Path(path).is_file()}
    if hasattr(pose, "mano_paths"):
        provenance["mano_assets"] = {name: str(path) for name, path in pose.mano_paths.items()}
    preview_output = settings.get("preview_output")
    if preview_output:
        if cancelled and cancelled(): raise InterruptedError("Processing cancelled before preview")
        preview_path = Path(preview_output)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        def preview_frames():
            for position, frame in enumerate(frames, 1):
                if cancelled and cancelled(): raise InterruptedError("Processing cancelled during preview")
                if progress: progress("Creating preview", position, len(frames))
                yield frame.rgb
        encode_result_preview(preview_frames(), result, preview_path,
                              float(settings.get("fps", 30.0)),
                              Path(input_path) if settings.get("preserve_audio") and input_path else None)
        provenance["outputs"].append(str(preview_path.name))
    if progress: progress("Saving results", 1, 1)
    return save_result(result, output_dir, provenance)
