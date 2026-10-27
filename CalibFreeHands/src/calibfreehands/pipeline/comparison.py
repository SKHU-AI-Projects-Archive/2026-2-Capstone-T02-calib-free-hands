from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from calibfreehands.app.configuration import ApplicationConfiguration
from calibfreehands.calibration import aggregate_focal, sample_indices
from calibfreehands.export.preview import encode_result_preview
from calibfreehands.media.ffmpeg import DecodedFrame
from .provenance import runtime_metadata, sha256_file
from .result import CanonicalResult, save_result


@dataclass(frozen=True)
class PipelineConfiguration:
    id: str
    name: str
    calibration_model_id: str
    reconstruction_model_id: str
    reference_focal_px: float | None = None


@dataclass(frozen=True)
class RawHandEstimate:
    is_right: bool
    keypoints_2d: np.ndarray
    keypoints_3d_relative: np.ndarray
    camera_translation_source: np.ndarray
    source_focal_px: float
    bbox: np.ndarray
    confidence: float
    vertices_relative: np.ndarray | None = None
    mano_pose: np.ndarray | None = None
    mano_shape: np.ndarray | None = None
    mano_faces: np.ndarray | None = None


@dataclass(frozen=True)
class CalibrationResult:
    id: str
    model_id: str
    focal_px: float
    metadata: dict


@dataclass
class RawPoseResult:
    frame_indices: np.ndarray
    timestamps: np.ndarray
    keypoints_2d: np.ndarray
    keypoints_3d_relative: np.ndarray
    detected: np.ndarray
    camera_translation_source: np.ndarray
    source_focal_px: np.ndarray
    image_size: tuple[int, int]
    vertices_relative: np.ndarray | None = None
    mano_faces: np.ndarray | None = None
    confidence: np.ndarray | None = None
    bbox: np.ndarray | None = None
    mano_pose: np.ndarray | None = None
    mano_shape: np.ndarray | None = None

    @classmethod
    def from_predictions(
        cls, frames: list[DecodedFrame], predictions: list[list[RawHandEstimate]]
    ) -> RawPoseResult:
        if len(frames) != len(predictions) or not frames:
            raise ValueError("raw pose predictions must contain one entry per decoded frame")
        count = len(frames)
        keypoints_2d = np.full((count, 2, 21, 2), np.nan, np.float32)
        keypoints_3d = np.full((count, 2, 21, 3), np.nan, np.float32)
        detected = np.zeros((count, 2), bool)
        translation = np.full((count, 2, 3), np.nan, np.float32)
        source_focal = np.full((count, 2), np.nan, np.float32)
        confidence = np.full((count, 2), np.nan, np.float32)
        bbox = np.full((count, 2, 4), np.nan, np.float32)
        vertices = mano_pose = mano_shape = mano_faces = None

        for frame_index, estimates in enumerate(predictions):
            for estimate in estimates:
                hand = 1 if estimate.is_right else 0
                if estimate.keypoints_2d.shape != (21, 2) or estimate.keypoints_3d_relative.shape != (21, 3):
                    raise ValueError("raw hand keypoints have an unexpected shape")
                if not np.isfinite(estimate.source_focal_px) or estimate.source_focal_px <= 0:
                    raise ValueError("raw hand source focal must be finite and positive")
                keypoints_2d[frame_index, hand] = estimate.keypoints_2d
                keypoints_3d[frame_index, hand] = estimate.keypoints_3d_relative
                detected[frame_index, hand] = True
                translation[frame_index, hand] = estimate.camera_translation_source
                source_focal[frame_index, hand] = estimate.source_focal_px
                confidence[frame_index, hand] = estimate.confidence
                bbox[frame_index, hand] = estimate.bbox
                if estimate.vertices_relative is not None:
                    value = np.asarray(estimate.vertices_relative, dtype=np.float32)
                    if vertices is None:
                        vertices = np.full((count, 2, *value.shape), np.nan, np.float32)
                    elif vertices.shape[2:] != value.shape:
                        raise ValueError("raw MANO vertex shape changed")
                    vertices[frame_index, hand] = value
                for field_name in ("mano_pose", "mano_shape"):
                    value = getattr(estimate, field_name)
                    if value is None:
                        continue
                    value = np.asarray(value, dtype=np.float32)
                    stored = mano_pose if field_name == "mano_pose" else mano_shape
                    if stored is None:
                        stored = np.full((count, 2, *value.shape), np.nan, np.float32)
                    elif stored.shape[2:] != value.shape:
                        raise ValueError(f"raw {field_name} shape changed")
                    stored[frame_index, hand] = value
                    if field_name == "mano_pose":
                        mano_pose = stored
                    else:
                        mano_shape = stored
                if estimate.mano_faces is not None and mano_faces is None:
                    faces = np.asarray(estimate.mano_faces, dtype=np.int32)
                    mano_faces = np.stack((faces[:, [0, 2, 1]], faces), axis=0) if faces.ndim == 2 else faces

        return cls(
            np.asarray([frame.index for frame in frames]),
            np.asarray([frame.timestamp_s for frame in frames], dtype=np.float64),
            keypoints_2d, keypoints_3d, detected, translation, source_focal,
            (frames[0].rgb.shape[1], frames[0].rgb.shape[0]),
            vertices, mano_faces, confidence, bbox, mano_pose, mano_shape,
        )

    def save(self, path: Path) -> None:
        values = {
            "frame_indices": self.frame_indices,
            "timestamps": self.timestamps,
            "keypoints_2d": self.keypoints_2d,
            "keypoints_3d_relative": self.keypoints_3d_relative,
            "detected": self.detected,
            "camera_translation_source": self.camera_translation_source,
            "source_focal_px": self.source_focal_px,
            "image_size": np.asarray(self.image_size, np.int32),
        }
        for name in ("vertices_relative", "mano_faces", "confidence", "bbox", "mano_pose", "mano_shape"):
            value = getattr(self, name)
            if value is not None:
                values[name] = value
        np.savez_compressed(path, **values)

    @classmethod
    def load(cls, path: Path) -> RawPoseResult:
        with np.load(path, allow_pickle=False) as values:
            optional = lambda name: values[name].copy() if name in values else None
            return cls(
                values["frame_indices"].copy(), values["timestamps"].copy(),
                values["keypoints_2d"].copy(), values["keypoints_3d_relative"].copy(),
                values["detected"].copy(), values["camera_translation_source"].copy(),
                values["source_focal_px"].copy(), tuple(int(v) for v in values["image_size"]),
                optional("vertices_relative"), optional("mano_faces"), optional("confidence"),
                optional("bbox"), optional("mano_pose"), optional("mano_shape"),
            )

    def apply_focal(self, target_focal_px: float, calibration_metadata: dict | None = None) -> CanonicalResult:
        if not np.isfinite(target_focal_px) or target_focal_px <= 0:
            raise ValueError("target focal must be finite and positive")
        corrected_translation = self.camera_translation_source.copy()
        rows, hands = np.where(self.detected)
        source = self.source_focal_px[rows, hands]
        if not np.isfinite(source).all() or (source <= 0).any():
            raise ValueError("detected raw hands require a finite positive source focal")
        corrected_translation[rows, hands, 2] *= target_focal_px / source
        keypoints_3d = np.full_like(self.keypoints_3d_relative, np.nan)
        keypoints_3d[self.detected] = (
            self.keypoints_3d_relative[self.detected] + corrected_translation[self.detected, None, :]
        )
        vertices = None
        if self.vertices_relative is not None:
            vertices = np.full_like(self.vertices_relative, np.nan)
            vertices[self.detected] = (
                self.vertices_relative[self.detected] + corrected_translation[self.detected, None, :]
            )
        return CanonicalResult(
            self.frame_indices.copy(), self.timestamps.copy(), self.keypoints_2d.copy(), keypoints_3d,
            self.detected.copy(), np.full(len(self.frame_indices), target_focal_px, np.float32),
            self.image_size, vertices, self.mano_faces, self.confidence, self.bbox,
            mano_pose=self.mano_pose, mano_shape=self.mano_shape,
            camera_translation=corrected_translation, video_focal_px=float(target_focal_px),
            media_metadata={"frame_count": len(self.frame_indices)},
            calibration_metadata=calibration_metadata,
        )


@dataclass
class ComparisonSession:
    id: str
    source_media: str
    selected_preset: str | None
    configurations: list[PipelineConfiguration]
    state: str = "processing"
    calibration_results: dict[str, dict] = field(default_factory=dict)
    raw_pose_results: dict[str, dict] = field(default_factory=dict)
    final_results: dict[str, dict] = field(default_factory=dict)
    reference_configuration_id: str | None = None
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "session_id": self.id,
            "source_media": self.source_media,
            "state": self.state,
            "selected_preset": self.selected_preset,
            "configurations": [asdict(item) for item in self.configurations],
            "calibration_results": self.calibration_results,
            "raw_pose_results": self.raw_pose_results,
            "final_results": self.final_results,
            "reference_configuration_id": self.reference_configuration_id,
            "provenance": self.provenance,
        }


def _write_session(path: Path, session: ComparisonSession) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(session.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def load_comparison_session(path: Path) -> ComparisonSession:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError(f"Unsupported comparison session schema: {data.get('schema_version')}")
    return ComparisonSession(
        id=data["session_id"], source_media=data["source_media"], selected_preset=data.get("selected_preset"),
        configurations=[PipelineConfiguration(**item) for item in data["configurations"]],
        state=data["state"], calibration_results=data["calibration_results"],
        raw_pose_results=data["raw_pose_results"], final_results=data["final_results"],
        reference_configuration_id=data.get("reference_configuration_id"), provenance=data["provenance"],
    )


def _validate_configurations(configurations: list[PipelineConfiguration], config: ApplicationConfiguration) -> None:
    if not configurations:
        raise ValueError("at least one pipeline configuration is required")
    ids = [item.id for item in configurations]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("pipeline configuration IDs must be non-empty and unique")
    for item in configurations:
        if not item.name.strip():
            raise ValueError("pipeline configuration names must not be empty")
        calibration = config.calibration_models.get(item.calibration_model_id)
        reconstruction = config.reconstruction_models.get(item.reconstruction_model_id)
        if calibration is None:
            raise KeyError(f"Unknown calibration model: {item.calibration_model_id}")
        if reconstruction is None:
            raise KeyError(f"Unknown reconstruction model: {item.reconstruction_model_id}")
        if not calibration.available:
            raise RuntimeError(f"Calibration model {calibration.display_name} is not available: {calibration.unavailable_reason}")
        if not reconstruction.available:
            raise RuntimeError(f"Reconstruction model {reconstruction.display_name} is not available: {reconstruction.unavailable_reason}")
        if calibration.backend == "reference_focal":
            if item.reference_focal_px is None or not np.isfinite(item.reference_focal_px) or item.reference_focal_px <= 0:
                raise ValueError(f"{item.name} requires a finite positive reference focal in pixels")


def _calibration_result_id(item: PipelineConfiguration, backend: str) -> str:
    if backend != "reference_focal":
        return item.calibration_model_id
    return f"{item.calibration_model_id}:{item.reference_focal_px:.6f}"


def _calibrate(
    model_id: str, frames: list[DecodedFrame], backend, settings: dict, cancelled, progress, offset: int, total: int
) -> CalibrationResult:
    if getattr(backend, "reference_focal_px", None) is not None:
        if cancelled and cancelled():
            raise InterruptedError("Processing cancelled during reference calibration")
        focal = float(backend.reference_focal_px)
        return CalibrationResult(model_id, model_id, focal, {
            "method": "explicit_reference_focal",
            "focal_px": focal,
            "sampling_mode": settings.get("sampling_mode", "uniform"),
            "requested_sample_count": int(settings.get("sample_count", 8)),
            "actual_sample_count": 0,
            "aggregation": settings.get("aggregation", "median"),
        })
    indices = sample_indices(len(frames), settings.get("sampling_mode", "uniform"), int(settings.get("sample_count", 8)))
    focals = []
    for position, index in enumerate(indices, 1):
        if cancelled and cancelled():
            raise InterruptedError("Processing cancelled during calibration")
        focals.append(float(backend.estimate_focal(frames[int(index)].rgb)))
        if progress:
            progress("Estimating focal", offset + position, total)
    mode = settings.get("aggregation", "median")
    focal = float(aggregate_focal(focals, mode))
    return CalibrationResult(model_id, model_id, focal, {
        "sampling_mode": settings.get("sampling_mode", "uniform"),
        "requested_sample_count": int(settings.get("sample_count", 8)),
        "actual_sample_count": len(indices),
        "aggregation": mode,
        "frame_indices": [int(frames[int(index)].index) for index in indices],
        "timestamps_s": [float(frames[int(index)].timestamp_s) for index in indices],
        "focal_px": focals,
        "aggregated_focal_px": focal,
    })


def _stable_file_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def process_comparison_session(
    frames: list[DecodedFrame],
    configurations: list[PipelineConfiguration],
    config: ApplicationConfiguration,
    project_root: Path,
    output_dir: Path,
    settings: dict,
    progress=None,
    cancelled: Callable[[], bool] | None = None,
    calibration_factory: Callable[[str, float | None], object] | None = None,
    pose_factory: Callable[[str], object] | None = None,
) -> ComparisonSession:
    _validate_configurations(configurations, config)
    if not frames:
        raise ValueError("at least one decoded frame is required")
    if calibration_factory is None:
        from calibfreehands.calibration import create_calibration_backend
        calibration_factory = lambda model_id, focal: create_calibration_backend(
            model_id, config, reference_focal_px=focal
        )
    if pose_factory is None:
        from calibfreehands.pose import create_pose_backend
        pose_factory = lambda model_id: create_pose_backend(model_id, config, Path(project_root))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_id = next((item.id for item in configurations
                         if config.calibration_models[item.calibration_model_id].backend == "reference_focal"), None)
    source_media = str(settings.get("input_path", ""))
    session = ComparisonSession(
        uuid.uuid4().hex, source_media, settings.get("preset_id"), list(configurations),
        reference_configuration_id=reference_id,
        provenance={"application_version": "0.1.0", "runtime": runtime_metadata()},
    )
    manifest_path = output_dir / "session.json"
    _write_session(manifest_path, session)
    started = time.monotonic()
    try:
        unique_calibrations = {}
        for item in configurations:
            model = config.calibration_models[item.calibration_model_id]
            result_id = _calibration_result_id(item, model.backend)
            if result_id in unique_calibrations:
                continue
            backend = calibration_factory(item.calibration_model_id, item.reference_focal_px)
            result = _calibrate(item.calibration_model_id, frames, backend, settings, cancelled, progress,
                                len(unique_calibrations), len(configurations))
            result = CalibrationResult(result_id, item.calibration_model_id, result.focal_px, result.metadata)
            relative_path = Path("calibration") / f"{_stable_file_key(result_id)}.json"
            calibration_path = output_dir / relative_path
            calibration_path.parent.mkdir(parents=True, exist_ok=True)
            calibration_path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True), encoding="utf-8")
            unique_calibrations[result_id] = result
            session.calibration_results[result_id] = {"path": relative_path.as_posix(), **asdict(result)}
            _write_session(manifest_path, session)

        unique_pose_models = list(dict.fromkeys(item.reconstruction_model_id for item in configurations))
        pose_metadata = {}
        for pose_index, model_id in enumerate(unique_pose_models, 1):
            if cancelled and cancelled():
                raise InterruptedError("Processing cancelled before pose inference")
            backend = pose_factory(model_id)
            preflight = backend.preflight() if hasattr(backend, "preflight") else {}
            predict_raw = getattr(backend, "predict_raw", None)
            if not callable(predict_raw):
                raise RuntimeError(f"Reconstruction backend {model_id} does not expose raw pose inference")
            raw_predictions = []
            for frame_index, frame in enumerate(frames, 1):
                if cancelled and cancelled():
                    raise InterruptedError("Processing cancelled during raw pose inference")
                raw_predictions.append(predict_raw(frame.rgb))
                if progress:
                    progress(f"Reconstructing raw poses ({model_id})", frame_index, len(frames))
            raw_result = RawPoseResult.from_predictions(frames, raw_predictions)
            relative_path = Path("pose") / f"{_stable_file_key(model_id)}.npz"
            raw_path = output_dir / relative_path
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_result.save(raw_path)
            pose_metadata[model_id] = {
                "name": getattr(backend, "name", type(backend).__name__),
                "checkpoint": str(getattr(backend, "checkpoint", "")),
                "checkpoint_sha256": sha256_file(backend.checkpoint)
                if getattr(backend, "checkpoint", None) and Path(backend.checkpoint).is_file() else None,
                "mano_assets": {name: str(path) for name, path in getattr(backend, "mano_paths", {}).items()},
                "preflight": preflight,
            }
            session.raw_pose_results[model_id] = {"path": relative_path.as_posix(), **pose_metadata[model_id]}
            del raw_result, raw_predictions, backend
            _write_session(manifest_path, session)

        for config_index, item in enumerate(configurations, 1):
            if cancelled and cancelled():
                raise InterruptedError("Processing cancelled before final result construction")
            calibration_model = config.calibration_models[item.calibration_model_id]
            result_id = _calibration_result_id(item, calibration_model.backend)
            calibration = unique_calibrations[result_id]
            pose_info = session.raw_pose_results[item.reconstruction_model_id]
            raw_path = output_dir / pose_info["path"]
            raw_pose = RawPoseResult.load(raw_path)
            canonical = raw_pose.apply_focal(calibration.focal_px, calibration.metadata)
            result_dir = output_dir / "results" / _stable_file_key(item.id)
            result_dir.mkdir(parents=True, exist_ok=True)
            preview_path = None
            if settings.get("create_preview", False):
                preview_path = result_dir / "preview.mp4"
                encode_result_preview(
                    (frame.rgb for frame in frames), canonical, preview_path,
                    float(settings.get("fps", 30.0)),
                    Path(source_media) if settings.get("preserve_audio") and source_media else None,
                )
            calibration_name = config.calibration_models[item.calibration_model_id].display_name
            pose_name = config.reconstruction_models[item.reconstruction_model_id].display_name
            provenance = {
                "schema_version": canonical.schema_version,
                "application_version": "0.1.0",
                "calibration_backend": calibration_name,
                "pose_backend": pose_info["name"],
                "inference_settings": settings,
                "coordinate_system": "camera_space",
                "unit": "metre",
                "elapsed_s": time.monotonic() - started,
                "applied_camera_parameters": {"focal_px": calibration.focal_px,
                                               "principal_point_applied": False,
                                               "distortion_applied": False},
                "runtime": session.provenance["runtime"],
                "outputs": ["results.npz", "run.json"] + (["preview.mp4"] if preview_path else []),
                "input_path": str(Path(source_media).resolve()) if source_media and Path(source_media).is_file() else source_media,
                "calibration": calibration.metadata,
                "model_selection": {
                    "configuration_id": item.id,
                    "configuration_name": item.name,
                    "calibration_model_id": item.calibration_model_id,
                    "calibration_display_name": calibration_name,
                    "reconstruction_model_id": item.reconstruction_model_id,
                    "reconstruction_display_name": pose_name,
                    "preset_id": settings.get("preset_id"),
                },
                "checkpoint_sha256": {"wilor": pose_info["checkpoint_sha256"]}
                if pose_info["checkpoint_sha256"] else {},
                "mano_assets": pose_info["mano_assets"],
            }
            if source_media and Path(source_media).is_file():
                provenance["input_sha256"] = sha256_file(Path(source_media))
            result_npz, run_json = save_result(canonical, result_dir, provenance)
            session.final_results[item.id] = {
                "name": item.name,
                "calibration_result_id": result_id,
                "raw_pose_model_id": item.reconstruction_model_id,
                "path": result_npz.relative_to(output_dir).as_posix(),
                "run_json": run_json.relative_to(output_dir).as_posix(),
                "focal_px": calibration.focal_px,
            }
            if preview_path:
                session.final_results[item.id]["preview"] = preview_path.relative_to(output_dir).as_posix()
            del raw_pose, canonical
            if progress:
                progress("Saving comparison results", config_index, len(configurations))
            _write_session(manifest_path, session)
        if cancelled and cancelled():
            raise InterruptedError("Processing cancelled before session completion")
        session.state = "completed"
        session.provenance["elapsed_s"] = time.monotonic() - started
        _write_session(manifest_path, session)
        return session
    except InterruptedError:
        session.state = "cancelled"
        session.provenance["elapsed_s"] = time.monotonic() - started
        _write_session(manifest_path, session)
        raise
    except Exception as exc:
        session.state = "failed"
        session.provenance["error"] = f"{type(exc).__name__}: {exc}"
        session.provenance["elapsed_s"] = time.monotonic() - started
        _write_session(manifest_path, session)
        raise
