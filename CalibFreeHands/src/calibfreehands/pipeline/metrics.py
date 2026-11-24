from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .result import CanonicalResult


class MetricStatus(str, Enum):
    READY = "READY"
    PROTOCOL_PENDING = "PROTOCOL_PENDING"
    UNAVAILABLE_FOR_DATA = "UNAVAILABLE_FOR_DATA"


@dataclass(frozen=True)
class MetricDefinition:
    metric_id: str
    name: str
    category: str
    unit: str
    status: MetricStatus
    scope: str = "current_frame_hand"
    data_requirements: tuple[str, ...] = ("canonical 21-joint camera-space result",)
    reference_required: bool = True
    protocol_version: str = "calibfreehands-reference-v1"
    description: str = ""


@dataclass(frozen=True)
class FrameHandMetrics:
    frame_index: int
    timestamp_s: float
    hand: str
    joint_errors_mm: np.ndarray
    valid_joint_count: int
    metrics: dict[str, object]
    metric_status: dict[str, MetricStatus]


_READY = MetricStatus.READY
_PENDING = MetricStatus.PROTOCOL_PENDING
_UNAVAILABLE = MetricStatus.UNAVAILABLE_FOR_DATA
_METRICS = (
    MetricDefinition("per_joint_error", "Per-Joint 3D Error", "Reference Accuracy", "mm", _READY),
    MetricDefinition("wrist_error", "Wrist Position Error", "Reference Accuracy", "mm", _READY),
    MetricDefinition("mpjpe", "Camera-space MPJPE", "Reference Accuracy", "mm", _READY),
    MetricDefinition("joint_error_median", "Median Joint Error", "Reference Accuracy", "mm", _READY),
    MetricDefinition("joint_error_max", "Max Joint Error", "Reference Accuracy", "mm", _READY),
    MetricDefinition("joint_error_rmse", "Joint Error RMSE", "Reference Accuracy", "mm", _READY),
    MetricDefinition("wrist_depth_error", "Wrist Depth Error", "Reference Accuracy", "mm", _READY),
    MetricDefinition("mean_joint_depth_error", "Mean Joint Depth Error", "Reference Accuracy", "mm", _READY),
    MetricDefinition("signed_mean_depth_bias", "Signed Mean Depth Bias", "Reference Accuracy", "mm", _READY),
    MetricDefinition("translation_error", "Translation Vector Error", "Reference Accuracy", "mm", _READY,
                     data_requirements=("compatible camera_translation arrays shaped [frame, hand, xyz]",)),
    MetricDefinition("translation_z_error", "Translation Z Error", "Reference Accuracy", "mm", _READY,
                     data_requirements=("compatible camera_translation arrays shaped [frame, hand, xyz]",)),
    MetricDefinition("translation_xy_error", "Translation XY Error", "Reference Accuracy", "mm", _READY,
                     data_requirements=("compatible camera_translation arrays shaped [frame, hand, xyz]",)),
    MetricDefinition("relative_wrist_error", "Relative Wrist Error", "Reference Accuracy", "%", _READY,
                     description="Wrist position error / ||reference wrist position|| * 100; denominator is camera-to-wrist distance."),
    MetricDefinition("mpvpe", "Camera-space MPVPE", "Pose / Mesh Shape", "mm", _READY,
                     data_requirements=("finite MANO vertices and identical face topology",)),
    MetricDefinition("pa_mpjpe", "PA-MPJPE", "Pose / Mesh Shape", "mm", _READY,
                     data_requirements=("at least three finite corresponding joints",)),
    MetricDefinition("pa_mpvpe", "PA-MPVPE", "Pose / Mesh Shape", "mm", _READY,
                     data_requirements=("at least three finite MANO vertices and identical face topology",)),
    *(MetricDefinition(metric_id, label, category, unit, _PENDING,
                       scope="sequence" if category == "Temporal Stability" else "sequence_or_benchmark",
                       data_requirements=("official benchmark/video evaluation protocol not yet reproduced",),
                       protocol_version="pending")
      for metric_id, label, category, unit in (
        ("f@5", "F@5", "Pose / Mesh Shape", "score"),
        ("f@15", "F@15", "Pose / Mesh Shape", "score"),
        ("auc_j", "AUC-J", "Pose / Mesh Shape", "score"),
        ("auc_v", "AUC-V", "Pose / Mesh Shape", "score"),
        ("mpfje", "MPFJE", "Temporal Stability", "protocol-defined"),
        ("mpfve", "MPFVE", "Temporal Stability", "protocol-defined"),
        ("jitter", "Jitter", "Temporal Stability", "protocol-defined"),
        ("rte", "RTE", "Temporal Stability", "protocol-defined"),
    )),
)


def metric_registry() -> tuple[MetricDefinition, ...]:
    return _METRICS


def _similarity_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    candidate_center = candidate.mean(axis=0)
    reference_center = reference.mean(axis=0)
    x = candidate - candidate_center
    y = reference - reference_center
    variance = np.sum(x * x)
    if not np.isfinite(variance) or variance <= np.finfo(np.float64).eps:
        return float("nan")
    u, singular_values, vt = np.linalg.svd(x.T @ y)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    scale = np.sum(singular_values * np.diag(correction)) / variance
    aligned = scale * x @ rotation + reference_center
    return float(np.linalg.norm(aligned - y - reference_center, axis=1).mean() * 1000.0)


def _mesh_compatible(reference: CanonicalResult, candidate: CanonicalResult) -> bool:
    return (
        reference.mano_vertices is not None
        and candidate.mano_vertices is not None
        and reference.mano_vertices.shape[2:] == candidate.mano_vertices.shape[2:]
        and reference.mano_faces is not None
        and candidate.mano_faces is not None
        and reference.mano_faces.shape == candidate.mano_faces.shape
        and np.array_equal(reference.mano_faces, candidate.mano_faces)
    )


def compare_results(reference: CanonicalResult, candidate: CanonicalResult) -> list[FrameHandMetrics]:
    """Compare matching source frames and hand slots without mutating canonical arrays."""
    for result in (reference, candidate):
        if result.keypoints_3d.ndim != 4 or result.keypoints_3d.shape[1:3] != (2, 21) or result.keypoints_3d.shape[-1] != 3:
            raise ValueError("comparison requires canonical [frame, 2 hands, 21 joints, xyz] keypoints_3d")

    reference_frames = {int(frame): position for position, frame in enumerate(reference.frame_indices)}
    candidate_translation_ok = (
        reference.camera_translation is not None
        and candidate.camera_translation is not None
        and reference.camera_translation.shape == (len(reference.frame_indices), 2, 3)
        and candidate.camera_translation.shape == (len(candidate.frame_indices), 2, 3)
    )
    mesh_ok = _mesh_compatible(reference, candidate)
    rows = []
    for candidate_pos, raw_frame in enumerate(candidate.frame_indices):
        frame = int(raw_frame)
        reference_pos = reference_frames.get(frame)
        if reference_pos is None:
            continue
        for hand_index, hand_name in enumerate(("left", "right")):
            if not (reference.detected[reference_pos, hand_index] and candidate.detected[candidate_pos, hand_index]):
                continue
            reference_points = np.asarray(reference.keypoints_3d[reference_pos, hand_index], dtype=np.float64)
            candidate_points = np.asarray(candidate.keypoints_3d[candidate_pos, hand_index], dtype=np.float64)
            valid = np.isfinite(reference_points).all(axis=1) & np.isfinite(candidate_points).all(axis=1)
            errors = np.full(21, np.nan, dtype=np.float64)
            errors[valid] = np.linalg.norm(candidate_points[valid] - reference_points[valid], axis=1) * 1000.0
            if not valid.any():
                continue
            distances = errors[valid]
            depth_delta = (candidate_points[valid, 2] - reference_points[valid, 2]) * 1000.0
            wrist_valid = bool(valid[0])
            wrist_error = float(errors[0]) if wrist_valid else None
            wrist_depth = None
            if wrist_valid:
                wrist_depth = float(abs(candidate_points[0, 2] - reference_points[0, 2]) * 1000.0)
                wrist_distance = float(np.linalg.norm(reference_points[0]))
                relative_wrist = wrist_error / wrist_distance * 100.0 if np.isfinite(wrist_distance) and wrist_distance > 0 else None
            else:
                relative_wrist = None
            values: dict[str, object] = {
                "per_joint_error": [float(value) if np.isfinite(value) else None for value in errors],
                "wrist_error": wrist_error,
                "mpjpe": float(distances.mean()),
                "joint_error_median": float(np.median(distances)),
                "joint_error_max": float(distances.max()),
                "joint_error_rmse": float(np.sqrt(np.mean(distances ** 2))),
                "wrist_depth_error": wrist_depth,
                "mean_joint_depth_error": float(np.abs(depth_delta).mean()),
                "signed_mean_depth_bias": float(depth_delta.mean()),
                "translation_error": None,
                "translation_z_error": None,
                "translation_xy_error": None,
                "relative_wrist_error": relative_wrist,
                "mpvpe": None,
                "pa_mpjpe": None,
                "pa_mpvpe": None,
            }
            if valid.sum() >= 3:
                pa_error = _similarity_error(candidate_points[valid], reference_points[valid])
                values["pa_mpjpe"] = pa_error if np.isfinite(pa_error) else None
            statuses = {metric.metric_id: metric.status for metric in _METRICS}
            for metric_id in ("wrist_error", "wrist_depth_error", "relative_wrist_error"):
                if values[metric_id] is None:
                    statuses[metric_id] = _UNAVAILABLE
            if candidate_translation_ok:
                delta = (candidate.camera_translation[candidate_pos, hand_index] -
                         reference.camera_translation[reference_pos, hand_index]).astype(np.float64)
                if np.isfinite(delta).all():
                    values["translation_error"] = float(np.linalg.norm(delta) * 1000.0)
                    values["translation_z_error"] = float(abs(delta[2]) * 1000.0)
                    values["translation_xy_error"] = float(np.linalg.norm(delta[:2]) * 1000.0)
            if values["translation_error"] is None:
                for metric_id in ("translation_error", "translation_z_error", "translation_xy_error"):
                    statuses[metric_id] = _UNAVAILABLE
            if mesh_ok:
                reference_vertices = np.asarray(reference.mano_vertices[reference_pos, hand_index], dtype=np.float64)
                candidate_vertices = np.asarray(candidate.mano_vertices[candidate_pos, hand_index], dtype=np.float64)
                vertex_valid = np.isfinite(reference_vertices).all(axis=1) & np.isfinite(candidate_vertices).all(axis=1)
                if vertex_valid.any():
                    vertex_delta = candidate_vertices[vertex_valid] - reference_vertices[vertex_valid]
                    vertex_errors = np.linalg.norm(vertex_delta, axis=1)
                    values["mpvpe"] = float(vertex_errors.mean() * 1000.0)
                    if vertex_valid.sum() >= 3:
                        pa_error = _similarity_error(candidate_vertices[vertex_valid], reference_vertices[vertex_valid])
                        values["pa_mpvpe"] = pa_error if np.isfinite(pa_error) else None
            for metric_id in ("mpvpe", "pa_mpvpe"):
                if values[metric_id] is None:
                    statuses[metric_id] = _UNAVAILABLE
            if values["pa_mpjpe"] is None:
                statuses["pa_mpjpe"] = _UNAVAILABLE
            rows.append(FrameHandMetrics(frame, float(candidate.timestamps[candidate_pos]), hand_name,
                                         errors, int(valid.sum()), values, statuses))
    return rows
