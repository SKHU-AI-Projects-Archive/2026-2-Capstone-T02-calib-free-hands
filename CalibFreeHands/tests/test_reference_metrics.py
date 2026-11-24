import numpy as np
import pytest

from calibfreehands.pipeline.result import CanonicalResult
from calibfreehands.pipeline.metrics import MetricStatus, compare_results, metric_registry


def _result(frame_indices=(0,), offset=(0, 0, 0), detected=None, translation=True, mesh=False):
    frames = len(frame_indices)
    points = np.full((frames, 2, 21, 3), np.nan, np.float32)
    points_2d = np.full((frames, 2, 21, 2), np.nan, np.float32)
    mask = np.zeros((frames, 2), bool) if detected is None else np.asarray(detected, bool)
    points[mask] = np.array((0.0, 0.0, 2.0), np.float32) + np.asarray(offset, np.float32)
    points_2d[mask] = 0
    vertices = faces = None
    if mesh:
        vertices = np.full((frames, 2, 4, 3), np.nan, np.float32)
        vertices[mask] = np.array(((0, 0, 2), (1, 0, 2), (0, 1, 2), (0, 0, 3)), np.float32) + offset
        faces = np.array(((0, 1, 2), (0, 2, 3)), np.int32)
    camera_translation = None
    if translation:
        camera_translation = np.full((frames, 2, 3), np.nan, np.float32)
        camera_translation[mask] = np.array((0, 0, 2), np.float32) + offset
    return CanonicalResult(np.asarray(frame_indices), np.arange(frames, dtype=float),
                           points_2d, points,
                           mask, np.full(frames, 500.0), (640, 480), vertices, faces,
                           camera_translation=camera_translation)


def test_exact_and_known_offset_errors_use_mm_and_align_frame_and_side():
    reference = _result((10, 20), detected=((1, 0), (0, 1)), mesh=True)
    candidate = _result((20, 10), offset=(0.01, 0, 0), detected=((0, 1), (1, 0)), mesh=True)
    rows = compare_results(reference, candidate)
    assert [(row.frame_index, row.hand) for row in rows] == [(20, "right"), (10, "left")]
    row = rows[0]
    assert row.metrics["wrist_error"] == pytest.approx(10)
    assert row.metrics["mpjpe"] == pytest.approx(10)
    assert row.metrics["joint_error_rmse"] == pytest.approx(10)
    assert row.metrics["wrist_depth_error"] == pytest.approx(0)
    assert row.metrics["relative_wrist_error"] == pytest.approx(10 / np.hypot(2, 0) * 100)
    assert row.metrics["mpvpe"] == pytest.approx(10)
    assert row.metrics["translation_error"] == pytest.approx(10)


def test_exact_match_and_25mm_depth_offset():
    reference = _result(detected=((1, 0),))
    exact = compare_results(reference, reference)[0]
    assert exact.metrics["mpjpe"] == 0
    assert exact.metrics["wrist_error"] == 0
    np.testing.assert_array_equal(exact.joint_errors_mm, np.zeros(21))
    candidate = _result(offset=(0, 0, 0.025), detected=((1, 0),))
    row = compare_results(reference, candidate)[0]
    assert row.metrics["mpjpe"] == pytest.approx(25, abs=0.001)
    assert row.metrics["wrist_depth_error"] == pytest.approx(25, abs=0.001)
    assert row.metrics["mean_joint_depth_error"] == pytest.approx(25, abs=0.001)


def test_joint_aggregates_depth_bias_outlier_and_nan_are_explicit():
    reference = _result(detected=((1, 0),))
    candidate = _result(offset=(0, 0.025, 0), detected=((1, 0),))
    candidate.keypoints_3d[0, 0, 1] = (np.nan, np.nan, np.nan)
    candidate.keypoints_3d[0, 0, 2] += np.array((0, 0, 0.025), np.float32)
    row = compare_results(reference, candidate)[0]
    assert row.valid_joint_count == 20
    np.testing.assert_allclose(row.joint_errors_mm[[0, 2]], [25, 35.35534], rtol=1e-5)
    assert row.metrics["joint_error_median"] == pytest.approx(25)
    assert row.metrics["joint_error_max"] == pytest.approx(35.35534, rel=1e-5)
    assert row.metrics["mean_joint_depth_error"] == pytest.approx(1.25, abs=1e-5)
    assert row.metrics["signed_mean_depth_bias"] == pytest.approx(1.25, abs=1e-5)


def test_missing_hands_and_incompatible_mesh_translation_are_unavailable():
    reference = _result(detected=((1, 0),), translation=False, mesh=True)
    candidate = _result(detected=((0, 1),), translation=True, mesh=True)
    assert compare_results(reference, candidate) == []
    candidate = _result(detected=((1, 0),), translation=True, mesh=True)
    candidate.mano_faces = np.array(((0, 1, 3), (0, 2, 3)), np.int32)
    row = compare_results(reference, candidate)[0]
    assert row.metric_status["translation_error"] is MetricStatus.UNAVAILABLE_FOR_DATA
    assert row.metrics["translation_error"] is None
    assert row.metric_status["mpvpe"] is MetricStatus.UNAVAILABLE_FOR_DATA
    assert row.metrics["mpvpe"] is None


def test_relative_wrist_rejects_zero_reference_denominator():
    reference = _result(detected=((1, 0),))
    candidate = _result(detected=((1, 0),))
    reference.keypoints_3d[0, 0, 0] = 0
    candidate.keypoints_3d[0, 0, 0] = (0.01, 0, 0)
    row = compare_results(reference, candidate)[0]
    assert row.metrics["relative_wrist_error"] is None
    assert row.metric_status["relative_wrist_error"] is MetricStatus.UNAVAILABLE_FOR_DATA


def test_procrustes_removes_known_similarity_transform_without_mutation():
    reference = _result(detected=((1, 0),))
    points = reference.keypoints_3d[0, 0].copy()
    points[:, 0] += np.linspace(-0.2, 0.2, 21)
    points[:, 1] += np.linspace(0.1, -0.1, 21) ** 2
    reference.keypoints_3d[0, 0] = points
    candidate = _result(detected=((1, 0),))
    angle = 0.7
    rotation = np.array(((np.cos(angle), -np.sin(angle), 0),
                         (np.sin(angle), np.cos(angle), 0), (0, 0, 1)))
    candidate.keypoints_3d[0, 0] = 1.7 * points @ rotation + (0.4, -0.2, 1.0)
    before = candidate.keypoints_3d.copy()
    row = compare_results(reference, candidate)[0]
    assert row.metrics["mpjpe"] > 100
    assert row.metrics["pa_mpjpe"] == pytest.approx(0, abs=1e-4)
    np.testing.assert_array_equal(candidate.keypoints_3d, before)


def test_pa_mpvpe_requires_matching_topology_and_aligns_mesh_vertices():
    reference = _result(detected=((1, 0),), mesh=True)
    candidate = _result(detected=((1, 0),), mesh=True)
    candidate.mano_vertices[0, 0] = 1.5 * reference.mano_vertices[0, 0] + (0.2, -0.4, 0.7)
    row = compare_results(reference, candidate)[0]
    assert row.metrics["mpvpe"] > 100
    assert row.metrics["pa_mpvpe"] == pytest.approx(0, abs=1e-4)


def test_metric_registry_marks_unreproduced_protocols_pending():
    registry = {metric.metric_id: metric for metric in metric_registry()}
    assert registry["mpjpe"].unit == "mm"
    assert "camera-to-wrist distance" in registry["relative_wrist_error"].description
    for metric_id in ("f@5", "f@15", "auc_j", "auc_v", "mpfje", "mpfve", "jitter", "rte"):
        assert registry[metric_id].status is MetricStatus.PROTOCOL_PENDING
