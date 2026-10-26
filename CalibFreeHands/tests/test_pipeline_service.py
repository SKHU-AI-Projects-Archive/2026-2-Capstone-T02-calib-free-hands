import json

import numpy as np
import pytest

from calibfreehands.media.ffmpeg import DecodedFrame
from calibfreehands.pipeline.service import HandEstimate, process_and_save, process_frames


class Calibrator:
    name = "test-calibrator"
    def estimate_focal(self, image): return 500.0


class Pose:
    name = "test-pose"
    def predict(self, image, focal_px):
        if image[0, 0, 0] == 0: return []
        return [HandEstimate(False, np.zeros((21, 2)), np.zeros((21, 3)), np.array([0, 0, 2.]), None, np.zeros(4), .9)]


class PoseWithVertices(Pose):
    def predict(self, image, focal_px):
        return [HandEstimate(False, np.zeros((21, 2)), np.zeros((21, 3)), np.array([0, 0, 2.]),
                             np.array([[1., 2., 3.]]), np.zeros(4), .9)]


class CalibratorByPixel:
    name = "pixel-calibrator"

    def estimate_focal(self, image):
        return float(int(image[0, 0, 0]) * 100)


class PoseWithMano:
    name = "mano-pose"

    def predict(self, image, focal_px):
        value = int(image[0, 0, 0])
        if value == 0:
            return []
        return [HandEstimate(
            is_right=False,
            keypoints_2d=np.full((21, 2), value, dtype=np.float32),
            keypoints_3d_relative=np.full((21, 3), value, dtype=np.float32),
            camera_translation=np.array([0, 0, 2], dtype=np.float32),
            vertices_relative=np.array([[value, value + 1, value + 2]], dtype=np.float32),
            bbox=np.full(4, value, dtype=np.float32),
            confidence=.9,
            mano_pose=np.arange(48, dtype=np.float32) + value,
            mano_shape=np.arange(10, dtype=np.float32) + value * 10,
        )]


class PoseWithInconsistentMano(PoseWithMano):
    def predict(self, image, focal_px):
        estimate = super().predict(image, focal_px)[0]
        if int(image[0, 0, 0]) == 2:
            return [HandEstimate(**{**estimate.__dict__, "mano_pose": np.zeros(47, dtype=np.float32)})]
        return [estimate]


def test_service_preserves_missing_hand_as_nan_and_mask():
    frames = [DecodedFrame(0, 0.0, np.zeros((8, 8, 3), np.uint8)), DecodedFrame(1, .1, np.ones((8, 8, 3), np.uint8))]
    result = process_frames(frames, Calibrator(), Pose())
    assert not result.detected[0, 0]
    assert np.isnan(result.keypoints_3d[0, 0]).all()
    assert result.detected[1, 0]
    assert result.keypoints_3d[1, 0, 0, 2] == 2.0
    assert result.mano_pose is None and result.mano_shape is None


def test_service_applies_camera_translation_once_to_vertices():
    frame = DecodedFrame(0, 0.0, np.ones((8, 8, 3), np.uint8))
    result = process_frames([frame], Calibrator(), PoseWithVertices())
    np.testing.assert_allclose(result.mano_vertices[0, 0, 0], [1., 2., 5.])


def test_service_cancellation_is_checked_between_reconstruction_frames():
    frames = [DecodedFrame(0, 0.0, np.ones((8, 8, 3), np.uint8)), DecodedFrame(1, 0.1, np.ones((8, 8, 3), np.uint8))]
    with np.testing.assert_raises(InterruptedError):
        process_frames(frames, Calibrator(), Pose(), progress=lambda *_: None, cancelled=lambda: True)


def test_service_preserves_mano_parameters_and_calibration_trace(tmp_path):
    frames = [
        DecodedFrame(11, 0.11, np.full((8, 8, 3), 4, np.uint8)),
        DecodedFrame(19, 0.19, np.full((8, 8, 3), 5, np.uint8)),
        DecodedFrame(29, 0.29, np.full((8, 8, 3), 6, np.uint8)),
        DecodedFrame(37, 0.37, np.full((8, 8, 3), 7, np.uint8)),
        DecodedFrame(47, 0.47, np.full((8, 8, 3), 8, np.uint8)),
    ]
    result = process_frames(frames, CalibratorByPixel(), PoseWithMano(), "uniform", 3, "median")

    assert result.mano_pose.shape == (5, 2, 48)
    assert result.mano_shape.shape == (5, 2, 10)
    assert result.mano_pose.dtype == np.float32
    assert result.mano_shape.dtype == np.float32
    np.testing.assert_array_equal(result.mano_pose[0, 0], np.arange(48, dtype=np.float32) + 4)
    np.testing.assert_array_equal(result.mano_shape[0, 0], np.arange(10, dtype=np.float32) + 40)
    assert np.isnan(result.mano_pose[0, 1]).all()
    assert np.isnan(result.mano_shape[0, 1]).all()
    assert result.calibration_metadata == {
        "sampling_mode": "uniform", "requested_sample_count": 3, "actual_sample_count": 3,
        "aggregation": "median", "frame_indices": [11, 29, 47], "timestamps_s": [0.11, 0.29, 0.47],
        "focal_px": [400.0, 600.0, 800.0], "aggregated_focal_px": 600.0,
    }
    np.testing.assert_allclose(result.keypoints_3d[0, 0, 0], [4, 4, 6])
    np.testing.assert_allclose(result.mano_vertices[0, 0, 0], [4, 5, 8])
    np.testing.assert_allclose(result.camera_translation[0, 0], [0, 0, 2])
    assert result.video_focal_px == 600.0

    npz_path, run_path = process_and_save(
        frames, CalibratorByPixel(), PoseWithMano(), tmp_path,
        {"sampling_mode": "uniform", "sample_count": 3, "aggregation": "median",
         "model_selection": {"calibration_model_id": "anycalib_original", "preset_id": "original"}},
    )
    with np.load(npz_path) as saved:
        assert {"mano_pose", "mano_shape"}.issubset(saved.files)
        assert int(saved["schema_version"]) == 1
        np.testing.assert_array_equal(saved["mano_pose"], result.mano_pose)
        np.testing.assert_array_equal(saved["mano_shape"], result.mano_shape)
    provenance = json.loads(run_path.read_text(encoding="utf-8"))
    assert provenance["calibration"] == result.calibration_metadata
    assert provenance["applied_camera_parameters"]["focal_px"] == result.video_focal_px
    assert provenance["model_selection"]["calibration_model_id"] == "anycalib_original"


@pytest.mark.parametrize(
    ("mode", "count", "aggregation", "expected_indices", "expected_focals", "expected_final"),
    [
        ("uniform", 8, "mean", [101, 103, 107], [400.0, 500.0, 600.0], 500.0),
        ("all", 8, "median", [101, 103, 107], [400.0, 500.0, 600.0], 500.0),
    ],
)
def test_service_calibration_trace_preserves_short_and_all_frame_selection(mode, count, aggregation, expected_indices, expected_focals, expected_final):
    frames = [
        DecodedFrame(101, 1.01, np.full((8, 8, 3), 4, np.uint8)),
        DecodedFrame(103, 1.03, np.full((8, 8, 3), 5, np.uint8)),
        DecodedFrame(107, 1.07, np.full((8, 8, 3), 6, np.uint8)),
    ]
    result = process_frames(frames, CalibratorByPixel(), Pose(), mode, count, aggregation)
    trace = result.calibration_metadata
    assert trace["requested_sample_count"] == 8
    assert trace["actual_sample_count"] == 3
    assert trace["sampling_mode"] == mode
    assert trace["aggregation"] == aggregation
    assert trace["frame_indices"] == expected_indices
    assert trace["timestamps_s"] == [1.01, 1.03, 1.07]
    assert trace["focal_px"] == expected_focals
    assert trace["aggregated_focal_px"] == expected_final


def test_service_rejects_changed_mano_parameter_shape():
    frames = [
        DecodedFrame(0, 0.0, np.full((8, 8, 3), 1, np.uint8)),
        DecodedFrame(1, 0.1, np.full((8, 8, 3), 2, np.uint8)),
    ]
    with pytest.raises(ValueError, match="mano_pose shape changed"):
        process_frames(frames, CalibratorByPixel(), PoseWithInconsistentMano())
