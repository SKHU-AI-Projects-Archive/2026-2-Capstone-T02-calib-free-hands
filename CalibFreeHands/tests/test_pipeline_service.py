import numpy as np

from calibfreehands.media.ffmpeg import DecodedFrame
from calibfreehands.pipeline.service import HandEstimate, process_frames


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


def test_service_preserves_missing_hand_as_nan_and_mask():
    frames = [DecodedFrame(0, 0.0, np.zeros((8, 8, 3), np.uint8)), DecodedFrame(1, .1, np.ones((8, 8, 3), np.uint8))]
    result = process_frames(frames, Calibrator(), Pose())
    assert not result.detected[0, 0]
    assert np.isnan(result.keypoints_3d[0, 0]).all()
    assert result.detected[1, 0]
    assert result.keypoints_3d[1, 0, 0, 2] == 2.0


def test_service_applies_camera_translation_once_to_vertices():
    frame = DecodedFrame(0, 0.0, np.ones((8, 8, 3), np.uint8))
    result = process_frames([frame], Calibrator(), PoseWithVertices())
    np.testing.assert_allclose(result.mano_vertices[0, 0, 0], [1., 2., 5.])


def test_service_cancellation_is_checked_between_reconstruction_frames():
    frames = [DecodedFrame(0, 0.0, np.ones((8, 8, 3), np.uint8)), DecodedFrame(1, 0.1, np.ones((8, 8, 3), np.uint8))]
    with np.testing.assert_raises(InterruptedError):
        process_frames(frames, Calibrator(), Pose(), progress=lambda *_: None, cancelled=lambda: True)
