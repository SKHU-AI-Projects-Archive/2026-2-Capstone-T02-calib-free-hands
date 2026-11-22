from fractions import Fraction

import numpy as np
import pytest

from calibfreehands.media.ffmpeg import DecodedFrame
from calibfreehands.pipeline.service import HandEstimate
from capstone.pipeline import run_video
from capstone.video import Recording


def test_pts_boundary_and_focal_passed_only_to_following_frames(tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"test")
    times = tuple(Fraction(value) for value in ("12.032", "15.157", "32.031", "32.032", "32.045", "33.252", "36.032"))
    images = [np.full((6, 8, 3), i, np.uint8) for i in range(len(times))]
    recording = Recording(source, (8, 6), times, lambda: iter(
        DecodedFrame(i, float(t), image) for i, (t, image) in enumerate(zip(times, images))
    ))
    calibrated = []
    focal_calls = []

    class Calibration:
        name = "Mock focal"
        def add(self, image):
            calibrated.append(int(image[0, 0, 0]))
        def estimate(self):
            return 1725.25

    class Pose:
        name = "Mock pose"
        def predict(self, image, focal_px):
            focal_calls.append((int(image[0, 0, 0]), focal_px))
            return [HandEstimate(
                True, np.zeros((21, 2)), np.ones((21, 3)), np.array([0, 0, 2.0]),
                None, np.zeros(4), 1.0,
            )]

    result, report = run_video(recording, Calibration(), Pose(), work_seconds=3)
    assert calibrated == [0, 1, 2]
    assert focal_calls == [(3, 1725.25), (4, 1725.25), (5, 1725.25)]
    np.testing.assert_allclose(result.timestamps, [20.0, 20.013, 21.22])
    assert result.detected[:, 1].all()
    assert report["estimator"] == "Mock focal"
    assert report["calibration_range_s"] == [0.0, 20.0]
    assert report["work_range_s"] == [20.0, 23.0]
    assert report["focal_px"] == report["pose_input_focal_px"] == 1725.25


def test_invalid_focal_stops_before_pose(tmp_path):
    source = tmp_path / "clip.mp4"
    source.touch()
    image = np.zeros((4, 6, 3), np.uint8)
    times = tuple(Fraction(value) for value in ("0", "10", "19.99", "20"))
    recording = Recording(source, (6, 4), times, lambda: iter(
        DecodedFrame(i, float(t), image) for i, t in enumerate(times)
    ))

    class Calibration:
        name = "Mock focal"
        def add(self, image):
            pass
        def estimate(self):
            return -1

    with pytest.raises(ValueError, match="focal"):
        run_video(recording, Calibration(), object())
