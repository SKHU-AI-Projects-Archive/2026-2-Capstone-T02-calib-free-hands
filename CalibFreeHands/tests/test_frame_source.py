import shutil
import subprocess
import types

import numpy as np
import pytest

from calibfreehands.media.ffmpeg import DecodedFrame
from calibfreehands.media.frames import ListFrameSource, VideoFrameSource, as_frame_source, frame_meta
from calibfreehands.pipeline.comparison import RawPoseResult
from test_comparison import _configurations, _run

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                reason="ffmpeg/ffprobe not installed")


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10", "-t", "3",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def test_meta_data_is_known_without_decoding_frames_into_memory(video):
    source = VideoFrameSource(video, 64, 48, 10.0)
    assert len(source) == 30 and source.image_size == (64, 48)
    assert source.timestamps[0] == pytest.approx(0.0) and source.timestamps[1] == pytest.approx(0.1)
    assert not hasattr(source, "_frames")  # nothing is kept: every pass decodes again


def test_every_pass_decodes_again(video):
    source = VideoFrameSource(video, 64, 48, 10.0)
    for _ in range(2):
        frames = list(source.iter_frames())
        assert len(frames) == 30 and frames[0].rgb.shape == (48, 64, 3)
        assert [frame.index for frame in frames] == list(range(30))


def test_list_and_video_sources_give_the_same_meta_data(video):
    decoded = list(VideoFrameSource(video, 64, 48, 10.0).iter_frames())
    list_meta, video_meta = frame_meta(decoded), frame_meta(VideoFrameSource(video, 64, 48, 10.0))
    assert np.array_equal(list_meta[0], video_meta[0]) and np.allclose(list_meta[1], video_meta[1])
    assert list_meta[2] == video_meta[2] == (64, 48)
    assert as_frame_source(as_frame_source(decoded)).__class__ is ListFrameSource


def test_pipeline_result_is_the_same_for_a_stream_and_a_list(video, tmp_path):
    source = VideoFrameSource(video, 64, 48, 10.0)
    in_memory = list(source.iter_frames())
    settings = {"input_path": "input.mp4", "sampling_mode": "uniform", "sample_count": 4,
                "aggregation": "median", "preserve_audio": False}
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    list_session, list_calibration, list_pose = _run(tmp_path / "a", settings=settings, frames=in_memory)
    stream_session, stream_calibration, stream_pose = _run(tmp_path / "b", settings=settings, frames=source)

    assert len(stream_calibration) == len(list_calibration) == 4  # only the sampled frames are estimated
    assert {k: len(v) for k, v in stream_pose.items()} == {k: len(v) for k, v in list_pose.items()}
    assert next(iter(stream_pose.values())) == ["pose"] * 30
    list_meta = next(iter(list_session.calibration_results.values()))["metadata"]
    stream_meta = next(iter(stream_session.calibration_results.values()))["metadata"]
    assert stream_meta["frame_indices"] == list_meta["frame_indices"] == [0, 9, 19, 29]
    assert stream_meta["timestamps_s"] == pytest.approx(list_meta["timestamps_s"])
    assert [r["focal_px"] for r in stream_session.final_results.values()] == \
           [r["focal_px"] for r in list_session.final_results.values()]


def test_all_frames_mode_streams_every_frame_through_the_estimator(video, tmp_path):
    settings = {"input_path": "input.mp4", "sampling_mode": "all", "sample_count": 8,
                "aggregation": "median", "preserve_audio": False}
    _, calibration_calls, _ = _run(tmp_path, settings=settings, frames=VideoFrameSource(video, 64, 48, 10.0))
    assert len(calibration_calls) == 30


def test_raw_pose_result_accepts_a_source_and_checks_the_count(video):
    source = VideoFrameSource(video, 64, 48, 10.0)
    with pytest.raises(ValueError):
        RawPoseResult.from_predictions(source, [[]] * 3)
    result = RawPoseResult.from_predictions(source, [[] for _ in range(len(source))])
    assert result.image_size == (64, 48) and len(result.frame_indices) == 30
