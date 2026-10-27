import json

import numpy as np
import pytest

from calibfreehands.app.configuration import ModelDefinition, load_configuration
from calibfreehands.calibration import ReferenceFocalBackend
from calibfreehands.media.ffmpeg import DecodedFrame
from calibfreehands.pipeline.comparison import (
    PipelineConfiguration,
    RawHandEstimate,
    load_comparison_session,
    process_comparison_session,
)


class CountingCalibration:
    name = "synthetic calibration"

    def __init__(self, focal, calls):
        self.focal = focal
        self.calls = calls

    def estimate_focal(self, image_rgb):
        self.calls.append("calibration")
        return self.focal


class CountingPose:
    name = "synthetic pose"
    checkpoint = None
    mano_paths = {}

    def __init__(self, source_focal, calls):
        self.source_focal = source_focal
        self.calls = calls

    def preflight(self):
        return {}

    def predict_raw(self, image_rgb):
        self.calls.append("pose")
        points_2d = np.full((21, 2), 3, np.float32)
        points_3d = np.zeros((21, 3), np.float32)
        points_3d[0] = [1, 2, 2]
        return [RawHandEstimate(
            is_right=True,
            keypoints_2d=points_2d,
            keypoints_3d_relative=points_3d,
            camera_translation_source=np.array([4, 5, 10], np.float32),
            source_focal_px=self.source_focal,
            vertices_relative=np.array([[1, 2, 2], [2, 3, 4]], np.float32),
            bbox=np.array([0, 0, 7, 7], np.float32),
            confidence=0.9,
            mano_pose=np.zeros(3, np.float32),
            mano_shape=np.zeros(2, np.float32),
            mano_faces=np.array([[0, 1, 0]], np.int32),
        )]


def _frames():
    return [
        DecodedFrame(4, 0.4, np.zeros((8, 8, 3), np.uint8)),
        DecodedFrame(9, 0.9, np.ones((8, 8, 3), np.uint8)),
    ]


def _configurations():
    return [
        PipelineConfiguration("cfg-a", "Baseline", "anycalib_original", "wilor_original"),
        PipelineConfiguration("cfg-b", "Proposed", "anycalib_original", "wilor_original"),
    ]


def _run(tmp_path, configs=None, cancelled=None):
    app_config = load_configuration()
    app_config.calibration_models["reference"] = ModelDefinition(
        "reference", "Checkerboard Reference", "reference_focal", True
    )
    app_config.reconstruction_models["wilor_alt"] = ModelDefinition(
        "wilor_alt", "Synthetic WiLoR variant", "wilor", True
    )
    calibration_calls = []
    pose_calls = {}

    def make_calibration(model_id, reference_focal_px):
        if model_id == "reference":
            return ReferenceFocalBackend(reference_focal_px)
        return CountingCalibration(500, calibration_calls)

    def make_pose(model_id):
        calls = pose_calls.setdefault(model_id, [])
        return CountingPose(500, calls)

    session = process_comparison_session(
        _frames(), configs or _configurations(), app_config, tmp_path, tmp_path / "session",
        {"input_path": "input.mp4", "sampling_mode": "uniform", "sample_count": 1,
         "aggregation": "median", "preserve_audio": False},
        calibration_factory=make_calibration,
        pose_factory=make_pose,
        cancelled=cancelled,
    )
    return session, calibration_calls, pose_calls


def test_identical_calibration_and_pose_work_is_reused_and_session_round_trips(tmp_path):
    session, calibration_calls, pose_calls = _run(tmp_path)

    assert len(calibration_calls) == 1
    assert len(pose_calls["wilor_original"]) == 2
    assert session.state == "completed"
    assert len(session.final_results) == 2
    restored = load_comparison_session(tmp_path / "session" / "session.json")
    assert [item.id for item in restored.configurations] == ["cfg-a", "cfg-b"]
    assert restored.final_results == session.final_results
    manifest = json.loads((tmp_path / "session" / "session.json").read_text(encoding="utf-8"))
    assert "keypoints_3d" not in json.dumps(manifest)
    assert manifest["raw_pose_results"]["wilor_original"]["path"].startswith("pose/")
    assert manifest["raw_pose_results"]["wilor_original"]["path"].endswith(".npz")


def test_shared_raw_pose_with_different_focals_produces_distinct_camera_space_results(tmp_path):
    configs = [
        PipelineConfiguration("base", "Baseline", "anycalib_original", "wilor_original"),
        PipelineConfiguration("reference", "Reference", "reference", "wilor_original", 1000.0),
    ]
    session, calibration_calls, pose_calls = _run(tmp_path, configs)
    baseline_path = tmp_path / "session" / session.final_results["base"]["path"]
    reference_path = tmp_path / "session" / session.final_results["reference"]["path"]

    assert len(pose_calls["wilor_original"]) == 2
    assert len(calibration_calls) == 1
    with np.load(baseline_path) as baseline, np.load(reference_path) as reference:
        assert baseline["keypoints_3d"][0, 1, 0, 0] == reference["keypoints_3d"][0, 1, 0, 0]
        assert baseline["keypoints_3d"][0, 1, 0, 1] == reference["keypoints_3d"][0, 1, 0, 1]
        assert baseline["keypoints_3d"][0, 1, 0, 2] == 12
        assert reference["keypoints_3d"][0, 1, 0, 2] == 22
        assert baseline["mano_vertices"][0, 1, 0, 2] == 12
        assert reference["mano_vertices"][0, 1, 0, 2] == 22


def test_same_calibration_with_different_pose_models_calibrates_once(tmp_path):
    configs = [
        PipelineConfiguration("original", "Original", "anycalib_original", "wilor_original"),
        PipelineConfiguration("variant", "Variant", "anycalib_original", "wilor_alt"),
    ]
    session, calibration_calls, pose_calls = _run(tmp_path, configs)

    assert len(calibration_calls) == 1
    assert set(pose_calls) == {"wilor_original", "wilor_alt"}
    assert all(len(calls) == 2 for calls in pose_calls.values())
    assert session.calibration_results.keys() == {"anycalib_original"}


def test_cancellation_writes_cancelled_manifest_not_completed(tmp_path):
    with pytest.raises(InterruptedError):
        _run(tmp_path, cancelled=lambda: True)

    session = load_comparison_session(tmp_path / "session" / "session.json")
    assert session.state == "cancelled"
    assert not session.final_results


@pytest.mark.parametrize(
    "configuration, error",
    [
        (PipelineConfiguration("bad", "Bad", "missing", "wilor_original"), KeyError),
        (PipelineConfiguration("disabled", "Disabled", "handcalib", "wilor_original"), RuntimeError),
    ],
)
def test_unknown_or_unavailable_backend_is_rejected_before_session_start(tmp_path, configuration, error):
    app_config = load_configuration()
    with pytest.raises(error):
        process_comparison_session(
            _frames(), [configuration], app_config, tmp_path, tmp_path / "session", {},
        )
    assert not (tmp_path / "session" / "session.json").exists()


def test_empty_or_duplicate_configuration_ids_are_rejected(tmp_path):
    app_config = load_configuration()
    with pytest.raises(ValueError, match="at least one"):
        process_comparison_session([], [], app_config, tmp_path, tmp_path / "empty", {})
    duplicate = [
        PipelineConfiguration("same", "One", "anycalib_original", "wilor_original"),
        PipelineConfiguration("same", "Two", "anycalib_original", "wilor_original"),
    ]
    with pytest.raises(ValueError, match="unique"):
        process_comparison_session(_frames(), duplicate, app_config, tmp_path, tmp_path / "duplicate", {})
