import json

import numpy as np

from calibfreehands.calibration import aggregate_focal, sample_indices
from calibfreehands.media import discover_media
from calibfreehands.pipeline import CanonicalResult, save_result


def test_folder_discovery_is_non_recursive(tmp_path):
    (tmp_path / "a.MP4").write_bytes(b"")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.mp4").write_bytes(b"")
    assert [p.name for p in discover_media(tmp_path)] == ["a.MP4"]


def test_sampling_and_aggregation():
    assert sample_indices(10, "uniform", 4).tolist() == [0, 3, 6, 9]
    assert aggregate_focal([100.0, np.nan, 200.0], "median") == 150.0


def test_result_round_trip(tmp_path):
    detected = np.array([[False, False], [True, False]])
    keypoints_2d = np.full((2, 2, 21, 2), np.nan)
    keypoints_3d = np.full((2, 2, 21, 3), np.nan)
    keypoints_2d[1, 0] = 0
    keypoints_3d[1, 0] = 0
    result = CanonicalResult(
        np.array([0, 1]), np.array([0.0, 0.04]),
        keypoints_2d, keypoints_3d, detected, np.array([500.0, 501.0]), (640, 480),
        video_focal_px=500.5, media_metadata={"format": "synthetic"},
    )
    npz, run = save_result(result, tmp_path, {"schema_version": 1, "coordinate_system": "camera"})
    loaded = np.load(npz)
    assert int(loaded["schema_version"]) == 1
    assert float(loaded["video_focal_px"]) == 500.5
    assert np.isnan(loaded["keypoints_3d"][0]).all()
    assert np.isfinite(loaded["keypoints_3d"][1, 0]).all()
    assert loaded["detected"].tolist() == [[False, False], [True, False]]
    assert json.loads(run.read_text(encoding="utf-8"))["media_metadata"]["format"] == "synthetic"
