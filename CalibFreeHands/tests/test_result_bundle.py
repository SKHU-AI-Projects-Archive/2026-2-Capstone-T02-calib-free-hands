from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

from calibfreehands.pipeline import CanonicalResult, save_result
from calibfreehands.pipeline.result_bundle import load_result_bundle, write_result_bundle


def _result(tmp_path):
    points2d = np.full((2, 2, 21, 2), np.nan, np.float32)
    points3d = np.full((2, 2, 21, 3), np.nan, np.float32)
    detected = np.array([[True, False], [False, True]])
    points2d[0, 0] = np.arange(42, dtype=np.float32).reshape(21, 2)
    points2d[1, 1] = 3
    points3d[0, 0] = 4
    points3d[1, 1] = 5
    result = CanonicalResult(
        np.array([2, 5]), np.array([0.0, 0.2]), points2d, points3d, detected,
        np.array([600.0, 600.0]), (64, 48),
        mano_vertices=np.full((2, 2, 3, 3), np.nan, np.float32),
        mano_faces=np.array([[[0, 1, 2]], [[0, 2, 1]]], np.int32),
        confidence=np.array([[0.9, np.nan], [np.nan, 0.8]], np.float32),
        video_focal_px=600.0,
    )
    npz, run = save_result(result, tmp_path / "internal", {"coordinate_system": "camera_space"})
    metadata = {
        "application_version": "0.1.0", "source_media_name": "clip.mp4",
        "source_media_path": str(tmp_path / "clip.mp4"), "source_media_sha256": "a" * 64,
        "calibration_model_id": "anycalib_original", "calibration_display_name": "Original AnyCalib",
        "reconstruction_model_id": "wilor_original", "reconstruction_display_name": "Original WiLoR",
        "coordinate_system": "camera_space", "unit": "metre", "focal_px": 600.0,
        "sampling_mode": "uniform", "sample_count": 8, "aggregation": "median",
        "principal_point_applied": False, "distortion_applied": False,
        "run_provenance": json.loads(run.read_text(encoding="utf-8")),
    }
    return result, npz, metadata


def test_bundle_round_trip_preserves_canonical_arrays_and_metadata(tmp_path):
    expected, npz, metadata = _result(tmp_path)
    path = tmp_path / "clip.cfh"
    write_result_bundle(npz, metadata, path)

    with zipfile.ZipFile(path) as archive:
        assert set(archive.namelist()) == {"result.npz", "metadata.json"}
        with np.load(io.BytesIO(archive.read("result.npz")), allow_pickle=False) as arrays:
            for name in arrays.files:
                actual = arrays[name]
                target = getattr(expected, name, None)
                if name == "image_size":
                    target = np.asarray(expected.image_size, np.int32)
                if name == "schema_version":
                    target = np.asarray(expected.schema_version, np.int32)
                if name == "video_focal_px":
                    target = np.asarray(expected.video_focal_px, np.float32)
                np.testing.assert_array_equal(actual, target)
    loaded = load_result_bundle(path)
    assert loaded.metadata == {"bundle_schema_version": 1, **metadata}
    np.testing.assert_array_equal(loaded.result.detected, expected.detected)
    np.testing.assert_array_equal(loaded.result.keypoints_3d, expected.keypoints_3d)
    assert loaded.result.mano_faces.shape == (2, 1, 3)


@pytest.mark.parametrize("contents", [b"not a zip", None])
def test_bundle_rejects_invalid_zip_and_missing_required_member(tmp_path, contents):
    path = tmp_path / "broken.cfh"
    if contents is not None:
        path.write_bytes(contents)
    else:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("result.npz", b"not npz")
    with pytest.raises(ValueError):
        load_result_bundle(path)


def test_bundle_rejects_extra_members_without_extracting_them(tmp_path):
    _canonical, npz, metadata = _result(tmp_path)
    path = tmp_path / "extra.cfh"
    with zipfile.ZipFile(path, "w") as archive:
        archive.write(npz, "result.npz")
        archive.writestr("metadata.json", json.dumps({"bundle_schema_version": 1, **metadata}))
        archive.writestr("../outside.txt", "must never be extracted")
    with pytest.raises(ValueError, match="exactly"):
        load_result_bundle(path)
    assert not (tmp_path.parent / "outside.txt").exists()


def test_bundle_rejects_unsupported_schema_and_malformed_metadata(tmp_path):
    path = tmp_path / "broken.cfh"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("result.npz", b"placeholder")
        archive.writestr("metadata.json", '{"bundle_schema_version": 99}')
    with pytest.raises(ValueError, match="schema"):
        load_result_bundle(path)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("result.npz", b"placeholder")
        archive.writestr("metadata.json", "{")
    with pytest.raises(ValueError, match="metadata"):
        load_result_bundle(path)


def test_bundle_rejects_raw_pose_npz_and_malformed_source_identity(tmp_path):
    path = tmp_path / "raw.cfh"
    raw = io.BytesIO()
    np.savez(raw, keypoints_3d_relative=np.zeros((1, 2, 21, 3)))
    _, _, metadata = _result(tmp_path)
    metadata["bundle_schema_version"] = 1
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("result.npz", raw.getvalue())
        archive.writestr("metadata.json", json.dumps(metadata))
    with pytest.raises(ValueError, match="raw-pose"):
        load_result_bundle(path)

    expected, npz, metadata = _result(tmp_path)
    metadata["source_media_sha256"] = "not-a-sha"
    with pytest.raises(ValueError, match="SHA256"):
        write_result_bundle(npz, metadata, tmp_path / "invalid.cfh")


def test_result_filename_sanitizes_model_and_source_names(tmp_path):
    from calibfreehands.export.paths import result_filename

    assert result_filename("CON. ", "Original: Any/Calib", "Original*WiLoR") == (
        "_CON__Original_-Any_Calib__Original_WiLoR.cfh"
    )
    assert result_filename("clip", "AnyCalib", "WiLoR").endswith(".cfh")


def test_result_file_collision_policies_are_file_scoped(tmp_path):
    from calibfreehands.export.paths import choose_output_file

    original = tmp_path / "clip.cfh"
    original.touch()
    assert choose_output_file(tmp_path, original.name, "auto_rename").name == "clip_1.cfh"
    assert choose_output_file(tmp_path, original.name, "skip") is None
    assert choose_output_file(tmp_path, original.name, "overwrite") == original
    with pytest.raises(FileExistsError, match="confirmation"):
        choose_output_file(tmp_path, original.name, "ask")


def test_failed_atomic_write_leaves_no_partial_bundle_or_temporary_file(tmp_path):
    _canonical, npz, metadata = _result(tmp_path)
    metadata["unexpected"] = object()
    output = tmp_path / "atomic.cfh"
    with pytest.raises(TypeError):
        write_result_bundle(npz, metadata, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".atomic.cfh.*.tmp"))

