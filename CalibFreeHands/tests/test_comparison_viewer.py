import json

import numpy as np
import pytest
from PySide6.QtWidgets import QCheckBox

from calibfreehands.pipeline import CanonicalResult, save_result
from calibfreehands.pipeline.result_bundle import write_result_bundle


def _session(tmp_path, count=3, mesh=False):
    configurations, final_results = [], {}
    for index in range(count):
        config_id = f"cfg-{index}"
        points2d = np.zeros((3, 2, 21, 2), np.float32)
        points3d = np.zeros((3, 2, 21, 3), np.float32)
        points3d[:, :, :, 0] = index
        points3d[:, :, :, 2] = 2 + index
        detected = np.ones((3, 2), bool)
        confidence = np.full((3, 2), 0.25 + index * 0.25, np.float32)
        translation = np.zeros((3, 2, 3), np.float32)
        translation[:, :, 2] = 2 + index
        result = CanonicalResult(np.array([0, 1, 2]), np.array([0., .1, .2]), points2d,
                                 points3d, detected, np.full(3, 500.), (32, 24),
                                 mano_vertices=np.zeros((3, 2, 3, 3), np.float32) if mesh else None,
                                 mano_faces=np.array([[[0, 1, 2]], [[0, 2, 1]]], np.int32) if mesh else None,
                                 confidence=confidence, camera_translation=translation)
        result_dir = tmp_path / config_id
        npz, run = save_result(result, result_dir, {"input_path": ""})
        configurations.append({"id": config_id, "name": f"Method {index}",
                               "calibration_model_id": "reference", "reconstruction_model_id": "pose"})
        final_results[config_id] = {"path": npz.relative_to(tmp_path).as_posix(),
                                    "run_json": run.relative_to(tmp_path).as_posix()}
    manifest = {"schema_version": 1, "session_id": "session", "source_media": "",
                "state": "completed", "selected_preset": None, "configurations": configurations,
                "calibration_results": {}, "raw_pose_results": {}, "final_results": final_results,
                "reference_configuration_id": "cfg-0", "provenance": {}}
    path = tmp_path / "session.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_comparison_metrics_and_exports_are_numerically_correct(tmp_path):
    from calibfreehands.visualization.comparison import calculate_reference_metrics, export_metrics_csv, export_metrics_json
    path = _session(tmp_path, 2)
    rows = calculate_reference_metrics(path)
    assert len(rows) == 6
    rows = [row for row in rows if row["hand"] == "left" and row["frame_index"] == 0]
    assert len(rows) == 1
    assert rows[0]["configuration_id"] == "cfg-1"
    assert rows[0]["joint_distance_mean_m"] == pytest.approx(np.sqrt(2))
    assert rows[0]["joint_distance_rmse_m"] == pytest.approx(np.sqrt(2))
    assert rows[0]["depth_error_mean_m"] == pytest.approx(1.0)
    assert rows[0]["translation_distance_m"] == pytest.approx(1.0)
    export_metrics_csv(path, tmp_path / "metrics.csv")
    export_metrics_json(path, tmp_path / "metrics.json")
    assert "joint_distance_mean_m" in (tmp_path / "metrics.csv").read_text(encoding="utf-8")
    assert json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))[0] == rows[0]


def test_comparison_viewer_loads_three_results_and_preserves_frame_across_modes(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer
    viewer = ComparisonViewer(_session(tmp_path))
    assert len(viewer.loaded_results) == 3
    assert viewer.mode == "overlay"
    loaded_identity = viewer.loaded_results[0]
    original_arrays = [loaded.result.keypoints_3d.copy() for loaded in viewer.loaded_results]
    viewer.slider.setValue(2)
    viewer.set_mode("split")
    assert viewer.slider.value() == 2
    assert [view.frame for view in viewer.split_views] == [2, 2, 2]
    viewer.set_mode("overlay")
    assert viewer.slider.value() == 2
    assert viewer.loaded_results[0] is loaded_identity
    for loaded, before in zip(viewer.loaded_results, original_arrays):
        np.testing.assert_array_equal(loaded.result.keypoints_3d, before)
    viewer.close()


def test_overlay_colors_visibility_confidence_and_linked_camera(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer
    viewer = ComparisonViewer(_session(tmp_path))
    assert len(set(viewer.configuration_colors)) == 3
    initial_params = viewer.split_views[0].cameraParams()
    for view in viewer.split_views[1:]:
        assert view.cameraParams()["distance"] == pytest.approx(initial_params["distance"])
        assert view.cameraParams()["azimuth"] == pytest.approx(initial_params["azimuth"])
        assert view.cameraParams()["elevation"] == pytest.approx(initial_params["elevation"])
    viewer.set_mode("overlay")
    alpha_low = viewer.overlay_view.confidence_alpha(0, 0)
    alpha_high = viewer.overlay_view.confidence_alpha(2, 0, viewer.loaded_results[2].result.confidence[2, 0])
    assert alpha_low < alpha_high
    missing = viewer.overlay_view.confidence_alpha(0, 1, confidence=np.nan)
    assert missing == pytest.approx(viewer.overlay_view.missing_confidence_alpha)
    base_color = np.asarray(viewer.overlay_view.joints[0].color)
    np.testing.assert_allclose(base_color[:3], viewer.configuration_colors[0])
    assert base_color[3] == pytest.approx(alpha_low)
    checks = [box for box in viewer.findChildren(QCheckBox) if box.text().startswith("Method")]
    checks[1].setChecked(False)
    assert not viewer.overlay_view.configuration_visible[1]
    viewer.set_mode("split")
    viewer._apply_camera_preset("front_view")
    params = viewer.split_views[0].cameraParams()
    for view in viewer.split_views[1:]:
        assert view.cameraParams()["distance"] == pytest.approx(params["distance"])
        np.testing.assert_allclose(view.cameraParams()["center"].toTuple(), params["center"].toTuple())
    for method in ("side_view", "top_view", "fit_hands", "reset_camera"):
        viewer._apply_camera_preset(method)
        params = viewer.split_views[0].cameraParams()
        for view in viewer.split_views[1:]:
            assert view.cameraParams()["distance"] == pytest.approx(params["distance"])
            assert view.cameraParams()["azimuth"] == pytest.approx(params["azimuth"])
            assert view.cameraParams()["elevation"] == pytest.approx(params["elevation"])
    source_view = viewer.split_views[0]
    source_view.orbit(17, -6)
    source_view.pan(0.1, -0.2, 0.3)
    source_view.opts["distance"] *= 0.8
    viewer._sync_camera(source_view)
    params = source_view.cameraParams()
    for view in viewer.split_views[1:]:
        assert view.cameraParams()["distance"] == pytest.approx(params["distance"])
        assert view.cameraParams()["azimuth"] == pytest.approx(params["azimuth"])
        assert view.cameraParams()["elevation"] == pytest.approx(params["elevation"])
        np.testing.assert_allclose(view.cameraParams()["center"].toTuple(), params["center"].toTuple())
    viewer.close()


def test_comparison_mesh_visibility_remains_exclusive_with_skeleton(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer
    viewer = ComparisonViewer(_session(tmp_path, mesh=True))
    viewer.mesh_checkbox.setChecked(True)
    assert viewer.mesh_checkbox.isChecked()
    assert not viewer.joint_checkbox.isChecked() and not viewer.bone_checkbox.isChecked()
    assert viewer.overlay_view.meshes[0].opts["color"][3] == pytest.approx(0.4)
    viewer.joint_checkbox.setChecked(True)
    assert viewer.joint_checkbox.isChecked() and not viewer.mesh_checkbox.isChecked()
    assert viewer.overlay_view.joints[0].visible()
    assert not viewer.overlay_view.meshes[0].visible()
    viewer.close()


def test_comparison_viewer_decodes_shared_source_media_once(tmp_path, monkeypatch, qapp):
    import calibfreehands.visualization.comparison as comparison
    source = tmp_path / "source.mp4"
    source.touch()
    manifest = _session(tmp_path)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["source_media"] = str(source)
    manifest.write_text(json.dumps(data), encoding="utf-8")
    image = np.zeros((24, 32, 3), np.uint8)
    calls = []
    monkeypatch.setattr(comparison, "probe_media", lambda path: type("Info", (), {"width": 32, "height": 24, "frame_rate": 10})())
    monkeypatch.setattr(comparison, "FFmpegDecoder", lambda: type("Decoder", (), {"decode_rgb": lambda self, *args: calls.append(args[0]) or [type("Frame", (), {"rgb": image})() for _ in range(3)]})())
    viewer = comparison.ComparisonViewer(manifest)
    assert len(calls) == 1
    assert all(loaded.media_frames is viewer.loaded_results[0].media_frames for loaded in viewer.loaded_results)
    viewer.close()


def _bundle(tmp_path, name, model_id, source_name, source_sha, index):
    result = CanonicalResult(
        np.array([0, 1]), np.array([0., .1]), np.zeros((2, 2, 21, 2), np.float32),
        np.full((2, 2, 21, 3), index, np.float32), np.ones((2, 2), bool),
        np.full(2, 500., np.float32), (32, 24), confidence=np.full((2, 2), .8, np.float32),
    )
    npz, _ = save_result(result, tmp_path / f"{name}-internal", {})
    metadata = {
        "application_version": "0.1.0", "source_media_name": source_name,
        "source_media_path": str(tmp_path / source_name), "source_media_sha256": source_sha,
        "calibration_model_id": model_id, "calibration_display_name": name,
        "reconstruction_model_id": "wilor_original", "reconstruction_display_name": "Original WiLoR",
        "configuration_id": model_id, "configuration_name": name,
        "reference_configuration_id": "reference", "coordinate_system": "camera_space", "unit": "metre",
        "focal_px": 500.0, "sampling_mode": "uniform", "sample_count": 8, "aggregation": "median",
        "principal_point_applied": False, "distortion_applied": False, "run_provenance": {},
    }
    path = write_result_bundle(npz, metadata, tmp_path / f"{name}.cfh")
    from calibfreehands.pipeline.result_bundle import load_result_bundle
    return load_result_bundle(path)


def test_comparison_viewer_accepts_same_source_bundles_and_keeps_existing_modes(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer
    bundles = [
        _bundle(tmp_path, "Reference", "reference", "source-a.mp4", "a" * 64, 1),
        _bundle(tmp_path, "Proposed", "proposed", "different-name.mp4", "a" * 64, 2),
    ]
    viewer = ComparisonViewer(bundles)
    assert len(viewer.loaded_results) == 2
    assert viewer.mode == "overlay"
    assert viewer.export_csv_button.isEnabled()
    viewer.set_mode("split")
    viewer.slider.setValue(1)
    assert viewer.slider.value() == 1
    assert all(item.result.keypoints_3d[1, 0, 0, 0] == index for index, item in enumerate(viewer.loaded_results, 1))
    viewer.close()


def test_comparison_viewer_rejects_different_source_hashes(tmp_path):
    from calibfreehands.visualization.comparison import ComparisonViewer
    bundles = [
        _bundle(tmp_path, "Reference", "reference", "same-name.mp4", "a" * 64, 1),
        _bundle(tmp_path, "Proposed", "proposed", "same-name.mp4", "b" * 64, 2),
    ]
    with pytest.raises(ValueError, match="different source media"):
        ComparisonViewer(bundles)
