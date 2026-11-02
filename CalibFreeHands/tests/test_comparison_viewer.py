import json

import numpy as np
import pytest
from PySide6.QtWidgets import QCheckBox

from calibfreehands.pipeline import CanonicalResult, save_result
from calibfreehands.pipeline.result_bundle import write_result_bundle
from calibfreehands.joint_schema import JOINT_NAMES


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
    csv_text = (tmp_path / "metrics.csv").read_text(encoding="utf-8")
    assert "metric_id" in csv_text and "protocol_version" in csv_text and "joint_index" in csv_text
    exported = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    mpjpe = next(row for row in exported if row["metric_id"] == "mpjpe" and row["frame_index"] == 0 and row["hand"] == "left")
    assert mpjpe["value"] == pytest.approx(np.sqrt(2) * 1000)
    assert mpjpe["unit"] == "mm" and mpjpe["valid_status"] == "READY"
    assert any(row["metric_id"] == "f@5" and row["valid_status"] == "PROTOCOL_PENDING" for row in exported)
    assert all(row["reference_result_id"] == "cfg-0" for row in exported)


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
    from calibfreehands.visualization.comparison import _hand_variant
    np.testing.assert_allclose(base_color[:3], _hand_variant(viewer.configuration_colors[0], 0))
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
    from calibfreehands.media.provider import MediaFrameProvider
    source = tmp_path / "source.mp4"
    source.touch()
    manifest = _session(tmp_path)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["source_media"] = str(source)
    manifest.write_text(json.dumps(data), encoding="utf-8")
    calls = []
    provider = MediaFrameProvider(source, frame_loader=lambda index: calls.append(index) or np.zeros((24, 32, 3), np.uint8))
    monkeypatch.setattr(comparison, "shared_media_provider", lambda *args: (provider, "available"))
    viewer = comparison.ComparisonViewer(manifest)
    assert calls in ([], [0])
    assert all(loaded.media_frames is viewer.loaded_results[0].media_frames for loaded in viewer.loaded_results)
    viewer.source_overlay.set_frame(1)
    deadline = __import__("time").monotonic() + 2
    while (not calls or calls[-1] != 1) and __import__("time").monotonic() < deadline:
        qapp.processEvents()
        __import__("time").sleep(.005)
    assert calls[-1] == 1
    provider.close()
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
    assert viewer.split_source_overlay in viewer.split_widget.findChildren(type(viewer.split_source_overlay))
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


def test_dashboard_reference_metrics_source_grid_and_native_visibility(tmp_path, qapp):
    from PySide6.QtWidgets import QCheckBox, QComboBox
    from calibfreehands.visualization.comparison import ComparisonViewer
    viewer = ComparisonViewer(_session(tmp_path, 5))
    assert isinstance(viewer.reference_selector, QComboBox)
    assert viewer.reference_selector.currentData() == "cfg-0"
    assert viewer.split_source_overlay in viewer.split_widget.findChildren(type(viewer.split_source_overlay))
    assert len(viewer.split_views) == 5
    assert viewer.split_columns == 3 and viewer.split_rows == 2
    assert not hasattr(viewer, "metric_table")
    candidate = viewer.split_panels[1]
    assert "Wrist Error" in candidate.metric_rows[("cfg-1", "wrist_error", "left")].text()
    assert viewer.overlay_panel.reference_badges["cfg-0"].text() == "REFERENCE"
    assert ("cfg-1", "joint_error_median", "left") not in viewer.overlay_panel.metric_rows
    viewer.metric_checks["joint_error_median"].setChecked(True)
    assert ("cfg-1", "joint_error_median", "left") in viewer.overlay_panel.metric_rows
    viewer.metric_checks["joint_error_median"].setChecked(False)
    assert ("cfg-1", "joint_error_median", "left") not in viewer.overlay_panel.metric_rows
    viewer.reference_selector.setCurrentIndex(1)
    assert viewer.reference_selector.currentData() == "cfg-1"
    viewer.slider.setValue(2)
    assert viewer.status.text().startswith("Frame 3")
    assert viewer.split_panels[1].reference_badges["cfg-1"].text() == "REFERENCE"
    assert all(box.styleSheet() == "" for box in viewer.visibility_checks)
    assert all(viewer.color_buttons)
    viewer.close()


def test_dashboard_metric_selection_error_vectors_and_confidence_are_runtime_only(tmp_path, qapp, monkeypatch):
    import calibfreehands.visualization.comparison as comparison
    from calibfreehands.visualization.comparison import ComparisonViewer
    viewer = ComparisonViewer(_session(tmp_path, 2))
    monkeypatch.setattr(comparison, "compare_results", lambda *args: pytest.fail("metrics recomputed during frame change"))
    assert viewer.selected_metric_ids == {"wrist_error", "mpjpe", "wrist_depth_error"}
    viewer.metric_checks["mpvpe"].setChecked(True)
    assert "mpvpe" in viewer.selected_metric_ids
    viewer.metric_checks["per_joint_error"].setChecked(True)
    panel = viewer.split_panels[1]
    assert ("cfg-1", "per_joint_error", "left") in panel.details_buttons
    panel.details_buttons[("cfg-1", "per_joint_error", "left")].click()
    from PySide6.QtWidgets import QTableWidget
    tables = viewer.findChildren(QTableWidget)
    assert tables and tables[-1].rowCount() == 21
    assert tables[-1].columnCount() == 3
    assert float(tables[-1].item(0, 1).text()) == pytest.approx(np.sqrt(2) * 1000, abs=0.01)
    viewer.slider.setValue(1)
    assert viewer.status.text().startswith("Frame 2")
    viewer.confidence_numeric_checkbox.setChecked(True)
    viewer.confidence_bar_checkbox.setChecked(True)
    assert "0.50" in panel.confidence_rows["cfg-1", "left"].text()
    assert set(panel.confidence_rows) == {("cfg-1", "left"), ("cfg-1", "right")}
    viewer.confidence_opacity_checkbox.setChecked(False)
    assert viewer.overlay_view.confidence_alpha(0, 0) < 1
    assert not viewer.overlay_view.confidence_opacity_enabled
    viewer.confidence_color_mode.setCurrentIndex(1)
    assert viewer.overlay_view.confidence_color_mode == "confidence"
    viewer.confidence_opacity_checkbox.setChecked(True)
    assert viewer.confidence_numeric_checkbox and viewer.confidence_bar_checkbox
    assert viewer.confidence_color_mode.currentData() in ("configuration", "confidence")
    assert viewer.error_vectors_checkbox
    assert viewer.error_candidate_selector.count() == 1
    before = [loaded.result.keypoints_3d.copy() for loaded in viewer.loaded_results]
    viewer._apply_configuration_color(1, (0.1, 0.2, 0.3))
    for loaded, array in zip(viewer.loaded_results, before):
        np.testing.assert_array_equal(loaded.result.keypoints_3d, array)
    viewer.error_vectors_checkbox.setChecked(True)
    assert any(item.visible() for item in viewer.overlay_view.error_items)
    viewer.close()


def test_dashboard_checkerboard_default_metric_export_reference_and_detach_sync(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer, export_metrics_json
    path = _session(tmp_path, 3)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["configurations"][2]["calibration_model_id"] = "checkerboard_reference"
    manifest["reference_configuration_id"] = "cfg-1"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    viewer = ComparisonViewer(path)
    assert viewer.reference_id == "cfg-2"
    export_metrics_json(path, tmp_path / "selected.json", viewer.reference_id)
    rows = json.loads((tmp_path / "selected.json").read_text(encoding="utf-8"))
    assert rows and {row["reference_result_id"] for row in rows} == {"cfg-2"}
    viewer.detach_all()
    window, source, detached_views = viewer._detached_windows[0]
    viewer.slider.setValue(2)
    assert source.frame == 2 and all(view.frame == 2 for view in detached_views)
    window.close()
    viewer.close()


def test_dashboard_missing_hand_is_na_and_fallback_is_visible(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer
    path = _session(tmp_path, 2)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["reference_configuration_id"] = "removed-result"
    npz_path = tmp_path / manifest["final_results"]["cfg-1"]["path"]
    with np.load(npz_path) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays["detected"][0, 0] = False
    arrays["keypoints_2d"][0, 0] = np.nan
    arrays["keypoints_3d"][0, 0] = np.nan
    np.savez_compressed(npz_path, **arrays)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    viewer = ComparisonViewer(path)
    assert viewer.reference_id == "cfg-0"
    assert "fallback" in viewer.reference_label.text()
    assert "L N/A" in viewer.split_panels[1].metric_rows[("cfg-1", "wrist_error", "left")].text()
    assert not viewer.metric_checks["mpvpe"].isEnabled()
    viewer.close()


def test_comparison_dashboard_uses_viewport_overlays_and_compact_menus(tmp_path, qapp):
    from PySide6.QtWidgets import QSizePolicy
    from calibfreehands.visualization.comparison import ComparisonViewer
    viewer = ComparisonViewer(_session(tmp_path, 5))
    assert not hasattr(viewer, "metric_table")
    assert len(viewer.split_panels) == 5
    assert viewer.split_views[1].minimumWidth() >= 320
    assert viewer.split_views[1].minimumHeight() >= 240
    assert set(viewer.mode_buttons) == {"overlay", "split"}
    assert viewer.results_menu_button.text().startswith("Results")
    assert viewer.confidence_menu_button.text() == "Detection"
    assert len(viewer.visibility_checks) == 5
    assert all(check.parentWidget() is not viewer for check in viewer.visibility_checks)
    for button in viewer.view_buttons + viewer.playback_buttons:
        assert button.sizePolicy().horizontalPolicy() != QSizePolicy.Policy.Expanding
    assert not viewer.metric_checks["f@5"].isEnabled()
    assert "pending" in viewer.metric_checks["f@5"].toolTip().lower()
    viewer.close()


def test_presentation_hud_is_viewport_child_and_reorder_preserves_result_data(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer
    viewer = ComparisonViewer(_session(tmp_path, 3))
    assert viewer.overlay_panel.parentWidget() is viewer.overlay_view
    assert viewer.split_panels[1].parentWidget() is viewer.split_views[1]
    assert viewer.overlay_panel.geometry() == viewer.overlay_view.rect()
    assert viewer.split_panels[1].geometry() == viewer.split_views[1].rect()
    overlay_splitter = viewer.views.widget(0)
    assert overlay_splitter.widget(1) is viewer.overlay_view
    original_media_cell = viewer.split_grid.itemAtPosition(0, 0).widget()
    before = [loaded.result.keypoints_3d.copy() for loaded in viewer.loaded_results]
    viewer.reorder_result("cfg-2", "cfg-0")
    assert viewer.split_result_order == [2, 0, 1]
    assert viewer.split_grid.itemAtPosition(0, 1).widget() is viewer.split_views[2]
    assert viewer.split_grid.itemAtPosition(0, 0).widget() is original_media_cell
    from calibfreehands.visualization.comparison_presentation import _ResultDragHandle
    handle = viewer.split_panels[2].findChild(_ResultDragHandle)
    assert handle is not None and not handle.isHidden()
    assert viewer.split_views[2].acceptDrops()
    assert viewer.results_menu.actions()[0].defaultWidget().findChild(QCheckBox).text() == "Method 2"
    for loaded, points in zip(viewer.loaded_results, before):
        np.testing.assert_array_equal(loaded.result.keypoints_3d, points)
    viewer.set_annotation_mode("all")
    viewer._set_annotation_content("error")
    candidate_labels = viewer.split_annotations[1].projected_annotations()
    reference_labels = viewer.split_annotations[0].projected_annotations()
    assert candidate_labels and not reference_labels
    assert candidate_labels[0][3] == "1414.2"
    viewer.split_views[1].show_left = False
    assert {row[1] for row in viewer.split_annotations[1].projected_annotations()} == {"right"}
    viewer.set_annotation_mode("all")
    viewer._set_annotation_content("name")
    assert {row[3] for row in viewer.split_annotations[1].projected_annotations()} <= set(JOINT_NAMES)
    viewer.set_annotation_mode("off")
    assert not viewer.split_annotations[1].projected_annotations()
    viewer.close()


def test_presentation_controls_keep_detection_and_mesh_opacity_distinct(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer
    viewer = ComparisonViewer(_session(tmp_path, 2, mesh=True))
    assert viewer.confidence_menu_button.text() == "Detection"
    assert viewer.confidence_opacity_checkbox.text() == "Use as opacity"
    assert viewer.confidence_numeric_checkbox.text() == "Numeric value"
    assert viewer.mesh_opacity_label.text() == "Mesh opacity"
    assert not viewer.mesh_opacity_slider.isEnabled()
    viewer.mesh_checkbox.setChecked(True)
    assert viewer.mesh_opacity_slider.isEnabled()
    viewer.mesh_opacity_slider.setValue(50)
    mesh_alpha = viewer.split_views[0].meshes[0].opts["color"][3]
    viewer.confidence_opacity_checkbox.setChecked(False)
    assert viewer.split_views[0].meshes[0].opts["color"][3] > mesh_alpha
    assert not viewer.joint_checkbox.isChecked() and not viewer.bone_checkbox.isChecked()
    assert viewer.more_menu.actions()[0].text() == "Detach panels"
    assert viewer.error_vectors_button.menu() is viewer.error_vectors_menu
    assert viewer.annotation_button.menu() is viewer.annotation_menu
    assert viewer.hud_button.menu() is viewer.hud_menu
    viewer._set_hud(opacity=0.5, scale="large", anchor="bottom-right")
    assert viewer.overlay_panel.opacity_effect.opacity() == pytest.approx(0.5)
    assert viewer.split_panels[0].scale == "large"
    assert viewer.split_panels[0].anchor == "bottom-right"
    viewer.close()


def test_presentation_auto_hide_activity_pause_and_fullscreen_keys(tmp_path, qapp):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    from calibfreehands.visualization.comparison import ComparisonViewer
    viewer = ComparisonViewer(_session(tmp_path, 2))
    viewer.show()
    qapp.processEvents()
    viewer.timer.start(1000)
    viewer._auto_hide_controls()
    assert viewer.chrome.isHidden()
    viewer.eventFilter(viewer, QEvent(QEvent.Type.MouseMove))
    assert not viewer.chrome.isHidden()
    viewer.reference_selector.showPopup()
    viewer._auto_hide_controls()
    assert not viewer.chrome.isHidden()
    viewer.reference_selector.hidePopup()
    viewer._auto_hide_controls()
    assert viewer.chrome.isHidden()
    viewer.timer.stop()
    viewer._play()
    assert not viewer.chrome.isHidden()
    f11 = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_F11, Qt.KeyboardModifier.NoModifier)
    viewer.eventFilter(viewer, f11)
    assert viewer.isFullScreen()
    escape = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    viewer.eventFilter(viewer, escape)
    assert not viewer.isFullScreen()
    viewer.close()


def test_mainwindow_checkerboard_focal_has_no_duplicate_row_label(tmp_path, monkeypatch, qapp):
    from PySide6.QtWidgets import QLabel
    from test_queue_workflow import make_window
    _, window, _ = make_window(tmp_path, monkeypatch)
    row = window.configuration_rows[0]
    row.calibration.setCurrentIndex(row.calibration.findData("checkerboard_reference"))
    assert window.rows_layout.itemAtPosition(0, 3).widget().text() == "Reference Focal"
    assert not any(label.text() == "Reference Focal" for label in row.reference_focal_container.findChildren(QLabel))
    assert row.reference_focal_container.layout().itemAt(0).widget() is row.reference_focal
    window.close()


def test_presentation_controls_overlay_without_changing_viewport_geometry(tmp_path, qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QGridLayout
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2))
    viewer.resize(1280, 800)
    viewer.show()
    qapp.processEvents()
    layout = viewer.layout()
    assert isinstance(layout, QGridLayout)
    viewport = viewer.views.geometry()
    viewer.chrome.hide()
    qapp.processEvents()
    assert viewer.views.geometry() == viewport
    viewer.chrome.show()
    qapp.processEvents()
    assert viewer.views.geometry() == viewport
    assert layout.itemAt(1).alignment() & Qt.AlignmentFlag.AlignBottom
    viewer.close()


def test_presentation_pin_and_annotation_density_modes(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2))
    assert hasattr(viewer, "pin_controls_button")
    viewer.pin_controls_button.setChecked(True)
    assert viewer.pin_controls_button.text() in {"Pinned", "Pin: On"}
    viewer.timer.start()
    viewer._auto_hide_controls()
    assert not viewer.chrome.isHidden()
    assert {"off", "hovered", "key_joints", "worst5", "all"} <= set(viewer.annotation_actions)
    viewer.set_annotation_mode("worst5")
    assert viewer.annotation_mode == "worst5"
    viewer.close()


def test_comparison_event_filter_accepts_qwindow_and_combo_close_target(tmp_path, qapp):
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QWindow
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2))
    window = QWindow()
    assert viewer.eventFilter(window, QEvent(QEvent.Type.MouseButtonPress)) is False
    viewer._hover_menu_target = viewer.reference_selector
    viewer._close_hover_menu()
    assert not viewer.reference_selector.view().isVisible()
    window.deleteLater()
    viewer.close()


def test_active_dropdown_filter_handles_qwindow_targets_without_widget_api_calls(tmp_path, qapp):
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QWindow
    from PySide6.QtWidgets import QWidget
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2))
    unrelated_window = QWindow()

    combo = viewer.reference_selector
    combo_view = combo.view()
    viewer._hover_menu_target = combo
    assert viewer._is_active_hover_menu(combo_view)
    assert viewer._is_active_hover_menu(combo_view.viewport())
    assert not viewer._is_active_hover_menu(QWidget())
    assert not viewer._is_active_hover_menu(unrelated_window)
    assert viewer.eventFilter(unrelated_window, QEvent(QEvent.Type.Hide)) is False
    combo.showPopup()
    qapp.processEvents()
    combo_popup_window = combo_view.window().windowHandle()
    assert combo_popup_window is not None
    assert viewer._is_active_hover_menu(combo_popup_window)
    viewer._hover_menu_latched = True
    assert viewer.eventFilter(combo_popup_window, QEvent(QEvent.Type.Hide)) is False
    assert not viewer._hover_menu_latched
    viewer._hide_hover_popup(combo)

    menu_button = viewer.results_menu_button
    menu = menu_button.menu()
    viewer._hover_menu_target = menu_button
    assert viewer._is_active_hover_menu(menu)
    assert viewer._is_active_hover_menu(menu.actions()[0].defaultWidget())
    assert not viewer._is_active_hover_menu(QWidget())
    menu.popup(viewer.mapToGlobal(viewer.rect().center()))
    qapp.processEvents()
    menu_popup_window = menu.windowHandle()
    assert menu_popup_window is not None
    assert viewer._is_active_hover_menu(menu_popup_window)
    viewer._hover_menu_latched = True
    assert viewer.eventFilter(menu_popup_window, QEvent(QEvent.Type.MouseButtonPress)) is False
    assert viewer._hover_menu_latched
    menu.hide()
    viewer._hover_menu_latched = True
    assert viewer.eventFilter(unrelated_window, QEvent(QEvent.Type.MouseButtonPress)) is False
    assert not viewer._hover_menu_latched

    unrelated_window.deleteLater()
    viewer.close()


def test_hover_menu_becomes_latched_by_click_and_closes_on_action(tmp_path, qapp):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2))
    viewer.resize(1280, 800); viewer.show(); qapp.processEvents()
    button = viewer.annotation_button
    menu = viewer.annotation_menu
    QTest.mouseMove(viewer.views, QPoint(40, 40)); qapp.processEvents()
    QTest.mouseMove(button, button.rect().center()); qapp.processEvents()
    assert menu.isVisible()
    QTest.mouseClick(button, Qt.MouseButton.LeftButton); qapp.processEvents()
    QTest.mouseMove(viewer.views, QPoint(40, 40)); qapp.processEvents()
    assert menu.isVisible()
    action = next(action for action in menu.actions() if action.isCheckable())
    QTest.mouseClick(menu, Qt.MouseButton.LeftButton, pos=menu.actionGeometry(action).center())
    qapp.processEvents()
    assert not menu.isVisible()
    viewer.close()


def test_clicked_menu_stays_latched_until_second_toggle_or_escape(tmp_path, qapp):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2))
    viewer.resize(1280, 800); viewer.show(); qapp.processEvents()
    button = viewer.hud_button
    QTest.mouseClick(button, Qt.MouseButton.LeftButton); qapp.processEvents()
    assert viewer.hud_menu.isVisible() and viewer._hover_menu_latched
    QTest.mouseMove(viewer.views, QPoint(40, 40)); qapp.processEvents()
    assert viewer.hud_menu.isVisible()
    QTest.mouseClick(button, Qt.MouseButton.LeftButton); qapp.processEvents()
    assert not viewer.hud_menu.isVisible()
    QTest.mouseClick(button, Qt.MouseButton.LeftButton); qapp.processEvents()
    assert viewer.hud_menu.isVisible()
    QTest.keyClick(viewer.hud_menu, Qt.Key.Key_Escape); qapp.processEvents()
    assert not viewer.hud_menu.isVisible()
    QTest.mouseClick(button, Qt.MouseButton.LeftButton); qapp.processEvents()
    assert viewer.hud_menu.isVisible()
    QTest.mouseClick(viewer.playback_buttons[-1], Qt.MouseButton.LeftButton); qapp.processEvents()
    assert not viewer.hud_menu.isVisible() and not viewer._hover_menu_latched
    viewer.close()


def test_joint_annotation_density_and_label_content_use_joint_errors(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2))
    layer = viewer.split_annotations[1]
    viewer.set_mode("split")
    viewer.set_annotation_mode("all")
    all_rows = layer.projected_annotations()
    assert len(all_rows) == 42
    assert all(row[3].replace(".", "", 1).isdigit() for row in all_rows)
    viewer.set_annotation_mode("key_joints")
    assert len(layer.projected_annotations()) == 12
    viewer.set_annotation_mode("worst5")
    worst = layer.projected_annotations()
    assert len(worst) == 10
    assert all(label == "1414.2" for *_, label, _, _ in worst)
    viewer.set_annotation_mode("hovered")
    layer.hover_position = __import__("PySide6.QtCore", fromlist=["QPoint"]).QPoint(320, 240)
    assert len(layer.projected_annotations()) <= 2
    viewer._set_annotation_content("name")
    named = layer.projected_annotations()
    assert all(row[3] in JOINT_NAMES for row in named)
    viewer.close()


def test_compact_presentation_explains_disabled_options_and_hud_background_default(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2))
    assert viewer.metric_checks["f@5"].toolTip().startswith("Protocol pending:")
    assert viewer.results_menu_button.text() == "Results (2/2)"
    assert viewer.error_vectors_button.text() == "Vectors: Off"
    assert viewer.overlay_panel.background == "none"
    assert "transparent" in viewer.overlay_panel.card.styleSheet()
    assert viewer.confidence_color_mode.itemText(1) == "Detection"
    viewer.close()


def test_toolbutton_menu_opens_immediately_and_closes_when_pointer_leaves(tmp_path, qapp):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2))
    viewer.resize(1280, 800); viewer.show(); qapp.processEvents()
    button = viewer.metric_menu_button
    QTest.mouseMove(viewer.views, QPoint(40, 40)); qapp.processEvents()
    QTest.mouseMove(button, button.rect().center())
    qapp.processEvents()
    assert viewer.metric_menu.isVisible()
    QTest.mouseMove(viewer.metric_menu, viewer.metric_menu.rect().center()); qapp.processEvents()
    assert viewer.metric_menu.isVisible()
    QTest.mouseMove(viewer.views, QPoint(40, 40))
    qapp.processEvents()
    assert not viewer.metric_menu.isVisible()
    viewer.close()


def test_combo_boxes_open_on_hover_and_close_after_click_latch(tmp_path, qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2))
    viewer.resize(1280, 800); viewer.show(); qapp.processEvents()
    combo = viewer.reference_selector
    QTest.mouseMove(combo, combo.rect().center())
    qapp.processEvents()
    assert combo.view().isVisible()
    QTest.mouseClick(combo, Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert combo.view().isVisible() and viewer._hover_menu_latched
    QTest.mouseClick(combo, Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert not combo.view().isVisible()
    viewer.close()


def test_comparison_frame_waits_for_matching_source_frame_before_presenting(tmp_path, qapp):
    from calibfreehands.media.provider import MediaFrameProvider
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2))
    provider = MediaFrameProvider(tmp_path / "source.mp4", frame_loader=lambda index: np.full((24, 32, 3), index, np.uint8))
    provider._store(0, np.zeros((24, 32, 3), np.uint8))
    for loaded in viewer.loaded_results:
        loaded.media_frames = provider
    viewer._watch_source_provider(provider)
    hud_frames = []
    original_update = viewer.overlay_panel.update_values
    viewer.overlay_panel.update_values = lambda owner: (hud_frames.append(owner.slider.value()), original_update(owner))

    viewer.slider.setValue(1)
    assert viewer._last_rendered_frame == 0
    assert viewer.slider.value() == 0
    assert viewer.overlay_view.frame == 0
    assert viewer.status.text().startswith("Frame 1 /")

    provider._store(1, np.ones((24, 32, 3), np.uint8))
    provider.signals.frame_ready.emit(1)
    assert viewer._last_rendered_frame == 1
    assert viewer.slider.value() == 1
    assert viewer.overlay_view.frame == 1
    assert viewer.status.text().startswith("Frame 2 /")
    assert hud_frames[-1] == 1
    assert viewer.presented_frame == viewer.requested_frame == 1
    assert viewer.source_overlay._display_image.pixelColor(0, 0).red() == 1
    provider.close()
    viewer.close()


def test_comparison_ignores_stale_source_completion_and_tracks_latest_request(tmp_path, qapp):
    from calibfreehands.media.provider import MediaFrameProvider
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2))
    provider = MediaFrameProvider(tmp_path / "source.mp4", frame_loader=lambda index: np.full((24, 32, 3), index, np.uint8))
    provider._store(0, np.zeros((24, 32, 3), np.uint8))
    for loaded in viewer.loaded_results:
        loaded.media_frames = provider
    viewer._watch_source_provider(provider)

    viewer._set_frame(1)
    viewer._set_frame(2)
    provider._store(1, np.ones((24, 32, 3), np.uint8))
    provider.signals.frame_ready.emit(1)
    assert viewer._last_rendered_frame == 0
    provider._store(2, np.full((24, 32, 3), 2, np.uint8))
    provider.signals.frame_ready.emit(2)
    assert viewer._last_rendered_frame == 2
    assert viewer.slider.value() == 2
    provider.close()
    viewer.close()


def test_pause_discards_pending_playback_frame_but_manual_seek_is_exact(tmp_path, qapp):
    from calibfreehands.media.provider import MediaFrameProvider
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2))
    provider = MediaFrameProvider(tmp_path / "source.mp4", frame_loader=lambda index: np.full((24, 32, 3), index, np.uint8))
    provider._store(0, np.zeros((24, 32, 3), np.uint8))
    for loaded in viewer.loaded_results:
        loaded.media_frames = provider
    viewer._watch_source_provider(provider)

    viewer.timer.start(1000)
    viewer._set_frame(1)
    viewer._play()
    provider._store(1, np.ones((24, 32, 3), np.uint8))
    provider.signals.frame_ready.emit(1)
    assert viewer.presented_frame == 0
    assert viewer.requested_frame == 0

    viewer.slider.setValue(2)
    assert viewer.presented_frame == 0
    provider._store(2, np.full((24, 32, 3), 2, np.uint8))
    provider.signals.frame_ready.emit(2)
    assert viewer.presented_frame == 2
    assert viewer.overlay_view.frame == 2
    assert viewer.slider.value() == 2
    assert viewer.status.text().startswith("Frame 3 /")
    provider.close()
    viewer.close()


@pytest.mark.parametrize("selector_name", ["reference_selector", "speed_selector", "confidence_color_mode"])
def test_dropdown_combos_open_on_hover_and_click_latches(tmp_path, qapp, selector_name):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2))
    viewer.resize(1280, 800); viewer.show(); qapp.processEvents()
    combo = getattr(viewer, selector_name)
    QTest.mouseMove(combo, combo.rect().center()); qapp.processEvents()
    assert combo.view().isVisible()
    QTest.mouseClick(combo, Qt.MouseButton.LeftButton); qapp.processEvents()
    assert combo.view().isVisible() and viewer._hover_menu_latched
    QTest.mouseMove(viewer.views, QPoint(40, 40)); qapp.processEvents()
    assert combo.view().isVisible()
    QTest.mouseClick(combo, Qt.MouseButton.LeftButton); qapp.processEvents()
    assert not combo.view().isVisible()
    viewer.close()


def test_combo_hover_popup_closes_when_pointer_leaves_control_and_popup(tmp_path, qapp):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2))
    viewer.resize(1280, 800); viewer.show(); qapp.processEvents()
    combo = viewer.reference_selector
    QTest.mouseMove(combo, combo.rect().center()); qapp.processEvents()
    assert combo.view().isVisible()
    QTest.mouseMove(viewer.views, QPoint(40, 40)); qapp.processEvents()
    assert not combo.view().isVisible()
    viewer.close()


def test_hud_is_text_only_and_has_no_scroll_area_or_group_separators(tmp_path, qapp):
    from PySide6.QtWidgets import QFrame, QLabel, QScrollArea
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 6))
    viewer.resize(1280, 800); viewer.show(); viewer.confidence_numeric_checkbox.setChecked(True)
    qapp.processEvents()
    panel = viewer.overlay_panel
    assert not panel.findChildren(QScrollArea)
    assert not any(isinstance(panel.body_layout.itemAt(i).widget(), QFrame)
                   and panel.body_layout.itemAt(i).widget().frameShape() == QFrame.Shape.HLine
                   for i in range(panel.body_layout.count()) if panel.body_layout.itemAt(i).widget())
    assert panel.card.geometry().right() <= panel.width()
    assert panel.card.geometry().bottom() <= panel.height()
    assert all(label.font().pointSizeF() >= 7 for label in panel.body.findChildren(QLabel))
    viewer.close()


def test_mouse_move_over_viewport_restores_hidden_controls(tmp_path, qapp):
    from PySide6.QtTest import QTest
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2))
    viewer.resize(1280, 800); viewer.show(); qapp.processEvents()
    viewer.timer.start(1000); viewer.chrome.hide()
    QTest.mouseMove(viewer.overlay_view, viewer.overlay_view.rect().center())
    qapp.processEvents()
    assert not viewer.chrome.isHidden()
    viewer.close()
