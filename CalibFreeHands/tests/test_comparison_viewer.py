import json

import numpy as np
import pytest
from PySide6.QtWidgets import QCheckBox

from calibfreehands.pipeline import CanonicalResult, save_result
from calibfreehands.pipeline.result_bundle import write_result_bundle
from calibfreehands.joint_schema import JOINT_NAMES


def _wait_for_metric_job(viewer, qapp):
    from PySide6.QtTest import QTest
    for _ in range(500):
        qapp.processEvents()
        if not viewer.metric_status.text().startswith("Updating comparison metrics"):
            return
        QTest.qWait(10)
    pytest.fail("comparison metric worker did not finish")


def _session(tmp_path, count=3, mesh=False, gt_index=None):
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
                               "calibration_model_id": "checkerboard_reference" if index == gt_index else "reference",
                               "reconstruction_model_id": "pose"})
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
    np.testing.assert_allclose(base_color[:3], viewer.configuration_colors[0])
    assert base_color[3] == pytest.approx(1.0)
    assert not viewer.confidence_opacity_checkbox.isChecked()
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


def test_comparison_camera_is_shared_across_modes_and_fit_all_is_explicit(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 3, mesh=True))
    default = viewer.overlay_view.cameraParams()
    assert all(view.cameraParams()["distance"] == pytest.approx(default["distance"])
               for view in viewer.split_views)
    for view in viewer.split_views:
        np.testing.assert_allclose(view.cameraParams()["center"].toTuple(), default["center"].toTuple())
        for index, grid in enumerate(view.grids):
            assert grid.size() == pytest.approx(viewer.overlay_view.grids[index].size())
            assert grid.spacing() == pytest.approx(viewer.overlay_view.grids[index].spacing())
            assert grid.transform() == viewer.overlay_view.grids[index].transform()
    camera_menu = next(action.menu() for action in viewer.more_menu.actions()
                       if action.menu() is not None and action.text() == "Camera")
    assert "Fit All" in [action.text() for action in camera_menu.actions()]
    viewer.geometry_checks["show_grid"].setChecked(False)
    assert all(label.visible() for label in viewer.overlay_view.numeric_labels)
    assert all(all(label.visible() for label in view.numeric_labels) for view in viewer.split_views)
    viewer.geometry_checks["show_grid"].setChecked(True)
    assert all(label.visible() for label in viewer.overlay_view.numeric_labels)

    overlay_pose = {"pos": (0.2, -0.6, 0.3), "distance": 7.0, "azimuth": 80.0, "elevation": 15.0}
    viewer.overlay_view.setCameraPosition(**overlay_pose)
    viewer.set_mode("split")
    split_pose = viewer.split_views[0].cameraParams()
    assert split_pose["distance"] == pytest.approx(overlay_pose["distance"])
    assert split_pose["azimuth"] == pytest.approx(overlay_pose["azimuth"])
    assert split_pose["elevation"] == pytest.approx(overlay_pose["elevation"])
    np.testing.assert_allclose(tuple(split_pose["center"]), overlay_pose["pos"])

    viewer.split_views[1].setCameraPosition(pos=(0.1, -0.4, 0.2), distance=5.0, azimuth=30.0, elevation=25.0)
    viewer._sync_camera(viewer.split_views[1])
    viewer.set_mode("overlay")
    overlay_synced = viewer.overlay_view.cameraParams()
    assert overlay_synced["distance"] == pytest.approx(5.0)
    assert overlay_synced["azimuth"] == pytest.approx(30.0)
    assert overlay_synced["elevation"] == pytest.approx(25.0)
    np.testing.assert_allclose(tuple(overlay_synced["center"]), (0.1, -0.4, 0.2))

    viewer.close()


def test_detached_comparison_views_share_current_pose_and_remain_linked(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 3))
    viewer.overlay_view.setCameraPosition(pos=(0.2, -0.8, 0.4), distance=6.0, azimuth=72.0, elevation=18.0)
    viewer.detach_all()
    window, _source, detached_views = viewer._detached_windows[0]
    for view in detached_views:
        params = view.cameraParams()
        assert params["distance"] == pytest.approx(6.0)
        assert params["azimuth"] == pytest.approx(72.0)
        assert params["elevation"] == pytest.approx(18.0)
        assert tuple(params["center"]) == pytest.approx((0.2, -0.8, 0.4))

    changed = detached_views[1]
    changed.setCameraPosition(pos=(0.1, -0.3, 0.2), distance=4.0, azimuth=30.0, elevation=25.0)
    changed.camera_changed(changed)
    for view in detached_views:
        params = view.cameraParams()
        assert params["distance"] == pytest.approx(4.0)
        assert params["azimuth"] == pytest.approx(30.0)
        assert params["elevation"] == pytest.approx(25.0)
        assert tuple(params["center"]) == pytest.approx((0.1, -0.3, 0.2))
    viewer.set_space_mode("hand")
    expected_bounds = viewer.overlay_view.workspace_bounds_m.copy()
    side_lengths = np.ptp(expected_bounds, axis=1)
    np.testing.assert_allclose(side_lengths, np.repeat(side_lengths[0], 3))
    for view in detached_views:
        np.testing.assert_allclose(view.workspace_bounds_m, expected_bounds)
    viewer.labels_switch.setChecked(False)
    assert all(not label.visible() for view in detached_views for label in view.numeric_labels)
    window.close()
    viewer.close()


def test_detach_all_initial_geometry_keeps_3d_viewports_renderable(tmp_path, qapp):
    from PySide6.QtTest import QTest
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 3))
    viewer.resize(1500, 900)
    viewer.show()
    QTest.qWait(50)
    viewer.detach_all()
    window, _source, detached_views = viewer._detached_windows[0]
    QTest.qWait(50)
    assert all(view.width() >= 340 and view.height() >= 260 for view in detached_views)
    window.close()
    viewer.close()


def test_fit_all_uses_visible_union_and_leaves_fixed_metric_reference_unchanged(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 3, mesh=True))
    viewer.loaded_results[2].result.keypoints_3d[..., 2] = 15.0
    viewer.loaded_results[2].result.mano_vertices[..., 2] = 15.0
    before = viewer.overlay_view.cameraParams()
    grid_data = [(grid.transform(), grid.size(), grid.spacing()) for grid in viewer.overlay_view.grids]
    axis_data = viewer.overlay_view.axis.pos.copy()
    label_data = [(label.text, label.pos.copy()) for label in viewer.overlay_view.numeric_labels]

    assert viewer.fit_all()
    all_results_camera = viewer.overlay_view.cameraParams()
    assert all_results_camera["distance"] > before["distance"]
    assert all_results_camera["center"].y() < before["center"].y()
    viewer.visibility_checks[2].setChecked(False)
    assert viewer.fit_all()
    visible_camera = viewer.overlay_view.cameraParams()
    assert visible_camera["distance"] < all_results_camera["distance"]

    for (transform, size, spacing), grid in zip(grid_data, viewer.overlay_view.grids):
        assert grid.transform() == transform
        assert grid.size() == pytest.approx(size)
        assert grid.spacing() == pytest.approx(spacing)
    np.testing.assert_array_equal(viewer.overlay_view.axis.pos, axis_data)
    for (text, position), label in zip(label_data, viewer.overlay_view.numeric_labels):
        assert label.text == text
        np.testing.assert_array_equal(label.pos, position)

    viewer.set_mode("split")
    assert viewer.fit_all()
    split_camera = viewer.split_views[0].cameraParams()
    for view, check in zip(viewer.split_views, viewer.visibility_checks):
        if check.isChecked():
            assert view.cameraParams()["distance"] == pytest.approx(split_camera["distance"])
            np.testing.assert_allclose(view.cameraParams()["center"].toTuple(), split_camera["center"].toTuple())
    viewer.set_mode("overlay")
    viewer.reset_camera()
    reset = viewer.overlay_view.cameraParams()
    assert reset["distance"] == pytest.approx(before["distance"])
    np.testing.assert_allclose(reset["center"].toTuple(), before["center"].toTuple())

    for loaded in viewer.loaded_results:
        loaded.result.detected[:] = False
    unchanged = viewer.overlay_view.cameraParams()
    assert not viewer.fit_all()
    assert viewer.overlay_view.cameraParams()["distance"] == pytest.approx(unchanged["distance"])
    np.testing.assert_allclose(viewer.overlay_view.cameraParams()["center"].toTuple(), unchanged["center"].toTuple())
    viewer.close()


def test_gt_change_refocuses_selected_hand_and_metric_overview_is_restorable(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer
    from calibfreehands.visualization.viewer import default_metric_camera_distance

    viewer = ComparisonViewer(_session(tmp_path, 3, gt_index=1))
    assert viewer.reference_id == "cfg-1"
    initial = viewer.overlay_view.cameraParams()
    np.testing.assert_allclose(initial["center"].toTuple(), (-1.0, -3.0, 0.0))
    for view in viewer.split_views:
        np.testing.assert_allclose(view.cameraParams()["center"].toTuple(), (-1.0, -3.0, 0.0))

    viewer.overlay_view.setCameraPosition(pos=(0.4, -0.8, 1.7), distance=3.25,
                                          azimuth=88.0, elevation=7.0)
    viewer._sync_camera(viewer.overlay_view)
    before_gt_change = viewer.overlay_view.cameraParams()

    viewer.reference_selector.setCurrentIndex(viewer.reference_selector.findData("cfg-2"))
    focused = viewer.overlay_view.cameraParams()
    np.testing.assert_allclose(tuple(focused["center"]), (-2.0, -4.0, 0.0))
    assert focused["distance"] != pytest.approx(before_gt_change["distance"])
    assert focused["azimuth"] == pytest.approx(before_gt_change["azimuth"])
    assert focused["elevation"] == pytest.approx(before_gt_change["elevation"])
    for view in viewer.split_views:
        params = view.cameraParams()
        np.testing.assert_allclose(tuple(params["center"]), tuple(focused["center"]))
        assert params["distance"] == pytest.approx(focused["distance"])
        assert params["azimuth"] == pytest.approx(before_gt_change["azimuth"])
        assert params["elevation"] == pytest.approx(before_gt_change["elevation"])
    assert viewer.hand_view_cache["cfg-2"][0] == pytest.approx((-2.0, -4.0, 0.0))

    viewer.hand_view()
    np.testing.assert_allclose(viewer.overlay_view.cameraParams()["center"].toTuple(), (-2.0, -4.0, 0.0))
    for view in viewer.split_views:
        np.testing.assert_allclose(view.cameraParams()["center"].toTuple(), (-2.0, -4.0, 0.0))

    cached_framings = dict(viewer.hand_view_cache)
    for frame in (1, 2, 0):
        viewer.slider.setValue(frame)
    assert viewer.hand_view_cache == cached_framings
    assert viewer.overlay_view.cameraParams()["center"].toTuple() == pytest.approx((-2.0, -4.0, 0.0))

    viewer.set_mode("split")
    viewer.metric_overview()
    metric = viewer.overlay_view.cameraParams()
    np.testing.assert_allclose(metric["center"].toTuple(), (0.0, -1.0, 0.0))
    assert metric["distance"] == pytest.approx(default_metric_camera_distance())
    for view in viewer.split_views:
        np.testing.assert_allclose(view.cameraParams()["center"].toTuple(), (0.0, -1.0, 0.0))
        assert view.cameraParams()["distance"] == pytest.approx(default_metric_camera_distance())
    viewer.reset_camera()
    for view in [viewer.overlay_view, *viewer.split_views]:
        np.testing.assert_allclose(view.cameraParams()["center"].toTuple(), (-2.0, -4.0, 0.0))
    camera_menu = next(action.menu() for action in viewer.more_menu.actions()
                       if action.menu() is not None and action.text() == "Camera")
    assert [action.text() for action in camera_menu.actions()] == [
        "Front", "Left", "Right", "Top", "Focus GT", "Space Overview", "Fit All"
    ]
    viewer.close()


def test_physical_and_hand_spaces_update_orbit_center_only_and_preserve_frame_playback(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer
    from calibfreehands.visualization.viewer import hand_space_bounds

    viewer = ComparisonViewer(_session(tmp_path, 3, gt_index=1))
    viewer.loaded_results[2].result.keypoints_3d[..., 2] = 15.0
    gt = viewer.loaded_results[1].result
    expected_gt_bounds = hand_space_bounds(gt.keypoints_3d, gt.detected)
    raw_before = [(loaded.result.keypoints_3d.copy(),
                   None if loaded.result.mano_vertices is None else loaded.result.mano_vertices.copy(),
                   None if loaded.result.camera_translation is None else loaded.result.camera_translation.copy(),
                   loaded.result.focal_px.copy(), loaded.result.detected.copy())
                  for loaded in viewer.loaded_results]
    viewer.slider.setValue(1)
    viewer.overlay_view.setCameraPosition(pos=(0.25, -0.75, 1.2), distance=4.5,
                                          azimuth=24.0, elevation=10.0)
    viewer._sync_camera(viewer.overlay_view)
    before_camera = viewer.overlay_view.cameraParams()
    before_frame = viewer.slider.value()
    viewer.timer.start()

    viewer.space_mode_switch.setChecked(True)

    assert viewer.space_mode == "hand"
    assert viewer.space_mode_switch.toolTip() == "Space: Hand"
    assert not hasattr(viewer, "space_mode_value")
    assert viewer.slider.value() == before_frame
    hand_center_display = viewer.overlay_view.workspace.display_center
    _expected_hand_center, expected_hand_distance = viewer._camera_framing_for("cfg-1", "hand")
    for view in [viewer.overlay_view, *viewer.split_views]:
        np.testing.assert_allclose(view.workspace_bounds_m, expected_gt_bounds)
        np.testing.assert_allclose(tuple(view.cameraParams()["center"]), hand_center_display)
        assert view.cameraParams()["distance"] == pytest.approx(expected_hand_distance)
        assert view.cameraParams()["azimuth"] == pytest.approx(before_camera["azimuth"])
        assert view.cameraParams()["elevation"] == pytest.approx(before_camera["elevation"])
    assert viewer.timer.isActive()
    assert np.max(np.ptp(viewer.overlay_view.workspace_bounds_m, axis=1)) < 1.0

    viewer.labels_switch.setChecked(False)
    assert viewer.overlay_view.show_labels is False
    assert all(not label.visible() for label in viewer.overlay_view.numeric_labels)
    assert all(grid.visible() for grid in viewer.overlay_view.grids)
    assert viewer.overlay_view.axis.visible()
    assert viewer.slider.value() == before_frame
    np.testing.assert_allclose(tuple(viewer.overlay_view.cameraParams()["center"]), hand_center_display)
    assert viewer.timer.isActive()
    viewer.labels_switch.setChecked(True)
    assert all(label.visible() for label in viewer.overlay_view.numeric_labels)

    hand_camera = viewer.overlay_view.cameraParams()
    viewer.reference_selector.setCurrentIndex(viewer.reference_selector.findData("cfg-0"))
    changed_camera = viewer.overlay_view.cameraParams()
    expected_hand_target, expected_hand_distance = viewer._camera_framing_for("cfg-0", "hand")
    assert tuple(changed_camera["center"]) == pytest.approx(expected_hand_target)
    assert changed_camera["distance"] == pytest.approx(expected_hand_distance)
    assert changed_camera["azimuth"] == pytest.approx(hand_camera["azimuth"])
    assert changed_camera["elevation"] == pytest.approx(hand_camera["elevation"])
    np.testing.assert_allclose(viewer.overlay_view.workspace_bounds_m,
                               hand_space_bounds(viewer.loaded_results[0].result.keypoints_3d,
                                                 viewer.loaded_results[0].result.detected))
    viewer.set_mode("split")
    assert viewer.space_mode == "hand"
    for view in viewer.split_views:
        np.testing.assert_allclose(view.workspace_bounds_m, viewer.overlay_view.workspace_bounds_m)
    viewer.set_mode("split")
    viewer.space_mode_switch.setChecked(False)
    assert viewer.space_mode == "physical"
    np.testing.assert_allclose(viewer.overlay_view.workspace_bounds_m, ((-1, 1), (-1, 1), (0, 2)))
    physical_target, physical_distance = viewer._camera_framing_for("cfg-0", "physical")
    physical_center_display = tuple(physical_target)
    for view in [viewer.overlay_view, *viewer.split_views]:
        np.testing.assert_allclose(tuple(view.cameraParams()["center"]), physical_center_display)
        assert view.cameraParams()["distance"] == pytest.approx(physical_distance)
        assert view.cameraParams()["azimuth"] == pytest.approx(before_camera["azimuth"])
        assert view.cameraParams()["elevation"] == pytest.approx(before_camera["elevation"])
    assert viewer.slider.value() == before_frame
    assert viewer.timer.isActive()
    viewer.timer.stop()
    for target_mode, target_center in (
            ("hand", tuple(expected_hand_target)), ("physical", physical_center_display),
            ("hand", tuple(expected_hand_target))):
        viewer.set_space_mode(target_mode)
        expected_distance = viewer._camera_states[target_mode]["distance"]
        for view in [viewer.overlay_view, *viewer.split_views]:
            np.testing.assert_allclose(tuple(view.cameraParams()["center"]), target_center)
            assert view.cameraParams()["distance"] == pytest.approx(expected_distance)
            assert view.cameraParams()["azimuth"] == pytest.approx(before_camera["azimuth"])
            assert view.cameraParams()["elevation"] == pytest.approx(before_camera["elevation"])
        assert viewer.slider.value() == before_frame
    for loaded, before in zip(viewer.loaded_results, raw_before):
        np.testing.assert_array_equal(loaded.result.keypoints_3d, before[0])
        if before[1] is not None:
            np.testing.assert_array_equal(loaded.result.mano_vertices, before[1])
        if before[2] is not None:
            np.testing.assert_array_equal(loaded.result.camera_translation, before[2])
        np.testing.assert_array_equal(loaded.result.focal_px, before[3])
        np.testing.assert_array_equal(loaded.result.detected, before[4])
    viewer.close()


@pytest.mark.parametrize("layout_mode", ("overlay", "split"))
def test_each_space_restores_its_last_camera_pose(tmp_path, qapp, layout_mode):
    from calibfreehands.visualization.comparison import ComparisonViewer

    def camera_pose(view):
        params = view.cameraParams()
        return (tuple(params["center"]), params["distance"], params["azimuth"], params["elevation"])

    viewer = ComparisonViewer(_session(tmp_path, 3, gt_index=1))
    if layout_mode == "split":
        viewer.set_mode("split")
        source = viewer.split_views[0]
    else:
        source = viewer.overlay_view
    initial_pose = camera_pose(source)
    stored_initial = viewer._camera_states["physical"]
    np.testing.assert_allclose(tuple(stored_initial["center"]), initial_pose[0])
    assert stored_initial["distance"] == pytest.approx(initial_pose[1])
    assert stored_initial["azimuth"] == pytest.approx(initial_pose[2])
    assert stored_initial["elevation"] == pytest.approx(initial_pose[3])
    physical_pose = {"pos": (0.4, -0.8, 1.1), "distance": 5.4, "azimuth": 27.0, "elevation": 13.0}
    source.setCameraPosition(**physical_pose)
    viewer._camera_interacted(source)
    expected_physical = camera_pose(source)

    viewer.set_space_mode("hand")
    first_hand = source.cameraParams()
    hand_workspace = viewer.overlay_view.workspace
    np.testing.assert_allclose(tuple(first_hand["center"]), hand_workspace.axis_intersection_display)
    expected_hand_center, expected_hand_distance = viewer._camera_framing_for("cfg-1", "hand")
    np.testing.assert_allclose(tuple(first_hand["center"]), expected_hand_center)
    assert first_hand["distance"] == pytest.approx(expected_hand_distance)
    assert first_hand["azimuth"] == pytest.approx(physical_pose["azimuth"])
    assert first_hand["elevation"] == pytest.approx(physical_pose["elevation"])

    hand_pose = {"pos": (-1.3, -2.1, 0.7), "distance": 2.8, "azimuth": 142.0, "elevation": 34.0}
    source.setCameraPosition(**hand_pose)
    viewer._camera_interacted(source)
    expected_hand = camera_pose(source)

    viewer.set_space_mode("physical")
    np.testing.assert_allclose(camera_pose(source)[0], expected_physical[0])
    assert camera_pose(source)[1:] == pytest.approx(expected_physical[1:])
    viewer.set_space_mode("hand")
    np.testing.assert_allclose(camera_pose(source)[0], expected_hand[0])
    assert camera_pose(source)[1:] == pytest.approx(expected_hand[1:])
    for view in [viewer.overlay_view, *viewer.split_views]:
        np.testing.assert_allclose(camera_pose(view)[0], expected_hand[0])
        assert camera_pose(view)[1:] == pytest.approx(expected_hand[1:])
    viewer.close()


def test_gt_changes_refocus_both_space_poses_without_rebase_drift(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer
    from calibfreehands.visualization.viewer import whole_video_hand_framing

    viewer = ComparisonViewer(_session(tmp_path, 3, gt_index=0))
    source = viewer.overlay_view
    source.setCameraPosition(pos=(0.25, -0.75, 1.2), distance=4.5,
                             azimuth=24.0, elevation=10.0)
    viewer._camera_interacted(source)
    viewer.set_space_mode("hand")
    viewer.overlay_view.setCameraPosition(pos=(1.0, 2.0, 3.0), distance=7.0,
                                          azimuth=142.0, elevation=34.0)
    viewer._camera_interacted(viewer.overlay_view)
    viewer.set_space_mode("physical")

    # A large, focal-like GT displacement and extent change must drive camera framing.
    target_result = viewer.loaded_results[1].result
    target_result.keypoints_3d[..., 0] += 120.0
    target_result.keypoints_3d[..., 2] += 25.0
    target_result.keypoints_3d *= 8.0
    viewer.hand_view_cache.pop("cfg-1", None)
    viewer.hand_space_cache.pop("cfg-1", None)
    expected_target, expected_distance = whole_video_hand_framing(
        target_result.keypoints_3d, target_result.detected)
    expected_hand_center, expected_hand_distance = viewer._camera_framing_for("cfg-1", "hand")
    viewer.reference_selector.setCurrentIndex(viewer.reference_selector.findData("cfg-1"))

    physical = viewer.overlay_view.cameraParams()
    np.testing.assert_allclose(tuple(physical["center"]), expected_target)
    assert physical["distance"] == pytest.approx(expected_distance)
    assert physical["azimuth"] == pytest.approx(24.0)
    assert physical["elevation"] == pytest.approx(10.0)
    assert not hasattr(viewer, "_pending_hand_camera_rebase")
    assert viewer._camera_states["hand"]["distance"] == pytest.approx(expected_hand_distance)
    np.testing.assert_allclose(tuple(viewer._camera_states["hand"]["center"]), expected_hand_center)
    assert viewer._camera_states["hand"]["azimuth"] == pytest.approx(142.0)
    assert viewer._camera_states["hand"]["elevation"] == pytest.approx(34.0)

    viewer.set_space_mode("hand")
    hand = viewer.overlay_view.cameraParams()
    np.testing.assert_allclose(tuple(hand["center"]), expected_hand_center)
    assert hand["distance"] == pytest.approx(expected_hand_distance)
    assert hand["azimuth"] == pytest.approx(142.0)
    assert hand["elevation"] == pytest.approx(34.0)

    for configuration_id in ("cfg-2", "cfg-0", "cfg-1"):
        viewer.reference_selector.setCurrentIndex(viewer.reference_selector.findData(configuration_id))
        active = viewer.overlay_view.cameraParams()
        expected, distance = viewer._camera_framing_for(configuration_id, "hand")
        np.testing.assert_allclose(tuple(active["center"]), expected)
        assert active["distance"] == pytest.approx(distance)
        assert not hasattr(viewer, "_pending_hand_camera_rebase")
    viewer.close()


def test_hand_gt_refocus_uses_rendered_workspace_axis_intersection(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer
    from calibfreehands.visualization.viewer import workspace_camera_framing, workspace_geometry

    viewer = ComparisonViewer(_session(tmp_path, 3, gt_index=0))
    result = viewer.loaded_results[1].result
    result.keypoints_3d[:, :, 1:, 0] += 6.0
    viewer.hand_view_cache.pop("cfg-1", None)
    viewer.hand_space_cache.pop("cfg-1", None)
    gt_center, _gt_distance = viewer._hand_view_for("cfg-1")
    hand_workspace = workspace_geometry("hand", viewer._hand_space_for("cfg-1"))
    hand_center = hand_workspace.display_center
    assert not np.allclose(gt_center, hand_center)
    active_view = viewer._camera_source()
    aspect = active_view.width() / active_view.height()
    expected_workspace_center, expected_hand_distance = workspace_camera_framing(
        hand_workspace, active_view.opts["fov"], aspect)
    np.testing.assert_allclose(expected_workspace_center, hand_center)

    viewer.reference_selector.setCurrentIndex(viewer.reference_selector.findData("cfg-1"))
    np.testing.assert_allclose(tuple(viewer._camera_states["physical"]["center"]), gt_center)
    np.testing.assert_allclose(tuple(viewer._camera_states["hand"]["center"]), hand_center)
    assert viewer._camera_states["physical"]["distance"] > 0
    assert viewer._camera_states["hand"]["distance"] == pytest.approx(expected_hand_distance)

    viewer.set_space_mode("hand")
    for view in [viewer.overlay_view, *viewer.split_views]:
        np.testing.assert_allclose(tuple(view.cameraParams()["center"]), hand_center)
        axis_centers = view.axis.pos.reshape(3, 2, 3).mean(axis=1)
        np.testing.assert_allclose(axis_centers, np.repeat(hand_center[None, :], 3, axis=0))
    assert viewer.overlay_view.cameraParams()["distance"] == pytest.approx(expected_hand_distance)
    viewer.close()


def test_gt_refocus_syncs_split_detached_views_and_preserves_playback_order(
        tmp_path, qapp, monkeypatch):
    from calibfreehands.visualization import comparison
    from calibfreehands.visualization.comparison import ComparisonViewer
    from calibfreehands.visualization.viewer import whole_video_hand_framing

    class FakeClock:
        value = 0.0

        def __call__(self):
            return self.value

    now = FakeClock()
    monkeypatch.setattr(comparison, "monotonic", now)
    viewer = ComparisonViewer(_session(tmp_path, 3, gt_index=0))
    viewer.set_mode("split")
    viewer.reorder_result("cfg-1", "cfg-0")
    viewer.detach_all()
    source = viewer.split_views[0]
    source.setCameraPosition(pos=(0.4, -0.8, 1.1), distance=5.4,
                             azimuth=27.0, elevation=13.0)
    viewer._camera_interacted(source)
    viewer.slider.setValue(1)
    panel_order = list(viewer.split_panel_order)
    viewer._play()
    now.value = 0.15
    playback_time_before = viewer.playback_clock.source_time_at()

    target = viewer.loaded_results[2].result
    target.keypoints_3d[..., 0] += 40
    target.keypoints_3d[..., 2] *= 5
    viewer.hand_view_cache.pop("cfg-2", None)
    viewer.hand_space_cache.pop("cfg-2", None)
    expected_center, expected_distance = whole_video_hand_framing(
        target.keypoints_3d, target.detected)
    viewer.reference_selector.setCurrentIndex(viewer.reference_selector.findData("cfg-2"))

    assert viewer.timer.isActive()
    assert viewer.slider.value() == 1
    assert viewer.playback_clock.source_time_at() == pytest.approx(playback_time_before)
    assert viewer.split_panel_order == panel_order
    assert viewer.reference_id == "cfg-2"
    assert viewer.configuration_colors[2] == pytest.approx((0x33 / 255, 1.0, 0x33 / 255))
    all_views = [viewer.overlay_view, *viewer.split_views,
                 *viewer._detached_windows[0][2]]
    for view in all_views:
        params = view.cameraParams()
        np.testing.assert_allclose(tuple(params["center"]), expected_center)
        assert params["distance"] == pytest.approx(expected_distance)
        assert params["azimuth"] == pytest.approx(27.0)
        assert params["elevation"] == pytest.approx(13.0)
    viewer._play()
    viewer._detached_windows[0][0].close()
    viewer.close()


def test_invalid_gt_geometry_refocuses_to_finite_fallback(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2, gt_index=0))
    invalid = viewer.loaded_results[1].result
    invalid.detected[:] = False
    invalid.keypoints_3d[:] = np.nan
    viewer.hand_view_cache.pop("cfg-1", None)
    viewer.hand_space_cache.pop("cfg-1", None)
    viewer.reference_selector.setCurrentIndex(viewer.reference_selector.findData("cfg-1"))
    viewer.set_space_mode("hand")
    params = viewer.overlay_view.cameraParams()
    assert np.isfinite(tuple(params["center"])).all()
    assert np.isfinite(params["distance"]) and params["distance"] > 0
    assert not hasattr(viewer, "_pending_hand_camera_rebase")
    np.testing.assert_allclose(tuple(params["center"]),
                               viewer.overlay_view.workspace.axis_intersection_display)
    axis_segments = viewer.overlay_view.axis.pos.reshape(3, 2, 3)
    axis_intersection = np.asarray(tuple(params["center"]))
    for start, end in axis_segments:
        direction = end - start
        fraction = np.dot(axis_intersection - start, direction) / np.dot(direction, direction)
        assert 0.0 <= fraction <= 1.0
        np.testing.assert_allclose(start + fraction * direction, axis_intersection)
    viewer.close()


def test_playback_frames_stay_synchronized_through_repeated_overlay_resizes(
        tmp_path, qapp, monkeypatch):
    from calibfreehands.visualization import comparison
    from calibfreehands.visualization.comparison import ComparisonViewer

    class FakeClock:
        value = 0.0

        def __call__(self):
            return self.value

    now = FakeClock()
    monkeypatch.setattr(comparison, "monotonic", now)
    viewer = ComparisonViewer(_session(tmp_path, 3))
    viewer.resize(1280, 800)
    viewer.show()
    viewer._play()
    for frame, elapsed in ((1, 0.12), (2, 0.22), (1, 0.45)):
        now.value = elapsed
        viewer._advance()
        sizes = viewer.overlay_splitter.sizes()
        delta = 12 if frame % 2 else -12
        left = max(100, min(viewer.overlay_splitter.width() - 100, sizes[0] + delta))
        viewer.overlay_splitter.setSizes([left, viewer.overlay_splitter.width() - left])
        qapp.processEvents()
        assert viewer.requested_frame == viewer.presented_frame == frame
        assert viewer.source_overlay.frame == frame
        assert viewer.overlay_view.frame == frame
    assert viewer.timer.isActive()
    viewer._play()
    viewer.close()


def test_camera_orientation_presets_and_recovery_have_deterministic_params(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer
    from calibfreehands.visualization.viewer import default_metric_camera_distance, whole_video_hand_framing

    viewer = ComparisonViewer(_session(tmp_path, 3, gt_index=1))
    view = viewer.overlay_view
    view.setCameraPosition(pos=(0.3, -0.9, 1.4), distance=6.2, azimuth=31.0, elevation=16.0)
    center = tuple(view.cameraParams()["center"])
    distance = view.cameraParams()["distance"]
    expected_orientations = {
        "front_view": (90.0, 0.0), "left_view": (0.0, 0.0),
        "right_view": (180.0, 0.0), "top_view": (0.0, 90.0),
    }
    for method, (azimuth, elevation) in expected_orientations.items():
        viewer._apply_camera_preset(method)
        params = view.cameraParams()
        assert tuple(params["center"]) == pytest.approx(center)
        assert params["distance"] == pytest.approx(distance)
        assert params["azimuth"] == pytest.approx(azimuth)
        assert params["elevation"] == pytest.approx(elevation)

    result = viewer.loaded_results[1].result
    viewer.loaded_results[2].result.keypoints_3d[...] += 20.0
    focus_target, focus_distance = whole_video_hand_framing(result.keypoints_3d, result.detected)
    view.setCameraPosition(pos=(30.0, 30.0, 30.0), distance=50.0, azimuth=220.0, elevation=-15.0)
    viewer._apply_camera_preset("focus_gt")
    focused = view.cameraParams()
    np.testing.assert_allclose(tuple(focused["center"]), focus_target)
    assert focused["distance"] == pytest.approx(focus_distance)

    viewer._apply_camera_preset("fit_all")
    fit_center = tuple(view.cameraParams()["center"])
    assert fit_center != pytest.approx(focus_target)
    viewer._apply_camera_preset("focus_gt")
    np.testing.assert_allclose(tuple(view.cameraParams()["center"]), focus_target)
    assert view.cameraParams()["distance"] == pytest.approx(focus_distance)

    viewer.set_space_mode("physical")
    viewer._apply_camera_preset("fit_all")
    viewer._apply_camera_preset("space_overview")
    overview = view.cameraParams()
    np.testing.assert_allclose(tuple(overview["center"]), (0.0, -1.0, 0.0))
    assert overview["distance"] == pytest.approx(default_metric_camera_distance())
    viewer.close()


def test_viewport_chrome_plain_text_stays_light_on_light_and_dark_palettes(tmp_path, qapp):
    from PySide6.QtGui import QColor, QPalette
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2))
    for background in (QColor("#f8f8f8"), QColor("#202020")):
        palette = QPalette(viewer.palette())
        palette.setColor(QPalette.ColorRole.Window, background)
        viewer.setPalette(palette)
        for label in viewer.chrome_text_labels:
            assert "#f2f3f4" in label.styleSheet().lower()
    assert not viewer.primary_controls.autoFillBackground()
    viewer.close()


def test_metric_pair_cache_reuses_ordered_pairs_when_gt_changes(monkeypatch, tmp_path, qapp):
    import calibfreehands.visualization.comparison as comparison

    viewer = comparison.ComparisonViewer(_session(tmp_path, 4, gt_index=0))
    initial_id = viewer.reference_id
    candidate_ids = [config["id"] for config in viewer.configurations if config["id"] != initial_id]
    calls = []
    original = comparison.compare_results

    def counted(reference, candidate):
        calls.append((id(reference), id(candidate)))
        return original(reference, candidate)

    monkeypatch.setattr(comparison, "compare_results", counted)
    viewer.reference_selector.setCurrentIndex(viewer.reference_selector.findData(candidate_ids[0]))
    _wait_for_metric_job(viewer, qapp)
    assert len(calls) == len(viewer.configurations) - 1
    reverse_pair = (candidate_ids[0], initial_id)
    assert reverse_pair in viewer._metric_pair_cache
    assert (initial_id, candidate_ids[0]) in viewer._metric_pair_cache
    assert reverse_pair != (initial_id, candidate_ids[0])

    viewer.reference_selector.setCurrentIndex(viewer.reference_selector.findData(initial_id))
    _wait_for_metric_job(viewer, qapp)
    assert len(calls) == len(viewer.configurations) - 1
    viewer.reference_selector.setCurrentIndex(viewer.reference_selector.findData(candidate_ids[0]))
    assert len(calls) == len(viewer.configurations) - 1
    viewer.close()


def test_uncached_large_metric_job_is_async_and_stale_reference_is_discarded(monkeypatch, tmp_path, qapp):
    from dataclasses import fields, replace
    from threading import Event
    import calibfreehands.visualization.comparison as comparison

    viewer = comparison.ComparisonViewer(_session(tmp_path, 4, gt_index=0))
    for loaded in viewer.loaded_results:
        result = loaded.result
        count = len(result.frame_indices)
        expanded = {}
        for field in fields(result):
            value = getattr(result, field.name)
            if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == count:
                expanded[field.name] = np.repeat(value[:1], 252, axis=0)
        expanded["frame_indices"] = np.arange(252)
        expanded["timestamps"] = np.arange(252, dtype=np.float64) / 30
        loaded.result = replace(result, **expanded)

    entered = Event()
    release = Event()
    original = comparison.compare_results
    blocked_once = False

    def delayed(reference, candidate):
        nonlocal blocked_once
        if not blocked_once:
            blocked_once = True
            entered.set()
            assert release.wait(5)
        return original(reference, candidate)

    monkeypatch.setattr(comparison, "compare_results", delayed)
    first_id, latest_id = "cfg-1", "cfg-2"
    viewer.reference_selector.setCurrentIndex(viewer.reference_selector.findData(first_id))
    assert viewer.metric_status.text().startswith("Updating comparison metrics")
    assert not viewer.export_csv_button.isEnabled()
    assert not viewer.export_json_button.isEnabled()
    assert entered.wait(2)
    viewer.reference_selector.setCurrentIndex(viewer.reference_selector.findData(latest_id))
    release.set()
    _wait_for_metric_job(viewer, qapp)

    assert viewer.reference_id == latest_id
    assert set(viewer._metric_cache) == {"cfg-0", "cfg-1", "cfg-3"}
    for candidate_id, pair_cache in viewer._metric_cache.items():
        assert pair_cache is viewer._metric_pair_cache[(latest_id, candidate_id)]
    assert viewer.export_csv_button.isEnabled()
    assert viewer.export_json_button.isEnabled()
    assert all(key[0] in {"cfg-0", first_id, latest_id} for key in viewer._metric_pair_cache)
    viewer.close()


def test_reference_change_performs_one_coherent_panel_refresh(monkeypatch, tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 3, gt_index=0))
    refresh_count = 0
    render_count = 0
    original_refresh = viewer._refresh_metric_panels
    original_render = viewer._render_active_frame

    def counted_refresh(*args, **kwargs):
        nonlocal refresh_count
        refresh_count += 1
        return original_refresh(*args, **kwargs)

    def counted_render(*args, **kwargs):
        nonlocal render_count
        render_count += 1
        return original_render(*args, **kwargs)

    monkeypatch.setattr(viewer, "_refresh_metric_panels", counted_refresh)
    monkeypatch.setattr(viewer, "_render_active_frame", counted_render)
    viewer.reference_selector.setCurrentIndex(viewer.reference_selector.findData("cfg-1"))
    assert refresh_count == 1
    assert render_count == 0
    _wait_for_metric_job(viewer, qapp)
    viewer.close()


def test_camera_actions_frame_distinct_gt_workspace_and_visible_union(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 3, gt_index=1))
    camera_menu = next(action.menu() for action in viewer.more_menu.actions()
                       if action.menu() is not None and action.text() == "Camera")
    assert [action.text() for action in camera_menu.actions()] == [
        "Front", "Left", "Right", "Top", "Focus GT", "Space Overview", "Fit All"
    ]
    initial = viewer.overlay_view.cameraParams()
    viewer.focus_gt()
    focus = viewer.overlay_view.cameraParams()
    viewer.space_overview()
    overview = viewer.overlay_view.cameraParams()

    assert focus["distance"] != pytest.approx(overview["distance"])
    assert tuple(focus["center"]) != pytest.approx(tuple(overview["center"]))
    assert initial["distance"] == pytest.approx(focus["distance"])

    viewer.space_mode_switch.setChecked(True)
    viewer.space_overview()
    hand_overview = viewer.overlay_view.cameraParams()
    assert hand_overview["distance"] < overview["distance"]
    assert tuple(hand_overview["center"]) != pytest.approx(tuple(overview["center"]))

    viewer.loaded_results[2].result.keypoints_3d[:] = (0.0, 0.0, 15.0)
    assert viewer.fit_all()
    fitted = viewer.overlay_view.cameraParams()
    assert fitted["distance"] > hand_overview["distance"]
    assert tuple(fitted["center"]) != pytest.approx(tuple(hand_overview["center"]))
    viewer.close()


def test_single_cfh_viewer_uses_sole_result_for_hand_space_and_focus(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer
    from calibfreehands.visualization.viewer import hand_space_bounds

    viewer = ComparisonViewer(_session(tmp_path, 1))
    assert viewer.single_mode
    assert viewer.space_mode == "physical"
    viewer.space_mode_switch.setChecked(True)
    expected = hand_space_bounds(viewer.loaded_results[0].result.keypoints_3d,
                                 viewer.loaded_results[0].result.detected)
    np.testing.assert_allclose(viewer.overlay_view.workspace_bounds_m, expected)
    hand_camera = viewer.overlay_view.cameraParams()
    np.testing.assert_allclose(tuple(hand_camera["center"]),
                               viewer.overlay_view.workspace.axis_intersection_display)
    expected_center, expected_distance = viewer._camera_framing_for("cfg-0", "hand")
    np.testing.assert_allclose(tuple(hand_camera["center"]), expected_center)
    assert hand_camera["distance"] == pytest.approx(expected_distance)
    camera_menu = next(action.menu() for action in viewer.more_menu.actions()
                       if action.menu() is not None and action.text() == "Camera")
    assert "Focus Hands" in [action.text() for action in camera_menu.actions()]
    frame = viewer.slider.value()
    viewer.focus_gt()
    assert viewer.slider.value() == frame
    viewer.close()


def test_scale_labels_switch_preserves_playback_and_workspace_visibility_state(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 3))
    viewer.slider.setValue(1)
    viewer.timer.start(1000)
    before_frame = viewer.slider.value()
    before_camera = viewer.overlay_view.cameraParams()
    before_grid = tuple(grid.visible() for grid in viewer.overlay_view.grids)
    before_axis = viewer.overlay_view.axis.visible()
    for visible in (False, True, False, True, False, True):
        viewer.labels_switch.setChecked(visible)
        assert viewer.timer.isActive()
        assert viewer.slider.value() == before_frame
        assert viewer.overlay_view.cameraParams() == before_camera
        assert tuple(grid.visible() for grid in viewer.overlay_view.grids) == before_grid
        assert viewer.overlay_view.axis.visible() == before_axis
        assert all(label.visible() == visible for label in viewer.overlay_view.numeric_labels)
    viewer.timer.stop()
    viewer.close()


def test_camera_direction_presets_preserve_manually_changed_center_and_distance(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 3))
    viewer.set_mode("split")
    source = viewer.split_views[0]
    source.setCameraPosition(pos=(0.25, -0.75, 1.2), distance=4.5, azimuth=24.0, elevation=10.0)
    before = source.cameraParams()
    for method in ("front_view", "side_view", "top_view"):
        viewer._apply_camera_preset(method)
        after = viewer.split_views[0].cameraParams()
        np.testing.assert_allclose(tuple(after["center"]), tuple(before["center"]))
        assert after["distance"] == pytest.approx(before["distance"])
        for view in viewer.split_views[1:]:
            np.testing.assert_allclose(tuple(view.cameraParams()["center"]), tuple(before["center"]))
            assert view.cameraParams()["distance"] == pytest.approx(before["distance"])
    viewer.close()


def test_comparison_mesh_visibility_remains_exclusive_with_skeleton(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer
    viewer = ComparisonViewer(_session(tmp_path, mesh=True))
    viewer.mesh_checkbox.setChecked(True)
    assert viewer.mesh_checkbox.isChecked()
    assert not viewer.joint_checkbox.isChecked() and not viewer.bone_checkbox.isChecked()
    assert viewer.overlay_view.meshes[0].opts["color"][3] == pytest.approx(1.0)
    viewer.joint_checkbox.setChecked(True)
    assert viewer.joint_checkbox.isChecked() and not viewer.mesh_checkbox.isChecked()
    assert viewer.overlay_view.joints[0].visible()
    assert not viewer.overlay_view.meshes[0].visible()
    viewer.close()


def test_gt_switching_recolors_results_and_keeps_candidate_base_colors(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 3, gt_index=0))
    green = (0x33 / 255, 1.0, 0x33 / 255)
    original = list(viewer.candidate_colors)
    assert original[1] == pytest.approx((0x33 / 255, 0x33 / 255, 1.0))
    assert original[2] == pytest.approx((1.0, 0x33 / 255, 1.0))
    viewer._apply_configuration_color(0, (0.25, 0.4, 0.8))
    custom_a = viewer.candidate_colors[0]
    assert viewer.configuration_colors[0] == pytest.approx(green)
    assert viewer.reference_label.text() == "GT"
    assert viewer.overlay_panel.reference_badges["cfg-0"].text() == "GT"
    assert (viewer.overlay_panel.group_widgets["cfg-0"].layout().itemAt(0).layout()
            .itemAt(2).widget().text() == viewer.configurations[0]["name"])
    initial_mpjpe = viewer._metric_cache["cfg-2"][(0, "left")].metrics["mpjpe"]
    viewer.set_annotation_mode("all")
    viewer._set_annotation_content("error")
    initial_labels = viewer.split_annotations[2].projected_annotations()
    initial_gt2_label = next(item[3] for item in initial_labels if item[1] == "left" and item[2] == 0)

    for selected_index, previous_index in ((1, 0), (2, 1), (0, 2)):
        viewer.reference_selector.setCurrentIndex(selected_index)
        _wait_for_metric_job(viewer, qapp)
        assert viewer.configuration_colors[selected_index] == pytest.approx(green)
        assert viewer.configuration_colors[previous_index] == pytest.approx(
            custom_a if previous_index == 0 else original[previous_index]
        )
        selected_id = viewer.configurations[selected_index]["id"]
        assert (viewer.overlay_panel.group_widgets[selected_id].layout().itemAt(0).layout()
                .itemAt(2).widget().text() == viewer.configurations[selected_index]["name"])
        assert list(viewer.overlay_panel.reference_badges) == [f"cfg-{selected_index}"]
        assert viewer.color_buttons[selected_index].toolTip() == "GT color is fixed to #33FF33"
        assert not viewer.color_buttons[selected_index].isEnabled()
        if selected_index == 1:
            assert viewer._metric_cache["cfg-2"][(0, "left")].metrics["mpjpe"] != initial_mpjpe
            updated_labels = viewer.split_annotations[2].projected_annotations()
            updated_gt2_label = next(item[3] for item in updated_labels if item[1] == "left" and item[2] == 0)
            assert updated_gt2_label != initial_gt2_label
    assert viewer.candidate_colors[0] == pytest.approx(custom_a)
    assert viewer.configuration_colors[1] == pytest.approx(original[1])
    assert viewer.configuration_colors[2] == pytest.approx(original[2])
    viewer.close()


def test_result_colors_are_stable_across_hands_and_confidence_only_changes_alpha(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer, _bone_color

    viewer = ComparisonViewer(_session(tmp_path, 2, mesh=True, gt_index=0))
    viewer.set_mode("overlay")
    gt_color = viewer.configuration_colors[0]
    candidate_color = viewer.configuration_colors[1]
    for hand in range(2):
        np.testing.assert_allclose(viewer.overlay_view.joints[hand].color[:3], gt_color)
        np.testing.assert_allclose(viewer.overlay_view.bones[hand].color[:3], _bone_color(gt_color))
    viewer.mesh_checkbox.setChecked(True)
    np.testing.assert_allclose(viewer.overlay_view.meshes[0].opts["color"][:3], gt_color)
    viewer.mesh_checkbox.setChecked(False)
    viewer.joint_checkbox.setChecked(True)
    viewer.bone_checkbox.setChecked(True)
    viewer.set_mode("overlay")
    viewer.confidence_opacity_checkbox.setChecked(True)
    high = viewer.overlay_view.extra_items[0][0][0].color
    low = viewer.overlay_view.joints[0].color
    np.testing.assert_allclose(high[:3], candidate_color)
    np.testing.assert_allclose(low[:3], gt_color)
    assert high[3] > low[3]
    viewer.confidence_opacity_checkbox.setChecked(False)
    assert viewer.overlay_view.joints[0].color[3] == pytest.approx(1.0)
    viewer.set_mode("split")
    for hand in range(2):
        np.testing.assert_allclose(viewer.split_views[1].joints[hand].color[:3], candidate_color)
        np.testing.assert_allclose(viewer.split_views[1].bones[hand].color[:3], _bone_color(candidate_color))
    viewer.mesh_checkbox.setChecked(True)
    np.testing.assert_allclose(viewer.split_views[0].meshes[0].opts["color"][:3], gt_color)
    np.testing.assert_allclose(viewer.split_views[1].meshes[0].opts["color"][:3], candidate_color)
    viewer.confidence_opacity_checkbox.setChecked(True)
    gt_mesh = viewer.split_views[0].meshes[0].opts["color"]
    candidate_mesh = viewer.split_views[1].meshes[0].opts["color"]
    np.testing.assert_allclose(gt_mesh[:3], gt_color)
    np.testing.assert_allclose(candidate_mesh[:3], candidate_color)
    assert candidate_mesh[3] > gt_mesh[3]
    viewer.confidence_opacity_checkbox.setChecked(False)
    viewer.close()


def test_comparison_toolbar_wording_and_unavailable_metrics(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 3, mesh=True))
    assert viewer.results_menu_button.text() == "Visible (3/3)"
    assert viewer.results_menu_button.toolTip() == "Show or hide comparison results"
    assert viewer.reference_label.text() == "GT"
    assert viewer.metric_menu_button.text() == "Error Types"
    assert viewer.metric_menu_button.toolTip() == "Select error types shown in the HUD"
    assert viewer.confidence_menu_button.text() == "Confidence Display"
    assert viewer.configuration_colors[1] == pytest.approx((0x33 / 255, 0x33 / 255, 1.0))
    assert not viewer.confidence_opacity_checkbox.isChecked()
    assert not viewer.confidence_numeric_checkbox.isChecked()
    assert not viewer.confidence_bar_checkbox.isChecked()
    assert viewer.toolbar.indexOf(viewer.error_vectors_button) == -1
    assert not hasattr(viewer, "confidence_color_mode")
    assert all(definition.status.value == "READY" for definition in viewer.metric_definitions
               if definition.metric_id in viewer.metric_checks)
    assert "f@5" not in viewer.metric_checks
    viewer.close()


def test_gt_default_prefers_unique_checkerboard_and_metric_menu_rebuilds(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 3, mesh=True, gt_index=2))
    assert viewer.reference_id == "cfg-2"
    assert "mpvpe" in viewer.metric_checks
    viewer.metric_checks["mpvpe"].setChecked(True)
    viewer.loaded_results[1].result.mano_vertices = None
    viewer.loaded_results[1].result.mano_faces = None
    viewer.reference_selector.setCurrentIndex(1)
    assert "mpvpe" not in viewer.metric_checks
    assert "mpvpe" not in viewer.selected_metric_ids
    viewer.close()


def test_phase_a_viewer_menu_wording_order_and_hud_visibility(tmp_path, qapp):
    from PySide6.QtWidgets import QLabel, QToolButton, QWidgetAction
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2, mesh=True))
    assert viewer.metric_menu_button.text() == "Error Types"
    assert viewer.confidence_menu_button.text() == "Confidence Display"
    toolbar_labels = [item.text() for index in range(viewer.toolbar.count())
                      if isinstance(item := viewer.toolbar.itemAt(index).widget(), QToolButton)
                      and item.text() in {"Error Types", "Confidence Display", "Annotations", "HUD", "More"}]
    assert toolbar_labels == ["Error Types", "Confidence Display", "Annotations", "HUD", "More"]
    assert [action.text() for action in viewer.annotation_menu.actions()] == [
        "All joints", "Hovered joint", "Key joints", "Worst 5", "Off", "", "Label"
    ]
    assert viewer.annotation_actions["off"].isChecked()
    hud_items = [action.defaultWidget().findChild(QLabel).text()
                 if isinstance(action, QWidgetAction) else action.text()
                 for action in viewer.hud_menu.actions()]
    assert hud_items == ["HUD opacity", "Size", "Background"]

    assert viewer.hud_button.isCheckable() and viewer.hud_button.isChecked()
    assert viewer.hud_button.toolTip() == "Hide HUD"
    viewer._set_hud(anchor="bottom-right")
    viewer.hud_menu.actions()[1].menu().actions()[-1].trigger()
    viewer.hud_background_actions["subtle"].trigger()
    assert all(panel.scale == "large" and panel.background == "subtle"
               for panel in (viewer.overlay_panel, *viewer.split_panels))
    viewer.hud_button.setChecked(False)
    assert all(panel.isHidden() for panel in (viewer.overlay_panel, *viewer.split_panels))
    assert viewer.hud_button.toolTip() == "Show HUD"
    viewer.hud_opacity_slider.setValue(60)
    assert all(panel.opacity == pytest.approx(0.6)
               for panel in (viewer.overlay_panel, *viewer.split_panels))
    viewer.set_mode("split")
    assert all(panel.isHidden() for panel in (viewer.overlay_panel, *viewer.split_panels))
    viewer.hud_button.setChecked(True)
    assert all(not panel.isHidden() for panel in (viewer.overlay_panel, *viewer.split_panels))
    viewer.set_mode("overlay")
    assert not viewer.overlay_panel.isHidden()
    assert viewer.overlay_panel.anchor == "bottom-right"
    assert viewer.overlay_panel.scale == "large"
    assert viewer.overlay_panel.opacity == pytest.approx(0.6)
    assert viewer.overlay_panel.background == "subtle"

    labels = [action.text() for action in viewer.more_menu.actions()]
    assert "Playback Stats" in labels
    assert "Performance stats" not in labels
    assert "Auto-hide controls while playing" not in labels
    assert not hasattr(viewer, "auto_hide_action")
    assert viewer.performance_stats_action.text() == "Playback Stats"
    viewer.performance_stats_action.setChecked(True)
    assert not viewer.performance_stats_label.isHidden()
    assert all(value in viewer.performance_stats_label.text()
               for value in ("Source", "Frame updates", "Skipped"))
    assert chr(0xD7) in viewer.performance_stats_label.text()
    viewer.performance_stats_action.setChecked(False)
    assert viewer.performance_stats_label.isHidden()
    assert viewer.performance_stats_label.text() == ""
    viewer.close()


def test_pin_is_the_only_auto_hide_setting_and_unpin_restores_timer(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2))
    viewer.show()
    qapp.processEvents()
    viewer.timer.start(1000)
    viewer._auto_hide_controls()
    assert viewer.chrome.isHidden()
    viewer.pin_controls_button.setChecked(True)
    assert not viewer.chrome.isHidden()
    viewer._auto_hide_controls()
    assert not viewer.chrome.isHidden()
    viewer.pin_controls_button.setChecked(False)
    assert viewer.auto_hide_timer.isActive()
    viewer._auto_hide_controls()
    assert viewer.chrome.isHidden()
    viewer.timer.stop()
    viewer._play()
    assert not viewer.chrome.isHidden()
    viewer.close()


def test_hud_split_button_body_toggles_visibility_and_arrow_opens_settings(tmp_path, qapp):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2))
    viewer.resize(1280, 800)
    viewer.show()
    qapp.processEvents()
    button = viewer.hud_button
    from PySide6.QtWidgets import QStyle, QStyleOptionToolButton
    option = QStyleOptionToolButton()
    button.initStyleOption(option)
    arrow = button.style().subControlRect(QStyle.ComplexControl.CC_ToolButton, option,
                                          QStyle.SubControl.SC_ToolButtonMenu, button)
    assert not arrow.isEmpty()

    body = QPoint(max(0, arrow.left() - 3), button.height() // 2)
    QTest.mouseClick(button, Qt.MouseButton.LeftButton, pos=body)
    assert not button.isChecked()
    assert all(panel.isHidden() for panel in (viewer.overlay_panel, *viewer.split_panels))
    QTest.mouseClick(button, Qt.MouseButton.LeftButton, pos=arrow.center())
    qapp.processEvents()
    assert viewer.hud_menu.isVisible()
    assert not button.isChecked()
    QTest.keyClick(viewer.hud_menu, Qt.Key.Key_Escape)
    QTest.mouseClick(button, Qt.MouseButton.LeftButton, pos=body)
    assert button.isChecked()
    assert all(not panel.isHidden() for panel in (viewer.overlay_panel, *viewer.split_panels))
    viewer.close()


@pytest.mark.parametrize("style_name", ["Fusion", "Windows", "windowsvista", "windows11"])
def test_hud_menu_opacity_row_matches_submenu_rows_in_dark_and_light_palettes(tmp_path, qapp, style_name):
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QColor, QPalette
    from PySide6.QtWidgets import QStyleFactory, QWidgetAction
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2))
    menu = viewer.hud_menu
    opacity_action, size_action, background_action = menu.actions()
    assert isinstance(opacity_action, QWidgetAction)
    for widget in (viewer, menu, viewer.hud_background_menu):
        widget.setStyle(QStyleFactory.create(style_name))
    viewer._fit_hud_opacity_row()

    for background, foreground in (("#202020", "#f2f2f2"), ("#f8f8f8", "#202020")):
        palette = QPalette(viewer.palette())
        palette.setColor(QPalette.ColorRole.Window, QColor(background))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(foreground))
        viewer.setPalette(palette)
        menu.setPalette(palette)
        menu.ensurePolished()
        menu.popup(QPoint(20, 20))
        qapp.processEvents()

        opacity_rect = menu.actionGeometry(opacity_action)
        size_rect = menu.actionGeometry(size_action)
        background_rect = menu.actionGeometry(background_action)
        assert abs(opacity_rect.height() - size_rect.height()) <= 1
        assert abs(opacity_rect.height() - background_rect.height()) <= 1
        slider_center = viewer.hud_opacity_slider.mapTo(menu, viewer.hud_opacity_slider.rect().center())
        assert abs(slider_center.y() - opacity_rect.center().y()) <= 1
        label_center = viewer.hud_opacity_label.mapTo(menu, viewer.hud_opacity_label.rect().center())
        assert abs(label_center.y() - opacity_rect.center().y()) <= 1
        assert menu.width() <= 280
        menu.hide()
        qapp.processEvents()
    assert viewer.hud_opacity_slider.minimum() == 20
    assert viewer.hud_opacity_slider.maximum() == 100
    assert viewer.hud_opacity_slider.value() == 100
    viewer.close()


@pytest.mark.parametrize("result_count", [1, 3])
def test_geometry_menu_actions_keep_mesh_and_skeleton_state_synchronized(tmp_path, qapp, result_count):
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, result_count, mesh=True))
    geometry_menu = next(action.menu() for action in viewer.more_menu.actions()
                         if action.menu() is not None and action.menu().title() == "Geometry")
    actions = {action.text(): action for action in geometry_menu.actions() if action.isCheckable()}
    assert set(actions) >= {"Joints", "Bones", "Mesh"}
    joint_action, bone_action, mesh_action = actions["Joints"], actions["Bones"], actions["Mesh"]
    views = [viewer.overlay_view, *viewer.split_views]

    def assert_state(joints, bones, mesh):
        assert viewer.joint_checkbox.isChecked() is joints
        assert viewer.bone_checkbox.isChecked() is bones
        assert viewer.mesh_checkbox.isChecked() is mesh
        assert joint_action.isChecked() is joints
        assert bone_action.isChecked() is bones
        assert mesh_action.isChecked() is mesh
        assert viewer.mesh_opacity_slider.isEnabled() is mesh
        for view in views:
            assert view.show_joints is joints
            assert view.show_bones is bones
            assert view.show_mesh is mesh
        rendered_views = [viewer.overlay_view] if viewer.mode == "overlay" else viewer.split_views
        for view in rendered_views:
            for geometry in ("joints", "bones", "meshes"):
                items = list(getattr(view, geometry))
                if hasattr(view, "extra_items"):
                    item_index = {"joints": 0, "bones": 1, "meshes": 2}[geometry]
                    items.extend(hand_items[item_index] for config_items in view.extra_items
                                 for hand_items in config_items)
                assert all(item.visible() is expected for item, expected in zip(
                    items, ([joints] if geometry == "joints" else
                            [bones] if geometry == "bones" else [mesh]) * len(items)))

    assert_state(True, True, False)
    modes = ("overlay", "split") if result_count > 1 else ("overlay",)
    for mode in modes:
        viewer.set_mode(mode)
        if viewer.mesh_checkbox.isChecked():
            joint_action.trigger()
            bone_action.trigger()
            assert_state(True, True, False)
        for _ in range(3):
            mesh_action.trigger()
            assert_state(False, False, True)
            joint_action.trigger()
            assert_state(True, False, False)
            bone_action.trigger()
            assert_state(True, True, False)
            mesh_action.trigger()
            assert_state(False, False, True)
            if _ < 2:
                joint_action.trigger()
                assert_state(True, False, False)
                bone_action.trigger()
                assert_state(True, True, False)
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


def _bundle(tmp_path, name, model_id, source_name, source_sha, index, mesh=False):
    result = CanonicalResult(
        np.array([0, 1]), np.array([0., .1]), np.zeros((2, 2, 21, 2), np.float32),
        np.full((2, 2, 21, 3), index, np.float32), np.ones((2, 2), bool),
        np.full(2, 500., np.float32), (32, 24),
        mano_vertices=np.zeros((2, 2, 3, 3), np.float32) if mesh else None,
        mano_faces=np.array([[[0, 1, 2]], [[0, 2, 1]]], np.int32) if mesh else None,
        confidence=np.full((2, 2), .8, np.float32),
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


def test_unified_viewer_single_result_hides_comparison_semantics_and_keeps_split_playback(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer

    bundle = _bundle(tmp_path, "Single method", "method", "missing.mp4", "a" * 64, 1)
    bundle.metadata.pop("source_media_sha256")
    viewer = ComparisonViewer([bundle])

    assert viewer.single_mode
    assert viewer.windowTitle() == "CalibFreeHands - Result"
    assert viewer.reference_id is None
    assert viewer.reference_label.isHidden()
    assert viewer.reference_selector.isHidden()
    assert viewer.metric_menu_button.isHidden()
    assert viewer.error_vectors_button.isHidden()
    assert viewer.export_csv_button.isHidden()
    assert viewer.export_json_button.isHidden()
    assert viewer.results_menu_button.isHidden()
    assert viewer._metric_cache == {}
    assert viewer.split_panels[0].reference_badges == {}
    assert viewer.split_panels[0].metric_rows == {}
    assert not viewer.annotation_content_actions["error"].isVisible()
    assert viewer.annotation_content == "name"
    assert not viewer.split_views[0].acceptDrops()
    assert len(viewer.split_views) == 1
    assert viewer.split_grid.count() == 2
    assert viewer.speed_value_label.text() == "1.0×"
    assert viewer.playback_clock.speed == pytest.approx(1.0)
    assert viewer.presented_frame == 0
    assert viewer.loaded_results[0].source_status == "missing"
    assert viewer.loaded_results[0].media_frames is None
    with pytest.raises(ValueError, match="only support joint names"):
        viewer._set_annotation_content("error")

    viewer.set_mode("split")
    assert viewer.views.currentIndex() == 1
    assert viewer.split_grid.count() == 2
    viewer.slider.setValue(1)
    assert viewer.presented_frame == 1
    viewer.close()


def test_single_result_playback_uses_timestamp_clock_at_one_and_two_times(tmp_path, qapp, monkeypatch):
    from calibfreehands.visualization import comparison
    from calibfreehands.visualization.comparison import ComparisonViewer

    class FakeClock:
        value = 0.0
        def __call__(self):
            return self.value

    now = FakeClock()
    monkeypatch.setattr(comparison, "monotonic", now)
    viewer = ComparisonViewer(_session(tmp_path, count=1))
    viewer._play()
    now.value = 0.15
    viewer._advance()
    assert viewer.presented_frame == viewer.requested_frame == 1

    viewer._step_playback_speed(1)
    viewer._step_playback_speed(1)
    viewer._step_playback_speed(1)
    now.value = 0.20
    viewer._advance()
    assert viewer.presented_frame == viewer.requested_frame == 2
    viewer._play()
    viewer.close()


def test_single_result_static_image_uses_one_cached_source_frame(tmp_path, qapp):
    import hashlib
    from PySide6.QtGui import QImage, QColor
    from calibfreehands.media.provider import MediaFrameProvider
    from calibfreehands.visualization.comparison import ComparisonViewer

    source = tmp_path / "still.png"
    image = QImage(7, 5, QImage.Format.Format_RGB888)
    image.fill(QColor("steelblue"))
    assert image.save(str(source))
    bundle = _bundle(tmp_path, "Single image", "method", source.name,
                     hashlib.sha256(source.read_bytes()).hexdigest(), 1)
    viewer = ComparisonViewer([bundle])
    provider = viewer.loaded_results[0].media_frames
    try:
        assert isinstance(provider, MediaFrameProvider)
        assert viewer.loaded_results[0].source_status == "available"
        assert provider._is_image
        assert provider.decoded_frames == 1
        assert provider.cache_bytes == 7 * 5 * 3
        viewer.slider.setValue(1)
        assert viewer.presented_frame == 1
        assert not viewer._presented_source_image.isNull()
        viewer.show()
        qapp.processEvents()
        viewer.source_overlay.repaint()
        first_image_key = viewer.source_overlay._display_image.cacheKey()
        viewer.source_overlay.repaint()
        assert viewer.source_overlay._display_image.cacheKey() == first_image_key
    finally:
        viewer.close()
    assert provider._closed
    assert provider.cache_bytes == 0


def test_comparison_viewers_share_source_until_last_viewer_closes(tmp_path, qapp):
    import hashlib
    from PySide6.QtGui import QImage, QColor
    import calibfreehands.media.provider as media_provider
    from calibfreehands.visualization.comparison import ComparisonViewer

    source = tmp_path / "shared.png"
    image = QImage(7, 5, QImage.Format.Format_RGB888)
    image.fill(QColor("steelblue"))
    assert image.save(str(source))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    bundle = _bundle(tmp_path, "Shared source", "method", source.name, digest, 1)
    first = ComparisonViewer([bundle])
    second = ComparisonViewer([bundle])
    provider = first.loaded_results[0].media_frames
    assert provider is second.loaded_results[0].media_frames
    assert media_provider.media_provider_reference_count(provider) == 2

    first.close()
    qapp.processEvents()
    assert not provider._closed
    assert media_provider.media_provider_reference_count(provider) == 1

    second.close()
    qapp.processEvents()
    assert provider._closed
    assert provider.cache_bytes == 0
    assert media_provider.media_provider_reference_count(provider) == 0


def test_single_result_mano_mesh_toggle_remains_available(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer

    bundle = _bundle(tmp_path, "Mesh result", "method", "missing.mp4", "a" * 64, 1, mesh=True)
    viewer = ComparisonViewer([bundle])
    assert viewer.mesh_checkbox.isEnabled()
    viewer.mesh_checkbox.setChecked(True)
    assert viewer.overlay_view.show_mesh
    assert viewer.split_views[0].show_mesh
    assert viewer.mesh_opacity_slider.isEnabled()
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
    assert viewer.overlay_panel.reference_badges["cfg-0"].text() == "GT"
    assert ("cfg-1", "joint_error_median", "left") not in viewer.overlay_panel.metric_rows
    viewer.metric_checks["joint_error_median"].setChecked(True)
    assert ("cfg-1", "joint_error_median", "left") in viewer.overlay_panel.metric_rows
    viewer.metric_checks["joint_error_median"].setChecked(False)
    assert ("cfg-1", "joint_error_median", "left") not in viewer.overlay_panel.metric_rows
    viewer.reference_selector.setCurrentIndex(1)
    assert viewer.reference_selector.currentData() == "cfg-1"
    viewer.slider.setValue(2)
    assert viewer.status.text().startswith("Frame 3")
    assert viewer.split_panels[1].reference_badges["cfg-1"].text() == "GT"
    assert all(box.styleSheet() == "" for box in viewer.visibility_checks)
    assert all(viewer.color_buttons)
    viewer.close()


def test_dashboard_metric_selection_error_vectors_and_confidence_are_runtime_only(tmp_path, qapp, monkeypatch):
    import calibfreehands.visualization.comparison as comparison
    from calibfreehands.visualization.comparison import ComparisonViewer
    viewer = ComparisonViewer(_session(tmp_path, 2))
    monkeypatch.setattr(comparison, "compare_results", lambda *args: pytest.fail("metrics recomputed during frame change"))
    assert viewer.selected_metric_ids == {"wrist_error", "mpjpe", "wrist_depth_error"}
    viewer.metric_checks["translation_error"].setChecked(True)
    assert "translation_error" in viewer.selected_metric_ids
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
    assert not hasattr(viewer, "confidence_color_mode")
    assert np.asarray(viewer.overlay_view.joints[0].color)[:3] == pytest.approx(viewer.configuration_colors[0])
    viewer.confidence_opacity_checkbox.setChecked(True)
    assert viewer.confidence_numeric_checkbox and viewer.confidence_bar_checkbox
    assert viewer.confidence_numeric_checkbox.isChecked() and viewer.confidence_bar_checkbox.isChecked()
    assert viewer.error_vectors_checkbox
    assert viewer.error_candidate_selector.count() == 1
    before = [loaded.result.keypoints_3d.copy() for loaded in viewer.loaded_results]
    viewer._apply_configuration_color(1, (0.1, 0.2, 0.3))
    for loaded, array in zip(viewer.loaded_results, before):
        np.testing.assert_array_equal(loaded.result.keypoints_3d, array)
    viewer.error_vectors_checkbox.setChecked(True)
    assert any(item.visible() for item in viewer.overlay_view.error_items)
    viewer.close()


def test_error_vector_actions_keep_candidate_state_consistent_across_repeated_switches(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 3))
    candidate_ids = [config["id"] for config in viewer.configurations if config["id"] != viewer.reference_id]

    def choose(candidate_id):
        viewer.error_vector_actions[candidate_id].trigger()
        qapp.processEvents()
        assert viewer.error_vectors_checkbox.isChecked()
        assert getattr(viewer, "selected_error_candidate_id", None) == candidate_id
        expected_index = next(i for i, config in enumerate(viewer.configurations)
                              if config["id"] == candidate_id)
        assert viewer.overlay_view.error_vectors_enabled
        assert viewer.overlay_view.error_candidate == expected_index
        assert viewer.error_vector_actions[candidate_id].isChecked()
        assert viewer.error_vectors_off.isChecked() is False
        assert not viewer.error_candidate_selector.isVisible()

    def disable():
        viewer.error_vectors_off.trigger()
        qapp.processEvents()
        assert not viewer.error_vectors_checkbox.isChecked()
        assert not viewer.overlay_view.error_vectors_enabled
        assert viewer.error_vectors_off.isChecked()
        assert all(not action.isChecked() for action in viewer.error_vector_actions.values())
        assert all(not item.visible() for item in viewer.overlay_view.error_items)
        assert not viewer.error_candidate_selector.isVisible()

    for _ in range(3):
        choose(candidate_ids[0])
        disable()
        choose(candidate_ids[0])
    choose(candidate_ids[1])
    disable()
    choose(candidate_ids[1])
    viewer.close()


def test_checkerboard_historical_viewer_name_normalizes_without_changing_saved_metadata():
    import calibfreehands.visualization.comparison as comparison
    display_name = getattr(comparison, "_manual_focal_display_name", None)
    assert callable(display_name), "Manual Focal presentation normalizer is missing"

    historical_metadata = {
        "calibration_model_id": "checkerboard_reference",
        "configuration_name": "Checkerboard Reference 1500 px",
    }
    display = display_name(
        historical_metadata["calibration_model_id"], historical_metadata["configuration_name"], 1500.0
    )
    assert display == "Manual Focal 1500\u00a0px"
    assert historical_metadata["configuration_name"] == "Checkerboard Reference 1500 px"
    assert display_name("checkerboard_reference", "Configuration 2", 1500.0) == "Manual Focal 1500\u00a0px"


def test_hud_requested_size_recovers_after_constrained_viewport_and_popup(tmp_path, qapp):
    from PySide6.QtWidgets import QLabel, QWidget
    from calibfreehands.visualization.comparison_presentation import _ViewportPanel

    view = QWidget()
    view.resize(500, 500)
    panel = _ViewportPanel(view, [0])
    for index in range(5):
        panel.body_layout.addWidget(QLabel(f"Result {index}: wrist error | L 123.4 / R 234.5 mm"))
    panel.set_presentation(scale="large")
    view.show()
    panel.show()
    qapp.processEvents()
    for requested, expected_points in (("small", 8.0), ("medium", 9.0), ("large", 10.0)):
        panel.set_presentation(scale=requested)
        qapp.processEvents()
        assert panel.requested_scale == requested
        assert panel.effective_point_size == expected_points
    requested = getattr(panel, "requested_scale", None)
    effective = getattr(panel, "effective_point_size", None)
    assert requested == "large", "HUD requested size is not stored independently"
    assert effective is not None, "HUD effective fitted size is not observable"
    assert panel.requested_scale == "large"
    roomy_size = panel.effective_point_size

    original_height = panel.view.height()
    original_minimum_height = panel.view.minimumHeight()
    panel.view.setMinimumHeight(1)
    panel.view.resize(panel.view.width(), 70)
    panel._position()
    qapp.processEvents()
    assert panel.view.height() < original_height
    assert panel.effective_point_size < roomy_size

    panel.view.setMinimumHeight(original_minimum_height)
    panel.view.resize(panel.view.width(), original_height)
    panel._position()
    qapp.processEvents()
    restored_size = panel.effective_point_size
    assert restored_size == roomy_size
    assert panel.requested_scale == "large"
    view.close()

    from calibfreehands.visualization.comparison import ComparisonViewer
    viewer = ComparisonViewer(_session(tmp_path, 2))
    viewer.resize(1280, 800)
    viewer.show()
    qapp.processEvents()
    panel = viewer.overlay_panel
    viewer._set_hud(scale="large")
    qapp.processEvents()
    restored_size = panel.effective_point_size
    original_height = panel.view.height()
    viewer.results_menu_button.menu().popup(viewer.results_menu_button.mapToGlobal(
        viewer.results_menu_button.rect().bottomLeft()
    ))
    qapp.processEvents()
    assert panel.view.height() == original_height
    assert panel.effective_point_size == restored_size
    viewer.results_menu_button.menu().hide()
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


def test_gt_presentation_wording_and_export_title_keep_internal_schema(tmp_path, qapp, monkeypatch):
    import calibfreehands.visualization.comparison as comparison
    viewer = comparison.ComparisonViewer(_session(tmp_path, 2))
    assert viewer.metric_status.toolTip() == "Camera-space differences from the selected GT."
    assert "these are not ground-truth errors" not in viewer.metric_status.toolTip()
    assert any(action.menu() is not None and action.menu().title() == "GT Differences"
               for action in viewer.metric_menu.actions())
    assert all(action.menu() is None or action.menu().title() != "Reference Accuracy"
               for action in viewer.metric_menu.actions())

    dialog_calls = []
    monkeypatch.setattr(comparison.QFileDialog, "getSaveFileName",
                        lambda *args: (dialog_calls.append(args) or ("", "")))
    viewer._export_json()
    assert dialog_calls[0][1] == "Export GT differences"
    assert "reference_result_id" in comparison.METRIC_FIELDS
    assert "candidate_result_id" in comparison.METRIC_FIELDS
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
    assert "mpvpe" not in viewer.metric_checks
    viewer.close()


def test_comparison_dashboard_uses_viewport_overlays_and_compact_menus(tmp_path, qapp):
    from PySide6.QtWidgets import QSizePolicy
    from calibfreehands.visualization.comparison import ComparisonViewer
    viewer = ComparisonViewer(_session(tmp_path, 5))
    assert not hasattr(viewer, "metric_table")
    assert len(viewer.split_panels) == 5
    assert viewer.split_views[1].minimumWidth() >= 320
    assert viewer.split_views[1].minimumHeight() >= 240
    assert viewer.mode_switch.isCheckable()
    assert viewer.results_menu_button.text().startswith("Visible")
    assert viewer.confidence_menu_button.text() == "Confidence Display"
    assert len(viewer.visibility_checks) == 5
    assert all(check.parentWidget() is not viewer for check in viewer.visibility_checks)
    for button in viewer.view_buttons + viewer.playback_buttons:
        assert button.sizePolicy().horizontalPolicy() != QSizePolicy.Policy.Expanding
    assert "f@5" not in viewer.metric_checks
    viewer.close()


def test_presentation_hud_is_viewport_child_and_reorder_preserves_result_data(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer
    viewer = ComparisonViewer(_session(tmp_path, 3))
    assert viewer.overlay_panel.parentWidget() is viewer.overlay_view
    assert viewer.split_panels[1].parentWidget() is viewer.split_views[1]
    assert viewer.overlay_panel.geometry() == viewer.overlay_view.rect()
    assert viewer.split_panels[1].geometry() == viewer.split_views[1].rect()
    overlay_splitter = viewer.views.widget(0)
    assert overlay_splitter.widget(1).content is viewer.overlay_view
    original_media_cell = viewer.split_grid.itemAtPosition(0, 0).widget()
    before = [loaded.result.keypoints_3d.copy() for loaded in viewer.loaded_results]
    viewer.reorder_result("cfg-2", "cfg-0")
    assert viewer.split_result_order == [2, 0, 1]
    assert [viewer.split_grid.itemAt(index).widget().panel_id
            for index in range(viewer.split_grid.count())] == viewer._visible_split_panel_ids()
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


def test_comparison_order_has_stable_panel_ids_and_user_order_survives_gt_change(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer
    viewer = ComparisonViewer(_session(tmp_path, 3))
    assert viewer.split_panel_order == ["source-media", "result:cfg-0", "result:cfg-1", "result:cfg-2"]
    viewer.reorder_result("cfg-2", "cfg-0")
    assert viewer.result_order_ids == ["cfg-2", "cfg-0", "cfg-1"]
    viewer.reference_selector.setCurrentIndex(2)
    assert viewer.reference_id == "cfg-2"
    assert viewer.result_order_ids == ["cfg-2", "cfg-0", "cfg-1"]
    assert viewer.user_ordered_panels
    viewer.close()


def test_gt_change_keeps_initial_automatic_priority_until_manual_reorder(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer
    viewer = ComparisonViewer(_session(tmp_path, 3))
    viewer.reference_selector.setCurrentIndex(2)
    assert viewer.result_order_ids == ["cfg-2", "cfg-0", "cfg-1"]
    assert not viewer.user_ordered_panels
    viewer.close()


def test_duplicate_names_and_hidden_results_keep_stable_order_across_modes(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer
    viewer = ComparisonViewer(_session(tmp_path, 3))
    viewer.configurations[0]["name"] = "Duplicate"
    viewer.configurations[1]["name"] = "Duplicate"
    ids = [surface.panel_id for surface in viewer.split_surfaces.values()]
    viewer._set_configuration_visible(1, False)
    viewer.reorder_result("cfg-2", "cfg-0")
    expected = ["cfg-2", "cfg-0", "cfg-1"]
    viewer.set_mode("split")
    assert viewer.result_order_ids == expected
    assert viewer.split_surfaces["result:cfg-1"].isHidden()
    viewer._set_configuration_visible(1, True)
    viewer.set_mode("overlay")
    assert viewer.result_order_ids == expected
    assert not viewer.split_surfaces["result:cfg-1"].isHidden()
    assert [surface.panel_id for surface in viewer.split_surfaces.values()] == ids
    viewer.close()


def test_result_noop_reorder_does_not_claim_custom_order(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer
    viewer = ComparisonViewer(_session(tmp_path, 3))
    before = viewer.result_order_ids
    viewer.reorder_result("cfg-0", "cfg-1")
    assert viewer.result_order_ids == before
    assert not viewer.user_ordered_panels
    viewer.close()


def test_hud_result_reorder_preserves_source_slot_in_split_order(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer
    viewer = ComparisonViewer(_session(tmp_path, 3))
    viewer.split_panel_order = ["result:cfg-0", "source-media", "result:cfg-1", "result:cfg-2"]
    viewer._sync_split_layout()
    viewer.reorder_result("cfg-2", "cfg-0")
    assert viewer.split_panel_order == ["result:cfg-2", "source-media", "result:cfg-0", "result:cfg-1"]
    assert viewer.result_order_ids == ["cfg-2", "cfg-0", "cfg-1"]
    viewer.close()


def test_overlay_panel_order_swap_preserves_original_widgets(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer
    viewer = ComparisonViewer(_session(tmp_path, 3))
    splitter = viewer.views.widget(0)
    source, overlay = viewer.source_overlay, viewer.overlay_view
    viewer.reorder_overlay_panel("overlay-3d", "source-media")
    assert viewer.overlay_panel_order == ["overlay-3d", "source-media"]
    assert splitter.widget(0).content is overlay
    assert splitter.widget(1).content is source
    viewer.reorder_overlay_panel("source-media", "overlay-3d")
    assert splitter.widget(0).content is source
    assert splitter.widget(1).content is overlay
    viewer.close()


def test_viewer_surfaces_have_no_titles_or_panel_drag_handles(tmp_path, qapp):
    from PySide6.QtWidgets import QLabel
    from calibfreehands.visualization.comparison import ComparisonViewer
    viewer = ComparisonViewer(_session(tmp_path, 3))
    assert all(surface.layout().count() == 1 for surface in viewer.overlay_surfaces.values())
    assert all(surface.layout().count() == 1 for surface in viewer.split_surfaces.values())
    assert all(surface.layout().count() == 1 for surface in viewer.findChildren(type(viewer.overlay_surfaces["source-media"])))
    assert not viewer.split_surfaces["source-media"].content.findChildren(QLabel)
    assert viewer.overlay_surfaces["source-media"].content is viewer.source_overlay
    assert viewer.overlay_surfaces["overlay-3d"].content is viewer.overlay_view
    assert viewer.split_surfaces["source-media"].content is viewer.split_source_overlay
    assert all(viewer.split_surfaces[f"result:{config['id']}"].content is viewer.split_views[index]
               for index, config in enumerate(viewer.configurations))
    viewer.close()


def test_split_surface_is_in_owning_layout_before_first_show(tmp_path, qapp, monkeypatch):
    from calibfreehands.visualization.comparison_presentation import _PanelSurface
    from calibfreehands.visualization.comparison import ComparisonViewer

    original_show = _PanelSurface.show
    first_show_ownership = []

    def record_first_show(surface):
        if surface.panel_id == "source-media" or surface.panel_id.startswith("result:"):
            parent = surface.parentWidget()
            layout_index = parent.layout().indexOf(surface) if parent and parent.layout() else -1
            first_show_ownership.append((surface.panel_id, parent, layout_index))
        original_show(surface)

    monkeypatch.setattr(_PanelSurface, "show", record_first_show)
    viewer = ComparisonViewer(_session(tmp_path, 3))
    assert first_show_ownership
    assert all(parent is viewer.split_widget and index >= 0
               for _panel_id, parent, index in first_show_ownership)
    viewer.close()


def test_overlay_divider_right_click_swap_preserves_slot_sizes_and_viewer_state(tmp_path, qapp):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QMenu
    from calibfreehands.visualization.comparison import ComparisonViewer
    viewer = ComparisonViewer(_session(tmp_path, 3))
    viewer.resize(1500, 900)
    viewer.show()
    qapp.processEvents()
    splitter = viewer.views.widget(0)
    assert splitter.handleWidth() <= 10
    handle = splitter.handle(1)
    assert handle.cursor().shape() == Qt.CursorShape.SplitHCursor
    assert "Right-click" in handle.toolTip()
    assert not hasattr(viewer, "overlay_swap_button")
    image = handle.grab().toImage()
    scale = image.devicePixelRatio()
    center = handle.rect().center()

    def luminance(color):
        channels = [component / 12.92 if component <= 0.04045 else ((component + 0.055) / 1.055) ** 2.4
                    for component in (color.redF(), color.greenF(), color.blueF())]
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]

    def contrast(first, second):
        light = max(luminance(first), luminance(second))
        dark = min(luminance(first), luminance(second))
        return (light + 0.05) / (dark + 0.05)

    background = image.pixelColor(0, 0)
    line_x = round(center.x() * scale)
    foreground_pixels = [contrast(background, image.pixelColor(line_x, y)) >= 3
                         for y in range(image.height())]
    runs = []
    start = None
    for y, is_dot in enumerate(foreground_pixels):
        if is_dot and start is None:
            start = y
        elif not is_dot and start is not None:
            runs.append((start, y - 1))
            start = None
    if start is not None:
        runs.append((start, image.height() - 1))
    expected_centers = [round((center.y() + offset) * scale) for offset in (-5, 0, 5)]
    assert len(runs) == 3, (runs, expected_centers)
    actual_centers = [round((start + end) / 2) for start, end in runs]
    assert all(abs(actual - expected) <= 1 for actual, expected in zip(actual_centers, expected_centers))
    assert all(abs((end - start + 1) - round(3 * scale)) <= 1 for start, end in runs)
    splitter.setSizes([420, 760])
    qapp.processEvents()
    before_sizes = splitter.sizes()
    before_frame = viewer.slider.value()
    before_clock = viewer.playback_clock
    before_camera = dict(viewer.overlay_view.opts)
    before_split_order = viewer.split_panel_order.copy()

    QTest.mouseClick(handle, Qt.MouseButton.LeftButton, pos=handle.rect().center())
    qapp.processEvents()
    assert viewer.overlay_panel_order == ["source-media", "overlay-3d"]
    sizes_before_drag = splitter.sizes()
    center = handle.rect().center()
    QTest.mousePress(handle, Qt.MouseButton.LeftButton, pos=center)
    QTest.mouseMove(handle, center + QPoint(45, 0), 100)
    QTest.mouseRelease(handle, Qt.MouseButton.LeftButton, pos=center + QPoint(45, 0))
    qapp.processEvents()
    assert splitter.sizes() != sizes_before_drag
    splitter.setSizes(before_sizes)
    qapp.processEvents()

    QTest.mouseClick(handle, Qt.MouseButton.RightButton, pos=handle.rect().center())
    qapp.processEvents()
    menu = handle.findChild(QMenu)
    assert menu is not None and menu.isVisible()
    assert [action.text() for action in menu.actions()] == ["Swap"]
    menu.hide()
    qapp.processEvents()
    assert viewer.overlay_panel_order == ["source-media", "overlay-3d"]
    QTest.mouseClick(handle, Qt.MouseButton.RightButton, pos=handle.rect().center())
    qapp.processEvents()
    menu = handle.findChild(QMenu)
    menu.actions()[0].trigger()
    menu.close()
    qapp.processEvents()
    assert viewer.overlay_panel_order == ["overlay-3d", "source-media"]
    assert splitter.widget(0).content is viewer.overlay_view
    assert splitter.widget(1).content is viewer.source_overlay
    assert splitter.sizes() == before_sizes
    assert viewer.slider.value() == before_frame
    assert viewer.playback_clock is before_clock
    assert viewer.overlay_view.opts == before_camera
    assert viewer.split_panel_order == before_split_order

    for expected_order in (["source-media", "overlay-3d"], ["overlay-3d", "source-media"]):
        QTest.mouseClick(handle, Qt.MouseButton.RightButton, pos=handle.rect().center())
        qapp.processEvents()
        menu = handle.findChild(QMenu)
        menu.actions()[0].trigger()
        menu.close()
        qapp.processEvents()
        assert viewer.overlay_panel_order == expected_order
    viewer.close()


def test_overlay_splitter_drag_coalesces_resize_to_playback_cadence(tmp_path, qapp, monkeypatch):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    import calibfreehands.visualization.comparison_presentation as presentation
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 3))
    viewer.resize(1500, 900)
    viewer.show()
    qapp.processEvents()
    splitter = viewer.overlay_splitter
    handle = splitter.handle(1)
    center = handle.rect().center()
    start_global = handle.mapToGlobal(center)
    fake_time = [0.0]
    monkeypatch.setattr(presentation, "monotonic", lambda: fake_time[0])
    initial_camera = dict(viewer.overlay_view.cameraParams())
    initial_reference = viewer.reference_id
    initial_results = tuple(map(id, viewer.loaded_results))
    viewer.timer.start()

    QTest.mousePress(handle, Qt.MouseButton.LeftButton, pos=center)
    QTest.mouseMove(handle, handle.mapFromGlobal(start_global + QPoint(20, 0)))
    first_resize = splitter.sizes()
    viewer._set_frame(1)
    assert viewer.requested_frame == viewer.presented_frame == 1
    assert viewer.source_overlay.frame == viewer.overlay_view.frame == 1
    assert viewer.reference_id == initial_reference
    assert tuple(map(id, viewer.loaded_results)) == initial_results
    assert viewer.overlay_view.cameraParams() == initial_camera
    fake_time[0] = 0.01
    QTest.mouseMove(handle, handle.mapFromGlobal(start_global + QPoint(40, 0)))
    assert splitter.sizes() == first_resize
    fake_time[0] = 0.04
    QTest.mouseMove(handle, handle.mapFromGlobal(start_global + QPoint(60, 0)))
    last_coalesced_resize = splitter.sizes()
    assert splitter.sizes() != first_resize
    fake_time[0] = 0.08
    QTest.mouseRelease(handle, Qt.MouseButton.LeftButton,
                       pos=handle.mapFromGlobal(start_global + QPoint(80, 0)))
    assert splitter.sizes()[0] == pytest.approx(first_resize[0] + 60, abs=4)
    assert splitter.sizes()[0] > last_coalesced_resize[0]
    assert viewer.timer.isActive()
    assert not splitter.drag_updates_coalesced

    viewer.timer.stop()
    before_paused_move = splitter.sizes()
    paused_center = handle.rect().center()
    paused_start_global = handle.mapToGlobal(paused_center)
    QTest.mousePress(handle, Qt.MouseButton.LeftButton, pos=paused_center)
    QTest.mouseMove(handle, handle.mapFromGlobal(paused_start_global + QPoint(-20, 0)))
    first_paused_move = splitter.sizes()
    QTest.mouseMove(handle, handle.mapFromGlobal(paused_start_global + QPoint(-50, 0)))
    assert first_paused_move != before_paused_move
    assert splitter.sizes() != first_paused_move
    QTest.mouseRelease(handle, Qt.MouseButton.LeftButton,
                       pos=handle.mapFromGlobal(paused_start_global + QPoint(-50, 0)))
    viewer.close()


def test_overlay_splitter_grip_pixels_at_light_dark_and_high_dpi(tmp_path, qapp):
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QColor, QImage, QPainter, QPalette
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2))
    handle = viewer.overlay_splitter.handle(1)
    palette = handle.palette()

    def luminance(color):
        components = [value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
                      for value in (color.redF(), color.greenF(), color.blueF())]
        return 0.2126 * components[0] + 0.7152 * components[1] + 0.0722 * components[2]

    def contrast(first, second):
        light = max(luminance(first), luminance(second))
        dark = min(luminance(first), luminance(second))
        return (light + 0.05) / (dark + 0.05)

    themes = ((QColor("#f2f2f2"), QColor("#151515")),
              (QColor("#2f3238"), QColor("#eeeeee")))
    for background, foreground in themes:
        palette.setColor(QPalette.ColorRole.Window, background)
        palette.setColor(QPalette.ColorRole.WindowText, foreground)
        handle.setPalette(palette)
        for scale in (1.0, 1.25, 1.5, 2.0):
            image = QImage(round(8 * scale), round(24 * scale),
                           QImage.Format.Format_ARGB32_Premultiplied)
            image.setDevicePixelRatio(scale)
            painter = QPainter(image)
            handle._paint_grip(painter, QRectF(0, 0, 8, 24))
            painter.end()
            line_x = round(4 * scale)
            background_pixel = image.pixelColor(0, 0)
            active = [contrast(background_pixel, image.pixelColor(line_x, y)) >= 3
                      for y in range(image.height())]
            runs = []
            start = None
            for y, is_dot in enumerate(active):
                if is_dot and start is None:
                    start = y
                elif not is_dot and start is not None:
                    runs.append((start, y - 1))
                    start = None
            if start is not None:
                runs.append((start, image.height() - 1))
            centers = [round((start + end) / 2) for start, end in runs]
            expected = [round((12 + offset) * scale) for offset in (-5, 0, 5)]
            assert len(runs) == 3, (scale, runs, expected)
            assert all(abs(actual - target) <= 1 for actual, target in zip(centers, expected))
            assert all(abs((end - start + 1) - round(3 * scale)) <= 1
                       for start, end in runs)
            assert all(contrast(background_pixel, image.pixelColor(line_x, y)) < 3
                       for y in (round(9.5 * scale), round(14.5 * scale)))
    viewer.close()


def test_split_reflow_keeps_native_gl_contexts_and_paints_frames(tmp_path, qapp):
    import warnings
    import shiboken6
    from OpenGL import GL
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from calibfreehands.visualization.comparison import ComparisonViewer

    if QApplication.platformName() == "offscreen":
        pytest.skip("requires a native platform plugin with a real OpenGL context")

    viewer = ComparisonViewer(_session(tmp_path, 3, mesh=True))
    viewer.resize(1500, 900)
    with warnings.catch_warnings(record=True) as render_warnings:
        warnings.simplefilter("always")
        viewer.show()
        QTest.qWait(250)
        viewer.set_mode("split")
        QTest.qWait(250)

        views = [viewer.overlay_view, *viewer.split_views]
        contexts = [view.context() for view in views]
        assert all(context is not None and context.isValid() for context in contexts)
        context_ids = [shiboken6.getCppPointer(context)[0] for context in contexts]
        destroyed = [[] for _ in views]
        for events, context in zip(destroyed, contexts):
            context.aboutToBeDestroyed.connect(lambda events=events: events.append(True))

        def assert_drawn(view):
            image = view.grabFramebuffer()
            assert not image.isNull()
            pixels = np.frombuffer(image.bits(), dtype=np.uint8)
            pixels = pixels.reshape(image.height(), image.bytesPerLine())
            rgb = pixels[:, :image.width() * 4].reshape(image.height(), image.width(), 4)[:, :, :3]
            assert len(np.unique(rgb.reshape(-1, 3), axis=0)) > 8
            view.makeCurrent()
            errors = []
            while (error := GL.glGetError()) != GL.GL_NO_ERROR:
                errors.append(error)
            view.doneCurrent()
            assert errors == []

        for view in views:
            assert_drawn(view)

        viewer._set_configuration_visible(1, False)
        viewer._set_configuration_visible(1, True)
        viewer.reorder_split_panel("result:cfg-2", "result:cfg-0")
        viewer.resize(1680, 960)
        QTest.qWait(250)
        viewer.set_mode("overlay")
        viewer.overlay_splitter.request_swap()
        viewer.overlay_splitter.request_swap()
        viewer.set_mode("split")
        QTest.qWait(300)

        assert [shiboken6.getCppPointer(view.context())[0] for view in views] == context_ids
        assert not any(destroyed)
        for view in views:
            assert_drawn(view)
        rendering_errors = [str(item.message) for item in render_warnings
                            if "Error while drawing item" in str(item.message)
                            or "GLError" in str(item.message)]
        assert rendering_errors == []
    viewer.close()


def test_overlay_hud_mouse_drag_inserts_using_hud_row_geometry(tmp_path, qapp):
    from PySide6.QtCore import QEvent, QPoint, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication
    from calibfreehands.visualization.comparison import ComparisonViewer
    from calibfreehands.visualization.comparison_presentation import _ResultDragHandle

    viewer = ComparisonViewer(_session(tmp_path, 4))
    viewer.resize(1500, 900)
    viewer.show()
    qapp.processEvents()
    hud = viewer.overlay_panel
    source_id = "cfg-3"
    handle = next(item for item in hud.findChildren(_ResultDragHandle)
                  if item.configuration_id == source_id)
    previous_center = hud.group_widgets["cfg-0"].geometry().center().y()
    target_center = hud.group_widgets["cfg-1"].geometry().center().y()
    insertion_y = round((previous_center + target_center) / 2)
    end = hud.body.mapToGlobal(QPoint(30, insertion_y))
    start = handle.mapToGlobal(handle.rect().center())
    QTest = __import__("PySide6.QtTest", fromlist=["QTest"]).QTest
    QTest.mousePress(handle, Qt.MouseButton.LeftButton, pos=handle.rect().center())
    move = QMouseEvent(QEvent.Type.MouseMove, handle.mapFromGlobal(end), end,
                       Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(handle, move)
    assert hud.insertion_marker.isVisible()
    QTest.mouseRelease(handle, Qt.MouseButton.LeftButton, pos=handle.mapFromGlobal(end))
    assert viewer.result_order_ids == ["cfg-0", "cfg-3", "cfg-1", "cfg-2"]
    assert not hud.insertion_marker.isVisible()
    assert not viewer._panel_reorder_active
    viewer.close()


@pytest.mark.parametrize(("source_id", "insertion", "expected_order"), [
    ("cfg-3", 0, ["cfg-3", "cfg-0", "cfg-1", "cfg-2"]),
    ("cfg-3", 1, ["cfg-0", "cfg-3", "cfg-1", "cfg-2"]),
    ("cfg-0", 3, ["cfg-1", "cfg-2", "cfg-0", "cfg-3"]),
    ("cfg-0", 4, ["cfg-1", "cfg-2", "cfg-3", "cfg-0"]),
    ("cfg-1", 2, ["cfg-0", "cfg-1", "cfg-2", "cfg-3"]),
])
def test_overlay_hud_marker_tracks_variable_group_bounds_and_drop_slot(
        tmp_path, qapp, source_id, insertion, expected_order):
    from PySide6.QtCore import QPoint
    from calibfreehands.visualization.comparison import ComparisonViewer
    from calibfreehands.visualization.comparison_presentation import _ResultDragHandle

    viewer = ComparisonViewer(_session(tmp_path, 4))
    viewer.resize(1500, 900)
    viewer.show()
    viewer.confidence_numeric_checkbox.setChecked(True)
    viewer.confidence_bar_checkbox.setChecked(True)
    qapp.processEvents()
    panel = viewer.overlay_panel
    group_ids = viewer.result_order_ids
    assert len({panel.group_widgets[cid].height() for cid in group_ids}) > 1
    handle = next(item for item in panel.findChildren(_ResultDragHandle)
                  if item.configuration_id == source_id)
    centers = [panel.group_widgets[cid].geometry().center().y() for cid in group_ids]
    if insertion == 0:
        body_y = panel.group_widgets[group_ids[0]].geometry().top()
    elif insertion == len(group_ids):
        body_y = panel.body.height() - 1
    else:
        body_y = round((centers[insertion - 1] + centers[insertion]) / 2)
    start = handle.mapToGlobal(handle.rect().center())
    target = panel.body.mapToGlobal(QPoint(24, body_y))
    handle._press = start
    handle._move_drag(target)
    assert handle._drop_target[1] == insertion

    marker = panel.insertion_marker
    assert marker.isVisible()
    marker_in_card = marker.geometry().translated(panel.body.mapTo(panel.card, QPoint(0, 0)))
    assert panel.card.rect().contains(marker_in_card)
    assert marker_in_card.left() >= 8
    assert marker_in_card.right() <= panel.card.width() - 8
    marker_rect = marker.geometry()
    if 0 < insertion < len(group_ids):
        previous = panel.group_widgets[group_ids[insertion - 1]].geometry()
        following = panel.group_widgets[group_ids[insertion]].geometry()
        assert previous.bottom() < marker_rect.center().y() < following.top()
    elif insertion == 0:
        assert marker_rect.bottom() <= panel.group_widgets[group_ids[0]].geometry().top() + 2
    else:
        assert marker_rect.top() >= panel.group_widgets[group_ids[-1]].geometry().bottom() - 2

    handle._finish_drag(target)
    assert viewer.result_order_ids == expected_order
    assert not marker.isVisible()
    viewer.close()


@pytest.mark.parametrize(("anchor", "scale"), [
    ("top-right", "small"), ("bottom-left", "large"),
])
def test_overlay_hud_marker_remains_inside_content_after_resize_and_reanchor(
        tmp_path, qapp, anchor, scale):
    from PySide6.QtCore import QPoint
    from calibfreehands.visualization.comparison import ComparisonViewer
    from calibfreehands.visualization.comparison_presentation import _ResultDragHandle

    viewer = ComparisonViewer(_session(tmp_path, 4))
    viewer.resize(1500, 900)
    viewer.show()
    qapp.processEvents()
    panel = viewer.overlay_panel
    panel.set_presentation(anchor=anchor, scale=scale)
    qapp.processEvents()
    source_id = "cfg-3"
    handle = next(item for item in panel.findChildren(_ResultDragHandle)
                  if item.configuration_id == source_id)
    destination = panel.body.mapToGlobal(QPoint(25, panel.body.height() - 1))
    handle._press = handle.mapToGlobal(handle.rect().center())
    handle._move_drag(destination)
    viewer.resize(1600, 960)
    qapp.processEvents()
    destination = panel.body.mapToGlobal(QPoint(25, panel.body.height() - 1))
    handle._move_drag(destination)
    marker = panel.insertion_marker
    marker_rect = marker.geometry().translated(panel.body.mapTo(panel.card, QPoint(0, 0)))
    assert panel.card.rect().contains(marker_rect)
    assert marker_rect.left() >= 8
    assert marker_rect.right() <= panel.card.width() - 8
    handle._panel_drag_cancelled()
    assert not marker.isVisible()
    viewer.close()


def test_overlay_hud_hidden_groups_and_outside_drop_cancel_marker(tmp_path, qapp):
    from PySide6.QtCore import QPoint
    from calibfreehands.visualization.comparison import ComparisonViewer
    from calibfreehands.visualization.comparison_presentation import _ResultDragHandle

    viewer = ComparisonViewer(_session(tmp_path, 4))
    viewer.resize(1500, 900)
    viewer.show()
    qapp.processEvents()
    viewer.visibility_checks[1].setChecked(False)
    qapp.processEvents()
    panel = viewer.overlay_panel
    assert "cfg-1" not in panel.group_widgets
    handle = next(item for item in panel.findChildren(_ResultDragHandle)
                  if item.configuration_id == "cfg-3")
    visible_ids = [cid for cid in viewer.result_order_ids if cid in panel.group_widgets]
    centers = [panel.group_widgets[cid].geometry().center().y() for cid in visible_ids]
    target = panel.body.mapToGlobal(QPoint(20, round((centers[0] + centers[1]) / 2)))
    original_order = viewer.result_order_ids.copy()
    handle._press = handle.mapToGlobal(handle.rect().center())
    handle._move_drag(target)
    assert handle._drop_target == (visible_ids, 1)
    assert panel.insertion_marker.isVisible()

    outside = viewer.overlay_view.mapToGlobal(QPoint(viewer.overlay_view.width() - 3,
                                                     viewer.overlay_view.height() - 3))
    handle._move_drag(outside)
    assert handle._drop_target is None
    assert not panel.insertion_marker.isVisible()
    handle._finish_drag(outside)
    assert viewer.result_order_ids == original_order
    assert not viewer._panel_reorder_active
    viewer.close()


def test_overlay_hud_marker_rebuilds_after_reference_change(tmp_path, qapp):
    from PySide6.QtCore import QPoint
    from calibfreehands.visualization.comparison import ComparisonViewer
    from calibfreehands.visualization.comparison_presentation import _ResultDragHandle

    viewer = ComparisonViewer(_session(tmp_path, 4, gt_index=0))
    viewer.resize(1500, 900)
    viewer.show()
    qapp.processEvents()
    panel = viewer.overlay_panel
    handle = next(item for item in panel.findChildren(_ResultDragHandle)
                  if item.configuration_id == "cfg-3")
    first_ids = viewer.result_order_ids.copy()
    initial_y = round((panel.group_widgets[first_ids[0]].geometry().center().y()
                       + panel.group_widgets[first_ids[1]].geometry().center().y()) / 2)
    handle._press = handle.mapToGlobal(handle.rect().center())
    handle._move_drag(panel.body.mapToGlobal(QPoint(20, initial_y)))
    assert panel.insertion_marker.isVisible()

    viewer.reference_selector.setCurrentIndex(viewer.reference_selector.findData("cfg-2"))
    _wait_for_metric_job(viewer, qapp)
    assert not panel.insertion_marker.isVisible()
    assert set(panel.group_widgets) == set(viewer.result_order_ids)

    reordered_ids = viewer.result_order_ids
    body_y = round((panel.group_widgets[reordered_ids[0]].geometry().center().y()
                    + panel.group_widgets[reordered_ids[1]].geometry().center().y()) / 2)
    handle._move_drag(panel.body.mapToGlobal(QPoint(20, body_y)))
    assert panel.insertion_marker.isVisible()
    assert handle._drop_target[0] == reordered_ids
    handle._panel_drag_cancelled()
    assert not panel.insertion_marker.isVisible()
    viewer.close()


def test_overlay_hud_click_does_not_reorder_results(tmp_path, qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from calibfreehands.visualization.comparison import ComparisonViewer
    from calibfreehands.visualization.comparison_presentation import _ResultDragHandle
    viewer = ComparisonViewer(_session(tmp_path, 3))
    handle = next(item for item in viewer.overlay_panel.findChildren(_ResultDragHandle)
                  if item.configuration_id == "cfg-2")
    before = viewer.result_order_ids
    QTest.mouseClick(handle, Qt.MouseButton.LeftButton)
    assert viewer.result_order_ids == before
    assert not viewer._panel_reorder_active
    viewer.close()


def test_escape_cancels_active_panel_reorder_without_sticking_controls(tmp_path, qapp):
    from PySide6.QtCore import QEvent, QPoint, Qt
    from PySide6.QtGui import QKeyEvent, QMouseEvent
    from PySide6.QtWidgets import QApplication
    from PySide6.QtTest import QTest
    from calibfreehands.visualization.comparison import ComparisonViewer
    from calibfreehands.visualization.comparison_presentation import _ResultDragHandle
    viewer = ComparisonViewer(_session(tmp_path, 3))
    viewer.resize(1500, 900)
    viewer.show()
    qapp.processEvents()
    handle = next(item for item in viewer.overlay_panel.findChildren(_ResultDragHandle)
                  if item.configuration_id == "cfg-2")
    end = viewer.overlay_panel.body.mapToGlobal(QPoint(20, 5))
    order = viewer.result_order_ids
    QTest.mousePress(handle, Qt.MouseButton.LeftButton, pos=handle.rect().center())
    move = QMouseEvent(QEvent.Type.MouseMove, handle.mapFromGlobal(end), end,
                       Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(handle, move)
    assert viewer._panel_reorder_active
    assert viewer.overlay_panel.insertion_marker.isVisible()
    QApplication.sendEvent(viewer, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape,
                                               Qt.KeyboardModifier.NoModifier))
    assert not viewer._panel_reorder_active
    assert not viewer.overlay_panel.insertion_marker.isVisible()
    assert viewer.result_order_ids == order
    QTest.mouseRelease(handle, Qt.MouseButton.LeftButton, pos=handle.mapFromGlobal(end))
    viewer.close()


def test_split_source_can_move_through_first_middle_last_without_duplicate_widgets(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer
    viewer = ComparisonViewer(_session(tmp_path, 3))
    source = viewer.split_surfaces["source-media"]
    viewer.reorder_split_panel("source-media", "result:cfg-1", after=True)
    assert viewer.split_panel_order == ["result:cfg-0", "result:cfg-1", "source-media", "result:cfg-2"]
    viewer.reorder_split_panel("source-media", "result:cfg-0")
    assert viewer.split_panel_order[0] == "source-media"
    viewer.reorder_split_panel("source-media", "result:cfg-2", after=True)
    assert viewer.split_panel_order[-1] == "source-media"
    widgets = [viewer.split_grid.itemAt(i).widget() for i in range(viewer.split_grid.count())]
    assert len(widgets) == 4 and len({id(widget) for widget in widgets}) == 4
    assert source in widgets
    assert viewer.split_surfaces["result:cfg-0"].content is viewer.split_views[0]
    viewer.close()


def test_shift_drag_over_split_panel_body_reorders_and_shows_live_target(tmp_path, qapp):
    from PySide6.QtCore import QEvent, QPoint, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication
    from PySide6.QtTest import QTest
    from calibfreehands.visualization.comparison import ComparisonViewer
    viewer = ComparisonViewer(_session(tmp_path, 3))
    viewer.resize(1500, 900)
    viewer.show()
    viewer.set_mode("split")
    qapp.processEvents()
    source_widget = viewer.split_source_overlay
    source_surface = viewer.split_surfaces["source-media"]
    target = viewer.split_surfaces["result:cfg-1"]
    end = target.mapToGlobal(target.rect().center())
    QTest.mousePress(source_widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier,
                     pos=source_widget.rect().center())
    move = QMouseEvent(QEvent.Type.MouseMove, source_widget.mapFromGlobal(end), end,
                       Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(target.content, move)
    assert viewer.split_reorder_controller.target_panel_id == "result:cfg-1"
    assert viewer._split_source_feedback.isVisible()
    assert viewer._split_target_feedback.isVisible()
    QTest.mouseRelease(source_widget, Qt.MouseButton.LeftButton, pos=source_widget.mapFromGlobal(end))
    assert viewer.split_panel_order.index("source-media") > viewer.split_panel_order.index("result:cfg-1")
    assert not viewer._panel_reorder_active
    assert not viewer._split_source_feedback.isVisible()
    assert not viewer._split_target_feedback.isVisible()
    viewer.close()


def test_shift_drag_moves_gl_result_diagonally_and_normal_drag_is_not_reordered(tmp_path, qapp):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication
    from PySide6.QtTest import QTest
    from calibfreehands.visualization.comparison import ComparisonViewer
    viewer = ComparisonViewer(_session(tmp_path, 4))
    viewer.resize(1500, 900)
    viewer.show()
    viewer.set_mode("split")
    qapp.processEvents()
    source = viewer.split_views[3]
    target = viewer.split_surfaces["result:cfg-0"]
    start = source.mapToGlobal(source.rect().center())
    end = target.mapToGlobal(target.rect().center())
    order = viewer.split_panel_order.copy()
    QTest.mousePress(source, Qt.MouseButton.LeftButton, pos=source.rect().center())
    move = QMouseEvent(QEvent.Type.MouseMove, source.mapFromGlobal(end), end,
                       Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(target.content, move)
    QTest.mouseRelease(source, Qt.MouseButton.LeftButton, pos=source.mapFromGlobal(end))
    assert viewer.split_panel_order == order
    assert not viewer._panel_reorder_active

    QTest.mousePress(source, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier,
                     pos=source.rect().center())
    move = QMouseEvent(QEvent.Type.MouseMove, source.mapFromGlobal(end), end,
                       Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier)
    QApplication.sendEvent(target, move)
    assert viewer.split_reorder_controller.target_panel_id == "result:cfg-0"
    QTest.mouseRelease(source, Qt.MouseButton.LeftButton, pos=source.rect().center())
    assert viewer.split_panel_order.index("result:cfg-3") < viewer.split_panel_order.index("result:cfg-0")
    assert not viewer._panel_reorder_active
    viewer.close()


def test_split_shift_drag_escape_and_outside_release_cancel_cleanly(tmp_path, qapp):
    from PySide6.QtCore import QEvent, QPoint, Qt
    from PySide6.QtGui import QKeyEvent, QMouseEvent
    from PySide6.QtWidgets import QApplication
    from PySide6.QtTest import QTest
    from calibfreehands.visualization.comparison import ComparisonViewer
    viewer = ComparisonViewer(_session(tmp_path, 3))
    viewer.resize(1500, 900)
    viewer.show()
    viewer.set_mode("split")
    qapp.processEvents()
    source = viewer.split_views[2]
    target = viewer.split_surfaces["source-media"]
    end = target.mapToGlobal(target.rect().center())
    order = viewer.split_panel_order.copy()
    QTest.mousePress(source, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier,
                     pos=source.rect().center())
    move = QMouseEvent(QEvent.Type.MouseMove, source.mapFromGlobal(end), end,
                       Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier)
    QApplication.sendEvent(source, move)
    assert viewer._split_target_feedback.isVisible()
    QApplication.sendEvent(viewer, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape,
                                               Qt.KeyboardModifier.NoModifier))
    assert viewer.split_panel_order == order
    assert not viewer._panel_reorder_active
    assert not viewer._split_target_feedback.isVisible()
    QTest.mouseRelease(source, Qt.MouseButton.LeftButton, pos=source.rect().center())

    outside = viewer.mapToGlobal(viewer.rect().bottomRight() + QPoint(200, 200))
    QTest.mousePress(source, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier,
                     pos=source.rect().center())
    move = QMouseEvent(QEvent.Type.MouseMove, source.mapFromGlobal(outside), outside,
                       Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier)
    QApplication.sendEvent(source, move)
    QTest.mouseRelease(source, Qt.MouseButton.LeftButton, pos=source.mapFromGlobal(outside))
    assert viewer.split_panel_order == order
    assert not viewer._panel_reorder_active
    viewer.close()


@pytest.mark.parametrize("results_to_hide,expected_count,expected_shape", [
    ([], 5, (3, 2)),
    (["cfg-3"], 4, (2, 2)),
    (["cfg-3", "cfg-2"], 3, (2, 2)),
    (["cfg-3", "cfg-2", "cfg-1"], 2, (2, 1)),
    (["cfg-3", "cfg-2", "cfg-1", "cfg-0"], 1, (1, 1)),
])
def test_split_grid_reflows_for_visible_panel_count(tmp_path, qapp, results_to_hide,
                                                    expected_count, expected_shape):
    from PySide6.QtTest import QTest
    from calibfreehands.visualization.comparison import ComparisonViewer
    viewer = ComparisonViewer(_session(tmp_path, 4))
    viewer.resize(1280, 720)
    viewer.show()
    viewer.set_mode("split")
    QTest.qWait(200)
    for config_id in results_to_hide:
        index = viewer._configuration_index(config_id)
        viewer._set_configuration_visible(index, False)
    qapp.processEvents()
    assert viewer.split_grid.count() == expected_count
    expected_columns, expected_rows = expected_shape
    assert viewer.split_columns == expected_columns, (
        f"shape={viewer.split_columns}x{viewer.split_rows}, "
        f"available={viewer._split_available_size().width()}x{viewer._split_available_size().height()}"
    )
    assert viewer.split_rows == expected_rows
    visible_ids = ["source-media", *(f"result:{config['id']}" for config in viewer.configurations
                                      if viewer.overlay_view.configuration_visible[
                                          viewer._configuration_index(config["id"])])]
    assert [viewer.split_grid.itemAt(index).widget().panel_id for index in range(viewer.split_grid.count())] == visible_ids
    positions = [viewer.split_grid.getItemPosition(index)[:2]
                 for index in range(viewer.split_grid.count())]
    assert positions == [(index // expected_columns, index % expected_columns)
                         for index in range(expected_count)]
    assert len({id(viewer.split_grid.itemAt(index).widget()) for index in range(viewer.split_grid.count())}) == expected_count
    margins = viewer.split_grid.contentsMargins()
    spacing = viewer.split_grid.spacing()
    assert viewer.split_widget.minimumWidth() == (
        expected_columns * 340 + max(0, expected_columns - 1) * spacing
        + margins.left() + margins.right()
    )
    assert viewer.split_widget.minimumHeight() == (
        expected_rows * 260 + max(0, expected_rows - 1) * spacing
        + margins.top() + margins.bottom()
    )
    if expected_count == 5:
        assert viewer.split_scroll.horizontalScrollBar().maximum() == 0
        assert viewer.split_scroll.verticalScrollBar().maximum() == 0
    viewer.close()


def test_split_grid_restore_and_resize_preserve_canonical_order(tmp_path, qapp):
    from PySide6.QtTest import QTest
    from calibfreehands.visualization.comparison import ComparisonViewer
    viewer = ComparisonViewer(_session(tmp_path, 8))
    viewer.resize(1280, 720)
    viewer.show()
    viewer.set_mode("split")
    qapp.processEvents()
    original_order = viewer.split_panel_order.copy()
    viewer._set_configuration_visible(7, False)
    assert viewer.split_grid.count() == 8
    viewer._set_configuration_visible(7, True)
    assert viewer.split_grid.count() == 9
    viewer.reorder_split_panel("result:cfg-7", "result:cfg-2")
    custom_order = viewer.split_panel_order.copy()
    viewer.resize(1920, 1080)
    QTest.qWait(200)
    assert viewer.split_panel_order == custom_order
    assert [viewer.split_grid.itemAt(index).widget().panel_id for index in range(viewer.split_grid.count())] == custom_order
    assert viewer.split_columns >= 2 and viewer.split_rows == int(np.ceil(9 / viewer.split_columns))
    assert len({id(viewer.split_grid.itemAt(index).widget()) for index in range(9)}) == 9
    assert original_order != custom_order
    viewer.close()


def test_split_grid_adapts_to_viewport_and_keeps_large_result_widgets(tmp_path, qapp):
    from PySide6.QtTest import QTest
    from calibfreehands.visualization.comparison import ComparisonViewer
    viewer = ComparisonViewer(_session(tmp_path, 12))
    viewer.resize(1280, 720)
    viewer.show()
    viewer.set_mode("split")
    QTest.qWait(200)
    widget_ids = {panel_id: id(surface) for panel_id, surface in viewer.split_surfaces.items()}

    def assert_row_major_and_usable():
        expected_ids = viewer._visible_split_panel_ids()
        assert viewer.split_grid.count() == len(expected_ids)
        actual = []
        for index in range(viewer.split_grid.count()):
            row, column, row_span, column_span = viewer.split_grid.getItemPosition(index)
            assert row_span == column_span == 1
            actual.append((row, column, viewer.split_grid.itemAt(index).widget().panel_id))
        assert actual == [(index // viewer.split_columns, index % viewer.split_columns, panel_id)
                          for index, panel_id in enumerate(expected_ids)]
        assert all(id(viewer.split_surfaces[panel_id]) == widget_ids[panel_id]
                   for panel_id in viewer.split_surfaces)
        assert all(viewer.split_surfaces[panel_id].minimumWidth() <= 340
                   and viewer.split_surfaces[panel_id].minimumHeight() <= 260
                   for panel_id in expected_ids)

    landscape_columns = viewer.split_columns
    assert_row_major_and_usable()
    viewer.resize(720, 1280)
    QTest.qWait(200)
    assert viewer.split_columns < landscape_columns
    assert_row_major_and_usable()
    viewer.resize(1920, 720)
    QTest.qWait(200)
    assert viewer.split_columns >= landscape_columns
    assert_row_major_and_usable()

    viewer._set_configuration_visible(10, False)
    hidden_id = "result:cfg-10"
    assert hidden_id not in viewer._laid_out_panel_ids
    viewer.reorder_split_panel(hidden_id, "result:cfg-2")
    remembered_order = viewer.split_panel_order.copy()
    viewer._set_configuration_visible(10, True)
    assert viewer.split_panel_order == remembered_order
    assert_row_major_and_usable()
    viewer.close()


def test_reorder_preserves_frame_playback_clock_camera_and_result_arrays(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer
    viewer = ComparisonViewer(_session(tmp_path, 3))
    viewer.slider.setValue(1)
    viewer.playback_clock.set_speed(1.25)
    viewer.playback_clock.play(1)
    viewer.timer.start(1000)
    provider_ids = tuple(id(bundle.media_frames) for bundle in viewer.loaded_results)
    arrays = [bundle.result.keypoints_3d.copy() for bundle in viewer.loaded_results]
    camera = dict(viewer.overlay_view.opts)
    clock = viewer.playback_clock
    viewer.reorder_result("cfg-2", "cfg-0")
    viewer.reorder_overlay_panel("overlay-3d", "source-media")
    assert viewer.slider.value() == 1
    assert viewer.playback_clock is clock and clock.speed == 1.25 and clock.playing
    assert viewer.timer.isActive()
    assert tuple(id(bundle.media_frames) for bundle in viewer.loaded_results) == provider_ids
    assert viewer.overlay_view.opts == camera
    for bundle, before in zip(viewer.loaded_results, arrays):
        np.testing.assert_array_equal(bundle.result.keypoints_3d, before)
    viewer.close()


def test_presentation_controls_keep_detection_and_mesh_opacity_distinct(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer
    viewer = ComparisonViewer(_session(tmp_path, 2, mesh=True))
    assert viewer.confidence_menu_button.text() == "Confidence Display"
    assert viewer.confidence_opacity_checkbox.text() == "Use as opacity"
    assert viewer.confidence_numeric_checkbox.text() == "Numeric value"
    assert viewer.mesh_opacity_label.text() == "Mesh opacity"
    assert not viewer.mesh_opacity_slider.isEnabled()
    viewer.mesh_checkbox.setChecked(True)
    assert viewer.mesh_opacity_slider.isEnabled()
    viewer.mesh_opacity_slider.setValue(50)
    mesh_alpha = viewer.split_views[0].meshes[0].opts["color"][3]
    viewer.confidence_opacity_checkbox.setChecked(True)
    assert viewer.split_views[0].meshes[0].opts["color"][3] < mesh_alpha
    viewer.confidence_opacity_checkbox.setChecked(False)
    assert viewer.split_views[0].meshes[0].opts["color"][3] == pytest.approx(0.5)
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
    assert window.rows_layout.itemAtPosition(0, 2).widget().text() == "Focal (px)"
    assert not any(label.text() == "Focal (px)" for label in row.reference_focal_container.findChildren(QLabel))
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
    from PySide6.QtWidgets import QAbstractButton
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2))
    assert hasattr(viewer, "pin_controls_button")
    assert isinstance(viewer.pin_controls_button, QAbstractButton)
    assert viewer.pin_controls_button.text() == ""
    assert viewer.pin_controls_button.accessibleName() == "Pin controls"
    viewer.pin_controls_button.setChecked(True)
    assert viewer.pin_controls_button.toolTip() == "Controls pinned"
    assert viewer.pin_controls_button.accessibleDescription() == "Pinned: controls remain visible"
    viewer.timer.start()
    viewer._auto_hide_controls()
    assert not viewer.chrome.isHidden()
    viewer.pin_controls_button.setChecked(False)
    assert viewer.pin_controls_button.toolTip() == "Keep controls visible"
    from PySide6.QtTest import QTest
    from PySide6.QtCore import Qt
    viewer.pin_controls_button.setFocus()
    QTest.keyClick(viewer.pin_controls_button, Qt.Key.Key_Space)
    assert viewer.pin_controls_button.isChecked()
    assert {"off", "hovered", "key_joints", "worst5", "all"} <= set(viewer.annotation_actions)
    viewer.set_annotation_mode("worst5")
    assert viewer.annotation_mode == "worst5"
    viewer.close()


def test_detection_bar_off_reflows_hud_before_any_pointer_event(tmp_path, qapp):
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 3))
    viewer.resize(1280, 800)
    viewer.show()
    viewer.confidence_numeric_checkbox.setChecked(True)
    viewer.confidence_bar_checkbox.setChecked(True)
    qapp.processEvents()
    panel = viewer.overlay_panel
    bar_height = panel.card.height()
    assert panel.confidence_bars

    viewer.confidence_bar_checkbox.setChecked(False)
    qapp.processEvents()
    off_height = panel.card.height()
    assert not panel.confidence_bars
    assert off_height < bar_height, "HUD retained its Bar-on geometry until a later pointer event"

    QApplication.sendEvent(panel.card, QEvent(QEvent.Type.Leave))
    qapp.processEvents()
    assert panel.card.height() == off_height
    for _ in range(10):
        viewer.confidence_bar_checkbox.setChecked(True)
        qapp.processEvents()
        on_height = panel.card.height()
        assert on_height == bar_height
        viewer.confidence_bar_checkbox.setChecked(False)
        qapp.processEvents()
        assert panel.card.height() == off_height
    viewer.close()


def test_mode_switch_has_compact_neutral_track_and_single_view_label(tmp_path, qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QPalette
    from PySide6.QtWidgets import QAbstractButton, QCheckBox, QLabel
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2))
    assert isinstance(viewer.mode_switch, QAbstractButton)
    assert 38 <= viewer.mode_switch.width() <= 44
    assert 20 <= viewer.mode_switch.height() <= 24
    assert viewer.mode_switch.isCheckable()
    assert viewer.mode_switch.accessibleName() == "View mode"
    assert viewer.mode_switch.accessibleDescription() == "Toggle between Overlay and Split views"
    assert viewer.mode_switch.toolTip() == "View: Overlay"
    assert viewer.mode_label.text() == "View"
    assert not any(label.text() == "Layout" for label in viewer.findChildren(QLabel))
    assert "#f2f3f4" in viewer.mode_label.styleSheet().lower()
    assert not hasattr(viewer, "overlay_mode_label")
    assert not hasattr(viewer, "split_mode_label")
    assert viewer.mode_switch.focusPolicy().name == "StrongFocus"
    assert viewer.mode_selector.layout().indexOf(viewer.mode_label) < viewer.mode_selector.layout().indexOf(viewer.mode_switch)
    assert viewer.mode_selector.layout().indexOf(viewer.pin_controls_button) == -1
    assert viewer.space_mode_switch.isCheckable() and viewer.labels_switch.isCheckable()
    assert not isinstance(viewer.labels_switch, QCheckBox)
    assert not hasattr(viewer, "space_mode_value")
    assert viewer.space_mode_switch.toolTip() == "Space: Physical"
    assert viewer.labels_switch.toolTip() == "Scale labels: On"
    assert viewer.space_mode_switch.accessibleName() == "3D space mode"
    assert viewer.labels_switch.accessibleName() == "Scale labels"
    controls = [viewer.mode_selector, viewer.space_mode_selector, viewer.labels_selector,
                viewer.pin_controls_button]
    assert [viewer.primary_controls.layout().indexOf(control) for control in controls] == [0, 1, 2, 3]
    toolbar_widgets = [viewer.toolbar.itemAt(index).widget() for index in range(viewer.toolbar.count())]
    toolbar_widgets = [widget for widget in toolbar_widgets if widget is not None]
    assert toolbar_widgets[:4] == [
        viewer.primary_controls, viewer.results_menu_button, viewer.reference_label, viewer.reference_selector
    ]
    assert 24 <= viewer.pin_controls_button.width() <= 26
    assert viewer.pin_controls_button.focusPolicy().name == "StrongFocus"
    for background, foreground, muted in (
            (QColor("#202020"), QColor("#f2f2f2"), QColor("#777777")),
            (QColor("#f8f8f8"), QColor("#202020"), QColor("#aaaaaa"))):
        palette = QPalette(viewer.palette())
        palette.setColor(QPalette.ColorRole.Window, background)
        palette.setColor(QPalette.ColorRole.WindowText, foreground)
        palette.setColor(QPalette.ColorRole.Mid, muted)
        viewer.setPalette(palette)
        assert viewer.mode_switch.palette().color(QPalette.ColorRole.Window) == background
        assert viewer.primary_controls.palette().color(QPalette.ColorRole.Window) == background
        assert all("#f2f3f4" in label.styleSheet().lower()
                   for label in (viewer.mode_label, viewer.space_label, viewer.labels_label))
        assert not viewer.primary_controls.autoFillBackground()
        assert viewer.primary_controls.sizeHint().width() < 400
    viewer.set_space_mode("hand")
    assert viewer.space_mode_switch.toolTip() == "Space: Hand"
    viewer.labels_switch.setChecked(False)
    assert viewer.labels_switch.toolTip() == "Scale labels: Off"
    viewer.labels_switch.setChecked(True)
    assert viewer.labels_switch.toolTip() == "Scale labels: On"
    for _ in range(4):
        viewer.mode_switch.click()
        assert viewer.mode == "split"
        assert viewer.views.currentIndex() == 1
        assert viewer.mode_switch.toolTip() == "View: Split"
        viewer.mode_switch.click()
        assert viewer.mode == "overlay"
        assert viewer.views.currentIndex() == 0
        assert viewer.mode_switch.toolTip() == "View: Overlay"
    viewer.show()
    qapp.processEvents()
    track_levels = []
    for background, expected_thumb in ((QColor("#202020"), QColor("#f2f3f4")),
                                       (QColor("#f8f8f8"), QColor("#f2f3f4"))):
        palette = QPalette(viewer.palette())
        palette.setColor(QPalette.ColorRole.Window, background)
        viewer.setPalette(palette)
        qapp.processEvents()
        image = viewer.mode_switch.grab().toImage()
        center = image.pixelColor(11, 11)
        assert abs(center.red() - expected_thumb.red()) <= 4
        track = image.pixelColor(21, 11)
        assert abs(track.red() - track.green()) < 12
        track_levels.append(track.red())
    assert track_levels[0] < track_levels[1]
    from PySide6.QtTest import QTest
    viewer.mode_switch.click()
    QTest.qWait(150)
    qapp.processEvents()
    image = viewer.mode_switch.grab().toImage()
    center = image.pixelColor(31, 11)
    assert abs(center.red() - 242) <= 4
    assert viewer.mode == "split"
    assert viewer.mode_switch.toolTip() == "View: Split"
    viewer.mode_switch.setFocus()
    QTest.keyClick(viewer.mode_switch, Qt.Key.Key_Space)
    assert viewer.mode == "overlay"
    viewer.close()


def test_pin_button_icon_is_palette_aware_and_view_precedes_results(tmp_path, qapp):
    from PySide6.QtGui import QColor, QPalette
    from PySide6.QtTest import QTest
    from PySide6.QtCore import Qt
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2))
    viewer.resize(1280, 720)
    viewer.show()
    qapp.processEvents()
    assert viewer.primary_controls.geometry().left() < viewer.results_menu_button.geometry().left()
    group_gap = viewer.results_menu_button.geometry().left() - viewer.primary_controls.geometry().right() - 1
    assert 10 <= group_gap <= 14

    for window, off_surface, off_ink in (
        (QColor("#202124"), QColor("#34383d"), QColor("#aab0b7")),
        (QColor("#f8f8f8"), QColor("#f0f1f2"), QColor("#666c74")),
    ):
        palette = QPalette(viewer.palette())
        palette.setColor(QPalette.ColorRole.Window, window)
        viewer.pin_controls_button.setPalette(palette)
        viewer.pin_controls_button.setChecked(False)
        qapp.processEvents()
        off = viewer.pin_controls_button.grab().toImage()
        viewer.pin_controls_button.setChecked(True)
        qapp.processEvents()
        on = viewer.pin_controls_button.grab().toImage()
        assert off != on
        assert off.pixelColor(4, 14).rgb() == off_surface.rgb()
        active_surface = on.pixelColor(4, 13)
        assert active_surface.rgb() == QColor("#33FF33").rgb()
        assert off.pixelColor(1, 14).rgb() != off.pixelColor(4, 14).rgb()
        assert off.pixelColor(14, 8).rgb() == off_ink.rgb()
        pin_ink = on.pixelColor(13, 8)
        assert pin_ink.rgb() == QColor("#111111").rgb()
        linear_channels = [
            channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
            for channel in (active_surface.redF(), active_surface.greenF(), active_surface.blueF())
        ]
        luminance = 0.2126 * linear_channels[0] + 0.7152 * linear_channels[1] + 0.0722 * linear_channels[2]
        dark_channels = [
            channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
            for channel in (pin_ink.redF(), pin_ink.greenF(), pin_ink.blueF())
        ]
        dark_ink_luminance = 0.2126 * dark_channels[0] + 0.7152 * dark_channels[1] + 0.0722 * dark_channels[2]
        dark_contrast = (luminance + 0.05) / (dark_ink_luminance + 0.05)
        assert dark_contrast >= 4.5
    viewer.pin_controls_button.setFocus()
    QTest.keyClick(viewer.pin_controls_button, Qt.Key.Key_Space)
    assert not viewer.pin_controls_button.isChecked()
    QTest.mouseClick(viewer.pin_controls_button, Qt.MouseButton.LeftButton)
    assert viewer.pin_controls_button.isChecked()
    QTest.mouseClick(viewer.pin_controls_button, Qt.MouseButton.LeftButton)
    assert not viewer.pin_controls_button.isChecked()
    viewer.close()


def test_global_space_still_controls_playback_without_control_focus(tmp_path, qapp):
    from PySide6.QtTest import QTest
    from PySide6.QtCore import Qt
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2))
    viewer.show()
    viewer.setFocus()
    qapp.processEvents()
    QTest.keyClick(viewer, Qt.Key.Key_Space)
    assert viewer.timer.isActive()
    QTest.keyClick(viewer, Qt.Key.Key_Space)
    assert not viewer.timer.isActive()
    viewer.close()


def test_speed_controls_form_a_compact_group_before_footer_stretch(tmp_path, qapp):
    from PySide6.QtWidgets import QSizePolicy
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2))
    viewer.resize(1280, 720)
    viewer.show()
    qapp.processEvents()
    group = viewer.speed_group
    assert group.sizePolicy().horizontalPolicy() in (QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)
    assert group.width() < 190
    gap_after_playback = group.geometry().left() - viewer.playback_group.geometry().right() - 1
    assert 12 <= gap_after_playback <= 18
    widgets = (viewer.speed_label, viewer.speed_down_button,
               viewer.speed_value_label, viewer.speed_up_button)
    for left, right in zip(widgets, widgets[1:]):
        gap = right.geometry().left() - left.geometry().right() - 1
        assert 0 <= gap <= 6
    assert viewer.playback_controls_layout.indexOf(group) < viewer.playback_controls_layout.count() - 1
    widest_value_width = viewer.speed_value_label.fontMetrics().horizontalAdvance("1.25" + chr(0xD7)) + 4
    assert viewer.speed_value_label.minimumWidth() == widest_value_width
    assert viewer.speed_value_label.maximumWidth() == widest_value_width
    viewer._step_playback_speed(1)
    assert viewer.speed_value_label.text() == "1.25" + chr(0xD7)
    viewer._step_playback_speed(-1)
    assert viewer.speed_value_label.text() == "1.0" + chr(0xD7)
    viewer.close()


def test_viewer_playback_controls_are_accessible_steppers_and_exclusive_modes(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2))
    assert viewer.speed_value_label.text() == "1.0×"
    assert viewer.speed_down_button.isEnabled()
    assert viewer.speed_up_button.isEnabled()
    assert viewer.mode_switch.isCheckable()
    assert not viewer.mode_switch.isChecked()
    assert viewer.mode_switch.accessibleName() == "View mode"
    viewer.mode_switch.click()
    assert viewer.mode == "split"
    assert viewer.mode_switch.isChecked()
    assert viewer.views.currentIndex() == 1
    assert viewer.mode_switch.accessibleDescription() == "Toggle between Overlay and Split views"
    assert viewer.mode_switch.toolTip() == "View: Split"
    assert viewer.annotation_button.text() == "Annotations"
    assert viewer.playback_buttons[0].accessibleName() == "Previous frame"
    assert viewer.playback_buttons[2].accessibleName() == "Next frame"
    assert "mode_buttons" not in viewer.__dict__
    viewer.close()


def test_playback_icons_follow_palette_and_use_single_frame_arrows(tmp_path, qapp):
    from PySide6.QtGui import QColor, QIcon, QPalette
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2))
    for color in (QColor("#f4f4f4"), QColor("#202020")):
        palette = QPalette(viewer.palette())
        palette.setColor(QPalette.ColorRole.ButtonText, color)
        disabled_color = QColor("#858585") if color.red() > 128 else QColor("#a0a0a0")
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled_color)
        viewer.setPalette(palette)
        qapp.processEvents()
        for button in viewer.playback_buttons:
            image = button.icon().pixmap(32, 32).toImage()
            matching = sum(
                1 for y in range(image.height()) for x in range(image.width())
                if image.pixelColor(x, y).alpha() > 0
                and image.pixelColor(x, y).red() == color.red()
                and image.pixelColor(x, y).green() == color.green()
                and image.pixelColor(x, y).blue() == color.blue()
            )
            assert matching > 0
        disabled_image = viewer.playback_buttons[0].icon().pixmap(32, 32, QIcon.Mode.Disabled).toImage()
        assert any(disabled_image.pixelColor(x, y) == disabled_color
                   for y in range(disabled_image.height()) for x in range(disabled_image.width()))
    assert viewer.playback_buttons[0].accessibleName() == "Previous frame"
    assert viewer.playback_buttons[2].accessibleName() == "Next frame"
    assert viewer.playback_icon_kinds == ["previous_frame", "play", "next_frame"]
    viewer.close()


def test_hud_layout_updates_are_coalesced_after_content_and_viewport_settle(tmp_path, qapp):
    from PySide6.QtTest import QTest
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 3))
    viewer.resize(1280, 800)
    viewer.show()
    viewer._set_hud(scale="large")
    qapp.processEvents()
    panels = [viewer.overlay_panel, *viewer.split_panels]
    calls = {panel: 0 for panel in panels}
    for panel in panels:
        fit = panel._fit_content
        panel._fit_content = lambda fit=fit, panel=panel: (calls.__setitem__(panel, calls[panel] + 1), fit())[-1]

    changes = [
        lambda: viewer.set_mode("split"),
        lambda: viewer.set_mode("overlay"),
        lambda: viewer.confidence_numeric_checkbox.setChecked(True),
        lambda: viewer.confidence_bar_checkbox.setChecked(True),
        lambda: viewer.visibility_checks[1].setChecked(False),
        lambda: viewer.visibility_checks[1].setChecked(True),
        lambda: viewer.reference_selector.setCurrentIndex(1),
        lambda: next(check for check in viewer.metric_checks.values() if check.isEnabled()).setChecked(True),
    ]
    for change in changes:
        before = dict(calls)
        change()
        assert all(not panel.card.updatesEnabled() for panel in panels)
        assert calls == before
        qapp.processEvents()
        QTest.qWait(1)
        qapp.processEvents()
        assert all(panel.card.updatesEnabled() for panel in panels), [
            (index, panel.card.updatesEnabled(), panel._fit_timer.isActive())
            for index, panel in enumerate(panels)
        ]
        assert all(calls[panel] - before[panel] <= 1 for panel in panels)
        assert all(panel.requested_scale == "large" for panel in panels)
        assert all(panel.effective_point_size >= 7.0 for panel in panels)
    before = dict(calls)
    viewer.set_mode("split")
    viewer.confidence_numeric_checkbox.setChecked(False)
    viewer.set_mode("overlay")
    viewer.confidence_numeric_checkbox.setChecked(True)
    viewer.reference_selector.setCurrentIndex(2)
    assert calls == before
    qapp.processEvents()
    assert all(calls[panel] - before[panel] <= 1 for panel in panels)
    assert all(panel.card.updatesEnabled() for panel in panels)
    assert all(panel.requested_scale == "large" for panel in panels)
    viewer.close()


def test_viewer_speed_stepper_bounds_and_play_state_icon(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer
    from calibfreehands.visualization.playback import PlaybackClock

    viewer = ComparisonViewer(_session(tmp_path, 2))
    viewer._step_playback_speed(-1)
    assert viewer.playback_clock.speed == PlaybackClock.SPEEDS[3]
    viewer._step_playback_speed(1)
    assert viewer.playback_clock.speed == 1.0
    viewer.playback_clock.set_speed(PlaybackClock.SPEEDS[0])
    viewer._sync_speed_controls()
    assert not viewer.speed_down_button.isEnabled()
    viewer.playback_clock.set_speed(PlaybackClock.SPEEDS[-1])
    viewer._sync_speed_controls()
    assert not viewer.speed_up_button.isEnabled()
    viewer._play()
    assert viewer.playback_buttons[1].accessibleName() == "Pause"
    assert viewer.playback_icon_kinds[1] == "pause"
    viewer._play()
    assert viewer.playback_buttons[1].accessibleName() == "Play"
    assert viewer.playback_icon_kinds[1] == "play"
    viewer.close()


def test_hud_can_be_dragged_and_snaps_to_nearest_corner_without_blocking(tmp_path, qapp):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2))
    viewer.resize(1280, 800); viewer.show(); qapp.processEvents()
    panel = viewer.overlay_panel
    panel._finish_drag(QPoint(20, 20))
    assert panel.anchor == "top-left"
    assert panel.snap_animation.duration() in range(120, 181)
    QTest.mousePress(panel.card, Qt.MouseButton.LeftButton, pos=panel.card.rect().center())
    QTest.mouseMove(panel.card, QPoint(panel.card.width() - 20, panel.card.height() - 20), 10)
    QTest.mouseRelease(panel.card, Qt.MouseButton.LeftButton, pos=QPoint(panel.card.width() - 20, panel.card.height() - 20))
    qapp.processEvents()
    assert panel.anchor in {"top-left", "top-right", "bottom-left", "bottom-right"}
    assert not viewer._panel_reorder_active
    viewer.resize(1440, 900); qapp.processEvents()
    assert panel.anchor in {"top-left", "top-right", "bottom-left", "bottom-right"}
    viewer.close()


def test_hud_snap_animation_reaches_each_corner_and_preserves_size(tmp_path, qapp):
    from PySide6.QtCore import QPoint
    from PySide6.QtTest import QTest
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2))
    viewer.resize(1280, 800); viewer.show(); qapp.processEvents()
    panel = viewer.overlay_panel
    requested_size = panel.requested_scale
    right = panel.width() - panel.card.width() - 12
    bottom = panel.height() - panel.card.height() - 12
    anchors = {
        "top-left": QPoint(12, 12),
        "top-right": QPoint(right, 12),
        "bottom-left": QPoint(12, bottom),
        "bottom-right": QPoint(right, bottom),
    }
    for name, position in anchors.items():
        panel.card.move(position + QPoint(3, 4))
        panel._finish_drag()
        QTest.qWait(175); qapp.processEvents()
        assert panel.anchor == name
        assert panel.card.pos() == position
        assert panel.requested_scale == requested_size
        assert panel.snap_animation.duration() in range(120, 181)
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
    from PySide6.QtWidgets import QStyle, QStyleOptionToolButton
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2))
    viewer.resize(1280, 800); viewer.show(); qapp.processEvents()
    button = viewer.hud_button
    option = QStyleOptionToolButton(); button.initStyleOption(option)
    arrow = button.style().subControlRect(QStyle.ComplexControl.CC_ToolButton, option,
                                          QStyle.SubControl.SC_ToolButtonMenu, button)
    arrow_position = arrow.center()
    QTest.mouseClick(button, Qt.MouseButton.LeftButton, pos=arrow_position); qapp.processEvents()
    assert viewer.hud_menu.isVisible() and viewer._hover_menu_latched
    QTest.mouseMove(viewer.views, QPoint(40, 40)); qapp.processEvents()
    assert viewer.hud_menu.isVisible()
    QTest.mouseClick(button, Qt.MouseButton.LeftButton, pos=arrow_position); qapp.processEvents()
    assert not viewer.hud_menu.isVisible()
    QTest.mouseClick(button, Qt.MouseButton.LeftButton, pos=arrow_position); qapp.processEvents()
    assert viewer.hud_menu.isVisible()
    QTest.keyClick(viewer.hud_menu, Qt.Key.Key_Escape); qapp.processEvents()
    assert not viewer.hud_menu.isVisible()
    QTest.mouseClick(button, Qt.MouseButton.LeftButton, pos=arrow_position); qapp.processEvents()
    assert viewer.hud_menu.isVisible()
    QTest.mouseClick(viewer.playback_buttons[-1], Qt.MouseButton.LeftButton); qapp.processEvents()
    assert not viewer.hud_menu.isVisible() and not viewer._hover_menu_latched
    viewer.close()


def test_joint_annotation_density_and_label_content_use_joint_errors(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2))
    layer = viewer.split_annotations[1]
    viewer.set_mode("overlay")
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


def test_identical_candidate_joint_labels_are_deterministically_staggered(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 3))
    viewer.set_mode("overlay")
    viewer._set_annotation_content("error")
    first, second = viewer.loaded_results[1].result, viewer.loaded_results[2].result
    second.keypoints_3d[:] = first.keypoints_3d
    layer = viewer.overlay_annotations
    for mode in ("key_joints", "worst5", "all"):
        viewer.set_annotation_mode(mode)
        labels = layer.projected_annotations()
        same_joint = [item for item in labels if item[1] == "left" and item[2] == 0]
        assert len(same_joint) == 2
        assert same_joint[0][4:] != same_joint[1][4:]
    viewer.close()


def test_compact_presentation_explains_disabled_options_and_hud_background_default(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer

    viewer = ComparisonViewer(_session(tmp_path, 2))
    assert "f@5" not in viewer.metric_checks
    assert viewer.results_menu_button.text() == "Visible (2/2)"
    assert viewer.error_vectors_button.text() == "Error vectors: Off"
    assert viewer.overlay_panel.background == "none"
    assert "transparent" in viewer.overlay_panel.card.styleSheet()
    assert viewer.metric_menu_button.text() == "Error Types"
    assert viewer.confidence_menu_button.toolTip().startswith("YOLO hand detector")
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


@pytest.mark.parametrize("selector_name", ["reference_selector"])
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
