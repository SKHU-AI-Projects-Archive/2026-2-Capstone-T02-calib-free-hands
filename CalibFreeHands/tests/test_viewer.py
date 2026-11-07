import numpy as np
from pyqtgraph.opengl.GLGraphicsItem import GLOptions
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QCheckBox, QSizePolicy, QWidget

from calibfreehands.pipeline import CanonicalResult, save_result
from calibfreehands.visualization.viewer import load_npz_result


def test_viewer_loads_saved_multiframe_result(tmp_path):
    keypoints_2d = np.full((2, 2, 21, 2), np.nan); keypoints_2d[0, 0] = 0; keypoints_2d[1] = 0
    keypoints_3d = np.full((2, 2, 21, 3), np.nan); keypoints_3d[0, 0] = 0; keypoints_3d[1] = 0
    result = CanonicalResult(
        np.array([0, 1]), np.array([0.0, 0.1]), keypoints_2d, keypoints_3d, np.array([[True, False], [True, True]]), np.array([500., 501.]), (640, 480),
        mano_vertices=np.zeros((2, 2, 10, 3)),
        mano_faces=np.array([[[0, 1, 2]], [[0, 2, 1]]], dtype=np.int32),
    )
    npz, _ = save_result(result, tmp_path, {"schema_version": 1})
    loaded = load_npz_result(npz)
    assert loaded.frame_count == 2
    assert loaded.has_mesh
    assert loaded.has_mesh_topology
    assert not loaded.detected[0, 1]


def test_legacy_raw_pose_npz_has_a_readable_validation_error(tmp_path):
    path = tmp_path / "raw.npz"
    np.savez(path, keypoints_3d_relative=np.zeros((1, 2, 21, 3)))
    with pytest.raises(ValueError, match="internal raw-pose cache"):
        load_npz_result(path)


def test_bundle_viewer_opens_3d_when_source_media_is_missing(tmp_path, qapp):
    from calibfreehands.pipeline.result_bundle import write_result_bundle
    from calibfreehands.visualization.viewer import ResultViewer

    result = _viewer_result(mesh=False)
    result_dir = tmp_path / "internal"
    npz, run = save_result(result, result_dir, {"input_path": str(tmp_path / "removed.mp4")})
    metadata = {
        "application_version": "0.1.0", "source_media_name": "removed.mp4",
        "source_media_path": str(tmp_path / "removed.mp4"), "source_media_sha256": "a" * 64,
        "calibration_model_id": "anycalib_original", "calibration_display_name": "Original AnyCalib",
        "reconstruction_model_id": "wilor_original", "reconstruction_display_name": "Original WiLoR",
        "coordinate_system": "camera_space", "unit": "metre", "focal_px": 500.0,
        "sampling_mode": "uniform", "sample_count": 8, "aggregation": "median",
        "principal_point_applied": False, "distortion_applied": False,
        "run_provenance": {"input_path": str(tmp_path / "removed.mp4")},
    }
    bundle = write_result_bundle(npz, metadata, tmp_path / "removed.cfh")
    viewer = ResultViewer(bundle)
    assert viewer.loaded.frame_count == 2
    assert viewer.loaded.media_frames is None
    assert viewer.loaded.source_status == "missing"
    assert "Source media missing" in viewer.status.text()
    viewer.close()


def test_viewer_widget_constructs_offscreen(tmp_path, qapp):
    result = CanonicalResult(np.array([0]), np.array([0.0]), np.zeros((1, 2, 21, 2)), np.zeros((1, 2, 21, 3)),
                             np.array([[True, True]]), np.array([500.]), (640, 480))
    npz, _ = save_result(result, tmp_path, {"schema_version": 1})
    from calibfreehands.visualization.viewer import ResultViewer
    viewer = ResultViewer(npz)
    assert viewer.windowTitle().startswith("CalibFreeHands")
    from pyqtgraph.opengl import GLViewWidget
    assert isinstance(viewer.gl_view, GLViewWidget)
    viewer.close()


def test_overlay_float32_and_nan_coordinates_render_without_qt_errors(qapp):
    from calibfreehands.visualization.viewer import LoadedResult, _OverlayCanvas
    keypoints = np.full((1, 2, 21, 2), np.nan, dtype=np.float32)
    keypoints[0, 0] = np.arange(42, dtype=np.float32).reshape(21, 2) + 20.5
    keypoints[0, 0, 7] = np.nan
    keypoints_3d = np.full((1, 2, 21, 3), np.nan, dtype=np.float32); keypoints_3d[0, 0] = 0
    result = CanonicalResult(np.array([0]), np.array([0.0]), keypoints, keypoints_3d,
                             np.array([[True, False]]), np.array([500.], dtype=np.float32), (64, 48))
    canvas = _OverlayCanvas(LoadedResult(result, [np.zeros((48, 64, 3), dtype=np.uint8)]))
    canvas.resize(64, 48)
    canvas.paintEvent(None)
    assert not canvas.isVisible() or canvas.size().width() == 64


def test_overlay_keeps_last_source_image_during_cache_miss_and_ignores_stale_ready(qapp):
    from calibfreehands.visualization.viewer import LoadedResult, _OverlayCanvas
    first = np.full((48, 64, 3), (210, 20, 30), dtype=np.uint8)
    keypoints_2d = np.full((2, 2, 21, 2), np.nan, dtype=np.float32)
    keypoints_3d = np.full((2, 2, 21, 3), np.nan, dtype=np.float32)
    result = CanonicalResult(np.array([0, 1]), np.array([0.0, .1]), keypoints_2d, keypoints_3d,
                             np.zeros((2, 2), bool), np.full(2, 500.), (64, 48))
    canvas = _OverlayCanvas(LoadedResult(result, [first, None]))
    canvas.resize(64, 48)
    canvas.paintEvent(None)
    assert canvas._display_image.pixelColor(0, 0).red() > 180
    canvas.set_frame(1)
    canvas.paintEvent(None)
    assert canvas._display_image.pixelColor(0, 0).red() > 180
    updates = []
    canvas.update = lambda: updates.append(True)
    canvas._frame_ready(0)
    assert not updates


def _viewer_result(mesh=True):
    keypoints_2d = np.full((2, 2, 21, 2), np.nan, dtype=np.float32); keypoints_2d[:, 0] = 0
    keypoints_3d = np.full((2, 2, 21, 3), np.nan, dtype=np.float32)
    keypoints_3d[:, 0] = np.array([1., 2., 3.], dtype=np.float32)
    vertices = np.full((2, 2, 4, 3), np.nan, dtype=np.float32)
    vertices[:, 0] = np.array(((0., 0., 0.), (2., 0., 0.), (0., 4., 0.), (0., 0., 6.)), dtype=np.float32)
    return CanonicalResult(np.array([0, 1]), np.array([0., .25]), keypoints_2d, keypoints_3d,
                           np.array([[True, False], [True, False]]), np.array([500., 500.]), (64, 48),
                           mano_vertices=vertices if mesh else None,
                           mano_faces=np.array([[[0, 1, 2]], [[0, 2, 3]]], dtype=np.int32) if mesh else None)


def test_viewer_mesh_grid_axis_and_fit_contract(tmp_path, qapp):
    result = _viewer_result(); before_joints = result.keypoints_3d.copy(); before_vertices = result.mano_vertices.copy()
    npz, _ = save_result(result, tmp_path, {"schema_version": 1})
    from calibfreehands.visualization.viewer import ResultViewer
    viewer = ResultViewer(npz)
    checkboxes = {box.text(): box for box in viewer.findChildren(QCheckBox)}
    assert checkboxes["Joint"].isChecked() and checkboxes["Bone"].isChecked()
    assert not checkboxes["Mesh"].isChecked() and checkboxes["Mesh"].isEnabled()
    assert not viewer.gl_view.show_mesh and len(viewer.gl_view.grids) == 3 and len(viewer.gl_view.axis_labels) == 3
    checkboxes["Grid"].setChecked(False)
    assert not any(grid.visible() for grid in viewer.gl_view.grids)
    assert not any(label.visible() for label in viewer.gl_view.numeric_labels)
    checkboxes["Grid"].setChecked(True)
    assert all(label.visible() for label in viewer.gl_view.numeric_labels)
    checkboxes["Axis"].setChecked(False)
    assert not viewer.gl_view.axis.visible() and not any(label.visible() for label in viewer.gl_view.axis_labels)
    viewer.gl_view.fit_hands(); viewer.gl_view.front_view(); viewer.gl_view.side_view(); viewer.gl_view.top_view(); viewer.slider.setValue(1)
    np.testing.assert_array_equal(result.keypoints_3d, before_joints)
    np.testing.assert_array_equal(result.mano_vertices, before_vertices)
    viewer.close()


def test_fixed_metric_workspace_planes_axes_and_major_labels(tmp_path, qapp):
    from PySide6.QtGui import QVector3D
    from calibfreehands.visualization.viewer import (
        GRID_MAJOR_SPACING_M, GRID_MINOR_SPACING_M, WORKSPACE_BOUNDS_M,
        GRID_LABEL_OFFSET_M, ResultViewer, _view_point_from_raw,
    )

    npz, _ = save_result(_viewer_result(mesh=False), tmp_path, {"schema_version": 1})
    viewer = ResultViewer(npz)
    view = viewer.gl_view
    assert WORKSPACE_BOUNDS_M == ((-1.0, 1.0), (-1.0, 1.0), (0.0, 2.0))
    assert GRID_MINOR_SPACING_M == pytest.approx(0.1)
    assert GRID_MAJOR_SPACING_M == pytest.approx(0.5)
    assert len(view.grids) == 3
    for grid in view.grids:
        assert grid.size() == pytest.approx([2.0, 2.0, 1.0])
        assert grid.spacing() == pytest.approx([0.1, 0.1, 1.0])

    planes = [grid.transform().map(QVector3D(0, 0, 0)) for grid in view.grids]
    assert [point.toTuple() for point in planes] == pytest.approx([(0, 0, 0), (0, -1, 0), (0, -1, 0)])
    assert view.grids[0].transform().map(QVector3D(0, 1, 0)).toTuple() == pytest.approx((0, 0, 1))
    assert view.grids[1].transform().map(QVector3D(0, 1, 0)).toTuple() == pytest.approx((0, 0, 0))
    assert view.grids[2].transform().map(QVector3D(1, 0, 0)).toTuple() == pytest.approx((0, -1, -1))
    assert view.axis.pos == pytest.approx(np.asarray((
        _view_point_from_raw((-1, 0, 0)), _view_point_from_raw((1, 0, 0)),
        _view_point_from_raw((0, 1, 0)), _view_point_from_raw((0, -1, 0)),
        _view_point_from_raw((0, 0, 0)), _view_point_from_raw((0, 0, 2)),
    )))

    assert len(view.numeric_labels) == 13
    assert sum(label.text == "0" for label in view.numeric_labels) == 1
    origin_label = next(label for label in view.numeric_labels if label.text == "0")
    np.testing.assert_allclose(origin_label.pos,
                               _view_point_from_raw((GRID_LABEL_OFFSET_M, -GRID_LABEL_OFFSET_M,
                                                     GRID_LABEL_OFFSET_M)))
    for text, raw_point in (("50 cm", (-GRID_LABEL_OFFSET_M, 0, 0.5)),
                            ("50 cm", (GRID_LABEL_OFFSET_M, -0.5, 0)),
                            ("-50 cm", (GRID_LABEL_OFFSET_M, 0.5, 0)),
                            ("100 cm", (1, -GRID_LABEL_OFFSET_M, 0)),
                            ("150 cm", (-GRID_LABEL_OFFSET_M, 0, 1.5))):
        assert any(label.text == text and np.allclose(label.pos, _view_point_from_raw(raw_point))
                   for label in view.numeric_labels)
    assert [label.text for label in view.axis_labels] == ["X", "Y", "Z"]
    for label, raw_point in zip(view.axis_labels, ((1.3, 0, 0), (0, -1.3, 0), (0, 0, 2.3))):
        np.testing.assert_allclose(label.pos, _view_point_from_raw(raw_point))
    assert all(label.color.getRgb() == view.numeric_label_color for label in view.numeric_labels)
    viewer.close()


def test_metric_overview_is_result_independent_and_fit_all_only_moves_camera(qapp):
    from calibfreehands.visualization.viewer import (
        LoadedResult, _GLResultView, default_metric_camera_distance,
        metric_workspace_center_display,
    )

    first_result = _viewer_result(mesh=True)
    second_result = _viewer_result(mesh=True)
    second_result.keypoints_3d[:, 0] = (0.0, 0.0, 15.0)
    second_result.mano_vertices[:, 0] += (0.0, 0.0, 15.0)
    raw_before = second_result.keypoints_3d.copy()
    mesh_before = second_result.mano_vertices.copy()
    first = _GLResultView(LoadedResult(first_result))
    second = _GLResultView(LoadedResult(second_result))
    first.metric_overview()
    second.metric_overview()
    first_camera = first.cameraParams()
    second_camera = second.cameraParams()
    np.testing.assert_allclose(first_camera["center"].toTuple(), metric_workspace_center_display())
    np.testing.assert_allclose(second_camera["center"].toTuple(), first_camera["center"].toTuple())
    assert first_camera["distance"] == pytest.approx(default_metric_camera_distance())
    assert second_camera["distance"] == pytest.approx(first_camera["distance"])

    grid_transforms = [grid.transform() for grid in (second.grids)]
    grid_positions = second.axis.pos.copy()
    label_positions = [label.pos.copy() for label in second.numeric_labels]
    reference_item_ids = (tuple(map(id, second.grids)), tuple(map(id, second.numeric_labels)))
    second.set_frame(1)
    assert second.cameraParams()["distance"] == pytest.approx(first_camera["distance"])
    assert (tuple(map(id, second.grids)), tuple(map(id, second.numeric_labels))) == reference_item_ids
    assert second.fit_all()
    fitted = second.cameraParams()
    assert fitted["distance"] != pytest.approx(first_camera["distance"])
    np.testing.assert_array_equal(second.axis.pos, grid_positions)
    for actual, expected in zip(second.grids, grid_transforms):
        assert actual.transform() == expected
    for label, position in zip(second.numeric_labels, label_positions):
        np.testing.assert_array_equal(label.pos, position)
    np.testing.assert_array_equal(second_result.keypoints_3d, raw_before)
    np.testing.assert_array_equal(second_result.mano_vertices, mesh_before)

    before_empty = second.cameraParams()
    second.show_left = False
    assert not second.fit_all()
    assert second.cameraParams()["distance"] == pytest.approx(before_empty["distance"])
    np.testing.assert_allclose(second.cameraParams()["center"].toTuple(), before_empty["center"].toTuple())
    first.close(); second.close()


def test_whole_video_hand_framing_uses_robust_detected_wrist_and_joint_statistics():
    from calibfreehands.visualization.viewer import whole_video_hand_framing

    points = np.empty((4, 2, 21, 3), dtype=np.float64)
    points[:] = (0.11, -0.05, 0.61)
    detected = np.ones((4, 2), dtype=bool)
    for frame, wrist in enumerate(((0.10, -0.05, 0.60), (0.12, -0.04, 0.62),
                                   (0.11, -0.06, 0.61))):
        points[frame, :, 0] = wrist
    points[3, 0, 0] = (4.0, 5.0, 12.0)
    points[0, 0, 0] = np.nan
    detected[3, 1] = False
    points[3, 1] = np.nan
    raw_before = points.copy()

    target, distance = whole_video_hand_framing(points, detected)

    np.testing.assert_allclose(target, (-0.115, -0.615, 0.045), atol=1e-6)
    assert 0.3 < distance < 1.0
    np.testing.assert_array_equal(points, raw_before)

    points[:, :, 0] = (0.11, -0.05, 0.61)
    _, without_outlier_distance = whole_video_hand_framing(points, detected)
    assert abs(distance - without_outlier_distance) < 0.1


def test_whole_video_hand_framing_falls_back_to_joints_then_metric_workspace():
    from calibfreehands.visualization.viewer import (
        default_metric_camera_distance, metric_workspace_center_display,
        whole_video_hand_framing,
    )

    points = np.full((2, 2, 21, 3), np.nan, dtype=np.float64)
    detected = np.ones((2, 2), dtype=bool)
    points[:, :, 1:] = (0.2, -0.1, 0.8)
    target, distance = whole_video_hand_framing(points, detected)
    np.testing.assert_allclose(target, (-0.2, -0.8, 0.1), atol=1e-6)
    assert distance > 0

    detected[:] = False
    fallback_target, fallback_distance = whole_video_hand_framing(points, detected)
    np.testing.assert_allclose(fallback_target, metric_workspace_center_display())
    assert fallback_distance == pytest.approx(default_metric_camera_distance())


def test_hand_space_bounds_are_robust_metric_and_fall_back_to_physical_workspace():
    from calibfreehands.visualization.viewer import (
        HAND_SPACE_MIN_SPAN_M, WORKSPACE_BOUNDS_M, hand_space_bounds, nice_grid_spacing,
    )

    keypoints = np.full((12, 2, 21, 3), np.nan, dtype=np.float64)
    detected = np.zeros((12, 2), dtype=bool)
    for frame in range(12):
        detected[frame, 0] = True
        keypoints[frame, 0, :, 0] = np.linspace(-0.12, 0.12, 21) + frame * 0.005
        keypoints[frame, 0, :, 1] = np.linspace(-0.08, 0.08, 21)
        keypoints[frame, 0, :, 2] = np.linspace(0.7, 1.0, 21)
    keypoints[-1, 0] += 50.0
    raw_before = keypoints.copy()

    bounds = hand_space_bounds(keypoints, detected)

    assert bounds.shape == (3, 2)
    assert np.isfinite(bounds).all()
    assert np.all(bounds[:, 1] > bounds[:, 0])
    assert np.all(bounds[:, 1] - bounds[:, 0] >= HAND_SPACE_MIN_SPAN_M)
    assert np.max(np.abs(bounds)) < 2.0
    np.testing.assert_array_equal(keypoints, raw_before)
    assert hand_space_bounds(keypoints, np.zeros_like(detected)).tolist() == [list(axis) for axis in WORKSPACE_BOUNDS_M]
    assert nice_grid_spacing(2.0) == pytest.approx(0.5)
    assert nice_grid_spacing(0.35) == pytest.approx(0.05)


def test_physical_y_labels_and_positive_axis_follow_y_up_display(tmp_path, qapp):
    from calibfreehands.visualization.viewer import ResultViewer, _view_point_from_raw

    result = _viewer_result(mesh=False)
    raw_before = result.keypoints_3d.copy()
    npz, _ = save_result(result, tmp_path, {"schema_version": 1})
    viewer = ResultViewer(npz)
    gl_view = viewer.gl_view

    y_segment = gl_view.axis.pos[2:4]
    assert y_segment[1, 2] > y_segment[0, 2]
    assert any(label.text == "100 cm" and np.allclose(label.pos, _view_point_from_raw((0.15, -1.0, 0.0)))
               for label in gl_view.numeric_labels)
    assert any(label.text == "-100 cm" and np.allclose(label.pos, _view_point_from_raw((0.15, 1.0, 0.0)))
               for label in gl_view.numeric_labels)
    np.testing.assert_array_equal(viewer.loaded.result.keypoints_3d, raw_before)
    viewer.close()


def test_single_result_viewer_starts_in_hand_view_and_exposes_camera_modes(tmp_path, qapp):
    from PySide6.QtWidgets import QPushButton
    from calibfreehands.visualization.viewer import (
        ResultViewer, whole_video_hand_framing,
    )

    result = _viewer_result(mesh=True)
    result.keypoints_3d[0, 0] = (0.1, -0.05, 0.6)
    result.keypoints_3d[1, 0] = (0.3, -0.15, 1.0)
    result.camera_translation = np.full((2, 2, 3), 0.25, dtype=np.float32)
    npz, _ = save_result(result, tmp_path, {"schema_version": 1})
    expected_target, expected_distance = whole_video_hand_framing(result.keypoints_3d, result.detected)
    viewer = ResultViewer(npz)
    raw_joints = viewer.loaded.result.keypoints_3d.copy()
    raw_vertices = viewer.loaded.result.mano_vertices.copy()
    raw_translation = viewer.loaded.result.camera_translation.copy()
    params = viewer.gl_view.cameraParams()
    np.testing.assert_allclose(params["center"].toTuple(), expected_target)
    assert params["distance"] == pytest.approx(expected_distance)
    assert {button.text() for button in viewer.findChildren(QPushButton)} >= {
        "Hand View", "Metric Overview", "Fit All"
    }
    assert "Fixed metric grid" in viewer.coordinate_info.text()
    assert "View centered on hands" not in viewer.coordinate_info.text()
    viewer._set_frame(1)
    frame_camera = viewer.gl_view.cameraParams()
    np.testing.assert_allclose(tuple(frame_camera["center"]), expected_target)
    assert frame_camera["distance"] == pytest.approx(expected_distance)

    viewer.gl_view.metric_overview()
    np.testing.assert_allclose(viewer.gl_view.cameraParams()["center"].toTuple(), (0, -1, 0))
    viewer.gl_view.hand_view()
    np.testing.assert_allclose(viewer.gl_view.cameraParams()["center"].toTuple(), expected_target)
    viewer.gl_view.reset_camera()
    np.testing.assert_allclose(viewer.gl_view.cameraParams()["center"].toTuple(), expected_target)
    np.testing.assert_array_equal(viewer.loaded.result.keypoints_3d, raw_joints)
    np.testing.assert_array_equal(viewer.loaded.result.mano_vertices, raw_vertices)
    np.testing.assert_array_equal(viewer.loaded.result.camera_translation, raw_translation)
    viewer.close()


def test_legacy_result_disables_mesh_checkbox(tmp_path, qapp):
    result = _viewer_result(mesh=False)
    npz, _ = save_result(result, tmp_path, {"schema_version": 1})
    from calibfreehands.visualization.viewer import ResultViewer
    viewer = ResultViewer(npz)
    mesh = next(box for box in viewer.findChildren(QCheckBox) if box.text() == "Mesh")
    assert not mesh.isChecked() and not mesh.isEnabled()
    viewer.close()


def test_mesh_and_skeleton_visibility_modes_remain_synchronized(tmp_path, qapp):
    npz, _ = save_result(_viewer_result(), tmp_path, {"schema_version": 1})
    from calibfreehands.visualization.viewer import ResultViewer

    viewer = ResultViewer(npz)
    checks = {box.text(): box for box in viewer.findChildren(QCheckBox)}

    def assert_mode(joints, bones, mesh):
        assert checks["Joint"].isChecked() is joints
        assert checks["Bone"].isChecked() is bones
        assert checks["Mesh"].isChecked() is mesh
        assert viewer.overlay.show_joints is joints
        assert viewer.overlay.show_bones is bones
        assert viewer.gl_view.show_joints is joints
        assert viewer.gl_view.show_bones is bones
        assert viewer.gl_view.show_mesh is mesh

    assert_mode(True, True, False)

    checks["Mesh"].setChecked(True)
    assert_mode(False, False, True)
    assert not viewer.gl_view.joints[0].visible()
    assert not viewer.gl_view.bones[0].visible()
    assert viewer.gl_view.meshes[0].visible()
    viewer.slider.setValue(1)
    assert_mode(False, False, True)
    assert viewer.gl_view.meshes[0].visible()

    checks["Joint"].setChecked(True)
    assert_mode(True, False, False)
    assert viewer.gl_view.joints[0].visible()
    assert not viewer.gl_view.bones[0].visible()
    assert not viewer.gl_view.meshes[0].visible()

    checks["Mesh"].setChecked(True)
    assert_mode(False, False, True)
    checks["Bone"].setChecked(True)
    assert_mode(False, True, False)
    assert not viewer.gl_view.joints[0].visible()
    assert viewer.gl_view.bones[0].visible()
    assert not viewer.gl_view.meshes[0].visible()

    checks["Joint"].setChecked(True)
    assert_mode(True, True, False)
    checks["Joint"].setChecked(False)
    assert_mode(False, True, False)
    checks["Bone"].setChecked(False)
    assert_mode(False, False, False)

    checks["Mesh"].setChecked(True)
    assert_mode(False, False, True)
    checks["Mesh"].setChecked(False)
    assert_mode(False, False, False)
    viewer.close()


def test_opacity_controls_use_compact_nested_layout(tmp_path, qapp):
    npz, _ = save_result(_viewer_result(), tmp_path, {"schema_version": 1})
    from calibfreehands.visualization.viewer import ResultViewer

    viewer = ResultViewer(npz)
    assert isinstance(viewer.opacity_container, QWidget)
    assert viewer.opacity_container.layout() is viewer.opacity_group
    assert viewer.opacity_container.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Fixed
    margins = viewer.opacity_group.contentsMargins()
    assert (margins.left(), margins.top(), margins.right(), margins.bottom()) == (0, 0, 0, 0)
    assert viewer.opacity_group.spacing() <= 6
    assert viewer.opacity_group.itemAt(0).widget().text() == "Opacity"
    assert viewer.opacity_group.itemAt(1).widget() is viewer.mesh_opacity_slider
    assert viewer.opacity_group.itemAt(2).widget() is viewer.mesh_opacity_value
    assert viewer.mesh_opacity_slider.minimumWidth() == viewer.mesh_opacity_slider.maximumWidth() == 120
    expected_value_width = viewer.mesh_opacity_value.fontMetrics().horizontalAdvance("100%") + 4
    assert viewer.mesh_opacity_value.minimumWidth() == viewer.mesh_opacity_value.maximumWidth() == expected_value_width
    assert viewer.mesh_opacity_value.alignment() & Qt.AlignmentFlag.AlignRight
    assert viewer.mesh_opacity_value.alignment() & Qt.AlignmentFlag.AlignVCenter
    container_width = viewer.opacity_container.sizeHint().width()
    for percent in (10, 90, 100):
        viewer.mesh_opacity_slider.setValue(percent)
        assert viewer.mesh_opacity_value.text() == f"{percent}%"
        assert viewer.opacity_container.sizeHint().width() == container_width
    viewer.close()


def test_fit_uses_visible_two_hand_bounds_and_handles_empty_result(qapp):
    from calibfreehands.visualization.topology import RAW_TO_VIEW
    from calibfreehands.visualization.viewer import LoadedResult, _GLResultView
    result = _viewer_result(); result.detected[:, 1] = True; result.keypoints_2d[:, 1] = 0; result.keypoints_3d[:, 1] = np.array([[-2., 1., 2.]], dtype=np.float32)
    result.mano_vertices[:, 1] = np.array(((-3., 0., 0.), (-1., 0., 0.), (-3., 2., 0.), (-3., 0., 2.)), dtype=np.float32)
    view = _GLResultView(LoadedResult(result)); view.show_mesh = True; assert view._frame_camera()
    points = np.concatenate((result.keypoints_3d[0].reshape(-1, 3), result.mano_vertices[0].reshape(-1, 3))) @ RAW_TO_VIEW.T
    points = points[np.isfinite(points).all(axis=1)]
    np.testing.assert_allclose(view.fit_center, (points.min(axis=0) + points.max(axis=0)) / 2)
    empty = _viewer_result(); empty.detected[:] = False; empty.keypoints_2d[:] = np.nan; empty.keypoints_3d[:] = np.nan; empty.mano_vertices[:] = np.nan
    assert not _GLResultView(LoadedResult(empty))._frame_camera()


def _mouse_press():
    return QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(5, 5),
        QPointF(5, 5),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )


def _key_press(key):
    return QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)


def test_mesh_opacity_controls_render_mode_without_changing_vertices(tmp_path, qapp):
    result = _viewer_result()
    before_vertices = result.mano_vertices.copy()
    npz, _ = save_result(result, tmp_path, {"schema_version": 1})
    from calibfreehands.visualization.viewer import ResultViewer

    viewer = ResultViewer(npz)
    assert viewer.gl_view.mesh_opacity == 1.0
    assert viewer.mesh_opacity_slider.value() == 100
    assert viewer.mesh_opacity_value.text() == "100%"
    assert all(np.isclose(mesh.opts["color"][3], 1.0) for mesh in viewer.gl_view.meshes)
    assert all(mesh._GLGraphicsItem__glOpts == GLOptions["opaque"] for mesh in viewer.gl_view.meshes)

    viewer.mesh_opacity_slider.setValue(50)
    assert all(np.isclose(mesh.opts["color"][3], 0.5) for mesh in viewer.gl_view.meshes)
    assert all(mesh._GLGraphicsItem__glOpts == GLOptions["translucent"] for mesh in viewer.gl_view.meshes)
    viewer.slider.setValue(1)
    assert all(np.isclose(mesh.opts["color"][3], 0.5) for mesh in viewer.gl_view.meshes)

    viewer.mesh_opacity_slider.setValue(100)
    assert all(np.isclose(mesh.opts["color"][3], 1.0) for mesh in viewer.gl_view.meshes)
    assert all(mesh._GLGraphicsItem__glOpts == GLOptions["opaque"] for mesh in viewer.gl_view.meshes)
    np.testing.assert_array_equal(result.mano_vertices, before_vertices)
    viewer.close()


def test_legacy_result_disables_mesh_opacity(tmp_path, qapp):
    result = _viewer_result(mesh=False)
    npz, _ = save_result(result, tmp_path, {"schema_version": 1})
    from calibfreehands.visualization.viewer import ResultViewer

    viewer = ResultViewer(npz)
    assert not viewer.mesh_opacity_slider.isEnabled()
    viewer.close()


def test_axis_ranges_remain_fixed_to_metric_workspace(qapp):
    from calibfreehands.visualization.viewer import LoadedResult, _GLResultView

    view = _GLResultView(LoadedResult(_viewer_result()))
    assert np.linalg.norm(view.axis.pos[1] - view.axis.pos[0]) == pytest.approx(2.0)
    assert np.linalg.norm(view.axis.pos[3] - view.axis.pos[2]) == pytest.approx(2.0)
    assert np.linalg.norm(view.axis.pos[5] - view.axis.pos[4]) == pytest.approx(2.0)
    np.testing.assert_allclose(view.axis_labels[0].pos, (-1.3, 0, 0))
    np.testing.assert_allclose(view.axis_labels[1].pos, (0, 0, 1.3))
    np.testing.assert_allclose(view.axis_labels[2].pos, (0, -2.3, 0))


def test_click_context_routes_arrows_without_mutating_frame_in_orbit_mode(tmp_path, qapp):
    result = _viewer_result()
    npz, _ = save_result(result, tmp_path, {"schema_version": 1})
    from calibfreehands.visualization.viewer import ResultViewer

    viewer = ResultViewer(npz)
    viewer.eventFilter(viewer.gl_view, _mouse_press())
    assert viewer._arrow_mode == "orbit"
    frame = viewer.slider.value()
    for key in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
        before = (viewer.gl_view.opts["azimuth"], viewer.gl_view.opts["elevation"])
        assert viewer.eventFilter(viewer.gl_view, _key_press(key))
        after = (viewer.gl_view.opts["azimuth"], viewer.gl_view.opts["elevation"])
        assert after != before
    assert viewer.slider.value() == frame

    viewer.slider.setValue(1)
    viewer.eventFilter(viewer.overlay, _mouse_press())
    assert viewer._arrow_mode == "frame"
    assert viewer.eventFilter(viewer.overlay, _key_press(Qt.Key.Key_Left))
    assert viewer.slider.value() == 0
    assert viewer.eventFilter(viewer.overlay, _key_press(Qt.Key.Key_Right))
    assert viewer.slider.value() == 1
    assert viewer.eventFilter(viewer.overlay, _key_press(Qt.Key.Key_Up))
    assert viewer.slider.value() == 1
    viewer.close()


def test_space_toggles_playback_in_both_click_contexts(tmp_path, qapp):
    npz, _ = save_result(_viewer_result(), tmp_path, {"schema_version": 1})
    from calibfreehands.visualization.viewer import ResultViewer

    viewer = ResultViewer(npz)
    viewer.eventFilter(viewer.gl_view, _mouse_press())
    assert viewer.eventFilter(viewer.gl_view, _key_press(Qt.Key.Key_Space))
    assert viewer.timer.isActive()
    viewer.eventFilter(viewer.overlay, _mouse_press())
    assert viewer.eventFilter(viewer.overlay, _key_press(Qt.Key.Key_Space))
    assert not viewer.timer.isActive()
    viewer.close()
