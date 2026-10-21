import numpy as np
from pyqtgraph.opengl.GLGraphicsItem import GLOptions
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
    checkboxes["Axis"].setChecked(False)
    assert not viewer.gl_view.axis.visible() and not any(label.visible() for label in viewer.gl_view.axis_labels)
    viewer.gl_view.fit_hands(); viewer.gl_view.front_view(); viewer.gl_view.side_view(); viewer.gl_view.top_view(); viewer.slider.setValue(1)
    np.testing.assert_array_equal(result.keypoints_3d, before_joints)
    np.testing.assert_array_equal(result.mano_vertices, before_vertices)
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
    view = _GLResultView(LoadedResult(result)); assert view._frame_camera()
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


def test_axis_length_is_sixty_five_percent_of_grid_size(qapp):
    from calibfreehands.visualization.viewer import LoadedResult, _GLResultView

    view = _GLResultView(LoadedResult(_viewer_result()))
    expected_axis_length = 9.0 * 0.65
    assert np.isclose(np.linalg.norm(view.axis.pos[1] - view.axis.pos[0]), expected_axis_length)
    label_position = np.asarray(view.axis_labels[0].pos)
    assert np.isclose(np.linalg.norm(label_position - view.fit_center), expected_axis_length * 1.08)


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
