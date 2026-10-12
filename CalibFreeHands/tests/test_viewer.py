import numpy as np

from calibfreehands.pipeline import CanonicalResult, save_result
from calibfreehands.visualization.viewer import load_npz_result


def test_viewer_loads_saved_multiframe_result(tmp_path):
    keypoints_2d = np.full((2, 2, 21, 2), np.nan); keypoints_2d[0, 0] = 0; keypoints_2d[1] = 0
    keypoints_3d = np.full((2, 2, 21, 3), np.nan); keypoints_3d[0, 0] = 0; keypoints_3d[1] = 0
    result = CanonicalResult(
        np.array([0, 1]), np.array([0.0, 0.1]), keypoints_2d, keypoints_3d, np.array([[True, False], [True, True]]), np.array([500., 501.]), (640, 480),
        mano_vertices=np.zeros((2, 2, 10, 3)),
    )
    npz, _ = save_result(result, tmp_path, {"schema_version": 1})
    loaded = load_npz_result(npz)
    assert loaded.frame_count == 2
    assert loaded.has_mesh
    assert not loaded.detected[0, 1]


def test_viewer_widget_constructs_offscreen(tmp_path, qapp):
    result = CanonicalResult(np.array([0]), np.array([0.0]), np.zeros((1, 2, 21, 2)), np.zeros((1, 2, 21, 3)),
                             np.array([[True, True]]), np.array([500.]), (640, 480))
    npz, _ = save_result(result, tmp_path, {"schema_version": 1})
    from calibfreehands.visualization.viewer import ResultViewer
    viewer = ResultViewer(npz)
    assert viewer.windowTitle().startswith("CalibFreeHands")
    viewer.close()
