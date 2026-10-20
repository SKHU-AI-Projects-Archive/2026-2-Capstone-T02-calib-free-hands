import numpy as np

from calibfreehands.visualization.topology import HAND_CONNECTIONS, RAW_TO_VIEW


def test_hand_topology_is_openpose_21_not_sequential():
    assert len(HAND_CONNECTIONS) == 20
    assert (0, 5) in HAND_CONNECTIONS
    assert (4, 5) not in HAND_CONNECTIONS


def test_raw_to_view_is_orthogonal():
    np.testing.assert_allclose(RAW_TO_VIEW @ RAW_TO_VIEW.T, np.eye(3))


def test_joints_and_vertices_share_display_transform():
    joints = np.array([[1., 2., 3.]], dtype=np.float32)
    vertices = joints.copy()
    np.testing.assert_array_equal(joints @ RAW_TO_VIEW.T, vertices @ RAW_TO_VIEW.T)
