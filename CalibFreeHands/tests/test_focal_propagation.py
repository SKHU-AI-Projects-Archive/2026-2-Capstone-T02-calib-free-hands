import numpy as np

from calibfreehands.calibration.focal import extract_pinhole_focal, rescale_camera_translation


def test_focal_extraction_and_translation_rescale():
    assert extract_pinhole_focal(np.array([700.0, 710.0, 320.0, 240.0])) == 705.0
    original = np.array([0.1, -0.2, 2.0])
    np.testing.assert_allclose(rescale_camera_translation(original, 500.0, 750.0), [0.1, -0.2, 3.0])
