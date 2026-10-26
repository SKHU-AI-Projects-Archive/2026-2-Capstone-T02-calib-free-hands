import numpy as np
import pytest

from calibfreehands.pipeline.result import CanonicalResult


def test_detected_hand_cannot_have_all_nan_coordinates():
    with pytest.raises(ValueError, match="detected hands"):
        CanonicalResult(np.array([0]), np.array([0.]), np.full((1, 2, 21, 2), np.nan),
                        np.full((1, 2, 21, 3), np.nan), np.array([[True, False]]), np.array([500.]), (10, 10))
