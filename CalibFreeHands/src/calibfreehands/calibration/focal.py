from __future__ import annotations

import numpy as np


def extract_pinhole_focal(intrinsics: np.ndarray) -> float:
    values = np.asarray(intrinsics, dtype=np.float64).reshape(-1)
    if values.size < 2 or not np.isfinite(values[:2]).all() or (values[0] <= 0 or values[1] <= 0):
        raise ValueError("AnyCalib returned invalid pinhole focal intrinsics")
    return float((values[0] + values[1]) / 2.0)


def rescale_camera_translation(cam_t: np.ndarray, source_focal_px: float, target_focal_px: float) -> np.ndarray:
    """Preserve WiLoR's perspective scale relation while changing focal length."""
    if source_focal_px <= 0 or target_focal_px <= 0:
        raise ValueError("focal lengths must be positive")
    translation = np.asarray(cam_t, dtype=np.float32).copy()
    translation[2] *= target_focal_px / source_focal_px
    return translation
