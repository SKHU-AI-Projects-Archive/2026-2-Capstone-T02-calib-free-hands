"""Common interface for single-image camera calibration models.

Every adapter returns the same record, always in **original-image pixels**.
Values a model does not predict stay ``None`` — they are never invented, and a
missing principal point is reported as NOT_PREDICTED rather than silently
filled with the image centre.

The conversion from each model's native output to original-image pixels is the
single most dangerous step in this benchmark, so every adapter stores its raw
output and the transform it applied alongside the converted value.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class CalibrationPrediction:
    model: str
    sequence: str = ""
    camera: str = ""
    frame: int = 0
    image_width: int = 0
    image_height: int = 0
    success: bool = False
    fx_px: float | None = None
    fy_px: float | None = None
    cx_px: float | None = None
    cy_px: float | None = None
    distortion: str = ""                 # raw, model-specific, not converted
    hfov_deg: float | None = None
    vfov_deg: float | None = None
    runtime_ms: float = float("nan")
    raw_output: str = ""                 # native output, before conversion
    conversion_note: str = ""            # what transform was applied
    failure_reason: str = ""
    extra: dict = field(default_factory=dict)


def fov_from_focal(f_px: float, extent_px: float) -> float:
    """Full field of view in degrees for a pinhole camera."""
    if not f_px or f_px <= 0 or extent_px <= 0:
        return float("nan")
    return float(2.0 * math.degrees(math.atan(0.5 * extent_px / f_px)))


def focal_from_fov(fov_deg: float, extent_px: float) -> float:
    """Inverse of :func:`fov_from_focal`."""
    if not fov_deg or fov_deg <= 0 or extent_px <= 0:
        return float("nan")
    return float(0.5 * extent_px / math.tan(0.5 * math.radians(fov_deg)))


class CalibrationAdapter:
    """Base adapter. Subclasses implement ``_load`` and ``_predict``."""

    name = "base"
    predicts_principal_point = False
    predicts_distortion = False

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None

    def load(self):
        if self._model is None:
            self._model = self._load()
        return self._model

    def _load(self):
        raise NotImplementedError

    def _predict(self, img_bgr, meta: dict) -> CalibrationPrediction:
        raise NotImplementedError

    def predict(self, img_bgr, meta: dict) -> CalibrationPrediction:
        import time
        self.load()
        t0 = time.perf_counter()
        try:
            pred = self._predict(img_bgr, meta)
            pred.success = pred.fx_px is not None and pred.fx_px > 0
        except Exception as exc:  # a model failure is data, not a crash
            pred = CalibrationPrediction(model=self.name, success=False,
                                         failure_reason=f"{type(exc).__name__}: {exc}")
        pred.runtime_ms = (time.perf_counter() - t0) * 1000.0
        pred.model = self.name
        pred.sequence = meta.get("sequence", "")
        pred.camera = meta.get("camera", "")
        pred.frame = int(meta.get("frame", 0))
        pred.image_width = int(meta.get("image_width", img_bgr.shape[1]))
        pred.image_height = int(meta.get("image_height", img_bgr.shape[0]))
        if pred.success:
            if pred.hfov_deg is None:
                pred.hfov_deg = fov_from_focal(pred.fx_px, pred.image_width)
            if pred.vfov_deg is None:
                pred.vfov_deg = fov_from_focal(pred.fy_px or pred.fx_px,
                                               pred.image_height)
        return pred
