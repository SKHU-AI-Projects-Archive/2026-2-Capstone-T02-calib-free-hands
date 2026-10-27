from __future__ import annotations

import math

from calibfreehands.app.configuration import ApplicationConfiguration
from .anycalib_backend import AnyCalibBackend


class ReferenceFocalBackend:
    name = "Checkerboard Reference"

    def __init__(self, focal_px: float) -> None:
        if not math.isfinite(focal_px) or focal_px <= 0:
            raise ValueError("reference focal must be finite and positive")
        self.reference_focal_px = float(focal_px)
        self.model_id = "checkerboard_reference"

    def estimate_focal(self, image_rgb) -> float:
        return self.reference_focal_px


def create_calibration_backend(
    model_id: str, config: ApplicationConfiguration, device: str = "cuda",
    reference_focal_px: float | None = None,
):
    try:
        model = config.calibration_models[model_id]
    except KeyError as exc:
        raise KeyError(f"Unknown calibration model: {model_id}") from exc
    if not model.available:
        raise RuntimeError(f"Calibration model {model.display_name} is not available: {model.unavailable_reason}")
    if model.backend == "reference_focal":
        if reference_focal_px is None:
            raise ValueError("Checkerboard Reference requires an explicit focal length in pixels")
        return ReferenceFocalBackend(reference_focal_px)
    if model.backend == "anycalib":
        return AnyCalibBackend(model_id=model.model_id or "anycalib_pinhole", device=device)
    raise RuntimeError(f"No runtime factory is registered for calibration backend: {model.backend}")
