from __future__ import annotations

from calibfreehands.app.configuration import ApplicationConfiguration
from .anycalib_backend import AnyCalibBackend


def create_calibration_backend(
    model_id: str, config: ApplicationConfiguration, device: str = "cuda"
) -> AnyCalibBackend:
    try:
        model = config.calibration_models[model_id]
    except KeyError as exc:
        raise KeyError(f"Unknown calibration model: {model_id}") from exc
    if not model.available:
        raise RuntimeError(f"Calibration model {model.display_name} is not available: {model.unavailable_reason}")
    if model.backend == "anycalib":
        return AnyCalibBackend(model_id=model.model_id or "anycalib_pinhole", device=device)
    raise RuntimeError(f"No runtime factory is registered for calibration backend: {model.backend}")
