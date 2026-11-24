from __future__ import annotations

from pathlib import Path

from calibfreehands.app.configuration import ApplicationConfiguration
from .wilor_backend import OriginalWiLoRBackend


def create_pose_backend(
    model_id: str, config: ApplicationConfiguration, project_root: Path, device: str = "cuda"
) -> OriginalWiLoRBackend:
    try:
        model = config.reconstruction_models[model_id]
    except KeyError as exc:
        raise KeyError(f"Unknown reconstruction model: {model_id}") from exc
    if not model.available:
        raise RuntimeError(f"Reconstruction model {model.display_name} is not available: {model.unavailable_reason}")
    if model.backend == "wilor":
        assets = model.assets or {}
        root = Path(project_root)
        return OriginalWiLoRBackend(
            root / assets["repository"], root / assets["checkpoint"], root / assets["config"], device=device
        )
    raise RuntimeError(f"No runtime factory is registered for reconstruction backend: {model.backend}")
