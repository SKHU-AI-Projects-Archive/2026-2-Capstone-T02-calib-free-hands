from __future__ import annotations

from pathlib import Path

import numpy as np

from .focal import extract_pinhole_focal


class AnyCalibBackend:
    """Lazy adapter for the repository's original AnyCalib package."""

    name = "AnyCalib"

    def __init__(self, model_id: str = "anycalib_pinhole", device: str = "cuda") -> None:
        self.model_id, self.device = model_id, device
        self._model = None

    def load(self) -> None:
        try:
            import torch
            from anycalib import AnyCalib
        except ImportError as exc:
            raise RuntimeError("Original AnyCalib is not installed in the active environment.") from exc
        self._model = AnyCalib(model_id=self.model_id).to(torch.device(self.device))
        self._model.eval()

    def estimate_focal(self, image_rgb: np.ndarray) -> float:
        if self._model is None:
            self.load()
        import torch
        image = torch.as_tensor(np.array(image_rgb, copy=True), dtype=torch.float32, device=self.device).permute(2, 0, 1) / 255.0
        with torch.no_grad():
            output = self._model.predict(image, cam_id="pinhole")
        intrinsics = output["intrinsics"].detach().cpu().numpy()
        return extract_pinhole_focal(intrinsics)
