from __future__ import annotations

import numpy as np

from calibfreehands.calibration.anycalib_backend import AnyCalibBackend
from calibfreehands.calibration.sampling import aggregate_focal


class HandCalib:
    name = "HandCalib · benchmark Calibrator (circle guide, 8f70b58)"

    def __init__(self, device: str):
        if device == "cuda":
            import torch
            import onnxruntime

            if not torch.cuda.is_available() or "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
                raise RuntimeError("HandCalib CUDA needs ONNX Runtime GPU and a working CUDA installation")
            onnxruntime.preload_dlls()
        from .benchmark_calibrator import Calibrator

        self.calibrator = Calibrator(device=device)

    def add(self, rgb: np.ndarray) -> None:
        self.calibrator.add(np.ascontiguousarray(rgb[:, :, ::-1]))

    def estimate(self) -> float:
        if len(self.calibrator.observations) < 6:
            raise ValueError(f"HandCalib detected only {len(self.calibrator.observations)} usable right-hand frames")
        return float(self.calibrator.estimate())

    def details(self) -> dict:
        return {"usable_hand_frames": len(self.calibrator.observations), "source_commit": "8f70b58"}


class AnyCalibIntegration:
    name = "AnyCalib 통합 검사 · anycalib_pinhole"

    def __init__(self, device: str):
        self.model = AnyCalibBackend(device=device)
        self.focals = []

    def add(self, rgb: np.ndarray) -> None:
        self.focals.append(self.model.estimate_focal(rgb))

    def estimate(self) -> float:
        return aggregate_focal([f for f in self.focals if f > 0], "median")

    def details(self) -> dict:
        return {"aggregation": "median", "model_id": self.model.model_id}
