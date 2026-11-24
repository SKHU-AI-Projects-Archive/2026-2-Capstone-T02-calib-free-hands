from __future__ import annotations

import numpy as np


def sample_indices(frame_count: int, mode: str = "uniform", count: int = 8) -> np.ndarray:
    if frame_count < 0 or count < 1:
        raise ValueError("frame_count must be non-negative and count must be positive")
    if frame_count == 0:
        return np.empty(0, dtype=np.int64)
    if mode == "all":
        return np.arange(frame_count, dtype=np.int64)
    if mode != "uniform":
        raise ValueError(f"unsupported sampling mode: {mode}")
    return np.unique(np.linspace(0, frame_count - 1, min(count, frame_count), dtype=np.int64))


def aggregate_focal(values: list[float] | np.ndarray, method: str = "median") -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("no finite focal estimates")
    if method == "median":
        return float(np.median(finite))
    if method == "mean":
        return float(np.mean(finite))
    raise ValueError(f"unsupported aggregation method: {method}")
