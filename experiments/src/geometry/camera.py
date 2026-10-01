"""Pinhole camera model shared by every dataset loader.

Every dataset in this project ends up expressed in the same canonical form:

    X_cam = R @ X_world + t        (R: 3x3, t: 3,)
    x_norm = X_cam[:2] / X_cam[2]
    (u, v) = K @ distort(x_norm)

Datasets differ only in how R/t/K/dist are *stored*; the loaders are
responsible for converting into this convention so that downstream
experiments never have to special-case a dataset again.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def quat_to_rotmat(q) -> np.ndarray:
    """COLMAP-style quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = np.asarray(q, dtype=float)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


@dataclass
class Camera:
    """A calibrated pinhole camera with optional OpenCV distortion."""

    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))
    width: int | None = None
    height: int | None = None
    name: str = ""

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=float).reshape(3, 3)
        self.R = np.asarray(self.R, dtype=float).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=float).reshape(3)
        self.dist = np.asarray(self.dist, dtype=float).reshape(-1)

    @property
    def has_distortion(self) -> bool:
        return bool(self.dist.size) and bool(np.any(self.dist != 0))

    @classmethod
    def from_world_to_cam(cls, K, R, t, **kw) -> "Camera":
        """R, t already map world -> camera (the canonical form)."""
        return cls(K=K, R=R, t=t, **kw)

    @classmethod
    def from_camera_pose(cls, K, camrot, campos, **kw) -> "Camera":
        """InterHand2.6M style: X_cam = camrot @ (X_world - campos)."""
        R = np.asarray(camrot, dtype=float).reshape(3, 3)
        t = -R @ np.asarray(campos, dtype=float).reshape(3)
        return cls(K=K, R=R, t=t, **kw)

    def to_camera(self, X_world: np.ndarray) -> np.ndarray:
        X = np.asarray(X_world, dtype=float).reshape(-1, 3)
        return (self.R @ X.T).T + self.t

    def project(self, X_world: np.ndarray, apply_distortion: bool = True):
        """Project world points. Returns (uv Nx2, depth N,).

        Points with depth <= 0 are returned as NaN so callers can treat them
        as invalid rather than silently producing a mirrored projection.
        """
        Xc = self.to_camera(X_world)
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            xn = Xc[:, 0] / z
            yn = Xc[:, 1] / z
        if apply_distortion and self.has_distortion:
            xn, yn = self._distort(xn, yn)
        u = self.K[0, 0] * xn + self.K[0, 1] * yn + self.K[0, 2]
        v = self.K[1, 1] * yn + self.K[1, 2]
        uv = np.stack([u, v], axis=1)
        uv[z <= 0] = np.nan
        return uv, z

    def _distort(self, xn, yn):
        """OpenCV radial-tangential model; dist = [k1, k2, p1, p2, (k3)]."""
        d = np.zeros(5)
        d[: min(5, self.dist.size)] = self.dist[:5]
        k1, k2, p1, p2, k3 = d
        r2 = xn * xn + yn * yn
        radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        x = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
        y = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
        return x, y

    def in_image(self, uv: np.ndarray) -> np.ndarray:
        if self.width is None or self.height is None:
            return np.ones(len(uv), dtype=bool)
        return (
            (uv[:, 0] >= 0) & (uv[:, 0] < self.width)
            & (uv[:, 1] >= 0) & (uv[:, 1] < self.height)
        )
