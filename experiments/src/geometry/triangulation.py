"""Robust multi-view triangulation, shared across experiments.

The routines here reconstruct 3D points from 2D observations and camera
calibration only. They never consult any dataset-provided 3D, which is what
makes them usable as an independent check of a released 3D annotation.

Distortion handling is explicit and symmetric:

* observations are converted to *normalised, undistorted* coordinates before
  any geometry, so the linear triangulation sees an ideal pinhole camera;
* residuals are evaluated back in real pixel coordinates, with distortion
  re-applied, so a reported reprojection error is a true image-space error.

``Camera`` here is the project's ``experiments.src.geometry.Camera``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TriangulationResult:
    """Outcome for a single 3D point."""

    point: np.ndarray | None                  # (3,) world coordinates, or None
    inliers: np.ndarray                       # bool mask over the observations
    residuals_px: np.ndarray                  # per-observation reprojection error
    n_observations: int = 0
    n_inliers: int = 0
    max_ray_angle_deg: float = 0.0            # geometric conditioning
    status: str = "ok"                        # ok | insufficient_observations |
                                              # insufficient_inliers | degenerate

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.point is not None

    def median_residual(self) -> float:
        r = self.residuals_px[self.inliers]
        return float(np.median(r)) if r.size else float("nan")

    def p90_residual(self) -> float:
        r = self.residuals_px[self.inliers]
        return float(np.percentile(r, 90)) if r.size else float("nan")


def normalize_observation(cam, uv: np.ndarray) -> np.ndarray:
    """Pixel -> normalised undistorted camera coordinates.

    Uses the camera's own distortion coefficients, so this is the exact
    inverse of ``Camera.project`` up to the iterative solver's tolerance.
    """
    import cv2
    pts = np.asarray(uv, dtype=float).reshape(-1, 1, 2)
    dist = cam.dist.reshape(-1).astype(float)
    if dist.size < 4:
        dist = np.zeros(4)
    out = cv2.undistortPoints(pts, cam.K.astype(float), dist[:5])
    return out.reshape(-1, 2)


def projection_matrix(cam) -> np.ndarray:
    """[R | t] for a camera whose observations are already normalised."""
    P = np.zeros((3, 4))
    P[:, :3] = cam.R
    P[:, 3] = cam.t
    return P


def triangulate_dlt(P_list, xn_list) -> np.ndarray | None:
    """Linear (DLT) triangulation from normalised observations.

    Each observation contributes two rows to a homogeneous system; the
    solution is the right singular vector of the smallest singular value.
    """
    rows = []
    for P, xn in zip(P_list, xn_list):
        x, y = float(xn[0]), float(xn[1])
        rows.append(x * P[2] - P[0])
        rows.append(y * P[2] - P[1])
    if len(rows) < 4:
        return None
    A = np.asarray(rows, dtype=float)
    if not np.isfinite(A).all():
        return None
    try:
        _, _, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    return X[:3] / X[3]


def reprojection_residuals(cameras, uv_obs, X) -> np.ndarray:
    """Pixel-space residuals, with distortion applied (NaN if behind camera)."""
    res = np.full(len(cameras), np.nan)
    for i, cam in enumerate(cameras):
        p, z = cam.project(np.asarray(X, float).reshape(1, 3), apply_distortion=True)
        if not np.isfinite(p).all() or z[0] <= 0:
            continue
        res[i] = float(np.linalg.norm(p[0] - np.asarray(uv_obs[i], float)))
    return res


def ray_directions(cameras) -> np.ndarray:
    """Unit viewing directions in world space (camera optical axes)."""
    return np.array([cam.R.T @ np.array([0.0, 0.0, 1.0]) for cam in cameras])


def max_ray_angle(cameras, X) -> float:
    """Largest angle (degrees) between rays from the cameras to the point.

    A small value means all cameras look from nearly the same direction, where
    depth is poorly constrained however small the reprojection error is.
    """
    X = np.asarray(X, float).reshape(3)
    dirs = []
    for cam in cameras:
        c = -cam.R.T @ cam.t                      # camera centre in world
        d = X - c
        n = np.linalg.norm(d)
        if n > 1e-12:
            dirs.append(d / n)
    if len(dirs) < 2:
        return 0.0
    D = np.asarray(dirs)
    G = np.clip(D @ D.T, -1.0, 1.0)
    return float(np.degrees(np.arccos(G.min())))


def robust_triangulate(cameras, uv_obs, threshold_px: float = 10.0,
                       min_inliers: int = 3, max_pairs: int = 60,
                       refine_iters: int = 2,
                       rng: np.random.Generator | None = None) -> TriangulationResult:
    """RANSAC-style triangulation tolerant of grossly wrong observations.

    A plain least-squares fit over all views is not usable here: a single view
    that annotated the wrong hand pulls the solution hundreds of millimetres.
    So 3D hypotheses are generated from camera pairs, scored by how many views
    they explain, and only the winning consensus set is used for the final fit.

    Selection depends solely on 2D observations and calibration; no external
    3D is consulted.
    """
    cameras = list(cameras)
    uv = np.asarray(uv_obs, dtype=float).reshape(len(cameras), 2)
    n = len(cameras)
    empty = np.zeros(n, dtype=bool)
    if n < 2:
        return TriangulationResult(None, empty, np.full(n, np.nan), n, 0, 0.0,
                                   "insufficient_observations")

    rng = rng or np.random.default_rng(0)
    P = [projection_matrix(c) for c in cameras]
    xn = [normalize_observation(c, uv[i])[0] for i, c in enumerate(cameras)]

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > max_pairs:
        sel = rng.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[k] for k in sel]

    best_mask, best_med, best_X = None, np.inf, None
    for i, j in pairs:
        X = triangulate_dlt([P[i], P[j]], [xn[i], xn[j]])
        if X is None:
            continue
        res = reprojection_residuals(cameras, uv, X)
        mask = np.isfinite(res) & (res <= threshold_px)
        if mask.sum() < 2:
            continue
        med = float(np.median(res[mask]))
        # prefer a larger consensus set; break ties on tighter residuals
        if (best_mask is None or mask.sum() > best_mask.sum()
                or (mask.sum() == best_mask.sum() and med < best_med)):
            best_mask, best_med, best_X = mask, med, X

    if best_mask is None:
        return TriangulationResult(None, empty, np.full(n, np.nan), n, 0, 0.0,
                                   "degenerate")

    # Refit on the consensus set, re-deciding membership each round.
    X = best_X
    mask = best_mask
    for _ in range(max(1, refine_iters)):
        idx = np.where(mask)[0]
        if idx.size < 2:
            break
        Xn = triangulate_dlt([P[k] for k in idx], [xn[k] for k in idx])
        if Xn is None:
            break
        X = Xn
        res = reprojection_residuals(cameras, uv, X)
        new_mask = np.isfinite(res) & (res <= threshold_px)
        if new_mask.sum() < 2:
            break
        if np.array_equal(new_mask, mask):
            mask = new_mask
            break
        mask = new_mask

    res = reprojection_residuals(cameras, uv, X)
    mask = np.isfinite(res) & (res <= threshold_px)
    n_in = int(mask.sum())
    angle = max_ray_angle([cameras[k] for k in np.where(mask)[0]], X) if n_in >= 2 else 0.0
    status = "ok" if n_in >= min_inliers else "insufficient_inliers"
    return TriangulationResult(np.asarray(X, float) if status == "ok" else None,
                               mask, res, n, n_in, angle, status)


def triangulate_hand(cameras, uv_per_joint, valid_per_joint=None,
                     threshold_px: float = 10.0, min_inliers: int = 3,
                     rng: np.random.Generator | None = None) -> list:
    """Triangulate each joint of a hand independently.

    ``uv_per_joint``   : (n_cameras, n_joints, 2)
    ``valid_per_joint``: (n_cameras, n_joints) bool, optional
    """
    uv = np.asarray(uv_per_joint, dtype=float)
    n_cams, n_joints = uv.shape[0], uv.shape[1]
    valid = (np.ones((n_cams, n_joints), bool) if valid_per_joint is None
             else np.asarray(valid_per_joint, bool))
    out = []
    for j in range(n_joints):
        keep = np.where(valid[:, j])[0]
        if keep.size < 2:
            out.append(TriangulationResult(None, np.zeros(n_cams, bool),
                                           np.full(n_cams, np.nan), int(keep.size), 0,
                                           0.0, "insufficient_observations"))
            continue
        r = robust_triangulate([cameras[k] for k in keep], uv[keep, j],
                               threshold_px=threshold_px, min_inliers=min_inliers,
                               rng=rng)
        # lift masks/residuals back to the full camera list
        full_mask = np.zeros(n_cams, bool)
        full_res = np.full(n_cams, np.nan)
        full_mask[keep] = r.inliers
        full_res[keep] = r.residuals_px
        out.append(TriangulationResult(r.point, full_mask, full_res, int(keep.size),
                                       r.n_inliers, r.max_ray_angle_deg, r.status))
    return out


class CameraBundle:
    """Stacked camera parameters for vectorised reprojection.

    ``robust_triangulate`` evaluates one candidate 3D point against every
    camera many times over, which dominates the runtime when done one camera at
    a time. This class pre-stacks intrinsics, extrinsics and distortion so a
    whole camera set is evaluated in a few array operations, with exactly the
    same maths as ``Camera.project``.
    """

    def __init__(self, cameras):
        self.cameras = list(cameras)
        n = len(self.cameras)
        self.R = np.stack([c.R for c in self.cameras]) if n else np.zeros((0, 3, 3))
        self.t = np.stack([c.t for c in self.cameras]) if n else np.zeros((0, 3))
        self.fx = np.array([c.K[0, 0] for c in self.cameras])
        self.fy = np.array([c.K[1, 1] for c in self.cameras])
        self.skew = np.array([c.K[0, 1] for c in self.cameras])
        self.cx = np.array([c.K[0, 2] for c in self.cameras])
        self.cy = np.array([c.K[1, 2] for c in self.cameras])
        d = np.zeros((n, 5))
        for i, c in enumerate(self.cameras):
            v = np.asarray(c.dist, float).reshape(-1)
            d[i, :min(5, v.size)] = v[:5]
        self.dist = d

    def project_point(self, X: np.ndarray) -> np.ndarray:
        """Project one world point into every camera. Returns (N, 2), NaN behind."""
        X = np.asarray(X, float).reshape(3)
        Xc = self.R @ X + self.t
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            xn = Xc[:, 0] / z
            yn = Xc[:, 1] / z
        k1, k2, p1, p2, k3 = (self.dist[:, i] for i in range(5))
        r2 = xn * xn + yn * yn
        radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
        yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
        u = self.fx * xd + self.skew * yd + self.cx
        v = self.fy * yd + self.cy
        uv = np.stack([u, v], axis=1)
        uv[z <= 0] = np.nan
        return uv

    def residuals(self, X: np.ndarray, uv_obs: np.ndarray) -> np.ndarray:
        p = self.project_point(X)
        return np.linalg.norm(p - np.asarray(uv_obs, float), axis=1)


def robust_triangulate_bundle(bundle, uv_obs, threshold_px: float = 10.0,
                              min_inliers: int = 3, max_pairs: int = 40,
                              refine_iters: int = 2, P_list=None, xn_list=None,
                              rng: np.random.Generator | None = None) -> TriangulationResult:
    """Same algorithm as ``robust_triangulate``, using a pre-stacked bundle."""
    cameras = bundle.cameras
    uv = np.asarray(uv_obs, dtype=float).reshape(len(cameras), 2)
    n = len(cameras)
    empty = np.zeros(n, dtype=bool)
    if n < 2:
        return TriangulationResult(None, empty, np.full(n, np.nan), n, 0, 0.0,
                                   "insufficient_observations")
    rng = rng or np.random.default_rng(0)
    P = P_list if P_list is not None else [projection_matrix(c) for c in cameras]
    xn = (xn_list if xn_list is not None
          else [normalize_observation(c, uv[i])[0] for i, c in enumerate(cameras)])

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > max_pairs:
        sel = rng.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[k] for k in sel]

    best_mask, best_med, best_X = None, np.inf, None
    for i, j in pairs:
        X = triangulate_dlt([P[i], P[j]], [xn[i], xn[j]])
        if X is None:
            continue
        res = bundle.residuals(X, uv)
        mask = np.isfinite(res) & (res <= threshold_px)
        k = int(mask.sum())
        if k < 2:
            continue
        med = float(np.median(res[mask]))
        if (best_mask is None or k > int(best_mask.sum())
                or (k == int(best_mask.sum()) and med < best_med)):
            best_mask, best_med, best_X = mask, med, X

    if best_mask is None:
        return TriangulationResult(None, empty, np.full(n, np.nan), n, 0, 0.0,
                                   "degenerate")
    X, mask = best_X, best_mask
    for _ in range(max(1, refine_iters)):
        idx = np.where(mask)[0]
        if idx.size < 2:
            break
        Xn = triangulate_dlt([P[k] for k in idx], [xn[k] for k in idx])
        if Xn is None:
            break
        X = Xn
        res = bundle.residuals(X, uv)
        new_mask = np.isfinite(res) & (res <= threshold_px)
        if int(new_mask.sum()) < 2 or np.array_equal(new_mask, mask):
            if int(new_mask.sum()) >= 2:
                mask = new_mask
            break
        mask = new_mask

    res = bundle.residuals(X, uv)
    mask = np.isfinite(res) & (res <= threshold_px)
    n_in = int(mask.sum())
    angle = max_ray_angle([cameras[k] for k in np.where(mask)[0]], X) if n_in >= 2 else 0.0
    status = "ok" if n_in >= min_inliers else "insufficient_inliers"
    return TriangulationResult(np.asarray(X, float) if status == "ok" else None,
                               mask, res, n, n_in, angle, status)


def triangulate_hand_fast(cameras, uv_per_joint, valid_per_joint=None,
                          threshold_px: float = 10.0, min_inliers: int = 3,
                          rng: np.random.Generator | None = None) -> list:
    """Vectorised equivalent of ``triangulate_hand`` for full-dataset passes."""
    uv = np.asarray(uv_per_joint, dtype=float)
    n_cams, n_joints = uv.shape[0], uv.shape[1]
    valid = (np.ones((n_cams, n_joints), bool) if valid_per_joint is None
             else np.asarray(valid_per_joint, bool))
    xn_all = np.full((n_cams, n_joints, 2), np.nan)
    for i, c in enumerate(cameras):
        xn_all[i] = normalize_observation(c, uv[i])
    P_all = [projection_matrix(c) for c in cameras]

    out = []
    for j in range(n_joints):
        keep = np.where(valid[:, j])[0]
        if keep.size < 2:
            out.append(TriangulationResult(None, np.zeros(n_cams, bool),
                                           np.full(n_cams, np.nan), int(keep.size),
                                           0, 0.0, "insufficient_observations"))
            continue
        bundle = CameraBundle([cameras[k] for k in keep])
        r = robust_triangulate_bundle(
            bundle, uv[keep, j], threshold_px=threshold_px, min_inliers=min_inliers,
            P_list=[P_all[k] for k in keep], xn_list=[xn_all[k, j] for k in keep],
            rng=rng)
        full_mask = np.zeros(n_cams, bool)
        full_res = np.full(n_cams, np.nan)
        full_mask[keep] = r.inliers
        full_res[keep] = r.residuals_px
        out.append(TriangulationResult(r.point, full_mask, full_res, int(keep.size),
                                       r.n_inliers, r.max_ray_angle_deg, r.status))
    return out
