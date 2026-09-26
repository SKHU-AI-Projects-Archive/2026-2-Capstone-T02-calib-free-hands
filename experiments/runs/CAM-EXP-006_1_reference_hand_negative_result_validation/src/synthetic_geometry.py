"""Deterministic synthetic scene generation for CAM-EXP-006.1.

Every trial is reproducible from the frozen seed. Nothing here reads a
GigaHands evaluation target.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import R006_CACHE, R006_FRAMES, SEED, read_csv  # noqa: E402

W, H = 1280, 720
F_TRUE = 900.0
DIST_RIG = np.array([-0.39218, 0.13298, 0.000538, -0.000224])
PP_OFFSET = (40.0, -24.0)


def rotation(rng):
    """A uniformly random rotation matrix via QR of a Gaussian matrix."""
    a = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(a)
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def diameter(pts):
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    return float(d.max())


def generic_cloud(rng, n=20):
    """A non-coplanar point cloud, redrawn if its third singular value is small."""
    for _ in range(200):
        p = rng.uniform(-0.5, 0.5, size=(n, 3))
        p -= p.mean(0)
        s = np.linalg.svd(p, compute_uv=False)
        if s[2] / s[0] > 0.25:        # clearly three-dimensional
            return p
    return p


def hand_shapes(limit=100):
    """Root-relative reference hands, sampled deterministically by stride.

    Selection uses the frozen CAM-006 manifest ORDER only. No focal error and
    no GT focal takes any part.
    """
    z = np.load(R006_CACHE)
    rows = read_csv(R006_FRAMES)
    keys = []
    for r in rows:
        for hand in ("left", "right"):
            keys.append(f"{r['sequence']}|{r['camera']}|{r['frame']}|{hand}")
    out = []
    stride = max(1, len(keys) // (limit * 3))
    for k in keys[::stride]:
        if k + "|use" not in z:
            continue
        use = z[k + "|use"]
        if use.sum() < 18:            # want well-populated hands for synthesis
            continue
        p = z[k + "|xyz"][use].astype(float)
        if not np.isfinite(p).all():
            continue
        p = p - p.mean(0)
        s = np.linalg.svd(p, compute_uv=False)
        if s[0] <= 0 or s[2] / s[0] < 0.02:
            continue
        out.append(p)
        if len(out) >= limit:
            break
    return out


def place(pts, rng, distance_over_diameter, depth_scale=1.0):
    """Rotate the shape and place it in front of the camera at a set distance.

    `depth_scale` shrinks the extent along the camera's viewing axis AFTER the
    rotation, so 0.0 gives an exactly fronto-parallel planar target.
    """
    R = rotation(rng)
    p = pts @ R.T
    if depth_scale != 1.0:
        p[:, 2] *= depth_scale
    d = diameter(pts)
    Z = distance_over_diameter * d
    t = np.array([0.0, 0.0, Z])
    return p + t


def project(pts_cam, f=F_TRUE, cx=None, cy=None, dist=None, fy_over_fx=1.0,
            noise_px=0.0, rng=None):
    """Project camera-frame points. Returns None if the trial is unsafe."""
    import cv2
    cx = W / 2.0 if cx is None else cx
    cy = H / 2.0 if cy is None else cy
    if np.any(pts_cam[:, 2] <= 1e-6):
        return None
    K = np.array([[f, 0.0, cx], [0.0, fy_over_fx * f, cy], [0.0, 0.0, 1.0]])
    d = None if dist is None else np.asarray(dist, float).reshape(1, -1)
    uv, _ = cv2.projectPoints(
        np.ascontiguousarray(pts_cam.reshape(-1, 1, 3)),
        np.zeros(3), np.zeros(3), K, d)
    uv = uv.reshape(-1, 2)
    if not np.isfinite(uv).all():
        return None
    margin = 2.0
    if (uv[:, 0].min() < -margin * W or uv[:, 0].max() > margin * W
            or uv[:, 1].min() < -margin * H or uv[:, 1].max() > margin * H):
        return None
    if noise_px > 0 and rng is not None:
        uv = uv + rng.normal(scale=noise_px, size=uv.shape)
    return uv


def make_trial(shape, rng, distance_over_diameter=2.5, depth_scale=1.0,
               noise_px=0.0, dist=None, cx=None, cy=None, fy_over_fx=1.0):
    """One synthetic trial: returns (points in the camera frame, 2D) or None.

    The solver is later given ONLY these two arrays plus the image size. It is
    never told f, R or T.
    """
    pc = place(shape, rng, distance_over_diameter, depth_scale)
    uv = project(pc, F_TRUE, cx, cy, dist, fy_over_fx, noise_px, rng)
    if uv is None:
        return None
    # express the shape in an arbitrary object frame, so the pose the solver
    # must recover is a general rigid transform rather than the identity
    Ro = rotation(rng)
    to = rng.uniform(-0.3, 0.3, size=3)
    obj = pc @ Ro.T + to
    return obj, uv
