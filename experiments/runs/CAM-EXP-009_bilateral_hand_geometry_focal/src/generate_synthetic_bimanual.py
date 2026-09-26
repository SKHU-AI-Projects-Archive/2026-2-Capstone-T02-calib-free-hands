"""Deterministic synthetic bimanual sequences with a known focal.

The solver is never told f_true, R, T or the bone vectors. Only the 2D
observations, the per-frame bone directions and the image size are handed over.

A synthetic subject has a LEFT and a RIGHT bone-length vector that are related
but not identical: `asymmetry` is the fractional per-bone deviation between
corresponding bones, because real people are not perfectly symmetric.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BONES, N_BONES, assemble  # noqa: E402

W, H = 1280, 720
F_TRUE = 900.0
DIST_RIG = np.array([-0.39218, 0.13298, 0.0, 0.0])

# a plausible generic hand: metacarpals longer than distal phalanges
BASE_LENGTHS = np.array([
    0.040, 0.035, 0.030, 0.025,          # thumb chain
    0.085, 0.040, 0.025, 0.020,          # index
    0.080, 0.045, 0.028, 0.021,          # middle
    0.075, 0.040, 0.026, 0.020,          # ring
    0.070, 0.032, 0.020, 0.017,          # little
], float)


def rotation(rng):
    a = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(a)
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def random_dirs(rng, spread=1.0):
    """Per-frame bone directions: a rest pose plus articulation jitter."""
    base = np.zeros((N_BONES, 3))
    for i, (p, c) in enumerate(BONES):
        finger = i // 4
        ang = -0.6 + 0.3 * finger
        base[i] = [np.sin(ang), -np.cos(ang), 0.0]
        if i % 4 == 0:
            base[i] = [0.35 * (finger - 2), -0.9, 0.0]
    d = base + spread * rng.normal(scale=0.35, size=(N_BONES, 3))
    n = np.linalg.norm(d, axis=1, keepdims=True)
    n = np.where(n < 1e-9, 1.0, n)
    return d / n


def make_subject(rng, asymmetry=0.0):
    lL = BASE_LENGTHS * (1.0 + rng.normal(scale=0.06, size=N_BONES))
    lL = np.clip(lL, 0.005, None)
    if asymmetry <= 0:
        lR = lL.copy()
    else:
        lR = lL * (1.0 + rng.normal(scale=asymmetry, size=N_BONES))
        lR = np.clip(lR, 0.005, None)
    return lL, lR


def project(X, K, dist, rng=None, noise_px=0.0):
    import cv2
    if np.any(X[:, 2] <= 1e-6):
        return None
    d = None if dist is None else np.asarray(dist, float).reshape(1, -1)
    uv, _ = cv2.projectPoints(np.ascontiguousarray(X.reshape(-1, 1, 3)),
                              np.zeros(3), np.zeros(3), K, d)
    uv = uv.reshape(-1, 2)
    if not np.isfinite(uv).all():
        return None
    if noise_px > 0 and rng is not None:
        uv = uv + rng.normal(scale=noise_px, size=uv.shape)
    return uv


def make_sequence(rng, n_frames=14, asymmetry=0.0, noise_px=0.0,
                  dist_over_diam=4.0, pose_spread=1.0, f_true=F_TRUE,
                  dist=None, both_visible_rate=1.0):
    """One synthetic bimanual sequence. Returns dirs + 2D per side."""
    dist = DIST_RIG if dist is None else dist
    K = np.array([[f_true, 0, W / 2.0], [0, f_true, H / 2.0], [0, 0, 1.0]])
    lL, lR = make_subject(rng, asymmetry)

    out = {"left": [], "right": [], "lL_true": lL, "lR_true": lR,
           "f_true": f_true, "K": K, "dist": dist}
    tries = 0
    while (len(out["left"]) < n_frames or len(out["right"]) < n_frames) \
            and tries < n_frames * 20:
        tries += 1
        for side, lvec in (("left", lL), ("right", lR)):
            if len(out[side]) >= n_frames:
                continue
            if side == "right" and rng.random() > both_visible_rate:
                continue
            dirs = random_dirs(rng, pose_spread)
            X = assemble(dirs, lvec)
            diam = float(np.linalg.norm(
                X[:, None, :] - X[None, :, :], axis=-1).max())
            R = rotation(rng)
            Xr = X @ R.T
            dirs_r = dirs @ R.T
            Z = dist_over_diam * diam
            off = np.array([rng.uniform(-0.15, 0.15) * Z,
                            rng.uniform(-0.10, 0.10) * Z, Z])
            Xc = Xr + off
            uv = project(Xc, K, dist, rng, noise_px)
            if uv is None:
                continue
            if (uv[:, 0].min() < -W or uv[:, 0].max() > 2 * W
                    or uv[:, 1].min() < -H or uv[:, 1].max() > 2 * H):
                continue
            use = np.ones(21, bool)
            out[side].append((dirs_r, uv, use))
    return out
