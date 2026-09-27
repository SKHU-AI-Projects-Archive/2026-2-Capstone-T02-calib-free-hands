"""Synthetic bimanual subjects with controlled, SUBJECT-LEVEL stress.

Anatomical asymmetry is a property of the subject: it is drawn once and held
fixed for the whole sequence. It is never resampled per frame.

Four asymmetry families, all applied to bone LENGTHS before normalisation:
  global   - the whole right hand scaled; should be absorbed by normalisation
  dense    - every corresponding bone perturbed independently
  sparse   - only 4 deterministic bones perturbed
  finger   - each finger chain scaled differently between the sides
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c92_common import N_FRAMES  # noqa: E402  (puts CAM-009.1/src on the path)
from common import assemble  # noqa: E402  (CAM-EXP-009.1 common, unchanged)

W, H = 1280, 720
F_TRUE = 900.0
DIST_REAL = np.array([-0.39218, 0.13298, 0.0, 0.0])
N_BONES = 20
SPARSE_BONES = [2, 7, 10, 17]          # deterministic, fixed in the spec

BASE_LENGTHS = np.array([
    0.040, 0.035, 0.030, 0.025,
    0.085, 0.040, 0.025, 0.020,
    0.080, 0.045, 0.028, 0.021,
    0.075, 0.040, 0.026, 0.020,
    0.070, 0.032, 0.020, 0.017], float)

BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
         (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def random_dirs(rng, spread=1.0, artic_deg=0.0):
    base = np.zeros((N_BONES, 3))
    for i in range(N_BONES):
        finger = i // 4
        ang = -0.6 + 0.3 * finger
        base[i] = ([0.35 * (finger - 2), -0.9, 0.0] if i % 4 == 0
                   else [np.sin(ang), -np.cos(ang), 0.0])
    d = base + spread * rng.normal(scale=0.35, size=(N_BONES, 3))
    n = np.linalg.norm(d, axis=1, keepdims=True)
    d = d / np.where(n < 1e-9, 1.0, n)
    if artic_deg > 0:
        # small random rotation of each bone direction; lengths untouched
        sig = np.deg2rad(artic_deg)
        pert = rng.normal(scale=sig, size=(N_BONES, 3))
        d = d + np.cross(pert, d)
        n = np.linalg.norm(d, axis=1, keepdims=True)
        d = d / np.where(n < 1e-9, 1.0, n)
    return d


def make_subject(rng, asym_kind="dense", asym=0.0):
    """LEFT and RIGHT bone lengths. Asymmetry is fixed for the subject."""
    lL = np.clip(BASE_LENGTHS * (1.0 + rng.normal(scale=0.06, size=N_BONES)),
                 0.005, None)
    lR = lL.copy()
    if asym > 0:
        if asym_kind == "global":
            lR = lL * (1.0 + asym)
        elif asym_kind == "dense":
            lR = lL * np.exp(rng.normal(scale=asym, size=N_BONES))
        elif asym_kind == "sparse":
            f = np.ones(N_BONES)
            f[SPARSE_BONES] = np.exp(
                rng.normal(scale=asym, size=len(SPARSE_BONES)))
            lR = lL * f
        elif asym_kind == "finger":
            f = np.ones(N_BONES)
            for k in range(5):
                f[4 * k:4 * k + 4] = np.exp(rng.normal(scale=asym))
            lR = lL * f
        else:
            raise ValueError(asym_kind)
        lR = np.clip(lR, 0.005, None)
    return lL, lR


def project(Xc, K, dist, rng=None, noise_px=0.0):
    import cv2
    if np.any(Xc[:, 2] <= 1e-6):
        return None
    d = None if dist is None else np.asarray(dist, float).reshape(1, -1)
    uv, _ = cv2.projectPoints(
        np.ascontiguousarray(Xc.reshape(-1, 1, 3)), np.zeros(3), np.zeros(3),
        K, d)
    uv = uv.reshape(-1, 2)
    if not np.isfinite(uv).all():
        return None
    if noise_px > 0 and rng is not None:
        uv = uv + rng.normal(scale=noise_px, size=uv.shape)
    return uv


def make_sequence(rng, n_frames=N_FRAMES, dist_over_diam=4.0, asym_kind="dense",
                  asym=0.0, noise_px=0.0, artic_deg=0.0, missing=0.0,
                  visibility=1.0, pose_spread=1.0, dist=None, f_true=F_TRUE):
    dist = DIST_REAL if dist is None else dist
    K = np.array([[f_true, 0, W / 2.0], [0, f_true, H / 2.0], [0, 0, 1.0]])
    lL, lR = make_subject(rng, asym_kind, asym)
    out = {"left": [], "right": [], "lL_true": lL, "lR_true": lR,
           "f_true": f_true, "K": K, "dist": dist}
    tries = 0
    while (len(out["left"]) < n_frames or len(out["right"]) < n_frames) \
            and tries < n_frames * 30:
        tries += 1
        for side, lvec in (("left", lL), ("right", lR)):
            if len(out[side]) >= n_frames:
                continue
            if rng.random() > visibility:
                continue
            dirs_true = random_dirs(rng, pose_spread, 0.0)
            X = assemble(dirs_true, lvec)
            diam = float(np.linalg.norm(
                X[:, None, :] - X[None, :, :], axis=-1).max())
            R = rotation(rng)
            Xr, dirs_r = X @ R.T, dirs_true @ R.T
            Z = dist_over_diam * diam
            Xc = Xr + np.array([rng.uniform(-.15, .15) * Z,
                                rng.uniform(-.10, .10) * Z, Z])
            uv = project(Xc, K, dist, rng, noise_px)
            if uv is None:
                continue
            if (uv[:, 0].min() < -W or uv[:, 0].max() > 2 * W
                    or uv[:, 1].min() < -H or uv[:, 1].max() > 2 * H):
                continue
            # the articulation the SOLVER sees may be noisy; the image is not
            dirs_obs = dirs_r
            if artic_deg > 0:
                sig = np.deg2rad(artic_deg)
                pert = rng.normal(scale=sig, size=(N_BONES, 3))
                dirs_obs = dirs_r + np.cross(pert, dirs_r)
                n = np.linalg.norm(dirs_obs, axis=1, keepdims=True)
                dirs_obs = dirs_obs / np.where(n < 1e-9, 1.0, n)
            use = np.ones(21, bool)
            if missing > 0:
                drop = rng.random(21) < missing
                drop[0] = False                 # keep the root
                use = ~drop
            if use.sum() < 8:
                continue
            out[side].append((dirs_obs, uv, use))
    return out
