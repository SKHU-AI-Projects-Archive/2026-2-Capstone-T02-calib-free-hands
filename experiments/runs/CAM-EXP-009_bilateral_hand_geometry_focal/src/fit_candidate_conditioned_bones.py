"""Estimate sequence-shared bone proportions AT a candidate focal.

For one side of one sequence, given per-frame bone directions and per-frame 2D
observations, and given a candidate camera K(f):

    find one shared bone-proportion vector l  (positive, sum = 1)
    and free per-frame poses R_t, T_t
    minimising the reprojection error over the sequence.

Why this is the whole point. The hand is LINEAR in l once the bone directions
are fixed:  X_j = sum_b M[j,b] * l_b * dir_b.  So with the pose held, the
collinearity equations

    (u - cx) * (R_3 . X + T_z) = fx * (R_1 . X + T_x)
    (v - cy) * (R_3 . X + T_z) = fy * (R_2 . X + T_y)

are linear in l. Alternating [PnP for pose] and [linear solve for l] therefore
converges quickly, and because fx enters those equations explicitly, the fitted
l DEPENDS ON THE CANDIDATE FOCAL. That dependence is what makes a bilateral
comparison of l_L(f) against l_R(f) a focal cue at all.

No reference bone-length vector is used as an initial value or as a prior
centre: the fit starts from a uniform vector.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MJB, N_BONES, assemble  # noqa: E402


def _undistort(uv, K, dist):
    """Map observed pixels to ideal pinhole pixels once, given fixed K."""
    import cv2
    if dist is None or not np.any(dist):
        return uv
    d = np.asarray(dist, np.float64).reshape(1, -1)
    pts = np.ascontiguousarray(uv.reshape(-1, 1, 2), np.float64)
    out = cv2.undistortPoints(pts, K, d, P=K)
    return out.reshape(-1, 2)


def _pose_from(l, dirs, uv, use, K, dist):
    import cv2
    X = assemble(dirs, l)
    obj = np.ascontiguousarray(X[use].reshape(-1, 1, 3), np.float64)
    img = np.ascontiguousarray(uv[use].reshape(-1, 1, 2), np.float64)
    d = None if dist is None else np.asarray(dist, np.float64).reshape(1, -1)
    for flag in (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_IPPE,
                 cv2.SOLVEPNP_ITERATIVE):
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, d, flags=flag)
        except cv2.error:
            ok = False
        if ok:
            R, _ = cv2.Rodrigues(rvec)
            return R, tvec.reshape(3)
    return None, None


def fit_bones(frames, K, dist, n_iter=6, l0=None):
    """frames: list of (dirs (B,3), uv (21,2), use (21,) bool).

    Returns (l, mean_reprojection_px) or (None, nan).
    """
    if not frames:
        return None, float("nan")
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    # undistort once per candidate K, so the linear system below is exact
    prepped = []
    for dirs, uv, use in frames:
        if use.sum() < 6:
            continue
        prepped.append((dirs, _undistort(uv, K, dist), use))
    if len(prepped) < 2:
        return None, float("nan")

    l = np.full(N_BONES, 1.0 / N_BONES) if l0 is None else np.asarray(l0,
                                                                     float)
    last = float("nan")
    for _ in range(n_iter):
        rows, rhs, errs = [], [], []
        any_pose = False
        for dirs, uv, use in prepped:
            R, T = _pose_from(l, dirs, uv, use, K, dist)
            if R is None:
                continue
            any_pose = True
            # coefficient of l in X_j is  M[j,b] * dir_b   -> (21,B,3)
            C = MJB[:, :, None] * dirs[None, :, :]
            # camera-frame coefficient: R @ coeff
            Cc = np.einsum("ij,kbj->kbi", R, C)          # (21,B,3)
            for j in np.flatnonzero(use):
                u, v = uv[j]
                # (u-cx)*(Cc_z . l + Tz) = fx*(Cc_x . l + Tx)
                rows.append((u - cx) * Cc[j, :, 2] - fx * Cc[j, :, 0])
                rhs.append(fx * T[0] - (u - cx) * T[2])
                rows.append((v - cy) * Cc[j, :, 2] - fy * Cc[j, :, 1])
                rhs.append(fy * T[1] - (v - cy) * T[2])
        if not any_pose or not rows:
            return None, float("nan")
        A = np.asarray(rows, float)
        b = np.asarray(rhs, float)
        # constrain sum(l) = 1 by appending a strongly weighted row, and keep
        # the solution positive by clipping then renormalising
        w = float(np.linalg.norm(A)) / max(N_BONES, 1) * 10.0
        A = np.vstack([A, np.full((1, N_BONES), w)])
        b = np.concatenate([b, [w]])
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None, float("nan")
        sol = np.clip(sol, 1e-4, None)
        s = sol.sum()
        if not np.isfinite(s) or s <= 0:
            return None, float("nan")
        new = sol / s
        if np.allclose(new, l, atol=1e-6):
            l = new
            break
        l = new

    # final reprojection error with the fitted l
    errs = []
    for dirs, uv, use in prepped:
        R, T = _pose_from(l, dirs, uv, use, K, dist)
        if R is None:
            continue
        X = assemble(dirs, l)
        Xc = (R @ X.T).T + T
        z = np.where(np.abs(Xc[:, 2]) < 1e-9, 1e-9, Xc[:, 2])
        proj = np.stack([fx * Xc[:, 0] / z + cx, fy * Xc[:, 1] / z + cy], 1)
        errs.append(np.median(np.linalg.norm(proj[use] - uv[use], axis=1)))
    return l, (float(np.median(errs)) if errs else float("nan"))


def bilateral_distance(lL, lR, correspondence=None):
    """Robust distance between corresponding, scale-normalised bone vectors.

    Both vectors already sum to 1, so overall hand size is removed and a person
    with larger hands is not penalised. What remains is shape proportion.
    """
    if lL is None or lR is None:
        return float("nan")
    a = np.asarray(lL, float)
    b = np.asarray(lR, float)
    if correspondence is not None:
        idx = np.asarray([j for _, j in correspondence], int)
        b = b[idx]
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(
            b).all():
        return float("nan")
    return float(np.mean(np.abs(a - b)))        # L1, robust to one odd bone
