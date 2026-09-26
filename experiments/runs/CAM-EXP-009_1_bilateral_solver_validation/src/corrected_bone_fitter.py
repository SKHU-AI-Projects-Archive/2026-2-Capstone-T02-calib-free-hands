"""Corrected bone fitter and projection, under UNDISTORT_ONCE_INTERNAL.

The two CAM-EXP-009 faults are fixed here:

  * the observations are undistorted exactly ONCE, with the CANDIDATE focal's
    own K, and everything inside the solver then runs with distCoeffs=None;
  * scoring is reported in BOTH spaces, each self-consistent - ideal (pinhole
    projection against undistorted pixels) and raw (cv2.projectPoints with the
    distortion against the raw pixels).

Nothing here ever receives the true focal, the true pose or the true bone
vector, except where a function is explicitly named an oracle diagnostic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MJB, N_BONES, N_ITER, assemble, dof_groups  # noqa: E402


def undistort_once(uv_raw, K_cand, dist):
    """Raw pixels -> ideal pinhole pixels under the CANDIDATE K.

    Recomputed for every candidate focal: reusing one undistortion made with
    the true K would leak the answer and is explicitly avoided.
    """
    import cv2
    if dist is None or not np.any(dist):
        return np.asarray(uv_raw, float).copy()
    d = np.asarray(dist, np.float64).reshape(1, -1)
    pts = np.ascontiguousarray(
        np.asarray(uv_raw, np.float64).reshape(-1, 1, 2))
    return cv2.undistortPoints(pts, K_cand, d, P=K_cand).reshape(-1, 2)


def pose_ideal(X, uv_ideal, use, K):
    """PnP in ideal space. distCoeffs is None by construction."""
    import cv2
    obj = np.ascontiguousarray(X[use].reshape(-1, 1, 3), np.float64)
    img = np.ascontiguousarray(uv_ideal[use].reshape(-1, 1, 2), np.float64)
    for flag in (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_IPPE,
                 cv2.SOLVEPNP_ITERATIVE):
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=flag)
        except cv2.error:
            ok = False
        if ok:
            return rvec, tvec
    return None, None


def reproj_ideal(X, rvec, tvec, K, uv_ideal, use):
    import cv2
    R, _ = cv2.Rodrigues(rvec)
    Xc = (R @ X.T).T + tvec.reshape(3)
    z = np.where(np.abs(Xc[:, 2]) < 1e-9, 1e-9, Xc[:, 2])
    proj = np.stack([K[0, 0] * Xc[:, 0] / z + K[0, 2],
                     K[1, 1] * Xc[:, 1] / z + K[1, 2]], 1)
    return float(np.median(np.linalg.norm(proj[use] - uv_ideal[use], axis=1)))


def reproj_raw(X, rvec, tvec, K, dist, uv_raw, use):
    """Raw-space score: full distortion model against the raw pixels."""
    import cv2
    d = None if dist is None else np.asarray(dist, np.float64).reshape(1, -1)
    proj, _ = cv2.projectPoints(
        np.ascontiguousarray(X.reshape(-1, 1, 3), np.float64),
        rvec, tvec, K, d)
    proj = proj.reshape(-1, 2)
    return float(np.median(np.linalg.norm(proj[use] - uv_raw[use], axis=1)))


def _expand(theta, groups):
    l = np.zeros(N_BONES)
    for g, val in zip(groups, theta):
        for b in g:
            l[b] = val
    return l


def fit_bones(frames, K, dist, dof="D20", l_init=None, n_iter=N_ITER):
    """frames: list of (dirs (20,3), uv_raw (21,2), use (21,) bool).

    Returns (l, ideal_px, raw_px). l sums to 1, so absolute hand size is never
    used as a cue.
    """
    groups = dof_groups(dof)
    prepped = []
    for dirs, uv_raw, use in frames:
        if use.sum() < 6:
            continue
        prepped.append((dirs, np.asarray(uv_raw, float),
                        undistort_once(uv_raw, K, dist), use))
    if len(prepped) < 2:
        return None, float("nan"), float("nan")

    l = (np.full(N_BONES, 1.0 / N_BONES) if l_init is None
         else np.asarray(l_init, float).copy())
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    for _ in range(n_iter):
        rows, rhs = [], []
        for dirs, _uvr, uvi, use in prepped:
            X = assemble(dirs, l)
            rvec, tvec = pose_ideal(X, uvi, use, K)
            if rvec is None:
                continue
            import cv2
            R, _ = cv2.Rodrigues(rvec)
            T = tvec.reshape(3)
            C = MJB[:, :, None] * dirs[None, :, :]          # (21,B,3)
            Cc = np.einsum("ij,kbj->kbi", R, C)
            # group the bone coefficients into the DOF parameterisation
            Cg = np.stack([Cc[:, g, :].sum(1) for g in groups], 1)
            for j in np.flatnonzero(use):
                u, v = uvi[j]
                rows.append((u - cx) * Cg[j, :, 2] - fx * Cg[j, :, 0])
                rhs.append(fx * T[0] - (u - cx) * T[2])
                rows.append((v - cy) * Cg[j, :, 2] - fy * Cg[j, :, 1])
                rhs.append(fy * T[1] - (v - cy) * T[2])
        if not rows:
            return None, float("nan"), float("nan")
        A = np.asarray(rows, float)
        b = np.asarray(rhs, float)
        # sum(l) = 1 as a strongly weighted equality row
        w = float(np.linalg.norm(A)) / max(len(groups), 1) * 10.0
        A = np.vstack([A, np.array([[w * len(g) for g in groups]])])
        b = np.concatenate([b, [w]])
        try:
            theta, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None, float("nan"), float("nan")
        theta = np.clip(theta, 1e-5, None)
        new = _expand(theta, groups)
        s = new.sum()
        if not np.isfinite(s) or s <= 0:
            return None, float("nan"), float("nan")
        new = new / s
        if np.allclose(new, l, atol=1e-7):
            l = new
            break
        l = new

    ideal, raw = [], []
    for dirs, uvr, uvi, use in prepped:
        X = assemble(dirs, l)
        rvec, tvec = pose_ideal(X, uvi, use, K)
        if rvec is None:
            continue
        ideal.append(reproj_ideal(X, rvec, tvec, K, uvi, use))
        raw.append(reproj_raw(X, rvec, tvec, K, dist, uvr, use))
    return (l,
            float(np.median(ideal)) if ideal else float("nan"),
            float(np.median(raw)) if raw else float("nan"))


def score_fixed_shape(frames, K, dist, l):
    """Reprojection of a GIVEN shape on GIVEN frames. Used for held-out
    evaluation and for the true-shape oracle profile."""
    ideal, raw = [], []
    for dirs, uv_raw, use in frames:
        if use.sum() < 6:
            continue
        uvi = undistort_once(uv_raw, K, dist)
        X = assemble(dirs, l)
        rvec, tvec = pose_ideal(X, uvi, use, K)
        if rvec is None:
            continue
        ideal.append(reproj_ideal(X, rvec, tvec, K, uvi, use))
        raw.append(reproj_raw(X, rvec, tvec, K, dist,
                              np.asarray(uv_raw, float), use))
    return (float(np.median(ideal)) if ideal else float("nan"),
            float(np.median(raw)) if raw else float("nan"))


def bilateral_distance(lL, lR):
    if lL is None or lR is None:
        return float("nan")
    a, b = np.asarray(lL, float), np.asarray(lR, float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(
            b).all():
        return float("nan")
    return float(np.mean(np.abs(a - b)))


def multistart_inits(rng, n=8):
    """Deterministic positive perturbations around uniform. No GT is used."""
    out = [np.full(N_BONES, 1.0 / N_BONES)]
    for _ in range(n - 1):
        v = np.clip(1.0 / N_BONES * (1.0 + rng.uniform(-0.6, 0.6, N_BONES)),
                    1e-4, None)
        out.append(v / v.sum())
    return out
