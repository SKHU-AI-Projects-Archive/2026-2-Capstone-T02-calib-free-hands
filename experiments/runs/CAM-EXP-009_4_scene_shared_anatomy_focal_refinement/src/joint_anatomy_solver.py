"""Sequence-shared anatomy solver under the CAM-EXP-009.1 conventions.

For one candidate focal f and one sequence-camera unit:

    p_seq = softmax(log(p0 + eps) + a)
    p_L   = softmax(log(p_seq + eps) + delta_L)
    p_R   = softmax(log(p_seq + eps) + delta_R)

`a`, `delta_L`, `delta_R` are fitted on the FIT frames of THIS unit only. Per
frame the pose (R, T) is a free nuisance solved by PnP. The anatomy is shared
across every frame of the unit and across both hands through `p_seq`.

Conventions inherited unchanged from CAM-EXP-009.1:
  * UNDISTORT_ONCE_INTERNAL - undistort once with the CANDIDATE K, then solve
    with distCoeffs=None. The distortion is never applied twice, and a pinhole
    prediction is never scored against raw distorted pixels.
  * 32 alternating iterations.
  * deterministic initialisation.

Gauge: softmax is shift-invariant, so `a` and each `delta` are mean-centred
after every update to remove the redundant direction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (EPS, MIN_OK_JOINTS, N_BONES, N_ITER, assemble,  # noqa: E402
                    softmax_from_log)


def undistort_once(uv_raw, K, dist):
    """Raw pixels -> ideal pinhole pixels under the CANDIDATE K."""
    import cv2
    if dist is None or not np.any(dist):
        return np.asarray(uv_raw, float).copy()
    d = np.asarray(dist, np.float64).reshape(1, -1)
    pts = np.ascontiguousarray(np.asarray(uv_raw, np.float64).reshape(-1, 1, 2))
    return cv2.undistortPoints(pts, K, d, P=K).reshape(-1, 2)


def pose_ideal(X, uv_ideal, use, K):
    import cv2
    idx = np.flatnonzero(use)
    if idx.size < 6:
        return None, None
    obj = np.ascontiguousarray(X[idx].reshape(-1, 1, 3))
    img = np.ascontiguousarray(uv_ideal[idx].reshape(-1, 1, 2))
    for flag in (getattr(cv2, "SOLVEPNP_SQPNP", None), cv2.SOLVEPNP_ITERATIVE):
        if flag is None:
            continue
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=flag)
        except cv2.error:
            continue
        if ok and np.isfinite(rvec).all() and np.isfinite(tvec).all():
            return rvec, tvec
    return None, None


def reproj_ideal(X, rvec, tvec, K, uv_ideal, use):
    import cv2
    proj, _ = cv2.projectPoints(np.ascontiguousarray(X.reshape(-1, 1, 3)),
                                rvec, tvec, K, None)
    proj = proj.reshape(-1, 2)
    idx = np.flatnonzero(use)
    return float(np.median(np.linalg.norm(proj[idx] - uv_ideal[idx], axis=1)))


class UnitFrames:
    """Prepared observations for one side of one unit at one candidate K."""

    def __init__(self, frames, K, dist):
        self.items = []
        for dirs, uv_raw, use in frames:
            if use.sum() < MIN_OK_JOINTS:
                continue
            self.items.append((dirs, np.asarray(uv_raw, float),
                               undistort_once(uv_raw, K, dist), use))

    def __len__(self):
        return len(self.items)


def _centre(v):
    return v - v.mean()


def _linear_rows(prepped, p, K, jac_scale):
    """Least-squares rows for a log-space anatomy update.

    The skeleton is linear in the bone LENGTHS once the directions and the pose
    are fixed. The parameters here are log-space offsets, so the lengths are
    linearised about the current proportions: dl_b ~= p_b * dtheta_b.
    """
    import cv2
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rows, rhs = [], []
    from common import BIDX, ORDER, PARENT
    for dirs, _uvr, uvi, use in prepped:
        X = assemble(dirs, p)
        rvec, tvec = pose_ideal(X, uvi, use, K)
        if rvec is None:
            continue
        R, _ = cv2.Rodrigues(rvec)
        T = tvec.reshape(3)
        # dX_j / dtheta_b  =  p_b * dir_b  for every bone b on the chain to j
        C = np.zeros((21, N_BONES, 3))
        for j in ORDER:
            par = PARENT[j]
            i = BIDX[(par, j)]
            C[j] = C[par]
            C[j, i] += p[i] * dirs[i]
        Cc = np.einsum("ij,kbj->kbi", R, C)
        Xc = (R @ X.T).T + T
        for j in np.flatnonzero(use):
            u, v = uvi[j]
            rows.append((u - cx) * Cc[j, :, 2] - fx * Cc[j, :, 0])
            rhs.append(fx * Xc[j, 0] - (u - cx) * Xc[j, 2])
            rows.append((v - cy) * Cc[j, :, 2] - fy * Cc[j, :, 1])
            rhs.append(fy * Xc[j, 1] - (v - cy) * Xc[j, 2])
    if not rows:
        return None, None
    return np.asarray(rows, float) * jac_scale, np.asarray(rhs, float)


def fit_unit(frames_L, frames_R, K, dist, p0, lam_generic, lam_side,
             shared=True, use_generic=True, n_iter=N_ITER):
    """Fit (a, delta_L, delta_R) on FIT frames. Returns a dict or None."""
    PL = UnitFrames(frames_L, K, dist)
    PR = UnitFrames(frames_R, K, dist)
    if len(PL) < 2 or len(PR) < 2:
        return None

    base = (np.asarray(p0, float) if use_generic
            else np.full(N_BONES, 1.0 / N_BONES))
    base = base / base.sum()
    log_base = np.log(base + EPS)

    a = np.zeros(N_BONES)
    dL = np.zeros(N_BONES)
    dR = np.zeros(N_BONES)

    for _ in range(n_iter):
        p_seq = softmax_from_log(log_base + a)
        pL = softmax_from_log(np.log(p_seq + EPS) + dL)
        pR = softmax_from_log(np.log(p_seq + EPS) + dR)

        AL, bL = _linear_rows(PL.items, pL, K, 1.0)
        AR, bR = _linear_rows(PR.items, pR, K, 1.0)
        if AL is None or AR is None:
            return None

        if shared:
            # parameters [a, dL, dR]; the left rows see a + dL, right a + dR
            n = N_BONES
            Z = np.zeros((AL.shape[0], n))
            top = np.hstack([AL, AL, Z])
            Z2 = np.zeros((AR.shape[0], n))
            bot = np.hstack([AR, Z2, AR])
            A = np.vstack([top, bot])
            b = np.concatenate([bL, bR])
            scale = float(np.linalg.norm(A)) / max(A.shape[1], 1) + 1e-12
            reg = np.zeros((3 * n, 3 * n))
            reg[:n, :n] = np.eye(n) * lam_generic * scale
            reg[n:2 * n, n:2 * n] = np.eye(n) * lam_side * scale
            reg[2 * n:, 2 * n:] = np.eye(n) * lam_side * scale
            A = np.vstack([A, reg])
            b = np.concatenate([b, np.zeros(3 * n)])
            try:
                theta, *_ = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                return None
            if not np.isfinite(theta).all():
                return None
            a = _centre(a + np.clip(theta[:N_BONES], -0.5, 0.5))
            dL = _centre(dL + np.clip(theta[N_BONES:2 * N_BONES], -0.5, 0.5))
            dR = _centre(dR + np.clip(theta[2 * N_BONES:], -0.5, 0.5))
        else:
            # independent sides: no shared p_seq
            out = []
            for A_, b_, cur in ((AL, bL, dL), (AR, bR, dR)):
                scale = float(np.linalg.norm(A_)) / max(N_BONES, 1) + 1e-12
                A2 = np.vstack([A_, np.eye(N_BONES) * lam_side * scale])
                b2 = np.concatenate([b_, np.zeros(N_BONES)])
                try:
                    th, *_ = np.linalg.lstsq(A2, b2, rcond=None)
                except np.linalg.LinAlgError:
                    return None
                out.append(_centre(cur + np.clip(th, -0.5, 0.5)))
            dL, dR = out
            a = np.zeros(N_BONES)

    p_seq = softmax_from_log(log_base + a)
    pL = softmax_from_log(np.log(p_seq + EPS) + dL)
    pR = softmax_from_log(np.log(p_seq + EPS) + dR)
    return {"a": a, "delta_L": dL, "delta_R": dR,
            "p_seq": p_seq, "p_L": pL, "p_R": pR}


def score_frames(frames, K, dist, p, diag_px):
    """Held-out score: anatomy FROZEN, only the pose is fitted per frame.

    Returned as a fraction of the image diagonal, so it is dimensionless and
    comparable across units without any candidate-dependent rescaling.
    """
    vals = []
    for dirs, uv_raw, use in frames:
        if use.sum() < MIN_OK_JOINTS:
            continue
        uvi = undistort_once(uv_raw, K, dist)
        X = assemble(dirs, p)
        rvec, tvec = pose_ideal(X, uvi, use, K)
        if rvec is None:
            continue
        vals.append(reproj_ideal(X, rvec, tvec, K, uvi, use))
    if not vals:
        return float("nan")
    return float(np.median(vals)) / float(diag_px)
