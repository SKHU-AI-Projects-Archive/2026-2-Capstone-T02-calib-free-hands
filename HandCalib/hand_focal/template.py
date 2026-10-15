"""Frontal template (similarity Procrustes mean) and the weak-perspective 2D tilt indicator theta2d."""
from __future__ import annotations

import numpy as np


def similarity_align(src: np.ndarray, dst: np.ndarray):
    """Least-squares similarity src -> dst (2D). Returns (s, R, t) with dst ~ s R src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a)
    D = np.diag([1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = float((S * np.diag(D)).sum() / max((a ** 2).sum(), 1e-12))
    return s, R, mu_d - s * R @ mu_s


def build(front_pts: np.ndarray, iters: int = 4) -> dict:
    """front_pts (N, J, 2) of frames with every joint visible -> template (J, 2), centred, in the window's pixel scale."""
    X = front_pts.astype(float)
    T = X[0] - X[0].mean(0)
    for _ in range(iters):
        aligned = []
        for P in X:
            s, R, t = similarity_align(P, T)
            aligned.append((s * (R @ P.T)).T + t)
        A = np.array(aligned)
        T = A.mean(0)
        T = T - T.mean(0)
    # scale = mean pixel size of the window
    sizes = [np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean()) for P in X]
    T = T * (np.mean(sizes) / np.sqrt((T ** 2).sum(1).mean()))
    res = []
    for P in X:                                     # residuals in each frame's own pixels
        s, R, t = similarity_align(T, P)
        res.append(np.linalg.norm((s * (R @ T.T)).T + t - P, axis=1))
    res = np.array(res)
    return {"template": T, "jitter_rms_px": float(np.sqrt((res ** 2).mean())), "n_frames": int(len(X))}


def affine_tilt(T: np.ndarray, P: np.ndarray):
    """Least-squares affine T -> P; theta2d = arccos(s2 / s1) of its linear part; axis = direction of the compressed image axis (deg, 0-180)."""
    A = np.c_[T, np.ones(len(T))]
    M, *_ = np.linalg.lstsq(A, P, rcond=None)
    L = M[:2].T                                       # 2x2 linear part, P ~ L T + t
    U, S, Vt = np.linalg.svd(L)
    ratio = float(np.clip(S[1] / S[0], 0.0, 1.0)) if S[0] > 0 else 0.0
    theta = float(np.degrees(np.arccos(ratio)))
    axis = float(np.degrees(np.arctan2(U[1, 1], U[0, 1])) % 180.0)
    return theta, axis


def finger_extension(kp: np.ndarray) -> dict:
    """MCP -> tip straight distance / sum of the three bone lengths, per finger (index .. pinky), on 2D points."""
    out = {}
    for name, ids in (("index", [5, 6, 7, 8]), ("middle", [9, 10, 11, 12]), ("ring", [13, 14, 15, 16]), ("pinky", [17, 18, 19, 20])):
        P = kp[:, ids]
        straight = np.linalg.norm(P[:, 3] - P[:, 0], axis=-1)
        path = np.linalg.norm(np.diff(P, axis=1), axis=-1).sum(1)
        out[name] = float(np.nanmedian(straight / np.maximum(path, 1e-9)))
    out["min"] = float(min(out.values()))
    out["mean"] = float(np.mean([out[k] for k in ("index", "middle", "ring", "pinky")]))
    return out
