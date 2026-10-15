"""f from a planar hand template: per-frame IPPE pose at every candidate f, robust mean reprojection cost.

fixed_template: the frontal template is the true planar shape; f from a log grid refined by a parabola in log f.
joint_delta:    the frontal hand was really tilted by delta towards phi; the template is stretched by 1 / cos(delta) along phi
                (first-order weak-perspective correction) and (f, delta, phi) is found on a grid and refined by Nelder-Mead, with a
                soft prior (delta / delta_prior)^2 in units of the per-point noise variance.
Pinhole, principal point fixed, square pixels, no distortion.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import minimize


class Problem:
    def __init__(self, template: np.ndarray, frames: list, cx: float, cy: float, huber_px: np.ndarray, min_points: int = 4):
        """template (J, 2); frames: list of (obs (J, 2), visible (J,)) ; huber_px: per frame Huber width."""
        self.T = np.asarray(template, float)
        self.frames = [(np.asarray(o, float), np.asarray(v, bool)) for o, v in frames]
        self.keep = [i for i, (o, v) in enumerate(self.frames) if v.sum() >= max(4, min_points)]
        self.cx, self.cy = float(cx), float(cy)
        self.huber = np.asarray(huber_px, float)
        self.n_points = int(sum(self.frames[i][1].sum() for i in self.keep))

    def subset(self, idx):
        p = Problem.__new__(Problem)
        p.T, p.frames, p.cx, p.cy, p.huber = self.T, self.frames, self.cx, self.cy, self.huber
        p.keep = [i for i in idx if i in set(self.keep)]
        p.n_points = int(sum(self.frames[i][1].sum() for i in p.keep))
        return p


def stretch(T: np.ndarray, delta_deg: float, phi_deg: float) -> np.ndarray:
    u = np.array([np.cos(np.radians(phi_deg)), np.sin(np.radians(phi_deg))])
    k = 1.0 / np.cos(np.radians(min(abs(delta_deg), 80.0))) - 1.0
    return T + k * (T @ u)[:, None] * u[None, :]


def _huber(r, d):
    return np.where(r <= d, r * r, 2 * d * r - d * d)


def frame_fit(T2: np.ndarray, obs: np.ndarray, vis: np.ndarray, K: np.ndarray):
    """IPPE pose of the planar template on the visible points; the lower-error of the two solutions. Returns (residuals, R, t)."""
    obj = np.c_[T2[vis], np.zeros(vis.sum())].astype(np.float64)
    img = obs[vis].astype(np.float64)
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
    k = int(np.argmin(np.asarray(errs).ravel())) if n > 1 else 0
    R, _ = cv2.Rodrigues(rvecs[k])
    t = tvecs[k].ravel()
    X = obj @ R.T + t
    proj = K[:2, :2] @ (X[:, :2] / X[:, 2:3]).T + K[:2, 2:3]
    return np.linalg.norm(proj.T - img, axis=1), R, t


def K_of(f, cx, cy):
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])


def cost(pb: Problem, f: float, T2: np.ndarray | None = None, details: bool = False):
    T2 = pb.T if T2 is None else T2
    K = K_of(f, pb.cx, pb.cy)
    per, out = [], []
    for i in pb.keep:
        o, v = pb.frames[i]
        r, R, t = frame_fit(T2, o, v, K)
        per.append(float(_huber(r, pb.huber[i]).mean()))
        if details:
            out.append({"frame": i, "res": r, "vis": v, "R": R, "t": t})
    c = float(np.mean(per)) if per else float("inf")
    return (c, out) if details else c


def f_grid(cfg_e: dict, long_side: float) -> np.ndarray:
    lo, hi = cfg_e["f_ratio_range"]
    return np.geomspace(lo * long_side, hi * long_side, int(cfg_e["f_grid_n"]))


def _parabola(lf: np.ndarray, c: np.ndarray, k: int) -> float:
    if k == 0 or k == len(lf) - 1:
        return float(lf[k])
    x, y = lf[k - 1:k + 2], c[k - 1:k + 2]
    A = np.vstack([x ** 2, x, np.ones(3)]).T
    a, b, _ = np.linalg.solve(A, y)
    return float(-b / (2 * a)) if a > 0 else float(lf[k])


def fixed_template(pb: Problem, grid: np.ndarray) -> dict:
    c = np.array([cost(pb, f) for f in grid])
    k = int(np.argmin(c))
    lf = _parabola(np.log(grid), c, k)
    f = float(np.exp(np.clip(lf, np.log(grid[0]), np.log(grid[-1]))))
    return {"f": f, "delta_deg": 0.0, "phi_deg": 0.0, "profile_f": grid.tolist(), "profile_cost": c.tolist(), "cost": cost(pb, f),
            "at_grid_edge": bool(k in (0, len(grid) - 1))}


def joint_delta(pb: Problem, grid: np.ndarray, jd: dict, sigma2: float, start=None) -> dict:
    """(f, delta, phi): grid then Nelder-Mead on the total = N * mean cost / sigma^2 + (delta / delta_prior)^2."""
    N = max(pb.n_points, 1)
    dp = float(jd["delta_prior_deg"])

    def total(lf, d, ph):
        return N * cost(pb, float(np.exp(lf)), stretch(pb.T, d, ph)) / sigma2 + (d / dp) ** 2

    phis = np.linspace(0, 180, int(jd["phi_n"]), endpoint=False)
    prof = np.full(len(grid), np.inf)
    best = (np.inf, None)
    if start is None:
        for d in jd["delta_grid_deg"]:
            for ph in (phis if d > 0 else [0.0]):
                T2 = stretch(pb.T, d, ph)
                for k, f in enumerate(grid):
                    v = N * cost(pb, f, T2) / sigma2 + (d / dp) ** 2
                    prof[k] = min(prof[k], v)
                    if v < best[0]:
                        best = (v, (np.log(f), float(d), float(ph)))
        x0 = best[1]
    else:
        x0 = start
    lo, hi = np.log(grid[0]), np.log(grid[-1])
    fun = lambda x: total(np.clip(x[0], lo, hi), abs(x[1]), x[2] % 180.0)   # noqa: E731
    r = minimize(fun, np.array(x0, float), method="Nelder-Mead",
                 options={"maxiter": int(jd["nm_max_iter"]), "xatol": 1e-4, "fatol": 1e-6, "initial_simplex": None})
    lf, d, ph = float(np.clip(r.x[0], lo, hi)), abs(float(r.x[1])), float(r.x[2] % 180.0)
    f = float(np.exp(lf))
    return {"f": f, "delta_deg": d, "phi_deg": ph, "profile_f": grid.tolist(), "profile_cost": (prof * sigma2 / N).tolist() if start is None else None,
            "cost": cost(pb, f, stretch(pb.T, d, ph)), "objective": float(r.fun), "nm_iters": int(r.nit), "at_grid_edge": bool(lf <= lo + 1e-6 or lf >= hi - 1e-6)}


def run(pb: Problem, method: str, grid: np.ndarray, ecfg: dict, sigma2: float, start=None) -> dict:
    if method == "fixed_template":
        return fixed_template(pb, grid)
    if method == "joint_delta":
        return joint_delta(pb, grid, ecfg["joint_delta"], sigma2, start)
    raise ValueError(method)


def diagnostics(pb: Problem, sol: dict, box_long_med: float, joint_ids: list, tilt_ids=None) -> dict:
    T2 = stretch(pb.T, sol["delta_deg"], sol["phi_deg"])
    c0, det = cost(pb, sol["f"], T2, details=True)
    res = np.concatenate([d["res"] for d in det]) if det else np.zeros(0)
    per_joint = {}
    J = len(joint_ids)
    acc = [[] for _ in range(J)]
    for d in det:
        for pos, j in enumerate(np.flatnonzero(d["vis"])):
            acc[j].append(d["res"][pos])
    for j in range(J):
        per_joint[int(joint_ids[j])] = float(np.sqrt(np.mean(np.square(acc[j])))) if acc[j] else float("nan")
    sel = [d for d in det if tilt_ids is None or d["frame"] in set(tilt_ids)]
    normals = np.array([d["R"][:, 2] for d in sel]) if sel else np.zeros((0, 3))
    az = np.degrees(np.arctan2(normals[:, 1], normals[:, 0])) % 360 if len(normals) else np.zeros(0)
    tilt = np.degrees(np.arccos(np.clip(np.abs(normals[:, 2]), 0, 1))) if len(normals) else np.zeros(0)
    cp = cost(pb, sol["f"] * 1.15, T2)
    cm = cost(pb, sol["f"] / 1.15, T2)
    rms = float(np.sqrt(np.mean(res ** 2))) if len(res) else float("nan")
    return {"reproj_rms_px": rms, "reproj_rms_rel": rms / box_long_med if box_long_med else float("nan"), "per_joint_rms_px": per_joint,
            "sharpness": float(min(cp, cm) / max(c0, 1e-12)), "cost": c0, "cost_plus15": cp, "cost_minus15": cm,
            "pnp_azimuth_deg": az.tolist(), "pnp_tilt_deg": tilt.tolist(), "azimuth_bins8": int(len(np.unique((az // 45).astype(int)))) if len(az) else 0,
            "frames_used": [int(d["frame"]) for d in det], "frame_rms_px": [float(np.sqrt(np.mean(d["res"] ** 2))) for d in det]}
