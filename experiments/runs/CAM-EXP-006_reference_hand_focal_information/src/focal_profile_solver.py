"""The PROFILED_PNP_FOCAL_DIAGNOSTIC solver, exactly as frozen in the spec.

For one candidate focal f the intrinsics are K(f) with the principal point
FIXED at the image centre. The per-hand rotation and translation are nuisance
parameters, re-fitted by PnP at every candidate f, and the objective is the
median reprojection error over joints. Profiling f over a log-spaced grid turns
a 7-parameter problem into a 1-D curve whose shape also tells us whether f is
identifiable at all.

Inputs to the solver are ONLY: the reference 3D, the provided 2D of the camera
under test, and the image size. GT focal, GT extrinsics and GT distortion are
never read here.
"""
from __future__ import annotations

import numpy as np

GAMMA_MIN, GAMMA_MAX, N_GRID = 0.25, 5.0, 121
FLAT_TOL = 0.10          # profile ends within 10 % of the minimum -> flat


def gamma_grid() -> np.ndarray:
    return np.geomspace(GAMMA_MIN, GAMMA_MAX, N_GRID)


def _pnp_error(xyz, uv, f, w, h):
    """Median reprojection error in px for one hand at one candidate focal.

    SQPNP is the frozen primary solver. It refuses coplanar point sets, which
    the PLANARIZED control produces by construction, so a planar-capable
    fallback (IPPE, then ITERATIVE) is tried in turn. Without it the control
    would fail for an implementation reason rather than a geometric one, and
    would falsely appear to support the hypothesis.
    """
    import cv2
    K = np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])
    obj = xyz.reshape(-1, 1, 3).astype(np.float64)
    img = uv.reshape(-1, 1, 2).astype(np.float64)
    ok = False
    for flag in (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_IPPE,
                 cv2.SOLVEPNP_ITERATIVE):
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=flag)
        except cv2.error:
            ok = False
        if ok:
            break
    if not ok:
        return np.nan
    proj, _ = cv2.projectPoints(xyz.astype(np.float64), rvec, tvec, K, None)
    return float(np.median(np.linalg.norm(
        proj.reshape(-1, 2) - uv, axis=1)))


def profile_one(xyz, uv, w, h, grid=None):
    """Objective curve over the focal grid for a single hand in a single frame."""
    grid = gamma_grid() if grid is None else grid
    s = max(w, h)
    return np.array([_pnp_error(xyz, uv, g * s, w, h) for g in grid])


def combine(profiles) -> np.ndarray:
    """Mean over frames of the per-frame log objective, as frozen in the spec."""
    p = np.asarray(profiles, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.log(np.clip(p, 1e-6, None))
    lg[~np.isfinite(p)] = np.nan
    if np.all(np.isnan(lg)):
        return np.full(p.shape[-1], np.nan)
    return np.nanmean(lg, axis=0)


def pick(curve, grid=None) -> dict:
    """argmin with a parabolic refinement in log-gamma, plus the spec's flags."""
    grid = gamma_grid() if grid is None else grid
    c = np.asarray(curve, float)
    if not np.isfinite(c).any():
        return {"gamma": np.nan, "status": "NO_SOLUTION",
                "flat_profile": 0, "boundary_solution": 0,
                "curvature": np.nan, "depth_of_minimum": np.nan}
    i = int(np.nanargmin(c))
    lg = np.log(grid)
    boundary = int(i == 0 or i == len(grid) - 1)
    if boundary:
        g_hat = float(grid[i])
        curv = np.nan
    else:
        y0, y1, y2 = c[i - 1], c[i], c[i + 1]
        x0, x1, x2 = lg[i - 1], lg[i], lg[i + 1]
        denom = (y0 - 2 * y1 + y2)
        if np.isfinite(denom) and abs(denom) > 1e-12:
            delta = 0.5 * (y0 - y2) / denom
            delta = float(np.clip(delta, -1.0, 1.0))
            g_hat = float(np.exp(x1 + delta * (x2 - x1)))
        else:
            g_hat = float(grid[i])
        curv = float(denom)

    ends = np.array([c[0], c[-1]])
    cmin = float(c[i])
    # a deep minimum means the ends are much WORSE than the optimum
    depth = float(np.nanmin(ends) - cmin)        # in log units
    flat = int(np.isfinite(depth) and depth < np.log1p(FLAT_TOL))
    status = "ok"
    if boundary:
        status = "BOUNDARY_SOLUTION"
    elif flat:
        status = "FLAT_PROFILE"
    return {"gamma": g_hat, "status": status, "flat_profile": flat,
            "boundary_solution": boundary, "curvature": curv,
            "depth_of_minimum": depth}


def solve(cases, w, h):
    """Solve one (view, frame set) from a list of (xyz, uv) hand observations."""
    grid = gamma_grid()
    profiles = [profile_one(xyz, uv, w, h, grid) for xyz, uv in cases]
    if not profiles:
        return {"gamma": np.nan, "status": "NO_CASES", "flat_profile": 0,
                "boundary_solution": 0, "curvature": np.nan,
                "depth_of_minimum": np.nan, "n_hands": 0}
    curve = combine(profiles)
    out = pick(curve, grid)
    out["n_hands"] = len(profiles)
    out["focal_px"] = out["gamma"] * max(w, h)
    out["min_objective_px"] = float(np.exp(np.nanmin(curve)))
    return out
