"""Generalised focal-profile solver for CAM-EXP-006.1.

It reduces EXACTLY to CAM-EXP-006's solver when
    cx = W/2, cy = H/2, dist = None, fy_over_fx = 1.0
and the hand-weighted aggregator is used. That identity is what makes the R0
reproduction meaningful, and it is asserted by tests in run_real_reanalysis.py.

The distortion coefficients, when supplied, are applied at EVERY candidate
focal, inside both solvePnP and projectPoints. The 2D is never pre-undistorted
once and reused, because K changes with the candidate focal.

GT focal MAGNITUDE is never an argument to anything in this file.
"""
from __future__ import annotations

import numpy as np

GAMMA_MIN, GAMMA_MAX, N_GRID = 0.25, 5.0, 121
FLAT_TOL = 0.10


def gamma_grid() -> np.ndarray:
    return np.geomspace(GAMMA_MIN, GAMMA_MAX, N_GRID)


def make_K(f, cx, cy, fy_over_fx=1.0):
    return np.array([[f, 0.0, cx], [0.0, fy_over_fx * f, cy], [0.0, 0.0, 1.0]])


def _pnp_error(xyz, uv, K, dist):
    """Median reprojection error in px for one hand at one candidate focal.

    SQPNP is the primary back-end, matching CAM-EXP-006. IPPE and ITERATIVE are
    tried only when SQPNP raises, which happens for coplanar input.
    """
    import cv2
    obj = np.ascontiguousarray(xyz.reshape(-1, 1, 3), dtype=np.float64)
    img = np.ascontiguousarray(uv.reshape(-1, 1, 2), dtype=np.float64)
    d = None if dist is None else np.asarray(dist, np.float64).reshape(1, -1)
    ok = False
    for flag in (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_IPPE,
                 cv2.SOLVEPNP_ITERATIVE):
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, d, flags=flag)
        except cv2.error:
            ok = False
        if ok:
            break
    if not ok:
        return np.nan
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, d)
    return float(np.median(np.linalg.norm(
        proj.reshape(-1, 2) - uv.reshape(-1, 2), axis=1)))


def profile_hand(xyz, uv, w, h, cx=None, cy=None, dist=None, fy_over_fx=1.0,
                 grid=None):
    """Objective curve in px over the focal grid, for one hand in one frame."""
    grid = gamma_grid() if grid is None else grid
    cx = w / 2.0 if cx is None else cx
    cy = h / 2.0 if cy is None else cy
    s = max(w, h)
    return np.array([_pnp_error(xyz, uv, make_K(g * s, cx, cy, fy_over_fx),
                                dist) for g in grid])


# ----------------------------------------------------------------- aggregation
def aggregate_hand_weighted(per_frame_hand_profiles):
    """CAM-EXP-006's original behaviour: mean of log over every HAND profile.

    A frame holding two usable hands therefore contributes twice.
    """
    flat = [p for ps in per_frame_hand_profiles for p in ps]
    if not flat:
        return None
    p = np.asarray(flat, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.log(np.clip(p, 1e-6, None))
    lg[~np.isfinite(p)] = np.nan
    if np.all(np.isnan(lg)):
        return np.full(p.shape[-1], np.nan)
    return np.nanmean(lg, axis=0)


def aggregate_frame_balanced(per_frame_hand_profiles, across="mean"):
    """CAM0061_FRAME_BALANCED: hands -> one frame score, then frames -> view.

    Step 2 is a median over the usable hands inside a frame, so every frame
    carries equal weight. Step 3 is unchanged from CAM-EXP-006: the mean, in
    log space, across frames.
    """
    frame_scores = []
    for ps in per_frame_hand_profiles:
        if not ps:
            continue
        a = np.asarray(ps, float)
        with np.errstate(invalid="ignore"):
            frame_scores.append(np.nanmedian(a, axis=0))
    if not frame_scores:
        return None
    p = np.asarray(frame_scores, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.log(np.clip(p, 1e-6, None))
    lg[~np.isfinite(p)] = np.nan
    if np.all(np.isnan(lg)):
        return np.full(p.shape[-1], np.nan)
    return (np.nanmedian(lg, axis=0) if across == "median"
            else np.nanmean(lg, axis=0))


# --------------------------------------------------------------------- picking
def pick(curve, grid=None) -> dict:
    """argmin with parabolic refinement in log-gamma. Identical to CAM-006."""
    grid = gamma_grid() if grid is None else grid
    c = np.asarray(curve, float)
    if c is None or not np.isfinite(c).any():
        return {"gamma": np.nan, "status": "NO_SOLUTION", "flat_profile": 0,
                "boundary_solution": 0, "curvature": np.nan,
                "depth_of_minimum": np.nan, "profile_width": np.nan}
    i = int(np.nanargmin(c))
    lg = np.log(grid)
    boundary = int(i == 0 or i == len(grid) - 1)
    if boundary:
        g_hat, curv = float(grid[i]), np.nan
    else:
        y0, y1, y2 = c[i - 1], c[i], c[i + 1]
        x1, x2 = lg[i], lg[i + 1]
        denom = (y0 - 2 * y1 + y2)
        if np.isfinite(denom) and abs(denom) > 1e-12:
            delta = float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))
            g_hat = float(np.exp(x1 + delta * (x2 - x1)))
        else:
            g_hat = float(grid[i])
        curv = float(denom)

    cmin = float(c[i])
    depth = float(np.nanmin(np.array([c[0], c[-1]])) - cmin)
    flat = int(np.isfinite(depth) and depth < np.log1p(FLAT_TOL))

    # profile width: the log-gamma span within 10 % of the minimum objective
    thr = cmin + np.log1p(0.10)
    within = np.flatnonzero(c <= thr)
    width = (float(lg[within[-1]] - lg[within[0]]) if within.size
             else float("nan"))

    status = ("BOUNDARY_SOLUTION" if boundary
              else "FLAT_PROFILE" if flat else "ok")
    return {"gamma": g_hat, "status": status, "flat_profile": flat,
            "boundary_solution": boundary, "curvature": curv,
            "depth_of_minimum": depth, "profile_width": width}


def solve_view(per_frame_hand_profiles, w, h, aggregation="frame_balanced",
               grid=None):
    grid = gamma_grid() if grid is None else grid
    agg = (aggregate_hand_weighted if aggregation == "hand_weighted"
           else aggregate_frame_balanced)
    curve = agg(per_frame_hand_profiles)
    n_frames = sum(1 for ps in per_frame_hand_profiles if ps)
    n_hands = sum(len(ps) for ps in per_frame_hand_profiles)
    if curve is None:
        return {"gamma": np.nan, "status": "NO_CASES", "flat_profile": 0,
                "boundary_solution": 0, "curvature": np.nan,
                "depth_of_minimum": np.nan, "profile_width": np.nan,
                "n_frames_used": 0, "n_hand_observations": 0,
                "focal_px": np.nan, "min_objective_px": np.nan}
    out = pick(curve, grid)
    out["n_frames_used"] = n_frames
    out["n_hand_observations"] = n_hands
    out["focal_px"] = out["gamma"] * max(w, h)
    out["min_objective_px"] = float(np.exp(np.nanmin(curve))) \
        if np.isfinite(curve).any() else np.nan
    return out
