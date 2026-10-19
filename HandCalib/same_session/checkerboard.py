"""Checkerboard reference f of one recording (rules of the experiment specification; video_calib's code was not available).

Sampling: frames of the board segment at sample_fps -> corner detection (findChessboardCornersSB on a downscaled grey frame, then
cornerSubPix at full resolution) -> up to max_frames frames chosen by farthest-point sampling of a pose descriptor.
Models (cv2.calibrateCamera):
  A  fx, fy, cx, cy free; k1 k2 p1 p2 k3
  B  fx, fy, cx, cy free; k1 k2
  C  fx = fy, cx, cy free; k1 k2
  D  fx = fy, principal point fixed at the image centre (W / 2, H / 2); k1 k2
  E  fx = fy, centre principal point, no distortion (the hand method's and AnyCalib's assumptions)
f_reference = median fx of the reference models (A-D).
"""
from __future__ import annotations

import cv2
import numpy as np

CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)


def detect(gray, inner, long_side, win):
    """Returns (corners (N, 2) full-resolution pixels, pattern size) or (None, None)."""
    h, w = gray.shape
    s = min(1.0, long_side / max(h, w))
    g = cv2.resize(gray, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else gray
    for ps in (tuple(inner), tuple(inner[::-1])):
        ok, c = cv2.findChessboardCornersSB(g, ps, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if ok:
            c = (c.reshape(-1, 2) / s).astype(np.float32).reshape(-1, 1, 2)
            c = cv2.cornerSubPix(gray, c, (win, win), (-1, -1), CRIT)
            return c.reshape(-1, 2).astype(np.float64), ps
    return None, None


def object_points(ps, square):
    nx, ny = ps
    g = np.zeros((nx * ny, 3), np.float64)
    g[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) * square
    return g


def _init_K(W, H):
    f = float(max(W, H))
    return np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])


def pose_descriptor(corners, ps, square, W, H):
    """Rough pose (K = long side, centre pp, no distortion): board normal (3), board centre in normalised image coordinates (2),
    log apparent size (1)."""
    obj = object_points(ps, square)
    ok, rv, tv = cv2.solvePnP(obj, corners, _init_K(W, H), None, flags=cv2.SOLVEPNP_IPPE)
    R, _ = cv2.Rodrigues(rv)
    n = R[:, 2] * np.sign(R[2, 2] or 1.0)
    c = corners.mean(0)
    size = np.sqrt(cv2.contourArea(cv2.convexHull(corners.astype(np.float32))))
    return np.r_[n, (c[0] - W / 2) / max(W, H) * 2, (c[1] - H / 2) / max(W, H) * 2, np.log(size / max(W, H))]


def farthest_point(desc, k, start=0):
    idx = [start]
    d = np.linalg.norm(desc - desc[start], axis=1)
    while len(idx) < min(k, len(desc)):
        j = int(np.argmax(d))
        idx.append(j)
        d = np.minimum(d, np.linalg.norm(desc - desc[j], axis=1))
    return sorted(idx)


FLAGS = {
    "A": 0,
    "B": cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3,
    "C": cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3,
    "D": cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3,
    "E": cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3,
}


def calibrate(views, W, H, model):
    """views: list of (obj (N, 3), img (N, 2)). Returns {fx, fy, cx, cy, dist, rms, rvecs, tvecs}."""
    obj = [v[0].astype(np.float32) for v in views]
    img = [v[1].astype(np.float32).reshape(-1, 1, 2) for v in views]
    K0 = _init_K(W, H)
    flags = FLAGS[model] | cv2.CALIB_USE_INTRINSIC_GUESS
    rms, K, dist, rv, tv = cv2.calibrateCamera(obj, img, (W, H), K0.copy(), np.zeros(5), flags=flags,
                                               criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-10))
    return {"fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]), "dist": np.ravel(dist)[:5].tolist(),
            "rms": float(rms), "rvecs": rv, "tvecs": tv}


def f_reference(views, W, H, ref_models):
    fx = [calibrate(views, W, H, m)["fx"] for m in ref_models]
    return float(np.median(fx))


def analyse(frames, ccfg: dict, W: int, H: int, log=print) -> dict:
    """frames: list of {index, time_s, gray (H, W) uint8}. Returns the full checkerboard result."""
    inner, sq = ccfg["inner_corners"], float(ccfg["square_mm"])
    det = []
    for fr in frames:
        c, ps = detect(fr["gray"], inner, ccfg["detect_long_side"], int(ccfg["subpix_window"]))
        if c is not None:
            det.append({"index": fr["index"], "time_s": fr["time_s"], "corners": c, "ps": ps})
    out = {"n_sampled": len(frames), "n_detected": len(det)}
    if len(det) < 5:
        out.update({"verdict": "REJECT", "failed": "검출된 체커보드 프레임이 5장 미만"})
        return out
    desc = np.array([pose_descriptor(d["corners"], d["ps"], sq, W, H) for d in det])
    start = int(np.argmin(np.arccos(np.clip(np.abs(desc[:, 2]), 0, 1))))      # most frontal frame first
    sel = farthest_point(desc, int(ccfg["max_frames"]), start)
    views = [(object_points(det[i]["ps"], sq), det[i]["corners"]) for i in sel]
    times = np.array([det[i]["time_s"] for i in sel])
    models = {}
    for m in ccfg["models"]:
        r = calibrate(views, W, H, m)
        models[m] = {k: r[k] for k in ("fx", "fy", "cx", "cy", "dist", "rms")}
        if m == "D":
            rD = r
    ref = [models[m]["fx"] for m in ccfg["reference_models"]]
    f_ref = float(np.median(ref))
    model_range = float((max(ref) - min(ref)) / f_ref)
    # block bootstrap over the time-ordered selected frames
    rng = np.random.default_rng(0)
    order = np.argsort(times)
    B = int(ccfg["bootstrap_block_frames"])
    blocks = [order[i:i + B] for i in range(0, len(order), B)]
    bs = []
    for _ in range(int(ccfg["n_bootstrap"])):
        pick = [blocks[j] for j in rng.integers(0, len(blocks), len(blocks))]
        vv = [views[i] for blk in pick for i in blk]
        bs.append(f_reference(vv, W, H, ccfg["reference_models"]))
    bs = np.array(bs)
    # time segments
    segs = np.array_split(order, int(ccfg["time_segments"]))
    seg_f = [f_reference([views[i] for i in s], W, H, ccfg["reference_models"]) if len(s) >= 4 else float("nan") for s in segs]
    segf = [x for x in seg_f if np.isfinite(x)]
    seg_range = float((max(segf) - min(segf)) / f_ref) if len(segf) >= 2 else float("nan")
    # pose / coverage metrics from model D's extrinsics
    tilts = []
    normals = []
    for rv in rD["rvecs"]:
        R, _ = cv2.Rodrigues(rv)
        n = R[:, 2]
        normals.append(n / np.linalg.norm(n))
        tilts.append(float(np.degrees(np.arccos(min(1.0, abs(n[2]) / np.linalg.norm(n))))))
    tilts = np.array(tilts)
    distinct = []
    for n in normals:
        if all(np.degrees(np.arccos(np.clip(abs(n @ m), 0, 1))) >= ccfg["distinct_pose_min_deg"] for m in distinct):
            distinct.append(n)
    gx, gy = ccfg["coverage_grid"]
    cells = np.zeros((gy, gx), int)
    for v in views:
        p = v[1]
        ix = np.clip((p[:, 0] / W * gx).astype(int), 0, gx - 1)
        iy = np.clip((p[:, 1] / H * gy).astype(int), 0, gy - 1)
        np.add.at(cells, (iy, ix), 1)
    coverage = float((cells > 0).mean())
    rms = float(np.median([models[m]["rms"] for m in ccfg["reference_models"]]))
    met = {"tilt_p90_deg": float(np.percentile(tilts, 90)), "tilt_max_deg": float(tilts.max()), "distinct_poses": len(distinct),
           "coverage": coverage, "model_range": model_range, "bootstrap_rel_std": float(np.std(bs, ddof=1) / np.mean(bs)),
           "segment_range": seg_range, "rms_px": rms}
    v = verdict(met, ccfg["verdict"])
    out.update({"n_selected": len(sel), "selected_times": times.tolist(), "selected_indices": [int(det[i]["index"]) for i in sel],
                "pattern_size": list(det[0]["ps"]), "models": models, "f_reference_px": f_ref, "f_model_E_px": models.get("E", {}).get("fx"),
                "bootstrap_f": bs.tolist(), "segment_f": seg_f, "metrics": met, "tilts_deg": tilts.tolist(), "coverage_cells": cells.tolist(), **v,
                "image_size": [W, H]})
    return out


def verdict(m, vc):
    def ok(t):
        return {"tilt_p90": m["tilt_p90_deg"] >= t["tilt_p90_min_deg"], "coverage": m["coverage"] >= t["coverage_min"],
                "poses": m["distinct_poses"] >= t["poses_min"], "model_range": m["model_range"] <= t["model_range_max"],
                "bootstrap": m["bootstrap_rel_std"] <= t["bootstrap_rel_std_max"],
                "segment_range": bool(np.isfinite(m["segment_range"]) and m["segment_range"] <= t["segment_range_max"]),
                "rms": m["rms_px"] <= t["rms_max_px"]}
    g, r = ok(vc["good"]), ok(vc["rough"])
    v = "GOOD" if all(g.values()) else ("ROUGH" if all(r.values()) else "REJECT")
    return {"verdict": v, "failed_good": [k for k, x in g.items() if not x], "failed_rough": [k for k, x in r.items() if not x]}
