"""Reprojection error rows + summary statistics.

The raw per-joint rows are the durable artefact: every table, figure and
statistic in later experiments must be derivable from them without
re-reading the datasets.
"""
from __future__ import annotations

import numpy as np

RAW_COLUMNS = [
    "dataset", "subset", "sequence", "camera", "frame", "hand", "joint",
    "u_gt", "v_gt", "u_projected", "v_projected", "error_px",
    "depth_mm_or_m", "visible", "valid", "in_image", "gt_in_image", "inside_bbox",
    "distortion_applied", "image_width", "image_height",
    "joints2d_source", "notes",
]


def _bbox_contains(bbox, uv, pad: float = 0.0):
    """bbox may be [x, y, w, h] or {'left': [...], 'right': [...]}."""
    if bbox is None or not np.isfinite(uv).all():
        return ""
    if isinstance(bbox, dict):
        return ""
    b = np.asarray(bbox, dtype=float).reshape(-1)
    if b.size != 4:
        return ""
    x, y, w, h = b
    return int(x - pad <= uv[0] <= x + w + pad and y - pad <= uv[1] <= y + h + pad)


def per_joint_rows(sample, apply_distortion: bool = True):
    """Yield one dict per joint for a Sample."""
    cam = sample.camera
    uv, z = cam.project(sample.joints3d_world, apply_distortion=apply_distortion)
    in_img = cam.in_image(uv)
    g = sample.joints2d_gt
    bbox = sample.extra.get("bbox")
    for j in range(len(uv)):
        has_gt = g is not None and np.isfinite(g[j]).all()
        err = float(np.linalg.norm(uv[j] - g[j])) if has_gt and np.isfinite(uv[j]).all() else ""
        yield {
            "dataset": sample.dataset, "subset": sample.subset,
            "sequence": sample.sequence, "camera": sample.camera_name,
            "frame": sample.frame, "hand": sample.hand, "joint": j,
            "u_gt": g[j][0] if has_gt else "", "v_gt": g[j][1] if has_gt else "",
            "u_projected": uv[j][0] if np.isfinite(uv[j][0]) else "",
            "v_projected": uv[j][1] if np.isfinite(uv[j][1]) else "",
            "error_px": err,
            "depth_mm_or_m": float(z[j]),
            "visible": int(bool(in_img[j])),
            "valid": int(bool(sample.valid[j])),
            "in_image": int(bool(in_img[j])),
            "gt_in_image": int(bool(has_gt and cam.in_image(g[j:j + 1])[0])) if has_gt else "",
            "inside_bbox": _bbox_contains(bbox, uv[j]),
            "distortion_applied": int(bool(apply_distortion and cam.has_distortion)),
            "image_width": cam.width if cam.width else "",
            "image_height": cam.height if cam.height else "",
            "joints2d_source": sample.joints2d_source,
            "notes": sample.notes,
        }


def summarize(errors) -> dict:
    """Standard error statistics. No pass/fail threshold is imposed here."""
    a = np.asarray([e for e in errors if e != "" and np.isfinite(e)], dtype=float)
    if a.size == 0:
        return {"n": 0, "mean_px": "", "median_px": "", "p90_px": "",
                "p95_px": "", "max_px": "", "rmse_px": ""}
    return {
        "n": int(a.size),
        "mean_px": round(float(a.mean()), 4),
        "median_px": round(float(np.median(a)), 4),
        "p90_px": round(float(np.percentile(a, 90)), 4),
        "p95_px": round(float(np.percentile(a, 95)), 4),
        "max_px": round(float(a.max()), 4),
        "rmse_px": round(float(np.sqrt((a ** 2).mean())), 4),
    }
