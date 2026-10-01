"""Shared helpers for CAM-EXP-001.1 (GigaHands bad-view diagnosis).

Design notes
------------
* The per-view statistics are rebuilt from the CAM-EXP-001 raw CSV, which is
  treated as read-only input. The dataset itself is re-read only where a test
  genuinely needs pixels or alternative projections (lag, distortion, flips,
  overlays).
* A finding from the first probe shapes everything here: GigaHands writes
  ``(0, 0)`` with ``confidence = 1.0`` into ``keypoints_2d`` when a hand is not
  detected in a view. These sentinel rows are not real annotations, so they are
  flagged explicitly rather than being allowed to inflate the error statistics.
"""
from __future__ import annotations

import csv
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RUN_DIR = Path(__file__).resolve().parents[1]
EXPERIMENTS = RUN_DIR.parents[1]
REPO = EXPERIMENTS.parent
SRC_ROOT = EXPERIMENTS.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

CAM_EXP_001 = EXPERIMENTS / "runs" / "CAM-EXP-001_gt_projection_validation"
RAW_CSV = CAM_EXP_001 / "results" / "raw" / "reprojection_per_joint.csv.gz"

VIEW_KEYS = ("sequence", "camera", "frame", "hand")
# A view is called a zero-sentinel view when every annotated joint sits exactly
# at the image origin. Observed to be strictly all-or-nothing (never partial).
SENTINEL = (0.0, 0.0)


def rel(p) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return p.as_posix()


def load_gigahands_views(raw_csv: Path = RAW_CSV) -> dict:
    """Rebuild per-view records from the CAM-EXP-001 raw per-joint table.

    Returns {view_key: record}, where each record separates the errors of real
    annotations from the zero-sentinel rows.
    """
    if not raw_csv.exists():
        raise SystemExit(f"missing CAM-EXP-001 raw CSV: {raw_csv}")
    views: dict = defaultdict(lambda: {
        "n_valid": 0, "n_zero": 0, "n_compared": 0,
        "errors": [], "errors_all": [], "in_image": 0,
        "image_width": "", "image_height": "", "distortion_applied": "",
    })
    with gzip.open(raw_csv, "rt", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["dataset"] != "gigahands" or r["valid"] != "1" or not r["error_px"]:
                continue
            key = tuple(r[k] for k in VIEW_KEYS)
            v = views[key]
            v["n_valid"] += 1
            v["image_width"] = r["image_width"]
            v["image_height"] = r["image_height"]
            v["distortion_applied"] = r["distortion_applied"]
            err = float(r["error_px"])
            v["errors_all"].append(err)
            if (float(r["u_gt"]), float(r["v_gt"])) == SENTINEL:
                v["n_zero"] += 1
                continue
            v["in_image"] += int(r["in_image"] == "1" and r["gt_in_image"] == "1")
            if r["in_image"] == "1" and r["gt_in_image"] == "1":
                v["n_compared"] += 1
                v["errors"].append(err)
    return dict(views)


def stats(errors) -> dict:
    a = np.asarray([e for e in errors if np.isfinite(e)], dtype=float)
    if a.size == 0:
        return {k: "" for k in ("n", "mean_px", "median_px", "p90_px",
                                "p95_px", "max_px")}
    return {"n": int(a.size),
            "mean_px": round(float(a.mean()), 4),
            "median_px": round(float(np.median(a)), 4),
            "p90_px": round(float(np.percentile(a, 90)), 4),
            "p95_px": round(float(np.percentile(a, 95)), 4),
            "max_px": round(float(a.max()), 4)}


def otsu_threshold_log(values: np.ndarray) -> float:
    """Pick the good/bad split from the data, not from a hard-coded constant.

    The per-view median error is strongly bimodal, so the threshold is chosen
    by maximising between-class variance (Otsu) on log10(error). The returned
    value is in pixels.
    """
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a) & (a > 0)]
    if a.size < 10:
        return float("nan")
    x = np.log10(a)
    edges = np.linspace(x.min(), x.max(), 256)
    best, best_var = edges[0], -1.0
    for t in edges[1:-1]:
        lo, hi = x[x <= t], x[x > t]
        if lo.size < 5 or hi.size < 5:
            continue
        w0, w1 = lo.size / x.size, hi.size / x.size
        var = w0 * w1 * (lo.mean() - hi.mean()) ** 2
        if var > best_var:
            best_var, best = var, t
    return float(10 ** best)


def write_csv(path: Path, rows: list) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys, seen = [], set()
    for r in rows:                      # tolerate heterogeneous row dicts
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"  wrote {rel(path)} ({len(rows)} rows)")


def similarity_residual(proj: np.ndarray, gt: np.ndarray) -> tuple:
    """Residual after the best similarity (scale+rotation+translation) fit.

    If a view becomes consistent under a similarity transform, the *shape* of
    the annotated skeleton matches the projected one and only its placement is
    wrong - the signature of a wrong hand / wrong instance being annotated,
    rather than a broken camera model.
    """
    P, Q = np.asarray(proj, float), np.asarray(gt, float)
    if len(P) < 3:
        return float("nan"), float("nan")
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    denom = (Pc ** 2).sum()
    if denom <= 0:
        return float("nan"), float("nan")
    U, S, Vt = np.linalg.svd(Pc.T @ Qc)
    R = (U @ Vt).T
    scale = S.sum() / denom
    resid = float(np.median(np.linalg.norm(Qc - scale * (Pc @ R.T), axis=1)))
    trans = float(np.median(np.linalg.norm((Q - P) - np.median(Q - P, axis=0), axis=1)))
    return resid, trans


def is_sentinel(gt2d: np.ndarray, mask: np.ndarray) -> bool:
    """True when every annotated joint of this view sits exactly at (0, 0)."""
    if mask.sum() == 0:
        return False
    return bool(np.all(gt2d[mask] == 0.0))
