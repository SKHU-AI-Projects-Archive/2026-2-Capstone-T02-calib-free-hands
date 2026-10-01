"""Build the GT-undistorted frame cache (condition C).

This is an ORACLE preprocessing step: it needs the ground-truth distortion, so
it can never be part of a deployable pipeline. Its only purpose is to answer
"if the lens distortion were removed, how much of the pinhole models' focal
error would disappear?".

For every frame the exact undistortion bookkeeping is stored, because the
easiest way to fake an improvement here is to undistort and then compare
against the ORIGINAL focal instead of the new one.
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from common import (RUN_DIR, UNDIST_CACHE, UNDISTORT_ALPHA, gt_K_D, load_frames,
                    new_camera_matrix, raw_frame_path, rel, undist_frame_path,
                    write_csv)


def main() -> None:
    UNDIST_CACHE.mkdir(parents=True, exist_ok=True)
    frames = load_frames()
    rows, n_new = [], 0
    for i, r in enumerate(frames):
        src = raw_frame_path(r)
        dst = undist_frame_path(r)
        K, D = gt_K_D(r)
        K_new, roi = new_camera_matrix(r)
        W, H = int(r["image_width"]), int(r["image_height"])
        if not dst.exists():
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if img is None:
                continue
            und = cv2.undistort(img, K, D, None, K_new)
            cv2.imwrite(str(dst), und, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            n_new += 1
        rows.append({
            "sequence": r["sequence"], "camera": r["camera"], "frame": int(r["frame"]),
            "image_width": W, "image_height": H,
            "orig_fx": float(r["gt_fx"]), "orig_fy": float(r["gt_fy"]),
            "orig_cx": float(r["gt_cx"]), "orig_cy": float(r["gt_cy"]),
            "gt_k1": float(r.get("gt_k1") or 0.0), "gt_k2": float(r.get("gt_k2") or 0.0),
            "gt_p1": float(r.get("gt_p1") or 0.0), "gt_p2": float(r.get("gt_p2") or 0.0),
            "undistort_alpha": UNDISTORT_ALPHA,
            "new_fx": round(float(K_new[0, 0]), 6), "new_fy": round(float(K_new[1, 1]), 6),
            "new_cx": round(float(K_new[0, 2]), 6), "new_cy": round(float(K_new[1, 2]), 6),
            "new_fx_over_orig_fx": round(float(K_new[0, 0] / float(r["gt_fx"])), 6),
            "valid_roi": str(tuple(int(v) for v in roi)),
            "output_width": W, "output_height": H,
            "cropped": 0,
            "undistorted_png": rel(dst),
            "note": ("cv2.undistort with GT K,D and K_new from "
                     "getOptimalNewCameraMatrix(alpha=0, same size); the GT focal for "
                     "this image is new_fx/new_fy, NOT the original"),
        })
        if (i + 1) % 300 == 0:
            print(f"  {i + 1}/{len(frames)}", flush=True)
    write_csv(RUN_DIR / "results" / "raw" / "gt_undistortion_bookkeeping.csv.gz", rows)
    ratios = np.array([r["new_fx_over_orig_fx"] for r in rows])
    summary = {"n_frames": len(rows), "n_newly_written": n_new,
               "alpha": UNDISTORT_ALPHA,
               "new_fx_over_orig_fx": {
                   "median": round(float(np.median(ratios)), 5),
                   "min": round(float(ratios.min()), 5),
                   "max": round(float(ratios.max()), 5)}}
    (RUN_DIR / "results" / "summary" / "_undistortion_meta.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
