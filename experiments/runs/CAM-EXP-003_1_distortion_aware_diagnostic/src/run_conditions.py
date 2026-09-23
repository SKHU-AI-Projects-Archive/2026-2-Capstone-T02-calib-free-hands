"""Run conditions B (raw + distortion-aware) and C (GT-undistorted + pinhole).

Condition A (raw + pinhole) is not re-run: CAM-EXP-003's predictions are reused
verbatim, which is both cheaper and guarantees the paired comparison uses
exactly the numbers already published there.

Metric definitions are imported from CAM-EXP-003's ``run_benchmark.to_row`` so
that "relative focal error" means precisely the same thing in both experiments.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "win32")

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from common import (CAM003_SRC, RUN_DIR, load_frames, new_camera_matrix,  # noqa: E402
                    raw_frame_path, read_csv, rel, undist_frame_path, write_csv)

REPO = SRC.parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

sys.path.insert(0, str(CAM003_SRC))
from run_benchmark import to_row  # noqa: E402  (CAM-EXP-003 metric code, reused)


def build(key: str):
    """Adapter + which image cache it consumes + how GT is defined for it."""
    from experiments.src.calibration.anycalib_adapter import AnyCalibAdapter
    from experiments.src.calibration.geocalib_adapter import GeoCalibAdapter
    from experiments.src.calibration.perspective_fields_adapter import (
        CENTERED, UNCENTERED, PerspectiveFieldsAdapter)

    table = {
        # --- condition B: raw image, distortion-aware camera model ----------
        "anycalib_dist_radial": (
            lambda: AnyCalibAdapter(model_id="anycalib_dist", cam_id="radial:2"),
            "raw", "B_RAW_DISTORTION_AWARE"),
        "anycalib_gen_radial": (
            lambda: AnyCalibAdapter(model_id="anycalib_gen", cam_id="radial:2"),
            "raw", "B_RAW_DISTORTION_AWARE"),
        "geocalib_distorted_radial": (
            lambda: GeoCalibAdapter(weights="distorted", camera_model="radial"),
            "raw", "B_RAW_DISTORTION_AWARE"),
        # --- condition C: GT-undistorted image, pinhole camera model --------
        "undist_anycalib_pinhole": (
            lambda: AnyCalibAdapter(model_id="anycalib_pinhole", cam_id="pinhole"),
            "undistorted", "C_GT_UNDISTORTED_PINHOLE"),
        "undist_geocalib_pinhole": (
            lambda: GeoCalibAdapter(weights="pinhole", camera_model="pinhole"),
            "undistorted", "C_GT_UNDISTORTED_PINHOLE"),
        "undist_pf_centered": (
            lambda: PerspectiveFieldsAdapter(version=CENTERED),
            "undistorted", "C_GT_UNDISTORTED_PINHOLE"),
        "undist_pf_uncentered": (
            lambda: PerspectiveFieldsAdapter(version=UNCENTERED),
            "undistorted", "C_GT_UNDISTORTED_PINHOLE"),
    }
    return table[key]


def meta_for(r, source: str) -> dict:
    """Frame metadata with the GT that is correct for this image.

    For the undistorted images the ground truth is K_new, not the original K:
    undistorting at alpha = 0 changes the focal by ~0.77x, so comparing an
    undistorted prediction against the original focal would invent a large
    apparent error (or, with the opposite sign, a fake improvement).
    """
    m = dict(r)
    if source == "undistorted":
        K_new, _ = new_camera_matrix(r)
        m["gt_fx"] = float(K_new[0, 0])
        m["gt_fy"] = float(K_new[1, 1])
        m["gt_cx"] = float(K_new[0, 2])
        m["gt_cy"] = float(K_new[1, 2])
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=0)
    args = ap.parse_args()

    import cv2
    ctor, source, condition = build(args.model)
    frames = load_frames()
    n_all = len(frames)
    if args.count:
        frames = frames[args.start:args.start + args.count]
    print(f"{args.model}: {len(frames)} of {n_all} frames "
          f"(source={source}, condition={condition})", flush=True)

    ad = ctor()
    ad.load()
    rows, t0 = [], time.perf_counter()
    for i, r in enumerate(frames):
        path = raw_frame_path(r) if source == "raw" else undist_frame_path(r)
        img = cv2.imread(str(path), cv2.IMREAD_COLOR) if path.exists() else None
        meta = meta_for(r, source)
        if img is None:
            rows.append({"model": args.model, "sequence": r["sequence"],
                         "camera": r["camera"], "frame": int(r["frame"]),
                         "success": 0, "failure_reason": "image not in cache",
                         "condition": condition, "image_source": source})
            continue
        pred = ad.predict(img, meta)
        row = to_row(pred, meta)
        row["condition"] = condition
        row["image_source"] = source
        row["model_key"] = args.model
        if source == "undistorted":
            row["orig_gt_fx"] = float(r["gt_fx"])
            row["gt_fx_is_K_new"] = 1
        rows.append(row)
        if (i + 1) % 300 == 0:
            print(f"  {i + 1}/{len(frames)} ({time.perf_counter() - t0:.0f}s)", flush=True)

    out = (RUN_DIR / "results" / "raw" / "_parts" /
           f"{args.model}_{args.start:05d}.csv.gz") if args.count else \
          (RUN_DIR / "results" / "raw" / f"{args.model}_predictions.csv.gz")
    write_csv(out, rows)
    n_ok = sum(1 for r in rows if r.get("success") == 1)
    print(f"[{args.model}] {n_ok}/{len(rows)} succeeded in "
          f"{time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
