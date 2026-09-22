"""Run every calibration model over the frozen benchmark frames.

Fairness rules enforced here, not left to convention:
  * the frame list is read from the frozen manifest, identical for every model;
  * a model failure is recorded as a row with success=0, never dropped;
  * no per-model frame filtering of any kind exists in this file.

Runs in ``experiments/.venv-calib``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from common import RUN_DIR, load_frames, rel, write_csv  # noqa: E402

REPO = SRC.parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def build_adapters(which: list | None):
    from experiments.src.calibration.anycalib_adapter import AnyCalibAdapter
    from experiments.src.calibration.geocalib_adapter import GeoCalibAdapter
    from experiments.src.calibration.perspective_fields_adapter import (
        CENTERED, UNCENTERED, PerspectiveFieldsAdapter)
    from experiments.src.calibration.fixed_baseline_adapter import (
        DemoFixedFocalAdapter, ImageCenterPrincipalPointAdapter, OracleGTAdapter)

    all_ad = {
        "anycalib": AnyCalibAdapter,
        "geocalib": GeoCalibAdapter,
        "pf_uncentered": lambda: PerspectiveFieldsAdapter(version=UNCENTERED),
        "pf_centered": lambda: PerspectiveFieldsAdapter(version=CENTERED),
        "demo_fixed": DemoFixedFocalAdapter,
        "image_center_pp": ImageCenterPrincipalPointAdapter,
        "oracle_gt": OracleGTAdapter,
    }
    keys = which or list(all_ad)
    return [(k, all_ad[k]) for k in keys if k in all_ad]


def frame_reader(frames):
    """Yield (meta, image) from the PNG frame cache.

    Reading a pre-decoded cache instead of seeking into the videos removes the
    dominant cost of the benchmark and guarantees that every model is given
    bit-identical pixels (see cache_frames.py).
    """
    import cv2
    from cache_frames import frame_path
    for r in frames:
        p = frame_path(r)
        img = cv2.imread(str(p), cv2.IMREAD_COLOR) if p.exists() else None
        yield r, img


def to_row(pred, meta) -> dict:
    gt_fx, gt_fy = float(meta["gt_fx"]), float(meta["gt_fy"])
    gt_cx, gt_cy = float(meta["gt_cx"]), float(meta["gt_cy"])
    W, H = int(meta["image_width"]), int(meta["image_height"])

    def rel_pct(p, g):
        return "" if (p is None or not np.isfinite(p) or g == 0) else round((p - g) / g * 100.0, 6)

    fx_s = rel_pct(pred.fx_px, gt_fx)
    fy_s = rel_pct(pred.fy_px, gt_fy)
    # primary focal metric compares the model's focal against GT fx; fy is kept
    # alongside because the GT is slightly anisotropic and no scalar convention
    # should be invented (see the audit)
    signed = fx_s
    row = {
        "model": pred.model, "sequence": pred.sequence, "camera": pred.camera,
        "frame": pred.frame, "image_width": W, "image_height": H,
        "gt_fx": gt_fx, "gt_fy": gt_fy, "gt_cx": gt_cx, "gt_cy": gt_cy,
        "pred_fx": "" if pred.fx_px is None else round(float(pred.fx_px), 6),
        "pred_fy": "" if pred.fy_px is None else round(float(pred.fy_px), 6),
        "pred_cx": "" if pred.cx_px is None else round(float(pred.cx_px), 6),
        "pred_cy": "" if pred.cy_px is None else round(float(pred.cy_px), 6),
        "pred_hfov": "" if pred.hfov_deg is None or not np.isfinite(pred.hfov_deg)
                     else round(float(pred.hfov_deg), 6),
        "pred_vfov": "" if pred.vfov_deg is None or not np.isfinite(pred.vfov_deg)
                     else round(float(pred.vfov_deg), 6),
        "signed_focal_error_pct": signed,
        "relative_focal_error_pct": "" if signed == "" else round(abs(signed), 6),
        "fx_relative_error_pct": "" if fx_s == "" else round(abs(fx_s), 6),
        "fy_relative_error_pct": "" if fy_s == "" else round(abs(fy_s), 6),
        "log_focal_error": ("" if (pred.fx_px is None or pred.fx_px <= 0)
                            else round(abs(float(np.log(pred.fx_px / gt_fx))), 6)),
        "runtime_ms": round(float(pred.runtime_ms), 3),
        "success": int(bool(pred.success)),
        "failure_reason": pred.failure_reason,
        "raw_output": pred.raw_output,
        "conversion_note": pred.conversion_note,
    }
    # GT fields of view, for angular error
    from experiments.src.calibration.base import fov_from_focal
    gt_h = fov_from_focal(gt_fx, W)
    gt_v = fov_from_focal(gt_fy, H)
    row["gt_hfov"] = round(gt_h, 6)
    row["gt_vfov"] = round(gt_v, 6)
    row["hfov_error_deg"] = ("" if row["pred_hfov"] == ""
                             else round(abs(row["pred_hfov"] - gt_h), 6))
    row["vfov_error_deg"] = ("" if row["pred_vfov"] == ""
                             else round(abs(row["pred_vfov"] - gt_v), 6))
    if pred.cx_px is not None and pred.cy_px is not None:
        dx, dy = pred.cx_px - gt_cx, pred.cy_px - gt_cy
        row["principal_point_error_px"] = round(float(np.hypot(dx, dy)), 4)
        row["principal_point_error_norm"] = round(
            float(np.hypot(dx / W, dy / H)), 6)
    else:
        row["principal_point_error_px"] = "NOT_PREDICTED"
        row["principal_point_error_norm"] = "NOT_PREDICTED"
    for k, v in (pred.extra or {}).items():
        row[f"extra_{k}"] = v
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0, help="first frame index")
    ap.add_argument("--count", type=int, default=0, help="frames this invocation")
    ap.add_argument("--trace", type=int, default=5,
                    help="frames per model written to the numeric trace")
    args = ap.parse_args()

    os.environ.setdefault("PYOPENGL_PLATFORM", "win32")
    frames = load_frames()
    if args.limit:
        frames = frames[:args.limit]
    all_n = len(frames)
    # Bounded slices with part-file checkpoints: this environment terminates
    # very long runs, so an expensive model is processed a chunk at a time and
    # the chunks are concatenated once complete.
    if args.count:
        frames = frames[args.start:args.start + args.count]
    print(f"benchmark frames: {len(frames)} of {all_n} (start={args.start})")

    adapters = build_adapters(args.models)
    all_rows, failures, trace = [], [], []
    availability = []

    # Frames are STREAMED per model rather than all decoded up front: holding
    # 1400 x 1280x720x3 arrays in memory costs ~4 GB and made the run swap.
    # Each model therefore re-decodes, which costs about a minute and keeps the
    # process small. Every model still sees exactly the same frame list.
    n_frames = len(frames)

    for key, ctor in adapters:
        # per-model resume: a completed model is reloaded from disk rather than
        # re-run, so an interrupted benchmark can be continued cheaply
        done_path = RUN_DIR / "results" / "raw" / f"{key}_predictions.csv.gz"
        if done_path.exists() and not args.count:
            from common import read_csv as _rc
            prev = _rc(done_path)
            if len(prev) >= n_frames:
                all_rows.extend(prev)
                n_ok = sum(1 for r in prev if str(r.get("success")) == "1")
                availability.append({"model_key": key,
                                     "model": prev[0].get("model", key),
                                     "status": "RUN_SUCCESS", "note": "resumed",
                                     "n_attempted": len(prev), "n_success": n_ok,
                                     "failure_rate": round(1 - n_ok / max(len(prev), 1), 4),
                                     "total_seconds": ""})
                print(f"[{key}] resumed {len(prev)} rows from disk")
                continue
        try:
            ad = ctor()
            ad.load()
            status = "RUN_SUCCESS"
            note = ""
        except Exception as exc:
            status = "BLOCKED_DEPENDENCY"
            note = f"{type(exc).__name__}: {exc}"
            availability.append({"model_key": key, "status": status, "note": note})
            print(f"[{key}] {status}: {note}")
            continue

        rows, t0, n_ok = [], time.perf_counter(), 0
        for i, (meta, img) in enumerate(frame_reader(frames)):
            if img is None:
                failures.append({"model": ad.name, "sequence": meta["sequence"],
                                 "camera": meta["camera"], "frame": meta["frame"],
                                 "stage": "decode", "reason": "frame decode failed"})
                continue
            pred = ad.predict(img, meta)
            row = to_row(pred, meta)
            rows.append(row)
            n_ok += row["success"]
            if not row["success"]:
                failures.append({"model": ad.name, "sequence": meta["sequence"],
                                 "camera": meta["camera"], "frame": meta["frame"],
                                 "stage": "inference",
                                 "reason": pred.failure_reason or "no focal returned"})
            if len(trace) < args.trace * (adapters.index((key, ctor)) + 1):
                trace.append({
                    "model": ad.name, "sequence": meta["sequence"],
                    "camera": meta["camera"], "frame": meta["frame"],
                    "orig_W": meta["image_width"], "orig_H": meta["image_height"],
                    "raw_output": pred.raw_output,
                    "conversion_note": pred.conversion_note,
                    "converted_fx_px": row["pred_fx"], "converted_fy_px": row["pred_fy"],
                    "gt_fx": meta["gt_fx"], "gt_fy": meta["gt_fy"],
                    "relative_focal_error_pct": row["relative_focal_error_pct"],
                })
            if (i + 1) % 200 == 0:
                el = time.perf_counter() - t0
                print(f"  [{ad.name}] {i + 1}/{n_frames} ({el:.1f}s)", flush=True)
        el = time.perf_counter() - t0
        if args.count:
            part = (RUN_DIR / "results" / "raw" / "_parts" /
                    f"{key}_{args.start:05d}.csv.gz")
            write_csv(part, rows)
        else:
            write_csv(RUN_DIR / "results" / "raw" / f"{key}_predictions.csv.gz", rows)
        all_rows.extend(rows)
        availability.append({
            "model_key": key, "model": ad.name, "status": status, "note": note,
            "n_attempted": len(rows), "n_success": n_ok,
            "failure_rate": round(1 - n_ok / max(len(rows), 1), 4),
            "total_seconds": round(el, 1)})
        print(f"[{key}] {n_ok}/{len(rows)} succeeded in {el:.1f}s")
        del ad

    write_csv(RUN_DIR / "results" / "raw" / "all_predictions.csv.gz", all_rows)
    write_csv(RUN_DIR / "results" / "raw" / "inference_failures.csv", failures)
    write_csv(RUN_DIR / "tables" / "adapter_numeric_trace.csv", trace)
    (RUN_DIR / "results" / "summary" / "_availability_run.json").write_text(
        json.dumps(availability, indent=2), encoding="utf-8")
    print(f"done: {len(all_rows)} prediction rows")


if __name__ == "__main__":
    main()
