"""Run AnyCalib-gen and GeoCalib-distorted on the frozen 64-frame manifest.

Design choices that matter:

* one pass per view: the video is decoded once, the frames are held in memory,
  and BOTH models see them. This avoids a ~10 GB lossless frame cache and halves
  the decode cost.
* all 64 frames are re-inferred, including the 8 that CAM-EXP-003.1 already ran.
  Re-running them is not waste: comparing the new predictions on those 8 frames
  against the stored ones is the reproducibility equivalence check.
* resumable: one part file per view, skipped if already present.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "win32")

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from common import FRAMES64_CSV, REPO, RUN_DIR, read_csv, write_csv  # noqa: E402

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Reuse CAM-EXP-003's metric code verbatim so "relative focal error" keeps the
# same meaning. It has its own `common` module, so it is imported with that
# directory first on the path and our own `common` temporarily shadowed.
CAM003_SRC = (RUN_DIR.parent / "CAM-EXP-003_single_frame_calibration_benchmark"
              / "src")


def _load_to_row():
    import importlib
    saved = sys.modules.pop("common", None)
    sys.path.insert(0, str(CAM003_SRC))
    try:
        mod = importlib.import_module("run_benchmark")
        return mod.to_row
    finally:
        sys.path.remove(str(CAM003_SRC))
        sys.modules.pop("common", None)
        if saved is not None:
            sys.modules["common"] = saved


to_row = _load_to_row()

PARTS = RUN_DIR / "results" / "raw" / "_parts64"


def build_models():
    from experiments.src.calibration.anycalib_adapter import AnyCalibAdapter
    from experiments.src.calibration.geocalib_adapter import GeoCalibAdapter
    a = AnyCalibAdapter(model_id="anycalib_gen", cam_id="radial:2")
    g = GeoCalibAdapter(weights="distorted", camera_model="radial")
    a.load()
    g.load()
    return {"anycalib_gen_radial": a, "geocalib_distorted_radial": g}


def decode(video_path: Path, wanted: list[int]):
    """Sequential decode, keeping only the wanted frame indices."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}
    want = set(wanted)
    hi = max(want)
    out, i = {}, 0
    while i <= hi:
        ok, frame = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = frame
        i += 1
    cap.release()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True,
                    help="path to the pre-registered view list (csv)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=0)
    ap.add_argument("--budget-seconds", type=float, default=480.0,
                    help="stop cleanly before the harness timeout")
    args = ap.parse_args()

    sel = [(r["sequence"], r["camera"]) for r in read_csv(Path(args.views))]
    if args.count:
        sel = sel[args.start:args.start + args.count]
    sel_set = set(sel)

    rows_by_view = defaultdict(list)
    for r in read_csv(FRAMES64_CSV):
        k = (r["sequence"], r["camera"])
        if k in sel_set:
            rows_by_view[k].append(r)

    todo = [k for k in sel
            if not (PARTS / f"{k[0]}__{k[1]}.csv.gz").exists()]
    print(f"{len(sel)} views selected, {len(todo)} still to do", flush=True)
    if not todo:
        return

    models = build_models()
    t0 = time.perf_counter()
    done = 0
    for key in todo:
        if time.perf_counter() - t0 > args.budget_seconds:
            print(f"budget reached after {done} views", flush=True)
            break
        rs = sorted(rows_by_view[key], key=lambda r: int(r["grid_pos"]))
        vid = REPO / rs[0]["video"]
        frames = [int(r["frame"]) for r in rs]
        imgs = decode(vid, frames)
        out = []
        for r in rs:
            f = int(r["frame"])
            img = imgs.get(f)
            meta = dict(r)
            for c in ("gt_fx", "gt_fy", "gt_cx", "gt_cy", "image_width",
                      "image_height"):
                meta[c] = float(r[c])
            if img is None:
                out.append({"model_key": "", "sequence": key[0], "camera": key[1],
                            "frame": f, "success": 0,
                            "failure_reason": "frame not decodable"})
                continue
            for mk, ad in models.items():
                row = to_row(ad.predict(img, meta), meta)
                row["model_key"] = mk
                row["grid_pos"] = int(r["grid_pos"])
                row["in_n8"] = int(r["in_n8"])
                row["in_n16"] = int(r["in_n16"])
                row["in_n32"] = int(r["in_n32"])
                row["reused_from_cam_exp_003"] = int(r["reused_from_cam_exp_003"])
                out.append(row)
        write_csv(PARTS / f"{key[0]}__{key[1]}.csv.gz", out)
        done += 1
        el = time.perf_counter() - t0
        print(f"  [{done}/{len(todo)}] {key[0]}/{key[1]} "
              f"{len(rs)} frames  {el:.0f}s total ({el / done:.1f}s/view)",
              flush=True)


if __name__ == "__main__":
    main()
