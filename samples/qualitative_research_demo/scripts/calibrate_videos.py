"""Estimate a video-level focal for each clip with the研究's calibration models.

Runs in experiments/.venv-calib. Uses the same adapters, checkpoints and
aggregation rules as CAM-EXP-003.1 / 004, unchanged:

  AnyCalib  anycalib_gen, cam_id "radial:2"   -> median over the frames
  GeoCalib  weights "distorted", camera_model "radial"
  Frozen E2 per experiments/manifests/cam_exp_004_e2_frozen_spec_v1.json:
            per frame sqrt(f_anycalib * f_geocalib), then median over frames

GeoCalib is known to vary between runs on identical input (CAM-EXP-004.1). This
script is run ONCE and its output is used as-is; no run is picked for looking
better than another.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "win32")

import cv2
import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from common import (CALIB, N_CALIB_FRAMES, REPO, videos, write_csv,  # noqa: E402
                    write_json)

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def sample_indices(n_frames: int, k: int) -> list[int]:
    """Uniform over the clip. No content, no GT and no model output is consulted."""
    return sorted(set(np.linspace(0, max(n_frames - 1, 0), k).astype(int).tolist()))


def grab(path: Path, idxs: list[int]):
    cap = cv2.VideoCapture(str(path))
    out = {}
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            out[int(i)] = frame
    cap.release()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="substring filter on the filename")
    ap.add_argument("--frames", type=int, default=N_CALIB_FRAMES)
    ap.add_argument("--skip-geocalib", action="store_true")
    args = ap.parse_args()

    from experiments.src.calibration.anycalib_adapter import AnyCalibAdapter
    any_ad = AnyCalibAdapter(model_id="anycalib_gen", cam_id="radial:2")
    any_ad.load()
    geo_ad = None
    if not args.skip_geocalib:
        try:
            from experiments.src.calibration.geocalib_adapter import GeoCalibAdapter
            geo_ad = GeoCalibAdapter(weights="distorted", camera_model="radial")
            geo_ad.load()
        except Exception as exc:  # recorded, never silently substituted
            print(f"[geocalib] unavailable: {type(exc).__name__}: {exc}")
            geo_ad = None

    for path in videos():
        if args.only and args.only not in path.name:
            continue
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        idxs = sample_indices(n, args.frames)
        frames = grab(path, idxs)
        rows = []
        t0 = time.perf_counter()
        for i in idxs:
            img = frames.get(i)
            if img is None:
                rows.append({"video": path.name, "frame": i, "status": "DECODE_FAILED"})
                continue
            meta = {"image_width": w, "image_height": h}
            r = {"video": path.name, "frame": i, "time_sec": round(i / fps, 3),
                 "status": "ok"}
            pa = any_ad.predict(img, meta)
            r["anycalib_fx_px"] = float(pa.fx_px) if pa.success else float("nan")
            r["anycalib_k1"] = pa.extra.get("k1", "")
            if geo_ad is not None:
                pg = geo_ad.predict(img, meta)
                r["geocalib_fx_px"] = float(pg.fx_px) if pg.success else float("nan")
                r["geocalib_k1"] = pg.extra.get("k1", "")
                if pa.success and pg.success:
                    r["e2_fx_px"] = float(np.sqrt(pa.fx_px * pg.fx_px))
            rows.append(r)
            print(f"  {path.name} f{i:5d}  anycalib={r.get('anycalib_fx_px', float('nan')):8.1f}"
                  f"  geocalib={r.get('geocalib_fx_px', float('nan')):8.1f}"
                  f"  e2={r.get('e2_fx_px', float('nan')):8.1f}", flush=True)
        write_csv(CALIB / f"{path.stem}_calibration_frames.csv", rows)

        def med(key):
            v = np.array([r[key] for r in rows if isinstance(r.get(key), float)
                          and np.isfinite(r.get(key))])
            return float(np.median(v)) if v.size else None

        est = {
            "video": path.name, "width": w, "height": h, "fps": round(fps, 3),
            "n_frames_total": n,
            "calibration_frames": idxs,
            "frame_selection": "uniform over the clip; no content, model output "
                               "or ground truth was consulted",
            "anycalib": {
                "checkpoint": "anycalib_gen", "camera_model": "radial:2",
                "aggregation": "median over the calibration frames "
                               "(the frozen CAM-EXP-004 rule)",
                "video_focal_px": med("anycalib_fx_px"),
            },
            "geocalib": {
                "weights": "distorted", "camera_model": "radial",
                "aggregation": "median over the calibration frames",
                "video_focal_px": med("geocalib_fx_px"),
                "note": "GeoCalib is not run-to-run deterministic on identical "
                        "input (CAM-EXP-004.1). This is one execution, used "
                        "as-is.",
            },
            "e2_frozen": {
                "spec": "experiments/manifests/cam_exp_004_e2_frozen_spec_v1.json",
                "definition": "per frame sqrt(f_anycalib * f_geocalib), then "
                              "median over frames",
                "video_focal_px": med("e2_fx_px"),
                "status": "frozen exploratory ensemble - not a proposed method",
            },
            "caveat": "These clips have NO dataset-provided camera calibration. "
                      "No estimate here may be called a ground-truth, correct or "
                      "true focal; they are model estimates.",
            "what_the_hand_pipeline_consumes": {
                "focal": "yes - only through tz = 2f/(s*box_size)",
                "principal_point": "NOT consumed; rgb_predictor uses the image "
                                   "centre (W/2, H/2)",
                "distortion": "NOT consumed; predicted by the calibration model "
                              "but never propagated into the hand 3D",
            },
        }
        write_json(CALIB / f"{path.stem}_video_estimate.json", est)
        print(f"[{path.name}] anycalib={est['anycalib']['video_focal_px']} "
              f"geocalib={est['geocalib']['video_focal_px']} "
              f"e2={est['e2_frozen']['video_focal_px']} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
