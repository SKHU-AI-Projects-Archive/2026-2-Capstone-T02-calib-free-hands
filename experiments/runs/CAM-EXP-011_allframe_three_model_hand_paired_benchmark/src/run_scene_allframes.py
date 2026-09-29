"""PHASE B — run a scene model over EVERY usable RGB frame.

Frames are decoded sequentially per video (not seeked), which is the dominant
cost saving. Predictions are cached per video so the run is resumable and so
aggregation can be revisited without re-running any model.

A frame where the model fails is written with `valid=0` and a reason. It is
never silently dropped: the failure rate is itself a result.

Reference focal is never read here.

    PYTHONPATH=<repo> .venv-calib/Scripts/python src/run_scene_allframes.py \
        --model ANYCALIB
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "win32")

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from c011_common import (CACHE, MANIFESTS, MODELS, RAW, REPO, read_csv,  # noqa: E402
                    scalar_focal, write_csv)

CACHE_DIR = {"ANYCALIB": "anycalib", "GEOCALIB": "geocalib",
             "PERSPECTIVE_FIELDS": "perspective_fields"}


def build_adapter(key):
    from experiments.src.calibration.anycalib_adapter import AnyCalibAdapter
    from experiments.src.calibration.geocalib_adapter import GeoCalibAdapter
    from experiments.src.calibration.perspective_fields_adapter import (
        CENTERED, PerspectiveFieldsAdapter)
    return {"anycalib": AnyCalibAdapter,
            "geocalib": GeoCalibAdapter,
            "pf_centered": lambda: PerspectiveFieldsAdapter(version=CENTERED),
            }[key]()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--limit-videos", type=int, default=0)
    args = ap.parse_args()

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import cv2
    import numpy as np
    from experiments.src.datasets.gigahands import takes

    out_dir = CACHE / CACHE_DIR[args.model]
    out_dir.mkdir(parents=True, exist_ok=True)

    man = [r for r in read_csv(MANIFESTS / "cam_exp_011_allframe_manifest_v1.csv.gz")
           if r["scene_input"] == "1"]
    by_video = defaultdict(list)
    for r in man:
        by_video[(r["sequence"], r["camera"])].append(int(r["frame"]))
    videos = sorted(by_video)
    if args.limit_videos:
        videos = videos[:args.limit_videos]

    todo = [v for v in videos
            if not (out_dir / ("%s__%s.csv.gz" % v)).exists()]
    print("model %s: %d videos, %d still to do"
          % (args.model, len(videos), len(todo)), flush=True)
    if not todo:
        print("nothing to do")
    else:
        take_by_name = {t.name: t for t in takes()}
        ad = build_adapter(MODELS[args.model]["adapter"])
        t0, done_frames = time.time(), 0
        for vi, (seq, cam) in enumerate(todo, 1):
            want = set(by_video[(seq, cam)])
            vid = take_by_name[seq].video_path(cam)
            cap = cv2.VideoCapture(str(vid))
            rows, idx = [], 0
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                if idx in want:
                    t1 = time.time()
                    try:
                        p = ad.predict(img, {"sequence": seq, "camera": cam,
                                             "frame": idx,
                                             "image_width": img.shape[1],
                                             "image_height": img.shape[0]})
                        fx, fy = p.fx_px, p.fy_px
                        ok_pred = (p.success and fx is not None
                                   and fy is not None
                                   and np.isfinite(fx) and np.isfinite(fy)
                                   and fx > 0 and fy > 0)
                        rows.append({
                            "sequence": seq, "camera": cam, "frame": idx,
                            "focal_pred_px": (scalar_focal(fx, fy)
                                              if ok_pred else ""),
                            "fx_px": fx if fx is not None else "",
                            "fy_px": fy if fy is not None else "",
                            "valid": int(bool(ok_pred)),
                            "failure_reason": ("" if ok_pred
                                               else (p.failure_reason
                                                     or "invalid_focal")),
                            "inference_time_ms": round(
                                (time.time() - t1) * 1000, 2),
                            "source": "NEW_INFERENCE"})
                    except Exception as exc:
                        rows.append({
                            "sequence": seq, "camera": cam, "frame": idx,
                            "focal_pred_px": "", "fx_px": "", "fy_px": "",
                            "valid": 0,
                            "failure_reason": "%s: %s" % (type(exc).__name__,
                                                          exc),
                            "inference_time_ms": "",
                            "source": "NEW_INFERENCE"})
                    done_frames += 1
                idx += 1
            cap.release()
            write_csv(out_dir / ("%s__%s.csv.gz" % (seq, cam)), rows)
            if vi % 5 == 0 or vi == len(todo):
                el = time.time() - t0
                rate = done_frames / max(el, 1e-9)
                left = sum(len(by_video[v]) for v in todo[vi:])
                print("  %3d/%d videos  %6d frames  %.1f min  %.1f fps  "
                      "eta %.1f min"
                      % (vi, len(todo), done_frames, el / 60, rate,
                         left / max(rate, 1e-9) / 60), flush=True)

    # ---- assemble
    allrows = []
    for v in videos:
        p = out_dir / ("%s__%s.csv.gz" % v)
        if p.exists():
            allrows.extend(read_csv(p))
    name = {"ANYCALIB": "anycalib", "GEOCALIB": "geocalib",
            "PERSPECTIVE_FIELDS": "perspective_fields"}[args.model]
    write_csv(RAW / ("%s_allframe_predictions.csv.gz" % name), allrows)
    nv = sum(1 for r in allrows if r["valid"] == "1")
    print("%s: %d rows, %d valid (%.2f%%), %d videos"
          % (args.model, len(allrows), nv,
             100.0 * nv / max(len(allrows), 1), len(videos)))


if __name__ == "__main__":
    main()
