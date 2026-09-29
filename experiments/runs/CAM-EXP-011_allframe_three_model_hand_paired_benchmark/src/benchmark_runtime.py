"""PHASE A — measure per-model throughput before committing to all-frame runs.

Runtime is measured so the plan can be sized, NOT so frames can be dropped.
If a model is slow the answer is caching and batching, never sampling. Any
compute decision is recorded here, before any focal result exists.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "win32")

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from c011_common import MODELS, REPO, SUM, usable_cameras, write_csv, write_json  # noqa

N_SMOKE = 60


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
    ap.add_argument("--models", default="ANYCALIB,GEOCALIB,PERSPECTIVE_FIELDS")
    ap.add_argument("--n", type=int, default=N_SMOKE)
    args = ap.parse_args()

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import cv2
    import torch
    from experiments.src.datasets.gigahands import takes

    cams = [k for k, v in usable_cameras().items() if v["usable"]]
    take_by_name = {t.name: t for t in takes()}
    seq, cam = sorted(cams)[0]
    vid = take_by_name[seq].video_path(cam)
    cap = cv2.VideoCapture(str(vid))
    frames = []
    while len(frames) < args.n:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    cap.release()
    print("smoke frames:", len(frames), "from", seq, cam, flush=True)

    total_scene = 52423
    rows = []
    for name in args.models.split(","):
        key = MODELS[name]["adapter"]
        t0 = time.time()
        ad = build_adapter(key)
        load_s = time.time() - t0
        meta = {"sequence": seq, "camera": cam, "frame": 0,
                "image_width": frames[0].shape[1],
                "image_height": frames[0].shape[0]}
        ad.predict(frames[0], meta)                # warm up
        t1 = time.time()
        ok = 0
        for i, img in enumerate(frames):
            meta["frame"] = i
            p = ad.predict(img, meta)
            ok += int(getattr(p, "success", True))
        dt = time.time() - t1
        fps = len(frames) / dt
        rows.append({
            "model": name, "adapter": key,
            "historical_key": MODELS[name]["historical_key"],
            "load_seconds": round(load_s, 2),
            "frames": len(frames), "successes": ok,
            "seconds_per_frame": round(dt / len(frames), 4),
            "frames_per_second": round(fps, 2),
            "estimated_hours_for_all_frames":
                round(total_scene / fps / 3600, 2),
        })
        print("%-20s %.3f s/frame  %.2f fps  -> %.2f h for %d frames"
              % (name, dt / len(frames), fps,
                 total_scene / fps / 3600, total_scene), flush=True)
        del ad
        torch.cuda.empty_cache()

    total_h = sum(r["estimated_hours_for_all_frames"] for r in rows)
    write_csv(SUM / "runtime_benchmark.csv", rows)
    write_json(SUM / "runtime_benchmark.json", {
        "smoke_frames": len(frames),
        "total_scene_input_frames": total_scene,
        "per_model": rows,
        "estimated_total_hours_three_models": round(total_h, 2),
        "gpu": (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "cpu"),
        "policy": "Runtime is measured to size the plan. Frames are NOT "
                  "reduced because a model is slow; all usable RGB frames are "
                  "processed.",
    })
    print("\nTOTAL estimated: %.2f h for three models over %d frames"
          % (total_h, total_scene))


if __name__ == "__main__":
    main()
