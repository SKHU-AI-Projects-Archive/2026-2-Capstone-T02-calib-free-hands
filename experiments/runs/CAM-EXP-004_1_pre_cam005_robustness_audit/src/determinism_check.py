"""Run-to-run determinism of each model on a byte-identical image.

This was not planned. It was forced by the equivalence check: re-running the
same 8 frames through GeoCalib did not reproduce CAM-EXP-003.1's numbers, and
before blaming the environment the input had to be ruled out. It was: the frame
decoded from the video is bit-identical to the cached PNG CAM-EXP-003.1 used
(max abs pixel difference 0).

So the question becomes whether the models themselves are deterministic.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "win32")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FRAMES64_CSV, REPO, RUN_DIR, read_csv, write_csv  # noqa: E402

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CACHE = (RUN_DIR.parent / "CAM-EXP-003_single_frame_calibration_benchmark"
         / "cache" / "frames")
N_FRAMES = 40
N_REPEATS = 3
SEED = 20260923


def main() -> None:
    import cv2
    from experiments.src.calibration.anycalib_adapter import AnyCalibAdapter
    from experiments.src.calibration.geocalib_adapter import GeoCalibAdapter

    rows = [r for r in read_csv(FRAMES64_CSV) if r["reused_from_cam_exp_003"] == "1"]
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(rows), size=N_FRAMES * 3, replace=False)

    imgs = []
    for i in pick:
        r = rows[int(i)]
        p = CACHE / f"{r['sequence']}__{r['camera']}__{int(r['frame']):06d}.png"
        if not p.exists():
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        imgs.append((r, img))
        if len(imgs) >= N_FRAMES:
            break
    print(f"{len(imgs)} frames loaded from the CAM-EXP-003 cache")

    models = {
        "anycalib_gen_radial": AnyCalibAdapter(model_id="anycalib_gen",
                                               cam_id="radial:2"),
        "geocalib_distorted_radial": GeoCalibAdapter(weights="distorted",
                                                     camera_model="radial"),
    }
    for m in models.values():
        m.load()

    out = []
    for mk, ad in models.items():
        for r, img in imgs:
            meta = {k: float(r[k]) for k in ("gt_fx", "gt_fy", "gt_cx", "gt_cy",
                                             "image_width", "image_height")}
            vals = np.array([ad.predict(img, meta).fx_px for _ in range(N_REPEATS)],
                            dtype=float)
            out.append({
                "model": mk, "sequence": r["sequence"], "camera": r["camera"],
                "frame": int(r["frame"]), "n_repeats": N_REPEATS,
                "fx_values": ";".join(f"{v:.4f}" for v in vals),
                "identical": int(np.all(vals == vals[0])),
                "spread_pct": float((vals.max() - vals.min()) / np.median(vals) * 100),
                "cv_pct": float(vals.std() / vals.mean() * 100),
                "gt_fx": float(r["gt_fx"]),
            })
    write_csv(RUN_DIR / "results" / "raw" / "model_determinism_repeats.csv.gz", out)

    summ = []
    for mk in models:
        d = [r for r in out if r["model"] == mk]
        sp = np.array([r["spread_pct"] for r in d])
        summ.append({
            "model": mk, "n_frames": len(d), "n_repeats": N_REPEATS,
            "pct_frames_bit_identical": float(np.mean([r["identical"] for r in d]) * 100),
            "median_spread_pct": float(np.median(sp)),
            "mean_spread_pct": float(sp.mean()),
            "p90_spread_pct": float(np.percentile(sp, 90)),
            "max_spread_pct": float(sp.max()),
            "pct_frames_spread_over_1pct": float((sp > 1).mean() * 100),
            "pct_frames_spread_over_5pct": float((sp > 5).mean() * 100),
            "verdict": ("DETERMINISTIC" if np.all(sp == 0)
                        else "NON_DETERMINISTIC"),
        })
    write_csv(RUN_DIR / "results" / "summary" / "model_determinism_check.csv", summ)
    for s in summ:
        print(f"  {s['model']:26s} {s['verdict']:18s} identical "
              f"{s['pct_frames_bit_identical']:5.1f}%  spread med "
              f"{s['median_spread_pct']:6.2f}%  p90 {s['p90_spread_pct']:6.2f}%  "
              f"max {s['max_spread_pct']:8.2f}%")


if __name__ == "__main__":
    main()
