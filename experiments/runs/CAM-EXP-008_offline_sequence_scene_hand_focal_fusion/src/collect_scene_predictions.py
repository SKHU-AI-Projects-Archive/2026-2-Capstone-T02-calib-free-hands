"""Run AnyCalib (anycalib_gen / radial:2) on every candidate frame.

TARGET-BLIND. Records fx, fy, cx, cy, k1, k2 per frame. One decode pass per
video. Resumable per view.
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ANYCALIB_CAM_ID, ANYCALIB_MODEL_ID, CACHE, MANIFESTS,
                    RAW, REPO, read_csv, write_csv, write_json)

FRAMES = MANIFESTS / "cam_exp_008_candidate_frames_v1.csv.gz"
OUTDIR = CACHE / "scene_pred"


def main() -> None:
    sys.path.insert(0, str(REPO))
    import cv2
    from experiments.src.calibration.anycalib_adapter import AnyCalibAdapter
    from experiments.src.datasets.gigahands import takes

    rows = read_csv(FRAMES)
    by = defaultdict(list)
    for r in rows:
        by[(r["sequence"], r["camera"])].append(int(r["frame"]))
    views = sorted(by)

    ad = AnyCalibAdapter(model_id=ANYCALIB_MODEL_ID, cam_id=ANYCALIB_CAM_ID)
    take_by_name = {t.name: t for t in takes()}
    OUTDIR.mkdir(parents=True, exist_ok=True)

    for vi, (seq, cam) in enumerate(views, 1):
        out = OUTDIR / f"{seq}__{cam}.csv.gz"
        if out.exists():
            continue
        want = sorted(by[(seq, cam)])
        take = take_by_name[seq]
        vc = cv2.VideoCapture(str(take.video_path(cam)))
        wanted, recs = set(want), []
        i, hi = 0, max(want)
        while i <= hi:
            ok, frame = vc.read()
            if not ok:
                break
            if i in wanted:
                H, W = frame.shape[:2]
                try:
                    p = ad.predict(frame, {})
                    d = np.asarray(p.distortion, float).reshape(-1) \
                        if p.distortion is not None else np.zeros(2)
                    recs.append({
                        "sequence": seq, "camera": cam, "frame": i,
                        "image_width": W, "image_height": H,
                        "success": int(bool(p.success)),
                        "pred_fx": p.fx_px, "pred_fy": p.fy_px,
                        "pred_cx": p.cx_px, "pred_cy": p.cy_px,
                        "k1": float(d[0]) if d.size > 0 else 0.0,
                        "k2": float(d[1]) if d.size > 1 else 0.0,
                    })
                except Exception as e:                        # noqa: BLE001
                    recs.append({"sequence": seq, "camera": cam, "frame": i,
                                 "image_width": W, "image_height": H,
                                 "success": 0, "pred_fx": "", "pred_fy": "",
                                 "pred_cx": "", "pred_cy": "", "k1": "",
                                 "k2": "",
                                 "failure": f"{type(e).__name__}"})
            i += 1
        vc.release()
        write_csv(out, recs)
        if vi % 5 == 0:
            print(f"  {vi}/{len(views)} views", flush=True)

    allrecs = []
    for (seq, cam) in views:
        f = OUTDIR / f"{seq}__{cam}.csv.gz"
        if f.exists():
            allrecs += read_csv(f)
    write_csv(RAW / "sequence_scene_predictions.csv.gz", allrecs)
    ok = sum(1 for r in allrecs if r["success"] == "1")
    write_json(RAW.parent / "summary" / "scene_prediction_meta.json", {
        "model": f"AnyCalib[{ANYCALIB_MODEL_ID}/{ANYCALIB_CAM_ID}]",
        "frames": len(allrecs), "success": ok,
        "success_rate": round(ok / max(len(allrecs), 1), 4),
        "views": len(views), "target_blind": True,
    })
    print(f"scene predictions {len(allrecs)}, success {ok}")


if __name__ == "__main__":
    main()
