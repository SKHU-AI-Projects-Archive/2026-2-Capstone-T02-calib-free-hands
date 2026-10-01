"""FULL_DURATION_64_APPROXIMATION inputs.

Rebuilds the candidate manifest on the frozen CAM-EXP-004.1 64-frame grid and
materialises the scene-prediction cache directly from that run's existing
AnyCalib outputs. NO new scene inference is performed and CAM-EXP-004.1 is
read-only.

TARGET-BLIND.
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CACHE, CAMERA_MANIFEST, FRAMES64, MANIFESTS, PRED64,
                    QC_MANIFEST, SUM, read_csv, sha256, write_csv, write_json)

CAND = MANIFESTS / "cam_exp_008_candidate_frames_v1.csv.gz"
SCENE_DIR = CACHE / "scene_pred"


def main() -> None:
    cam_ok = {(r["sequence"], r["camera"]) for r in read_csv(CAMERA_MANIFEST)
              if r["usable_for_camera_benchmark"] == "1"}
    grid = defaultdict(set)
    for r in read_csv(FRAMES64):
        k = (r["sequence"], r["camera"])
        if k in cam_ok:
            grid[k].add(int(r["frame"]))

    hands = defaultdict(list)
    for r in read_csv(QC_MANIFEST):
        k = (r["sequence"], r["camera"])
        if k not in cam_ok or int(r["frame"]) not in grid[k]:
            continue
        if r["qc_status"] not in ("PASS_STRICT", "PASS_SINGLE_HAND"):
            continue
        if r["zero_pattern"] == "1" or r["chosen"] != "1":
            continue
        hands[(k[0], k[1], int(r["frame"]))].append(r["hand"])

    rows = [{"sequence": s, "camera": c, "frame": f, "n_qc_hands": len(h),
             "hands": "+".join(sorted(set(h)))}
            for (s, c, f), h in sorted(hands.items())]
    write_csv(CAND, rows)

    # scene predictions: reuse CAM-EXP-004.1's existing AnyCalib outputs
    pred = defaultdict(list)
    for r in read_csv(PRED64):
        if r["model_key"] != "anycalib_gen_radial":
            continue
        pred[(r["sequence"], r["camera"])].append(r)
    SCENE_DIR.mkdir(parents=True, exist_ok=True)
    keep = {(r["sequence"], r["camera"]) for r in rows}
    frames_by = defaultdict(set)
    for r in rows:
        frames_by[(r["sequence"], r["camera"])].add(r["frame"])
    nsc = 0
    for k, prs in pred.items():
        if k not in keep:
            continue
        recs = [{"sequence": k[0], "camera": k[1], "frame": int(p["frame"]),
                 "image_width": p["image_width"],
                 "image_height": p["image_height"],
                 "success": p["success"], "pred_fx": p["pred_fx"],
                 "pred_fy": p["pred_fy"], "pred_cx": p["pred_cx"],
                 "pred_cy": p["pred_cy"], "k1": p["extra_k1"],
                 "k2": p["extra_k2"]}
                for p in prs if int(p["frame"]) in frames_by[k]]
        write_csv(SCENE_DIR / f"{k[0]}__{k[1]}.csv.gz", recs)
        nsc += len(recs)

    by = defaultdict(list)
    for r in rows:
        by[(r["sequence"], r["camera"])].append(r["frame"])
    n = [len(v) for v in by.values()]
    write_json(SUM / "candidate_manifest_meta.json", {
        "mode": "FULL_DURATION_64_APPROXIMATION",
        "manifest": CAND.name, "sha256": sha256(CAND),
        "rows": len(rows), "views": len(by),
        "physical_cameras": len({c for _, c in by}),
        "sequences": len({s for s, _ in by}),
        "frames_per_view": {"min": int(min(n)), "median": int(np.median(n)),
                            "max": int(max(n)), "total": int(sum(n))},
        "scene_prediction_rows_reused": nsc,
        "scene_prediction_source": "CAM-EXP-004.1 extended_frame_predictions "
                                   "(anycalib_gen_radial), reused read-only; "
                                   "no new inference",
        "eligibility": "existing QC pass, chosen, not the all-zero pattern, on "
                       "a camera marked usable_for_camera_benchmark, "
                       "intersected with the frozen 64-frame grid",
        "target_blind": True,
    })
    print(f"candidate frames {len(rows)}, views {len(by)}, "
          f"cameras {len({c for _, c in by})}, "
          f"median frames/view {int(np.median(n))}, scene rows {nsc}")


if __name__ == "__main__":
    main()
