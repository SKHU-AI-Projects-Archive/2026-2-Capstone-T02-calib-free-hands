"""Build the ALL_COMMON candidate frame manifest. TARGET-BLIND.

Eligibility uses pre-existing QC and data availability ONLY. No focal error,
no reference focal and no fusion outcome takes any part.
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CAMERA_MANIFEST, MANIFESTS, QC_MANIFEST, SUM, read_csv,
                    sha256, write_csv, write_json)

OUT = MANIFESTS / "cam_exp_008_candidate_frames_v1.csv.gz"


def main() -> None:
    cam_ok = {(r["sequence"], r["camera"]) for r in read_csv(CAMERA_MANIFEST)
              if r["usable_for_camera_benchmark"] == "1"}
    qc = read_csv(QC_MANIFEST)

    # a frame is a candidate if at least one hand passes the existing QC
    hands = defaultdict(list)
    for r in qc:
        key = (r["sequence"], r["camera"], int(r["frame"]))
        if key[:2] not in cam_ok:
            continue
        if r["qc_status"] not in ("PASS_STRICT", "PASS_SINGLE_HAND"):
            continue
        if r["zero_pattern"] == "1" or r["chosen"] != "1":
            continue
        hands[key].append(r["hand"])

    rows = []
    for (seq, cam, fr), hs in sorted(hands.items()):
        rows.append({"sequence": seq, "camera": cam, "frame": fr,
                     "n_qc_hands": len(hs),
                     "hands": "+".join(sorted(set(hs)))})
    write_csv(OUT, rows)

    by = defaultdict(list)
    for r in rows:
        by[(r["sequence"], r["camera"])].append(r["frame"])
    n = [len(v) for v in by.values()]
    cov = [{"sequence": s, "camera": c, "n_candidate_frames": len(v),
            "first_frame": min(v), "last_frame": max(v)}
           for (s, c), v in sorted(by.items())]
    write_csv(SUM / "candidate_frame_coverage.csv", cov)
    write_json(SUM / "candidate_manifest_meta.json", {
        "manifest": OUT.name, "sha256": sha256(OUT),
        "rows": len(rows), "views": len(by),
        "physical_cameras": len({c for _, c in by}),
        "sequences": len({s for s, _ in by}),
        "frames_per_view": {"min": int(min(n)), "median": int(np.median(n)),
                            "max": int(max(n)), "total": int(sum(n))},
        "eligibility": "existing QC pass (PASS_STRICT or PASS_SINGLE_HAND), "
                       "chosen, not the all-zero 2D pattern, on a camera "
                       "marked usable_for_camera_benchmark",
        "target_blind": True,
    })
    print(f"candidate frames {len(rows)}, views {len(by)}, "
          f"cameras {len({c for _, c in by})}, "
          f"frames/view median {int(np.median(n))}")


if __name__ == "__main__":
    main()
