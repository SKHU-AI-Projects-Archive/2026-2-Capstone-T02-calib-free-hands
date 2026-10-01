"""Freeze the nested N = 1, 2, 4, 8, 16 frame grid for CAM-EXP-006.

The grid is a strict subset of CAM-EXP-004.1's frozen 64-frame manifest, and is
nested: the N = 1 frame is in N = 2, which is in N = 4, and so on. Frames are
ordered by the farthest-point insertion order already frozen in that manifest
(the `grid_pos` column), so nothing here is chosen by looking at a result.

Selection inputs: the frozen 64-frame grid and the frame index only. No focal
error, no solver output and no GT focal took any part.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CAMERA_MANIFEST, FRAMES64, MANIFESTS, read_csv, sha256,  # noqa: E402
                    write_csv, write_json)

OUT = MANIFESTS / "cam_exp_006_reference_hand_frames_v1.csv.gz"
COUNTS = (1, 2, 4, 8, 16)


def main() -> None:
    cam_ok = {(r["sequence"], r["camera"]) for r in read_csv(CAMERA_MANIFEST)
              if r["usable_for_camera_benchmark"] == "1"}

    by_view = defaultdict(list)
    for r in read_csv(FRAMES64):
        key = (r["sequence"], r["camera"])
        if key in cam_ok:
            by_view[key].append(r)

    rows, nested_ok = [], True
    for (seq, cam), rs in sorted(by_view.items()):
        rs.sort(key=lambda r: int(r.get("grid_pos", r["frame"])))
        order = [int(r["frame"]) for r in rs]
        sets = {n: set(order[:n]) for n in COUNTS}
        for a, b in zip(COUNTS, COUNTS[1:]):
            if not sets[a] <= sets[b]:
                nested_ok = False
        for pos, f in enumerate(order[:max(COUNTS)]):
            rows.append({
                "sequence": seq, "camera": cam, "frame": f,
                "grid_pos": pos,
                **{f"in_n{n}": int(pos < n) for n in COUNTS},
            })

    write_csv(OUT, rows)
    meta = {
        "manifest": OUT.name,
        "n_views": len(by_view),
        "frame_counts": list(COUNTS),
        "rows": len(rows),
        "nesting_verified": nested_ok,
        "source_grid": FRAMES64.name,
        "ordering": "farthest-point insertion order frozen in CAM-EXP-004.1 "
                    "(grid_pos)",
        "selection_independence": "frames were ordered before any CAM-006 "
                                  "solver ran; no focal error, no GT focal and "
                                  "no solver output took any part",
        "sha256": sha256(OUT),
    }
    write_json(MANIFESTS / "_cam_exp_006_frames_meta.json", meta)
    print(f"{len(by_view)} views, {len(rows)} rows, nesting verified: "
          f"{nested_ok}")


if __name__ == "__main__":
    main()
