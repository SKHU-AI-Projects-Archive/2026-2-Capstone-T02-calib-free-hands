"""Freeze the 64-frame-per-view manifest, with nested 8 < 16 < 32 < 64 sets.

Selection rules, all data-blind:

* the candidate grid is `np.linspace(0, len(union_sorted)-1, 64)`, i.e. the same
  uniform temporal grid CAM-EXP-003 used, refined 8x. Because 63 = 9 x 7, the
  grid positions 0, 9, 18, ... 63 land on exactly the same real values as
  CAM-EXP-003's `linspace(..., 8)`, so **the 8 existing frames are an exact
  subset of the 64** and their predictions can be reused.
* the 16- and 32-frame sets are grown from the frozen 8 by farthest-point
  insertion on that grid (always add the candidate furthest from everything
  already chosen; ties break to the lower index). This is deterministic, needs
  no data, and guarantees 8 < 16 < 32 < 64 while keeping the spacing as even as
  the constraint allows.

Nothing about the GT focal, any model prediction, hand annotation or scene
content enters the choice.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (FRAMES64_CSV, FRAMES8_CSV, REPO, RUN_DIR, rel,  # noqa: E402
                    read_csv, sha256, write_csv, write_json)

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

N_MAX = 64
NESTED = (8, 16, 32, 64)


def farthest_point_sets(n_grid: int, seed_positions: list[int]):
    """Return {N: sorted list of grid positions} with each set nested in the next."""
    chosen = sorted(seed_positions)
    out = {len(chosen): list(chosen)}
    remaining = [p for p in range(n_grid) if p not in set(chosen)]
    for target in NESTED[1:]:
        while len(chosen) < target and remaining:
            dists = [min(abs(p - c) for c in chosen) for p in remaining]
            best = int(np.argmax(dists))          # ties -> lowest index
            chosen.append(remaining.pop(best))
            chosen.sort()
        out[target] = list(chosen)
    return out


def main() -> None:
    from experiments.src.datasets.gigahands import takes

    old = read_csv(FRAMES8_CSV)
    old_by_view = {}
    for r in old:
        old_by_view.setdefault((r["sequence"], r["camera"]), []).append(int(r["frame"]))

    take_by_name = {t.name: t for t in takes()}
    rows, checks = [], []
    for (seq, cam), old_frames in sorted(old_by_view.items()):
        take = take_by_name[seq]
        frames = take.union_sorted
        grid = np.unique(np.linspace(0, len(frames) - 1, num=N_MAX).astype(int))
        n_grid = grid.size
        # the CAM-EXP-003 frames, located on this finer grid
        old_set = sorted(set(old_frames))
        seed_pos = [p for p in range(n_grid)
                    if int(frames[grid[p]]) in set(old_set)]
        sets = farthest_point_sets(n_grid, seed_pos)
        exact_superset = set(old_set).issubset({int(frames[grid[p]])
                                                for p in range(n_grid)})
        checks.append({"sequence": seq, "camera": cam,
                       "n_valid_frames": int(len(frames)), "n_grid": int(n_grid),
                       "n_old_frames": len(old_set),
                       "n_old_recovered_on_grid": len(seed_pos),
                       "old_8_is_exact_subset_of_64": int(exact_superset),
                       "n_8": len(sets[8]), "n_16": len(sets[16]),
                       "n_32": len(sets[32]), "n_64": len(sets[64]),
                       "nested_ok": int(all(set(sets[a]).issubset(set(sets[b]))
                                            for a, b in zip(NESTED, NESTED[1:])))})
        src = {r["camera"]: r for r in old if r["sequence"] == seq}[cam]
        for p in range(n_grid):
            f = int(frames[grid[p]])
            rows.append({
                "sequence": seq, "take": src["take"], "camera": cam,
                "frame": f, "grid_pos": p,
                "in_n8": int(p in set(sets[8])), "in_n16": int(p in set(sets[16])),
                "in_n32": int(p in set(sets[32])), "in_n64": 1,
                "reused_from_cam_exp_003": int(f in set(old_set)),
                "video": src["video"],
                "image_width": src["image_width"], "image_height": src["image_height"],
                "gt_fx": src["gt_fx"], "gt_fy": src["gt_fy"],
                "gt_cx": src["gt_cx"], "gt_cy": src["gt_cy"],
                "gt_k1": src["gt_k1"], "gt_k2": src["gt_k2"],
                "gt_p1": src["gt_p1"], "gt_p2": src["gt_p2"],
            })

    write_csv(FRAMES64_CSV, rows)
    write_csv(RUN_DIR / "tables" / "manifest64_nesting_check.csv", checks)
    digest = sha256(FRAMES64_CSV)
    write_json(RUN_DIR / "results" / "summary" / "_manifest64_meta.json", {
        "path": rel(FRAMES64_CSV), "sha256": digest, "n_rows": len(rows),
        "n_views": len(checks),
        "grid": "np.linspace(0, n_valid_frames-1, 64).astype(int), deduplicated",
        "nested_sets": "8 < 16 < 32 < 64 by farthest-point insertion seeded with "
                       "the frozen CAM-EXP-003 8-frame set",
        "views_where_old_8_is_exact_subset":
            sum(c["old_8_is_exact_subset_of_64"] for c in checks),
        "views_with_full_nesting": sum(c["nested_ok"] for c in checks),
        "selection_inputs": "video length only - no GT, no predictions, no hand "
                            "annotation, no scene content",
    })
    n_new = sum(1 for r in rows if not r["reused_from_cam_exp_003"])
    print(f"wrote {len(rows)} rows over {len(checks)} views; sha256 {digest[:16]}")
    print(f"  old-8 exact subset in {sum(c['old_8_is_exact_subset_of_64'] for c in checks)}"
          f"/{len(checks)} views; nesting ok in "
          f"{sum(c['nested_ok'] for c in checks)}/{len(checks)}")
    print(f"  frames needing NEW inference: {n_new}")
    print(f"  grid sizes: {sorted({c['n_grid'] for c in checks})}")


if __name__ == "__main__":
    main()
