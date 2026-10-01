"""Sanity test: are the GT intrinsics really constant within one (sequence, camera)?

The whole premise of multi-frame aggregation is that the camera does not change
during a take. If the GT itself varied per frame there would be nothing to
aggregate towards, so this is checked before any aggregation is run.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BENCH_CSV, FRAMES_CSV, RUN_DIR, read_csv, write_csv  # noqa: E402

PARAMS = ("gt_fx", "gt_fy", "gt_cx", "gt_cy", "gt_k1", "gt_k2", "gt_p1", "gt_p2")


def check(rows, source: str) -> list[dict]:
    by_view = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["sequence"], r["camera"])
        for p in PARAMS:
            if p in r and r[p] not in ("", None):
                by_view[key][p].append(float(r[p]))
    out = []
    for (seq, cam), d in sorted(by_view.items()):
        row = {"source": source, "sequence": seq, "camera": cam,
               "n_frames": max(len(v) for v in d.values())}
        constant = True
        for p in PARAMS:
            vals = np.asarray(d.get(p, []), dtype=float)
            if vals.size == 0:
                row[f"{p}_status"] = "ABSENT"
                continue
            spread = float(vals.max() - vals.min())
            row[f"{p}_value"] = round(float(vals[0]), 8)
            row[f"{p}_max_abs_spread"] = spread
            if spread > 0:
                constant = False
        row["all_constant"] = int(constant)
        out.append(row)
    return out


def main() -> None:
    rows = check(read_csv(FRAMES_CSV), "cam_exp_003_frames_manifest")
    n_bad = sum(1 for r in rows if not r["all_constant"])
    print(f"frame manifest: {len(rows)} views, {n_bad} with non-constant GT")

    # The camera-benchmark manifest carries usability flags rather than
    # intrinsics, so it cannot be checked for constancy. What it can confirm is
    # that every view we aggregate is camera-usable, and that the filter used is
    # the camera one - hand annotation quality must not gate a calibration view.
    if BENCH_CSV.exists():
        bench = {(r["sequence"], r["camera"]): r for r in read_csv(BENCH_CSV)}
        n_usable = sum(1 for r in bench.values()
                       if r.get("usable_for_camera_benchmark") == "1")
        missing = [k for k in ((r["sequence"], r["camera"]) for r in rows)
                   if bench.get(k, {}).get("usable_for_camera_benchmark") != "1"]
        print(f"camera benchmark manifest: {len(bench)} views, {n_usable} usable; "
              f"{len(missing)} of our views not flagged usable")
        for r in rows:
            b = bench.get((r["sequence"], r["camera"]), {})
            r["usable_for_camera_benchmark"] = b.get("usable_for_camera_benchmark", "")
            r["hand_pass_strict_rate"] = b.get("hand_pass_strict_rate", "")

    write_csv(RUN_DIR / "tables" / "static_camera_gt_check.csv", rows)
    total_bad = sum(1 for r in rows if not r["all_constant"])
    print(f"VERDICT: {'STATIC_CONFIRMED' if total_bad == 0 else 'NOT_STATIC'} "
          f"({total_bad} violating views)")


if __name__ == "__main__":
    main()
