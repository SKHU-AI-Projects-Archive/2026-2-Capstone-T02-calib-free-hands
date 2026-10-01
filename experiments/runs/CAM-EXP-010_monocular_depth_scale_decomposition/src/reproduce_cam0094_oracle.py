"""Reproduce CAM-EXP-009.4's REF_FOCAL_ORACLE row before anything is built on it.

If the O0 baseline here does not match that run's oracle-focal result, the whole
decomposition would be resting on a different pipeline and must stop.

The two are not expected to be bit-identical: CAM-EXP-009.4 aggregated over all
of its frames, while this run reports EVAL frames of units that also have enough
FIT frames. So the check is on the FULL frame set (matching CAM-EXP-009.4's
population), and the tolerance is stated rather than assumed to be zero.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from common import (RAW, SOURCE_ARTIFACTS, SUM, TAB, fnum, med,  # noqa: E402
                    read_csv, write_csv, write_json)

TOL_MM = 1.0


def main():
    mine = read_csv(RAW / "frame_translation_errors.csv.gz")
    theirs = [r for r in read_csv(SOURCE_ARTIFACTS["cam0094_abs3d"])
              if r["method"] == "REF_FOCAL_ORACLE"]

    # unit-level medians on both sides, matched on the same units
    def unit_med(rows, key, getter):
        by = defaultdict(list)
        for r in rows:
            by[(r["sequence"], r["camera"])].append(getter(r))
        return {k: float(np.median([x for x in v if np.isfinite(x)]))
                for k, v in by.items()
                if any(np.isfinite(x) for x in v)}

    a_root = unit_med(theirs, None, lambda r: fnum(r["wrist_root_error_mm"]))
    b_root = unit_med(mine, None, lambda r: fnum(r["root_error_mm"]))
    a_dep = unit_med(theirs, None, lambda r: fnum(r["root_depth_error_mm"]))
    b_dep = unit_med(mine, None, lambda r: fnum(r["depth_error_mm"]))

    ks = sorted(set(a_root) & set(b_root))
    rows = []
    for name, A, B in (("wrist_root_error_mm", a_root, b_root),
                       ("root_depth_error_mm", a_dep, b_dep)):
        va = med([A[k] for k in ks if k in A])
        vb = med([B[k] for k in ks if k in B])
        diffs = [abs(A[k] - B[k]) for k in ks if k in A and k in B]
        rows.append({"metric": name, "CAM0094_median_mm": va,
                     "CAM010_median_mm": vb,
                     "abs_diff_mm": abs(va - vb),
                     "median_per_unit_abs_diff_mm": med(diffs),
                     "max_per_unit_abs_diff_mm": (float(np.max(diffs))
                                                  if diffs else np.nan),
                     "tolerance_mm": TOL_MM,
                     "PASS": int(abs(va - vb) <= TOL_MM)})
    write_csv(TAB / "cam0094_oracle_reproduction.csv", rows)
    ok = all(r["PASS"] for r in rows)
    write_json(SUM / "cam0094_oracle_reproduction.json", {
        "source_run": "CAM-EXP-009_4_scene_shared_anatomy_focal_refinement",
        "source_commit_expected": "1c36d92",
        "n_units_matched": len(ks),
        "metrics": rows, "ALL_REPRODUCED": ok, "tolerance_mm": TOL_MM,
    })
    for r in rows:
        print("%-24s CAM-009.4 %8.2f   CAM-010 %8.2f   diff %6.3f mm  %s"
              % (r["metric"], r["CAM0094_median_mm"], r["CAM010_median_mm"],
                 r["abs_diff_mm"], "PASS" if r["PASS"] else "FAIL"))
    print("units matched:", len(ks), "| ALL_REPRODUCED:", ok)
    if not ok:
        raise SystemExit("CAM-EXP-009.4 oracle row NOT reproduced - stop")


if __name__ == "__main__":
    main()
