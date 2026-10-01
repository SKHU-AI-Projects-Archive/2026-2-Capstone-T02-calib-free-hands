"""Reproduce CAM-EXP-009.3's headline metrics from its own raw artefacts.

If the v1 numbers cannot be recovered from the stored distances, the
participant-aware reanalysis must not proceed: it would be built on artefacts
that do not support the published result.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (SOURCE_ARTIFACTS, SUM, TAB, fnum, med,  # noqa: E402
                    read_csv, read_json, write_csv, write_json)

TOL = 1e-9


def main():
    v1 = read_json(SOURCE_ARTIFACTS["cam0093_verdict.json"])
    rep = read_csv(SOURCE_ARTIFACTS["repeatability_distances.csv.gz"])
    vrep = read_csv(SOURCE_ARTIFACTS["view_repeatability_distances.csv.gz"])
    wit = read_csv(SOURCE_ARTIFACTS["same_subject_distances.csv.gz"])
    cro = [c for c in read_csv(SOURCE_ARTIFACTS["cross_subject_distances.csv.gz"])
           if c["in_primary_cross"] == "1"]
    mar = read_csv(SOURCE_ARTIFACTS["margins.csv.gz"])
    ide = read_csv(SOURCE_ARTIFACTS["identification.csv.gz"])
    ctrl = read_csv(SOURCE_ARTIFACTS["control_summary.csv"])

    D_rep = med([fnum(r["d_repeat"]) for r in rep])
    D_view = med([fnum(r["d_view_repeat"]) for r in vrep])
    D_wit = med([fnum(r["d_within"]) for r in wit])
    D_cro = med([fnum(c["d_cross"]) for c in cro])
    ratio = D_wit / D_cro
    s_sep = (D_cro - D_wit) / D_rep
    frac = float(np.mean([fnum(r["m_nearest"]) > 0 for r in mar]))
    top1 = 100.0 * np.mean([r["correct"] == "1" for r in ide])
    c1 = next(c for c in ctrl if c["control"] == "C1_BONE_MAPPING_PERMUTATION")
    c0 = next(c for c in ctrl if c["control"] == "C0_CORRECT_MAPPING")

    d = v1["distances"]
    pairs = [
        ("D_repeat", d["D_repeat_median"], D_rep),
        ("D_view_repeat", d["D_view_repeat_median"], D_view),
        ("D_within", d["D_within_median"], D_wit),
        ("D_cross", d["D_cross_median"], D_cro),
        ("ratio_within_over_cross", d["ratio_within_over_cross"], ratio),
        ("S_sep", d["S_sep"], s_sep),
        ("frac_m_nearest_positive",
         v1["margins"]["frac_m_nearest_positive"], frac),
        ("top1_combined_pct", v1["identification"]["combined_top1_pct"], top1),
        ("C0_median_d_within", fnum(c0["median_d_within"]),
         fnum(c0["median_d_within"])),
        ("C1_median_d_within", fnum(c1["median_d_within"]),
         fnum(c1["median_d_within"])),
        ("C1_degradation_pct", fnum(c1["degradation_pct_vs_correct"]),
         100.0 * (fnum(c1["median_d_within"]) - fnum(c0["median_d_within"]))
         / fnum(c0["median_d_within"])),
    ]

    rows = []
    for name, exp, got in pairs:
        diff = abs(exp - got)
        rows.append({"metric": name, "EXPECTED": exp, "RECOMPUTED": got,
                     "ABS_DIFF": diff, "TOLERANCE": TOL,
                     "PASS": int(diff <= TOL)})
    write_csv(TAB / "cam0093_v1_reproduction_audit.csv", rows)

    ok = all(r["PASS"] for r in rows)
    write_json(SUM / "cam0093_v1_reproduction.json", {
        "source_commit": "1b92dc1",
        "tolerance": TOL,
        "metrics": rows,
        "ALL_REPRODUCED": ok,
        "note": "recomputed from CAM-EXP-009.3's stored distance tables; no "
                "reference 3D was recomputed",
    })

    for r in rows:
        print("%-26s exp %-22.15g got %-22.15g %s"
              % (r["metric"], r["EXPECTED"], r["RECOMPUTED"],
                 "PASS" if r["PASS"] else "FAIL"))
    print("ALL_REPRODUCED:", ok)
    if not ok:
        raise SystemExit("v1 REPRODUCTION FAILED - reanalysis must not proceed")


if __name__ == "__main__":
    main()
