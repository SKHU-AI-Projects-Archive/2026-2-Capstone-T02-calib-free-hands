"""p41 cross-session diagnostic — the same participant in two sessions.

This is the only comparison in the local subset that can separate a
person-specific signature from a session-specific one, and it rests on
ONE participant. It is descriptive; no population claim follows from it.

Four camera-matched comparisons between `p41-boxing-0021` and
`p41-plant-0004`:

  A  boxing LEFT  vs plant LEFT     (same side)
  B  boxing RIGHT vs plant RIGHT    (same side)
  C  boxing LEFT  vs plant RIGHT    (opposite side)
  D  boxing RIGHT vs plant LEFT     (opposite side)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (MULTI_SESSION, PARTICIPANT_OF, RAW,  # noqa: E402
                    SOURCE_ARTIFACTS, SUM, d_primary, fnum, med, read_csv,
                    write_csv, zof)

COMPARISONS = [
    ("A_boxingL_vs_plantL", "left", "left", "SAME_SIDE"),
    ("B_boxingR_vs_plantR", "right", "right", "SAME_SIDE"),
    ("C_boxingL_vs_plantR", "left", "right", "OPPOSITE_SIDE"),
    ("D_boxingR_vs_plantL", "right", "left", "OPPOSITE_SIDE"),
]


def main():
    rows = read_csv(SOURCE_ARTIFACTS["geometry_templates.csv.gz"])
    full = {(r["sequence"], r["camera"], r["hand"]): zof(r)
            for r in rows if r["template_type"] == "FULL"}
    cams = sorted(set(c for (_s, c, _h) in full))

    sA, sB = MULTI_SESSION["p41"]
    out = []
    for name, hA, hB, kind in COMPARISONS:
        for cam in cams:
            a = full.get((sA, cam, hA))
            b = full.get((sB, cam, hB))
            if a is None or b is None:
                continue
            out.append({"comparison": name, "kind": kind, "camera": cam,
                        "sequence_a": sA, "hand_a": hA,
                        "sequence_b": sB, "hand_b": hB,
                        "participant": "p41", "d": d_primary(a, b)})
    write_csv(RAW / "p41_cross_session_distances.csv.gz", out)

    cross = read_csv(RAW / "cross_participant_distances.csv.gz")
    within = read_csv(RAW / "within_session_distances.csv.gz")
    d_cross = med([fnum(r["d"]) for r in cross])
    d_within_p41 = med([fnum(r["d"]) for r in within
                        if r["participant"] == "p41"])

    summary = []
    for name, _hA, _hB, kind in COMPARISONS:
        v = [r["d"] for r in out if r["comparison"] == name]
        v = [x for x in v if np.isfinite(x)]
        summary.append({
            "comparison": name, "kind": kind, "n_cameras": len(v),
            "median": med(v),
            "p25": float(np.percentile(v, 25)) if v else np.nan,
            "p75": float(np.percentile(v, 75)) if v else np.nan,
            "vs_cross_participant_median": d_cross,
            "smaller_than_cross_participant": int(med(v) < d_cross),
        })
    for kind in ("SAME_SIDE", "OPPOSITE_SIDE"):
        v = [r["d"] for r in out if r["kind"] == kind]
        summary.append({"comparison": "POOLED_" + kind, "kind": kind,
                        "n_cameras": len(v), "median": med(v),
                        "p25": float(np.percentile(v, 25)) if v else np.nan,
                        "p75": float(np.percentile(v, 75)) if v else np.nan,
                        "vs_cross_participant_median": d_cross,
                        "smaller_than_cross_participant":
                            int(med(v) < d_cross)})
    summary.append({"comparison": "REFERENCE_within_session_p41",
                    "kind": "WITHIN_SESSION", "n_cameras": "",
                    "median": d_within_p41,
                    "vs_cross_participant_median": d_cross,
                    "smaller_than_cross_participant":
                        int(d_within_p41 < d_cross)})
    summary.append({"comparison": "REFERENCE_cross_participant",
                    "kind": "CROSS_PARTICIPANT", "n_cameras": len(cross),
                    "median": d_cross,
                    "vs_cross_participant_median": d_cross,
                    "smaller_than_cross_participant": 0})
    write_csv(SUM / "p41_cross_session_summary.csv", summary)

    for r in summary:
        print("%-32s %-16s n=%-5s median %.5f  %s"
              % (r["comparison"], r["kind"], r["n_cameras"], r["median"],
                 "< cross-participant" if r["smaller_than_cross_participant"]
                 else ">= cross-participant"))


if __name__ == "__main__":
    main()
