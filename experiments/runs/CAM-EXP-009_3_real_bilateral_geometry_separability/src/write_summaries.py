"""Per-sequence summary tables required by the spec output structure."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, SUM, fnum, read_csv, write_csv  # noqa: E402


def agg(rows, key, group):
    by = defaultdict(list)
    for r in rows:
        by[group(r)].append(fnum(r[key]))
    out = []
    for g, v in sorted(by.items()):
        v = [x for x in v if np.isfinite(x)]
        if not v:
            continue
        out.append({"group": g, "n": len(v), "median": float(np.median(v)),
                    "p25": float(np.percentile(v, 25)),
                    "p75": float(np.percentile(v, 75)),
                    "min": float(np.min(v)), "max": float(np.max(v))})
    return out


def main():
    rep = read_csv(RAW / "repeatability_distances.csv.gz")
    wit = read_csv(RAW / "same_subject_distances.csv.gz")
    cro = [c for c in read_csv(RAW / "cross_subject_distances.csv.gz")
           if c["in_primary_cross"] == "1"]

    write_csv(SUM / "repeatability_summary.csv",
              agg(rep, "d_repeat", lambda r: r["sequence"] + "|" + r["hand"])
              + agg(rep, "d_repeat", lambda r: "ALL"))
    write_csv(SUM / "same_subject_summary.csv",
              agg(wit, "d_within", lambda r: r["sequence"])
              + agg(wit, "d_within", lambda r: "ALL"))
    write_csv(SUM / "cross_subject_summary.csv",
              agg(cro, "d_cross",
                  lambda r: r["left_sequence"] + "->" + r["right_sequence"])
              + agg(cro, "d_cross", lambda r: "ALL"))
    print("wrote repeatability/same_subject/cross_subject summaries")


if __name__ == "__main__":
    main()
