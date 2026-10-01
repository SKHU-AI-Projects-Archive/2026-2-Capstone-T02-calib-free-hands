"""Concatenate chunked part files into one predictions file per model.

The expensive models are run in bounded slices (see run_benchmark --start /
--count) because this machine terminates very long processes; this step joins
the slices back together and rebuilds all_predictions.csv.gz.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from common import RUN_DIR, load_frames, read_csv, rel, write_csv

PARTS = RUN_DIR / "results" / "raw" / "_parts"


def main() -> dict:
    n_expected = len(load_frames())
    by_model = defaultdict(list)
    for p in sorted(PARTS.glob("*.csv.gz")):
        key = p.stem.rsplit("_", 1)[0]
        by_model[key].extend(read_csv(p))

    status = {}
    for key, rows in sorted(by_model.items()):
        # a frame may appear twice if a slice was re-run; keep one per frame
        seen, uniq = set(), []
        for r in rows:
            k = (r["sequence"], r["camera"], r["frame"])
            if k in seen:
                continue
            seen.add(k)
            uniq.append(r)
        out = RUN_DIR / "results" / "raw" / f"{key}_predictions.csv.gz"
        write_csv(out, uniq)
        status[key] = {"rows": len(uniq), "expected": n_expected,
                       "complete": len(uniq) >= n_expected}

    all_rows = []
    for p in sorted((RUN_DIR / "results" / "raw").glob("*_predictions.csv.gz")):
        if p.name == "all_predictions.csv.gz":
            continue
        all_rows.extend(read_csv(p))
    write_csv(RUN_DIR / "results" / "raw" / "all_predictions.csv.gz", all_rows)
    print(json.dumps(status, indent=2))
    return status


if __name__ == "__main__":
    main()
