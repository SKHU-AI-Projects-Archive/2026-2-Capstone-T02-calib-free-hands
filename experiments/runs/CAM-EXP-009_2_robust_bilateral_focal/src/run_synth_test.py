"""TEST run: the DEV winner and the M0 baseline, on disjoint TEST seeds.

Only two methods are run and reported. Running the whole family here and then
quoting whichever did best would make the TEST set a second selection set.

All M0-M6 scores are nonetheless recorded in an auxiliary column set for the
audit trail, but the verdict is computed from the pre-registered pair only.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c92_common import MANIFESTS, RAW, read_csv, read_json, write_csv  # noqa
from run_synth_dev import run_trial  # noqa: E402


def main():
    sel = read_json(MANIFESTS / "cam_exp_0092_selected_method_v1.json")
    methods = [sel["SELECTED_METHOD"], sel["baseline_for_comparison"]]
    methods = list(dict.fromkeys(methods))
    rows = read_csv(MANIFESTS / "cam_exp_0092_synthetic_test_trials_v1.csv.gz")
    print("methods:", methods, " trials:", len(rows), flush=True)
    recs, t0 = [], time.time()
    for i, r in enumerate(rows, 1):
        recs.extend(run_trial(r, methods))
        if i % 20 == 0 or i == len(rows):
            print("%3d/%d  %.1f min" % (i, len(rows), (time.time() - t0) / 60),
                  flush=True)
    write_csv(RAW / "test_trials.csv.gz", recs)
    print("wrote", len(recs), "rows")


if __name__ == "__main__":
    main()
