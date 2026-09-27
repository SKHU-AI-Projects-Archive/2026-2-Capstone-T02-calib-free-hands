"""Select exactly one method from the DEV run, by the pre-registered rule.

Selection score = median over the 7 DEV cells of each cell's median EVAL focal
error. EVAL (odd frames) is used so the choice is not made on the same frames
the shape was fitted on. Ties break by the frozen METHODS order, so the rule is
deterministic and does not involve a second look at the numbers.

The winner is frozen to a manifest. The TEST run then reports only that method
and the M0 baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c92_common import (MANIFESTS, RAW, SUM, TAB, fnum, read_csv,  # noqa: E402
                        sha256, write_csv, write_json)
from robust_bilateral_scores import METHODS  # noqa: E402


def main():
    rows = read_csv(RAW / "dev_trials.csv.gz")
    cells = sorted(set(r["cell"] for r in rows))
    per_cell, table = {}, []
    for m in METHODS:
        med = []
        for c in cells:
            v = [fnum(r["err_pct_eval"]) for r in rows
                 if r["method"] == m and r["cell"] == c]
            v = [x for x in v if np.isfinite(x)]
            cm = float(np.median(v)) if v else float("nan")
            med.append(cm)
            table.append({"method": m, "cell": c, "n": len(v),
                          "median_err_pct_eval": cm})
        per_cell[m] = med

    scores = {m: (float(np.median(v)) if np.isfinite(v).all()
                  else float("inf")) for m, v in per_cell.items()}
    winner = min(METHODS, key=lambda m: (scores[m], METHODS.index(m)))

    write_csv(TAB / "dev_method_by_cell.csv", table)
    sel = {
        "selection_rule": "median over the 7 DEV cells of each cell's median "
                          "EVAL focal error; ties break by frozen METHODS "
                          "order",
        "selected_on": "DEV seeds only (namespace CAM0092_DEV), disjoint from "
                       "TEST",
        "cells": cells,
        "selection_score_pct": scores,
        "per_cell_median_err_pct_eval": {m: per_cell[m] for m in METHODS},
        "SELECTED_METHOD": winner,
        "baseline_for_comparison": "M0_BASELINE_RAW_L1",
        "note": "DEV numbers are a selection instrument, not evidence of "
                "performance. Only the TEST run is reported as a result.",
    }
    p = MANIFESTS / "cam_exp_0092_selected_method_v1.json"
    write_json(p, sel)
    write_json(SUM / "dev_selection.json", sel)

    for m in METHODS:
        print("%-28s %7.3f" % (m, scores[m]))
    print("\nSELECTED:", winner)
    print("manifest sha256:", sha256(p)[:16])


if __name__ == "__main__":
    main()
