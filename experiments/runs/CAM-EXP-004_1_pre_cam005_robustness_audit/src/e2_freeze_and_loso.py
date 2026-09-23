"""Freeze the E2 ensemble specification, then audit how it was selected.

E2 was picked because it scored best among E1-E5 on the same 175 views it is
reported on. That makes 6.14 % an EXPLORATORY number, not a confirmatory one.
This script does two things:

1. writes the E2 definition to a frozen spec file so it can never quietly
   change in CAM-EXP-005/006;
2. runs leave-one-sequence-out selection: on 4 sequences, pick whichever of
   E1-E5 looks best; evaluate that pick on the held-out sequence. This is
   RETROSPECTIVE_INTERNAL_VALIDATION - the same rig, the same models, the same
   5 sequences - and is never called an independent test.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CAM004, E2_SPEC, RUN_DIR, read_csv, write_csv,  # noqa: E402
                    write_json)

RAW = RUN_DIR / "results" / "raw"
SUM = RUN_DIR / "results" / "summary"
ENSEMBLES = ["E1_perframe_mean", "E2_perframe_geomean", "E3_perview_mean",
             "E4_perview_geomean", "E5_three_model_median"]


def freeze_e2():
    spec = {
        "id": "CAM_EXP_004_E2_FROZEN_V1",
        "frozen_on": "2026-09-23",
        "frozen_in": "CAM-EXP-004.1",
        "name": "E2_perframe_geomean",
        "definition": {
            "step_1_per_frame": ("f_t = sqrt(f_anycalib_t * f_geocalib_t) - the "
                                 "geometric mean of the two model focals on the "
                                 "same frame"),
            "step_2_per_view": ("f_view = median over the frames of one "
                                "(sequence, camera) static view"),
        },
        "members": [
            {"model": "AnyCalib", "checkpoint": "anycalib_gen",
             "camera_model": "radial:2", "output": "pred_fx in original pixels",
             "adapter": "experiments/src/calibration/anycalib_adapter.py"},
            {"model": "GeoCalib", "weights": "distorted",
             "camera_model": "radial", "output": "pred_fx in original pixels",
             "adapter": "experiments/src/calibration/geocalib_adapter.py"},
        ],
        "parameters_fitted": "NONE - no weights, no thresholds, no tuning. The "
                             "geometric mean is equal-weight by construction.",
        "aggregation_unit": "(sequence, camera) - one static camera in one take",
        "inputs": "raw images, no undistortion",
        "evidence_status": "EXPLORATORY - selected as the best of E1-E5 on the "
                           "same 175 views it is reported on",
        "reported_performance_on_selection_set": {
            "dataset": "GigaHands demo, 175 views, N=8",
            "median_rel_focal_err_pct": 6.14, "p90_pct": 19.18,
            "within_5_pct": 38.3, "within_10_pct": 68.6, "within_20_pct": 90.3,
            "signed_median_pct": -2.36,
            "caveat": "selection-set performance; expected to be optimistic",
        },
        "change_policy": ("This definition is FROZEN. CAM-EXP-005 and CAM-EXP-006 "
                          "use it verbatim as a baseline and must not modify the "
                          "formula, the members or the aggregation. A different "
                          "combination rule is a NEW estimator with a new name, "
                          "compared against this one."),
    }
    write_json(E2_SPEC, spec)
    return spec


def loso():
    ens = read_csv(CAM004 / "results" / "raw" / "cross_model_ensemble.csv.gz")
    rows = [r for r in ens if r["N"] == "8"]
    by = defaultdict(dict)   # ensemble -> view -> err
    for r in rows:
        by[r["ensemble"]][(r["sequence"], r["camera"])] = float(r["mean_rel_err_pct"])
    seqs = sorted({k[0] for k in by["E2_perframe_geomean"]})

    raw, summary = [], []
    n_e2_selected = 0
    for held in seqs:
        train_scores, test_scores = {}, {}
        for e in ENSEMBLES:
            tr = np.array([v for k, v in by[e].items() if k[0] != held])
            te = np.array([v for k, v in by[e].items() if k[0] == held])
            train_scores[e] = float(np.median(tr))
            test_scores[e] = float(np.median(te))
            raw.append({"held_out_sequence": held, "ensemble": e,
                        "train_median_rel_err_pct": train_scores[e],
                        "train_n_views": int(tr.size),
                        "heldout_median_rel_err_pct": test_scores[e],
                        "heldout_n_views": int(te.size),
                        "heldout_within_5_pct": float(np.mean(te <= 5) * 100),
                        "heldout_within_10_pct": float(np.mean(te <= 10) * 100)})
        picked = min(train_scores, key=train_scores.get)
        n_e2_selected += picked == "E2_perframe_geomean"
        te_pick = np.array([v for k, v in by[picked].items() if k[0] == held])
        te_e2 = np.array([v for k, v in by["E2_perframe_geomean"].items()
                          if k[0] == held])
        summary.append({
            "held_out_sequence": held,
            "n_heldout_views": int(te_e2.size),
            "selected_on_training_folds": picked,
            "train_median_of_selected_pct": train_scores[picked],
            "runner_up": sorted(train_scores, key=train_scores.get)[1],
            "train_median_of_runner_up_pct":
                train_scores[sorted(train_scores, key=train_scores.get)[1]],
            "train_gap_to_runner_up_pp":
                train_scores[sorted(train_scores, key=train_scores.get)[1]]
                - train_scores[picked],
            "heldout_median_of_selected_pct": float(np.median(te_pick)),
            "heldout_median_of_fixed_E2_pct": float(np.median(te_e2)),
            "heldout_within_5_fixed_E2_pct": float(np.mean(te_e2 <= 5) * 100),
            "selection_cost_pp": float(np.median(te_pick) - np.median(te_e2)),
            "status": "RETROSPECTIVE_INTERNAL_VALIDATION",
        })
    write_csv(RAW / "ensemble_loso_selection.csv.gz", raw)
    write_csv(SUM / "ensemble_selection_stability.csv", summary)
    return summary, n_e2_selected, len(seqs)


def main() -> None:
    freeze_e2()
    print(f"frozen E2 spec -> {E2_SPEC}")
    summ, n_e2, n_folds = loso()
    print(f"E2 selected on the training folds in {n_e2}/{n_folds} folds")
    for r in summ:
        print(f"  hold out {r['held_out_sequence']:22s} "
              f"picked={r['selected_on_training_folds']:22s} "
              f"heldout(pick)={r['heldout_median_of_selected_pct']:6.2f}% "
              f"heldout(fixed E2)={r['heldout_median_of_fixed_E2_pct']:6.2f}% "
              f"w5={r['heldout_within_5_fixed_E2_pct']:5.1f}%")


if __name__ == "__main__":
    main()
