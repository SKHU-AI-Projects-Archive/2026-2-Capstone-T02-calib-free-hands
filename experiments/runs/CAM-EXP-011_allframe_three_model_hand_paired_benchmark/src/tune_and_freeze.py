"""PHASE D/E — nested lambda selection per model, then FREEZE the predictions.

The tuning procedure is identical for all three models: the same lambda grid,
the same physical-camera outer folds, selection on TRAIN cameras only. The
selected value may differ between models because their scene scores are
calibrated differently, and that is allowed — what must not differ is the
procedure.

The TEST cameras' reference focal is not read here. `evaluate.py` is the first
script that opens it, after these predictions are frozen and hashed.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from c011_common import (LAMBDA_GRID, MANIFESTS, MIN_SCENE_COVERAGE,  # noqa
                         MODELS, N_OUTER_FOLDS, RAW, SUM, TAB, fnum, med,
                         read_csv, scalar_focal, sha256, write_csv, write_json)

PRIMARY_SET = "STRICT"


def train_reference_focal(cameras):
    """Reference focal for TRAIN cameras ONLY. Refuses anything else."""
    from c011_common import SCENE_SRC_64
    out = {}
    for r in read_csv(SCENE_SRC_64):
        if r["camera"] not in cameras:
            continue
        k = (r["sequence"], r["camera"])
        if k in out:
            continue
        fx, fy = fnum(r.get("gt_fx")), fnum(r.get("gt_fy"))
        if np.isfinite(fx) and np.isfinite(fy):
            out[k] = scalar_focal(fx, fy)
    return out


def main():
    cand = read_csv(RAW / "six_condition_candidates.csv.gz")
    folds = {r["camera"]: int(r["outer_fold"])
             for r in read_csv(MANIFESTS
                               / "cam_exp_011_outer_camera_folds_v1.csv")}

    # eligible videos: coverage rule only, never performance
    ok_video = {}
    for r in cand:
        if r["frame_set"] != PRIMARY_SET:
            continue
        k = (r["sequence"], r["camera"], r["model"])
        if r["condition"] == "SCENE_ONLY":
            ok_video[k] = (r["meets_min_coverage"] == "1")

    sel_rows, frozen = [], []
    for setname in ("STRICT", "NATIVE"):
        rows = [r for r in cand if r["frame_set"] == setname]
        for m in MODELS:
            for fold in range(N_OUTER_FOLDS):
                train_cams = {c for c, f in folds.items() if f != fold}
                ref = train_reference_focal(train_cams)
                best, best_err = None, np.inf
                for lam in LAMBDA_GRID:
                    errs = []
                    for r in rows:
                        if (r["model"] != m or r["condition"] != "SCENE_PLUS_HAND"
                                or fnum(r["lambda"]) != lam
                                or folds.get(r["camera"]) == fold):
                            continue
                        g = ref.get((r["sequence"], r["camera"]))
                        f = fnum(r["f_pred"])
                        if g and np.isfinite(f):
                            errs.append(100.0 * abs(f - g) / g)
                    e = med(errs)
                    sel_rows.append({"frame_set": setname, "model": m,
                                     "outer_fold": fold, "lambda": lam,
                                     "n_train_videos": len(errs),
                                     "train_median_err_pct": e})
                    if np.isfinite(e) and e < best_err:
                        best, best_err = lam, e
                if best is None:
                    best = 0.0
                for r in rows:
                    if r["model"] != m or folds.get(r["camera"]) != fold:
                        continue
                    if r["condition"] == "SCENE_ONLY":
                        frozen.append({**{k: r[k] for k in
                                          ("sequence", "camera", "participant",
                                           "model", "frame_set",
                                           "n_frames_used", "coverage",
                                           "meets_min_coverage",
                                           "within_video_spread_pct",
                                           "f_scene", "hand_eligible")},
                                       "condition": "SCENE_ONLY",
                                       "outer_fold": fold,
                                       "lambda_selected": "",
                                       "f_pred": r["f_pred"],
                                       "boundary": r["boundary"]})
                    elif fnum(r["lambda"]) == best:
                        frozen.append({**{k: r[k] for k in
                                          ("sequence", "camera", "participant",
                                           "model", "frame_set",
                                           "n_frames_used", "coverage",
                                           "meets_min_coverage",
                                           "within_video_spread_pct",
                                           "f_scene", "hand_eligible")},
                                       "condition": "SCENE_PLUS_HAND",
                                       "outer_fold": fold,
                                       "lambda_selected": best,
                                       "f_pred": r["f_pred"],
                                       "boundary": r["boundary"]})
    write_csv(RAW / "fold_lambda_selection.csv", sel_rows)
    write_csv(TAB / "lambda_selection.csv", sel_rows)
    p = RAW / "final_six_condition_predictions.csv.gz"
    write_csv(p, frozen)
    write_json(SUM / "frozen_predictions_hash.json", {
        "file": p.name, "sha256": sha256(p), "n_rows": len(frozen),
        "note": "frozen BEFORE any TEST reference focal was opened",
    })

    chosen = defaultdict(dict)
    for r in sel_rows:
        if r["frame_set"] != PRIMARY_SET:
            continue
        k = (r["model"], r["outer_fold"])
        e = r["train_median_err_pct"]
        if np.isfinite(e) and (k not in chosen or e < chosen[k][1]):
            chosen[k] = (r["lambda"], e)
    for (m, f), (lam, e) in sorted(chosen.items()):
        print("%-20s fold %d  lambda=%-5s train %.3f%%" % (m, f, lam, e))
    print("frozen rows:", len(frozen), "sha256", sha256(p)[:16])


if __name__ == "__main__":
    main()
