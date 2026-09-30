"""PHASE G-K — alpha, nested lambda selection, and frozen TEST predictions.

Order, and what may see what:

  * alpha is computed from TRAIN videos only, and never sees a reference focal.
  * lambda is selected by inner camera-grouped CV inside the OUTER TRAIN set.
    Inner-validation reference focals may be read for that selection; the
    OUTER TEST reference focal may not.
  * the outer alpha is then recomputed on the full OUTER TRAIN set, frozen
    together with the selected lambda, and applied once to the OUTER TEST
    videos.
  * the shuffled and wrong-bone controls reuse the SAME alpha and the SAME
    selected lambda. Only the hand curve differs.

lambda = 0 is not a candidate. If every positive lambda is worse on the inner
validation, the least-bad positive lambda is selected rather than falling back
to scene-only.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from c012_common import (CACHE, LAMBDA_GRID, MANIFESTS, MODELS,  # noqa: E402
                         N_INNER_FOLDS, N_OUTER_FOLDS, RAW, SCENE_SRC_64, SUM,
                         TAB, fnum, med, read_csv, scalar_focal, sha256,
                         write_csv, write_json)
from fusion import alpha_from_train, fuse, local_magnitudes  # noqa: E402


def load_curves(variant):
    out = {}
    d = CACHE / ("%s_hand_curves" % variant)
    for p in sorted(d.glob("*.npz")):
        seq, cam = p.stem.split("__", 1)
        z = np.load(p)
        out[(seq, cam)] = (z["f"], z["score"])
    return out


def reference_focal(cameras):
    """Reference focal restricted to the given cameras. Refuses others."""
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
    pv = read_csv(TAB / "primary_video_set.csv")
    prim = [r for r in pv if r["primary_set"] == "1"]
    outer = {r["camera"]: int(r["outer_fold"])
             for r in read_csv(MANIFESTS
                               / "cam_exp_012_outer_camera_folds_v1.csv")}
    inner = {r["camera"]: int(r["inner_fold"])
             for r in read_csv(MANIFESTS
                               / "cam_exp_012_inner_camera_folds_v1.csv")}
    shuf = {(r["sequence"], r["camera"]): (r["donor_sequence"],
                                           r["donor_camera"])
            for r in read_csv(MANIFESTS
                              / "cam_exp_012_shuffled_hand_mapping_v1.csv")
            if r["has_donor"] == "1"}

    correct = load_curves("correct")
    wrong = load_curves("wrong_bone")
    have_wrong = len(wrong) > 0
    print("curves loaded: correct %d, wrong_bone %d"
          % (len(correct), len(wrong)))

    scene = {}
    for r in prim:
        k = (r["sequence"], r["camera"])
        for m in MODELS:
            f = fnum(r["f_scene_%s" % m])
            s = fnum(r["sigma_%s" % m])
            if np.isfinite(f) and np.isfinite(s):
                scene[(k, m)] = (f, s)

    videos = [(r["sequence"], r["camera"]) for r in prim
              if (r["sequence"], r["camera"]) in correct]
    print("videos with a correct curve:", len(videos))

    # ---------------- local magnitudes (label-free)
    mag = {}
    for k in videos:
        hf, hs = correct[k]
        for m in MODELS:
            if (k, m) not in scene:
                continue
            f, s = scene[(k, m)]
            mag[(k, m)] = local_magnitudes(f, s, hf, hs)

    # ---------------- nested CV
    lam_rows, alpha_rows, frozen = [], [], []
    for fold in range(N_OUTER_FOLDS):
        train_v = [k for k in videos if outer.get(k[1], -1) != fold]
        test_v = [k for k in videos if outer.get(k[1], -1) == fold]
        if not test_v:
            continue
        for m in MODELS:
            # ---- inner CV over lambda
            scores = {}
            for lam in LAMBDA_GRID:
                errs = []
                for inn in range(N_INNER_FOLDS):
                    itr = [k for k in train_v if inner.get(k[1], -1) != inn]
                    iva = [k for k in train_v if inner.get(k[1], -1) == inn]
                    if not iva or not itr:
                        continue
                    pairs = [mag[(k, m)] for k in itr if (k, m) in mag
                             and np.isfinite(mag[(k, m)][1])]
                    a_in, _, _ = alpha_from_train(pairs)
                    if not np.isfinite(a_in):
                        continue
                    ref = reference_focal({k[1] for k in iva})
                    for k in iva:
                        if (k, m) not in scene or k not in correct:
                            continue
                        f, s = scene[(k, m)]
                        hf, hs = correct[k]
                        fp, _ = fuse(f, s, hf, hs, a_in, lam)
                        g = ref.get(k)
                        if g and np.isfinite(fp):
                            errs.append(100.0 * abs(fp - g) / g)
                scores[lam] = med(errs)
                lam_rows.append({"model": m, "outer_fold": fold,
                                 "lambda": lam, "n_inner_videos": len(errs),
                                 "inner_median_err_pct": scores[lam]})
            valid = {l: v for l, v in scores.items() if np.isfinite(v)}
            if not valid:
                continue
            best = min(sorted(valid), key=lambda l: (valid[l], l))  # tie -> smaller
            boundary = int(best in (LAMBDA_GRID[0], LAMBDA_GRID[-1]))

            # ---- outer alpha on the FULL outer-train set
            pairs = [mag[(k, m)] for k in train_v if (k, m) in mag
                     and np.isfinite(mag[(k, m)][1])]
            a_out, d_s, d_h = alpha_from_train(pairs)
            alpha_rows.append({
                "model": m, "outer_fold": fold, "n_train_videos": len(pairs),
                "D_scene": d_s, "D_hand_correct": d_h,
                "before_ratio": (d_h / d_s) if d_s else np.nan,
                "alpha_correct": a_out,
                "after_ratio": ((a_out * d_h) / d_s)
                if (d_s and np.isfinite(a_out)) else np.nan,
                "selected_lambda": best, "lambda_boundary": boundary,
                "degenerate": int(not np.isfinite(a_out))})
            if not np.isfinite(a_out):
                continue

            # ---- frozen TEST predictions, all four conditions
            for k in test_v:
                if (k, m) not in scene:
                    continue
                f, s = scene[(k, m)]
                base = {"sequence": k[0], "camera": k[1], "model": m,
                        "outer_fold": fold, "alpha": a_out,
                        "lambda": best, "f_scene": f, "sigma": s}
                frozen.append({**base, "condition": "SCENE_ONLY",
                               "f_pred": f, "boundary": 0})
                hf, hs = correct[k]
                fp, b = fuse(f, s, hf, hs, a_out, best)
                frozen.append({**base,
                               "condition": "SCENE_PLUS_CORRECT_HAND",
                               "f_pred": fp, "boundary": b})
                fp1, b1 = fuse(f, s, hf, hs, a_out, 1.0)
                frozen.append({**base, "condition": "CORRECT_HAND_LAMBDA1",
                               "lambda": 1.0, "f_pred": fp1, "boundary": b1})
                d = shuf.get(k)
                if d and d in correct:
                    df, ds_ = correct[d]
                    fp2, b2 = fuse(f, s, df, ds_, a_out, best)
                    frozen.append({**base,
                                   "condition": "SCENE_PLUS_SHUFFLED_HAND",
                                   "f_pred": fp2, "boundary": b2,
                                   "donor_sequence": d[0],
                                   "donor_camera": d[1]})
                if have_wrong and k in wrong:
                    wf, ws = wrong[k]
                    fp3, b3 = fuse(f, s, wf, ws, a_out, best)
                    frozen.append({**base,
                                   "condition": "SCENE_PLUS_WRONG_BONE",
                                   "f_pred": fp3, "boundary": b3})

    write_csv(RAW / "inner_cv_lambda_scores.csv", lam_rows)
    write_csv(TAB / "selected_lambda_by_fold.csv", alpha_rows)
    write_csv(RAW / "loss_scale_by_fold.csv", alpha_rows)
    write_csv(TAB / "loss_scale_normalization.csv", alpha_rows)
    p = RAW / "frozen_test_predictions.csv.gz"
    write_csv(p, frozen)
    write_json(SUM / "frozen_predictions_hash.json", {
        "file": p.name, "sha256": sha256(p), "n_rows": len(frozen),
        "note": "frozen BEFORE any OUTER TEST reference focal was opened"})

    print("\n%-20s %5s %10s %10s %8s %10s %8s"
          % ("model", "fold", "D_scene", "D_hand", "alpha", "after", "lambda"))
    for r in alpha_rows:
        print("%-20s %5d %10.4g %10.4g %8.3g %10.3f %8s%s"
              % (r["model"], r["outer_fold"], r["D_scene"],
                 r["D_hand_correct"], r["alpha_correct"], r["after_ratio"],
                 r["selected_lambda"], " *BOUNDARY" if r["lambda_boundary"]
                 else ""))
    print("\nfrozen rows:", len(frozen), "sha256", sha256(p)[:16])


if __name__ == "__main__":
    main()
