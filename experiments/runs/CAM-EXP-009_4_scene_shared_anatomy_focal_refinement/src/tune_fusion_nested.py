"""PHASE D/E — nested fusion-lambda tuning, then FREEZE the test predictions.

Fusion, in the log-focal domain so both terms are dimensionless:

    L_total(q) = L_scene(q) + lambda * L_hand(q)
    L_scene(q) = ((log f_cand - log f_scene) / sigma_scene)^2
    L_hand(q)  = held-out reprojection / image diagonal, rescaled by the unit's
                 own fixed scene scale so no candidate-dependent min-max
                 rescaling is applied

`lambda` is chosen per outer fold on the TRAIN physical cameras only. The TEST
cameras' reference focal is never read here: this script opens the reference
focal for TRAIN cameras exclusively, and `evaluate_focal.py` is the first place
a TEST focal is touched.

Outer split unit = PHYSICAL CAMERA, so every sequence of a camera stays on one
side of the split.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from common import (FUSION_LAMBDA_GRID, MANIFESTS, RAW, SUM, TAB,  # noqa: E402
                    fnum, med, q_grid, read_csv, read_json, sha256,
                    scalar_focal, write_csv, write_json)

CURVES = RAW / "candidate_score_curves.csv.gz"


def reference_focal_map(cameras=None):
    """Reference focal per (sequence, camera). RESTRICTED to `cameras`.

    This is the only reader of the reference focal in PHASE D, and it refuses
    to return a camera outside the requested TRAIN set.
    """
    from common import SCENE_SRC
    rows = read_csv(SCENE_SRC)
    out = {}
    for r in rows:
        if cameras is not None and r["camera"] not in cameras:
            continue
        k = (r["sequence"], r["camera"])
        if k in out:
            continue
        fx, fy = fnum(r.get("gt_fx")), fnum(r.get("gt_fy"))
        if np.isfinite(fx) and np.isfinite(fy):
            out[k] = scalar_focal(fx, fy)
    return out


def load_curves(path=CURVES):
    qs = q_grid()
    rows = [r for r in read_csv(path) if r.get("status") == "ok"]
    out = []
    for r in rows:
        c = np.array([fnum(r.get("c%02d" % k, "")) for k in range(len(qs))])
        out.append({"sequence": r["sequence"], "camera": r["camera"],
                    "participant": r["participant"],
                    "outer_fold": int(r["outer_fold"]),
                    "method": r["method"],
                    "f_scene": fnum(r["f_scene"]),
                    "sigma_scene": fnum(r["sigma_scene"]),
                    "curve": c,
                    "hand_boundary": int(fnum(r["hand_boundary"])),
                    "hand_score_range": fnum(r["hand_score_range"]),
                    "norm_a": fnum(r["norm_a"]),
                    "norm_delta_L": fnum(r["norm_delta_L"]),
                    "norm_delta_R": fnum(r["norm_delta_R"])})
    return out, qs


def fuse(rec, qs, lam):
    """Fused focal for one unit at one lambda. Returns (f, boundary)."""
    c = rec["curve"]
    ok = np.isfinite(c)
    if ok.sum() < 5:
        return float("nan"), 1
    lq = np.log(qs)
    L_scene = (lq / rec["sigma_scene"]) ** 2
    # the hand term is put on the same dimensionless footing by dividing by its
    # own fixed scale: the unit's median finite score. No candidate-dependent
    # min-max rescaling (that would change what lambda means).
    scale = float(np.nanmedian(c[ok]))
    if not np.isfinite(scale) or scale <= 0:
        return float("nan"), 1
    L_hand = c / scale
    tot = np.where(ok, L_scene + lam * L_hand, np.inf)
    j = int(np.argmin(tot))
    idx = np.flatnonzero(ok)
    return float(rec["f_scene"] * qs[j]), int(j == idx[0] or j == idx[-1])


def main():
    recs, qs = load_curves()
    methods = sorted(set(r["method"] for r in recs))
    folds = sorted(set(r["outer_fold"] for r in recs))
    cam_fold = {r["camera"]: r["outer_fold"] for r in recs}

    sel_rows, test_rows = [], []
    for fold in folds:
        train_cams = {c for c, f in cam_fold.items() if f != fold}
        ref_train = reference_focal_map(train_cams)
        for m in methods:
            tr = [r for r in recs if r["method"] == m
                  and r["outer_fold"] != fold]
            best, best_err = None, np.inf
            for lam in FUSION_LAMBDA_GRID:
                errs = []
                for r in tr:
                    ref = ref_train.get((r["sequence"], r["camera"]))
                    if ref is None or not np.isfinite(ref):
                        continue
                    f, _ = fuse(r, qs, lam)
                    if np.isfinite(f):
                        errs.append(100.0 * abs(f - ref) / ref)
                e = med(errs)
                sel_rows.append({"outer_fold": fold, "method": m,
                                 "lambda": lam, "n_train_units": len(errs),
                                 "train_median_err_pct": e})
                if np.isfinite(e) and e < best_err:
                    best, best_err = lam, e
            if best is None:
                best = 0.0
            for r in recs:
                if r["method"] != m or r["outer_fold"] != fold:
                    continue
                f, b = fuse(r, qs, best)
                f0 = r["f_scene"]
                test_rows.append({
                    "sequence": r["sequence"], "camera": r["camera"],
                    "participant": r["participant"], "outer_fold": fold,
                    "method": m, "lambda_selected": best,
                    "train_median_err_pct": best_err,
                    "f_scene": f0, "f_pred": f, "boundary": b,
                    "hand_boundary": r["hand_boundary"],
                    "hand_score_range": r["hand_score_range"],
                    "norm_a": r["norm_a"], "norm_delta_L": r["norm_delta_L"],
                    "norm_delta_R": r["norm_delta_R"],
                    "delta_log_f": (float(np.log(f / f0))
                                    if np.isfinite(f) and f > 0 else np.nan),
                })
    # M0 is the scene-only estimate; it needs no lambda
    seen = set()
    for r in recs:
        k = (r["sequence"], r["camera"])
        if k in seen:
            continue
        seen.add(k)
        test_rows.append({
            "sequence": r["sequence"], "camera": r["camera"],
            "participant": r["participant"], "outer_fold": r["outer_fold"],
            "method": "M0_SCENE_ONLY", "lambda_selected": "",
            "f_scene": r["f_scene"], "f_pred": r["f_scene"], "boundary": 0,
            "delta_log_f": 0.0})

    write_csv(RAW / "fold_lambda_selection.csv", sel_rows)
    write_csv(TAB / "lambda_selection.csv",
              [r for r in sel_rows if r["train_median_err_pct"] is not None])
    p = RAW / "frozen_test_predictions.csv"
    write_csv(p, test_rows)
    write_json(SUM / "frozen_predictions_hash.json", {
        "file": p.name, "sha256": sha256(p), "n_rows": len(test_rows),
        "note": "frozen BEFORE any TEST reference focal was opened; never "
                "regenerated after results",
    })

    chosen = {}
    for r in sel_rows:
        k = (r["outer_fold"], r["method"])
        e = r["train_median_err_pct"]
        if np.isfinite(e) and (k not in chosen or e < chosen[k][1]):
            chosen[k] = (r["lambda"], e)
    for (fold, m), (lam, e) in sorted(chosen.items()):
        print("fold %d  %-30s lambda=%-5s train_err %.3f%%"
              % (fold, m, lam, e))
    print("frozen test predictions:", len(test_rows), "sha256",
          sha256(p)[:16])


if __name__ == "__main__":
    main()
