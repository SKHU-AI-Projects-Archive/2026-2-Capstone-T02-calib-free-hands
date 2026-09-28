"""PHASE F/G — open the TEST reference focal and score the frozen predictions.

This is the FIRST script permitted to read the test reference focal. It writes
`results/raw/REFERENCE_FOCAL_OPENED.txt` recording the moment, the git HEAD and
the hashes of the frozen predictions and the method spec. After this point no
prediction may be regenerated; a new question needs a new experiment.
"""
from __future__ import annotations

import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from common import (GATE, MANIFESTS, RAW, REPO, SCENE_SRC, SUM,  # noqa: E402
                    TAB, fnum, med, paired_cluster_bootstrap, read_csv,
                    scalar_focal, sha256, write_csv, write_json)

FROZEN = RAW / "frozen_test_predictions.csv"


def reference_focal_all():
    rows = read_csv(SCENE_SRC)
    out = {}
    for r in rows:
        k = (r["sequence"], r["camera"])
        if k in out:
            continue
        fx, fy = fnum(r.get("gt_fx")), fnum(r.get("gt_fy"))
        if np.isfinite(fx) and np.isfinite(fy):
            out[k] = scalar_focal(fx, fy)
    return out


def open_log():
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()
    spec = MANIFESTS / "cam_exp_0094_method_spec_v1.json"
    txt = (
        "REFERENCE FOCAL OPENED\n"
        "timestamp_utc: %s\n"
        "git_head: %s\n"
        "frozen_test_predictions_sha256: %s\n"
        "method_spec_sha256: %s\n"
        "lambda_selection_sha256: %s\n"
        "After this point no method prediction may be regenerated.\n"
        % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), head,
           sha256(FROZEN), sha256(spec),
           sha256(RAW / "fold_lambda_selection.csv")))
    (RAW / "REFERENCE_FOCAL_OPENED.txt").write_text(txt, encoding="utf-8")
    print(txt)


def main():
    open_log()
    ref = reference_focal_all()
    rows = read_csv(FROZEN)

    # RIG_MEDIAN_TRAIN_ONLY diagnostic: train cameras' reference focal median
    cam_fold = {}
    for r in rows:
        cam_fold[r["camera"]] = int(r["outer_fold"])
    rig = {}
    for fold in sorted(set(cam_fold.values())):
        tr = [ref[k] for k in ref if cam_fold.get(k[1], -1) not in (fold, -1)]
        rig[fold] = float(np.median(tr)) if tr else float("nan")

    out = []
    for r in rows:
        k = (r["sequence"], r["camera"])
        f_ref = ref.get(k)
        f = fnum(r["f_pred"])
        if f_ref is None or not np.isfinite(f_ref):
            continue
        out.append({**r, "f_ref": f_ref,
                    "err_pct": (100.0 * abs(f - f_ref) / f_ref
                                if np.isfinite(f) else np.nan),
                    "signed_err_pct": (100.0 * (f - f_ref) / f_ref
                                       if np.isfinite(f) else np.nan)})
    # rig diagnostic rows
    for k, f_ref in ref.items():
        fold = cam_fold.get(k[1])
        if fold is None:
            continue
        fr = rig[fold]
        out.append({"sequence": k[0], "camera": k[1],
                    "participant": "", "outer_fold": fold,
                    "method": "RIG_MEDIAN_TRAIN_ONLY", "f_pred": fr,
                    "f_ref": f_ref,
                    "err_pct": 100.0 * abs(fr - f_ref) / f_ref,
                    "signed_err_pct": 100.0 * (fr - f_ref) / f_ref,
                    "role": "DATASET_CONFOUND_DIAGNOSTIC"})
    write_csv(RAW / "focal_unit_results.csv.gz", out)

    methods = ["M0_SCENE_ONLY", "M1_SCENE_GENERIC_INDEPENDENT",
               "M2_SCENE_SHARED_NO_GENERIC", "M3_SCENE_SHARED_GENERIC_FULL",
               "RIG_MEDIAN_TRAIN_ONLY"]
    # PRIMARY_COMMON_SET: units with a finite result for every real method
    per = defaultdict(dict)
    for r in out:
        if np.isfinite(fnum(r["err_pct"])):
            per[(r["sequence"], r["camera"])][r["method"]] = fnum(r["err_pct"])
    real = methods[:4]
    common = [k for k, v in per.items() if all(m in v for m in real)]
    write_csv(TAB / "common_set.csv",
              [{"sequence": k[0], "camera": k[1]} for k in sorted(common)])

    summ = []
    for m in methods:
        vals = [per[k][m] for k in common if m in per[k]]
        full = [fnum(r["err_pct"]) for r in out if r["method"] == m]
        full = [x for x in full if np.isfinite(x)]
        summ.append({
            "method": m, "n_common": len(vals), "n_full_eligible": len(full),
            "median_err_pct": med(vals), "mean_err_pct":
                float(np.mean(vals)) if vals else np.nan,
            "p75": float(np.percentile(vals, 75)) if vals else np.nan,
            "p90": float(np.percentile(vals, 90)) if vals else np.nan,
            "within5_pct": 100.0 * np.mean([v <= 5 for v in vals])
            if vals else np.nan,
            "within10_pct": 100.0 * np.mean([v <= 10 for v in vals])
            if vals else np.nan,
        })
    write_csv(SUM / "focal_summary.csv", summ)
    write_csv(TAB / "focal_main_table.csv", summ)

    # PRIMARY paired comparison M3 vs M0
    paired = []
    for k in common:
        g = per[k]["M0_SCENE_ONLY"] - per[k]["M3_SCENE_SHARED_GENERIC_FULL"]
        paired.append({"sequence": k[0], "camera": k[1],
                       "M0": per[k]["M0_SCENE_ONLY"],
                       "M3": per[k]["M3_SCENE_SHARED_GENERIC_FULL"],
                       "gain_pp": g})
    gains = [p["gain_pp"] for p in paired]
    cams = [p["camera"] for p in paired]
    pt, lo, hi = paired_cluster_bootstrap(gains, cams)
    m0 = med([p["M0"] for p in paired])
    m3 = med([p["M3"] for p in paired])
    rel = 100.0 * (m0 - m3) / m0 if m0 else np.nan

    pairsum = {
        "n_units": len(paired), "n_cameras": len(set(cams)),
        "M0_median_err_pct": m0, "M3_median_err_pct": m3,
        "relative_reduction_pct": rel,
        "paired_median_gain_pp": pt,
        "paired_gain_ci95": [lo, hi],
        "cluster": "PHYSICAL CAMERA", "n_boot": 10000,
    }
    write_json(SUM / "focal_paired_summary.json", pairsum)
    write_csv(SUM / "focal_paired_summary.csv", paired)

    tag = bool(m3 < m0 and rel >= GATE["focal_relative_reduction_pct"]
               and pt > 0 and lo > 0)
    write_json(SUM / "focal_verdict.json", {
        "CAM0094_FOCAL_INCREMENTAL_SIGNAL": tag,
        "criteria": {
            "M3_below_M0": bool(m3 < m0),
            "relative_reduction_ge_10pct":
                bool(rel >= GATE["focal_relative_reduction_pct"]),
            "paired_median_gain_positive": bool(pt > 0),
            "bootstrap_lower_bound_positive": bool(lo > 0)},
        **pairsum,
    })

    # correction-direction audit vs the rig median
    corr = []
    for r in out:
        if r["method"] != "M3_SCENE_SHARED_GENERIC_FULL":
            continue
        fold = int(r["outer_fold"])
        f0, f3 = fnum(r["f_scene"]), fnum(r["f_pred"])
        fr = rig.get(fold, np.nan)
        if not all(np.isfinite([f0, f3, fr])) or abs(f3 - f0) < 1e-9:
            continue
        toward = int(abs(f3 - fr) < abs(f0 - fr))
        corr.append({"sequence": r["sequence"], "camera": r["camera"],
                     "f_scene": f0, "f_M3": f3, "rig_median_train": fr,
                     "moved_toward_rig_median": toward,
                     "delta_log_f": float(np.log(f3 / f0))})
    write_csv(SUM / "correction_direction_audit.csv", corr)
    frac = float(np.mean([c["moved_toward_rig_median"] for c in corr])) \
        if corr else np.nan

    refs = [ref[k] for k in common]
    write_json(SUM / "rig_confound_summary.json", {
        "test_reference_focal": {
            "min": float(np.min(refs)), "max": float(np.max(refs)),
            "median": float(np.median(refs)),
            "cv_pct": float(100.0 * np.std(refs) / np.mean(refs)),
            "relative_span_pct": float(100.0 * (np.max(refs) - np.min(refs))
                                       / np.median(refs))},
        "rig_median_train_only_median_err_pct":
            next(s["median_err_pct"] for s in summ
                 if s["method"] == "RIG_MEDIAN_TRAIN_ONLY"),
        "fraction_M3_corrections_toward_rig_median": frac,
        "n_units_corrected": len(corr),
        "caveat": "GigaHands uses a narrow-focal capture rig. A constant is a "
                  "strong baseline here, so a positive result on this dataset "
                  "cannot establish general camera generalisation.",
    })

    for s in summ:
        print("%-32s n=%-4s median %7.3f%%  <=5%% %5.1f  <=10%% %5.1f"
              % (s["method"], s["n_common"], s["median_err_pct"],
                 s["within5_pct"], s["within10_pct"]))
    print("\nPRIMARY M3 vs M0: %.3f%% -> %.3f%% (rel %.1f%%)"
          % (m0, m3, rel))
    print("paired gain %.3f pp  CI [%.3f, %.3f]  (%d units, %d cameras)"
          % (pt, lo, hi, len(paired), len(set(cams))))
    print("FOCAL_INCREMENTAL_SIGNAL:", tag)
    print("corrections toward rig median: %.1f%%" % (100 * frac))


if __name__ == "__main__":
    main()
