"""Structure of the depth-scale error: stability, consistency, correlates.

Runs after the oracle hierarchy. Everything here is ERROR ANALYSIS on data
whose reference has already been opened; nothing feeds back into a method.
"""
from __future__ import annotations

import itertools
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from common import (RAW, SUM, TAB, fnum, mad, med, pct, read_csv,  # noqa: E402
                    spearman, write_csv, write_json)


def main():
    fr = read_csv(RAW / "frame_translation_errors.csv.gz")
    par = read_csv(RAW / "sequence_scale_parameters.csv.gz")
    for r in fr:
        for k in ("log_z_ratio", "z_ratio", "dx_mm", "dy_mm", "dz_mm",
                  "root_error_mm", "xy_error_mm", "depth_error_mm",
                  "bbox_frac", "cam_t_z", "hand_extent_m", "score",
                  "z_pred", "z_ref"):
            r[k] = fnum(r[k])

    # ---------------------------------------- translation error budget
    budget = {
        "n_frames": len(fr),
        "median_abs_dx_mm": med([abs(r["dx_mm"]) for r in fr]),
        "median_abs_dy_mm": med([abs(r["dy_mm"]) for r in fr]),
        "median_abs_dz_mm": med([abs(r["dz_mm"]) for r in fr]),
        "median_xy_error_mm": med([r["xy_error_mm"] for r in fr]),
        "median_depth_error_mm": med([r["depth_error_mm"] for r in fr]),
        "median_root_error_mm": med([r["root_error_mm"] for r in fr]),
        "signed_median_dz_mm": med([r["dz_mm"] for r in fr]),
        "p75_depth_error_mm": pct([r["depth_error_mm"] for r in fr], 75),
        "p90_depth_error_mm": pct([r["depth_error_mm"] for r in fr], 90),
    }
    rr = budget["median_root_error_mm"]
    budget["depth_share_of_root_error_pct"] = (
        100.0 * budget["median_depth_error_mm"] / rr if rr else np.nan)
    budget["xy_share_of_root_error_pct"] = (
        100.0 * budget["median_xy_error_mm"] / rr if rr else np.nan)
    write_json(SUM / "translation_error_budget.json", budget)
    write_csv(TAB / "translation_component_summary.csv",
              [{"component": k, "value": v} for k, v in budget.items()])

    # ---------------------------------------- within-unit scale stability
    by_unit = defaultdict(list)
    for r in fr:
        by_unit[(r["sequence"], r["camera"])].append(r)
    stab = []
    for (seq, cam), rs in sorted(by_unit.items()):
        lr = [r["log_z_ratio"] for r in rs]
        lr = [x for x in lr if np.isfinite(x)]
        if len(lr) < 6:
            continue
        m = float(np.median(lr))
        stab.append({
            "sequence": seq, "camera": cam,
            "participant": rs[0]["participant"], "n_frames": len(lr),
            "median_log_z_ratio": m,
            "alpha": float(np.exp(m)),
            "mad_log_z_ratio": mad(lr),
            "iqr_log_z_ratio": float(np.percentile(lr, 75)
                                     - np.percentile(lr, 25)),
            "cv_z_ratio": float(np.std([np.exp(x) for x in lr])
                                / max(np.mean([np.exp(x) for x in lr]), 1e-9)),
        })
    write_csv(SUM / "depth_scale_summary.csv", stab)
    write_csv(TAB / "depth_scale_stability.csv", stab)

    scale_stats = {
        "n_units": len(stab),
        "median_alpha": med([s["alpha"] for s in stab]),
        "alpha_p25": pct([s["alpha"] for s in stab], 25),
        "alpha_p75": pct([s["alpha"] for s in stab], 75),
        "across_unit_mad_log_alpha": mad([s["median_log_z_ratio"]
                                          for s in stab]),
        "median_within_unit_mad_log_z_ratio": med([s["mad_log_z_ratio"]
                                                   for s in stab]),
        "median_within_unit_cv": med([s["cv_z_ratio"] for s in stab]),
    }
    a = scale_stats["median_within_unit_mad_log_z_ratio"]
    b = scale_stats["across_unit_mad_log_alpha"]
    scale_stats["within_over_across_spread"] = (a / b if b else np.nan)
    scale_stats["reading"] = (
        "within_over_across_spread << 1 would mean the depth-scale error is "
        "nearly constant inside a video and differs between videos, i.e. a "
        "sequence-shared bias. Near or above 1 means frame-to-frame depth "
        "variation is as large as the between-video difference.")
    write_json(SUM / "depth_scale_stats.json", scale_stats)

    # ---------------------------------------- left/right consistency
    lrrows = []
    for p in par:
        al, ar = fnum(p["alpha_left"]), fnum(p["alpha_right"])
        if np.isfinite(al) and np.isfinite(ar) and al > 0 and ar > 0:
            lrrows.append({"sequence": p["sequence"], "camera": p["camera"],
                           "alpha_left": al, "alpha_right": ar,
                           "abs_log_ratio": abs(float(np.log(al / ar)))})
    write_csv(SUM / "left_right_consistency.csv", lrrows)
    lr_stats = {
        "n_units": len(lrrows),
        "median_abs_log_alpha_ratio": med([r["abs_log_ratio"]
                                           for r in lrrows]),
        "median_pct_disagreement": 100.0 * (np.exp(med(
            [r["abs_log_ratio"] for r in lrrows])) - 1.0) if lrrows else np.nan,
        "spearman_left_right": spearman([r["alpha_left"] for r in lrrows],
                                        [r["alpha_right"] for r in lrrows]),
    }
    write_json(SUM / "left_right_stats.json", lr_stats)
    write_csv(TAB / "left_right_scale_consistency.csv",
              [{"metric": k, "value": v} for k, v in lr_stats.items()])

    # ------------------------- same camera across sequences; p41 diagnostic
    bycam = defaultdict(list)
    for p in par:
        bycam[p["camera"]].append(p)
    pairs = []
    for cam, ps in sorted(bycam.items()):
        for a1, a2 in itertools.combinations(sorted(
                ps, key=lambda z: z["sequence"]), 2):
            x, y = fnum(a1["alpha"]), fnum(a2["alpha"])
            if not (np.isfinite(x) and np.isfinite(y) and x > 0 and y > 0):
                continue
            pairs.append({
                "camera": cam, "sequence_a": a1["sequence"],
                "sequence_b": a2["sequence"],
                "participant_a": a1["participant"],
                "participant_b": a2["participant"],
                "same_participant": int(a1["participant"] == a2["participant"]),
                "alpha_a": x, "alpha_b": y,
                "abs_log_ratio": abs(float(np.log(x / y)))})
    write_csv(RAW / "camera_sequence_scale_pairs.csv.gz", pairs)
    cam_stats = {
        "n_pairs": len(pairs),
        "median_abs_log_ratio_same_camera_diff_sequence":
            med([p["abs_log_ratio"] for p in pairs]),
        "median_abs_log_ratio_same_participant":
            med([p["abs_log_ratio"] for p in pairs
                 if p["same_participant"] == 1]),
        "n_same_participant_pairs": sum(p["same_participant"] for p in pairs),
        "caveat": "GigaHands ties camera, rig and session together, so a stable "
                  "residual is described as camera-ASSOCIATED, never as caused "
                  "by the camera.",
    }
    write_json(SUM / "camera_stability_summary.json", cam_stats)
    write_csv(TAB / "camera_sequence_scale_consistency.csv",
              [{"metric": k, "value": v} for k, v in cam_stats.items()])

    # ---------------------------------------- correlates (effect sizes)
    targets = {"log_z_ratio": [r["log_z_ratio"] for r in fr],
               "dz_mm": [r["dz_mm"] for r in fr],
               "root_error_mm": [r["root_error_mm"] for r in fr]}
    preds = {"z_pred": [r["z_pred"] for r in fr],
             "cam_t_z": [r["cam_t_z"] for r in fr],
             "bbox_frac": [r["bbox_frac"] for r in fr],
             "hand_extent_m": [r["hand_extent_m"] for r in fr],
             "detection_score": [r["score"] for r in fr],
             "is_right": [1.0 if r["hand"] == "right" else 0.0 for r in fr]}
    corr = []
    for tn, tv in targets.items():
        for pn, pv in preds.items():
            corr.append({"target": tn, "predictor": pn,
                         "spearman_rho": spearman(pv, tv), "n": len(tv)})
    write_csv(SUM / "correlates.csv", corr)

    print("root %.1f mm = xy %.1f (%.0f%%) + depth %.1f (%.0f%%)"
          % (budget["median_root_error_mm"], budget["median_xy_error_mm"],
             budget["xy_share_of_root_error_pct"],
             budget["median_depth_error_mm"],
             budget["depth_share_of_root_error_pct"]))
    print("alpha median %.4f  within-unit MAD(log) %.4f  across-unit %.4f  "
          "ratio %.2f"
          % (scale_stats["median_alpha"],
             scale_stats["median_within_unit_mad_log_z_ratio"],
             scale_stats["across_unit_mad_log_alpha"],
             scale_stats["within_over_across_spread"]))
    print("left/right alpha: median |log ratio| %.4f (%.1f%%), spearman %.3f"
          % (lr_stats["median_abs_log_alpha_ratio"],
             lr_stats["median_pct_disagreement"],
             lr_stats["spearman_left_right"]))
    print("same camera, different sequence: median |log ratio| %.4f (%d pairs)"
          % (cam_stats["median_abs_log_ratio_same_camera_diff_sequence"],
             cam_stats["n_pairs"]))
    print("\ntop correlates of log_z_ratio:")
    for c in sorted([c for c in corr if c["target"] == "log_z_ratio"],
                    key=lambda z: -abs(z["spearman_rho"] if
                                       np.isfinite(z["spearman_rho"]) else 0)):
        print("   %-18s rho %+.3f" % (c["predictor"], c["spearman_rho"]))


if __name__ == "__main__":
    main()
