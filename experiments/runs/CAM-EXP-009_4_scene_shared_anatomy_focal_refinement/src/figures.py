"""Figures for CAM-EXP-009.4."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIG, RAW, SUM, TAB, fnum, med, read_csv, read_json  # noqa

C0, C3, CR = "#889", "#c44", "#47a"


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=140)
    plt.close(fig)
    print("  ", name)


def main():
    summ = read_csv(SUM / "focal_summary.csv")
    paired = read_csv(SUM / "focal_paired_summary.csv")
    pv = read_json(SUM / "focal_verdict.json")
    rig = read_json(SUM / "rig_confound_summary.json")
    gate = read_json(SUM / "synthetic_gate_summary.json")
    abl = read_csv(SUM / "ablation_summary.csv")
    a3 = read_csv(SUM / "absolute3d_summary.csv") \
        if (SUM / "absolute3d_summary.csv").exists() else []

    # 01 synthetic gate
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    cond = gate["conditions"]
    ks = list(cond)
    ax.bar(range(len(ks)), [cond[k]["median_err_pct"] for k in ks],
           color=[C3 if "CLEAN" in k or "MODERATE" in k else C0 for k in ks])
    ax.axhline(5.0, ls="--", c="k", lw=1, label="5 % gate")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([k.replace("_", "\n") for k in ks], fontsize=8)
    ax.set_ylabel("median focal error (%)")
    ax.legend(fontsize=8)
    ax.set_title("01 synthetic implementation gate — PASSED\n"
                 "sanity of the solver, NOT evidence the method works on real "
                 "data", fontsize=9)
    _save(fig, "01_synthetic_gate.png")

    # 02 focal main comparison
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ms = [s["method"] for s in summ]
    vals = [fnum(s["median_err_pct"]) for s in summ]
    cols = [C3 if m == "M3_SCENE_SHARED_GENERIC_FULL" else
            (CR if m == "RIG_MEDIAN_TRAIN_ONLY" else C0) for m in ms]
    ax.bar(range(len(ms)), vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v, " %.2f" % v, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels([m.replace("_", "\n") for m in ms], fontsize=7)
    ax.set_ylabel("median focal error (%)")
    ax.set_title("02 focal error by method (PRIMARY_COMMON_SET)\n"
                 "blue = dataset-confound diagnostic, not a deployment method",
                 fontsize=9)
    _save(fig, "02_focal_main.png")

    # 03 paired M0 vs M3
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    x = [fnum(p["M0"]) for p in paired]
    y = [fnum(p["M3"]) for p in paired]
    ax.scatter(x, y, s=12, alpha=.5)
    lim = max([v for v in x + y if np.isfinite(v)] + [1])
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlabel("M0 scene-only focal error (%)")
    ax.set_ylabel("M3 full model focal error (%)")
    ax.set_title("03 paired per-unit\nbelow the line = M3 better", fontsize=9)
    _save(fig, "03_paired_focal.png")

    # 04 paired gain distribution
    fig, ax = plt.subplots(figsize=(7, 4.2))
    g = [fnum(p["gain_pp"]) for p in paired]
    g = [v for v in g if np.isfinite(v)]
    ax.hist(g, bins=25, color=C3, alpha=.8)
    ax.axvline(0, c="k", lw=1)
    ax.axvline(pv["paired_median_gain_pp"], c="r", ls="--",
               label="median %+.3f pp" % pv["paired_median_gain_pp"])
    ax.set_xlabel("gain (M0 error - M3 error), pp")
    ax.set_ylabel("units")
    ax.legend(fontsize=8)
    ax.set_title("04 paired gain; camera-cluster CI [%.3f, %.3f]"
                 % tuple(pv["paired_gain_ci95"]), fontsize=9)
    _save(fig, "04_paired_gain.png")

    # 05 ablations
    fig, ax = plt.subplots(figsize=(8.5, 4))
    labs = [a["comparison"] for a in abl]
    v = [fnum(a["median_paired_gain_pp"]) for a in abl]
    ax.barh(range(len(labs)), v, color=[C3 if x > 0 else C0 for x in v])
    ax.axvline(0, c="k", lw=1)
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels(labs, fontsize=8)
    ax.set_xlabel("median paired gain (pp), positive = the added term helps")
    ax.set_title("05 ablations", fontsize=9)
    _save(fig, "05_ablations.png")

    # 06 GigaHands focal confound
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    r = rig["test_reference_focal"]
    ax.bar([0, 1, 2],
           [fnum(next(s["median_err_pct"] for s in summ
                      if s["method"] == "M0_SCENE_ONLY")),
            fnum(next(s["median_err_pct"] for s in summ
                      if s["method"] == "M3_SCENE_SHARED_GENERIC_FULL")),
            rig["rig_median_train_only_median_err_pct"]],
           color=[C0, C3, CR],
           tick_label=["M0 scene-only", "M3 full", "RIG_MEDIAN\n(diagnostic)"])
    ax.set_ylabel("median focal error (%)")
    ax.set_title("06 the GigaHands narrow-focal confound\n"
                 "reference focal CV %.2f %%, relative span %.2f %%"
                 % (r["cv_pct"], r["relative_span_pct"]), fontsize=9)
    _save(fig, "06_rig_confound.png")

    # 07 absolute 3D
    if a3:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        mets = ["median_wrist_root_error_mm", "median_absolute_mpjpe_mm",
                "median_root_aligned_mpjpe_mm"]
        titles = ["wrist/root error (mm)", "absolute MPJPE (mm)",
                  "root-aligned MPJPE (mm)\n(control: should barely move)"]
        sub = [s for s in a3 if s["hand"] == "BIMANUAL_COMBINED"]
        ms2 = [s["method"] for s in sub]
        for ax, mt, ti in zip(axes, mets, titles):
            vv = [fnum(s[mt]) for s in sub]
            ax.bar(range(len(ms2)), vv,
                   color=[C3 if "M3" in m else C0 for m in ms2])
            ax.set_xticks(range(len(ms2)))
            ax.set_xticklabels([m.replace("_", "\n") for m in ms2], fontsize=6)
            ax.set_title(ti, fontsize=9)
        fig.suptitle("07 absolute 3D under each method's focal", fontsize=10)
        _save(fig, "07_absolute_3d.png")

        # 08 left/right/bimanual
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        hands = ["left", "right", "BIMANUAL_COMBINED"]
        w = 0.35
        for i, m in enumerate(("M0_SCENE_ONLY",
                               "M3_SCENE_SHARED_GENERIC_FULL")):
            vv = [fnum(next((s["median_wrist_root_error_mm"] for s in a3
                             if s["method"] == m and s["hand"] == h), np.nan))
                  for h in hands]
            ax.bar(np.arange(3) + (i - .5) * w, vv, w,
                   label=m, color=[C0, C3][i])
        ax.set_xticks(range(3))
        ax.set_xticklabels(hands)
        ax.set_ylabel("median wrist/root error (mm)")
        ax.legend(fontsize=8)
        ax.set_title("08 per-hand downstream error", fontsize=9)
        _save(fig, "08_left_right.png")

    # main explanation
    fig, ax = plt.subplots(figsize=(9, 5))
    m0 = fnum(next(s["median_err_pct"] for s in summ
                   if s["method"] == "M0_SCENE_ONLY"))
    m3 = fnum(next(s["median_err_pct"] for s in summ
                   if s["method"] == "M3_SCENE_SHARED_GENERIC_FULL"))
    rg = rig["rig_median_train_only_median_err_pct"]
    ax.bar([0, 1, 2], [m0, m3, rg], color=[C0, C3, CR], width=.55,
           tick_label=["scene only\n(M0)", "scene + both hands\n(M3)",
                       "constant rig focal\n(diagnostic)"])
    for i, v in enumerate([m0, m3, rg]):
        ax.text(i, v, " %.2f %%" % v, ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("median focal error (%)")
    ax.set_title("CAM-EXP-009.4 — does whole-sequence hand geometry add to a "
                 "scene focal estimate?\n%s"
                 % ("YES on this common set"
                    if pv["CAM0094_FOCAL_INCREMENTAL_SIGNAL"]
                    else "No measurable incremental gain"), fontsize=10)
    _save(fig, "CAM_EXP_009_4_MAIN_EXPLANATION.png")


if __name__ == "__main__":
    main()
