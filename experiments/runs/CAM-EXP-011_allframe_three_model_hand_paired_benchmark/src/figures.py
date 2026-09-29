"""Presentation figures for CAM-EXP-011."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c011_common import (DISPLAY_SEQUENCE, FIG, SUM, TAB, fnum,  # noqa: E402
                         read_csv, read_json)

C_OFF, C_ON = "#8894a8", "#c1442e"


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    print("  ", name)


def main():
    main_tbl = read_csv(TAB / "presentation_main_table.csv")
    paired = read_csv(SUM / "paired_gain_summary.csv")
    ap = read_csv(TAB / "presentation_accuracy_precision.csv")
    inv = read_csv(TAB / "sequence_frame_inventory.csv")
    per_seq = read_csv(SUM / "per_sequence_results.csv")

    # ---------------- 1. the main presentation figure
    models = [r["Model"] for r in main_tbl]
    off = [fnum(r["Scene only"]) for r in main_tbl]
    on = [fnum(r["Scene + Hand"]) for r in main_tbl]
    x = np.arange(len(models))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, off, w, label="Scene only", color=C_OFF)
    ax.bar(x + w / 2, on, w, label="Scene + Hand", color=C_ON)
    for i, (a, b) in enumerate(zip(off, on)):
        ax.text(i - w / 2, a, "%.2f%%" % a, ha="center", va="bottom",
                fontsize=10)
        ax.text(i + w / 2, b, "%.2f%%" % b, ha="center", va="bottom",
                fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("median relative focal error (%)", fontsize=11)
    n = main_tbl[0]["N videos"] if main_tbl else "?"
    ax.set_title("Focal error with and without whole-video hand structure\n"
                 "%s paired videos, every usable RGB frame" % n, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=.25)
    ax.set_axisbelow(True)
    _save(fig, "presentation_scene_vs_hand.png")

    # ---------------- 2. frame counts
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    seqs = [r["display"] for r in inv]
    total = [fnum(r["scene_input_frames"]) for r in inv]
    per_video = [fnum(r["sequence_total_frames_per_video"]) for r in inv]
    cams = [fnum(r["usable_rgb_cameras"]) for r in inv]
    ax.bar(range(len(seqs)), total, color="#4a6fa5")
    for i, (t, p, c) in enumerate(zip(total, per_video, cams)):
        ax.text(i, t, "%d\n(%d x %d)" % (t, int(p), int(c)), ha="center",
                va="bottom", fontsize=9)
    ax.set_xticks(range(len(seqs)))
    ax.set_xticklabels(seqs, fontsize=10)
    ax.set_ylabel("RGB frames processed", fontsize=11)
    ax.set_title("Scene frames used per sequence\n"
                 "(frames per video x usable cameras)", fontsize=12)
    ax.grid(axis="y", alpha=.25)
    ax.set_axisbelow(True)
    _save(fig, "presentation_sequence_frame_counts.png")

    # ---------------- 3. accuracy / precision targets
    rng = np.random.default_rng(20260930)
    fig, axes = plt.subplots(1, len(ap), figsize=(4.2 * len(ap), 4.6))
    if len(ap) == 1:
        axes = [axes]
    lim = max(max(abs(fnum(r["median_signed_bias_pct"]))
                  + fnum(r["median_within_video_spread_pct"]) for r in ap), 30)
    for ax, r in zip(axes, ap):
        bias = fnum(r["median_signed_bias_pct"])
        spread = fnum(r["median_within_video_spread_pct"])
        for rad in (10, 20, 30):
            ax.add_patch(plt.Circle((0, 0), rad, fill=False, color="#bbb",
                                    lw=1))
        ax.plot(0, 0, "+", color="k", ms=10)
        sd = max(spread / 2.563, 1e-6)          # P90-P10 -> sigma
        n = 220
        xs = rng.normal(bias, sd, n)
        ys = rng.normal(0.0, sd, n)
        ax.scatter(xs, ys, s=14, alpha=.55, color=C_OFF, edgecolors="none")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("under-estimate  <-   focal error (%)   ->  over-estimate",
                      fontsize=8)
        ax.set_title("%s\nbias %+.1f %%, spread %.1f %%"
                     % (r["Model"], bias, spread), fontsize=10)
    fig.suptitle("Scene-only accuracy vs precision — synthetic dots generated "
                 "from the MEASURED bias and spread,\nnot a scatter of real "
                 "frames", fontsize=10)
    _save(fig, "presentation_accuracy_precision_targets.png")

    # ---------------- 4. per-sequence OFF vs ON
    seqs = sorted(set(r["display"] for r in per_seq))
    models = sorted(set(r["model"] for r in per_seq))
    fig, axes = plt.subplots(1, len(models), figsize=(5.0 * len(models), 4.4),
                             sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, m in zip(axes, models):
        o = [fnum(next((r["median_error_pct"] for r in per_seq
                        if r["display"] == s and r["model"] == m
                        and r["condition"] == "SCENE_ONLY"), np.nan))
             for s in seqs]
        h = [fnum(next((r["median_error_pct"] for r in per_seq
                        if r["display"] == s and r["model"] == m
                        and r["condition"] == "SCENE_PLUS_HAND"), np.nan))
             for s in seqs]
        xx = np.arange(len(seqs))
        ax.bar(xx - .2, o, .4, label="Scene only", color=C_OFF)
        ax.bar(xx + .2, h, .4, label="Scene + Hand", color=C_ON)
        ax.set_xticks(xx)
        ax.set_xticklabels(seqs, rotation=25, ha="right", fontsize=8)
        ax.set_title(m, fontsize=10)
        ax.grid(axis="y", alpha=.25)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("median relative focal error (%)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Per-sequence breakdown — is any single sequence driving the "
                 "pooled result?", fontsize=11)
    _save(fig, "presentation_per_sequence.png")

    # ---------------- 5. paired gain with CI
    fig, ax = plt.subplots(figsize=(8, 4.2))
    disp = [r["display"] for r in paired]
    g = [fnum(r["paired_median_gain_pp"]) for r in paired]
    lo = [g[i] - fnum(paired[i]["ci95_low"]) for i in range(len(paired))]
    hi = [fnum(paired[i]["ci95_high"]) - g[i] for i in range(len(paired))]
    ax.bar(range(len(disp)), g, color=[C_ON if v > 0 else C_OFF for v in g],
           yerr=[lo, hi], capsize=5)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(range(len(disp)))
    ax.set_xticklabels(disp, fontsize=10)
    ax.set_ylabel("paired gain (percentage points)")
    ax.set_title("Paired improvement from hand structure\n"
                 "positive = hand helps; bars are 95 % camera-cluster CI",
                 fontsize=11)
    _save(fig, "presentation_paired_gain.png")


if __name__ == "__main__":
    main()
