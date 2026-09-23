"""The one figure that has to carry CAM-EXP-004.1."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUN_DIR  # noqa: E402

SUM = RUN_DIR / "results" / "summary"
FIG = RUN_DIR / "figures"
FIG.mkdir(parents=True, exist_ok=True)
COL = {"anycalib_gen": "#1f77b4", "geocalib_distorted": "#d62728",
       "E2_frozen": "#7f2f8f"}


def load(n):
    return list(csv.DictReader(open(SUM / n, encoding="utf-8")))


def main() -> None:
    c31 = load("cam0031_statistical_reanalysis.csv")
    c04 = load("cam004_statistical_reanalysis.csv")
    loso = load("ensemble_selection_stability.csv")
    fc = load("frame_count_8_16_32_64.csv")
    det = load("model_determinism_check.csv")
    verd = json.loads((SUM / "frame_count_saturation_verdict.json").read_text())
    anyc = json.loads((SUM / "anycam_identifiability.json").read_text())

    fig = plt.figure(figsize=(17.5, 10.6))
    gs = fig.add_gridspec(2, 3, hspace=0.46, wspace=0.30)

    # 1 --- cluster CIs for CAM-EXP-003.1 ------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    order = ["LEGACY_FRAME_BOOTSTRAP", "VIEW_CLUSTER",
             "PHYSICAL_CAMERA_CLUSTER", "SEQUENCE_CLUSTER"]
    runs = [("anycalib_gen_radial", "#1f77b4"), ("geocalib_distorted_radial", "#d62728")]
    y = 0
    ticks, labels = [], []
    for run, col in runs:
        for lvl in order:
            r = next(x for x in c31 if x["model_run"] == run
                     and x["resampling_unit"] == lvl)
            lo, hi, pt = float(r["ci_lo"]), float(r["ci_hi"]), float(r["median_improvement_pp"])
            ax.plot([lo, hi], [y, y], color=col, lw=3,
                    alpha=0.35 if lvl == "LEGACY_FRAME_BOOTSTRAP" else 1.0)
            ax.plot([pt], [y], "o", color=col, ms=6)
            ticks.append(y)
            labels.append(f"{run.split('_')[0]} / {lvl.replace('_CLUSTER','').replace('LEGACY_FRAME_BOOTSTRAP','frame (legacy)').lower()}")
            y += 1
        y += 0.6
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("paired improvement from distortion-aware model (pp)")
    ax.set_title("1. CAM-EXP-003.1 survives the right\nresampling unit "
                 "(CIs widen ~2.5x, none cross 0)", fontsize=10)
    ax.grid(alpha=0.3, axis="x")

    # 2 --- CAM-EXP-004 ensemble under cluster resampling --------------------
    ax = fig.add_subplot(gs[0, 1])
    stat = "median_rel_err_improvement_pp"
    lvls = ["VIEW_CLUSTER", "PHYSICAL_CAMERA_CLUSTER", "SEQUENCE_CLUSTER"]
    for i, lvl in enumerate(lvls):
        r = next(x for x in c04 if x["statistic"] == stat
                 and x["resampling_unit"] == lvl)
        ax.plot([float(r["ci_lo"]), float(r["ci_hi"])], [i, i], color="#7f2f8f", lw=3)
        ax.plot([float(r["value"])], [i], "o", color="#7f2f8f", ms=7)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(range(len(lvls)))
    ax.set_yticklabels([f"{l.replace('_CLUSTER','').lower()}\n(n={next(x for x in c04 if x['statistic']==stat and x['resampling_unit']==l)['n_clusters']})"
                        for l in lvls], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("E2 improvement over AnyCalib-gen (pp)")
    ax.set_title("2. The ensemble gain holds at every\ncluster level", fontsize=10)
    ax.grid(alpha=0.3, axis="x")

    # 3 --- LOSO stability ----------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    seqs = [r["held_out_sequence"] for r in loso]
    e2 = [float(r["heldout_median_of_fixed_E2_pct"]) for r in loso]
    x = np.arange(len(seqs))
    ax.bar(x, e2, color="#7f2f8f")
    ax.axhline(6.14, color="k", ls="--", lw=1.4,
               label="6.14 % on the full selection set")
    ax.axhline(5, color="k", ls=":", lw=1, label="5 % target")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("-", "\n", 1) for s in seqs], fontsize=6.5)
    ax.set_ylabel("held-out median focal error (%)")
    ax.set_title("3. E2 selected on the training folds in\n5/5 folds; held-out "
                 "5.73-8.80 %", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    for i, v in enumerate(e2):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    # 4 --- real 8/16/32/64 ---------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    NS = [8, 16, 32, 64]
    for est in ("anycalib_gen", "geocalib_distorted", "E2_frozen"):
        v = [float(next(r for r in fc if r["estimator"] == est
                        and int(r["N"]) == n)["median_rel_err_pct"]) for n in NS]
        ax.plot(NS, v, "o-", color=COL[est], lw=2, label=est)
    ax.axhline(5, color="k", ls=":", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(NS)
    ax.set_xticklabels([str(n) for n in NS])
    ax.set_ylim(0, 14)
    ax.set_xlabel("frames aggregated per static camera (N)")
    ax.set_ylabel("median relative focal error (%)")
    ax.set_title("4. 16/32/64 frames actually measured:\nflat. "
                 f"{verd['OVERALL']}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 5 --- determinism + AnyCam ---------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    names = [r["model"].replace("_radial", "") for r in det]
    med = [float(r["median_spread_pct"]) for r in det]
    p90 = [float(r["p90_spread_pct"]) for r in det]
    mx = [float(r["max_spread_pct"]) for r in det]
    x = np.arange(len(names))
    w = 0.27
    ax.bar(x - w, med, w, label="median", color="#2980b9")
    ax.bar(x, p90, w, label="p90", color="#e67e22")
    ax.bar(x + w, mx, w, label="max", color="#c0392b")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("spread over 3 repeats on the SAME image (%)")
    ax.set_title("5. New problem found: GeoCalib is not\nreproducible run to run",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # 6 --- what we can and cannot claim -------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    a8 = float(next(r for r in fc if r["estimator"] == "anycalib_gen"
                    and int(r["N"]) == 8)["median_rel_err_pct"])
    e64 = float(next(r for r in fc if r["estimator"] == "E2_frozen"
                     and int(r["N"]) == 64)["median_rel_err_pct"])
    txt = (
        "6.  EVIDENCE GRADING AFTER THIS AUDIT\n"
        "\n"
        "CONFIRMED\n"
        "  distortion-aware > pinhole: holds at frame,\n"
        "    view, camera and sequence clustering\n"
        "  more frames do not help: 8/16/32/64 measured,\n"
        f"    AnyCalib {a8:.2f} % flat to N=64\n"
        "  error is per-camera bias, not frame noise\n"
        "  AnyCam is not identifiable under a static\n"
        "    camera (its own scoring function, spread\n"
        f"    across 32 focal candidates = {anyc['results'][0]['spread_across_candidates']:.1e})\n"
        "\n"
        "ROBUST_BUT_SINGLE_DATASET\n"
        "  everything above: one GigaHands rig, 5 seqs,\n"
        "    40 physical cameras\n"
        "\n"
        "EXPLORATORY\n"
        f"  E2 = 6.14 % (here {e64:.2f} % at N=64): best of 5\n"
        "    candidates on its own evaluation set\n"
        "\n"
        "PENDING_EXTERNAL_CONFIRMATION\n"
        "  no local dataset can confirm the static-camera\n"
        "    result - needs an image download decision\n"
        "\n"
        "OPEN_ISSUE\n"
        "  GeoCalib nondeterminism (+/-0.35 pp on medians)\n"
        "  hand-QC manual validation, CAM-002 scope\n"
        "\n"
        "USE AS FIXED BASELINE FROM CAM-EXP-005\n"
        "  frozen E2 spec + leave-one-sequence-out"
    )
    ax.text(0, 1, txt, va="top", ha="left", fontsize=8.6, family="monospace")

    fig.suptitle("CAM-EXP-004.1 - Robustness, statistics, reproducibility and "
                 "holdout audit: what is confirmed, what is still exploratory",
                 fontsize=14, y=0.985)
    fig.savefig(FIG / "CAM_EXP_004_1_MAIN_EXPLANATION.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / "CAM_EXP_004_1_MAIN_EXPLANATION.png")


if __name__ == "__main__":
    main()
