"""Figures for CAM-EXP-010."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (FIG, RAW, SUM, fnum, med, read_csv, read_json)  # noqa

CXY, CZ, CO = "#47a", "#c44", "#4a7"


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=140)
    plt.close(fig)
    print("  ", name)


def main():
    fr = read_csv(RAW / "frame_translation_errors.csv.gz")
    par = read_csv(RAW / "sequence_scale_parameters.csv.gz")
    hier = read_csv(SUM / "oracle_hierarchy_summary.csv")
    budget = read_json(SUM / "translation_error_budget.json")
    stab = read_csv(SUM / "depth_scale_summary.csv")
    ident = read_json(SUM / "identifiability_summary.json")
    pairs = read_csv(RAW / "camera_sequence_scale_pairs.csv.gz")

    clean = lambda a: [x for x in a if np.isfinite(x)]  # noqa: E731

    # 01 error budget
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    vals = [budget["median_abs_dx_mm"], budget["median_abs_dy_mm"],
            budget["median_abs_dz_mm"], budget["median_xy_error_mm"],
            budget["median_root_error_mm"]]
    labs = ["|dx|", "|dy|", "|dz|", "xy", "root (3D)"]
    ax.bar(range(5), vals, color=[CXY, CXY, CZ, CXY, "#666"])
    for i, v in enumerate(vals):
        ax.text(i, v, " %.1f" % v, ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(5))
    ax.set_xticklabels(labs)
    ax.set_ylabel("median error (mm)")
    ax.set_title("01 absolute translation error under the ORACLE FOCAL\n"
                 "depth is %.0f %% of the root error, x/y %.0f %%"
                 % (budget["depth_share_of_root_error_pct"],
                    budget["xy_share_of_root_error_pct"]), fontsize=10)
    _save(fig, "01_absolute_error_budget.png")

    # 02 log z ratio distribution
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    lr = clean([fnum(r["log_z_ratio"]) for r in fr])
    ax.hist(lr, bins=80, color=CZ, alpha=.85)
    ax.axvline(0, c="k", lw=1, label="perfect depth")
    ax.axvline(float(np.median(lr)), c="b", ls="--",
               label="median %.4f (alpha %.3f)"
               % (np.median(lr), np.exp(np.median(lr))))
    ax.set_xlabel("log(z_ref / z_pred)")
    ax.set_ylabel("frames")
    ax.legend(fontsize=8)
    ax.set_title("02 frame-wise depth ratio", fontsize=10)
    _save(fig, "02_frame_scale_ratio_distribution.png")

    # 03 within-sequence stability
    fig, ax = plt.subplots(figsize=(10, 4.4))
    s = sorted(stab, key=lambda z: fnum(z["alpha"]))
    x = np.arange(len(s))
    a = [fnum(r["alpha"]) for r in s]
    e = [fnum(r["mad_log_z_ratio"]) for r in s]
    ax.errorbar(x, a, yerr=[np.array(a) * np.array(e)], fmt="o", ms=3,
                lw=.8, color=CZ, ecolor="#aaa")
    ax.axhline(1.0, c="k", ls="--", lw=1, label="no correction needed")
    ax.set_xlabel("sequence-camera unit (sorted)")
    ax.set_ylabel("alpha = z_ref / z_pred")
    ax.legend(fontsize=8)
    ax.set_title("03 per-unit depth scale; error bars = within-unit MAD\n"
                 "tight bars would mean one constant per video explains it",
                 fontsize=10)
    _save(fig, "03_within_sequence_scale_stability.png")

    # 04 left vs right alpha
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    al = [fnum(r["alpha_left"]) for r in par]
    ar = [fnum(r["alpha_right"]) for r in par]
    m = [i for i in range(len(al)) if np.isfinite(al[i]) and np.isfinite(ar[i])]
    ax.scatter([al[i] for i in m], [ar[i] for i in m], s=14, alpha=.6)
    lim = [min(clean(al + ar) + [1]) * .95, max(clean(al + ar) + [1]) * 1.05]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("alpha fitted on LEFT only")
    ax.set_ylabel("alpha fitted on RIGHT only")
    ax.set_title("04 do the two hands of one video need the same scale?",
                 fontsize=10)
    _save(fig, "04_left_vs_right_alpha.png")

    # 05 O0 vs O1 paired
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    res = read_csv(RAW / "oracle_corrected_frame_results.csv.gz")
    by = defaultdict(lambda: defaultdict(list))
    for r in res:
        by[r["oracle"]][(r["sequence"], r["camera"])].append(
            fnum(r["depth_error_mm"]))
    o0 = {k: np.median(v) for k, v in by["O0_ORACLE_FOCAL_ONLY"].items()}
    o1 = {k: np.median(v)
          for k, v in by["O1_SEQUENCE_MULTIPLICATIVE_Z_SCALE"].items()}
    ks = sorted(set(o0) & set(o1))
    ax.scatter([o0[k] for k in ks], [o1[k] for k in ks], s=16, alpha=.6,
               color=CZ)
    lim = max([o0[k] for k in ks] + [o1[k] for k in ks] + [1])
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlabel("O0 depth error (mm)")
    ax.set_ylabel("O1 sequence z-scale, held out (mm)")
    ax.set_title("05 does one scale per video help, on held-out frames?\n"
                 "below the line = it helps", fontsize=10)
    _save(fig, "05_sequence_scale_oracle_gain.png")

    # 06 waterfall
    for metric, fname, title in (
            ("median_root_error_mm", "06_oracle_hierarchy_waterfall.png",
             "wrist/root error"),
            ("median_absolute_mpjpe_mm", "06b_waterfall_mpjpe.png",
             "absolute MPJPE")):
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        names = [h["oracle"] for h in hier]
        vals = [fnum(h[metric]) for h in hier]
        floor = fnum(hier[0]["median_root_aligned_mpjpe_mm"])
        ax.bar(range(len(names)), vals,
               color=[CO if i == 0 else CZ for i in range(len(names))])
        ax.axhline(floor, ls=":", c="k",
                   label="root-aligned pose floor %.1f mm" % floor)
        for i, v in enumerate(vals):
            ax.text(i, v, " %.1f" % v, ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=6)
        ax.set_ylabel("%s (mm)" % title)
        ax.legend(fontsize=8)
        ax.set_title("06 oracle hierarchy — %s (held-out EVAL frames)" % title,
                     fontsize=10)
        _save(fig, fname)

    # 07 same camera across sequences
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    if pairs:
        xa = [fnum(p["alpha_a"]) for p in pairs]
        yb = [fnum(p["alpha_b"]) for p in pairs]
        sp = [p["same_participant"] == "1" for p in pairs]
        ax.scatter([xa[i] for i in range(len(xa)) if not sp[i]],
                   [yb[i] for i in range(len(xa)) if not sp[i]], s=16,
                   alpha=.6, label="different participant")
        ax.scatter([xa[i] for i in range(len(xa)) if sp[i]],
                   [yb[i] for i in range(len(xa)) if sp[i]], s=30, alpha=.9,
                   color=CZ, label="same participant (p41)")
        lim = [min(clean(xa + yb) + [1]) * .95, max(clean(xa + yb) + [1]) * 1.05]
        ax.plot(lim, lim, "k--", lw=1)
        ax.legend(fontsize=8)
    ax.set_xlabel("alpha, sequence A")
    ax.set_ylabel("alpha, sequence B")
    ax.set_title("07 same physical camera, different sequence\n"
                 "does the required scale persist?", fontsize=10)
    _save(fig, "07_camera_sequence_scale_stability.png")

    # 08 metric scale ambiguity
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.axis("off")
    t = ident["tests"]
    ax.text(.02, .92, "Scale every 3D point AND the camera translation by s:",
            fontsize=11, weight="bold")
    y = .74
    for r in t:
        k = ("max_reprojection_difference_px"
             if "max_reprojection_difference_px" in r
             else "max_proportion_difference")
        ax.text(.04, y, "%-32s  max difference %.2e   %s"
                % (r["test"], r[k], "PASS" if r["PASS"] else "FAIL"),
                fontsize=9, family="monospace")
        y -= .12
    ax.text(.02, .18,
            "The 2D projections are identical to machine precision.\n"
            "s = 0.5, 1, 2, 5 are indistinguishable from the images alone.\n"
            "Multi-frame and bilateral consistency do NOT break the tie.",
            fontsize=10, va="top")
    ax.set_title("08 metric scale is not geometrically identifiable "
                 "from the allowed cues", fontsize=11)
    _save(fig, "08_metric_scale_ambiguity.png")

    # main explanation
    fig, ax = plt.subplots(figsize=(9.5, 5))
    keep = ["O0_ORACLE_FOCAL_ONLY", "O1_SEQUENCE_MULTIPLICATIVE_Z_SCALE",
            "O5_PER_FRAME_DEPTH_ORACLE",
            "O6_PER_FRAME_ROOT_TRANSLATION_ORACLE"]
    vals = [fnum(next(h["median_root_error_mm"] for h in hier
                      if h["oracle"] == k)) for k in keep]
    ax.bar(range(4), vals, color=[CO, CZ, CZ, CXY], width=.6)
    for i, v in enumerate(vals):
        ax.text(i, v, " %.1f mm" % v, ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["perfect focal\n(where 009.4 ended)",
                        "+ one depth scale\nper video",
                        "+ perfect depth\nevery frame",
                        "+ perfect x/y too"], fontsize=9)
    ax.set_ylabel("median wrist/root error (mm)")
    ax.set_title("CAM-EXP-010 — where the remaining absolute error lives\n"
                 "all bars are ORACLE DIAGNOSTICS, not deployable methods",
                 fontsize=10)
    _save(fig, "CAM_EXP_010_MAIN_EXPLANATION.png")


if __name__ == "__main__":
    main()
