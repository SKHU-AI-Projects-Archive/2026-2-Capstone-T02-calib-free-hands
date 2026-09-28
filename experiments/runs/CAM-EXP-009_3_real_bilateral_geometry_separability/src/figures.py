"""Figures for CAM-EXP-009.3."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIG, RAW, SUM, fnum, read_csv, read_json  # noqa: E402

C_REP, C_WIT, C_CRO = "#4a7", "#c44", "#47a"


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=140)
    plt.close(fig)
    print("  ", name)


def main():
    v = read_json(SUM / "cam0093_verdict.json")
    rep = [fnum(r["d_repeat"]) for r in
           read_csv(RAW / "repeatability_distances.csv.gz")]
    vrep = [fnum(r["d_view_repeat"]) for r in
            read_csv(RAW / "view_repeatability_distances.csv.gz")]
    wit = [fnum(r["d_within"]) for r in
           read_csv(RAW / "same_subject_distances.csv.gz")]
    crossr = read_csv(RAW / "cross_subject_distances.csv.gz")
    cro = [fnum(r["d_cross"]) for r in crossr if r["in_primary_cross"] == "1"]
    margins = read_csv(RAW / "margins.csv.gz")
    ident = read_csv(RAW / "identification.csv.gz")
    bone = read_csv(SUM / "per_bone_summary.csv")
    ctrl = read_csv(SUM / "control_summary.csv")
    csame = read_csv(RAW / "cross_sequence_same_subject.csv.gz")

    clean = lambda a: [x for x in a if np.isfinite(x)]  # noqa: E731
    rep, vrep, wit, cro = map(clean, (rep, vrep, wit, cro))

    # 01 core question
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot([rep, wit, cro], tick_labels=[
        "same hand\n(repeatability)", "same sequence\nLEFT vs RIGHT",
        "different sequence\nLEFT vs RIGHT"], showfliers=False)
    for i, (d, c) in enumerate(zip((rep, wit, cro), (C_REP, C_WIT, C_CRO)), 1):
        ax.scatter(np.random.default_rng(i).normal(i, 0.05, len(d)), d, s=4,
                   alpha=.25, color=c)
    ax.set_ylabel("median |log bone-proportion difference|")
    ax.set_title("01 The core question: is a sequence's own LEFT/RIGHT pair\n"
                 "closer than a pair across sequences?", fontsize=10)
    _save(fig, "01_core_question.png")

    # 02 ECDF
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for d, lab, c in ((rep, "D_repeat", C_REP), (vrep, "D_view_repeat", "#999"),
                      (wit, "D_within", C_WIT), (cro, "D_cross", C_CRO)):
        if not d:
            continue
        x = np.sort(d)
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), label=lab, color=c)
    ax.set_xlabel("distance")
    ax.set_ylabel("ECDF")
    ax.legend(fontsize=8)
    ax.set_title("02 distance distributions", fontsize=10)
    _save(fig, "02_distance_distributions.png")

    # 03 same vs nearest other
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    xs = [fnum(r["d_same"]) for r in margins]
    ys = [fnum(r["d_nearest_other"]) for r in margins]
    ax.scatter(xs, ys, s=6, alpha=.35)
    lim = max(clean(xs + ys) + [1e-3])
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlabel("D to its OWN sequence's other hand")
    ax.set_ylabel("D to the NEAREST other sequence's hand")
    ax.set_title("03 above the line = own pair is closer (good)\n"
                 "below the line = an outsider is closer", fontsize=9)
    _save(fig, "03_same_vs_nearest_other.png")

    # 04 distance matrix
    seqs = sorted(set(r["left_sequence"] for r in crossr))
    M = np.full((len(seqs), len(seqs)), np.nan)
    acc = defaultdict(list)
    for r in crossr:
        acc[(r["left_sequence"], r["right_sequence"])].append(fnum(r["d_cross"]))
    for r in read_csv(RAW / "same_subject_distances.csv.gz"):
        acc[(r["sequence"], r["sequence"])].append(fnum(r["d_within"]))
    for i, a in enumerate(seqs):
        for j, b in enumerate(seqs):
            vv = clean(acc.get((a, b), []))
            if vv:
                M[i, j] = np.median(vv)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(M, cmap="viridis")
    ax.set_xticks(range(len(seqs)))
    ax.set_xticklabels(seqs, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(seqs)))
    ax.set_yticklabels(seqs, fontsize=7)
    for i in range(len(seqs)):
        ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False,
                                   edgecolor="red", lw=2))
    fig.colorbar(im, ax=ax, label="median distance")
    ax.set_xlabel("RIGHT hand from")
    ax.set_ylabel("LEFT hand from")
    ax.set_title("04 sequence distance matrix\n"
                 "red = same sequence (its own pair)", fontsize=10)
    _save(fig, "04_subject_distance_matrix.png")

    # 05 identification
    fig, ax = plt.subplots(figsize=(6.5, 4))
    dirs = sorted(set(r["direction"] for r in ident))
    accs = [100 * np.mean([r["correct"] == "1" for r in ident
                           if r["direction"] == d]) for d in dirs]
    ch = v["identification"]["chance_pct"]
    ax.bar(range(len(dirs)), accs, color=C_WIT, width=.5)
    ax.axhline(ch, ls="--", c="k", label="chance = %.1f %%" % ch)
    ax.set_xticks(range(len(dirs)))
    ax.set_xticklabels(dirs, fontsize=8)
    ax.set_ylabel("top-1 accuracy (%)")
    ax.legend(fontsize=8)
    ax.set_title("05 does the other hand identify its own sequence?",
                 fontsize=10)
    _save(fig, "05_subject_identification.png")

    # 06 repeatability vs separation
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    d = v["distances"]
    ax.bar([0, 1], [d["D_repeat_median"], d["separation"]],
           color=[C_REP, C_CRO],
           tick_label=["reference repeatability\n(D_repeat)",
                       "separation\n(D_cross - D_within)"])
    ax.set_ylabel("distance units")
    ax.set_title("06 is the separation bigger than the reference noise?\n"
                 "S_sep = %.2f" % d["S_sep"], fontsize=10)
    _save(fig, "06_repeatability_vs_separation.png")

    # 07 per bone
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(bone))
    ax.bar(x - .25, [fnum(b["V_repeat"]) for b in bone], .25, label="repeat",
           color=C_REP)
    ax.bar(x, [fnum(b["V_within"]) for b in bone], .25, label="within",
           color=C_WIT)
    ax.bar(x + .25, [fnum(b["V_cross"]) for b in bone], .25, label="cross",
           color=C_CRO)
    ax.set_xticks(x)
    ax.set_xticklabels([b["bone"] for b in bone], rotation=70, ha="right",
                       fontsize=7)
    ax.set_ylabel("median |delta z|")
    ax.legend(fontsize=8)
    ax.set_title("07 per-bone structure (DESCRIPTIVE ONLY - the primary "
                 "metric keeps all 20 bones)", fontsize=10)
    _save(fig, "07_per_bone_structure.png")

    # 08 bone permutation control
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    labs = [c["control"].replace("_", "\n") for c in ctrl]
    axes[0].bar(range(len(ctrl)), [fnum(c["median_d_within"]) for c in ctrl],
                color=["#889", C_WIT], width=.5)
    axes[0].set_xticks(range(len(ctrl)))
    axes[0].set_xticklabels(labs, fontsize=7)
    axes[0].set_ylabel("median within-sequence distance")
    axes[1].bar(range(len(ctrl)), [fnum(c["top1_accuracy_pct"]) for c in ctrl],
                color=["#889", C_WIT], width=.5)
    axes[1].axhline(ch, ls="--", c="k")
    axes[1].set_xticks(range(len(ctrl)))
    axes[1].set_xticklabels(labs, fontsize=7)
    axes[1].set_ylabel("top-1 accuracy (%)")
    fig.suptitle("08 bone-mapping permutation control", fontsize=10)
    _save(fig, "08_bone_permutation_control.png")

    # 09 cross-sequence same candidate participant
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    a = clean([fnum(r["d"]) for r in csame if r["comparison"] == "SAME_SIDE"])
    b = clean([fnum(r["d"]) for r in csame
               if r["comparison"] == "OPPOSITE_SIDE"])
    if a or b:
        ax.boxplot([a or [np.nan], b or [np.nan], wit, cro],
                   tick_labels=["same prefix\nsame side",
                                "same prefix\nopposite side",
                                "within\nsequence", "cross\nsequence"],
                   showfliers=False)
    ax.set_ylabel("distance")
    ax.set_title("09 the one same-prefix sequence pair (descriptive;\n"
                 "its subject relationship is exactly what is unverified)",
                 fontsize=9)
    _save(fig, "09_cross_sequence_same_subject.png")

    # main explanation
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    vals = [d["D_repeat_median"], d["D_within_median"], d["D_cross_median"]]
    ax.bar(range(3), vals, color=[C_REP, C_WIT, C_CRO], width=.55)
    for i, val in enumerate(vals):
        ax.text(i, val, "  %.4f" % val, ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["same hand,\nmeasured twice",
                        "same sequence,\nleft vs right",
                        "different sequences,\nleft vs right"], fontsize=9)
    ax.set_ylabel("median |log bone-proportion difference|")
    ax.set_title("CAM-EXP-009.3 — ratio within/cross = %.2f,  S_sep = %.2f\n%s"
                 % (d["ratio_within_over_cross"], d["S_sep"],
                    " + ".join(v["VERDICT_TAGS"])), fontsize=9)
    _save(fig, "CAM_EXP_009_3_MAIN_EXPLANATION.png")


if __name__ == "__main__":
    main()
