"""Figures for CAM-EXP-009.3.1."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (FIG, PARTICIPANTS, RAW, SOURCE_ARTIFACTS, SUM,  # noqa: E402
                    fnum, read_csv, read_json)

C_REP, C_WIT, C_XS, C_CRO = "#4a7", "#c44", "#e90", "#47a"


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=140)
    plt.close(fig)
    print("  ", name)


def main():
    v = read_json(SUM / "cam00931_verdict.json")
    d = v["distances"]
    rep = [fnum(r["d_repeat"]) for r in
           read_csv(SOURCE_ARTIFACTS["repeatability_distances.csv.gz"])]
    wit = [fnum(r["d"]) for r in read_csv(RAW / "within_session_distances.csv.gz")]
    cro = [fnum(r["d"]) for r in
           read_csv(RAW / "cross_participant_distances.csv.gz")]
    p41 = [fnum(r["d"]) for r in
           read_csv(RAW / "p41_cross_session_distances.csv.gz")]
    p41rows = read_csv(RAW / "p41_cross_session_distances.csv.gz")
    mat = read_csv(RAW / "participant_distance_matrix.csv.gz")
    ctrl = read_csv(SOURCE_ARTIFACTS["control_summary.csv"])
    pident = read_csv(SUM / "participant_identification_summary.csv")
    clean = lambda a: [x for x in a if np.isfinite(x)]  # noqa: E731
    rep, wit, cro, p41 = map(clean, (rep, wit, cro, p41))

    # 01 identity correction
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.axis("off")
    ax.text(.02, .86, "CAM-EXP-009.3", fontsize=12, weight="bold")
    ax.text(.02, .70, "local files only -> no participant field found\n"
                      "5 sequences, identity UNRESOLVED\n"
                      "analysed WITHIN_SEQUENCE vs CROSS_SEQUENCE",
            fontsize=9.5, va="top")
    ax.text(.55, .86, "CAM-EXP-009.3.1", fontsize=12, weight="bold")
    ax.text(.55, .70, "official GigaHands README documents\n"
                      "  p<participant id>-<scene>-<squence id>/\n"
                      "5 sequences -> 4 participants\n"
                      "p41 = two sessions (boxing, plant)",
            fontsize=9.5, va="top")
    ax.axvline(.52, color="#bbb")
    ax.text(.02, .14, "Same reference geometry, reused byte-identically. "
                      "Only the grouping changed.", fontsize=9, style="italic")
    ax.set_title("01 identity correction", fontsize=11)
    _save(fig, "01_identity_correction.png")

    # 02 four distance levels - the core figure
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [rep, wit, cro, p41]
    labs = ["same hand\nmeasured twice\n(D_repeat)",
            "same participant\nSAME session\n(within-session)",
            "DIFFERENT\nparticipants\n(cross-participant)",
            "same participant\nDIFFERENT session\n(p41 only)"]
    cols = [C_REP, C_WIT, C_CRO, C_XS]
    bp = ax.boxplot(data, tick_labels=labs, showfliers=False,
                    patch_artist=True)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
        patch.set_alpha(.55)
    for i, (dd, c) in enumerate(zip(data, cols), 1):
        ax.scatter(np.random.default_rng(i).normal(i, .05, len(dd)), dd, s=5,
                   alpha=.2, color=c)
    ax.set_ylabel("median |log bone-proportion difference|")
    ax.set_title("02 the four levels — a participant's own hands in a DIFFERENT\n"
                 "session are further apart than two different people's",
                 fontsize=10)
    _save(fig, "02_four_distance_levels.png")

    # 03 participant pair matrix
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, direction in zip(axes, ("LEFT_TO_RIGHT", "RIGHT_TO_LEFT")):
        M = np.full((len(PARTICIPANTS), len(PARTICIPANTS)), np.nan)
        for r in mat:
            if r["direction"] != direction:
                continue
            i = PARTICIPANTS.index(r["query_participant"])
            j = PARTICIPANTS.index(r["target_participant"])
            M[i, j] = fnum(r["d"])
        im = ax.imshow(M, cmap="viridis")
        ax.set_xticks(range(len(PARTICIPANTS)))
        ax.set_xticklabels(PARTICIPANTS)
        ax.set_yticks(range(len(PARTICIPANTS)))
        ax.set_yticklabels(PARTICIPANTS)
        for i in range(len(PARTICIPANTS)):
            ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False,
                                       edgecolor="red", lw=2))
        for i in range(len(PARTICIPANTS)):
            for j in range(len(PARTICIPANTS)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, "%.3f" % M[i, j], ha="center", va="center",
                            fontsize=7, color="w")
        ax.set_title(direction, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=.046)
    fig.suptitle("03 participant template distances (red = same participant)",
                 fontsize=11)
    _save(fig, "03_participant_pair_matrix.png")

    # 04 within-session by participant
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ws = read_csv(RAW / "within_session_distances.csv.gz")
    pp = v["per_participant_within_session"]
    for i, p in enumerate(PARTICIPANTS):
        vals = clean([fnum(r["d"]) for r in ws if r["participant"] == p])
        ax.scatter(np.random.default_rng(i).normal(i, .06, len(vals)), vals,
                   s=6, alpha=.3, color=C_WIT)
        ax.scatter([i], [pp[p]], s=110, marker="_", color="k", lw=2.5)
    ax.axhline(d["D_cross_participant_pair_equal_weight"], ls="--", c=C_CRO,
               label="cross-participant")
    ax.axhline(d["D_repeat"], ls=":", c=C_REP, label="D_repeat")
    ax.set_xticks(range(len(PARTICIPANTS)))
    ax.set_xticklabels(["%s\n(%d seq)" % (p, 2 if p == "p41" else 1)
                        for p in PARTICIPANTS])
    ax.set_ylabel("within-session L/R distance")
    ax.legend(fontsize=8)
    ax.set_title("04 within-session distance by participant\n"
                 "black bar = participant median (p41's two sessions "
                 "equally weighted)", fontsize=10)
    _save(fig, "04_within_session_by_participant.png")

    # 05 p41 cross session
    fig, ax = plt.subplots(figsize=(9, 4.6))
    names = ["A_boxingL_vs_plantL", "B_boxingR_vs_plantR",
             "C_boxingL_vs_plantR", "D_boxingR_vs_plantL"]
    data = [clean([fnum(r["d"]) for r in p41rows if r["comparison"] == n])
            for n in names]
    data.append(cro)
    data.append(clean([fnum(r["d"]) for r in ws if r["participant"] == "p41"]))
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, tick_labels=[
        "L-L", "R-R", "L-R", "R-L", "cross-\nparticipant",
        "p41 within-\nsession"])
    for patch, c in zip(bp["boxes"], [C_XS] * 4 + [C_CRO, C_WIT]):
        patch.set_facecolor(c)
        patch.set_alpha(.55)
    ax.set_ylabel("distance")
    ax.set_title("05 p41 across two sessions (boxing vs plant)\n"
                 "all four cross-session comparisons exceed the "
                 "cross-participant level — ONE participant", fontsize=10)
    _save(fig, "05_p41_cross_session.png")

    # 06 generic bone identity control
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    b = v["bone_mapping_control"]
    ax.bar([0, 1], [b["correct"], b["permuted"]], color=["#889", C_WIT],
           width=.5, tick_label=["correct bone\nmapping",
                                 "within-finger\npermuted mapping"])
    ax.set_ylabel("median within-session distance")
    ax.set_title("06 generic bone-identity control: %+.0f %%"
                 % b["degradation_pct"], fontsize=10)
    _save(fig, "06_generic_bone_identity_control.png")

    # 07 participant template identification
    fig, ax = plt.subplots(figsize=(6.5, 4))
    dirs = [r["direction"] for r in pident]
    accs = [fnum(r["top1_accuracy_pct"]) for r in pident]
    ax.bar(range(len(dirs)), accs, color=C_WIT, width=.5)
    ax.axhline(25.0, ls="--", c="k", label="chance = 25 %")
    ax.set_xticks(range(len(dirs)))
    ax.set_xticklabels(dirs, fontsize=8, rotation=15)
    ax.set_ylabel("top-1 accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.set_title("07 participant-template identification\n"
                 "4 participants, 4 queries per direction — descriptive only",
                 fontsize=10)
    _save(fig, "07_participant_template_identification.png")

    # 08 old vs corrected interpretation
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.axis("off")
    ax.text(.01, .93, "CAM-EXP-009.3 (sequence-level)", fontsize=11,
            weight="bold")
    ax.text(.01, .78,
            "\"a sequence's own L/R pair is closer than a\n"
            " cross-sequence pair\"  ratio 0.618, S_sep 1.61\n"
            "identity unresolved; p41 pair excluded as ambiguous",
            fontsize=9, va="top")
    ax.text(.01, .42, "CAM-EXP-009.3.1 (participant-aware)", fontsize=11,
            weight="bold")
    ax.text(.01, .27,
            "\"a participant's own L/R pair is closer than a different\n"
            " participant's\"  ratio %.3f, separation/repeat %.2f\n"
            "BUT the same participant across sessions is FARTHER (%.3f)\n"
            "than different participants (%.3f)  ->  the advantage is not\n"
            "a persistent person-level signature"
            % (d["ratio_within_over_cross"],
               d["separation_over_repeatability"],
               v["p41_cross_session"]["same_side_pooled"],
               v["p41_cross_session"]["cross_participant_reference"]),
            fontsize=9, va="top")
    ax.set_title("08 what changed in the interpretation", fontsize=11)
    _save(fig, "08_old_vs_corrected_interpretation.png")

    # main explanation
    fig, ax = plt.subplots(figsize=(9, 5))
    vals = [d["D_repeat"], d["D_within_session_participant_equal_weight"],
            d["D_cross_participant_pair_equal_weight"],
            v["p41_cross_session"]["same_side_pooled"]]
    cols = [C_REP, C_WIT, C_CRO, C_XS]
    ax.bar(range(4), vals, color=cols, width=.55)
    for i, val in enumerate(vals):
        ax.text(i, val, " %.4f" % val, ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["same hand\nmeasured twice",
                        "same person\nsame session",
                        "different\npeople",
                        "same person\nDIFFERENT session\n(p41 only)"],
                       fontsize=9)
    ax.set_ylabel("median |log bone-proportion difference|")
    ax.set_title("CAM-EXP-009.3.1 — the same person in a different session is "
                 "the FURTHEST\n"
                 "within-session similarity is not a persistent person "
                 "signature", fontsize=10)
    _save(fig, "CAM_EXP_009_3_1_MAIN_EXPLANATION.png")


if __name__ == "__main__":
    main()
