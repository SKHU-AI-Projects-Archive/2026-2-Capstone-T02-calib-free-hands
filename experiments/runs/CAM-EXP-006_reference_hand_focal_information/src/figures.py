"""Figures for CAM-EXP-006."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIG, RAW, SUM, TAB, fnum, read_csv, read_json  # noqa: E402

COUNTS = (1, 2, 4, 8, 16)
COND_LABEL = {
    "MAIN": "reference hand (main)",
    "JOINT_PERMUTATION": "control: joints permuted",
    "WRONG_POSE": "control: wrong pose",
    "PLANARIZED": "control: planarized",
}
COND_COLOR = {"MAIN": "#1f4e79", "JOINT_PERMUTATION": "#b0b0b0",
              "WRONG_POSE": "#c0504d", "PLANARIZED": "#e0a030"}
NOTE = ("INTERNAL_REFERENCE_HAND_DIAGNOSTIC - an upper bound under an oracle "
        "reference hand, not a deployable method")


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  " + name)


def main() -> None:
    summ = read_csv(TAB / "solver_summary.csv")
    comp = read_csv(TAB / "comparators.csv")
    est = read_csv(RAW / "view_focal_estimates.csv.gz")
    verdict = read_json(SUM / "cam006_verdict.json")

    def series(cond, col="median_rel_focal_err_pct"):
        d = {int(s["n_frames"]): fnum(s[col]) for s in summ
             if s["condition"] == cond}
        return [d.get(n, np.nan) for n in COUNTS]

    oracle = next(fnum(c["median_rel_focal_err_pct"]) for c in comp
                  if c["comparator"] == "CONSTANT_RIG_FOCAL_ORACLE")

    # Fig 01 - error vs frame count, all conditions
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for cond in COND_LABEL:
        ax.plot(COUNTS, series(cond), "o-", label=COND_LABEL[cond],
                color=COND_COLOR[cond], lw=2)
    lo = [fnum(s["ci95_lo"]) for s in summ if s["condition"] == "MAIN"]
    hi = [fnum(s["ci95_hi"]) for s in summ if s["condition"] == "MAIN"]
    if len(lo) == len(COUNTS):
        ax.fill_between(COUNTS, lo, hi, color=COND_COLOR["MAIN"], alpha=0.15)
    ax.axhline(oracle, color="k", ls="--", lw=1.2,
               label=f"constant-rig focal oracle ({oracle:.2f} %)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(COUNTS)
    ax.set_xticklabels(COUNTS)
    ax.set_xlabel("frames aggregated per view")
    ax.set_ylabel("median relative focal error (%)")
    ax.set_title("Fig 01 - focal recoverable from reference hand geometry")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.text(0.01, -0.04, NOTE, fontsize=7, color="#555")
    save(fig, "fig01_error_vs_frames.png")

    # Fig 02 - comparators
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    names = [c["comparator"] for c in comp] + ["REFERENCE_HAND_N16"]
    vals = [fnum(c["median_rel_focal_err_pct"]) for c in comp] + \
           [series("MAIN")[-1]]
    cols = ["#7f7f7f"] * len(comp) + [COND_COLOR["MAIN"]]
    for i, c in enumerate(comp):
        if c["comparator"] == "CONSTANT_RIG_FOCAL_ORACLE":
            cols[i] = "#000000"
    ax.barh(names, vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.2f} %", va="center", fontsize=8)
    ax.set_xlabel("median relative focal error (%)")
    ax.set_title("Fig 02 - reference-hand ceiling against the frozen baselines")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.text(0.01, -0.06, "the constant-rig oracle is a "
             "POST_HOC_ORACLE_SANITY_CONTROL, not a method", fontsize=7,
             color="#555")
    save(fig, "fig02_comparators.png")

    # Fig 03 - identifiability flags
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    x = np.arange(len(COUNTS))
    ax.bar(x - 0.2, series("MAIN", "share_flat_profile"), 0.4,
           label="flat profile", color="#9ecae1")
    ax.bar(x + 0.2, series("MAIN", "share_boundary"), 0.4,
           label="boundary solution", color="#fdae6b")
    ax.set_xticks(x)
    ax.set_xticklabels(COUNTS)
    ax.set_xlabel("frames aggregated per view")
    ax.set_ylabel("share of views")
    ax.set_title("Fig 03 - is the focal identifiable at all? (main condition)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig03_identifiability.png")

    # Fig 04 - distribution of estimated focal at N=16 vs the reference
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    m = [r for r in est if r["condition"] == "MAIN"
         and int(r["n_frames"]) == 16]
    hat = np.array([fnum(r["focal_hat_px"]) for r in m])
    ref = np.array([fnum(r["gt_reference_focal_px"]) for r in m])
    ax.scatter(ref, hat, s=14, alpha=0.6, color=COND_COLOR["MAIN"])
    lim = [0, max(np.nanpercentile(hat, 99), ref.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1, label="perfect recovery")
    ax.set_xlim(ref.min() * 0.97, ref.max() * 1.03)
    ax.set_ylim(lim)
    ax.set_xlabel("GT_EFFECTIVE_FOCAL of the view (px)")
    ax.set_ylabel("focal estimated from the reference hand (px)")
    ax.set_title("Fig 04 - estimated against reference focal, N = 16")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.text(0.01, -0.05, "the reference focal spans only ~1.9 % across the "
             "rig (SINGLE_FOCAL_RIG_CONFOUND), so the x-axis is nearly a "
             "constant", fontsize=7, color="#555")
    save(fig, "fig04_estimated_vs_reference.png")

    # Fig 05 - error distribution per condition at N=16
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    data, labels, colors = [], [], []
    for cond in COND_LABEL:
        e = [fnum(r["rel_focal_err_pct"]) for r in est
             if r["condition"] == cond and int(r["n_frames"]) == 16]
        e = [v for v in e if np.isfinite(v)]
        if e:
            data.append(e)
            labels.append(COND_LABEL[cond].replace(": ", ":\n"))
            colors.append(COND_COLOR[cond])
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.axhline(oracle, color="k", ls="--", lw=1.2,
               label="constant-rig focal oracle")
    ax.set_yscale("log")
    ax.set_ylabel("relative focal error (%), log scale")
    ax.set_title("Fig 05 - main condition against the three negative controls "
                 "(N = 16)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig05_controls.png")

    # Fig 06 - main explanation panel
    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28)

    a = fig.add_subplot(gs[0, 0])
    for cond in COND_LABEL:
        a.plot(COUNTS, series(cond), "o-", color=COND_COLOR[cond], lw=1.8,
               label=COND_LABEL[cond])
    a.axhline(oracle, color="k", ls="--", lw=1.1)
    a.set_xscale("log", base=2); a.set_xticks(COUNTS)
    a.set_xticklabels(COUNTS)
    a.set_xlabel("frames per view"); a.set_ylabel("median rel. focal err (%)")
    a.set_title("how much focal comes out of the hand", fontsize=10)
    a.legend(fontsize=6.5); a.grid(alpha=0.3)

    b = fig.add_subplot(gs[0, 1])
    b.barh(names, vals, color=cols)
    b.invert_yaxis(); b.tick_params(labelsize=7)
    b.set_xlabel("median rel. focal err (%)")
    b.set_title("against the frozen baselines", fontsize=10)
    b.grid(axis="x", alpha=0.3)

    c = fig.add_subplot(gs[1, 0])
    c.bar(x - 0.2, series("MAIN", "share_flat_profile"), 0.4,
          color="#9ecae1", label="flat")
    c.bar(x + 0.2, series("MAIN", "share_boundary"), 0.4,
          color="#fdae6b", label="boundary")
    c.set_xticks(x); c.set_xticklabels(COUNTS)
    c.set_xlabel("frames per view"); c.set_ylabel("share of views")
    c.set_title("is the focal identifiable at all?", fontsize=10)
    c.legend(fontsize=7); c.grid(axis="y", alpha=0.3)

    d = fig.add_subplot(gs[1, 1]); d.axis("off")
    lines = ["VERDICT", ""]
    for k in ("H1_reference_hand_carries_focal_information",
              "H2_the_information_is_geometric", "H3_more_frames_help"):
        lines.append(f"{'PASS' if verdict[k] else 'FAIL'}  "
                     f"{k.split('_', 1)[0]}: {k.split('_', 1)[1]}")
    lines += [""] + verdict["decision_tags"]
    lines += ["", "SCOPE: upper bound under an oracle reference hand;",
              "not a deployable calibration method."]
    d.text(0, 1, "\n".join(lines), va="top", fontsize=8, family="monospace")

    fig.suptitle("CAM-EXP-006 - how much focal information is in the "
                 "reference hand?", fontsize=12)
    save(fig, "CAM_EXP_006_MAIN_EXPLANATION.png")


if __name__ == "__main__":
    main()
