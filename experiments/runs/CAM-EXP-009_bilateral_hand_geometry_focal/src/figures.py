"""Figures for CAM-EXP-009 (synthetic phase only — the gate failed)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIG, RAW, SUM, fnum, read_csv, read_json, q_grid  # noqa: E402

CAND = "#1f4e79"
FIXED = "#b0b0b0"
TRUTH = "#c0504d"


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  " + name)


def main() -> None:
    audit = read_json(SUM / "focal_dependence_audit.json")
    curve = read_csv(RAW / "focal_dependence_curve.csv")
    summ = read_csv(SUM / "synthetic_summary.csv")
    gate = read_json(SUM / "synthetic_gate.json")
    grid = q_grid()

    q = np.array([fnum(r["q"]) for r in curve])
    cand = np.array([fnum(r["candidate_conditioned_bilateral"])
                     for r in curve])
    fixed = np.array([fnum(r["fixed_length_bilateral"]) for r in curve])
    eL = np.array([fnum(r["left_reproj_px"]) for r in curve])

    # ---- 01 the core idea, and why the naive version cannot work --------
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.axis("off")
    ax.text(0.02, 0.93, "The idea under test", fontsize=12, weight="bold")
    ax.text(0.03, 0.80, "the same person's LEFT and RIGHT hands have related\n"
            "bone proportions:  LEFT 3-4 <-> RIGHT 3-4,  LEFT 7-8 <-> "
            "RIGHT 7-8,  ...", fontsize=10, family="monospace")
    ax.text(0.02, 0.62, "Why the obvious version cannot work", fontsize=12,
            weight="bold", color=TRUTH)
    ax.text(0.03, 0.48, "compare the FIXED reference 3D bone lengths  ->  "
            "the value does not depend on f at all\n"
            "measured range over a +-50 % focal sweep:  "
            f"{audit['claim_1_fixed_length_bilateral_is_focal_invariant']['value_range_over_+-50pct_focal_sweep']:.1e}",
            fontsize=10, family="monospace", color=TRUTH)
    ax.text(0.02, 0.30, "What CAM-EXP-009 does instead", fontsize=12,
            weight="bold", color=CAND)
    ax.text(0.03, 0.14, "at EVERY candidate focal, refit the shared bone\n"
            "proportions from each side's own 2D, then compare:\n"
            "     L_bilateral(f) = || l_L(f) - l_R(f) ||",
            fontsize=10, family="monospace", color=CAND)
    ax.set_title("Fig 01 - the bilateral core idea")
    save(fig, "01_bilateral_core_idea.png")

    # ---- 02 fixed vs candidate-conditioned ------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    axes[0].plot(q, fixed, color=FIXED, lw=2.5)
    axes[0].axvline(1.0, color=TRUTH, ls="--", lw=1.5, label="true focal")
    axes[0].set_title("fixed reference lengths\nFOCAL-INVARIANT (range = 0)",
                      fontsize=10)
    axes[0].set_ylabel("bilateral distance")
    axes[1].plot(q, cand, color=CAND, lw=2.5)
    axes[1].axvline(1.0, color=TRUTH, ls="--", lw=1.5, label="true focal")
    axes[1].axvline(grid[-1], color="k", ls=":", lw=1.5,
                    label="grid boundary (where it actually minimises)")
    axes[1].set_title("candidate-conditioned refit\nfocal-DEPENDENT, but "
                      "minimised at the boundary", fontsize=10)
    for a in axes:
        a.set_xlabel("candidate focal / true focal")
        a.legend(fontsize=7.5)
        a.grid(alpha=0.3)
    fig.suptitle("Fig 02 - the objective does depend on the focal, but it "
                 "does not prefer the right one", fontsize=12, y=1.03)
    fig.text(0.01, -0.05, "this is the whole result: focal DEPENDENCE was "
             "achieved, focal IDENTIFIABILITY was not", fontsize=7.5,
             color="#555")
    save(fig, "02_fixed_vs_candidate_conditioned.png")

    # ---- 03 why: reprojection itself runs to the boundary ---------------
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    ax.plot(q, eL, color=CAND, lw=2.5, label="reprojection error of the fit")
    ax.axvline(1.0, color=TRUTH, ls="--", lw=1.5, label="true focal")
    ax.set_xlabel("candidate focal / true focal")
    ax.set_ylabel("median reprojection error (px)")
    ax.set_title("Fig 03 - with 20 free bone lengths, a larger focal always "
                 "fits better")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.text(0.01, -0.05, "the fit reshapes the hand to absorb the focal "
             "error, so the data never argues for the true focal",
             fontsize=7.5, color="#555")
    save(fig, "03_synthetic_positive_control.png")

    # ---- 04/05 sweeps ----------------------------------------------------
    def grp(prefix):
        return [r for r in summ if r["condition"].startswith(prefix)]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    for ax, (pref, xlab) in zip(axes, [("DIST_", "distance / hand diameter"),
                                       ("ASYM_", "bilateral asymmetry"),
                                       ("NOISE_", "2D noise (px)")]):
        rows = grp(pref)
        names = [r["condition"].replace(pref, "") for r in rows]
        ax.bar(names, [fnum(r["bilat_median_focal_err_pct"]) for r in rows],
               color=CAND)
        ax.axhline(5, color=TRUTH, ls="--", lw=1.4, label="5 % gate")
        ax.set_xlabel(xlab)
        ax.set_ylabel("median focal error (%)")
        ax.tick_params(axis="x", labelsize=8)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Fig 04 - every sweep sits at the 50 % ceiling: the minimum "
                 "is always at the grid boundary", fontsize=12, y=1.04)
    save(fig, "04_synthetic_asymmetry.png")

    fig, ax = plt.subplots(figsize=(9, 4.3))
    rows = grp("DIST_")
    ax.plot([r["condition"].replace("DIST_", "") for r in rows],
            [fnum(r["bilat_boundary_rate"]) * 100 for r in rows], "o-",
            color=CAND, lw=2, label="bilateral")
    ax.plot([r["condition"].replace("DIST_", "") for r in rows],
            [fnum(r["heldreproj_boundary_rate"]) * 100 for r in rows], "s--",
            color="#888", lw=2, label="held-out reprojection")
    ax.set_xlabel("distance / hand diameter")
    ax.set_ylabel("share of trials minimising at the boundary (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Fig 05 - the held-out split did not restore "
                 "identifiability")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "05_synthetic_distance_noise.png")

    # ---- main explanation ------------------------------------------------
    fig = plt.figure(figsize=(14, 7.6))
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.3)

    a = fig.add_subplot(gs[0, 0])
    a.plot(q, fixed, color=FIXED, lw=2.5)
    a.axvline(1.0, color=TRUTH, ls="--", lw=1.4)
    a.set_title("1. fixed lengths:\nflat, range = 0", fontsize=10)
    a.set_xlabel("f / f_true"); a.grid(alpha=0.3)

    b = fig.add_subplot(gs[0, 1])
    b.plot(q, cand, color=CAND, lw=2.5)
    b.axvline(1.0, color=TRUTH, ls="--", lw=1.4)
    b.set_title("2. candidate-conditioned:\nvaries, but runs to the edge",
                fontsize=10)
    b.set_xlabel("f / f_true"); b.grid(alpha=0.3)

    c = fig.add_subplot(gs[0, 2])
    c.plot(q, eL, color=CAND, lw=2.5)
    c.axvline(1.0, color=TRUTH, ls="--", lw=1.4)
    c.set_title("3. why: reprojection itself\nprefers a bigger focal",
                fontsize=10)
    c.set_xlabel("f / f_true"); c.grid(alpha=0.3)

    d = fig.add_subplot(gs[1, :]); d.axis("off")
    d.text(0, 1,
           "VERDICT  " + gate["verdict"] + "\n\n"
           f"  g1 bilateral objective is NOT focal-invariant            "
           f"{gate['g1_bilateral_not_focal_invariant']}\n"
           f"  g2 true focal near the minimum (noiseless)               "
           f"{gate['g2_true_focal_near_minimum_noiseless']}"
           f"   (best over all distances: "
           f"{gate['g2_best_case_any_distance']} % error)\n"
           f"  g3 held-out reprojection identifies the focal            "
           f"{gate['g3_held_out_reprojection_identifies_focal']}\n"
           f"  g4 wrong bone correspondence degrades                    "
           f"{gate['g4_wrong_correspondence_degrades']}\n"
           f"  g5 fixed-length control is flat                          "
           f"{gate['g5_fixed_length_control_is_flat']}\n\n"
           "The idea is coherent and the naive version is provably useless, "
           "which is worth knowing.\n"
           "But the candidate-conditioned version buys focal DEPENDENCE at "
           "the cost of 20 free bone\n"
           "lengths per side, and those absorb the focal error: a larger "
           "focal always explains the\n"
           "images better. The minimum sits at the search boundary in "
           "90-100 % of trials, at every\n"
           "distance, asymmetry and noise level tested.\n\n"
           "Pre-registered consequence: the GigaHands real-data phase was "
           "NOT RUN.",
           va="top", fontsize=9, family="monospace")

    fig.suptitle("CAM-EXP-009 - is bilateral bone consistency a focal cue?",
                 fontsize=14)
    save(fig, "CAM_EXP_009_MAIN_EXPLANATION.png")


if __name__ == "__main__":
    main()
