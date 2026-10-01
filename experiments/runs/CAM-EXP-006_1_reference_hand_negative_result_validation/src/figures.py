"""Figures for CAM-EXP-006.1.

Synthetic and real results are kept in separate panels throughout. Synthetic
evidence validates the implementation and the theory; real evidence is the
empirical result. They are never substituted for one another.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import camera_model_solver as S  # noqa: E402
from common import COUNTS, FIG, RAW, SUM, TAB, fnum, read_csv, read_json  # noqa: E402

SYN = "#2e7d32"
REALC = "#1f4e79"
ORACLE = "#c0504d"


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  " + name)


def main() -> None:
    syn = read_csv(SUM / "synthetic_validation_summary.csv")
    dist = read_csv(SUM / "distance_sweep_summary.csv")
    plan = read_csv(SUM / "planarity_sweep_summary.csv")
    cm_syn = read_csv(SUM / "camera_model_synthetic_summary.csv")
    real = read_csv(SUM / "real_condition_summary.csv")
    track = read_csv(SUM / "real_tracking_summary.csv")
    stress = read_csv(SUM / "nondefault_focal_stress_summary.csv")
    est = read_csv(RAW / "real_validation_view_estimates.csv.gz")
    prof = read_csv(RAW / "real_objective_profiles.csv.gz")
    ev = read_json(SUM / "real_evaluation.json")
    sv = read_json(SUM / "synthetic_validation_verdict.json")
    ps = read_json(SUM / "perspective_strength_verdict.json")
    grid = S.gamma_grid()

    def R(c, n, col):
        return fnum(next(r for r in real if r["condition"] == c
                         and r["n_frames"] == str(n))[col])

    # ---- 01 synthetic positive control ----------------------------------
    sprof = read_csv(RAW / "synthetic_profiles.csv.gz")
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    n = 0
    for r in sprof:
        if r["condition"] != "SYNTH_HAND_STRONG_PERSPECTIVE":
            continue
        c = np.array([fnum(r[f"g{i}"]) for i in range(len(grid))])
        ax.plot(grid, c, color=SYN, alpha=0.25, lw=1)
        n += 1
    ax.axvline(900.0 / 1280, color="k", ls="--", lw=1.6,
               label="true focal (900 px)")
    g = next(r for r in syn
             if r["suite"] == "SYNTH_HAND_STRONG_PERSPECTIVE")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"candidate focal  $\gamma=f/\max(W,H)$")
    ax.set_ylabel("median reprojection error (px)")
    ax.set_title("Fig 01 - synthetic positive control: real hand shapes, "
                 "strong perspective")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.text(0.01, -0.06,
             f"median focal error {fnum(g['median_rel_focal_err_pct']):.2f} %, "
             f"{100*fnum(g['share_within_5pct']):.0f} % within 5 %, "
             f"flat {100*fnum(g['share_flat_profile']):.0f} %  ->  "
             f"the solver works when the focal is identifiable",
             fontsize=7.5, color="#555")
    save(fig, "01_synthetic_positive_control.png")

    # ---- 02 distance vs identifiability ---------------------------------
    d0 = [r for r in dist if r["noise_px"] in ("", "0.0")]
    d1 = [r for r in dist if r["noise_px"] == "1.0"]
    x = [fnum(r["distance_over_diameter"]) for r in d0]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    axes[0].plot(x, [fnum(r["median_profile_width"]) for r in d0], "o-",
                 color=SYN, lw=2)
    axes[0].set_ylabel("profile width (log-gamma)")
    axes[0].set_title("how sharp is the minimum?", fontsize=10)
    axes[1].plot(x, [100 * fnum(r["share_flat_profile"]) for r in d0], "o-",
                 color=SYN, lw=2)
    axes[1].set_ylabel("flat-profile share (%)")
    axes[1].set_title("is it flagged flat?", fontsize=10)
    axes[2].plot(x, [fnum(r["median_rel_focal_err_pct"]) for r in d0], "o-",
                 color="#9ecae1", lw=2, label="noiseless 2D")
    axes[2].plot([fnum(r["distance_over_diameter"]) for r in d1],
                 [fnum(r["median_rel_focal_err_pct"]) for r in d1], "o-",
                 color=SYN, lw=2, label="1 px 2D noise")
    axes[2].set_ylabel("median focal error (%)")
    axes[2].set_title("what it costs in focal error", fontsize=10)
    axes[2].legend(fontsize=7.5)
    real_d = ps["distance_over_diameter"]["median"]
    for a in axes:
        a.set_xscale("log"); a.set_xticks(x)
        a.set_xticklabels([str(int(v)) for v in x])
        a.set_xlabel("camera distance / hand diameter")
        a.axvline(real_d, color=ORACLE, ls=":", lw=1.8)
        a.grid(alpha=0.3)
    axes[0].text(real_d * 1.06, axes[0].get_ylim()[1] * 0.55,
                 f"GigaHands\nmedian {real_d:.1f}x", fontsize=7,
                 color=ORACLE)
    fig.suptitle("Fig 02 - SYNTHETIC: focal identifiability falls away with "
                 "distance", fontsize=12)
    fig.text(0.01, -0.04, "noiseless data still pins the minimum of a flat "
             "curve, so the loss shows up as WIDTH; with noise it becomes "
             "focal error", fontsize=7.5, color="#555")
    save(fig, "02_distance_vs_identifiability.png")

    # ---- 03 planarity sweep ---------------------------------------------
    x = [fnum(r["depth_scale"]) for r in plan]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    axes[0].plot(x, [fnum(r["median_profile_width"]) for r in plan], "o-",
                 color=SYN, lw=2)
    axes[0].set_ylabel("profile width (log-gamma)")
    axes[1].plot(x, [fnum(r["median_rel_focal_err_pct"]) for r in plan], "o-",
                 color=SYN, lw=2)
    axes[1].set_ylabel("median focal error (%)")
    axes[1].set_yscale("log")
    for a in axes:
        a.set_xlabel("depth scale (1.0 = real hand, 0.0 = exactly planar)")
        a.invert_xaxis(); a.grid(alpha=0.3)
        a.axvspan(0.05, -0.02, color="#eee", zorder=0)
    fig.suptitle("Fig 03 - SYNTHETIC: flatten the hand and the focal "
                 "information disappears", fontsize=12)
    fig.text(0.01, -0.05, "the exactly planar endpoint (grey) forces a "
             "different PnP back-end and is diagnostic only; 0.25 and 0.1 "
             "carry the evidence", fontsize=7.5, color="#555")
    save(fig, "03_planarity_sweep.png")

    # ---- 04 frame weighting ---------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for col, ylab, ax, scale in [
            ("median_rel_focal_err_pct", "median focal error (%)", axes[0], 1),
            ("share_flat_profile", "flat-profile share (%)", axes[1], 100),
            ("share_cleanly_identifiable", "cleanly identifiable (%)",
             axes[2], 100)]:
        ax.plot(COUNTS, [scale * R("R0", n, col) for n in COUNTS], "o-",
                color="#7f7f7f", lw=2, label="R0 CAM-006 original (hand-weighted)")
        ax.plot(COUNTS, [scale * R("R1", n, col) for n in COUNTS], "s--",
                color=REALC, lw=2, label="R1 frame-balanced (corrected)")
        ax.set_xscale("log", base=2); ax.set_xticks(COUNTS)
        ax.set_xticklabels(COUNTS); ax.set_xlabel("frames per view")
        ax.set_ylabel(ylab); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle("Fig 04 - REAL: the hand-weighting mismatch was real but "
                 "did not matter", fontsize=12)
    fw = ev["frame_weighting"]
    fig.text(0.01, -0.04,
             f"N=16 difference: {fw['delta_median_err_pp']:.2f} pp error, "
             f"{fw['delta_flat_pp']:.2f} pp flat, "
             f"{fw['delta_identifiable_pp']:.2f} pp identifiable  ->  "
             f"{fw['verdict']}", fontsize=7.5, color="#555")
    save(fig, "04_frame_weighting_check.png")

    # ---- 05 camera-model oracle conditions ------------------------------
    conds = ["R0", "R1", "R2", "R3", "R4", "R5"]
    lbl = {"R0": "R0 original", "R1": "R1 frame-balanced",
           "R2": "R2 +GT principal point", "R3": "R3 +GT distortion",
           "R4": "R4 +GT PP & distortion", "R5": "R5 +GT PP, dist, fy/fx"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    err = [R(c, 16, "median_rel_focal_err_pct") for c in conds]
    flat = [100 * R(c, 16, "share_flat_profile") for c in conds]
    iden = [100 * R(c, 16, "share_cleanly_identifiable") for c in conds]
    cols = [REALC if c in ("R0", "R1") else ORACLE for c in conds]
    axes[0].bar([lbl[c] for c in conds], err, color=cols)
    for i, v in enumerate(err):
        axes[0].text(i, v, f"{v:.0f} %", ha="center", va="bottom", fontsize=8)
    oc = fnum(ev["constant_rig_focal_oracle"]["median_rel_focal_err_pct"])
    axes[0].axhline(oc, color="k", ls="--", lw=1.3,
                    label=f"constant-rig focal oracle ({oc:.2f} %)")
    axes[0].set_ylabel("median focal error (%), N=16")
    axes[0].legend(fontsize=7.5)
    axes[1].bar(np.arange(len(conds)) - 0.2, flat, 0.4, color="#9ecae1",
                label="flat profile")
    axes[1].bar(np.arange(len(conds)) + 0.2, iden, 0.4, color="#fdae6b",
                label="cleanly identifiable")
    axes[1].set_xticks(np.arange(len(conds)))
    axes[1].set_xticklabels(conds)
    axes[1].set_ylabel("share of views (%)")
    axes[1].legend(fontsize=7.5)
    for a in axes:
        a.grid(axis="y", alpha=0.3)
        a.tick_params(axis="x", labelsize=7.5, rotation=20)
    fig.suptitle("Fig 05 - REAL: giving the solver the true camera model "
                 "helps a lot, but not enough", fontsize=12)
    fig.text(0.01, -0.08, "R2-R5 are ORACLE_DIAGNOSTIC_ONLY: they consume "
             "provided camera metadata and are not deployable. The focal "
             "MAGNITUDE is never supplied in any condition.",
             fontsize=7.5, color="#555")
    save(fig, "05_camera_model_oracle_conditions.png")

    # ---- 06 real objective profiles, R1 vs R5 ---------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
    refs = {(r["sequence"], r["camera"]): fnum(r["gt_reference_focal_px"])
            for r in est}
    for ax, cond, title in [(axes[0], "R1", "R1 frame-balanced\n"
                             "(centre PP, no distortion)"),
                            (axes[1], "R5", "R5 full camera-model oracle\n"
                             "(GT PP, distortion, fy/fx)")]:
        cs = []
        for r in prof:
            if r["condition"] != cond:
                continue
            c = np.array([fnum(r[f"g{i}"]) for i in range(len(grid))])
            ax.plot(grid, np.exp(c), color=REALC, alpha=0.18, lw=1)
            cs.append(c)
        if cs:
            ax.plot(grid, np.exp(np.nanmedian(np.array(cs), axis=0)),
                    color=REALC, lw=2.6, label="median over views")
        ax.axvline(np.median(list(refs.values())) / 1280, color="k", ls="--",
                   lw=1.5, label="provided reference focal")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"candidate focal  $\gamma=f/\max(W,H)$")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7.5); ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("aggregated reprojection objective (px)")
    fig.suptitle("Fig 06 - REAL: the objective stays flat even with the true "
                 "camera model", fontsize=12, y=1.06)
    save(fig, "06_real_objective_profiles_original_vs_full_oracle.png")

    # ---- 07 estimated vs reference focal --------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, cond in zip(axes, ("R1", "R5")):
        m = [r for r in est if r["condition"] == cond
             and int(r["n_frames"]) == 16]
        hat = np.array([fnum(r["focal_hat_px"]) for r in m])
        ref = np.array([fnum(r["gt_reference_focal_px"]) for r in m])
        ax.scatter(ref, hat, s=16, alpha=0.6, color=REALC)
        ax.axhline(float(np.median(ref)), color="k", ls=":", lw=1.2,
                   label="rig median reference focal")
        lo, hi = ref.min() * 0.97, ref.max() * 1.03
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect recovery")
        ax.set_xlim(lo, hi); ax.set_yscale("log")
        t = next(x for x in track if x["condition"] == cond
                 and x["n_frames"] == "16")
        ax.set_title(f"{lbl[cond]}\nSpearman {fnum(t['spearman_hat_vs_ref']):+.3f} "
                     f"CI [{fnum(t['spearman_ci95_lo']):+.3f}, "
                     f"{fnum(t['spearman_ci95_hi']):+.3f}]", fontsize=9.5)
        ax.set_xlabel("provided reference focal (px)")
        ax.legend(fontsize=7.5); ax.grid(alpha=0.3)
    axes[0].set_ylabel("estimated focal (px), log scale")
    fig.suptitle("Fig 07 - REAL: the estimates do not track the reference "
                 "focal", fontsize=12)
    fig.text(0.01, -0.06, "the reference focal spans only 1.90 % (CV) across "
             "the rig while the estimates span 65-86 %: scatter, not "
             "collapse to a constant", fontsize=7.5, color="#555")
    save(fig, "07_estimated_vs_reference_focal.png")

    # ---- 08 stress subsets ----------------------------------------------
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    subs = ["S2", "S5"]
    w = 0.35
    for i, c in enumerate(("R1", "R5")):
        vals = [fnum(next(s for s in stress if s["subset"] == t
                          and s["condition"] == c)["median_rel_focal_err_pct"])
                for t in subs]
        ax.bar(np.arange(len(subs)) + (i - 0.5) * w, vals, w,
               color=[REALC, ORACLE][i], label=lbl[c])
    cvals = [fnum(next(s for s in stress if s["subset"] == t
                       and s["condition"] == "R1")
                  ["constant_oracle_median_err_pct"]) for t in subs]
    ax.plot(np.arange(len(subs)), cvals, "k*--", ms=13,
            label="constant-rig focal oracle")
    lbls = []
    for t in subs:
        s = next(s for s in stress if s["subset"] == t
                 and s["condition"] == "R1")
        tag = " UNDERPOWERED" if s["underpowered"] == "1" else ""
        lbls.append(f"{t}\n{s['n_views']} views, "
                    f"{s['n_physical_cameras']} cameras{tag}")
    ax.set_xticks(np.arange(len(subs))); ax.set_xticklabels(lbls, fontsize=8)
    ax.set_yscale("log"); ax.set_ylabel("median focal error (%), N=16")
    ax.set_title("Fig 08 - REAL: views whose focal differs most from the rig "
                 "median")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    save(fig, "08_nondefault_focal_stress.png")

    # ---- 09 perspective strength ----------------------------------------
    cases = read_csv(RAW / "perspective_strength_cases.csv.gz")
    dzz = np.array([fnum(r["dZ_over_Z"]) for r in cases])
    dod = np.array([fnum(r["distance_over_diameter"]) for r in cases])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axes[0].hist(dzz, bins=60, color=REALC, alpha=0.8)
    axes[0].axvline(float(np.median(dzz)), color="k", ls="--",
                    label=f"median {np.median(dzz):.3f}")
    axes[0].set_xlabel(r"relative depth extent  $\Delta Z / Z$")
    axes[0].set_ylabel("hand observations")
    axes[0].set_title("how deep is the hand, relative to its distance?",
                      fontsize=10)
    axes[1].hist(dod, bins=60, range=(0, 15), color=REALC, alpha=0.8)
    axes[1].axvline(float(np.median(dod)), color="k", ls="--",
                    label=f"median {np.median(dod):.2f}x")
    axes[1].set_xlabel("camera distance / hand diameter")
    axes[1].set_title("how far away is the camera?", fontsize=10)
    for a in axes:
        a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.suptitle("Fig 09 - REAL: the imaging regime the hands were actually "
                 "in", fontsize=12)
    fig.text(0.01, -0.06, "ORACLE_PERSPECTIVE_STRENGTH_DIAGNOSTIC: the "
             "provided reference focal is used only to fit the pose for this "
             "causal diagnostic, never as a focal estimator input",
             fontsize=7.5, color="#555")
    save(fig, "09_perspective_strength.png")

    # ---- main explanation -------------------------------------------------
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.3)

    a = fig.add_subplot(gs[0, 0])
    g = next(r for r in syn if r["suite"] == "SYNTH_HAND_STRONG_PERSPECTIVE")
    a.bar(["generic", "real hands"],
          [fnum(next(r for r in syn
                     if r["suite"] == "SYNTH_GENERIC_STRONG_PERSPECTIVE")
                ["median_rel_focal_err_pct"]),
           fnum(g["median_rel_focal_err_pct"])], color=SYN)
    a.set_ylabel("median focal error (%)")
    a.set_title("1. does the solver work?\nSYNTHETIC, strong perspective",
                fontsize=9.5)
    a.text(0.5, 0.75, "SOLVER_VALIDATED\n~0.12 % error", transform=a.transAxes,
           ha="center", fontsize=9, color=SYN, weight="bold")
    a.grid(axis="y", alpha=0.3)

    b = fig.add_subplot(gs[0, 1])
    x = [fnum(r["distance_over_diameter"]) for r in d0]
    b.plot(x, [fnum(r["median_profile_width"]) for r in d0], "o-", color=SYN,
           lw=2)
    b.axvline(real_d, color=ORACLE, ls=":", lw=2)
    b.set_xscale("log"); b.set_xticks(x)
    b.set_xticklabels([str(int(v)) for v in x])
    b.set_xlabel("distance / hand diameter")
    b.set_ylabel("profile width")
    b.set_title("2. why it fails far away\nSYNTHETIC", fontsize=9.5)
    b.grid(alpha=0.3)

    c = fig.add_subplot(gs[0, 2])
    c.plot(COUNTS, [R("R0", n, "median_rel_focal_err_pct") for n in COUNTS],
           "o-", color="#7f7f7f", lw=2, label="R0 original")
    c.plot(COUNTS, [R("R1", n, "median_rel_focal_err_pct") for n in COUNTS],
           "s--", color=REALC, lw=2, label="R1 frame-balanced")
    c.set_xscale("log", base=2); c.set_xticks(COUNTS)
    c.set_xticklabels(COUNTS); c.set_xlabel("frames per view")
    c.set_ylabel("median focal error (%)")
    c.set_title("3. was it the weighting bug?\nREAL - no", fontsize=9.5)
    c.legend(fontsize=7); c.grid(alpha=0.3)

    d = fig.add_subplot(gs[1, 0])
    d.bar([lbl[k].split(" ", 1)[0] for k in conds], err, color=cols)
    d.axhline(oc, color="k", ls="--", lw=1.2)
    d.set_ylabel("median focal error (%)")
    d.set_title("4. was it the camera model?\nREAL - partly", fontsize=9.5)
    d.grid(axis="y", alpha=0.3)

    e = fig.add_subplot(gs[1, 1])
    e.bar(np.arange(len(conds)) - 0.2, flat, 0.4, color="#9ecae1",
          label="flat")
    e.bar(np.arange(len(conds)) + 0.2, iden, 0.4, color="#fdae6b",
          label="identifiable")
    e.set_xticks(np.arange(len(conds))); e.set_xticklabels(conds, fontsize=8)
    e.set_ylabel("share of views (%)")
    e.set_title("5. is it identifiable now?\nREAL - still mostly not",
                fontsize=9.5)
    e.legend(fontsize=7); e.grid(axis="y", alpha=0.3)

    f = fig.add_subplot(gs[1, 2])
    m = [r for r in est if r["condition"] == "R5" and int(r["n_frames"]) == 16]
    hat = np.array([fnum(r["focal_hat_px"]) for r in m])
    ref = np.array([fnum(r["gt_reference_focal_px"]) for r in m])
    f.scatter(ref, hat, s=12, alpha=0.6, color=REALC)
    f.plot([ref.min(), ref.max()], [ref.min(), ref.max()], "k--", lw=1)
    f.set_yscale("log"); f.set_xlabel("reference focal (px)")
    f.set_ylabel("estimated (px)")
    f.set_title("6. does R5 track the focal?\nREAL - no", fontsize=9.5)
    f.grid(alpha=0.3)

    g2 = fig.add_subplot(gs[2, :]); g2.axis("off")
    v = read_json(SUM / "cam0061_verdict.json") if (
        SUM / "cam0061_verdict.json").exists() else {}
    txt = v.get("plain_summary", "see report.md")
    g2.text(0, 1, txt, va="top", fontsize=9, family="monospace")

    fig.suptitle("CAM-EXP-006.1 - was the CAM-EXP-006 negative result real?",
                 fontsize=14)
    save(fig, "CAM_EXP_006_1_MAIN_EXPLANATION.png")


if __name__ == "__main__":
    main()
