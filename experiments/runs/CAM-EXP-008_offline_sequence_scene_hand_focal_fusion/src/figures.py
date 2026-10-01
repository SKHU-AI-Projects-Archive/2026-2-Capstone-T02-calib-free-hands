"""Figures for CAM-EXP-008."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, FIG, RAW, SUM, TAB, fnum, read_csv, read_json  # noqa: E402

SCENE = "#2e7d32"
FUSE = "#1f4e79"
CTRL = "#b0b0b0"
ORACLE = "#c0504d"
PRIMARY = "N64"
NOTE = ("INTERNAL INFORMATION CEILING on the GigaHands development rig - "
        "the hand geometry is not available at deployment")


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  " + name)


def main() -> None:
    pg = read_csv(SUM / "paired_gain_summary.csv")
    folds = read_csv(RAW / "control_fold_predictions.csv.gz")
    so = read_csv(SUM / "scene_only_summary.csv")
    h0 = read_csv(SUM / "hand_only_summary.csv")
    ver = read_json(SUM / "cam008_verdict.json")
    stress = read_csv(SUM / "nondefault_focal_stress_summary.csv")
    lam = read_csv(TAB / "lambda_selection_summary.csv")
    corc = read_csv(SUM / "constant_rig_oracle.csv")[0]

    def P(subset, cond, proto="LOCO_PHYSICAL_CAMERA"):
        return next((r for r in pg if r["subset"] == subset
                     and r["condition"] == cond
                     and r["protocol"] == proto), None)

    def sel(subset, cond, proto="LOCO_PHYSICAL_CAMERA"):
        return [r for r in folds if r["subset"] == subset
                and r["condition"] == cond and r["protocol"] == proto]

    # ---- 01 the core question -------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.axis("off")
    ax.text(0.02, 0.90, "CAM-EXP-008 asks one paired question", fontsize=12,
            weight="bold")
    ax.text(0.03, 0.72, "same recorded video  →  scene calibration  →  "
            "SCENE-ONLY focal", fontsize=11, family="monospace",
            color=SCENE)
    ax.text(0.03, 0.56, "same recorded video  →  scene calibration\n"
            "                     + sequence hand geometry  →  "
            "SCENE+HAND focal", fontsize=11, family="monospace", color=FUSE)
    ax.text(0.03, 0.30, "ONLY DIFFERENCE = THE HAND TERM", fontsize=13,
            weight="bold", color="#b00")
    ax.text(0.03, 0.16, "identical sequence · identical frame IDs · identical "
            "scene predictions\nidentical nuisance parameters · identical "
            "candidate grid", fontsize=9.5, color="#555")
    ax.text(0.03, 0.02, "This is NOT CAM-EXP-006: the hand does not determine "
            "the focal on its own.\nIt only expresses a preference inside the "
            "region the scene already selected.", fontsize=9.5, color="#333")
    ax.set_title("Fig 01 - the paired comparison")
    fig.text(0.01, -0.03, NOTE, fontsize=7.5, color="#555")
    save(fig, "01_cam008_core_question.png")

    # ---- 02 paired scatter ----------------------------------------------
    s = sel(PRIMARY, "S1")
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    if s:
        ref = np.array([fnum(r["f_reference"]) for r in s])
        e0 = 100 * np.abs(np.array([fnum(r["f_scene_only"]) for r in s])
                          - ref) / ref
        e1 = 100 * np.abs(np.array([fnum(r["f_fused"]) for r in s])
                          - ref) / ref
        ax.scatter(e0, e1, s=22, alpha=0.7, color=FUSE)
        m = max(e0.max(), e1.max()) * 1.05
        ax.plot([0, m], [0, m], "k--", lw=1.2, label="no change")
        ax.set_xlim(0, m); ax.set_ylim(0, m)
        ax.set_xlabel("scene-only focal error (%)")
        ax.set_ylabel("scene+hand focal error (%)")
        better = float(np.mean(e1 < e0)) * 100
        ax.set_title(f"Fig 02 - paired, per sequence-camera unit\n"
                     f"{better:.0f} % of units below the line "
                     f"(fusion helped)", fontsize=10)
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.text(0.01, -0.04, "points below the dashed line = the hand term "
             "improved that video", fontsize=7.5, color="#555")
    save(fig, "02_paired_scene_vs_scene_hand.png")

    # ---- 03 paired gain distribution ------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    if s:
        d = e0 - e1
        cams = np.array([r["physical_camera_id"] for r in s])
        order = np.argsort([np.median(d[cams == c]) for c in np.unique(cams)])
        uc = np.unique(cams)[order]
        data = [d[cams == c] for c in uc]
        bp = ax.boxplot(data, showfliers=False, patch_artist=True,
                        tick_labels=[c.replace("brics-odroid-", "")
                                     for c in uc])
        for p in bp["boxes"]:
            p.set_facecolor(FUSE); p.set_alpha(0.5)
        ax.axhline(0, color="k", lw=1.2)
        ax.set_ylabel("paired gain (pp)\nscene-only err - scene+hand err")
        ax.tick_params(axis="x", labelsize=6, rotation=90)
        pr = ver.get("primary_paired") or {}
        ax.set_title(f"Fig 03 - paired gain by held-out physical camera "
                     f"(median {pr.get('median_paired_gain_pp', float('nan')):+.4f} pp, "
                     f"CI [{pr.get('gain_ci_lo', float('nan')):+.4f}, "
                     f"{pr.get('gain_ci_hi', float('nan')):+.4f}])",
                     fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "03_paired_gain_distribution.png")

    # ---- 04 frame count --------------------------------------------------
    subs = [x for x in ("N8", "N16", "N32", "N64") if P(x, "S1")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axes[0].plot(subs, [fnum(P(x, "S1")["S0_median_err_pct"]) for x in subs],
                 "o-", color=SCENE, lw=2, label="scene-only")
    axes[0].plot(subs, [fnum(P(x, "S1")["S1_median_err_pct"]) for x in subs],
                 "s--", color=FUSE, lw=2, label="scene+hand")
    axes[0].set_ylabel("median focal error (%)")
    axes[0].legend(fontsize=8)
    axes[1].bar(subs, [fnum(P(x, "S1")["median_paired_gain_pp"])
                       for x in subs], color=FUSE)
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set_ylabel("paired gain (pp)")
    for a in axes:
        a.set_xlabel("frames used per video (offline evidence accumulation)")
        a.grid(alpha=0.3)
    fig.suptitle("Fig 04 - how much temporal evidence the hand term needs",
                 fontsize=12, y=1.02)
    fig.text(0.01, -0.05, "this is OFFLINE evidence accumulation over an "
             "already-recorded video, not a realtime update", fontsize=7.5,
             color="#555")
    save(fig, "04_frame_count.png")

    # ---- 05 hand profile examples ---------------------------------------
    prof = sorted((CACHE / "hand_profiles" / "real").glob("*.npz"))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    picks = []
    for p in prof[:60]:
        z = np.load(p, allow_pickle=True)
        c = np.asarray(z["curves"], float)
        if c.size == 0:
            continue
        C = np.nanmedian(c, axis=0)
        if not np.isfinite(C).any():
            continue
        L = C - np.nanmin(C)
        picks.append((float(np.nanmax(L)), p.stem, np.asarray(z["grid"],
                                                              float), L,
                      float(z["f_scene"])))
    picks.sort(key=lambda t: -t[0])
    show = [picks[0], picks[len(picks) // 2], picks[-1]] if len(picks) >= 3 \
        else picks
    titles = ["most informative", "typical", "flattest"]
    for ax, (depth, name, grid, L, fs), t in zip(axes, show, titles):
        ax.plot(grid, L, color=FUSE, lw=2)
        ax.axvline(1.0, color=SCENE, ls="--", lw=1.5, label="scene focal")
        ax.set_xlabel("candidate focal / scene focal")
        ax.set_title(f"{t}\n{name[:28]}", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("hand reprojection excess (px)")
    fig.suptitle("Fig 05 - what the hand term actually contributes",
                 fontsize=12, y=1.04)
    fig.text(0.01, -0.06, "a flat curve cannot move the scene estimate, which "
             "is why L_hand is an absolute pixel excess and is never "
             "per-view min-max normalised", fontsize=7.5, color="#555")
    save(fig, "05_hand_profile_examples.png")

    # ---- 06 controls -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    conds = [("S1", "real hand"), ("C1", "wrong-frame hand"),
             ("C2", "view-shuffled hand")]
    vals, los, his = [], [], []
    for c, _ in conds:
        p = P(PRIMARY, c)
        vals.append(fnum(p["median_paired_gain_pp"]) if p else np.nan)
        los.append(fnum(p["gain_ci_lo"]) if p else np.nan)
        his.append(fnum(p["gain_ci_hi"]) if p else np.nan)
    x = np.arange(len(conds))
    ax.bar(x, vals, color=[FUSE, CTRL, CTRL])
    ax.errorbar(x, vals, yerr=[np.array(vals) - np.array(los),
                               np.array(his) - np.array(vals)],
                fmt="none", ecolor="k", capsize=4)
    ax.axhline(0, color="k", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in conds], fontsize=9)
    ax.set_ylabel("median paired gain (pp)")
    ax.set_title("Fig 06 - does the gain survive destroying the hand "
                 "correspondence?")
    ax.grid(axis="y", alpha=0.3)
    fig.text(0.01, -0.05, "if the real bar is not clearly above the control "
             "bars, the gain is not attributable to hand geometry",
             fontsize=7.5, color="#555")
    save(fig, "06_shuffle_wrong_pose_controls.png")

    # ---- 07 pose diversity ----------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, n in zip(axes, (16, 32)):
        rows = [(f"LOW{n}", "low diversity"), (f"DIVERSE{n}", "high diversity")]
        g = [fnum(P(a, "S1")["median_paired_gain_pp"]) if P(a, "S1") else np.nan
             for a, _ in rows]
        ax.bar([b for _, b in rows], g, color=[CTRL, FUSE])
        ax.axhline(0, color="k", lw=1)
        ax.set_ylabel("median paired gain (pp)")
        ax.set_title(f"N = {n}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Fig 07 - does hand-pose variety make the hand term more "
                 "useful?", fontsize=12, y=1.02)
    fig.text(0.01, -0.05, "compared as PAIRED gains, because the two subsets "
             "contain different frames and their scene baselines differ",
             fontsize=7.5, color="#555")
    save(fig, "07_pose_diversity.png")

    # ---- 08 stress subsets ----------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.2))
    subs2 = ["S2", "S5"]
    w = 0.35
    for i, (c, lab) in enumerate((("S1", "scene+hand"), ("C1", "wrong-frame"))):
        v = []
        for t in subs2:
            m = [r for r in stress if r["subset"] == t and r["condition"] == c]
            v.append(fnum(m[0]["median_gain_pp"]) if m else np.nan)
        ax.bar(np.arange(len(subs2)) + (i - 0.5) * w, v, w,
               color=[FUSE, CTRL][i], label=lab)
    labs = []
    for t in subs2:
        m = [r for r in stress if r["subset"] == t and r["condition"] == "S1"]
        if m:
            tag = " UNDERPOWERED" if m[0]["underpowered"] == "1" else ""
            labs.append(f"{t}\n{m[0]['n_views']} views, "
                        f"{m[0]['n_physical_cameras']} cameras{tag}")
        else:
            labs.append(t)
    ax.set_xticks(np.arange(len(subs2)))
    ax.set_xticklabels(labs, fontsize=8)
    ax.axhline(0, color="k", lw=1)
    ax.set_ylabel("median paired gain (pp)")
    ax.set_title("Fig 08 - cameras whose focal differs most from the rig "
                 "median")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    save(fig, "08_nondefault_focal_stress.png")

    # ---- 09 lambda -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ls = [fnum(r["lambda"]) for r in lam
          if r["subset"] == PRIMARY and r["condition"] == "S1"
          and r["protocol"] == "LOCO_PHYSICAL_CAMERA"]
    if ls:
        u, cnt = np.unique(ls, return_counts=True)
        ax.bar([str(v) for v in u], cnt, color=FUSE)
    ax.set_xlabel("lambda selected inside the training cameras")
    ax.set_ylabel("outer folds")
    ax.set_title("Fig 09 - lambda was never chosen on the held-out camera")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "09_lambda_selection.png")

    # ---- main explanation -------------------------------------------------
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(3, 3, hspace=0.62, wspace=0.32)

    a = fig.add_subplot(gs[0, 0]); a.axis("off")
    a.text(0, 1, "1. THE QUESTION\n\nSame recorded video.\n"
                 "Scene calibration picks a focal.\n\n"
                 "Does adding the hand geometry\nobserved across the whole "
                 "video\npick a better one?\n\n"
                 "ONLY DIFFERENCE = HAND TERM", va="top", fontsize=9.5)

    b = fig.add_subplot(gs[0, 1])
    if s:
        b.scatter(e0, e1, s=14, alpha=0.65, color=FUSE)
        m = max(e0.max(), e1.max()) * 1.05
        b.plot([0, m], [0, m], "k--", lw=1)
        b.set_xlim(0, m); b.set_ylim(0, m)
    b.set_xlabel("scene-only err (%)"); b.set_ylabel("scene+hand err (%)")
    b.set_title("2. paired, per video", fontsize=10); b.grid(alpha=0.3)

    c = fig.add_subplot(gs[0, 2])
    c.bar([n for _, n in conds], vals, color=[FUSE, CTRL, CTRL])
    c.axhline(0, color="k", lw=1)
    c.set_ylabel("median gain (pp)")
    c.set_title("3. controls", fontsize=10)
    c.tick_params(axis="x", labelsize=7, rotation=12)
    c.grid(axis="y", alpha=0.3)

    d = fig.add_subplot(gs[1, 0])
    d.plot(subs, [fnum(P(x, "S1")["S0_median_err_pct"]) for x in subs], "o-",
           color=SCENE, lw=2, label="scene-only")
    d.plot(subs, [fnum(P(x, "S1")["S1_median_err_pct"]) for x in subs], "s--",
           color=FUSE, lw=2, label="scene+hand")
    d.set_ylabel("median err (%)")
    d.set_title("4. frames per video", fontsize=10)
    d.legend(fontsize=7); d.grid(alpha=0.3)

    e = fig.add_subplot(gs[1, 1])
    names = ["scene-only", "scene+hand", "hand-only", "constant\nrig oracle"]
    pr = ver.get("primary_paired") or {}
    hv = next((fnum(r["median_rel_focal_err_pct"]) for r in h0
               if r["subset"] == PRIMARY), np.nan)
    e.bar(names, [pr.get("S0_median_err_pct", np.nan),
                  pr.get("S1_median_err_pct", np.nan), hv,
                  fnum(corc["median_rel_focal_err_pct"])],
          color=[SCENE, FUSE, "#999", ORACLE])
    e.set_ylabel("median err (%)")
    e.set_yscale("log")
    e.set_title("5. all conditions", fontsize=10)
    e.tick_params(axis="x", labelsize=7)
    e.grid(axis="y", alpha=0.3)

    f = fig.add_subplot(gs[1, 2])
    for n, col in ((16, CTRL), (32, FUSE)):
        gg = [fnum(P(f"LOW{n}", "S1")["median_paired_gain_pp"])
              if P(f"LOW{n}", "S1") else np.nan,
              fnum(P(f"DIVERSE{n}", "S1")["median_paired_gain_pp"])
              if P(f"DIVERSE{n}", "S1") else np.nan]
        f.plot(["low", "high"], gg, "o-", color=col, lw=2, label=f"N={n}")
    f.axhline(0, color="k", lw=1)
    f.set_ylabel("paired gain (pp)")
    f.set_title("6. pose diversity", fontsize=10)
    f.legend(fontsize=7); f.grid(alpha=0.3)

    g2 = fig.add_subplot(gs[2, :]); g2.axis("off")
    cr = ver.get("criteria", {})
    g2.text(0, 1,
            "VERDICT  " + " | ".join(ver.get("decision_tags", [])) + "\n\n"
            f"mode: {ver.get('mode')}  (the full-sequence run was a "
            f"pre-registered compute blocker; this is a 64-frame "
            f"full-duration approximation)\n\n"
            f"primary {PRIMARY} / leave-one-physical-camera-out:  "
            f"scene-only {pr.get('S0_median_err_pct')} %   "
            f"scene+hand {pr.get('S1_median_err_pct')} %   "
            f"median paired gain {pr.get('median_paired_gain_pp')} pp  "
            f"CI [{pr.get('gain_ci_lo')}, {pr.get('gain_ci_hi')}]\n\n"
            f"criteria: beats_S0={cr.get('cond1_beats_S0_by_threshold')}  "
            f"CI_excludes_0={cr.get('cond2_gain_ci_excludes_zero')}  "
            f"beats_controls={cr.get('cond3_beats_controls')}  "
            f"not_collapse={cr.get('cond4_not_constant_collapse')}\n\n"
            "Scope: the hand geometry here is other-camera-only reference 3D "
            "paired with dataset-provided 2D. That is an internal information "
            "ceiling,\nnot something a single deployed camera has.",
            va="top", fontsize=9, family="monospace")

    fig.suptitle("CAM-EXP-008 - does sequence-level hand geometry improve a "
                 "scene-anchored focal estimate?", fontsize=14)
    save(fig, "CAM_EXP_008_MAIN_EXPLANATION.png")


if __name__ == "__main__":
    main()
