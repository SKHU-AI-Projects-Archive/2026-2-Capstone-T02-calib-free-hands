"""Figures for CAM-EXP-007."""
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

HAND = "#1f4e79"
SCENE = "#2e7d32"
BASE = "#7f7f7f"
ORACLE = "#c0504d"
SHUF = "#b0b0b0"
NOTE = ("LEARNED_HAND_CAMERA_INFORMATION_DIAGNOSTIC - no network was trained "
        "and no calibration method is proposed")


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  " + name)


def main() -> None:
    loco = {r["probe"]: r for r in read_csv(SUM / "loco_probe_summary.csv")}
    loso = {r["probe"]: r for r in read_csv(SUM / "loso_probe_summary.csv")}
    ver = read_json(SUM / "cam007_verdict.json")
    preds = read_csv(RAW / "probe_fold_predictions.csv.gz")
    assoc = read_csv(SUM / "explicit_feature_associations.csv")
    inc = read_csv(SUM / "scene_incremental_summary.csv")
    stress = read_csv(SUM / "nondefault_focal_stress_summary.csv")
    cov = ver["coverage"]

    def M(p, k):
        return fnum(loco[p][k])

    # ---- 01 the question --------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.axis("off")
    ax.text(0.02, 0.92, "CAM-EXP-006 / 006.1 asked:", fontsize=11,
            weight="bold")
    ax.text(0.04, 0.78, "RGB  ->  reference 3D hand  ->  profiled PnP  ->  "
            "focal", fontsize=11, family="monospace")
    ax.text(0.04, 0.66, "answer: the focal is not identifiable this way "
            "under this regime", fontsize=10, color=ORACLE)
    ax.text(0.02, 0.46, "CAM-EXP-007 asks a DIFFERENT question:", fontsize=11,
            weight="bold")
    ax.text(0.04, 0.32, "RGB  ->  frozen hand model  ->  outputs / latent  "
            "->  can a linear probe predict\n"
            "                                             AnyCalib's "
            "view-level focal residual?", fontsize=11, family="monospace")
    ax.text(0.04, 0.12, "This is NOT the CAM-006 PnP solver repeated with a "
            "predicted hand.\nIt asks whether the learned representation "
            "holds camera information at all.", fontsize=10, color=HAND)
    ax.set_title("Fig 01 - what CAM-EXP-007 actually tests")
    fig.text(0.01, -0.02, NOTE, fontsize=7.5, color="#555")
    save(fig, "01_cam007_question.png")

    # ---- 02 LOCO performance by feature group ----------------------------
    order = ["B0", "B1", "B2", "P1", "P2", "P3", "P4", "P5", "P6", "P7",
             "P8", "P9", "P10", "C1", "B3"]
    lbl = {"B0": "B0 raw AnyCalib", "B1": "B1 global-bias corrected",
           "B2": "B2 CAM-005 scene", "P1": "F0 box/visibility",
           "P2": "F1 pred_cam head", "P3": "F2 hand geometry",
           "P4": "F3 temporal", "P5": "all explicit",
           "P6": "latent", "P7": "explicit+latent",
           "P8": "scene+explicit", "P9": "scene+latent",
           "P10": "scene+all hand", "C1": "C1 AnyCalib-focal only",
           "B3": "B3 constant-reference oracle"}
    cols = []
    for p in order:
        cols.append(ORACLE if p in ("C1", "B3") else
                    BASE if p.startswith("B") else
                    SCENE if p in ("P8", "P9", "P10") else HAND)
    vals = [M(p, "median_rel_focal_err_pct") for p in order]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.barh([lbl[p] for p in order], vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)
    ax.axvline(M("B1", "median_rel_focal_err_pct"), color="k", ls="--", lw=1.3,
               label="B1 practical baseline")
    ax.invert_yaxis()
    ax.set_xlabel("median relative focal error (%), leave-one-physical-"
                  "camera-out")
    ax.set_title("Fig 02 - no hand feature group beats the global-bias "
                 "baseline")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.text(0.01, -0.03, "red = oracle / confound controls, not deployable "
             "methods", fontsize=7.5, color="#555")
    save(fig, "02_feature_group_loco_performance.png")

    # ---- 03 bias prediction scatter --------------------------------------
    best = min(["P1", "P2", "P3", "P4", "P5", "P6", "P7"],
               key=lambda p: M(p, "median_rel_focal_err_pct"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, p in zip(axes, [best, "B2"]):
        rows = [r for r in preds
                if r["protocol"] == "LOCO_PHYSICAL_CAMERA"
                and r["probe"] == p]
        yt = np.array([fnum(r["y_true"]) for r in rows])
        yp = np.array([fnum(r["y_pred"]) for r in rows])
        ax.scatter(yt, yp, s=16, alpha=0.6,
                   color=HAND if p != "B2" else SCENE)
        lim = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        ax.plot(lim, lim, "k--", lw=1, label="perfect")
        ax.set_xlabel("actual signed log AnyCalib bias")
        ax.set_title(f"{lbl.get(p, p)}\nSpearman "
                     f"{M(p, 'spearman_pred_vs_actual'):+.3f} "
                     f"CI [{M(p, 'spearman_ci_lo'):+.3f}, "
                     f"{M(p, 'spearman_ci_hi'):+.3f}]", fontsize=10)
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("predicted signed log bias")
    fig.suptitle("Fig 03 - predicted against actual residual bias, unseen "
                 "physical cameras", fontsize=12, y=1.03)
    save(fig, "03_bias_prediction_scatter.png")

    # ---- 04 corrected focal error ----------------------------------------
    sel = ["B0", "B1", "B2", "P5", "P6", "P10"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    axes[0].bar([lbl[p] for p in sel],
                [M(p, "median_rel_focal_err_pct") for p in sel],
                color=[BASE, BASE, SCENE, HAND, HAND, SCENE])
    axes[0].set_ylabel("median relative focal error (%)")
    axes[1].bar([lbl[p] for p in sel],
                [100 * M(p, "within_5pct") for p in sel],
                color=[BASE, BASE, SCENE, HAND, HAND, SCENE])
    axes[1].set_ylabel("within +-5 % (%)")
    for a in axes:
        a.tick_params(axis="x", labelsize=7.5, rotation=25)
        a.grid(axis="y", alpha=0.3)
    fig.suptitle("Fig 04 - corrected focal on the common eligible view set",
                 fontsize=12, y=1.03)
    save(fig, "04_corrected_focal_error.png")

    # ---- 05 shuffle control ----------------------------------------------
    sh = ver["shuffle_control_medians"]
    ps = sorted(sh)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(ps))
    axes[0].bar(x - 0.2, [M(p, "median_rel_focal_err_pct") for p in ps], 0.4,
                color=HAND, label="real hand features")
    axes[0].bar(x + 0.2, [sh[p]["median_rel_focal_err_pct"] for p in ps], 0.4,
                color=SHUF, label="view-shuffled")
    axes[0].set_ylabel("median relative focal error (%)")
    axes[1].bar(x - 0.2, [M(p, "spearman_pred_vs_actual") for p in ps], 0.4,
                color=HAND, label="real")
    axes[1].bar(x + 0.2, [sh[p]["spearman"] for p in ps], 0.4, color=SHUF,
                label="shuffled")
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set_ylabel("Spearman, predicted vs actual bias")
    for a in axes:
        a.set_xticks(x)
        a.set_xticklabels([lbl.get(p, p) for p in ps], fontsize=8)
        a.legend(fontsize=8)
        a.grid(axis="y", alpha=0.3)
    fig.suptitle("Fig 05 - destroying the hand-view correspondence changes "
                 "almost nothing", fontsize=12, y=1.03)
    fig.text(0.01, -0.05, "if a real hand cue were being used, shuffling the "
             "features should clearly hurt. It does not.", fontsize=7.5,
             color="#555")
    save(fig, "05_shuffle_control.png")

    # ---- 06 latent vs explicit -------------------------------------------
    sel = ["P5", "P6", "P7"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    for ax, k, t in [(axes[0], "median_rel_focal_err_pct",
                      "median focal error (%)"),
                     (axes[1], "bias_mae", "bias MAE (log units)"),
                     (axes[2], "spearman_pred_vs_actual", "Spearman")]:
        ax.bar([lbl[p] for p in sel], [M(p, k) for p in sel], color=HAND)
        ax.set_ylabel(t)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        if k == "spearman_pred_vs_actual":
            ax.axhline(0, color="k", lw=1)
    fig.suptitle("Fig 06 - the learned latent adds nothing over the explicit "
                 "outputs", fontsize=12, y=1.04)
    save(fig, "06_latent_vs_explicit.png")

    # ---- 07 scene + hand increment ---------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.2))
    names = ["B2 scene only"] + [lbl[r["probe"]] for r in inc]
    vals = [M("B2", "median_rel_focal_err_pct")] + \
           [fnum(r["with_hand_median_err"]) for r in inc]
    ax.bar(names, vals, color=[SCENE] + [HAND] * len(inc))
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(vals[0], color=SCENE, ls="--", lw=1.2)
    ax.set_ylabel("median relative focal error (%)")
    ax.set_title("Fig 07 - adding hand features to the scene baseline does "
                 "not help")
    ax.tick_params(axis="x", labelsize=8, rotation=12)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "07_scene_plus_hand_increment.png")

    # ---- 08 focal tracking -----------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, p in zip(axes, ["B1", "P5", "P6"]):
        rows = [r for r in preds
                if r["protocol"] == "LOCO_PHYSICAL_CAMERA"
                and r["probe"] == p]
        fc = np.array([fnum(r["f_corrected"]) for r in rows])
        fr = np.array([fnum(r["f_reference"]) for r in rows])
        ax.scatter(fr, fc, s=14, alpha=0.6, color=HAND)
        ax.plot([fr.min(), fr.max()], [fr.min(), fr.max()], "k--", lw=1)
        ax.set_xlabel("reference focal (px)")
        ax.set_title(f"{lbl[p]}\ncorrected CV "
                     f"{M(p, 'corrected_focal_cv_pct'):.2f} % vs reference CV "
                     f"{M(p, 'reference_focal_cv_pct'):.2f} %", fontsize=9.5)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("corrected focal (px)")
    fig.suptitle("Fig 08 - constant-collapse check", fontsize=12, y=1.03)
    save(fig, "08_focal_tracking.png")

    # ---- 09 stress subsets -----------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    subs = ["S2", "S5"]
    probes = ["B1", "B2", "P5", "P6"]
    w = 0.2
    for i, p in enumerate(probes):
        vals = []
        for s in subs:
            m = [r for r in stress if r["subset"] == s and r["probe"] == p]
            vals.append(fnum(m[0]["median_rel_focal_err_pct"]) if m else np.nan)
        ax.bar(np.arange(len(subs)) + (i - 1.5) * w, vals, w, label=lbl[p])
    labs = []
    for s in subs:
        m = [r for r in stress if r["subset"] == s and r["probe"] == "B1"][0]
        tag = " UNDERPOWERED" if m["underpowered"] == "1" else ""
        labs.append(f"{s}\n{m['n_views']} views, "
                    f"{m['n_physical_cameras']} cameras{tag}")
    ax.set_xticks(np.arange(len(subs)))
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("median relative focal error (%)")
    ax.set_title("Fig 09 - views whose focal differs most from the rig median")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "09_nondefault_focal_stress.png")

    # ---- main explanation -------------------------------------------------
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(3, 3, hspace=0.6, wspace=0.3)

    a = fig.add_subplot(gs[0, 0])
    a.axis("off")
    a.text(0, 1, "1. WHY THE QUESTION CHANGED\n\n"
                 "CAM-006/006.1: fitting an oracle 3D\n"
                 "hand by PnP cannot identify focal\n"
                 "in this regime.\n\n"
                 "But that does not mean the learned\n"
                 "model holds no camera information.\n"
                 "CAM-007 probes its outputs and\n"
                 "its latent instead.", va="top", fontsize=9)

    b = fig.add_subplot(gs[0, 1:])
    o2 = ["B0", "B1", "B2", "P5", "P6", "P7", "P10", "C1", "B3"]
    c2 = [BASE, BASE, SCENE, HAND, HAND, HAND, SCENE, ORACLE, ORACLE]
    b.bar([lbl[p] for p in o2],
          [M(p, "median_rel_focal_err_pct") for p in o2], color=c2)
    b.axhline(M("B1", "median_rel_focal_err_pct"), color="k", ls="--", lw=1.2)
    b.set_ylabel("median focal err (%)")
    b.set_title("2. unseen physical cameras: nothing beats the simple "
                "constant correction", fontsize=10)
    b.tick_params(axis="x", labelsize=7, rotation=20)
    b.grid(axis="y", alpha=0.3)

    c = fig.add_subplot(gs[1, 0])
    x = np.arange(len(ps))
    c.bar(x - 0.2, [M(p, "median_rel_focal_err_pct") for p in ps], 0.4,
          color=HAND, label="real")
    c.bar(x + 0.2, [sh[p]["median_rel_focal_err_pct"] for p in ps], 0.4,
          color=SHUF, label="shuffled")
    c.set_xticks(x)
    c.set_xticklabels([lbl.get(p, p) for p in ps], fontsize=7)
    c.set_ylabel("median focal err (%)")
    c.set_title("3. shuffling hand features\nbarely changes anything",
                fontsize=10)
    c.legend(fontsize=7)
    c.grid(axis="y", alpha=0.3)

    d = fig.add_subplot(gs[1, 1])
    d.bar(["scene", "scene+explicit", "scene+latent", "scene+all"],
          [M("B2", "median_rel_focal_err_pct"),
           M("P8", "median_rel_focal_err_pct"),
           M("P9", "median_rel_focal_err_pct"),
           M("P10", "median_rel_focal_err_pct")],
          color=[SCENE, HAND, HAND, HAND])
    d.set_ylabel("median focal err (%)")
    d.set_title("4. no incremental value\nover scene cues", fontsize=10)
    d.tick_params(axis="x", labelsize=7, rotation=15)
    d.grid(axis="y", alpha=0.3)

    e = fig.add_subplot(gs[1, 2])
    e.bar(["LOCO camera", "LOSO sequence"],
          [M("P7", "median_rel_focal_err_pct"),
           fnum(loso["P7"]["median_rel_focal_err_pct"])], color=HAND)
    e.set_ylabel("median focal err (%)")
    e.set_title("5. LOSO is no better either:\nnot even rig repetition",
                fontsize=10)
    e.grid(axis="y", alpha=0.3)

    f = fig.add_subplot(gs[2, :])
    f.axis("off")
    f.text(0, 1,
           "VERDICT  " + " | ".join(ver["decision_tags"]) + "\n\n"
           f"coverage: {cov['views_hand_feature_eligible']}/"
           f"{cov['views_total']} views hand-eligible, "
           f"{cov['views_with_latent']} with latent, latent dim "
           f"{cov['latent_dim']}\n\n"
           "No linearly decodable camera-residual signal was detected in the "
           "selected frozen representation\nunder this dataset and validation "
           "protocol. This does NOT show the network contains no camera "
           "information.\n\n"
           "The AnyCalib-focal-only control reaches "
           f"{M('C1', 'median_rel_focal_err_pct'):.2f} % and the constant "
           f"reference oracle {M('B3', 'median_rel_focal_err_pct'):.2f} % - "
           "both exploit the rig's 1.9 % focal spread,\nnot a camera cue. "
           "They are confound controls, not methods.",
           va="top", fontsize=9, family="monospace")

    fig.suptitle("CAM-EXP-007 - is there a learned hand-derived camera cue?",
                 fontsize=14)
    save(fig, "CAM_EXP_007_MAIN_EXPLANATION.png")


if __name__ == "__main__":
    main()
