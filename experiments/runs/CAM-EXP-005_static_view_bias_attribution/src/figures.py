"""Figures for CAM-EXP-005."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIG, RAW, SUM, read_csv, read_json  # noqa: E402

FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white", "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linestyle": ":"})


def f01():
    views = read_csv(RAW / "view_targets.csv.gz")
    seqs = sorted({v["sequence"] for v in views})
    cams = sorted({v["camera"] for v in views})
    M = np.full((len(seqs), len(cams)), np.nan)
    for v in views:
        M[seqs.index(v["sequence"]), cams.index(v["camera"])] = \
            (np.exp(float(v["anycalib_signed_log_bias"])) - 1) * 100
    order = np.argsort(np.nanmean(M, axis=0))
    M, cams = M[:, order], [cams[i] for i in order]
    fig, ax = plt.subplots(figsize=(15, 3.6))
    lim = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(cams)))
    ax.set_xticklabels([c.replace("brics-odroid-", "").replace("_cam", "/")
                        for c in cams], rotation=90, fontsize=6)
    ax.set_yticks(range(len(seqs)))
    ax.set_yticklabels(seqs, fontsize=8)
    ax.set_xlabel("physical camera id (sorted by mean bias)")
    fig.colorbar(im, ax=ax, label="signed focal bias (%)")
    ax.set_title("Per-view signed focal bias, AnyCalib-gen radial:2\n"
                 "columns stay the same colour down the rows: the bias travels "
                 "with the camera, not with the sequence", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "01_camera_sequence_bias_heatmap.png", dpi=200)
    plt.close(fig)


def f02():
    rep = read_csv(SUM / "camera_across_sequence_repeatability.csv")
    pairs = [r for r in read_csv(SUM / "sequence_pair_camera_correlations.csv")
             if r.get("status") == "ok"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    mean = np.array([float(r["mean_signed_log_bias"]) for r in rep])
    sd = np.array([float(r["between_sequence_sd_log"]) for r in rep])
    axes[0].errorbar(np.arange(len(rep)), mean * 100, yerr=sd * 100, fmt="o",
                     ms=4, lw=1, color="black", ecolor="#888888", capsize=2)
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_xlabel("physical camera (one point per camera)")
    axes[0].set_ylabel("signed bias, % (mean +/- sd across sequences)")
    axes[0].set_title("Same camera, different sequences:\n"
                      "the offset repeats, the scatter is smaller", fontsize=10.5)
    r = np.array([float(p["pearson_r"]) for p in pairs])
    axes[1].bar(range(len(pairs)), r, color="#555555")
    axes[1].axhline(0, color="black", lw=1)
    axes[1].axhline(float(np.median(r)), color="#d62728", ls="--", lw=1.2,
                    label=f"median r = {np.median(r):+.3f}")
    axes[1].set_xticks(range(len(pairs)))
    axes[1].set_xticklabels([f"{p['sequence_a'][:6]}\nvs\n{p['sequence_b'][:6]}"
                             for p in pairs], fontsize=6)
    axes[1].set_ylabel("Pearson r of shared-camera bias")
    axes[1].set_title("Sequence-pair agreement on which camera is biased\n"
                      "(shared cameras only)", fontsize=10.5)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "02_camera_bias_repeatability.png", dpi=200)
    plt.close(fig)


def f03():
    m = read_csv(SUM / "camera_sequence_descriptive_models.csv")
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    names = [r["model"].replace("_", " ") for r in m]
    r2 = [float(r["r2_in_sample"]) for r in m]
    adj = [float(r["adjusted_r2"]) for r in m]
    x = np.arange(len(m))
    ax.bar(x - 0.2, r2, 0.4, color="white", edgecolor="black", hatch="///",
           label="in-sample R^2")
    ax.bar(x + 0.2, adj, 0.4, color="#cccccc", edgecolor="black",
           label="adjusted R^2")
    for xi, (a, b) in enumerate(zip(r2, adj)):
        ax.text(xi - 0.2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(xi + 0.2, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=8)
    ax.set_ylabel("descriptive explanatory fit")
    ax.set_title("What describes the per-view bias: camera identity, not "
                 "sequence\n(in-sample fit, NOT a causal variance share; camera "
                 "id is DIAGNOSTIC_ONLY)", fontsize=10.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "03_camera_vs_sequence_explanatory_fit.png", dpi=200)
    plt.close(fig)


def f04():
    a = [r for r in read_csv(SUM / "feature_bias_associations.csv")
         if r["target"] == "signed_log_bias"]
    a.sort(key=lambda r: -abs(float(r["spearman_rho"])))
    top = a[:14]
    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    y = np.arange(len(top))
    rho = [float(r["spearman_rho"]) for r in top]
    lo = [float(r["ci_lo_camera_cluster"]) for r in top]
    hi = [float(r["ci_hi_camera_cluster"]) for r in top]
    col = {"F_MODEL_SELF": "#d62728", "MODEL_DISAGREEMENT": "#ff7f0e",
           "F_SCENE_GEOM": "#1f77b4", "F_IMAGE_GLOBAL": "#2ca02c",
           "F_BORDER_SCENE": "#9467bd"}
    for i, r in enumerate(top):
        ax.plot([lo[i], hi[i]], [i, i], lw=2.5, color=col.get(r["group"], "gray"))
        ax.plot(rho[i], i, "o", ms=6, color=col.get(r["group"], "gray"))
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['feature']}  [{r['group'].replace('F_','')}]"
                        for r in top], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman rho vs signed log bias (95 % physical-camera "
                  "cluster bootstrap)")
    ax.set_title("Top-ranked among the 42 PRE-REGISTERED features\n"
                 "the strongest are the model's own outputs, not the scene",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "04_scene_feature_correlations.png", dpi=200)
    plt.close(fig)


def f05():
    o = [r for r in read_csv(SUM / "oracle_camera_property_associations.csv")
         if r["target"] == "signed_log_bias"]
    o.sort(key=lambda r: -abs(float(r["spearman_rho"])))
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for i, r in enumerate(o):
        ax.plot([float(r["ci_lo_camera_cluster"]), float(r["ci_hi_camera_cluster"])],
                [i, i], lw=2.5, color="#777777")
        ax.plot(float(r["spearman_rho"]), i, "s", ms=6, color="black")
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(range(len(o)))
    ax.set_yticklabels([r["camera_property"] for r in o], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman rho vs signed log bias (camera-cluster CI)")
    ax.set_title("ORACLE DIAGNOSTIC ONLY - provided camera properties\n"
                 "no property's interval excludes zero", fontsize=11,
                 color="#b30000")
    fig.tight_layout()
    fig.savefig(FIG / "05_oracle_camera_property_diagnostics.png", dpi=200)
    plt.close(fig)


def _probe_fig(csv_name, out_name, title, note):
    rows = read_csv(SUM / csv_name)
    order = ["P0_baseline", "P1", "P2", "P3", "P5", "P6", "P4", "P7",
             "D_CAMERA_ID", "D_SEQUENCE_ID"]
    rows = [r for n in order for r in rows if r["feature_set"] == n]
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    mae = [float(r["mae_log_bias"]) * 100 for r in rows]
    cols = ["#999999" if r["feature_set"].startswith("D_")
            else "#d62728" if r["feature_set"] in ("P4", "P7")
            else "#1f77b4" for r in rows]
    x = np.arange(len(rows))
    ax.bar(x, mae, 0.62, color=cols, edgecolor="black")
    base = next(float(r["mae_log_bias"]) * 100 for r in rows
                if r["feature_set"] == "P0_baseline")
    ax.axhline(base, color="black", ls="--", lw=1.2,
               label=f"P0 baseline = {base:.2f}")
    for xi, v in zip(x, mae):
        ax.text(xi, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([r["feature_set"] for r in rows], fontsize=8, rotation=20)
    ax.set_ylabel("MAE of predicted bias (% equivalent)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    fig.text(0.5, -0.03, note, ha="center", fontsize=8.4)
    fig.tight_layout()
    fig.savefig(FIG / out_name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def f06():
    _probe_fig("probe_summary_loso.csv", "06_probe_loso.png",
               "LINEAR PROBE DIAGNOSTIC - leave one SEQUENCE out\n"
               "a content-shift test; the same physical camera is usually in "
               "training too",
               "red = groups containing the model's own predicted focal; grey = "
               "diagnostic-only identity models, not deployable")


def f07():
    _probe_fig("probe_summary_leave_camera_out.csv",
               "07_probe_leave_camera_out.png",
               "LINEAR PROBE DIAGNOSTIC - leave one PHYSICAL CAMERA out\n"
               "the new-camera test, and the primary decision criterion",
               "scene groups (P1, P2, P3, P6) do not beat the P0 baseline on an "
               "unseen camera")


def f08():
    rows = read_csv(SUM / "probe_correction_summary.csv")
    for proto, tag in (("B_LOCO", "08_probe_corrected_focal_error.png"),):
        rs = [r for r in rows if r["protocol"] == proto]
        order = ["ORIGINAL_uncorrected", "P0_baseline", "P1", "P6", "P4", "P7",
                 "D_CAMERA_ID", "CONSTANT_RIG_FOCAL_ORACLE"]
        rs = [r for n in order for r in rs if r["feature_set"] == n]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        x = np.arange(len(rs))
        med = [float(r["median_rel_focal_err_pct"]) for r in rs]
        lo = [med[i] - float(r["median_ci_lo_camera_cluster"]) for i, r in enumerate(rs)]
        hi = [float(r["median_ci_hi_camera_cluster"]) - med[i] for i, r in enumerate(rs)]
        cols = ["#333333" if r["feature_set"] == "ORIGINAL_uncorrected"
                else "#b30000" if "ORACLE" in r["feature_set"]
                or r["feature_set"].startswith("D_")
                else "#d62728" if r["feature_set"] in ("P4", "P7")
                else "#1f77b4" for r in rs]
        axes[0].bar(x, med, 0.6, yerr=[lo, hi], color=cols, edgecolor="black",
                    capsize=3)
        axes[0].axhline(5, color="black", ls=":", lw=1)
        for xi, v in zip(x, med):
            axes[0].text(xi, v + 0.15, f"{v:.2f}", ha="center", fontsize=8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([r["feature_set"].replace("_", "\n") for r in rs],
                                fontsize=7)
        axes[0].set_ylabel("median relative focal error (%)")
        axes[0].set_title("LINEAR PROBE DIAGNOSTIC correction, leave-one-camera-out",
                          fontsize=10.5)
        w5 = [float(r["within_5_pct"]) for r in rs]
        axes[1].bar(x, w5, 0.6, color=cols, edgecolor="black")
        for xi, v in zip(x, w5):
            axes[1].text(xi, v + 1, f"{v:.0f}", ha="center", fontsize=8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([r["feature_set"].replace("_", "\n") for r in rs],
                                fontsize=7)
        axes[1].set_ylabel("views within +/-5 % (%)")
        axes[1].set_title("within +/-5 %", fontsize=10.5)
        fig.suptitle("NOT a calibration method. Dark red = oracle / identity "
                     "references.", fontsize=10, y=1.02)
        fig.text(0.5, -0.05,
                 "The decisive control is CONSTANT_RIG_FOCAL_ORACLE: ignore the "
                 "image entirely and output the training folds' median reference "
                 "focal. It beats every probe,\nbecause this rig's reference "
                 "focal varies by only ~1.9 % (CV). No probe here has "
                 "demonstrated a calibration signal.", ha="center", fontsize=8.6)
        fig.tight_layout()
        fig.savefig(FIG / tag, dpi=200, bbox_inches="tight")
        plt.close(fig)


def main_figure():
    views = read_csv(RAW / "view_targets.csv.gz")
    models = read_csv(SUM / "camera_sequence_descriptive_models.csv")
    loco = read_csv(SUM / "probe_summary_leave_camera_out.csv")
    corr = [r for r in read_csv(SUM / "probe_correction_summary.csv")
            if r["protocol"] == "B_LOCO"]
    assoc = [r for r in read_csv(SUM / "feature_bias_associations.csv")
             if r["target"] == "signed_log_bias"]

    fig = plt.figure(figsize=(17, 10.4))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    seqs = sorted({v["sequence"] for v in views})
    cams = sorted({v["camera"] for v in views})
    M = np.full((len(seqs), len(cams)), np.nan)
    for v in views:
        M[seqs.index(v["sequence"]), cams.index(v["camera"])] = \
            (np.exp(float(v["anycalib_signed_log_bias"])) - 1) * 100
    order = np.argsort(np.nanmean(M, axis=0))
    lim = np.nanmax(np.abs(M))
    im = ax.imshow(M[:, order], cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_yticks(range(len(seqs)))
    ax.set_yticklabels([s[:12] for s in seqs], fontsize=7)
    ax.set_xticks([])
    ax.set_xlabel("40 physical cameras (sorted)")
    fig.colorbar(im, ax=ax, label="signed bias (%)")
    ax.set_title("1. The bias travels with the CAMERA\nnot with the sequence",
                 fontsize=10.5)

    ax = fig.add_subplot(gs[0, 1])
    names = [r["model"].split("_")[0] for r in models]
    adj = [float(r["adjusted_r2"]) for r in models]
    ax.bar(range(len(models)), adj, 0.6, color=["#bbbbbb", "#2ca02c", "#1f77b4",
                                                "#7f2f8f"], edgecolor="black")
    for i, v in enumerate(adj):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(["intercept", "sequence", "camera", "both"], fontsize=8)
    ax.set_ylabel("adjusted R^2 (in-sample, descriptive)")
    ax.set_title("2. Camera identity describes it;\nsequence identity does not",
                 fontsize=10.5)

    ax = fig.add_subplot(gs[0, 2])
    top = sorted(assoc, key=lambda r: -abs(float(r["spearman_rho"])))[:8]
    col = {"F_MODEL_SELF": "#d62728", "MODEL_DISAGREEMENT": "#ff7f0e",
           "F_SCENE_GEOM": "#1f77b4", "F_IMAGE_GLOBAL": "#2ca02c",
           "F_BORDER_SCENE": "#9467bd"}
    for i, r in enumerate(top):
        ax.plot([float(r["ci_lo_camera_cluster"]), float(r["ci_hi_camera_cluster"])],
                [i, i], lw=2.5, color=col.get(r["group"], "gray"))
        ax.plot(float(r["spearman_rho"]), i, "o", ms=5,
                color=col.get(r["group"], "gray"))
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([r["feature"][:30] for r in top], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman rho (camera-cluster CI)")
    ax.set_title("3. Associations: the model's own outputs,\nnot the scene "
                 "(blue = scene geometry)", fontsize=10.5)

    ax = fig.add_subplot(gs[1, 0])
    sets = ["P0_baseline", "P1", "P6", "P4", "P7"]
    mae = [float(next(r for r in loco if r["feature_set"] == s)["mae_log_bias"]) * 100
           for s in sets]
    ax.bar(range(len(sets)), mae, 0.6,
           color=["#bbbbbb", "#1f77b4", "#1f77b4", "#d62728", "#d62728"],
           edgecolor="black")
    ax.axhline(mae[0], color="black", ls="--", lw=1.2)
    for i, v in enumerate(mae):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels(["baseline", "scene\ngeom", "all\nscene", "model\nself",
                        "all"], fontsize=8)
    ax.set_ylabel("MAE of predicted bias (% equiv.)")
    ax.set_title("4. NEW-CAMERA test (leave one camera out)\nscene features do "
                 "not beat the baseline", fontsize=10.5)

    ax = fig.add_subplot(gs[1, 1])
    sets2 = ["ORIGINAL_uncorrected", "P0_baseline", "P1", "P4",
             "CONSTANT_RIG_FOCAL_ORACLE"]
    med = [float(next(r for r in corr if r["feature_set"] == s)
                 ["median_rel_focal_err_pct"]) for s in sets2]
    ax.bar(range(len(sets2)), med, 0.6,
           color=["#333333", "#bbbbbb", "#1f77b4", "#d62728", "#b30000"],
           edgecolor="black")
    for i, v in enumerate(med):
        ax.text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=8.5)
    ax.axhline(5, color="black", ls=":", lw=1)
    ax.set_xticks(range(len(sets2)))
    ax.set_xticklabels(["original", "mean-bias\nbaseline", "scene\ngeom",
                        "model\nself", "CONSTANT\nfocal oracle"], fontsize=7.5)
    ax.set_ylabel("median relative focal error (%)")
    ax.set_title("5. Probe correction diagnostic\nthe image-free constant beats "
                 "everything", fontsize=10.5)

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    txt = (
        "6.  WHAT THIS EXPERIMENT FOUND\n"
        "\n"
        "CONFIRMED - the bias is tied to the camera.\n"
        "  Camera identity: adjusted R2 0.38 (in-sample).\n"
        "  Sequence identity: adjusted R2 -0.01.\n"
        "  Sequence-pair agreement on which camera is\n"
        "  biased: median Pearson r = +0.42.\n"
        "  So it is a per-VIEW / per-camera property,\n"
        "  not an activity or content effect.\n"
        "\n"
        "NOT FOUND - a deployable RGB cue.\n"
        "  On an unseen camera, no scene-geometry,\n"
        "  image-global or border feature group beat the\n"
        "  simple mean-bias baseline.\n"
        "\n"
        "NOT FOUND - an oracle camera property either.\n"
        "  No provided property (k1, k2, principal point,\n"
        "  fx/fy, camera pose) had a camera-cluster CI\n"
        "  excluding zero.\n"
        "\n"
        "THE CONFOUND THAT DOMINATES THIS RIG.\n"
        "  The reference focal varies by only ~1.9 % (CV)\n"
        "  across all 175 views. An estimator that ignores\n"
        "  the image and always outputs the rig's median\n"
        "  focal reaches 0.88 % median error and 97.7 %\n"
        "  within 5 % - better than every probe. Any group\n"
        "  containing the predicted focal 'wins' by\n"
        "  memorising that constant, not by calibrating.\n"
        "\n"
        "NEXT: CAM-EXP-006 (hand cues), and a rig with\n"
        "  genuinely different focal lengths."
    )
    ax.text(0, 1, txt, va="top", ha="left", fontsize=8.9, family="monospace")

    fig.suptitle("CAM-EXP-005 - Where the static-view focal bias comes from: "
                 "the camera, but no deployable RGB cue recovers it",
                 fontsize=14, y=0.985)
    fig.savefig(FIG / "CAM_EXP_005_MAIN_EXPLANATION.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    f01(); f02(); f03(); f04(); f05(); f06(); f07(); f08()
    main_figure()
    for p in sorted(FIG.glob("*.png")):
        print(f"  {p.name}  {p.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
