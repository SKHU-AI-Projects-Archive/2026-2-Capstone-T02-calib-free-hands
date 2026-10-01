"""Human-readable figures for CAM-EXP-004."""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODELS, N_VALUES, RUN_DIR, load_views, read_csv  # noqa: E402

FIG = RUN_DIR / "figures"
SUM = RUN_DIR / "results" / "summary"
RAW = RUN_DIR / "results" / "raw"
FIG.mkdir(parents=True, exist_ok=True)

COL = {"anycalib_gen": "#1f77b4", "geocalib_distorted": "#d62728",
       "pf_uncentered": "#2ca02c"}
ECOL = {"E1_perframe_mean": "#9467bd", "E2_perframe_geomean": "#8c564b",
        "E3_perview_mean": "#e377c2", "E4_perview_geomean": "#7f2f8f",
        "E5_three_model_median": "#ff7f0e"}


def load(name):
    return list(csv.DictReader(open(SUM / name, encoding="utf-8")))


def curve(rows, key, val, col, N=N_VALUES):
    d = {int(r["N"]): float(r[col]) for r in rows if r[key] == val}
    return [d.get(n, np.nan) for n in N]


# ------------------------------------------------------------------ 1 & 2
def fig_vs_frame_count():
    evc = load("error_vs_frame_count.csv")
    med = [r for r in evc if r["method"] == "median"]
    ens = load("ensemble_summary.csv")

    for col, ylab, fname, title in (
        ("median_rel_err_pct", "median relative focal error (%)",
         "focal_error_vs_frame_count.png",
         "More frames barely help a biased model"),
        ("within_5_pct", "views within 5 % of GT focal (%)",
         "within5_rate_vs_frame_count.png",
         "Fraction of static cameras calibrated to within 5 %"),
    ):
        fig, ax = plt.subplots(figsize=(8.2, 5.4))
        for mk in MODELS:
            ax.plot(N_VALUES, curve(med, "model", mk, col), "o-", color=COL[mk],
                    lw=2, label=f"{MODELS[mk]['label']} (median agg)")
        for ek in ("E2_perframe_geomean", "E4_perview_geomean",
                   "E5_three_model_median"):
            ax.plot(N_VALUES, curve(ens, "ensemble", ek, col), "s--",
                    color=ECOL[ek], lw=1.8, label=f"ensemble {ek}")
        if col == "median_rel_err_pct":
            ax.axhline(5, color="k", ls=":", lw=1)
            ax.text(8.1, 5.2, "5 % target", fontsize=8, ha="right")
        ax.set_xscale("log", base=2)
        ax.set_xticks(N_VALUES)
        ax.set_xticklabels([str(n) for n in N_VALUES])
        ax.set_xlabel("frames aggregated per static camera (N)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / fname, dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------- 3
def fig_bias_noise():
    bn = load("bias_noise_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    names = [r["model"] for r in bn]
    bias = [float(r["bias_fraction_pct"]) for r in bn]
    within = [float(r["within_fraction_pct"]) for r in bn]
    y = np.arange(len(names))
    axes[0].barh(y, bias, color="#c0392b", label="between-view bias")
    axes[0].barh(y, within, left=bias, color="#2980b9", label="within-view noise")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([MODELS[n]["label"] for n in names], fontsize=8)
    axes[0].set_xlabel("share of total squared log focal error (%)")
    axes[0].set_title("Where the error lives")
    axes[0].legend(fontsize=8)
    for i, (b, w) in enumerate(zip(bias, within)):
        axes[0].text(b / 2, i, f"{b:.0f}%", ha="center", va="center",
                     color="w", fontsize=9)
        axes[0].text(b + w / 2, i, f"{w:.0f}%", ha="center", va="center",
                     color="w", fontsize=9)

    w = 0.35
    x = np.arange(len(names))
    axes[1].bar(x - w / 2, [float(r["median_abs_view_bias_pct"]) for r in bn],
                w, color="#c0392b", label="median |per-view bias| (%)")
    axes[1].bar(x + w / 2, [float(r["median_within_view_std_pct"]) for r in bn],
                w, color="#2980b9", label="median within-view sd (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, fontsize=8, rotation=10)
    axes[1].set_ylabel("% of focal")
    axes[1].set_title("Magnitude: bias dwarfs frame-to-frame scatter")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle("Averaging removes noise, not bias", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "bias_vs_noise_decomposition.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------- 4
def fig_oracle():
    orc = load("oracle_summary.csv")
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for mk in MODELS:
        rows = [r for r in orc if r["model"] == mk]
        ax.plot(N_VALUES, curve(rows, "model", mk, "median_rel_err_pct"), "^--",
                color=COL[mk], lw=2,
                label=f"{mk}: ORACLE best-of-N (uses GT)")
        ax.plot(N_VALUES,
                curve(rows, "model", mk, "deployable_median_for_reference_pct"),
                "o-", color=COL[mk], lw=2, alpha=0.55,
                label=f"{mk}: deployable median agg")
    ens = load("ensemble_summary.csv")
    ax.plot(N_VALUES, curve(ens, "ensemble", "E2_perframe_geomean",
                            "median_rel_err_pct"), "s-", color="#8c564b", lw=2.4,
            label="DEPLOYABLE 2-model ensemble (E2)")
    ax.axhline(5, color="k", ls=":", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(N_VALUES)
    ax.set_xticklabels([str(n) for n in N_VALUES])
    ax.set_xlabel("frames per static camera (N)")
    ax.set_ylabel("median relative focal error (%)")
    ax.set_title("Oracle frame selection vs deployable aggregation\n"
                 "(dashed = cheating with GT; the ensemble beats the primary "
                 "model's oracle)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "oracle_best_vs_aggregation.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------- 5
def fig_ensemble():
    ens = [r for r in load("ensemble_summary.csv") if r["N"] == "8"]
    ens.sort(key=lambda r: float(r["median_rel_err_pct"]))
    names = [r["ensemble"] for r in ens]
    med = [float(r["median_rel_err_pct"]) for r in ens]
    w5 = [float(r["within_5_pct"]) for r in ens]
    sgn = [float(r["signed_median_err_pct"]) for r in ens]
    cols = ["#7f7f7f" if n.startswith("SINGLE") else "#7f2f8f" for n in names]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0))
    y = np.arange(len(names))
    for ax, vals, lab in ((axes[0], med, "median relative focal error (%)"),
                          (axes[1], w5, "views within 5 % (%)")):
        ax.barh(y, vals, color=cols)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(lab)
        ax.grid(alpha=0.3, axis="x")
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.1f}", va="center", fontsize=8)
    axes[0].axvline(5, color="k", ls=":", lw=1)
    axes[2].barh(y, sgn, color=cols)
    axes[2].axvline(0, color="k", lw=1)
    axes[2].set_yticks(y)
    axes[2].set_yticklabels([])
    axes[2].invert_yaxis()
    axes[2].set_xlabel("signed median error (%)")
    axes[2].set_title("opposite biases cancel")
    axes[2].grid(alpha=0.3, axis="x")
    fig.suptitle("Parameter-free cross-model ensembles vs single models (N = 8)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "cross_model_ensemble.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------- 6
def fig_agreement():
    ag = read_csv(SUM / "model_agreement_vs_error.csv")
    corr = load("agreement_correlation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    x = np.array([float(r["disagreement_2model"]) for r in ag]) * 100
    y = np.array([float(r["ensemble_e5_rel_err_pct"]) for r in ag])
    axes[0].scatter(x, y, s=16, alpha=0.6, color="#2f5d8f")
    axes[0].set_xlabel("AnyCalib vs GeoCalib disagreement (%, GT-blind)")
    axes[0].set_ylabel("ensemble E5 relative focal error (%)")
    rho = next(float(r["spearman_rho"]) for r in corr
               if r["agreement_metric"] == "disagreement_2model"
               and r["target"] == "ensemble_e5_rel_err_pct")
    axes[0].set_title(f"Spearman rho = {rho:+.3f}")
    axes[0].grid(alpha=0.3)

    labels, vals = [], []
    for r in corr:
        labels.append(f"{r['agreement_metric'].replace('disagreement_', '')} "
                      f"-> {r['target'].replace('_rel_err_pct', '')}")
        vals.append([float(r[f"bin{b}_median_err_pct"]) for b in range(4)])
    xb = np.arange(4)
    for lab, v in zip(labels, vals):
        axes[1].plot(xb, v, "o-", lw=1.6, label=lab)
    axes[1].set_xticks(xb)
    axes[1].set_xticklabels(["Q1\n(most agree)", "Q2", "Q3", "Q4\n(least agree)"],
                            fontsize=8)
    axes[1].set_ylabel("median relative focal error (%)")
    axes[1].set_title("Error by quartile of GT-blind disagreement")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.suptitle("Does model disagreement predict error? Weakly, and "
                 "inconsistently.", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "model_agreement_vs_error.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------- 7
def fig_trajectories():
    bn = read_csv(RAW / "bias_noise_decomposition.csv.gz")
    prim = [r for r in bn if r["model"] == "anycalib_gen"]
    for r in prim:
        r["_b"] = abs(float(r["view_bias_pct"]))
        r["_n"] = float(r["within_view_std_pct"])
    good = min(prim, key=lambda r: r["_b"] + r["_n"])
    biased = max((r for r in prim if r["_n"] < 3), key=lambda r: r["_b"])
    noisy = max(prim, key=lambda r: r["_n"])
    picks = [("well calibrated\n(low bias, low noise)", good),
             ("stable but biased\n(more frames cannot help)", biased),
             ("noisy view\n(averaging does help here)", noisy)]

    views = {mk: load_views(mk) for mk in MODELS}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharex=True)
    for ax, (title, r) in zip(axes, picks):
        key = (r["sequence"], r["camera"])
        gt = views["anycalib_gen"][key]["gt_fx"]
        for mk in MODELS:
            v = views[mk][key]
            ax.plot(v["frame"], v["fx"], "o-", color=COL[mk], ms=4, lw=1.4,
                    label=MODELS[mk]["label"] if ax is axes[0] else None)
        ax.axhline(gt, color="k", lw=2, ls="--",
                   label="GT focal (constant)" if ax is axes[0] else None)
        ax.set_title(f"{title}\n{key[0]} / {key[1]}", fontsize=8.5)
        ax.set_xlabel("frame id")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("predicted fx (px)")
    axes[0].legend(fontsize=7.5)
    fig.suptitle("Per-view prediction trajectories: the camera never changes, "
                 "so every wobble and every offset is model error", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "per_view_trajectories.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------- MAIN
def fig_main():
    evc = [r for r in load("error_vs_frame_count.csv") if r["method"] == "median"]
    ens = load("ensemble_summary.csv")
    orc = load("oracle_summary.csv")
    bn = load("bias_noise_summary.csv")

    fig = plt.figure(figsize=(16.5, 10.2))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28)

    # (1) the premise
    ax = fig.add_subplot(gs[0, 0])
    v = load_views("anycalib_gen")
    key = sorted(v)[0]
    d = v[key]
    ax.plot(d["frame"], d["fx"], "o-", color=COL["anycalib_gen"], label="predicted fx")
    ax.axhline(d["gt_fx"], color="k", ls="--", lw=2, label="true fx (constant)")
    ax.fill_between(d["frame"], d["gt_fx"] * 0.95, d["gt_fx"] * 1.05,
                    color="k", alpha=0.10, label="+/-5 % target")
    ax.set_title("1. The camera is static, so the true focal\n"
                 "is one number for the whole video", fontsize=10)
    ax.set_xlabel("frame")
    ax.set_ylabel("fx (px)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    # (2) error vs N
    ax = fig.add_subplot(gs[0, 1])
    for mk in MODELS:
        ax.plot(N_VALUES, curve(evc, "model", mk, "median_rel_err_pct"), "o-",
                color=COL[mk], lw=2, label=mk)
    ax.plot(N_VALUES, curve(ens, "ensemble", "E2_perframe_geomean",
                            "median_rel_err_pct"), "s-", color="#8c564b", lw=2.4,
            label="2-model ensemble")
    ax.axhline(5, color="k", ls=":", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(N_VALUES)
    ax.set_xticklabels([str(n) for n in N_VALUES])
    ax.set_title("2-3. More frames: almost nothing for the\nprimary model, "
                 "real gains only for the noisy one", fontsize=10)
    ax.set_xlabel("frames aggregated (N)")
    ax.set_ylabel("median focal error (%)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    # (3) bias vs noise
    ax = fig.add_subplot(gs[0, 2])
    names = [r["model"] for r in bn]
    bias = [float(r["bias_fraction_pct"]) for r in bn]
    within = [float(r["within_fraction_pct"]) for r in bn]
    y = np.arange(len(names))
    ax.barh(y, bias, color="#c0392b", label="stable per-view BIAS")
    ax.barh(y, within, left=bias, color="#2980b9", label="frame NOISE")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("share of squared log error (%)")
    ax.set_title("4. Why: the error is bias, and averaging\n"
                 "cannot remove a bias", fontsize=10)
    ax.legend(fontsize=7.5)

    # (4) ensemble
    ax = fig.add_subplot(gs[1, 0])
    rows = [r for r in ens if r["N"] == "8"]
    rows.sort(key=lambda r: float(r["median_rel_err_pct"]))
    nm = [r["ensemble"].replace("_", " ") for r in rows]
    md = [float(r["median_rel_err_pct"]) for r in rows]
    cols = ["#7f7f7f" if r["ensemble"].startswith("SINGLE") else "#7f2f8f"
            for r in rows]
    yy = np.arange(len(nm))
    ax.barh(yy, md, color=cols)
    ax.set_yticks(yy)
    ax.set_yticklabels(nm, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(5, color="k", ls=":", lw=1)
    ax.set_xlabel("median focal error (%)")
    ax.set_title("5. Combining two models beats either one:\n"
                 "their biases point in opposite directions", fontsize=10)
    ax.grid(alpha=0.3, axis="x")

    # (5) oracle
    ax = fig.add_subplot(gs[1, 1])
    for mk in MODELS:
        rws = [r for r in orc if r["model"] == mk]
        ax.plot(N_VALUES, curve(rws, "model", mk, "median_rel_err_pct"), "^--",
                color=COL[mk], lw=1.8, label=f"{mk} ORACLE best-of-N")
        ax.plot(N_VALUES,
                curve(rws, "model", mk, "deployable_median_for_reference_pct"),
                "o-", color=COL[mk], alpha=0.5, lw=1.8, label=f"{mk} deployable")
    ax.plot(N_VALUES, curve(ens, "ensemble", "E2_perframe_geomean",
                            "median_rel_err_pct"), "s-", color="#8c564b", lw=2.4,
            label="2-model ensemble (deployable)")
    ax.axhline(5, color="k", ls=":", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(N_VALUES)
    ax.set_xticklabels([str(n) for n in N_VALUES])
    ax.set_xlabel("frames (N)")
    ax.set_ylabel("median focal error (%)")
    ax.set_title("6. Even cheating by picking the single best\n"
                 "frame with GT does not reach 5 %", fontsize=10)
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3)

    # (6) conclusion
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    e2 = next(r for r in ens if r["ensemble"] == "E2_perframe_geomean"
              and r["N"] == "8")
    a8 = next(r for r in evc if r["model"] == "anycalib_gen" and r["N"] == "8")
    a1 = next(r for r in evc if r["model"] == "anycalib_gen" and r["N"] == "1")
    txt = (
        "7.  WHAT MORE FRAMES DO AND DO NOT FIX\n"
        "\n"
        "FIXED - frame-to-frame scatter.\n"
        f"  GeoCalib {float(next(r for r in evc if r['model']=='geocalib_distorted' and r['N']=='1')['median_rel_err_pct']):.1f} %"
        f" -> {float(next(r for r in evc if r['model']=='geocalib_distorted' and r['N']=='8')['median_rel_err_pct']):.1f} % "
        "(it is the noisy one).\n"
        "\n"
        "NOT FIXED - the per-view offset.\n"
        f"  AnyCalib-gen {float(a1['median_rel_err_pct']):.2f} % -> "
        f"{float(a8['median_rel_err_pct']):.2f} % over 8 frames.\n"
        "  98 % of its error is a fixed offset per camera.\n"
        "  Intuitively: if a model says 1000 px every time and\n"
        "  the truth is 920 px, averaging 100 shots still says\n"
        "  1000 px. That is bias, not noise.\n"
        "\n"
        "WHAT DOES HELP - a second model.\n"
        f"  2-model ensemble: {float(e2['median_rel_err_pct']):.2f} % median, "
        f"{float(e2['within_5_pct']):.0f} % of cameras within 5 %\n"
        f"  (single best model: {float(a8['median_rel_err_pct']):.2f} %, "
        f"{float(a8['within_5_pct']):.0f} %).\n"
        "  One over-predicts, the other under-predicts.\n"
        "\n"
        "STILL SHORT OF THE TARGET.\n"
        "  The +/-5 % focal goal is not reached by using more\n"
        "  frames alone."
    )
    ax.text(0, 1, txt, va="top", ha="left", fontsize=9.2, family="monospace")

    fig.suptitle("CAM-EXP-004 - Static-camera multi-frame aggregation: "
                 "more frames remove noise, not bias", fontsize=14, y=0.985)
    fig.savefig(FIG / "CAM_EXP_004_MAIN_EXPLANATION.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig_vs_frame_count()
    fig_bias_noise()
    fig_oracle()
    fig_ensemble()
    fig_agreement()
    fig_trajectories()
    fig_main()
    for p in sorted(FIG.glob("*.png")):
        print(f"  {p.name}  {p.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
