"""Figures for CAM-EXP-003.

ORACLE_GT appears only as a reference line, never as a ranked estimator.
Models that could not be run produce no figure at all rather than an empty one.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import RUN_DIR, read_csv, rel

log = logging.getLogger("cam-exp-003")
FIG = RUN_DIR / "figures"
THRESHOLDS = (5.0, 10.0, 20.0)
REFERENCE_ONLY = {"ORACLE_GT"}
# CAM-EXP-002 measured this on the same data; quoted as a target, never as a
# depth error produced by CAM-EXP-003 itself.
CAM002_MM_AT_5PCT = 32.8

COLORS = ["#2563eb", "#16a34a", "#f59e0b", "#7c3aed", "#dc2626", "#0891b2"]


def num(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def load():
    rows = read_csv(RUN_DIR / "results" / "raw" / "all_predictions.csv.gz")
    summ = read_csv(RUN_DIR / "results" / "summary" / "model_summary.csv")
    return rows, summ


def errors_by_model(rows, exclude_reference=True) -> dict:
    g = defaultdict(list)
    for r in rows:
        if r["success"] != "1":
            continue
        if exclude_reference and r["model"] in REFERENCE_ONLY:
            continue
        v = num(r["relative_focal_error_pct"])
        if v is not None:
            g[r["model"]].append(v)
    return {k: np.asarray(v) for k, v in sorted(g.items(),
                                                key=lambda kv: np.median(kv[1]))}


def short(name: str) -> str:
    return (name.replace("PerspectiveFields", "PF").replace("AnyCalib[anycalib_pinhole/pinhole]",
            "AnyCalib").replace("GeoCalib[pinhole]", "GeoCalib")
            .replace("DEMO_FIXED_5000", "DEMO f=5000"))


def fig_boxplot(g):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    names = list(g)
    data = [g[k] for k in names]
    bp = ax.boxplot(data, tick_labels=[short(n) for n in names], showfliers=False,
                    patch_artist=True, widths=0.55)
    for patch, c in zip(bp["boxes"], COLORS * 3):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    for t, ls in zip(THRESHOLDS, ("-", "--", ":")):
        ax.axhline(t, color="#64748b", ls=ls, lw=1, label=f"{t:.0f} %")
    ax.set_yscale("log")
    ax.set_ylabel("relative focal error [%] (log)", fontsize=9)
    ax.set_title("Single-frame focal error by model (box = IQR, whiskers 1.5 IQR)",
                 fontsize=11)
    ax.legend(fontsize=8, title="targets", title_fontsize=8)
    plt.setp(ax.get_xticklabels(), fontsize=8, rotation=12)
    fig.tight_layout()
    fig.savefig(FIG / "focal_error_boxplot.png", dpi=130)
    plt.close(fig)
    print("  wrote", rel(FIG / "focal_error_boxplot.png"))


def fig_cdf(g):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for (name, v), c in zip(g.items(), COLORS * 3):
        x = np.sort(v)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, lw=1.9, color=c, label=f"{short(name)} (median {np.median(v):.1f} %)")
    for t, ls in zip(THRESHOLDS, ("-", "--", ":")):
        ax.axvline(t, color="#64748b", ls=ls, lw=1.2)
        ax.text(t, 0.02, f" {t:.0f}%", fontsize=8, color="#334155")
    ax.set_xscale("log")
    ax.set_xlim(0.3, 1000)
    ax.set_ylim(0, 1)
    ax.set_xlabel("relative focal error [%] (log)", fontsize=9)
    ax.set_ylabel("fraction of frames with error <= x", fontsize=9)
    ax.set_title("Cumulative distribution of single-frame focal error", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "focal_error_cdf.png", dpi=130)
    plt.close(fig)
    print("  wrote", rel(FIG / "focal_error_cdf.png"))


def fig_within(summ):
    rows = [s for s in summ if s["is_reference_only"] != "1"
            and s["focal_error_pct_median"] != ""]
    rows.sort(key=lambda s: -float(s["within_5pct_rate"]))
    names = [short(s["model"]) for s in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for i, (t, c) in enumerate(zip(THRESHOLDS, ("#16a34a", "#f59e0b", "#94a3b8"))):
        vals = [float(s[f"within_{int(t)}pct_rate"]) for s in rows]
        b = ax.bar(x + (i - 1) * 0.27, vals, 0.26, color=c, label=f"within {t:.0f} %")
        for bb, v in zip(b, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v, f"{v:.0%}", ha="center",
                    va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8, rotation=12)
    ax.set_ylabel("fraction of successful frames", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_title("Success rate within focal-error thresholds\n"
                 "(success-only; coverage-aware values in model_summary.csv)",
                 fontsize=10.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "within_threshold_rates.png", dpi=130)
    plt.close(fig)
    print("  wrote", rel(FIG / "within_threshold_rates.png"))


def fig_scatter(rows):
    g = defaultdict(list)
    for r in rows:
        if r["success"] != "1" or r["model"] in REFERENCE_ONLY:
            continue
        gt, pr = num(r["gt_fx"]), num(r["pred_fx"])
        if gt and pr:
            g[r["model"]].append((gt, pr))
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    lo, hi = 800, 1200
    for (name, pts), c in zip(sorted(g.items()), COLORS * 3):
        a = np.asarray(pts)
        ax.scatter(a[:, 0], a[:, 1], s=5, alpha=0.32, color=c, label=short(name))
        lo = min(lo, a[:, 0].min()); hi = max(hi, a[:, 1].max())
    ax.plot([700, 1200], [700, 1200], "k--", lw=1.2, label="perfect (y = x)")
    ax.set_xlim(880, 1010)
    ax.set_yscale("log")
    ax.set_xlabel("GT focal fx [px]", fontsize=9)
    ax.set_ylabel("predicted focal [px] (log)", fontsize=9)
    ax.set_title("Predicted vs GT focal\n"
                 "GT spans a narrow range, so a useful model must sit on the dashed line",
                 fontsize=10)
    ax.legend(fontsize=8, markerscale=2)
    fig.tight_layout()
    fig.savefig(FIG / "predicted_vs_gt_focal.png", dpi=130)
    plt.close(fig)
    print("  wrote", rel(FIG / "predicted_vs_gt_focal.png"))


def fig_stability():
    st = read_csv(RUN_DIR / "results" / "summary" / "single_frame_stability.csv")
    g = defaultdict(list)
    for r in st:
        if r["model"] in REFERENCE_ONLY:
            continue
        v = num(r["range_pct_of_mean"])
        if v is not None:
            g[r["model"]].append(v)
    if not g:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    names = sorted(g, key=lambda k: np.median(g[k]))
    bp = axes[0].boxplot([g[k] for k in names],
                         tick_labels=[short(n) for n in names],
                         showfliers=False, patch_artist=True, widths=0.55)
    for patch, c in zip(bp["boxes"], COLORS * 3):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    axes[0].set_ylabel("focal spread over 8 frames [% of mean]", fontsize=9)
    axes[0].set_title("Same static camera, 8 different frames:\n"
                      "how much does the prediction move?", fontsize=10)
    plt.setp(axes[0].get_xticklabels(), fontsize=8, rotation=12)

    # per-view trace for the best model
    best = names[0]
    sub = [r for r in st if r["model"] == best][:40]
    for i, r in enumerate(sub):
        mean = num(r["pred_focal_mean_px"])
        lo, hi = num(r["pred_focal_min_px"]), num(r["pred_focal_max_px"])
        if None in (mean, lo, hi):
            continue
        axes[1].plot([i, i], [lo, hi], color="#94a3b8", lw=1)
        axes[1].plot(i, mean, "o", color="#2563eb", ms=3)
    axes[1].set_xlabel("static camera view (first 40)", fontsize=9)
    axes[1].set_ylabel("predicted focal [px]", fontsize=9)
    axes[1].set_title(f"{short(best)}: per-view min/mean/max across 8 frames", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "single_frame_stability.png", dpi=130)
    plt.close(fig)
    print("  wrote", rel(FIG / "single_frame_stability.png"))


def fig_camera_heatmap(rows):
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["success"] != "1" or r["model"] in REFERENCE_ONLY:
            continue
        v = num(r["relative_focal_error_pct"])
        if v is not None:
            g[r["model"]][r["camera"]].append(v)
    models = sorted(g)
    cams = sorted({c for m in g for c in g[m]})
    if not models or not cams:
        return
    M = np.full((len(models), len(cams)), np.nan)
    for i, m in enumerate(models):
        for j, c in enumerate(cams):
            if g[m].get(c):
                M[i, j] = float(np.median(g[m][c]))
    fig, ax = plt.subplots(figsize=(max(12, len(cams) * 0.32), 1.1 + 0.55 * len(models)))
    im = ax.imshow(M, aspect="auto", cmap="inferno_r", vmin=0,
                   vmax=float(np.nanpercentile(M, 95)))
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([short(m) for m in models], fontsize=8)
    ax.set_xticks(range(len(cams)))
    ax.set_xticklabels([c.replace("brics-odroid-", "") for c in cams], rotation=90,
                       fontsize=6)
    ax.set_title("Median focal error [%] per camera view", fontsize=10)
    fig.colorbar(im, ax=ax, label="median relative focal error [%]")
    fig.tight_layout()
    fig.savefig(FIG / "per_camera_error_heatmap.png", dpi=130)
    plt.close(fig)
    print("  wrote", rel(FIG / "per_camera_error_heatmap.png"))


def fig_runtime(summ):
    rows = [s for s in summ if s["is_reference_only"] != "1"
            and s["focal_error_pct_median"] not in ("", None)
            and s["runtime_ms_median"] not in ("", None)]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for s, c in zip(rows, COLORS * 3):
        x, y = float(s["runtime_ms_median"]), float(s["focal_error_pct_median"])
        ax.scatter(x, y, s=110, color=c, alpha=0.85)
        ax.annotate(short(s["model"]), (x, y), textcoords="offset points",
                    xytext=(8, 5), fontsize=8)
    ax.axhline(5, color="#16a34a", ls="--", lw=1, label="5 % target")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("median runtime per frame [ms] (log)", fontsize=9)
    ax.set_ylabel("median focal error [%] (log)", fontsize=9)
    ax.set_title("Runtime vs accuracy", fontsize=10.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "runtime_vs_accuracy.png", dpi=130)
    plt.close(fig)
    print("  wrote", rel(FIG / "runtime_vs_accuracy.png"))


def _frame_image(seq, cam, frame):
    import cv2
    from experiments.src.datasets.gigahands import takes
    global _TAKES
    try:
        _TAKES
    except NameError:
        _TAKES = {t.name: t for t in takes()}
    t = _TAKES.get(seq)
    if t is None:
        return None
    v = t.video_path(cam)
    if v is None:
        return None
    cap = cv2.VideoCapture(str(v))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, img = cap.read()
    cap.release()
    return img[:, :, ::-1] if ok else None


def fig_examples(rows):
    """Best / typical / worst frames per model, with the numbers on the image."""
    reps = read_csv(RUN_DIR / "tables" / "representative_frames.csv")
    by_model = defaultdict(lambda: defaultdict(list))
    for r in reps:
        by_model[r["model"]][r["kind"]].append(r)
    out_dir = FIG / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    for model, kinds in sorted(by_model.items()):
        picks = []
        for kind in ("best", "typical", "worst"):
            picks += [(kind, r) for r in kinds.get(kind, [])[:3]]
        if not picks:
            continue
        fig, axes = plt.subplots(3, 3, figsize=(13, 8))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (kind, r) in zip(axes.ravel(), picks):
            img = _frame_image(r["sequence"], r["camera"], int(r["frame"]))
            if img is None:
                continue
            ax.axis("on"); ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
            gt, pr = float(r["gt_fx"]), float(r["pred_fx"])
            ax.set_title(
                f"[{kind}] {r['sequence']} {r['camera'].replace('brics-odroid-', '')} "
                f"f{r['frame']}\nGT {gt:.0f}px / pred {pr:.0f}px  "
                f"err {float(r['relative_focal_error_pct']):.1f}%\n"
                f"GT HFOV {float(r['gt_hfov']):.1f}° / pred {float(r['pred_hfov']):.1f}°",
                fontsize=7.5)
        fig.suptitle(f"{model} - best / typical / worst single-frame predictions",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        key = (model.split("[")[0] + ("_" + model.split("[")[1].rstrip("]")
                                      if "[" in model else "")).replace("/", "_")
        out = out_dir / f"{key}_best_worst_grid.png"
        fig.savefig(out, dpi=100)
        plt.close(fig)
        print("  wrote", rel(out))


def main_figure(rows, summ):
    g = errors_by_model(rows)
    ranked = [s for s in summ if s["is_reference_only"] != "1"
              and s["focal_error_pct_median"] != ""]
    ranked.sort(key=lambda s: float(s["focal_error_pct_median"]))

    fig = plt.figure(figsize=(17.5, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.26, height_ratios=[1, 1, 0.85])

    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    ax.text(0, 1.0, "The question", fontsize=12.5, fontweight="bold", va="top")
    best = ranked[0] if ranked else None
    ax.text(0, 0.85,
            "Can an existing single-image calibration model\n"
            "recover the physical focal of a static workspace\n"
            "camera to within 5 %?\n\n"
            "Why 5 %: CAM-EXP-002 measured on this same data\n"
            f"that a 5 % focal error displaces the reconstructed\n"
            f"hand by about {CAM002_MM_AT_5PCT:.0f} mm in depth, which is the\n"
            "point where the focal stops dominating the\n"
            "pipeline's own residual error.\n\n"
            "Every model sees the SAME 1400 frames from\n"
            "175 static camera views (8 frames each).\n"
            "Failures are counted, not dropped.",
            fontsize=9, va="top")

    ax = fig.add_subplot(gs[0, 1:])
    names = list(g)
    bp = ax.boxplot([g[k] for k in names], tick_labels=[short(n) for n in names],
                    showfliers=False, patch_artist=True, widths=0.5, vert=True)
    for patch, c in zip(bp["boxes"], COLORS * 3):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    for t, ls in zip(THRESHOLDS, ("-", "--", ":")):
        ax.axhline(t, color="#64748b", ls=ls, lw=1)
        ax.text(0.45, t, f"{t:.0f}%", fontsize=7.5, color="#334155", va="bottom")
    ax.set_yscale("log")
    ax.set_ylabel("relative focal error [%] (log)", fontsize=9)
    ax.set_title("Single-frame focal error (1400 frames per model)", fontsize=11)
    plt.setp(ax.get_xticklabels(), fontsize=8, rotation=10)

    ax = fig.add_subplot(gs[1, 0])
    for (name, v), c in zip(g.items(), COLORS * 3):
        x = np.sort(v); y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, lw=1.8, color=c, label=short(name))
    for t in THRESHOLDS:
        ax.axvline(t, color="#64748b", ls=":", lw=1)
    ax.set_xscale("log"); ax.set_xlim(0.3, 1000); ax.set_ylim(0, 1)
    ax.set_xlabel("relative focal error [%]", fontsize=9)
    ax.set_ylabel("fraction of frames", fontsize=9)
    ax.set_title("CDF", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[1, 1])
    x = np.arange(len(ranked))
    for i, (t, c) in enumerate(zip(THRESHOLDS, ("#16a34a", "#f59e0b", "#94a3b8"))):
        vals = [float(s[f"within_{int(t)}pct_rate"]) for s in ranked]
        ax.bar(x + (i - 1) * 0.27, vals, 0.26, color=c, label=f"<= {t:.0f} %")
    ax.set_xticks(x)
    ax.set_xticklabels([short(s["model"]) for s in ranked], fontsize=7.5, rotation=14)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of frames", fontsize=9)
    ax.set_title("Within-threshold success rate", fontsize=10)
    ax.legend(fontsize=7.5)

    ax = fig.add_subplot(gs[1, 2])
    st = read_csv(RUN_DIR / "results" / "summary" / "single_frame_stability.csv")
    gs2 = defaultdict(list)
    for r in st:
        if r["model"] in REFERENCE_ONLY:
            continue
        v = num(r["range_pct_of_mean"])
        if v is not None:
            gs2[r["model"]].append(v)
    nm = sorted(gs2, key=lambda k: np.median(gs2[k]))
    if nm:
        bp = ax.boxplot([gs2[k] for k in nm], tick_labels=[short(n) for n in nm],
                        showfliers=False, patch_artist=True, widths=0.5)
        for patch, c in zip(bp["boxes"], COLORS * 3):
            patch.set_facecolor(c); patch.set_alpha(0.55)
    ax.set_ylabel("focal spread over 8 frames [% of mean]", fontsize=9)
    ax.set_title("Same camera, different frames:\nsingle-frame instability", fontsize=10)
    plt.setp(ax.get_xticklabels(), fontsize=7.5, rotation=14)

    ax = fig.add_subplot(gs[2, :]); ax.axis("off")
    lines = [f"{'model':34s} {'median':>8s} {'p90':>8s} {'<=5%':>7s} {'<=10%':>7s} "
             f"{'<=20%':>7s} {'fail':>6s} {'ms':>7s}  principal point"]
    lines.append("-" * 118)
    for s in ranked:
        pp = s["principal_point_error_px_median"]
        pp = "NOT_PREDICTED" if pp == "NOT_PREDICTED" else f"{float(pp):.0f} px median"
        lines.append(
            f"{short(s['model']):34s} {float(s['focal_error_pct_median']):7.1f}% "
            f"{float(s['focal_error_pct_p90']):7.1f}% "
            f"{float(s['within_5pct_rate']):6.1%} {float(s['within_10pct_rate']):6.1%} "
            f"{float(s['within_20pct_rate']):6.1%} "
            f"{float(s['failure_rate']):5.1%} {float(s['runtime_ms_median']):6.0f}  {pp}")
    if best:
        lines += ["", f"Best model: {short(best['model'])} - "
                      f"{float(best['within_5pct_rate']):.1%} of single frames land within "
                      f"5 % of the true focal.",
                  "ORACLE_GT (0 % by construction) is a reference line only and is not "
                  "ranked as an estimator."]
    ax.text(0, 1.0, "\n".join(lines), fontsize=9, va="top", family="monospace")

    fig.suptitle("CAM-EXP-003 - can one RGB frame give the physical focal of a static "
                 "workspace camera?\n"
                 "GigaHands, 175 static views x 8 frames, identical frames for every model",
                 fontsize=12.5)
    fig.subplots_adjust(top=0.90, bottom=0.03, left=0.04, right=0.98)
    out = FIG / "CAM_EXP_003_MAIN_EXPLANATION.png"
    fig.savefig(out, dpi=105)
    plt.close(fig)
    print("  wrote", rel(out))


def main() -> dict:
    FIG.mkdir(parents=True, exist_ok=True)
    rows, summ = load()
    g = errors_by_model(rows)
    fig_boxplot(g)
    fig_cdf(g)
    fig_within(summ)
    fig_scatter(rows)
    fig_stability()
    fig_camera_heatmap(rows)
    fig_runtime(summ)
    fig_examples(rows)
    main_figure(rows, summ)
    return {"models_plotted": len(g)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print(json.dumps(main(), indent=2))
