"""Figures for CAM-EXP-003.1."""
from __future__ import annotations

import json
import logging
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CAM003, RUN_DIR, read_csv, rel

log = logging.getLogger("cam-exp-003.1")
FIG = RUN_DIR / "figures"
THRESHOLDS = (5.0, 10.0, 20.0)

COND_COLOR = {"A_RAW_PINHOLE": "#dc2626",
              "B_RAW_DISTORTION_AWARE": "#2563eb",
              "C_GT_UNDISTORTED_PINHOLE": "#16a34a"}
COND_LABEL = {"A_RAW_PINHOLE": "A raw + pinhole",
              "B_RAW_DISTORTION_AWARE": "B raw + distortion-aware",
              "C_GT_UNDISTORTED_PINHOLE": "C GT-undistorted + pinhole (oracle)"}

# Families: which runs belong together, in condition order
FAMILIES = {
    "AnyCalib": [("anycalib (CAM-EXP-003)", "A_RAW_PINHOLE"),
                 ("anycalib_dist_radial", "B_RAW_DISTORTION_AWARE"),
                 ("anycalib_gen_radial", "B_RAW_DISTORTION_AWARE"),
                 ("undist_anycalib_pinhole", "C_GT_UNDISTORTED_PINHOLE")],
    "GeoCalib": [("geocalib (CAM-EXP-003)", "A_RAW_PINHOLE"),
                 ("geocalib_distorted_radial", "B_RAW_DISTORTION_AWARE"),
                 ("undist_geocalib_pinhole", "C_GT_UNDISTORTED_PINHOLE")],
    "PF centered": [("pf_centered (CAM-EXP-003)", "A_RAW_PINHOLE"),
                    ("undist_pf_centered", "C_GT_UNDISTORTED_PINHOLE")],
    "PF uncentered": [("pf_uncentered (CAM-EXP-003)", "A_RAW_PINHOLE"),
                      ("undist_pf_uncentered", "C_GT_UNDISTORTED_PINHOLE")],
}
SHORT = {"anycalib (CAM-EXP-003)": "A pinhole", "anycalib_dist_radial": "B dist",
         "anycalib_gen_radial": "B gen", "undist_anycalib_pinhole": "C undist",
         "geocalib (CAM-EXP-003)": "A pinhole",
         "geocalib_distorted_radial": "B radial",
         "undist_geocalib_pinhole": "C undist",
         "pf_centered (CAM-EXP-003)": "A pinhole", "undist_pf_centered": "C undist",
         "pf_uncentered (CAM-EXP-003)": "A pinhole",
         "undist_pf_uncentered": "C undist"}


def num(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def load_errors() -> dict:
    """relative focal error arrays per run."""
    out = {}
    for key in ("anycalib", "geocalib", "pf_centered", "pf_uncentered"):
        p = CAM003 / "results" / "raw" / f"{key}_predictions.csv.gz"
        if p.exists():
            out[f"{key} (CAM-EXP-003)"] = np.array(
                [num(r["relative_focal_error_pct"]) for r in read_csv(p)
                 if r["success"] == "1"], dtype=float)
    for p in sorted((RUN_DIR / "results" / "raw").glob("*_predictions.csv.gz")):
        key = p.name.replace("_predictions.csv.gz", "")
        out[key] = np.array([num(r["relative_focal_error_pct"]) for r in read_csv(p)
                             if str(r.get("success")) == "1"], dtype=float)
    return {k: v[np.isfinite(v)] for k, v in out.items() if v.size}


def fig_family_comparison(errs, summ):
    n = len(FAMILIES)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 5.0), sharey=True)
    for ax, (fam, runs) in zip(np.atleast_1d(axes), FAMILIES.items()):
        runs = [(r, c) for r, c in runs if r in errs]
        data = [errs[r] for r, _ in runs]
        bp = ax.boxplot(data, tick_labels=[SHORT.get(r, r) for r, _ in runs],
                        showfliers=False, patch_artist=True, widths=0.55)
        for patch, (_, c) in zip(bp["boxes"], runs):
            patch.set_facecolor(COND_COLOR[c]); patch.set_alpha(0.55)
        for i, (r, _) in enumerate(runs, start=1):
            ax.text(i, np.median(errs[r]), f" {np.median(errs[r]):.1f}%",
                    fontsize=8, va="bottom", ha="center")
        for t, ls in zip(THRESHOLDS, ("-", "--", ":")):
            ax.axhline(t, color="#64748b", ls=ls, lw=1)
        ax.set_yscale("log")
        ax.set_title(fam, fontsize=11)
        plt.setp(ax.get_xticklabels(), fontsize=8, rotation=12)
    np.atleast_1d(axes)[0].set_ylabel("relative focal error [%] (log)", fontsize=9)
    handles = [plt.Line2D([], [], color=COND_COLOR[c], lw=8, alpha=0.55,
                          label=COND_LABEL[c]) for c in COND_COLOR]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Does modelling lens distortion explain the CAM-EXP-003 focal error?\n"
                 "same 1400 frames, paired per frame", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    fig.savefig(FIG / "pinhole_vs_distortion_aware_focal_error.png", dpi=130)
    plt.close(fig)
    print("  wrote", rel(FIG / "pinhole_vs_distortion_aware_focal_error.png"))


def fig_within(summ):
    rows = [s for s in summ if s["focal_error_pct_median"] != ""]
    order = []
    for fam, runs in FAMILIES.items():
        for r, _ in runs:
            m = next((s for s in rows if s["model_run"] == r), None)
            if m:
                order.append((fam, m))
    fig, ax = plt.subplots(figsize=(13, 4.8))
    x = np.arange(len(order))
    for i, (t, c) in enumerate(zip(THRESHOLDS, ("#16a34a", "#f59e0b", "#94a3b8"))):
        vals = [float(m[f"within_{int(t)}pct_rate"]) for _, m in order]
        b = ax.bar(x + (i - 1) * 0.27, vals, 0.26, color=c, label=f"<= {t:.0f} %")
        for bb, v in zip(b, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v, f"{v:.0%}", ha="center",
                    va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{fam}\n{SHORT.get(m['model_run'], m['model_run'])}"
                        for fam, m in order], fontsize=7.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("fraction of frames", fontsize=9)
    ax.set_title("Within-threshold rate by condition", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "within_threshold_comparison.png", dpi=130)
    plt.close(fig)
    print("  wrote", rel(FIG / "within_threshold_comparison.png"))


def fig_per_camera():
    rows = read_csv(RUN_DIR / "results" / "summary" / "per_camera_summary.csv")
    g = defaultdict(list)
    for r in rows:
        v = num(r["improvement_pct_points"])
        if v is not None:
            g[r["model_run"]].append(v)
    if not g:
        return
    names = sorted(g, key=lambda k: -np.median(g[k]))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    bp = ax.boxplot([g[k] for k in names], tick_labels=names, showfliers=False,
                    patch_artist=True, widths=0.55)
    for patch in bp["boxes"]:
        patch.set_facecolor("#2563eb"); patch.set_alpha(0.5)
    ax.axhline(0, color="#dc2626", ls="--", lw=1.2, label="no change")
    ax.set_ylabel("per-camera improvement [percentage points]", fontsize=9)
    ax.set_title("Improvement over raw+pinhole, per camera view\n"
                 "(positive = distortion handling helps)", fontsize=11)
    plt.setp(ax.get_xticklabels(), fontsize=7, rotation=14)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "per_camera_improvement.png", dpi=130)
    plt.close(fig)
    print("  wrote", rel(FIG / "per_camera_improvement.png"))


def fig_distortion_vs_improvement():
    rows = read_csv(RUN_DIR / "results" / "summary" / "per_camera_summary.csv")
    corr = {r["model_run"]: r for r in
            read_csv(RUN_DIR / "results" / "summary" / "distortion_vs_improvement.csv")}
    runs = ["anycalib_gen_radial", "geocalib_distorted_radial",
            "undist_anycalib_pinhole", "undist_geocalib_pinhole"]
    runs = [r for r in runs if any(x["model_run"] == r for x in rows)]
    fig, axes = plt.subplots(1, len(runs), figsize=(4.1 * len(runs), 4.2), sharey=True)
    for ax, run in zip(np.atleast_1d(axes), runs):
        pts = [(num(r["gt_abs_k1_median"]), num(r["improvement_pct_points"]))
               for r in rows if r["model_run"] == run and r["gt_abs_k1_median"] != ""]
        pts = [(a, b) for a, b in pts if a is not None and b is not None]
        if not pts:
            continue
        a = np.asarray(pts)
        ax.scatter(a[:, 0], a[:, 1], s=26, color="#2563eb", alpha=0.75)
        ax.axhline(0, color="#dc2626", ls="--", lw=1)
        c = corr.get(run, {})
        ax.set_title(f"{run}\nPearson r={c.get('pearson_r','?')}, "
                     f"Spearman r={c.get('spearman_r','?')}", fontsize=8.5)
        ax.set_xlabel("GT |k1| of the camera", fontsize=8.5)
    np.atleast_1d(axes)[0].set_ylabel("improvement [pp]", fontsize=9)
    fig.suptitle("Do cameras with stronger distortion gain more?\n"
                 "GT |k1| spans only 0.34-0.43 here, so this tests a narrow range",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(FIG / "distortion_magnitude_vs_improvement.png", dpi=130)
    plt.close(fig)
    print("  wrote", rel(FIG / "distortion_magnitude_vs_improvement.png"))


def fig_bias_before_after():
    st = {r["model_run"]: r for r in
          read_csv(RUN_DIR / "results" / "summary" / "stability_summary.csv")}
    pairs = [("A:anycalib", "anycalib_gen_radial", "AnyCalib"),
             ("A:anycalib", "undist_anycalib_pinhole", "AnyCalib (undist)"),
             ("A:geocalib", "geocalib_distorted_radial", "GeoCalib"),
             ("A:geocalib", "undist_geocalib_pinhole", "GeoCalib (undist)"),
             ("A:pf_uncentered", "undist_pf_uncentered", "PF unc. (undist)"),
             ("A:pf_centered", "undist_pf_centered", "PF cen. (undist)")]
    pairs = [(a, b, l) for a, b, l in pairs if a in st and b in st]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    x = np.arange(len(pairs))
    before = [float(st[a]["median_abs_view_bias_pct"]) for a, _, _ in pairs]
    after = [float(st[b]["median_abs_view_bias_pct"]) for _, b, _ in pairs]
    ax.bar(x - 0.2, before, 0.38, color="#dc2626", alpha=0.8, label="raw + pinhole")
    ax.bar(x + 0.2, after, 0.38, color="#2563eb", alpha=0.8, label="distortion handled")
    for i, (b1, b2) in enumerate(zip(before, after)):
        ax.text(i - 0.2, b1, f"{b1:.1f}", ha="center", va="bottom", fontsize=7.5)
        ax.text(i + 0.2, b2, f"{b2:.1f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, _, l in pairs], fontsize=8, rotation=10)
    ax.set_ylabel("median |per-view focal bias| [%]", fontsize=9)
    ax.set_title("Static-camera bias before and after handling distortion\n"
                 "(bias = median signed error of the 8 frames of one fixed camera)",
                 fontsize=10.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "static_camera_bias_before_after.png", dpi=130)
    plt.close(fig)
    print("  wrote", rel(FIG / "static_camera_bias_before_after.png"))


def fig_examples():
    """Raw vs GT-undistorted frames for a low- and a high-|k1| camera."""
    import cv2
    from common import raw_frame_path, undist_frame_path, load_frames
    frames = load_frames()
    by_cam = {}
    for r in frames:
        by_cam.setdefault((r["sequence"], r["camera"]), r)
    ordered = sorted(by_cam.values(), key=lambda r: abs(float(r.get("gt_k1") or 0)))
    picks = [("weakest distortion", ordered[0]), ("median distortion",
             ordered[len(ordered) // 2]), ("strongest distortion", ordered[-1])]

    preds = {}
    for key in ("anycalib_gen_radial", "undist_anycalib_pinhole"):
        p = RUN_DIR / "results" / "raw" / f"{key}_predictions.csv.gz"
        if p.exists():
            preds[key] = {(r["sequence"], r["camera"], str(int(float(r["frame"])))): r
                          for r in read_csv(p)}
    pa = CAM003 / "results" / "raw" / "anycalib_predictions.csv.gz"
    preds["anycalib_pinhole"] = {
        (r["sequence"], r["camera"], str(int(float(r["frame"])))): r
        for r in read_csv(pa)} if pa.exists() else {}

    fig, axes = plt.subplots(len(picks), 2, figsize=(11, 4.0 * len(picks)))
    for row, (label, r) in enumerate(picks):
        k = (r["sequence"], r["camera"], str(int(r["frame"])))
        raw = cv2.imread(str(raw_frame_path(r)))
        und = cv2.imread(str(undist_frame_path(r)))
        for col, (img, tag) in enumerate(((raw, "RAW"), (und, "GT-UNDISTORTED"))):
            ax = axes[row][col]
            ax.set_xticks([]); ax.set_yticks([])
            if img is None:
                ax.axis("off"); continue
            ax.imshow(img[:, :, ::-1])
            if col == 0:
                pin = preds["anycalib_pinhole"].get(k, {})
                da = preds.get("anycalib_gen_radial", {}).get(k, {})
                ax.set_title(
                    f"{tag} | {label}\n{r['sequence']} {r['camera'].replace('brics-odroid-','')} "
                    f"f{r['frame']}  GT k1={float(r.get('gt_k1') or 0):.3f} "
                    f"GT fx={float(r['gt_fx']):.0f}\n"
                    f"A pinhole {pin.get('pred_fx','?')} ({pin.get('relative_focal_error_pct','?')}%)  "
                    f"B dist-aware {da.get('pred_fx','?')} ({da.get('relative_focal_error_pct','?')}%)",
                    fontsize=7.5)
            else:
                uc = preds.get("undist_anycalib_pinhole", {}).get(k, {})
                ax.set_title(
                    f"{tag}\nGT focal for this image (K_new) = {uc.get('gt_fx','?')}\n"
                    f"C undist pinhole {uc.get('pred_fx','?')} "
                    f"({uc.get('relative_focal_error_pct','?')}%)", fontsize=7.5)
    fig.suptitle("Representative frames: raw vs GT-undistorted, with AnyCalib predictions",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG / "examples" / "raw_vs_undistorted_examples.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print("  wrote", rel(out))


def main_figure(errs, summ):
    fig = plt.figure(figsize=(17.5, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.26, height_ratios=[1, 1, 0.85])

    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    ax.text(0, 1.0, "Why this experiment", fontsize=12.5, fontweight="bold", va="top")
    ax.text(0, 0.86,
            "CAM-EXP-003 benchmarked calibration models with\n"
            "PINHOLE camera models, but the GigaHands lenses\n"
            "have real barrel distortion (GT |k1| = 0.34-0.43).\n\n"
            "So: how much of that 16-25 % focal error was just\n"
            "the missing distortion term?\n\n"
            "Three conditions, same 1400 frames, paired:\n"
            "  A  raw image + pinhole      (CAM-EXP-003, reused)\n"
            "  B  raw image + distortion-aware model\n"
            "  C  GT-undistorted image + pinhole  (ORACLE:\n"
            "     needs the true distortion, not deployable)\n\n"
            "For C the ground truth is K_new from\n"
            "getOptimalNewCameraMatrix, not the original focal -\n"
            "undistortion changes the focal by ~0.77x.",
            fontsize=8.6, va="top")

    ax = fig.add_subplot(gs[0, 1:])
    runs, cols = [], []
    for fam, rr in FAMILIES.items():
        for r, c in rr:
            if r in errs:
                runs.append((fam, r, c)); cols.append(COND_COLOR[c])
    bp = ax.boxplot([errs[r] for _, r, _ in runs],
                    tick_labels=[f"{f}\n{SHORT.get(r, r)}" for f, r, _ in runs],
                    showfliers=False, patch_artist=True, widths=0.55)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    for i, (_, r, _) in enumerate(runs, start=1):
        ax.text(i, np.median(errs[r]), f" {np.median(errs[r]):.1f}%", fontsize=7.5,
                va="bottom", ha="center")
    for t, ls in zip(THRESHOLDS, ("-", "--", ":")):
        ax.axhline(t, color="#64748b", ls=ls, lw=1)
    ax.set_yscale("log")
    ax.set_ylabel("relative focal error [%] (log)", fontsize=9)
    ax.set_title("Focal error by condition (median printed)", fontsize=11)
    plt.setp(ax.get_xticklabels(), fontsize=7)

    ax = fig.add_subplot(gs[1, 0])
    order = [(f, next((s for s in summ if s["model_run"] == r), None))
             for f, r, _ in runs]
    order = [(f, m) for f, m in order if m]
    x = np.arange(len(order))
    for i, (t, c) in enumerate(zip(THRESHOLDS, ("#16a34a", "#f59e0b", "#94a3b8"))):
        ax.bar(x + (i - 1) * 0.27, [float(m[f"within_{int(t)}pct_rate"]) for _, m in order],
               0.26, color=c, label=f"<= {t:.0f}%")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT.get(m["model_run"], m["model_run"]) for _, m in order],
                       fontsize=6.5, rotation=20)
    ax.set_ylabel("fraction of frames", fontsize=9)
    ax.set_title("Within-threshold rate", fontsize=10)
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[1, 1])
    st = {r["model_run"]: r for r in
          read_csv(RUN_DIR / "results" / "summary" / "stability_summary.csv")}
    pairs = [("A:anycalib", "anycalib_gen_radial", "AnyCalib"),
             ("A:geocalib", "geocalib_distorted_radial", "GeoCalib"),
             ("A:pf_uncentered", "undist_pf_uncentered", "PF unc."),
             ("A:pf_centered", "undist_pf_centered", "PF cen.")]
    pairs = [p for p in pairs if p[0] in st and p[1] in st]
    x = np.arange(len(pairs))
    ax.bar(x - 0.2, [float(st[a]["median_abs_view_bias_pct"]) for a, _, _ in pairs],
           0.38, color="#dc2626", alpha=0.8, label="raw + pinhole")
    ax.bar(x + 0.2, [float(st[b]["median_abs_view_bias_pct"]) for _, b, _ in pairs],
           0.38, color="#2563eb", alpha=0.8, label="distortion handled")
    ax.set_xticks(x); ax.set_xticklabels([l for _, _, l in pairs], fontsize=8)
    ax.set_ylabel("median |per-view bias| [%]", fontsize=9)
    ax.set_title("Static-camera bias", fontsize=10)
    ax.legend(fontsize=7.5)

    ax = fig.add_subplot(gs[1, 2])
    pc = read_csv(RUN_DIR / "results" / "summary" / "per_camera_summary.csv")
    g = defaultdict(list)
    for r in pc:
        v = num(r["improvement_pct_points"])
        if v is not None:
            g[r["model_run"]].append(v)
    nm = sorted(g, key=lambda k: -np.median(g[k]))
    bp = ax.boxplot([g[k] for k in nm], tick_labels=nm, showfliers=False,
                    patch_artist=True, widths=0.5)
    for patch in bp["boxes"]:
        patch.set_facecolor("#2563eb"); patch.set_alpha(0.5)
    ax.axhline(0, color="#dc2626", ls="--", lw=1.2)
    ax.set_ylabel("per-camera improvement [pp]", fontsize=9)
    ax.set_title("Improvement per camera view", fontsize=10)
    plt.setp(ax.get_xticklabels(), fontsize=5.5, rotation=22)

    ax = fig.add_subplot(gs[2, :]); ax.axis("off")
    ps = read_csv(RUN_DIR / "results" / "summary" / "paired_improvement_summary.csv")
    lines = [f"{'run':30s} {'cond':6s} {'median':>8s} {'p90':>8s} {'<=5%':>7s} "
             f"{'<=10%':>7s} {'paired gain':>13s} {'improved':>9s}"]
    lines.append("-" * 104)
    for fam, r, c in runs:
        m = next((s for s in summ if s["model_run"] == r), None)
        if not m:
            continue
        pr = next((s for s in ps if s["model_run"] == r), None)
        gain = (f"{float(pr['median_improvement_pct_points']):+7.2f} pp" if pr else "  baseline")
        frac = f"{float(pr['fraction_improved']):8.1%}" if pr else "        -"
        lines.append(f"{r[:30]:30s} {c[0]:6s} {float(m['focal_error_pct_median']):7.2f}% "
                     f"{float(m['focal_error_pct_p90']):7.2f}% "
                     f"{float(m['within_5pct_rate']):6.1%} {float(m['within_10pct_rate']):6.1%} "
                     f"{gain:>13s} {frac:>9s}")
    lines += ["",
              "Distortion explains a large but not decisive part: the best AnyCalib median "
              "falls 16.3 % -> 9.6 % and <=5 % rises 11.4 % -> 20.1 %,",
              "yet 4 frames in 5 are still outside the +-5 % target that CAM-EXP-002 set. "
              "Conditions B and C agree, so this is the distortion term itself,",
              "not an artefact of one model's parameterisation. Per-view bias, not "
              "frame noise, remains the dominant residual."]
    ax.text(0, 1.0, "\n".join(lines), fontsize=8.6, va="top", family="monospace")

    fig.suptitle("CAM-EXP-003.1 - how much of the single-frame focal error was lens "
                 "distortion?\nGigaHands, same 1400 frames as CAM-EXP-003, paired "
                 "per frame", fontsize=12.5)
    fig.subplots_adjust(top=0.90, bottom=0.03, left=0.04, right=0.98)
    out = FIG / "CAM_EXP_003_1_MAIN_EXPLANATION.png"
    fig.savefig(out, dpi=105)
    plt.close(fig)
    print("  wrote", rel(out))


def main() -> dict:
    FIG.mkdir(parents=True, exist_ok=True)
    errs = load_errors()
    summ = read_csv(RUN_DIR / "results" / "summary" / "model_summary.csv")
    fig_family_comparison(errs, summ)
    fig_within(summ)
    fig_per_camera()
    fig_distortion_vs_improvement()
    fig_bias_before_after()
    fig_examples()
    main_figure(errs, summ)
    return {"runs_plotted": len(errs)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print(json.dumps(main(), indent=2))
