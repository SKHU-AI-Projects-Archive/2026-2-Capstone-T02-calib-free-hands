"""Figures for CAM-EXP-002.

Absolute error and incremental focal effect are always plotted as separate
quantities, never merged: the first contains hand-model and detector error as
well, the second isolates the focal.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ALPHAS, RUN_DIR, read_csv, rel

log = logging.getLogger("cam-exp-002")
FIG = RUN_DIR / "figures"
GT_C, BASE_C = "#2563eb", "#dc2626"


def load():
    rows = read_csv(RUN_DIR / "results" / "raw" / "focal_sweep_per_hand.csv.gz")
    frames = read_csv(RUN_DIR / "results" / "raw" / "focal_sweep_per_frame.csv.gz")
    return rows, frames


def by_alpha(rows, field):
    g = defaultdict(list)
    for r in rows:
        if r["condition"] == "PIPELINE_BASELINE" or r.get(field) in ("", None):
            continue
        try:
            g[float(r["alpha"])].append(float(r[field]))
        except ValueError:
            continue
    return {a: np.asarray(v) for a, v in sorted(g.items())}


def baseline_vals(rows, field):
    return np.asarray([float(r[field]) for r in rows
                       if r["condition"] == "PIPELINE_BASELINE"
                       and r.get(field) not in ("", None)])


def _band(ax, g, color, label, absolute=True):
    a = sorted(g)
    med = [np.median(g[x]) for x in a]
    p25 = [np.percentile(g[x], 25) for x in a]
    p75 = [np.percentile(g[x], 75) for x in a]
    ax.plot(a, med, "o-", color=color, label=label, lw=1.8)
    ax.fill_between(a, p25, p75, color=color, alpha=0.18)
    return a, med


def fig_depth_error(rows):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    g = by_alpha(rows, "abs_z_error_mm")
    _band(axes[0], g, GT_C, "absolute |Z error| vs provided 3D")
    axes[0].set_xlabel("focal scale alpha (1.0 = physical GT focal)", fontsize=9)
    axes[0].set_ylabel("absolute depth error [mm]", fontsize=9)
    axes[0].set_title("Absolute depth error (median, IQR band)", fontsize=10)
    axes[0].axvline(1.0, color="#94a3b8", ls=":", lw=1)
    axes[0].legend(fontsize=8)

    gs = by_alpha(rows, "signed_z_error_mm")
    _band(axes[1], gs, "#7c3aed", "signed Z error")
    gd = by_alpha(rows, "delta_from_gt_focal_z_mm")
    _band(axes[1], gd, "#059669", "incremental shift vs alpha=1.0")
    axes[1].axhline(0, color="#94a3b8", ls=":", lw=1)
    axes[1].axvline(1.0, color="#94a3b8", ls=":", lw=1)
    axes[1].set_xlabel("focal scale alpha", fontsize=9)
    axes[1].set_ylabel("signed depth [mm]", fontsize=9)
    axes[1].set_title("Signed error and isolated focal effect", fontsize=10)
    axes[1].legend(fontsize=8)
    fig.suptitle("CAM-EXP-002 - focal scale error vs predicted depth", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "focal_scale_vs_depth_error.png", dpi=130)
    plt.close(fig)
    print("  wrote", rel(FIG / "focal_scale_vs_depth_error.png"))


def fig_root_error(rows):
    fig, ax = plt.subplots(figsize=(7, 4.3))
    _band(ax, by_alpha(rows, "root_xyz_error_mm"), GT_C, "root XYZ error vs provided 3D")
    b = baseline_vals(rows, "root_xyz_error_mm")
    if b.size:
        ax.axhline(float(np.median(b)), color=BASE_C, ls="--", lw=1.5,
                   label=f"PIPELINE_BASELINE median {np.median(b):.0f} mm")
    ax.axvline(1.0, color="#94a3b8", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("focal scale alpha", fontsize=9)
    ax.set_ylabel("root 3D error [mm] (log)", fontsize=9)
    ax.set_title("Root 3D placement error vs focal scale", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "focal_scale_vs_root_3d_error.png", dpi=130)
    plt.close(fig)
    print("  wrote", rel(FIG / "focal_scale_vs_root_3d_error.png"))


def fig_mpjpe(rows):
    fig, ax = plt.subplots(figsize=(7, 4.3))
    _band(ax, by_alpha(rows, "absolute_mpjpe_mm"), GT_C, "absolute 21-joint MPJPE")
    g = by_alpha(rows, "root_aligned_mpjpe_mm")
    a = sorted(g)
    ax.plot(a, [np.median(g[x]) for x in a], "s--", color="#f59e0b",
            label="root-aligned MPJPE (pose only)")
    ax.axvline(1.0, color="#94a3b8", ls=":", lw=1)
    ax.set_xlabel("focal scale alpha", fontsize=9)
    ax.set_ylabel("MPJPE [mm]", fontsize=9)
    ax.set_title("Absolute vs root-aligned joint error\n"
                 "root-aligned is flat: the focal moves the hand, it does not deform it",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "focal_scale_vs_absolute_mpjpe.png", dpi=130)
    plt.close(fig)
    print("  wrote", rel(FIG / "focal_scale_vs_absolute_mpjpe.png"))


def fig_incremental(rows):
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for field, color, lbl in (("delta_from_gt_focal_root_mm", "#0ea5e9",
                               "root shift |dT|"),
                              ("delta_from_gt_focal_joint_mm", "#16a34a",
                               "mean joint shift")):
        g = by_alpha(rows, field)
        a = sorted(g)
        ax.plot(a, [np.median(g[x]) for x in a], "o-", color=color, label=lbl)
    ax.axvline(1.0, color="#94a3b8", ls=":", lw=1)
    ax.set_xlabel("focal scale alpha", fontsize=9)
    ax.set_ylabel("displacement from the alpha=1.0 prediction [mm]", fontsize=9)
    ax.set_title("Isolated focal effect: how far the hand moves when only the\n"
                 "focal changes (same detection, same crop, same regression)",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "focal_scale_vs_incremental_shift.png", dpi=130)
    plt.close(fig)
    print("  wrote", rel(FIG / "focal_scale_vs_incremental_shift.png"))


def fig_depth_ratio(rows):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    g = by_alpha(rows, "pred_z_ratio_vs_alpha1")
    a = sorted(g)
    med = [np.median(g[x]) for x in a]
    ax.plot([min(a), max(a)], [min(a), max(a)], "k--", lw=1.2, label="y = x (exact Z ∝ f)")
    ax.plot(a, med, "o", color=GT_C, ms=8, label="measured median ratio")
    resid = max(abs(m - x) for m, x in zip(med, a))
    ax.set_xlabel("focal scale alpha", fontsize=9)
    ax.set_ylabel("predicted Z(alpha) / Z(alpha=1)", fontsize=9)
    ax.set_title(f"Depth proportionality check\nmax |median ratio - alpha| = {resid:.2e}",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "predicted_depth_ratio_vs_focal_scale.png", dpi=130)
    plt.close(fig)
    print("  wrote", rel(FIG / "predicted_depth_ratio_vs_focal_scale.png"))
    return resid


def fig_baseline_vs_gt(rows, frames):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    pairs = [("abs_z_error_mm", "absolute depth error"),
             ("root_xyz_error_mm", "root 3D error"),
             ("absolute_mpjpe_mm", "absolute 21-joint MPJPE")]
    for ax, (field, label) in zip(axes, pairs):
        b = baseline_vals(rows, field)
        g = np.asarray([float(r[field]) for r in rows
                        if r["condition"] == "GT_x1.00" and r[field] not in ("", None)])
        data = [b[np.isfinite(b)], g[np.isfinite(g)]]
        bp = ax.boxplot(data, tick_labels=["PIPELINE\nBASELINE\n(f=5000px)",
                                      "PHYSICAL GT\n(f≈920px)"],
                        showfliers=False, patch_artist=True, widths=0.55)
        for patch, c in zip(bp["boxes"], [BASE_C, GT_C]):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        ax.set_yscale("log")
        ax.set_ylabel(f"{label} [mm] (log)", fontsize=9)
        ax.set_title(f"{label}\nmedian {np.median(data[0]):.0f} -> {np.median(data[1]):.0f} mm",
                     fontsize=9.5)
    fig.suptitle("Existing pipeline focal vs the camera's physical focal", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "baseline_vs_physical_gt_focal.png", dpi=130)
    plt.close(fig)
    print("  wrote", rel(FIG / "baseline_vs_physical_gt_focal.png"))


def fig_per_camera(rows):
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["condition"] in ("GT_x1.00", "GT_x1.10"):
            try:
                g[r["camera"]][r["condition"]].append(float(r["root_xyz_error_mm"]))
            except (ValueError, KeyError):
                pass
    cams = sorted(g)
    if not cams:
        return
    x = np.arange(len(cams))
    v1 = [np.median(g[c].get("GT_x1.00", [np.nan])) for c in cams]
    v2 = [np.median(g[c].get("GT_x1.10", [np.nan])) for c in cams]
    fig, ax = plt.subplots(figsize=(max(11, len(cams) * 0.35), 4.3))
    ax.bar(x - 0.2, v1, 0.4, label="GT focal (alpha=1.00)", color=GT_C)
    ax.bar(x + 0.2, v2, 0.4, label="GT focal +10 % (alpha=1.10)", color="#f97316")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("brics-odroid-", "") for c in cams], rotation=90,
                       fontsize=6)
    ax.set_ylabel("median root 3D error [mm]", fontsize=9)
    ax.set_title("Per-camera sensitivity", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "per_camera_sensitivity.png", dpi=130)
    plt.close(fig)
    print("  wrote", rel(FIG / "per_camera_sensitivity.png"))


def main_figure(rows, frames, resid):
    fig = plt.figure(figsize=(17, 11.5))
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.28, height_ratios=[1, 1, 0.75])

    # --- concept ---
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    ax.text(0, 1.0, "What is measured", fontsize=12, fontweight="bold", va="top")
    ax.text(0, 0.86,
            "The existing pipeline never measures a focal.\n"
            "It assumes  f = FOCAL_LENGTH/IMAGE_SIZE x max(W,H)\n"
            "          = 1000/256 x 1280 = 5000 px\n\n"
            "The camera's real focal is fx ≈ 920 px,\n"
            "so the assumed focal is 5.4x too long.\n\n"
            "In rgb_predictor.py the focal enters only here:\n"
            "     tz = 2 f / (s · B)        <- depth\n"
            "     tx, ty : no focal term    <- unchanged\n\n"
            "So a wrong focal slides the hand along Z only,\n"
            "and the 2D reprojection does not change at all.\n"
            "One inference per frame is cached; only this\n"
            "conversion is recomputed per condition.",
            fontsize=8.6, va="top", family="monospace")

    # --- depth ratio ---
    ax = fig.add_subplot(gs[0, 1])
    g = by_alpha(rows, "pred_z_ratio_vs_alpha1")
    a = sorted(g)
    ax.plot([min(a), max(a)], [min(a), max(a)], "k--", lw=1.2, label="y = x")
    ax.plot(a, [np.median(g[x]) for x in a], "o", color=GT_C, ms=8, label="measured")
    ax.set_xlabel("focal scale alpha", fontsize=9)
    ax.set_ylabel("Z(alpha) / Z(1.0)", fontsize=9)
    ax.set_title(f"Depth scales exactly with focal\nmax deviation {resid:.1e}", fontsize=10)
    ax.legend(fontsize=8)

    # --- incremental effect ---
    ax = fig.add_subplot(gs[0, 2])
    gi = by_alpha(rows, "delta_from_gt_focal_root_mm")
    a = sorted(gi)
    med = [np.median(gi[x]) for x in a]
    ax.plot(a, med, "o-", color="#0ea5e9")
    for x, m in zip(a, med):
        if abs(x - 1.0) > 1e-9:
            ax.annotate(f"{m:.0f}", (x, m), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7.5)
    ax.set_xlabel("focal scale alpha", fontsize=9)
    ax.set_ylabel("hand displacement [mm]", fontsize=9)
    ax.set_title("Isolated focal effect\n(how far the hand moves, same inference)",
                 fontsize=10)

    # --- absolute depth error ---
    ax = fig.add_subplot(gs[1, 0])
    _band(ax, by_alpha(rows, "abs_z_error_mm"), GT_C, "|Z error|")
    ax.axvline(1.0, color="#94a3b8", ls=":", lw=1)
    ax.set_xlabel("focal scale alpha", fontsize=9)
    ax.set_ylabel("absolute depth error [mm]", fontsize=9)
    ax.set_title("Absolute depth error vs provided 3D", fontsize=10)
    ax.legend(fontsize=8)

    # --- absolute vs root-aligned ---
    ax = fig.add_subplot(gs[1, 1])
    _band(ax, by_alpha(rows, "absolute_mpjpe_mm"), GT_C, "absolute MPJPE")
    gr = by_alpha(rows, "root_aligned_mpjpe_mm")
    a = sorted(gr)
    ax.plot(a, [np.median(gr[x]) for x in a], "s--", color="#f59e0b",
            label="root-aligned MPJPE")
    ax.set_xlabel("focal scale alpha", fontsize=9)
    ax.set_ylabel("MPJPE [mm]", fontsize=9)
    ax.set_title("Pose is untouched; only placement moves", fontsize=10)
    ax.legend(fontsize=8)

    # --- baseline vs GT ---
    ax = fig.add_subplot(gs[1, 2])
    b = baseline_vals(rows, "root_xyz_error_mm")
    g1 = np.asarray([float(r["root_xyz_error_mm"]) for r in rows
                     if r["condition"] == "GT_x1.00"
                     and r["root_xyz_error_mm"] not in ("", None)])
    bp = ax.boxplot([b[np.isfinite(b)], g1[np.isfinite(g1)]],
                    tick_labels=["PIPELINE\nBASELINE", "PHYSICAL\nGT FOCAL"],
                    showfliers=False, patch_artist=True, widths=0.55)
    for patch, c in zip(bp["boxes"], [BASE_C, GT_C]):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.set_yscale("log")
    ax.set_ylabel("root 3D error [mm] (log)", fontsize=9)
    ax.set_title(f"median {np.median(b):.0f} mm  ->  {np.median(g1):.0f} mm",
                 fontsize=10)

    # --- numbers ---
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    sens = read_csv(RUN_DIR / "results" / "summary" / "focal_sensitivity_summary.csv")
    lines = ["focal error   |  isolated hand displacement (median)  |  absolute depth error (median)"]
    lines.append("-" * 96)
    ga = by_alpha(rows, "abs_z_error_mm")
    for s in sens:
        al = float(s["alpha"])
        if abs(al - 1.0) < 1e-9:
            continue
        shift = s.get("incremental_root_shift_median_mm", "")
        absz = np.median(ga[al]) if al in ga else float("nan")
        lines.append(f"{(al - 1) * 100:+6.0f} %       |  {float(shift):9.1f} mm"
                     f"                          |  {absz:9.1f} mm")
    n_hands = len({(r['sequence'], r['camera'], r['frame'], r['hand']) for r in rows})
    base_med = np.median(b[np.isfinite(b)])
    gt_med = np.median(g1[np.isfinite(g1)])
    lines += ["", f"clean hand observations: {n_hands}   "
                  f"(bimanual-clean PASS_STRICT frames with usable RGB)",
              f"PIPELINE_BASELINE root error median {base_med:.0f} mm    "
              f"PHYSICAL GT focal median {gt_med:.0f} mm    "
              f"improvement {base_med / max(gt_med, 1e-9):.1f}x",
              "root-aligned pose error is identical in every condition: the focal is a "
              "global placement problem, not a pose problem."]
    ax.text(0, 1.0, "\n".join(lines), fontsize=9.2, va="top", family="monospace")

    fig.suptitle(
        "CAM-EXP-002 - how wrong is the absolute 3D hand when the focal length is wrong?\n"
        "Existing AnyHand/WiLoR pipeline, GigaHands bimanual-clean subset, one inference "
        "per frame with only the focal-dependent conversion recomputed",
        fontsize=12.5)
    fig.subplots_adjust(top=0.90, bottom=0.04, left=0.05, right=0.97)
    out = FIG / "CAM_EXP_002_MAIN_EXPLANATION.png"
    fig.savefig(out, dpi=105)
    plt.close(fig)
    print("  wrote", rel(out))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    rows, frames = load()
    fig_depth_error(rows)
    fig_root_error(rows)
    fig_mpjpe(rows)
    fig_incremental(rows)
    resid = fig_depth_ratio(rows)
    fig_baseline_vs_gt(rows, frames)
    fig_per_camera(rows)
    main_figure(rows, frames, resid)
    return {"max_depth_ratio_deviation": resid}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print(json.dumps(main(), indent=2))
