"""Report figures: white background, print-safe, readable without colour.

No new research result is computed. Every figure reads canonical results and
writes the exact numbers it plotted to figures/main/data/FigNN.csv.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (PKG, R002, R003, R0031, R004, R0041, read_csv,  # noqa: E402
                    read_json, write_csv)

FIG = PKG / "figures" / "main"
DATA = FIG / "data"
FIG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

NUM = read_json(PKG / "report_numbers.json")["numbers"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": ":",
    "axes.spines.top": False, "axes.spines.right": False,
})

# print-safe: distinguished by marker + line style + hatch, not by colour alone
STYLE = {
    "anycalib_gen": dict(color="#2b2b2b", marker="o", ls="-", mfc="white"),
    "geocalib_distorted": dict(color="#6a6a6a", marker="s", ls="--", mfc="white"),
    "E2_frozen": dict(color="#111111", marker="^", ls="-.", mfc="#cccccc"),
}
LABEL = {"anycalib_gen": "AnyCalib-gen (radial)",
         "geocalib_distorted": "GeoCalib-distorted (radial)",
         "E2_frozen": "Frozen E2 ensemble (exploratory)"}


def v(k):
    return NUM[k]["value"]


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png / .pdf")


# ------------------------------------------------------------------- Fig 01
def fig01():
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(-7, 40)
    ax.axis("off")
    boxes = [("RGB video\n(monocular)", 2), ("hand detection\n+ crop", 21),
             ("hand pose model\nrelative 3D shape", 40),
             ("camera-dependent\ntranslation", 61),
             ("absolute 3D\nin camera space", 82)]
    for i, (txt, x) in enumerate(boxes):
        hl = i == 3
        ax.add_patch(FancyBboxPatch((x, 20), 16, 11,
                                    boxstyle="round,pad=0.6",
                                    linewidth=2.4 if hl else 1.2,
                                    edgecolor="black",
                                    facecolor="#e2e2e2" if hl else "white",
                                    zorder=2))
        ax.text(x + 8, 25.5, txt, ha="center", va="center", fontsize=10,
                fontweight="bold" if hl else "normal", zorder=3)
        if i < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((x + 16.4, 25.5), (boxes[i + 1][1] - 0.4, 25.5),
                                         arrowstyle="-|>", mutation_scale=16,
                                         linewidth=1.3, color="black"))
    ax.annotate("this step needs the camera's focal length",
                xy=(69, 19.6), xytext=(69, 12.5), ha="center", fontsize=10.5,
                arrowprops=dict(arrowstyle="-|>", lw=1.3, color="black"))
    f_virtual = v("cam002_pipeline_virtual_focal_px")
    f_phys = v("cam002_gt_effective_focal_median_px")
    ax.text(50, 7.2,
            r"$t_z = 2f\,/\,(s\,B)$" + "   -  the depth the hand is placed at is "
            "proportional to the focal length used",
            ha="center", fontsize=11)
    ax.text(50, 2.6,
            f"pipeline focal convention:  f = {f_virtual:,.0f} px          "
            f"dataset-provided reference focal (median):  "
            f"f $\\approx$ {f_phys:,.0f} px",
            ha="center", fontsize=10.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor="black", linewidth=1.2))
    ax.text(50, -3.2,
            "left: a training-convention virtual focal, FOCAL_LENGTH/IMAGE_SIZE "
            "x max(W,H), never read from any calibration.\n"
            "right: the dataset-provided camera intrinsic fx, verified to be in "
            "the same original full-image pixel convention.",
            ha="center", fontsize=8.6, style="italic")
    ax.set_title("Where the camera enters the existing hand pipeline", pad=6)
    save(fig, "Fig01_existing_pipeline_and_camera_role")
    write_csv(DATA / "Fig01.csv", [
        {"quantity": "pipeline virtual focal convention "
                     "(PIPELINE_BASELINE_FOCAL = FOCAL_LENGTH/IMAGE_SIZE "
                     "* max(W,H))", "value_px": f_virtual,
         "source": NUM["cam002_pipeline_virtual_focal_px"]["source_file"]},
        {"quantity": "dataset-provided reference focal, median "
                     "(GT_EFFECTIVE_FOCAL = GT_NATIVE_FX)",
         "value_px": f_phys,
         "source": NUM["cam002_gt_effective_focal_median_px"]["source_file"]}])


# ------------------------------------------------------------------- Fig 02
def fig02():
    steps = [
        ("Experiment 1\nData & convention validation",
         "QC established; released 2D vs provided 3D\n"
         f"self-consistent to {v('cam0013_recon_vs_provided3d_same_hand_median_mm')} mm "
         "(median)"),
        ("Experiment 2\nFocal sensitivity of absolute depth",
         f"a 5 % focal error moves the hand by about\n"
         f"{v('cam002_focal_5pct_root_shift_median_mm'):.0f} mm in this "
         "working-distance regime"),
        ("Experiment 3\nExisting calibration baselines",
         "distortion-aware modelling roughly halves the\n"
         f"error ({v('cam003_anycalib_pinhole_focal_err_median')} % -> "
         f"{v('cam0031_anycalib_gen_radial_focal_err_median')} %), "
         "but ~10 % remains"),
        ("Experiment 4\nMulti-frame use & bias analysis",
         f"{v('cam004_anycalib_gen_bias_fraction_pct'):.0f} % of the residual is a "
         "stable between-view\ncomponent; 8-64 frames do not remove it"),
        ("Current bottleneck",
         "a stable per-(sequence, camera) view bias,\n"
         "not frame count and not frame choice"),
        ("Next: CAM-EXP-005",
         "why does each view have its own bias?\n(scene / background / viewpoint cues)"),
        ("Later",
         "CAM-EXP-006 hand cues, then a single\nsealed external confirmation"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 11.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(steps) * 1.55)
    ax.axis("off")
    for i, (title, finding) in enumerate(steps):
        y = (len(steps) - i - 1) * 1.55 + 0.25
        last = i >= 5
        ax.add_patch(FancyBboxPatch((0.4, y), 9.2, 1.15,
                                    boxstyle="round,pad=0.12",
                                    linewidth=1.8 if i == 4 else 1.1,
                                    edgecolor="black",
                                    facecolor="#ececec" if i == 4 else "white",
                                    linestyle="--" if last else "-"))
        ax.text(0.75, y + 0.83, title, fontsize=10.6, fontweight="bold", va="center")
        ax.text(0.75, y + 0.36, finding, fontsize=9.6, va="center")
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((5, y - 0.02), (5, y - 0.38),
                                         arrowstyle="-|>", mutation_scale=14,
                                         linewidth=1.2, color="black"))
    ax.text(5, len(steps) * 1.55 - 0.12,
            "Research flow and what each step actually established",
            ha="center", fontsize=12.5, fontweight="bold")
    ax.text(5, 0.02,
            "dashed = planned, not yet performed.  The +/-5 % focal target is a "
            "working target derived from Experiment 2\nin this "
            "working-distance regime, not a universal requirement.",
            ha="center", fontsize=8.8, style="italic")
    save(fig, "Fig02_research_flow_with_findings")


# ------------------------------------------------------------------- Fig 03
def fig03():
    qc_total = v("gigahands_qc_total_observations")
    fig, ax = plt.subplots(figsize=(12.6, 11.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(-16, 100)
    ax.axis("off")

    def box(x, y, w, h, txt, bold=False, fc="white", hatch=None, ls="-",
            fs=9.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5",
                                    linewidth=1.8 if bold else 1.1,
                                    edgecolor="black", facecolor=fc,
                                    hatch=hatch, linestyle=ls, zorder=2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal", zorder=3)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=14, lw=1.2, color="black"))

    box(30, 88, 40, 8, "GigaHands released data\n"
                       "(RGB video, per-view 2D observations, provided 3D, "
                       "COLMAP cameras)", bold=True)
    arrow(50, 87.6, 50, 82.5)
    box(30, 74, 40, 8, "camera / coordinate convention validation\n"
                       "(CAM-EXP-001)")
    arrow(50, 73.6, 50, 68.5)
    box(14, 56, 72, 12,
        "diagnosis of PART of the error tail (CAM-EXP-001.1 / 001.2)\n"
        f"all-zero 2D pattern: {v('gigahands_qc_zero_pattern_n'):,} observations   |   "
        f"left/right identity swaps within a view: "
        f"{v('cam0013_identity_confirmed_2d_hand_identity_swap_n'):,}\n"
        "RGB segment matching   |   chosen-frame index audit against official "
        "source\n"
        f"NOT explained: {v('cam0013_identity_bad_2d_geometry_n'):,} "
        "BAD_2D_GEOMETRY, "
        f"{v('cam0013_identity_unresolved_insufficient_geometry_n'):,} "
        "UNRESOLVED_INSUFFICIENT_GEOMETRY")
    arrow(50, 55.6, 50, 51.5)
    box(22, 40, 56, 11,
        "independent multi-view reconstruction + leave-one-camera-out\n"
        "(CAM-EXP-001.3; the provided 3D is never an input)\n"
        f"agreement with provided 3D: "
        f"{v('cam0013_recon_vs_provided3d_same_hand_median_mm')} mm median  |  "
        f"wrong-hand control: "
        f"{v('cam0013_recon_vs_provided3d_other_hand_median_mm'):.0f} mm")
    arrow(50, 39.6, 50, 34.5)
    box(24, 26, 52, 8,
        f"QC manifest - {qc_total:,} hand observations labelled\n"
        f"PASS_STRICT {v('gigahands_qc_pass_strict_n'):,}  |  "
        f"PASS_SINGLE_HAND {v('gigahands_qc_pass_single_hand_n'):,}  |  "
        f"REVIEW {v('gigahands_qc_review_n'):,}  |  "
        f"EXCLUDE {v('gigahands_qc_exclude_n'):,}", bold=True)
    arrow(40, 25.6, 25, 18.6)
    arrow(60, 25.6, 75, 18.6)
    box(2, 1, 46, 17,
        "A. CAMERA-CLEAN subset\n(needs RGB + valid camera parameters)\n\n"
        f"{v('camera_benchmark_views_usable')} of 200 views, "
        f"{v('gigahands_unique_physical_cameras')} physical cameras\n"
        f"{v('camera_benchmark_views_excluded')} excluded: no usable RGB segment\n"
        "hand annotation quality is NOT a filter here\n\n"
        "-> Experiments 3 and 4", bold=True, fc="#ededed", fs=8.8)
    box(52, 1, 46, 17,
        "B. HAND-CLEAN / BIMANUAL-CLEAN ELIGIBLE POOL\n"
        "(needs valid hand annotation;\nbimanual needs BOTH hands)\n\n"
        f"{v('gigahands_bimanual_clean_frames'):,} frames over "
        f"{v('gigahands_bimanual_clean_views')} views\n\n"
        "-> the pool Experiment 2 draws from,\nand later hand-aware work",
        bold=True, fc="white", ls="--", fs=8.8)
    arrow(75, 0.6, 75, -3.0)
    box(48, -14, 50, 11,
        "stratified sample ACTUALLY EVALUATED in CAM-EXP-002\n\n"
        f"{v('cam002_frames_attempted'):,} frames drawn  ->  "
        f"{v('cam002_frames_evaluated'):,} evaluated  ->  "
        f"{v('cam002_hands_evaluated'):,} hands\n"
        f"{v('cam002_views_evaluated')} views, "
        f"{v('cam002_unique_physical_cameras_evaluated')} physical cameras, "
        f"{v('cam002_sequences_evaluated')} sequences\n"
        f"(dropped: {v('cam002_model_failures')} no-detection, "
        f"{v('cam002_hand_association_failures')} ambiguous association)")
    ax.set_title("GigaHands validation and QC flow, and why the experiments use "
                 "different subsets\n"
                 "(an eligible pool is not the sample an experiment evaluated)",
                 fontsize=12, pad=10)
    save(fig, "Fig03_gigahands_validation_and_qc_flow")
    def role(k):
        if k.startswith("cam002_"):
            return "CAM-EXP-002 evaluated sample"
        if k.startswith("gigahands_bimanual"):
            return "eligible pool (NOT the evaluated sample)"
        if k.startswith("camera_benchmark"):
            return "camera-clean subset"
        return "QC labelling"

    write_csv(DATA / "Fig03.csv", [
        {"quantity": k, "value": v(k), "role": role(k),
         "source": NUM[k]["source_file"]}
        for k in ("gigahands_qc_total_observations", "gigahands_qc_pass_strict_n",
                  "gigahands_qc_pass_single_hand_n", "gigahands_qc_review_n",
                  "gigahands_qc_exclude_n", "gigahands_qc_zero_pattern_n",
                  "camera_benchmark_views_usable", "camera_benchmark_views_excluded",
                  "gigahands_bimanual_clean_frames", "gigahands_bimanual_clean_views",
                  "cam002_frames_attempted", "cam002_frames_evaluated",
                  "cam002_hands_evaluated", "cam002_views_evaluated",
                  "cam002_unique_physical_cameras_evaluated",
                  "cam002_sequences_evaluated",
                  "gigahands_unique_physical_cameras")])


# ------------------------------------------------------------------- Fig 04
def fig04():
    sens = read_csv(R002 / "results" / "summary" / "focal_sensitivity_summary.csv")
    rows = sorted(sens, key=lambda r: float(r["focal_error_percent"]))
    x = np.array([float(r["focal_error_percent"]) for r in rows])
    y = np.array([float(r["incremental_signed_dz_median_mm"]) for r in rows])
    p90 = np.array([float(r["incremental_root_shift_p90_mm"]) for r in rows])

    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    ax.axhline(0, color="black", lw=0.9)
    ax.axvline(0, color="black", lw=0.9)
    ax.plot(x, y, "o-", color="black", mfc="white", ms=7, lw=1.8,
            label="median signed depth displacement")
    ax.plot(x, np.sign(x) * p90, "s--", color="#6a6a6a", mfc="white", ms=6,
            lw=1.3, label="p90 magnitude (signed for readability)")
    for xi, yi in zip(x, y):
        if xi != 0:
            ax.annotate(f"{yi:+.0f} mm", (xi, yi), textcoords="offset points",
                        xytext=(0, -16 if xi > 0 else 8), ha="center", fontsize=9)
    for t in (-5, 5):
        ax.axvline(t, color="#999999", ls=":", lw=1.1)
    ax.set_xlabel("focal-length perturbation (%)")
    ax.set_ylabel("absolute depth displacement of the hand (mm)")
    ax.set_title("Focal-length sensitivity of absolute hand depth\n"
                 f"evaluated sample: {v('cam002_hands_evaluated'):,} hands / "
                 f"{v('cam002_frames_evaluated'):,} frames / "
                 f"{v('cam002_views_evaluated')} views, drawn from the "
                 f"{v('gigahands_bimanual_clean_frames'):,}-frame\n"
                 "bimanual-clean eligible pool", fontsize=10.5)
    ax.legend(loc="upper left", frameon=False)
    ax.text(0.98, 0.04,
            "the near-linear relation is expected from the pipeline equation "
            r"$Z \propto f$",
            transform=ax.transAxes, ha="right", fontsize=9, style="italic")
    save(fig, "Fig04_focal_error_vs_absolute_depth_shift")
    write_csv(DATA / "Fig04.csv", [
        {"focal_error_percent": r["focal_error_percent"],
         "median_signed_dz_mm": r["incremental_signed_dz_median_mm"],
         "median_root_shift_mm": r["incremental_root_shift_median_mm"],
         "p90_root_shift_mm": r["incremental_root_shift_p90_mm"],
         "n_hands": r["n_hands"],
         "evaluated_sample_note":
             f"{v('cam002_hands_evaluated')} hands / "
             f"{v('cam002_frames_evaluated')} frames / "
             f"{v('cam002_views_evaluated')} views, sampled from a "
             f"{v('gigahands_bimanual_clean_frames')}-frame eligible pool",
         "source": "experiments/runs/CAM-EXP-002_camera_focal_sensitivity/"
                   "results/summary/focal_sensitivity_summary.csv"} for r in rows])


# ------------------------------------------------------------------- Fig 05
def fig05():
    ms3 = read_csv(R003 / "results" / "summary" / "model_summary.csv")
    ms31 = read_csv(R0031 / "results" / "summary" / "model_summary.csv")

    def g3(model, col):
        return float(next(r for r in ms3 if r["model"] == model)[col])

    def g31(run, col):
        return float(next(r for r in ms31 if r["model_run"] == run)[col])

    entries = [
        ("AnyCalib\npinhole", g3("AnyCalib[anycalib_pinhole/pinhole]",
                                 "focal_error_pct_median"),
         g3("AnyCalib[anycalib_pinhole/pinhole]", "within_5pct_rate") * 100,
         g3("AnyCalib[anycalib_pinhole/pinhole]", "within_10pct_rate") * 100,
         g3("AnyCalib[anycalib_pinhole/pinhole]", "within_20pct_rate") * 100, False),
        ("AnyCalib\ndistortion-aware", g31("anycalib_gen_radial", "focal_error_pct_median"),
         g31("anycalib_gen_radial", "within_5pct_rate") * 100,
         g31("anycalib_gen_radial", "within_10pct_rate") * 100,
         g31("anycalib_gen_radial", "within_20pct_rate") * 100, True),
        ("GeoCalib\npinhole", g3("GeoCalib[pinhole]", "focal_error_pct_median"),
         g3("GeoCalib[pinhole]", "within_5pct_rate") * 100,
         g3("GeoCalib[pinhole]", "within_10pct_rate") * 100,
         g3("GeoCalib[pinhole]", "within_20pct_rate") * 100, False),
        ("GeoCalib\ndistortion-aware",
         g31("geocalib_distorted_radial", "focal_error_pct_median"),
         g31("geocalib_distorted_radial", "within_5pct_rate") * 100,
         g31("geocalib_distorted_radial", "within_10pct_rate") * 100,
         g31("geocalib_distorted_radial", "within_20pct_rate") * 100, True),
        ("PerspectiveFields\nuncentered (no distortion variant)",
         g3("PerspectiveFields[uncentered]", "focal_error_pct_median"),
         g3("PerspectiveFields[uncentered]", "within_5pct_rate") * 100,
         g3("PerspectiveFields[uncentered]", "within_10pct_rate") * 100,
         g3("PerspectiveFields[uncentered]", "within_20pct_rate") * 100, False),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8),
                             gridspec_kw={"width_ratios": [1.05, 1]})
    names = [e[0] for e in entries]
    med = [e[1] for e in entries]
    xpos = np.arange(len(entries))
    ax = axes[0]
    bars = ax.bar(xpos, med, 0.62, color="white", edgecolor="black", linewidth=1.4,
                  hatch=["" if not e[5] else "///" for e in entries])
    ax.axhline(5, color="black", ls=":", lw=1.2)
    ax.text(len(entries) - 0.4, 5.6, "5 % target", ha="right", fontsize=9)
    for b, m in zip(bars, med):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.5, f"{m:.1f}",
                ha="center", fontsize=10)
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, fontsize=8.6)
    ax.set_ylabel("median relative focal error (%)")
    ax.set_title("Single-frame focal error\n(hatched = distortion-aware)",
                 fontsize=11.5)

    ax = axes[1]
    w = 0.26
    for i, (lab, idx, hatch) in enumerate((("within 5 %", 2, ""),
                                           ("within 10 %", 3, "..."),
                                           ("within 20 %", 4, "xxx"))):
        ax.bar(xpos + (i - 1) * w, [e[idx] for e in entries], w, color="white",
               edgecolor="black", linewidth=1.2, hatch=hatch, label=lab)
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, fontsize=8.6)
    ax.set_ylabel("share of evaluated cases (%)")
    ax.set_title("Accuracy within tolerance", fontsize=11.5)
    ax.legend(frameon=False)

    fig.suptitle("Existing single-frame calibration methods, with and without a "
                 "distortion-aware camera model", fontsize=12.5, y=1.02)
    fig.text(0.5, -0.035,
             "175 fixed-camera views, 8 frames per view, 1400 frames in total. "
             "Statistical unit = view; the 1400 frames are repeated observations, "
             "not independent samples.\n"
             "Paired view-clustered gain from the distortion-aware model: "
             f"AnyCalib {v('cam0031_anycalib_gen_radial_improvement_pp_view')[0]:+.2f} pp "
             f"[{v('cam0031_anycalib_gen_radial_improvement_pp_view')[1]:+.2f}, "
             f"{v('cam0031_anycalib_gen_radial_improvement_pp_view')[2]:+.2f}], "
             f"GeoCalib {v('cam0031_geocalib_distorted_radial_improvement_pp_view')[0]:+.2f} pp "
             f"[{v('cam0031_geocalib_distorted_radial_improvement_pp_view')[1]:+.2f}, "
             f"{v('cam0031_geocalib_distorted_radial_improvement_pp_view')[2]:+.2f}].",
             ha="center", fontsize=9)
    save(fig, "Fig05_single_frame_calibration_and_distortion")
    write_csv(DATA / "Fig05.csv", [
        {"model": e[0].replace("\n", " "), "median_rel_focal_err_pct": round(e[1], 3),
         "within_5_pct": round(e[2], 2), "within_10_pct": round(e[3], 2),
         "within_20_pct": round(e[4], 2),
         "distortion_aware": int(e[5]),
         "source": "CAM-EXP-003 / CAM-EXP-003.1 model_summary.csv"} for e in entries])


# ------------------------------------------------------------------- Fig 06
def fig06():
    rows = read_csv(DATA / "Fig06.csv")
    orig = read_csv(DATA / "Fig06_cam004_original.csv")
    NS = [1, 2, 4, 8, 16, 32, 64]
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    for est in ("anycalib_gen", "geocalib_distorted", "E2_frozen"):
        y = [float(next(r for r in rows if r["estimator"] == est
                        and int(r["N"]) == n)["median_rel_err_pct"]) for n in NS]
        ax.plot(NS, y, ms=7, lw=1.9, label=LABEL[est], **STYLE[est])
    for est in ("geocalib_distorted", "E2_frozen"):
        y = [float(next(r for r in orig if r["estimator"] == est
                        and int(r["N"]) == n)["median_rel_err_pct"])
             for n in (1, 2, 4, 8)]
        ax.plot([1, 2, 4, 8], y, marker="x", ls="None", ms=7, color="#9a9a9a")
    ax.plot([], [], marker="x", ls="None", color="#9a9a9a",
            label="CAM-EXP-004 original run (shown, not joined)")
    ax.axhline(5, color="black", ls=":", lw=1.2)
    ax.text(64, 5.35, "5 % target", ha="right", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(NS)
    ax.set_xticklabels([str(n) for n in NS])
    ax.set_ylim(0, 16)
    ax.set_xlabel("frames aggregated per static camera (N)")
    ax.set_ylabel("median relative focal error (%)")
    ax.set_title("Aggregating more frames of one static camera", fontsize=12)
    ax.legend(frameon=False, loc="upper right")
    fig.text(0.5, -0.06,
             "175 fixed-camera views; statistical unit = view. Every point is "
             "recomputed from ONE inference run (CAM-EXP-004.1), so no step in "
             "the curve is a run boundary.\nFor N <= 8 all C(8,N) frame subsets "
             "are evaluated and averaged per view. The crosses show the "
             "CAM-EXP-004 original run at the same N for comparison; they differ "
             "by at most 0.52 pp,\nwhich is the GeoCalib run-to-run variation, "
             "and are deliberately not joined into the curve.",
             ha="center", fontsize=8.8)
    save(fig, "Fig06_multiframe_1_to_64")


# ------------------------------------------------------------------- Fig 07
def fig07():
    bn = read_csv(R004 / "results" / "summary" / "bias_noise_summary.csv")
    fc = read_csv(DATA / "Fig06.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2))

    ax = axes[0]
    names = [r["model"] for r in bn]
    bias = [float(r["bias_fraction_pct"]) for r in bn]
    within = [100 - b for b in bias]
    y = np.arange(len(names))
    ax.barh(y, bias, color="white", edgecolor="black", linewidth=1.3, hatch="///",
            label="stable per-view bias")
    ax.barh(y, within, left=bias, color="#d9d9d9", edgecolor="black",
            linewidth=1.3, label="frame-to-frame noise")
    for i, b in enumerate(bias):
        ax.text(b / 2, i, f"{b:.0f} %", ha="center", va="center", fontsize=9.5)
    ax.set_yticks(y)
    ax.set_yticklabels(["AnyCalib-gen", "GeoCalib-distorted", "PF-uncentered"],
                       fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("share of total squared log focal error (%)")
    ax.set_title("A. Where the error lives", fontsize=11.5)
    ax.legend(frameon=False, loc="lower right", fontsize=8.8)

    ax = axes[1]
    ests = ["anycalib_gen", "geocalib_distorted", "E2_frozen"]
    med = [float(next(r for r in fc if r["estimator"] == e
                      and int(r["N"]) == 8)["median_rel_err_pct"]) for e in ests]
    w5 = [float(next(r for r in fc if r["estimator"] == e
                     and int(r["N"]) == 8)["within_5_pct"]) for e in ests]
    x = np.arange(3)
    ax.bar(x, med, 0.55, color="white", edgecolor="black", linewidth=1.4,
           hatch=["", "", "///"])
    ax.axhline(5, color="black", ls=":", lw=1.2)
    for xi, m, w in zip(x, med, w5):
        ax.text(xi, m + 0.25, f"{m:.2f} %", ha="center", fontsize=10)
        ax.text(xi, 0.4, f"{w:.0f} % within 5 %", ha="center", fontsize=8.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["AnyCalib-gen", "GeoCalib-\ndistorted", "Frozen E2\n(hatched)"],
                       fontsize=9.3)
    ax.set_ylabel("median relative focal error (%)")
    ax.set_title("B. Error at N = 8 frames", fontsize=11.5)

    ax = axes[2]
    sgn = []
    for e, key in ((ests[0], "cam0031_anycalib_gen_radial_focal_err_signed_median"),
                   (ests[1], "cam0031_geocalib_distorted_radial_focal_err_signed_median")):
        sgn.append(v(key))
    sgn.append(-2.45)   # E2, from the CAM-EXP-004.1 re-run
    ax.barh(np.arange(3), sgn, 0.55, color="white", edgecolor="black",
            linewidth=1.4, hatch=["", "", "///"])
    ax.axvline(0, color="black", lw=1.1)
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(["AnyCalib-gen", "GeoCalib-distorted", "Frozen E2"],
                       fontsize=9.5)
    ax.invert_yaxis()
    for i, s in enumerate(sgn):
        ax.text(s + (0.5 if s > 0 else -0.5), i, f"{s:+.2f} %",
                va="center", ha="left" if s > 0 else "right", fontsize=9.5)
    ax.set_xlabel("signed median focal error (%)")
    ax.set_title("C. The two families err in opposite directions", fontsize=11.5)

    fig.suptitle("Why more frames cannot help, and what can: the residual is bias, "
                 "and two model families have opposite bias", fontsize=12.5, y=1.03)
    fig.text(0.5, -0.05,
             "Frozen E2 is an exploratory ensemble selected among five candidates "
             "on the same 175 views it is reported on; it is not a final proposed "
             "method.\nIts performance is quoted as "
             f"{v('e2_or_model_rerun_E2_frozen_cam004_median')} % (selection set), "
             f"{v('e2_or_model_rerun_E2_frozen_cam0041_median')} % (independent "
             f"re-run) and {v('e2_loso_heldout_median_range_pct')[0]}-"
             f"{v('e2_loso_heldout_median_range_pct')[1]} % "
             "(leave-one-sequence-out).", ha="center", fontsize=8.8)
    save(fig, "Fig07_bias_noise_and_ensemble_summary")
    write_csv(DATA / "Fig07.csv", [
        {"panel": "A", "model": r["model"],
         "bias_fraction_pct": r["bias_fraction_pct"],
         "within_fraction_pct": r["within_fraction_pct"],
         "median_abs_view_bias_pct": r["median_abs_view_bias_pct"],
         "median_within_view_std_pct": r["median_within_view_std_pct"],
         "source": "CAM-EXP-004/results/summary/bias_noise_summary.csv"}
        for r in bn] + [
        {"panel": "B/C", "model": e, "median_rel_err_pct_N8": m,
         "within_5_pct_N8": w, "signed_median_pct": s,
         "source": "figures/main/data/Fig06.csv (a single execution over the "
                   "same 175 benchmark views) and CAM-EXP-003.1 "
                   "model_summary.csv for the signed values"}
        for e, m, w, s in zip(ests, med, w5, sgn)])


def main() -> None:
    print("writing report figures:")
    fig01()
    fig02()
    fig03()
    fig04()
    fig05()
    fig06()
    fig07()


if __name__ == "__main__":
    main()
