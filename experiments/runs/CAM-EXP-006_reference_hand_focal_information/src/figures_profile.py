"""Fig 06 / 07 - the shape of the objective curve, which is the core evidence."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import focal_profile_solver as S  # noqa: E402
from common import FIG, R005, SUM, fnum, read_csv, read_json  # noqa: E402

MAN = Path("D:/hand-demo/experiments/manifests/"
           "cam_exp_006_reference_hand_frames_v1.csv.gz")


def main() -> None:
    z = np.load("cache/reference_3d_v1.npz")
    tgt = {(r["sequence"], r["camera"]): r
           for r in read_csv(R005 / "results" / "raw" / "view_targets.csv.gz")}
    grid = S.gamma_grid()

    curves, gts = [], []
    seen = set()
    for r in read_csv(MAN):
        key = (r["sequence"], r["camera"])
        if key in seen or int(r["grid_pos"]) != 0 or len(curves) >= 60:
            continue
        for hand in ("left", "right"):
            k = f"{r['sequence']}|{r['camera']}|{r['frame']}|{hand}"
            if k + "|use" not in z:
                continue
            use = z[k + "|use"]
            if use.sum() < 12:
                continue
            c = S.profile_one(z[k + "|xyz"][use].astype(float),
                              z[k + "|uv"][use].astype(float), 1280, 720, grid)
            if np.isfinite(c).any():
                curves.append(c)
                gts.append(fnum(tgt[key]["gt_reference_focal_px"]) / 1280.0)
                seen.add(key)
            break

    # Fig 06 - the objective curves themselves
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for c in curves:
        ax.plot(grid, c, color="#1f4e79", alpha=0.18, lw=1)
    med = np.nanmedian(np.array(curves), axis=0)
    ax.plot(grid, med, color="#1f4e79", lw=2.6, label="median over views")
    ax.axvline(float(np.median(gts)), color="k", ls="--", lw=1.4,
               label="true focal of the rig")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"candidate focal  $\gamma = f\,/\,\max(W,H)$")
    ax.set_ylabel("median reprojection error (px)")
    ax.set_title("Fig 06 - the objective barely changes with the focal")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.text(0.01, -0.05, "a focal seven times too large fits the same hand "
             "about as well as the true one: the weak-perspective degeneracy",
             fontsize=7, color="#555")
    fig.savefig(FIG / "fig06_objective_profiles.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  fig06_objective_profiles.png")

    # Fig 07 - sanity check: flat, not broken
    s = read_json(SUM / "solver_sanity_check.json")
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    keys = [("median_objective_at_profile_minimum_px", "at the profile minimum"),
            ("median_objective_at_true_focal_px", "at the TRUE focal"),
            ("median_objective_at_gamma_5p0_px", r"at $\gamma=5.0$ (≈7x too large)"),
            ("median_objective_at_gamma_0p25_px", r"at $\gamma=0.25$ (too small)")]
    vals = [s[k] for k, _ in keys]
    cols = ["#1f4e79", "#2e7d32", "#c0504d", "#e0a030"]
    ax.barh([lbl for _, lbl in keys], vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.2f} px", va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("median reprojection error (px)")
    ax.set_title("Fig 07 - the solver is correct; the profile is flat")
    ax.grid(axis="x", alpha=0.3)
    fig.text(0.01, -0.07, "the true focal sits only ~11 % above the minimum, "
             "so the solver works - but a far larger focal fits slightly "
             "BETTER, so the focal is not identifiable", fontsize=7,
             color="#555")
    fig.savefig(FIG / "fig07_solver_sanity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig07_solver_sanity.png")


if __name__ == "__main__":
    main()
