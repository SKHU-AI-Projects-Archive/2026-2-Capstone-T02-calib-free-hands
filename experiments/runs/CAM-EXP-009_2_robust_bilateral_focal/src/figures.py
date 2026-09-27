"""Figures for CAM-EXP-009.2.

Each figure is labelled with what it is evidence FOR, and none of them is
drawn only for conditions that happened to look good.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c92_common import (FIG, MANIFESTS, RAW, fnum, read_csv,  # noqa: E402
                        read_json)
from robust_bilateral_scores import METHODS  # noqa: E402


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=140)
    plt.close(fig)
    print("  ", name)


def errs(rows, method, cell, key="err_pct_eval"):
    v = [fnum(r[key]) for r in rows
         if r["method"] == method and r["cell"] == cell]
    return [x for x in v if np.isfinite(x)]


def main():
    sel = read_json(MANIFESTS / "cam_exp_0092_selected_method_v1.json")
    win, base = sel["SELECTED_METHOD"], sel["baseline_for_comparison"]
    dev = read_csv(RAW / "dev_trials.csv.gz")
    test = read_csv(RAW / "test_trials.csv.gz")
    ctrl = read_csv(RAW / "control_trials.csv.gz")
    cells = sorted(set(r["cell"] for r in test))

    # F1 - DEV selection score per method
    sc = sel["selection_score_pct"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(METHODS)), [sc[m] for m in METHODS],
           color=["#c44" if m == win else "#889" for m in METHODS])
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(METHODS, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("DEV selection score (% focal error)")
    ax.set_title("F1 DEV selection only - NOT evidence of performance\n"
                 "selected: %s" % win, fontsize=9)
    _save(fig, "F1_dev_selection.png")

    # F2 - TEST winner vs M0 per cell
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(cells))
    mw = [np.median(errs(test, win, c)) if errs(test, win, c) else np.nan
          for c in cells]
    mb = [np.median(errs(test, base, c)) if errs(test, base, c) else np.nan
          for c in cells]
    ax.bar(x - 0.2, mb, 0.4, label=base, color="#889")
    ax.bar(x + 0.2, mw, 0.4, label=win, color="#c44")
    ax.axhline(5.0, ls="--", c="k", lw=1, label="5 % gate")
    ax.set_xticks(x)
    ax.set_xticklabels(cells, rotation=70, ha="right", fontsize=6)
    ax.set_ylabel("median focal error (%), EVAL frames")
    ax.set_title("F2 TEST: every cell, both methods", fontsize=9)
    ax.legend(fontsize=7)
    _save(fig, "F2_test_per_cell.png")

    # F3 - asymmetry response
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for kind, mk in (("GLOBAL", "o"), ("DENSE", "s"), ("SPARSE", "^"),
                     ("FINGER", "d")):
        xs, ys = [], []
        for a in (1, 2, 5):
            c = "TEST_ASYM_%s_%02d" % (kind, a)
            e = errs(test, win, c)
            if e:
                xs.append(a)
                ys.append(np.median(e))
        ax.plot(xs, ys, marker=mk, label=kind)
    e0 = errs(test, win, "TEST_CLEAN")
    if e0:
        ax.axhline(np.median(e0), ls=":", c="k", lw=1, label="clean")
    ax.axhline(5.0, ls="--", c="r", lw=1)
    ax.set_xlabel("bilateral asymmetry (%)")
    ax.set_ylabel("median focal error (%)")
    ax.set_title("F3 response to subject-level bilateral asymmetry", fontsize=9)
    ax.legend(fontsize=7)
    _save(fig, "F3_asymmetry.png")

    # F4 - noise and articulation
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (pre, lv, xl) in zip(axes, (
            ("TEST_NOISE_%sPX", (0.5, 1.0, 2.0), "2D noise (px)"),
            ("TEST_ARTIC_%sDEG", (0.5, 1.0, 2.0), "articulation noise (deg)"))):
        xs, ys, yb = [], [], []
        for v in lv:
            c = pre % v
            if errs(test, win, c):
                xs.append(v)
                ys.append(np.median(errs(test, win, c)))
                yb.append(np.median(errs(test, base, c)))
        ax.plot(xs, ys, "o-", c="#c44", label=win)
        ax.plot(xs, yb, "s--", c="#889", label=base)
        ax.axhline(5.0, ls="--", c="k", lw=1)
        ax.set_xlabel(xl)
        ax.set_ylabel("median focal error (%)")
        ax.legend(fontsize=7)
    fig.suptitle("F4 measurement-noise response", fontsize=9)
    _save(fig, "F4_noise.png")

    # F5 - controls
    cs = sorted(set(r["control"] for r in ctrl))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = [[fnum(r["err_pct"]) for r in ctrl if r["control"] == c]
            for c in cs]
    ax.boxplot(data, tick_labels=cs)
    ax.set_xticklabels(cs, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("focal error (%)")
    ax.set_title("F5 negative controls (COMBINED_MODERATE)\n"
                 "C3 is vacuous by construction; C4 is an oracle diagnostic",
                 fontsize=9)
    _save(fig, "F5_controls.png")

    # F6 - C4 invariance
    fig, ax = plt.subplots(figsize=(7, 4))
    rr = [fnum(r["curve_range"]) for r in ctrl
          if r["control"] == "C4_FIXED_LENGTH_INVARIANT"]
    r0 = [fnum(r["curve_range"]) for r in ctrl if r["control"] == "C0_CORRECT"]
    ax.bar([0, 1], [np.median(r0), np.median(rr)],
           color=["#c44", "#889"], tick_label=["C0 correct", "C4 fixed-length"])
    ax.set_ylabel("score-curve range over the focal grid")
    ax.set_title("F6 the naive fixed-length formulation is exactly flat "
                 "(range 0)", fontsize=9)
    _save(fig, "F6_invariance.png")

    # F7 - FIT vs EVAL agreement
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    a = [fnum(r["err_pct_fit"]) for r in test if r["method"] == win]
    b = [fnum(r["err_pct_eval"]) for r in test if r["method"] == win]
    ax.scatter(a, b, s=7, alpha=0.4)
    lim = max([x for x in a + b if np.isfinite(x)] + [1])
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlabel("focal error, FIT frames (%)")
    ax.set_ylabel("focal error, EVAL frames (%)")
    ax.set_title("F7 FIT vs EVAL - a shared bias is not exposed by\n"
                 "held-out frames of the same hand", fontsize=9)
    _save(fig, "F7_fit_vs_eval.png")

    # F8 - boundary rate per cell
    fig, ax = plt.subplots(figsize=(11, 4))
    br = [np.mean([fnum(r["boundary_eval"]) for r in test
                   if r["method"] == win and r["cell"] == c]) for c in cells]
    ax.bar(np.arange(len(cells)), br, color="#47a")
    ax.axhline(0.10, ls="--", c="r", lw=1, label="0.10 gate")
    ax.set_xticks(np.arange(len(cells)))
    ax.set_xticklabels(cells, rotation=70, ha="right", fontsize=6)
    ax.set_ylabel("fraction of trials at a grid boundary")
    ax.set_title("F8 boundary rate - a boundary minimum is a failure, "
                 "not an estimate", fontsize=9)
    ax.legend(fontsize=7)
    _save(fig, "F8_boundary.png")

    # F9 - error distribution on the two headline cells
    fig, ax = plt.subplots(figsize=(7, 4.5))
    dat, lab = [], []
    for c in ("TEST_CLEAN", "TEST_COMBINED_MODERATE"):
        for m, t in ((base, "M0"), (win, "winner")):
            dat.append(errs(test, m, c))
            lab.append("%s\n%s" % (c.replace("TEST_", ""), t))
    ax.boxplot(dat, tick_labels=lab)
    ax.axhline(5.0, ls="--", c="r", lw=1)
    ax.set_ylabel("focal error (%)")
    ax.set_title("F9 headline cells", fontsize=9)
    ax.tick_params(labelsize=7)
    _save(fig, "F9_headline.png")

    # F10 - visibility and missingness (secondary)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for pre, lv, lb in (("TEST_VIS_%d", (75, 50, 25), "per-side visibility %"),
                        ("TEST_MISSING_%d", (10, 20), "joint missingness %")):
        xs, ys = [], []
        for v in lv:
            c = pre % v
            if errs(test, win, c):
                xs.append(v)
                ys.append(np.median(errs(test, win, c)))
        ax.plot(xs, ys, "o-", label=lb)
    ax.axhline(5.0, ls="--", c="r", lw=1)
    ax.set_ylabel("median focal error (%)")
    ax.set_xlabel("level (%)")
    ax.set_title("F10 observability stress (secondary cells, 8 subjects)",
                 fontsize=9)
    ax.legend(fontsize=7)
    _save(fig, "F10_observability.png")

    # F11 - DEV vs TEST for the winner, the selection-optimism check
    fig, ax = plt.subplots(figsize=(6, 4))
    d = [fnum(r["err_pct_eval"]) for r in dev if r["method"] == win]
    t = [fnum(r["err_pct_eval"]) for r in test if r["method"] == win]
    ax.boxplot([[x for x in d if np.isfinite(x)],
                [x for x in t if np.isfinite(x)]],
               tick_labels=["DEV (selected on)", "TEST (disjoint)"])
    ax.set_ylabel("focal error (%)")
    ax.set_title("F11 selection optimism check", fontsize=9)
    _save(fig, "F11_dev_vs_test.png")

    print("MAIN_EXPLANATION: F2 and F5 together - F2 shows where the estimate "
          "stands against the gate across every stress cell, and F5 shows "
          "whether the bilateral correspondence is what produces it.")


if __name__ == "__main__":
    main()
