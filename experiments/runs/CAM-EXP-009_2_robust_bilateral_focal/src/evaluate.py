"""Apply the frozen gate to the TEST and control results.

Every threshold comes from GATE in c92_common.py, which was written before the
DEV run. Nothing here chooses a threshold, a cell or a comparison after seeing
a number.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c92_common import (GATE, MANIFESTS, RAW, SUM, TAB,  # noqa: E402
                        cluster_bootstrap, fnum, read_csv, read_json,
                        write_csv, write_json)

CLEAN = "TEST_CLEAN"
MODERATE = "TEST_COMBINED_MODERATE"


def med(rows, key="err_pct_eval"):
    v = [fnum(r[key]) for r in rows]
    v = [x for x in v if np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def within(rows, key="err_pct_eval", thr=5.0):
    v = [fnum(r[key]) for r in rows]
    v = [x for x in v if np.isfinite(x)]
    return 100.0 * float(np.mean([x <= thr for x in v])) if v else float("nan")


def main():
    sel = read_json(MANIFESTS / "cam_exp_0092_selected_method_v1.json")
    win, base = sel["SELECTED_METHOD"], sel["baseline_for_comparison"]
    rows = read_csv(RAW / "test_trials.csv.gz")
    ctrl = read_csv(RAW / "control_trials.csv.gz")
    cells = sorted(set(r["cell"] for r in rows))

    # ---- per-cell table
    table = []
    for c in cells:
        rec = {"cell": c}
        for m, tag in ((win, "winner"), (base, "M0")):
            sub = [r for r in rows if r["cell"] == c and r["method"] == m]
            rec["%s_median_err_pct_eval" % tag] = med(sub)
            rec["%s_median_err_pct_fit" % tag] = med(sub, "err_pct_fit")
            rec["%s_within5_pct" % tag] = within(sub)
            rec["%s_boundary_rate" % tag] = float(np.mean(
                [fnum(r["boundary_eval"]) for r in sub])) if sub else np.nan
            rec["%s_n" % tag] = len(sub)
        w, b = (rec["winner_median_err_pct_eval"], rec["M0_median_err_pct_eval"])
        rec["relative_reduction_vs_M0_pct"] = (
            100.0 * (b - w) / b if np.isfinite(w) and np.isfinite(b) and b > 0
            else float("nan"))
        table.append(rec)
    write_csv(TAB / "test_per_cell.csv", table)
    by = {r["cell"]: r for r in table}

    # ---- bootstrap CIs on the two headline cells (cluster = subject)
    cis = {}
    for c in (CLEAN, MODERATE):
        for m, tag in ((win, "winner"), (base, "M0")):
            sub = [r for r in rows if r["cell"] == c and r["method"] == m]
            v = [fnum(r["err_pct_eval"]) for r in sub]
            k = [r["subject"] for r in sub]
            pt, lo, hi = cluster_bootstrap(v, k)
            cis["%s|%s" % (c, tag)] = {"median_err_pct": pt, "lo": lo, "hi": hi}

    # ---- controls
    cstat = {}
    for c in sorted(set(r["control"] for r in ctrl)):
        sub = [r for r in ctrl if r["control"] == c]
        cstat[c] = {
            "median_err_pct": med(sub, "err_pct"),
            "within5_pct": within(sub, "err_pct"),
            "boundary_rate": float(np.mean([fnum(r["boundary"])
                                            for r in sub])),
            "curve_range": float(np.nanmedian([fnum(r["curve_range"])
                                               for r in sub])),
        }
    c0 = cstat.get("C0_CORRECT", {}).get("median_err_pct", float("nan"))

    def degrades(name):
        """A control degrades if it is at least GATE pp worse than C0."""
        e = cstat.get(name, {}).get("median_err_pct", float("nan"))
        return bool(np.isfinite(e) and np.isfinite(c0)
                    and e - c0 >= GATE["control_degradation_pp"])

    # ---- the frozen gate
    clean_err = by[CLEAN]["winner_median_err_pct_eval"]
    mod_err = by[MODERATE]["winner_median_err_pct_eval"]
    mod_red = by[MODERATE]["relative_reduction_vs_M0_pct"]
    fit_eval = abs(by[MODERATE]["winner_median_err_pct_fit"] - mod_err)

    gate = {
        "g1_clean_identifiable": bool(
            np.isfinite(clean_err) and clean_err <= GATE["clean_focal_err_pct"]),
        "g2_moderate_identifiable": bool(
            np.isfinite(mod_err) and mod_err <= GATE["moderate_focal_err_pct"]),
        "g3_beats_M0_on_moderate": bool(
            np.isfinite(mod_red)
            and mod_red >= GATE["relative_reduction_vs_M0_pct"]),
        "g4_boundary_rate_ok": bool(
            by[MODERATE]["winner_boundary_rate"] <= GATE["boundary_rate"]),
        "g5_fit_eval_consistent": bool(
            np.isfinite(fit_eval) and fit_eval <= GATE["fit_eval_diff_pct"]),
        "g6_wrong_mapping_degrades": degrades("C1_WRONG_BONE_MAPPING"),
        "g7_subject_swap_degrades": degrades("C2_SUBJECT_SWAP"),
        "g8_fixed_length_flat": bool(
            cstat.get("C4_FIXED_LENGTH_INVARIANT", {}).get("curve_range", 1.0)
            == 0.0),
    }
    required = ["g2_moderate_identifiable", "g3_beats_M0_on_moderate",
                "g6_wrong_mapping_degrades", "g7_subject_swap_degrades"]
    passed = all(gate[k] for k in required)

    out = {
        "selected_method": win, "baseline": base,
        "headline": {
            "clean_median_err_pct_eval": clean_err,
            "moderate_median_err_pct_eval": mod_err,
            "moderate_M0_median_err_pct_eval":
                by[MODERATE]["M0_median_err_pct_eval"],
            "moderate_relative_reduction_vs_M0_pct": mod_red,
            "fit_vs_eval_abs_diff_pct": fit_eval,
        },
        "bootstrap_ci_2p5_97p5": cis,
        "controls": cstat,
        "control_notes": {
            "C3_TEMPORAL_SHUFFLE": "CONTROL_VACUOUS_BY_CONSTRUCTION - the "
                                   "objective never pairs a left frame with a "
                                   "right frame, so there is no temporal "
                                   "pairing to destroy; C3 collapses exactly "
                                   "onto C2. Carries no evidential weight.",
            "C4_FIXED_LENGTH_INVARIANT": "Oracle diagnostic; handed the true "
                                         "bone lengths and not runnable on "
                                         "real data.",
        },
        "gate": gate,
        "required_gates": required,
        "GATE_PASSED": passed,
        "real_phase": ("REAL_FOCAL_PHASE_ELIGIBLE" if passed
                       else "REAL_FOCAL_PHASE_NOT_RUN"),
        "thresholds": GATE,
    }
    write_json(SUM / "gate.json", out)

    for k in sorted(gate):
        print("%-32s %s%s" % (k, gate[k], "  (required)" if k in required
                              else ""))
    print("\nclean %.3f%%   moderate %.3f%% (M0 %.3f%%, reduction %.1f%%)"
          % (clean_err, mod_err, by[MODERATE]["M0_median_err_pct_eval"],
             mod_red))
    print("GATE_PASSED:", passed, "->", out["real_phase"])


if __name__ == "__main__":
    main()
