"""Freeze the CAM-EXP-009.2 candidate family, stress grid and seeds.

Ordering enforced by the spec: (A) robust candidate family, (B) synthetic
generator / stress grid, (C) DEV seeds, (D) TEST seeds - all written before
any DEV result is produced, and none of them revisited afterwards.

Trial counts were fixed from a MEASURED runtime of 10.9 s per trial (12 frames,
161-point grid, both sides, FIT and EVAL profiles, warm-started), not from any
observed performance.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c92_common import (  # noqa: E402
    COMBINED_MODERATE, COMBINED_STRONG, DEV_SUBJECTS, GATE, MANIFESTS,
    MIN_FIT_FRAMES, N_FRAMES, Q_MAX, Q_MIN, Q_N, RUN_DIR, SUM,
    TEST_SUBJECTS_PRIMARY, TEST_SUBJECTS_SECONDARY, sha256, stable_seed,
    write_csv, write_json)
from generate_stress_synthetic import BASE_LENGTHS, SPARSE_BONES  # noqa: E402
from robust_bilateral_scores import (  # noqa: E402
    DEFINITIONS, METHODS, audit_r_LR_redundancy)

SEC_PER_TRIAL = 10.9          # measured, see module docstring

BASE = dict(dist_over_diam=4, asym_kind="dense", asym=0.0, noise_px=0.0,
            artic_deg=0.0, missing=0.0, visibility=1.0)


def cell(name, tier, **kw):
    d = dict(BASE)
    d.update(kw)
    d["cell"] = name
    d["tier"] = tier
    return d


# ---------------------------------------------------------------- DEV
# Seven cells, spanning the stress axes CAM-EXP-009.1 showed the cue is
# sensitive to. The DEV set exists only to pick ONE method; it is deliberately
# small and its seeds are disjoint from TEST.
DEV_CELLS = [
    cell("DEV_CLEAN", "dev"),
    cell("DEV_ASYM_DENSE_02", "dev", asym_kind="dense", asym=0.02),
    cell("DEV_ASYM_SPARSE_05", "dev", asym_kind="sparse", asym=0.05),
    cell("DEV_ASYM_FINGER_03", "dev", asym_kind="finger", asym=0.03),
    cell("DEV_NOISE_1PX", "dev", noise_px=1.0),
    cell("DEV_ARTIC_1DEG", "dev", artic_deg=1.0),
    cell("DEV_COMBINED_MODERATE", "dev", **COMBINED_MODERATE),
]

# ---------------------------------------------------------------- TEST
TEST_CELLS = [cell("TEST_CLEAN", "primary")]
for kind in ("global", "dense", "sparse", "finger"):
    for a in (0.01, 0.02, 0.05):
        TEST_CELLS.append(
            cell("TEST_ASYM_%s_%02d" % (kind.upper(), int(a * 100)), "primary",
                 asym_kind=kind, asym=a))
for n in (0.5, 1.0, 2.0):
    TEST_CELLS.append(cell("TEST_NOISE_%sPX" % n, "primary", noise_px=n))
for a in (0.5, 1.0, 2.0):
    TEST_CELLS.append(cell("TEST_ARTIC_%sDEG" % a, "primary", artic_deg=a))
for m in (0.10, 0.20):
    TEST_CELLS.append(cell("TEST_MISSING_%d" % int(m * 100), "secondary",
                           missing=m))
for v in (0.75, 0.50, 0.25):
    TEST_CELLS.append(cell("TEST_VIS_%d" % int(v * 100), "secondary",
                           visibility=v))
for d in (2, 8, 16):
    TEST_CELLS.append(cell("TEST_DIST_%dX" % d, "secondary",
                           dist_over_diam=d))
TEST_CELLS.append(cell("TEST_COMBINED_MODERATE", "primary",
                       **COMBINED_MODERATE))
TEST_CELLS.append(cell("TEST_COMBINED_STRONG", "primary", **COMBINED_STRONG))

CONTROLS = [
    {"control": "C0_CORRECT", "expect": "best of the controls",
     "note": "correct bone correspondence; the reference the others degrade "
             "from"},
    {"control": "C1_WRONG_BONE_MAPPING", "expect": "degrade",
     "note": "right-hand bones permuted within each finger before comparison"},
    {"control": "C2_SUBJECT_SWAP", "expect": "degrade",
     "note": "right hand taken from a DIFFERENT synthetic subject"},
    {"control": "C3_TEMPORAL_SHUFFLE", "expect": "degrade or flat",
     "note": "right-hand frames paired with unrelated left-hand frames"},
    {"control": "C4_FIXED_LENGTH_INVARIANT", "expect": "exactly flat",
     "note": "CAM-EXP-009's naive formulation; bone lengths held fixed, so "
             "the score cannot depend on the candidate focal"},
]


def rows(cells, n_subj_primary, n_subj_secondary, namespace):
    out = []
    for c in cells:
        n = (n_subj_primary if c["tier"] in ("dev", "primary")
             else n_subj_secondary)
        for s in range(n):
            r = dict(c)
            r["subject"] = s
            r["trial_id"] = "%s|S%03d" % (c["cell"], s)
            r["seed"] = stable_seed("%s|%s" % (namespace, r["trial_id"]))
            out.append(r)
    return out


def main():
    dev = rows(DEV_CELLS, DEV_SUBJECTS, DEV_SUBJECTS, "CAM0092_DEV")
    test = rows(TEST_CELLS, TEST_SUBJECTS_PRIMARY, TEST_SUBJECTS_SECONDARY,
                "CAM0092_TEST")
    dev_seeds = set(r["seed"] for r in dev)
    test_seeds = set(r["seed"] for r in test)
    assert not (dev_seeds & test_seeds), "DEV and TEST seeds must be disjoint"

    MANIFESTS.mkdir(parents=True, exist_ok=True)
    dev_p = MANIFESTS / "cam_exp_0092_synthetic_dev_trials_v1.csv.gz"
    test_p = MANIFESTS / "cam_exp_0092_synthetic_test_trials_v1.csv.gz"
    write_csv(dev_p, dev)
    write_csv(test_p, test)

    audit = audit_r_LR_redundancy()
    spec = {
        "experiment": "CAM-EXP-009.2",
        "frozen_before_any_dev_result": True,
        "solver": {
            "source": "CAM-EXP-009_1_bilateral_solver_validation/src/"
                      "corrected_bone_fitter.py",
            "imported_unchanged": True,
            "convention": "UNDISTORT_ONCE_INTERNAL",
            "dof": "D20 only",
            "dof_exclusion": "CAM-EXP-009.1's D10/D5 groupings force equal "
                             "bone lengths within a group, which no real hand "
                             "satisfies. They are NOT reused here.",
        },
        "grid": {"q_min": Q_MIN, "q_max": Q_MAX, "q_n": Q_N,
                 "expressed_relative_to": "PIPELINE_BASELINE_FOCAL, never the "
                                          "true focal"},
        "frames": {"n_frames": N_FRAMES, "min_fit_frames": MIN_FIT_FRAMES,
                   "split": "frame parity: even = FIT, odd = EVAL"},
        "methods": METHODS,
        "method_definitions": DEFINITIONS,
        "method_family_closed": "M0-M6 are frozen here. No loss, no tau and no "
                                "trim fraction may be added after a result is "
                                "seen.",
        "r_LR_redundancy_audit": audit,
        "asymmetry_is_subject_level": "Bilateral asymmetry is drawn once per "
                                      "synthetic subject and held fixed for "
                                      "the whole sequence. The two hands are "
                                      "never assumed identical.",
        "sparse_bones": SPARSE_BONES,
        "base_bone_lengths_m": [float(x) for x in BASE_LENGTHS],
        "dev": {"cells": [c["cell"] for c in DEV_CELLS],
                "subjects_per_cell": DEV_SUBJECTS, "trials": len(dev),
                "purpose": "select exactly one method; never reported as "
                           "evidence of performance"},
        "test": {"cells": [c["cell"] for c in TEST_CELLS],
                 "subjects_primary": TEST_SUBJECTS_PRIMARY,
                 "subjects_secondary": TEST_SUBJECTS_SECONDARY,
                 "trials": len(test),
                 "methods_run": "the DEV winner and M0_BASELINE_RAW_L1"},
        "controls": CONTROLS,
        "gate": GATE,
        "compute": {
            "measured_seconds_per_trial": SEC_PER_TRIAL,
            "dev_estimated_minutes": round(len(dev) * SEC_PER_TRIAL / 60, 1),
            "test_estimated_hours": round(len(test) * SEC_PER_TRIAL / 3600, 2),
            "counts_fixed_from_runtime_before_any_result": True,
        },
        "real_phase_rule": "The GigaHands phase runs only if the synthetic "
                           "gate passes. If it fails, the recorded status is "
                           "REAL_FOCAL_PHASE_NOT_RUN.",
        "never_used": ["the true/reference focal as an optimisation input",
                       "GT principal point or distortion as a primary input",
                       "any absolute hand-size prior",
                       "the FINAL_CONFIRMATORY_HOLDOUT"],
    }
    spec_p = MANIFESTS / "cam_exp_0092_robust_candidate_spec_v1.json"
    write_json(spec_p, spec)

    hashes = dict((p.name, sha256(p)) for p in (spec_p, dev_p, test_p))
    write_json(SUM / "manifest_hashes.json", hashes)
    write_csv(RUN_DIR / "tables" / "robust_method_definitions.csv", DEFINITIONS)
    write_json(SUM / "r_lr_redundancy_audit.json", audit)

    print("DEV  trials %4d  ~%.0f min" % (len(dev), len(dev) * SEC_PER_TRIAL / 60))
    print("TEST trials %4d  ~%.2f h" % (len(test), len(test) * SEC_PER_TRIAL / 3600))
    print("r_LR redundant:", audit["redundant"], "max score change %.2e"
          % audit["max_abs_score_change_under_global_right_scale"])
    for k, v in hashes.items():
        print("  %s  %s" % (k, v[:16]))


if __name__ == "__main__":
    main()
