"""Build report_numbers.json - the single source of truth for report figures.

Every entry is READ from a canonical result file. Nothing is recomputed, and no
value is typed in by hand except where the source is a prose report, in which
case the source file and the reason are recorded.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (MANIFESTS, PKG, REPO, R001, R0013, R002, R003, R0031, R004,  # noqa: E402
                    R0041, pick, read_csv, read_json, rel, write_json)

N = {}


def add(key, value, unit, experiment, source_file, source_key,
        statistical_unit, evidence_level, notes=""):
    N[key] = {
        "value": value, "unit": unit, "experiment": experiment,
        "source_file": rel(source_file), "source_column_or_key": source_key,
        "statistical_unit": statistical_unit, "evidence_level": evidence_level,
        "notes": notes,
    }


def f(x, nd=None):
    v = float(x)
    return round(v, nd) if nd is not None else v


# ---------------------------------------------------------------- dataset
def dataset_numbers():
    qc = read_csv(MANIFESTS / "gigahands_demo_qc_v1.csv.gz")
    cb = read_csv(MANIFESTS / "gigahands_demo_camera_benchmark_v1.csv.gz")
    bm = read_csv(MANIFESTS / "gigahands_demo_bimanual_clean_v1.csv.gz")
    fr = read_csv(MANIFESTS / "gigahands_demo_cam_exp_003_frames_v1.csv.gz")
    fr64 = read_csv(MANIFESTS / "gigahands_demo_cam_exp_0041_64frames_v1.csv.gz")
    st = Counter(r["qc_status"] for r in qc)

    src = rel(MANIFESTS / "gigahands_demo_qc_v1.csv.gz")
    for k, v in (("PASS_STRICT", st["PASS_STRICT"]),
                 ("PASS_SINGLE_HAND", st["PASS_SINGLE_HAND"]),
                 ("REVIEW", st["REVIEW"]), ("EXCLUDE", st["EXCLUDE"])):
        add(f"gigahands_qc_{k.lower()}_n", v, "hand observations", "CAM-EXP-001.3",
            MANIFESTS / "gigahands_demo_qc_v1.csv.gz", f"qc_status == {k}",
            "hand observation (sequence, camera, frame, hand)",
            "CONFIRMED_IN_CURRENT_ENVIRONMENT",
            "counted directly from the manifest")
    add("gigahands_qc_total_observations", len(qc), "hand observations",
        "CAM-EXP-001.3", MANIFESTS / "gigahands_demo_qc_v1.csv.gz", "row count",
        "hand observation", "CONFIRMED_IN_CURRENT_ENVIRONMENT",
        "2 hands x 54,120 frame-view pairs")
    add("gigahands_qc_zero_pattern_n", sum(1 for r in qc if r["zero_pattern"] == "1"),
        "hand observations", "CAM-EXP-001.2 / 001.3",
        MANIFESTS / "gigahands_demo_qc_v1.csv.gz", "zero_pattern == 1",
        "hand observation", "CONFIRMED_IN_CURRENT_ENVIRONMENT",
        "the observed invalid all-zero 2D pattern; NOT a documented sentinel")
    add("gigahands_bimanual_clean_frames", len(bm), "frames", "CAM-EXP-001.3",
        MANIFESTS / "gigahands_demo_bimanual_clean_v1.csv.gz", "row count",
        "frame within a view", "CONFIRMED_IN_CURRENT_ENVIRONMENT",
        "frames where BOTH hands pass strict QC. This is the ELIGIBLE POOL that "
        "CAM-EXP-002 draws from; it is NOT the sample CAM-EXP-002 evaluated - "
        "see cam002_frames_attempted / cam002_hands_evaluated.")
    add("gigahands_bimanual_clean_views", len({(r["sequence"], r["camera"]) for r in bm}),
        "views", "CAM-EXP-001.3", MANIFESTS / "gigahands_demo_bimanual_clean_v1.csv.gz",
        "distinct (sequence, camera)", "view", "CONFIRMED_IN_CURRENT_ENVIRONMENT", "")
    add("camera_benchmark_views_usable",
        sum(1 for r in cb if r["usable_for_camera_benchmark"] == "1"), "views",
        "CAM-EXP-003", MANIFESTS / "gigahands_demo_camera_benchmark_v1.csv.gz",
        "usable_for_camera_benchmark == 1", "view",
        "CONFIRMED_IN_CURRENT_ENVIRONMENT",
        "camera usability only; hand annotation quality is NOT a filter here")
    add("camera_benchmark_views_excluded",
        sum(1 for r in cb if r["usable_for_camera_benchmark"] != "1"), "views",
        "CAM-EXP-003", MANIFESTS / "gigahands_demo_camera_benchmark_v1.csv.gz",
        "usable_for_camera_benchmark == 0", "view",
        "CONFIRMED_IN_CURRENT_ENVIRONMENT",
        "all 25 lack a usable RGB segment matching the annotation")
    add("gigahands_unique_physical_cameras", len({r["camera"] for r in cb}), "cameras",
        "all", MANIFESTS / "gigahands_demo_camera_benchmark_v1.csv.gz",
        "distinct camera id", "physical camera", "CONFIRMED_IN_CURRENT_ENVIRONMENT",
        "each appears in several sequences; used as a coarser cluster unit")
    add("gigahands_sequences", len({r["sequence"] for r in cb}), "sequences", "all",
        MANIFESTS / "gigahands_demo_camera_benchmark_v1.csv.gz",
        "distinct sequence", "sequence", "CONFIRMED_IN_CURRENT_ENVIRONMENT", "")
    add("cam003_frames_total", len(fr), "frames", "CAM-EXP-003",
        MANIFESTS / "gigahands_demo_cam_exp_003_frames_v1.csv.gz", "row count",
        "frame (REPEATED OBSERVATION, not an independent sample)",
        "CONFIRMED_IN_CURRENT_ENVIRONMENT",
        "175 views x 8 frames; the statistical unit is the view, not the frame")
    add("cam0041_frames_total", len(fr64), "frames", "CAM-EXP-004.1",
        MANIFESTS / "gigahands_demo_cam_exp_0041_64frames_v1.csv.gz", "row count",
        "frame (repeated observation)", "CONFIRMED_IN_CURRENT_ENVIRONMENT",
        "175 views x 64 frames; the CAM-EXP-003 8 frames are an exact subset")


# --------------------------------------------------------------- CAM-EXP-001
def exp001_numbers():
    tri = read_csv(R0013 / "results" / "summary" / "triangulation_summary.csv")
    same = pick(tri, metric="reconstruction_vs_provided_same_hand_mpjpe")
    other = pick(tri, metric="reconstruction_vs_provided_other_hand_mpjpe")
    rep = pick(tri, metric="triangulation_median_reprojection_px")
    note = ("SELF-CONSISTENCY between the released 2D observations and the "
            "provided 3D. It is NOT accuracy against an independent external "
            "ground truth, and must never be written as such.")
    add("cam0013_recon_vs_provided3d_same_hand_median_mm", f(same["median"], 4), "mm",
        "CAM-EXP-001.3", R0013 / "results/summary/triangulation_summary.csv",
        "median", "hand observation", "CONFIRMED_IN_CURRENT_ENVIRONMENT", note)
    add("cam0013_recon_vs_provided3d_same_hand_p90_mm", f(same["p90"], 4), "mm",
        "CAM-EXP-001.3", R0013 / "results/summary/triangulation_summary.csv",
        "p90", "hand observation", "CONFIRMED_IN_CURRENT_ENVIRONMENT", note)
    add("cam0013_recon_vs_provided3d_other_hand_median_mm", f(other["median"], 4), "mm",
        "CAM-EXP-001.3", R0013 / "results/summary/triangulation_summary.csv",
        "median", "hand observation", "CONFIRMED_IN_CURRENT_ENVIRONMENT",
        "the wrong-hand control: ~48x larger, which is what makes the same-hand "
        "agreement meaningful rather than trivial")
    add("cam0013_triangulation_median_reprojection_px", f(rep["median"], 4), "px",
        "CAM-EXP-001.3", R0013 / "results/summary/triangulation_summary.csv",
        "median", "hand observation", "CONFIRMED_IN_CURRENT_ENVIRONMENT", "")
    add("cam0013_n_triangulated_observations", int(same["n"]), "hand observations",
        "CAM-EXP-001.3", R0013 / "results/summary/triangulation_summary.csv", "n",
        "hand observation", "CONFIRMED_IN_CURRENT_ENVIRONMENT", "")

    # every verdict class, including the ones that were NOT explained. Leaving
    # UNRESOLVED_INSUFFICIENT_GEOMETRY out would make the diagnosis look more
    # complete than it is.
    ident = read_csv(R0013 / "results" / "summary" / "identity_verdict_summary.csv")
    for case in sorted(r["case"] for r in ident):
        r = pick(ident, case=case)
        add(f"cam0013_identity_{case.lower()}_n", int(r["n"]), "hand observations",
            "CAM-EXP-001.3", R0013 / "results/summary/identity_verdict_summary.csv",
            f"case == {case}", "hand observation",
            "CONFIRMED_IN_CURRENT_ENVIRONMENT", "")

    ds = read_csv(R001 / "results" / "summary" / "dataset_summary.csv")
    g = pick(ds, dataset="gigahands")
    add("cam001_gigahands_reprojection_median_px", f(g["inimg_median_px"], 4), "px",
        "CAM-EXP-001", R001 / "results/summary/dataset_summary.csv",
        "inimg_median_px", "joint", "CONFIRMED_IN_CURRENT_ENVIRONMENT",
        "provided 3D projected with the provided camera vs the provided 2D; the "
        "long tail here is what CAM-EXP-001.1/001.2/001.3 went on to explain")
    add("cam001_gigahands_reprojection_frac_under_10px", f(g["inimg_frac_under_10px"], 4),
        "fraction", "CAM-EXP-001", R001 / "results/summary/dataset_summary.csv",
        "inimg_frac_under_10px", "joint", "CONFIRMED_IN_CURRENT_ENVIRONMENT", "")


# --------------------------------------------------------------- CAM-EXP-002
def exp002_usage():
    """What CAM-EXP-002 actually ran on, as opposed to what it could have run on.

    The bimanual-clean manifest is the ELIGIBLE POOL. The experiment drew a
    stratified sample from it, and frames were lost to detector and
    hand-association failures before evaluation. Those are different numbers and
    are recorded separately.
    """
    meta = read_json(R002 / "results" / "summary" / "_inference_meta.json")
    cache = read_csv(R002 / "results" / "raw" / "inference_cache_index.csv")
    mfail = read_csv(R002 / "results" / "raw" / "model_failures.csv")
    afail = read_csv(R002 / "results" / "raw" / "hand_association_failures.csv")
    hands = [r for r in read_csv(R002 / "results" / "raw"
                                 / "focal_sweep_per_hand.csv.gz")
             if r["condition"] == "GT_x1.00"]
    frames = [r for r in read_csv(R002 / "results" / "raw"
                                  / "focal_sweep_per_frame.csv.gz")
              if r["condition"] == "GT_x1.00"]
    ok = [r for r in cache if r["status"] != "no_detection"]
    src = R002 / "results" / "raw" / "inference_cache_index.csv"

    add("cam002_frames_attempted", len(cache), "frames", "CAM-EXP-002", src,
        "row count", "frame", "CONFIRMED_IN_CURRENT_ENVIRONMENT",
        f"stratified sample drawn from the bimanual-clean eligible pool "
        f"(target {meta['target_n']}, seed {meta['seed']}); grouped by "
        "(sequence, camera), shuffled, taken round-robin and evenly spaced in "
        "time within a group")
    add("cam002_frames_inference_successful", len(ok), "frames", "CAM-EXP-002",
        src, "status != no_detection", "frame",
        "CONFIRMED_IN_CURRENT_ENVIRONMENT", "")
    add("cam002_model_failures", len(mfail), "frames", "CAM-EXP-002",
        R002 / "results" / "raw" / "model_failures.csv", "row count", "frame",
        "CONFIRMED_IN_CURRENT_ENVIRONMENT", "no hand detected")
    add("cam002_hand_association_failures", len(afail), "frames", "CAM-EXP-002",
        R002 / "results" / "raw" / "hand_association_failures.csv", "row count",
        "frame", "CONFIRMED_IN_CURRENT_ENVIRONMENT",
        "AMBIGUOUS_HAND_ASSOCIATION; the frame is dropped rather than guessed")
    hsrc = R002 / "results" / "raw" / "focal_sweep_per_hand.csv.gz"
    add("cam002_frames_evaluated", len({(r["sequence"], r["camera"], r["frame"])
                                        for r in hands}), "frames",
        "CAM-EXP-002", hsrc, "distinct (sequence, camera, frame) at GT_x1.00",
        "frame", "CONFIRMED_IN_CURRENT_ENVIRONMENT",
        "frames contributing at least one evaluated hand")
    add("cam002_hands_evaluated", len(hands), "hands", "CAM-EXP-002", hsrc,
        "rows at condition GT_x1.00", "hand",
        "CONFIRMED_IN_CURRENT_ENVIRONMENT",
        "the analysis unit; matches n_hands in the summary tables")
    add("cam002_views_evaluated", len({(r["sequence"], r["camera"])
                                       for r in hands}), "views", "CAM-EXP-002",
        hsrc, "distinct (sequence, camera) at GT_x1.00", "view",
        "CONFIRMED_IN_CURRENT_ENVIRONMENT", "")
    add("cam002_unique_physical_cameras_evaluated",
        len({r["camera"] for r in hands}), "cameras", "CAM-EXP-002", hsrc,
        "distinct camera id at GT_x1.00", "physical camera",
        "CONFIRMED_IN_CURRENT_ENVIRONMENT", "")
    add("cam002_sequences_evaluated", len({r["sequence"] for r in hands}),
        "sequences", "CAM-EXP-002", hsrc, "distinct sequence at GT_x1.00",
        "sequence", "CONFIRMED_IN_CURRENT_ENVIRONMENT", "")
    add("cam002_bimanual_frames_evaluated",
        len({(r["sequence"], r["camera"], r["frame"]) for r in frames}),
        "frames", "CAM-EXP-002",
        R002 / "results" / "raw" / "focal_sweep_per_frame.csv.gz",
        "distinct frames at GT_x1.00", "frame",
        "CONFIRMED_IN_CURRENT_ENVIRONMENT",
        "frames where BOTH hands were evaluated, used for the bimanual metrics")


def exp002_numbers():
    exp002_usage()
    bl = read_csv(R002 / "results" / "summary" / "baseline_vs_gt_focal.csv")
    base = pick(bl, condition="PIPELINE_BASELINE")
    gt = pick(bl, condition="GT_x1.00")
    scope = ("SCOPE: focal-length sensitivity of ABSOLUTE hand depth. This "
             "experiment did not test principal point, anisotropic focal or "
             "distortion downstream.")
    add("cam002_pipeline_virtual_focal_px", f(base["focal_used_px_median"]), "px",
        "CAM-EXP-002", R002 / "results/summary/baseline_vs_gt_focal.csv",
        "focal_used_px_median @ PIPELINE_BASELINE", "hand",
        "CONFIRMED_IN_CURRENT_ENVIRONMENT",
        "the deployed pipeline's assumed focal, FOCAL_LENGTH/IMAGE_SIZE*max(W,H)")
    add("cam002_gigahands_physical_focal_median_px", f(gt["focal_used_px_median"], 2),
        "px", "CAM-EXP-002", R002 / "results/summary/baseline_vs_gt_focal.csv",
        "focal_used_px_median @ GT_x1.00", "hand",
        "CONFIRMED_IN_CURRENT_ENVIRONMENT",
        "median dataset-provided focal over the evaluated hands")
    add("cam002_baseline_root_error_median_mm", f(base["root_xyz_error_mm_median"], 3),
        "mm", "CAM-EXP-002", R002 / "results/summary/baseline_vs_gt_focal.csv",
        "root_xyz_error_mm_median @ PIPELINE_BASELINE", "hand",
        "CONFIRMED_IN_CURRENT_ENVIRONMENT", scope)
    add("cam002_physical_focal_root_error_median_mm", f(gt["root_xyz_error_mm_median"], 3),
        "mm", "CAM-EXP-002", R002 / "results/summary/baseline_vs_gt_focal.csv",
        "root_xyz_error_mm_median @ GT_x1.00", "hand",
        "CONFIRMED_IN_CURRENT_ENVIRONMENT", scope)
    add("cam002_baseline_absolute_mpjpe_median_mm", f(base["absolute_mpjpe_mm_median"], 3),
        "mm", "CAM-EXP-002", R002 / "results/summary/baseline_vs_gt_focal.csv",
        "absolute_mpjpe_mm_median @ PIPELINE_BASELINE", "hand",
        "CONFIRMED_IN_CURRENT_ENVIRONMENT", scope)
    add("cam002_physical_focal_absolute_mpjpe_median_mm",
        f(gt["absolute_mpjpe_mm_median"], 3), "mm", "CAM-EXP-002",
        R002 / "results/summary/baseline_vs_gt_focal.csv",
        "absolute_mpjpe_mm_median @ GT_x1.00", "hand",
        "CONFIRMED_IN_CURRENT_ENVIRONMENT", scope)
    add("cam002_root_aligned_mpjpe_median_mm", f(gt["root_aligned_mpjpe_mm_median"], 3),
        "mm", "CAM-EXP-002", R002 / "results/summary/baseline_vs_gt_focal.csv",
        "root_aligned_mpjpe_mm_median (identical in both conditions)", "hand",
        "CONFIRMED_IN_CURRENT_ENVIRONMENT",
        "IDENTICAL under both focals - focal changes metric placement, not hand "
        "shape. This is the control that makes the comparison interpretable.")
    add("cam002_n_hands", int(base["n_hands"]), "hands", "CAM-EXP-002",
        R002 / "results/summary/baseline_vs_gt_focal.csv", "n_hands", "hand",
        "CONFIRMED_IN_CURRENT_ENVIRONMENT", "")

    sens = read_csv(R002 / "results" / "summary" / "focal_sensitivity_summary.csv")
    for pct in ("5.0", "10.0", "20.0"):
        r = pick(sens, focal_error_percent=pct)
        add(f"cam002_focal_{int(float(pct))}pct_root_shift_median_mm",
            f(r["incremental_root_shift_median_mm"], 3), "mm", "CAM-EXP-002",
            R002 / "results/summary/focal_sensitivity_summary.csv",
            "incremental_root_shift_median_mm", "hand",
            "CONFIRMED_IN_CURRENT_ENVIRONMENT",
            "INCREMENTAL displacement caused by perturbing focal alone, at this "
            "working distance. Not a measure of calibration accuracy. " + scope)
        add(f"cam002_focal_{int(float(pct))}pct_root_shift_p90_mm",
            f(r["incremental_root_shift_p90_mm"], 3), "mm", "CAM-EXP-002",
            R002 / "results/summary/focal_sensitivity_summary.csv",
            "incremental_root_shift_p90_mm", "hand",
            "CONFIRMED_IN_CURRENT_ENVIRONMENT", scope)


# ------------------------------------------------------- CAM-EXP-003 / 003.1
def exp003_numbers():
    ms = read_csv(R003 / "results" / "summary" / "model_summary.csv")
    unit = ("view (175). The 1400 frames are 8 repeated observations of each of "
            "175 static cameras and are NOT 1400 independent samples.")
    for key, model in (("anycalib_pinhole", "AnyCalib[anycalib_pinhole/pinhole]"),
                       ("geocalib_pinhole", "GeoCalib[pinhole]"),
                       ("pf_centered", "PerspectiveFields[centered]"),
                       ("pf_uncentered", "PerspectiveFields[uncentered]"),
                       ("demo_fixed_5000", "DEMO_FIXED_5000")):
        r = pick(ms, model=model)
        for metric, col in (("median", "focal_error_pct_median"),
                            ("p90", "focal_error_pct_p90"),
                            ("within5", "within_5pct_rate")):
            v = f(r[col], 4)
            if metric == "within5":
                v = round(v * 100, 2)
            add(f"cam003_{key}_focal_err_{metric}", v,
                "%" if metric != "within5" else "% of views",
                "CAM-EXP-003", R003 / "results/summary/model_summary.csv", col,
                unit, "ROBUST_BUT_SINGLE_DATASET",
                "single frame, raw image, pinhole camera model")

    ms31 = read_csv(R0031 / "results" / "summary" / "model_summary.csv")
    for key in ("anycalib_gen_radial", "geocalib_distorted_radial"):
        r = pick(ms31, model_run=key)
        for metric, col, scale in (("median", "focal_error_pct_median", 1),
                                   ("p90", "focal_error_pct_p90", 1),
                                   ("within5", "within_5pct_rate", 100),
                                   ("within10", "within_10pct_rate", 100),
                                   ("within20", "within_20pct_rate", 100),
                                   ("signed_median", "signed_focal_error_pct_median", 1)):
            add(f"cam0031_{key}_focal_err_{metric}", round(f(r[col]) * scale, 3),
                "%" if scale == 1 else "% of frames",
                "CAM-EXP-003.1", R0031 / "results/summary/model_summary.csv", col,
                "frame for the rate; the view-clustered CI for the PAIRED "
                "comparison is in CAM-EXP-004.1",
                "ROBUST_BUT_SINGLE_DATASET",
                "single frame, raw image, distortion-aware camera model")

    cl = read_csv(R0041 / "results" / "summary" / "cam0031_statistical_reanalysis.csv")
    for run in ("anycalib_gen_radial", "geocalib_distorted_radial"):
        for lvl in ("LEGACY_FRAME_BOOTSTRAP", "VIEW_CLUSTER",
                    "PHYSICAL_CAMERA_CLUSTER", "SEQUENCE_CLUSTER"):
            r = pick(cl, model_run=run, resampling_unit=lvl)
            lab = {"LEGACY_FRAME_BOOTSTRAP": "frame_legacy", "VIEW_CLUSTER": "view",
                   "PHYSICAL_CAMERA_CLUSTER": "camera",
                   "SEQUENCE_CLUSTER": "sequence"}[lvl]
            add(f"cam0031_{run}_improvement_pp_{lab}",
                [f(r["median_improvement_pp"], 3), f(r["ci_lo"], 3), f(r["ci_hi"], 3)],
                "percentage points [median, ci_lo, ci_hi]", "CAM-EXP-004.1",
                R0041 / "results/summary/cam0031_statistical_reanalysis.csv",
                "median_improvement_pp, ci_lo, ci_hi",
                f"{lvl} ({r['n_clusters']} clusters)",
                "CONFIRMED_IN_CURRENT_ENVIRONMENT" if lvl != "SEQUENCE_CLUSTER"
                else "EXPLORATORY_INTERNAL_VALIDATION",
                "point estimate identical at every level; only the CI changes. "
                + ("5 clusters - sensitivity analysis only, not a trustworthy CI"
                   if lvl == "SEQUENCE_CLUSTER" else
                   "pseudo-replicated - retained for comparison only"
                   if lvl == "LEGACY_FRAME_BOOTSTRAP" else "primary"))


# ------------------------------------------------------- CAM-EXP-004 / 004.1
def exp004_numbers():
    bn = read_csv(R004 / "results" / "summary" / "bias_noise_summary.csv")
    for r in bn:
        m = r["model"]
        add(f"cam004_{m}_bias_fraction_pct", f(r["bias_fraction_pct"], 2), "%",
            "CAM-EXP-004", R004 / "results/summary/bias_noise_summary.csv",
            "bias_fraction_pct", "view", "ROBUST_BUT_SINGLE_DATASET",
            "share of the total squared log focal error that is a stable "
            "BETWEEN-VIEW component: the decomposition unit is the "
            "(sequence, camera) static view, with frames treated as repeated "
            "observations inside it. This is NOT a physical-camera-level "
            "statement; physical camera is a RESAMPLING unit used in "
            "CAM-EXP-004.1, not the decomposition unit.")
        add(f"cam004_{m}_median_abs_view_bias_pct", f(r["median_abs_view_bias_pct"], 2),
            "%", "CAM-EXP-004", R004 / "results/summary/bias_noise_summary.csv",
            "median_abs_view_bias_pct", "view", "ROBUST_BUT_SINGLE_DATASET", "")
        add(f"cam004_{m}_median_within_view_std_pct",
            f(r["median_within_view_std_pct"], 2), "%", "CAM-EXP-004",
            R004 / "results/summary/bias_noise_summary.csv",
            "median_within_view_std_pct", "view", "ROBUST_BUT_SINGLE_DATASET", "")

    fc = read_csv(R0041 / "results" / "summary" / "frame_count_8_16_32_64.csv")
    for est in ("anycalib_gen", "geocalib_distorted", "E2_frozen"):
        for n in (8, 16, 32, 64):
            r = pick(fc, estimator=est, N=n)
            add(f"cam0041_{est}_N{n}_focal_err_median", f(r["median_rel_err_pct"], 3),
                "%", "CAM-EXP-004.1",
                R0041 / "results/summary/frame_count_8_16_32_64.csv",
                "median_rel_err_pct", "view (175)", "ROBUST_BUT_SINGLE_DATASET",
                "single run; all four N come from the same inference pass")
            add(f"cam0041_{est}_N{n}_within5", f(r["within_5_pct"], 2), "% of views",
                "CAM-EXP-004.1", R0041 / "results/summary/frame_count_8_16_32_64.csv",
                "within_5_pct", "view (175)", "ROBUST_BUT_SINGLE_DATASET", "")

    rr = read_csv(R0041 / "results" / "summary" / "independent_rerun_comparison.csv")
    for r in rr:
        est = r["estimator"]
        add(f"e2_or_model_rerun_{est}_cam004_median", f(r["cam_exp_004_median_pct"], 3),
            "%", "CAM-EXP-004", R0041 / "results/summary/independent_rerun_comparison.csv",
            "cam_exp_004_median_pct", "view (175)", "ROBUST_BUT_SINGLE_DATASET",
            "original CAM-EXP-004 run at N=8")
        add(f"e2_or_model_rerun_{est}_cam0041_median",
            f(r["cam_exp_0041_rerun_median_pct"], 3), "%", "CAM-EXP-004.1",
            R0041 / "results/summary/independent_rerun_comparison.csv",
            "cam_exp_0041_rerun_median_pct", "view (175)",
            "ROBUST_BUT_SINGLE_DATASET",
            "independent re-run of the identical 8 frames, same environment")

    lo = read_csv(R0041 / "results" / "summary" / "ensemble_selection_stability.csv")
    vals = [f(r["heldout_median_of_fixed_E2_pct"], 3) for r in lo]
    add("e2_loso_heldout_median_range_pct", [min(vals), max(vals)], "%",
        "CAM-EXP-004.1", R0041 / "results/summary/ensemble_selection_stability.csv",
        "heldout_median_of_fixed_E2_pct", "sequence fold (5 folds)",
        "EXPLORATORY_INTERNAL_VALIDATION",
        "RETROSPECTIVE_INTERNAL_VALIDATION - same rig, same 5 sequences; not an "
        "independent test")
    add("e2_loso_heldout_medians_pct", vals, "%", "CAM-EXP-004.1",
        R0041 / "results/summary/ensemble_selection_stability.csv",
        "heldout_median_of_fixed_E2_pct", "sequence fold",
        "EXPLORATORY_INTERNAL_VALIDATION",
        "order: " + ", ".join(r["held_out_sequence"] for r in lo))
    add("e2_loso_folds_selecting_e2",
        sum(1 for r in lo if r["selected_on_training_folds"] == "E2_perframe_geomean"),
        "of 5 folds", "CAM-EXP-004.1",
        R0041 / "results/summary/ensemble_selection_stability.csv",
        "selected_on_training_folds", "sequence fold",
        "EXPLORATORY_INTERNAL_VALIDATION", "selection cost 0.00 pp in every fold")

    det = read_csv(R0041 / "results" / "summary" / "model_determinism_check.csv")
    for r in det:
        m = r["model"]
        for k, col in (("median_spread_pct", "median_spread_pct"),
                       ("p90_spread_pct", "p90_spread_pct"),
                       ("max_spread_pct", "max_spread_pct"),
                       ("pct_bit_identical", "pct_frames_bit_identical")):
            add(f"determinism_{m}_{k}", f(r[col], 2), "%", "CAM-EXP-004.1",
                R0041 / "results/summary/model_determinism_check.csv", col,
                "frame (40 frames x 3 repeats on byte-identical images)",
                "CONFIRMED_IN_CURRENT_ENVIRONMENT",
                "run-to-run variation on the SAME input; the input was verified "
                "bit-identical (max abs pixel difference 0)")

    ai = read_json(R0041 / "results" / "summary" / "anycam_identifiability.json")
    for r in ai["results"]:
        add(f"anycam_{r['condition'].lower()}_candidate_spread",
            float(r["spread_across_candidates"]), "induced-flow units",
            "CAM-EXP-004.1", R0041 / "results/summary/anycam_identifiability.json",
            "results[].spread_across_candidates", "synthetic test case",
            "CONFIRMED_IN_CURRENT_ENVIRONMENT",
            "spread of AnyCam's own focal-scoring objective across its 32 focal "
            "candidates; flat means the focal is not identifiable")


def main() -> None:
    dataset_numbers()
    exp001_numbers()
    exp002_numbers()
    exp003_numbers()
    exp004_numbers()

    out = {
        "_meta": {
            "id": "PROGRESS_REPORT_V1_CANONICAL_NUMBERS",
            "purpose": "single source of truth for every number that may appear "
                       "in the progress report. Read from canonical result files; "
                       "nothing recomputed, nothing hand-edited.",
            "rule": "if a number is not in here, it is not ready to be written in "
                    "the report. If a number in the report disagrees with here, "
                    "here wins.",
            "evidence_levels": [
                "CONFIRMED_IN_CURRENT_ENVIRONMENT",
                "ROBUST_BUT_SINGLE_DATASET",
                "EXPLORATORY_INTERNAL_VALIDATION",
                "PENDING_EXTERNAL_CONFIRMATION",
                "OPEN_ISSUE"],
            "n_entries": len(N),
            "generated_by": "experiments/report_prep/progress_report_v1/scripts/"
                            "build_report_numbers.py",
        },
        "numbers": N,
    }
    write_json(PKG / "report_numbers.json", out)
    print(f"report_numbers.json: {len(N)} canonical entries")
    missing = [k for k, v in N.items()
               if not (REPO / v["source_file"]).exists()]
    print(f"missing source files: {missing if missing else 'none'}")


if __name__ == "__main__":
    main()
