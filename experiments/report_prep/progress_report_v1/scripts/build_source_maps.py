"""Traceability: every report-facing artifact back to the file it came from."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PKG, RUNS, write_csv, write_md_table  # noqa: E402

ARTIFACTS = [
    ("Fig01", "pipeline virtual focal convention (5000 px) and the "
     "dataset-provided reference focal (GT_EFFECTIVE_FOCAL)",
     "CAM-EXP-002",
     "runs/CAM-EXP-002_camera_focal_sensitivity/results/summary/baseline_vs_gt_focal.csv",
     "focal_used_px_median", "none - read directly", "scripts/make_figures.py::fig01"),
    ("Fig02", "one headline finding per experiment", "all",
     "report_numbers.json", "several keys",
     "none - each value is quoted from report_numbers.json",
     "scripts/make_figures.py::fig02"),
    ("Fig03", "QC counts, camera-clean and bimanual-clean subset sizes",
     "CAM-EXP-001.3 / CAM-EXP-003",
     "manifests/gigahands_demo_qc_v1.csv.gz; "
     "manifests/gigahands_demo_camera_benchmark_v1.csv.gz; "
     "manifests/gigahands_demo_bimanual_clean_v1.csv.gz",
     "qc_status, usable_for_camera_benchmark, row counts",
     "counting only", "scripts/build_report_numbers.py, scripts/make_figures.py::fig03"),
    ("Fig04", "median and p90 depth displacement per focal perturbation",
     "CAM-EXP-002",
     "runs/CAM-EXP-002_camera_focal_sensitivity/results/summary/"
     "focal_sensitivity_summary.csv",
     "incremental_signed_dz_median_mm, incremental_root_shift_p90_mm",
     "none - read directly", "scripts/make_figures.py::fig04"),
    ("Fig05", "single-frame focal error, pinhole vs distortion-aware",
     "CAM-EXP-003 / CAM-EXP-003.1",
     "runs/CAM-EXP-003_single_frame_calibration_benchmark/results/summary/"
     "model_summary.csv; runs/CAM-EXP-003_1_distortion_aware_diagnostic/"
     "results/summary/model_summary.csv",
     "focal_error_pct_median, within_5pct_rate, within_10pct_rate, "
     "within_20pct_rate",
     "rates multiplied by 100 for display", "scripts/make_figures.py::fig05"),
    ("Fig05 caption", "view-clustered paired CI for the distortion-aware gain",
     "CAM-EXP-004.1",
     "runs/CAM-EXP-004_1_pre_cam005_robustness_audit/results/summary/"
     "cam0031_statistical_reanalysis.csv",
     "median_improvement_pp, ci_lo, ci_hi (resampling_unit = VIEW_CLUSTER)",
     "none", "scripts/build_report_numbers.py"),
    ("Fig06", "median focal error for N = 1..64, one run", "CAM-EXP-004.1",
     "runs/CAM-EXP-004_1_pre_cam005_robustness_audit/results/raw/"
     "extended_frame_predictions.csv.gz",
     "pred_fx, gt_fx, grid_pos, in_n8, in_n16, in_n32",
     "re-aggregated with the frozen CAM-EXP-004 rules; for N<=8 all C(8,N) "
     "subsets of the frozen 8-frame set, averaged per view. NO new inference.",
     "scripts/build_multiframe_curve.py"),
    ("Fig06 crosses", "CAM-EXP-004 original run at N = 1,2,4,8", "CAM-EXP-004",
     "runs/CAM-EXP-004_static_camera_multiframe_aggregation/results/raw/"
     "per_view_aggregation.csv.gz; .../cross_model_ensemble.csv.gz",
     "mean_rel_err_pct", "median over views; plotted as points, never joined "
     "to the curve", "scripts/build_multiframe_curve.py"),
    ("Fig07 panel A", "bias vs noise share of the squared log focal error",
     "CAM-EXP-004",
     "runs/CAM-EXP-004_static_camera_multiframe_aggregation/results/summary/"
     "bias_noise_summary.csv",
     "bias_fraction_pct, within_fraction_pct", "none",
     "scripts/make_figures.py::fig07"),
    ("Fig07 panel B", "median error and within-5 % rate at N = 8",
     "CAM-EXP-004.1", "figures/main/data/Fig06.csv",
     "median_rel_err_pct, within_5_pct", "single-run re-aggregation",
     "scripts/build_multiframe_curve.py, scripts/make_figures.py::fig07"),
    ("Fig07 panel C", "signed median focal error per model", "CAM-EXP-003.1",
     "runs/CAM-EXP-003_1_distortion_aware_diagnostic/results/summary/"
     "model_summary.csv",
     "signed_focal_error_pct_median", "E2's value is taken from the "
     "CAM-EXP-004.1 re-run aggregation", "scripts/make_figures.py::fig07"),
    ("tables/report_dataset_usage", "per-experiment data counts", "all",
     "the four GigaHands manifests plus each run's summary", "several",
     "counting only", "scripts/build_tables.py"),
    ("tables/hypothesis_result_evidence", "hypothesis, result, evidence level",
     "all", "report_numbers.json", "several keys", "none",
     "scripts/build_tables.py"),
    ("tables/claim_evidence_ledger", "allowed claims and wording", "all",
     "report_numbers.json", "several keys", "none", "scripts/build_tables2.py"),
    ("tables/report_main_results", "headline numbers per section", "all",
     "report_numbers.json", "several keys", "none", "scripts/build_tables2.py"),
    ("tables/multiframe_run_agreement",
     "difference between the two runs at the overlapping N", "CAM-EXP-004 vs 004.1",
     "figures/main/data/Fig06.csv and Fig06_cam004_original.csv",
     "median_rel_err_pct", "subtraction", "scripts/build_multiframe_curve.py"),
    ("report_numbers.json", "every report-facing number", "all",
     "the canonical summary files listed in each entry", "per entry",
     "read only", "scripts/build_report_numbers.py"),
]

APPENDIX = [
    ("CAM_EXP_004_MAIN_EXPLANATION.png",
     "runs/CAM-EXP-004_static_camera_multiframe_aggregation/figures/",
     "the seven-panel summary of the multi-frame result", "A.4 Experiment 4",
     "DROP", "its content is now split across Fig06 and Fig07, which are "
             "print-safe; keeping both would duplicate the message"),
    ("per_view_trajectories.png",
     "runs/CAM-EXP-004_static_camera_multiframe_aggregation/figures/",
     "predicted fx per frame for a good, a biased and a noisy view",
     "A.4 Experiment 4", "KEEP",
     "the single clearest picture of 'the camera never changed, so every offset "
     "is model error'"),
    ("oracle_best_vs_aggregation.png",
     "runs/CAM-EXP-004_static_camera_multiframe_aggregation/figures/",
     "oracle best-of-N vs deployable aggregation", "A.4 Experiment 4", "KEEP",
     "shows that even GT-assisted frame selection does not reach the target, "
     "which is why frame selection was deprioritised"),
    ("bias_vs_noise_decomposition.png",
     "runs/CAM-EXP-004_static_camera_multiframe_aggregation/figures/",
     "bias vs noise, with magnitudes", "A.4 Experiment 4", "DROP",
     "superseded by Fig07 panel A"),
    ("model_agreement_vs_error.png",
     "runs/CAM-EXP-004_static_camera_multiframe_aggregation/figures/",
     "GT-blind model disagreement vs actual error", "A.4 Experiment 4", "KEEP",
     "a weak but honest negative-ish result worth showing rather than hiding"),
    ("CAM_EXP_003_1_MAIN_EXPLANATION.png",
     "runs/CAM-EXP-003_1_distortion_aware_diagnostic/figures/",
     "distortion-aware diagnostic summary", "A.3 Experiment 3", "DROP",
     "superseded by Fig05"),
    ("distortion_magnitude_vs_improvement.png",
     "runs/CAM-EXP-003_1_distortion_aware_diagnostic/figures/",
     "|k1| vs improvement, with the restricted-range caveat", "A.3 Experiment 3",
     "KEEP", "documents an inconclusive analysis that a reader would otherwise "
             "expect us to have done"),
    ("static_camera_bias_before_after.png",
     "runs/CAM-EXP-003_1_distortion_aware_diagnostic/figures/",
     "per-view bias before and after distortion handling", "A.3 Experiment 3",
     "KEEP", "links Experiment 3 to Experiment 4's bias finding"),
    ("examples/raw_vs_undistorted_examples.png",
     "runs/CAM-EXP-003_1_distortion_aware_diagnostic/figures/",
     "what the undistortion actually did to the images", "A.3 Experiment 3",
     "KEEP", "makes the oracle condition concrete and shows it was not free"),
    ("CAM_EXP_004_1_MAIN_EXPLANATION.png",
     "runs/CAM-EXP-004_1_pre_cam005_robustness_audit/figures/",
     "robustness audit summary: cluster CIs, LOSO, 8-64, determinism",
     "A.5 Robustness and reproducibility", "KEEP",
     "the only figure that shows the statistical re-analysis and the determinism "
     "finding together"),
    ("qc_examples/qc_status_grid.png",
     "runs/CAM-EXP-001_3_gigahands_multiview_triangulation/figures/",
     "PASS_STRICT / REVIEW / EXCLUDE example crops", "A.1 Experiment 1", "KEEP",
     "lets a reader judge the QC labels instead of trusting them"),
    ("overlay figures", "runs/CAM-EXP-001_1_gigahands_bad_view_diagnosis/figures/",
     "bad-view overlays showing the all-zero pattern and hand swaps",
     "A.1 Experiment 1", "KEEP",
     "the visual evidence behind two of Experiment 1's conclusions; pick 2-3, "
     "not the whole set"),
]


def main() -> None:
    rows = [{"artifact": a, "claim_or_metric": b, "source_experiment": c,
             "source_file": d, "source_columns": e, "transformation": f,
             "script": g} for a, b, c, d, e, f, g in ARTIFACTS]
    write_csv(PKG / "source_map" / "report_artifact_sources.csv", rows)
    write_md_table(PKG / "source_map" / "report_artifact_sources.md", rows,
                   title="Where every report-facing number comes from",
                   notes="Paths are relative to `experiments/` unless they start "
                         "with `runs/` or `manifests/`, which are also under "
                         "`experiments/`. Scripts are relative to this package.")

    ap = [{"figure": a, "source": b, "purpose": c,
           "recommended_appendix_section": d, "keep_or_drop": e, "reason": f}
          for a, b, c, d, e, f in APPENDIX]
    write_csv(PKG / "tables" / "appendix_figure_index.csv", ap)
    write_md_table(PKG / "tables" / "appendix_figure_index.md", ap,
                   title="Appendix figure index",
                   notes="Figures already produced by the experiment runs. "
                         "Nothing is copied into this package; the appendix "
                         "should reference these paths directly.")
    missing = [r for r in ap if r["keep_or_drop"] == "KEEP"
               and not (RUNS.parent / r["source"].replace("runs/", "runs/")
                        ).exists()]
    print(f"{len(rows)} artifact rows, {len(ap)} appendix figures "
          f"({sum(1 for r in ap if r['keep_or_drop'] == 'KEEP')} KEEP)")


if __name__ == "__main__":
    main()
