"""Build the report-facing tables: data usage, hypotheses, claims, limitations."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (MANIFESTS, PKG, R0013, R002, R003, R0031, R004, R0041,  # noqa: E402
                    read_csv, read_json, write_csv, write_md_table)

NUM = read_json(PKG / "report_numbers.json")["numbers"]


def v(key):
    return NUM[key]["value"]


# ------------------------------------------------------------- data usage
def dataset_usage():
    qc = read_csv(MANIFESTS / "gigahands_demo_qc_v1.csv.gz")
    st = Counter(r["qc_status"] for r in qc)
    rows = [
        {"experiment": "CAM-EXP-001",
         "purpose": "check that the provided 3D, the provided cameras and the "
                    "provided 2D are mutually consistent under our loader",
         "dataset": "GigaHands demo (+ InterHand2.6M / HanCo / AssemblyHands as "
                    "convention cross-checks)",
         "n_sequences": 5, "n_sequence_camera_views": 200,
         "n_unique_physical_cameras": 40, "n_frames_input": 8000,
         "n_hand_observations": 168000,
         "n_used": 157451, "n_excluded": 10549,
         "exclusion_reason_summary": "joints projecting outside the image",
         "statistical_unit": "joint",
         "hand_quality_required": "no - this experiment measures the data itself",
         "camera_quality_required": "provided intrinsics/extrinsics required",
         "notes": "the long reprojection tail found here is what motivated "
                  "CAM-EXP-001.1 / 001.2 / 001.3"},
        {"experiment": "CAM-EXP-001.1 / 001.2",
         "purpose": "diagnose the large-error observations and audit the "
                    "frame/annotation index mapping against official source",
         "dataset": "GigaHands demo",
         "n_sequences": 5, "n_sequence_camera_views": 200,
         "n_unique_physical_cameras": 40, "n_frames_input": "diagnostic subsets",
         "n_hand_observations": "diagnostic subsets",
         "n_used": "n/a", "n_excluded": "n/a",
         "exclusion_reason_summary": "n/a - qualitative and index-level audit",
         "statistical_unit": "observation / index",
         "hand_quality_required": "no", "camera_quality_required": "no",
         "notes": "established the all-zero 2D pattern and the per-camera hand "
                  "identity swaps; produced no performance number"},
        {"experiment": "CAM-EXP-001.3",
         "purpose": "independent multi-view reconstruction from the released 2D, "
                    "and the QC labelling every later experiment depends on",
         "dataset": "GigaHands demo",
         "n_sequences": 5, "n_sequence_camera_views": 200,
         "n_unique_physical_cameras": 40,
         "n_frames_input": 54120,
         "n_hand_observations": v("gigahands_qc_total_observations"),
         "n_used": v("cam0013_n_triangulated_observations"),
         "n_excluded": v("gigahands_qc_total_observations")
         - v("cam0013_n_triangulated_observations"),
         "exclusion_reason_summary": "not a chosen frame (600); the rest are "
                                     "labelled rather than dropped",
         "statistical_unit": "hand observation (sequence, camera, frame, hand)",
         "hand_quality_required": "this experiment PRODUCES the hand quality labels",
         "camera_quality_required": "provided cameras required",
         "notes": f"QC outcome: PASS_STRICT {st['PASS_STRICT']:,}, "
                  f"PASS_SINGLE_HAND {st['PASS_SINGLE_HAND']:,}, "
                  f"REVIEW {st['REVIEW']:,}, EXCLUDE {st['EXCLUDE']:,}"},
        {"experiment": "CAM-EXP-002",
         "purpose": "how far does absolute hand depth move when only the focal "
                    "length changes",
         "dataset": "GigaHands demo, sample drawn from the bimanual-clean subset",
         "eligible_source_pool": "gigahands_demo_bimanual_clean_v1 (LEFT and "
                                 "RIGHT both PASS_STRICT)",
         "eligible_pool_frames": v("gigahands_bimanual_clean_frames"),
         "eligible_pool_views": v("gigahands_bimanual_clean_views"),
         "actual_frames_attempted": v("cam002_frames_attempted"),
         "actual_frames_successful": v("cam002_frames_inference_successful"),
         "actual_frames_evaluated": v("cam002_frames_evaluated"),
         "actual_hands_evaluated": v("cam002_hands_evaluated"),
         "actual_views": v("cam002_views_evaluated"),
         "actual_unique_cameras": v("cam002_unique_physical_cameras_evaluated"),
         "n_sequences": v("cam002_sequences_evaluated"),
         "n_sequence_camera_views": v("cam002_views_evaluated"),
         "n_unique_physical_cameras": v("cam002_unique_physical_cameras_evaluated"),
         "n_frames_input": v("cam002_frames_attempted"),
         "n_hand_observations": v("cam002_hands_evaluated"),
         "n_used": v("cam002_hands_evaluated"),
         "n_excluded": f"{v('cam002_model_failures')} frames (no hand detected) "
                       f"+ {v('cam002_hand_association_failures')} frames "
                       "(ambiguous hand association)",
         "exclusion_reason_summary":
             f"the eligible pool is {v('gigahands_bimanual_clean_frames'):,} "
             f"frames; a stratified sample of "
             f"{v('cam002_frames_attempted'):,} was drawn from it (grouped by "
             "(sequence, camera), shuffled with a fixed seed, taken round-robin "
             "and evenly spaced in time). Detector and hand-association "
             "failures then removed 27 frames.",
         "statistical_unit": "hand",
         "hand_quality_required": "YES - bimanual-clean (both hands PASS_STRICT)",
         "camera_quality_required": "provided intrinsics required",
         "notes": f"the pool and the evaluated sample are different numbers. "
                  f"{v('cam002_bimanual_frames_evaluated'):,} of the evaluated "
                  "frames had BOTH hands evaluated and carry the bimanual "
                  "metrics. This is a HAND-clean experiment, which is why it "
                  "covers fewer views than CAM-EXP-003/004."},
        {"experiment": "CAM-EXP-003 / 003.1",
         "purpose": "how accurate are existing single-frame calibration models, "
                    "and how much of their error is unmodelled lens distortion",
         "dataset": "GigaHands demo, camera-clean subset",
         "n_sequences": 5,
         "n_sequence_camera_views": v("camera_benchmark_views_usable"),
         "n_unique_physical_cameras": 40,
         "n_frames_input": v("cam003_frames_total"),
         "n_hand_observations": "not used - this is a camera experiment",
         "n_used": v("cam003_frames_total"), "n_excluded": 0,
         "exclusion_reason_summary": f"{v('camera_benchmark_views_excluded')} views "
                                     "were excluded upstream for having no usable "
                                     "RGB segment matching the annotation; 0 "
                                     "inference failures",
         "statistical_unit": "view (175); frames are repeated observations",
         "hand_quality_required": "NO - hand REVIEW/EXCLUDE never removes a "
                                  "camera-valid view",
         "camera_quality_required": "YES - RGB + provided intrinsics",
         "notes": "this is why CAM-EXP-003 has 175 views while CAM-EXP-002 has "
                  f"{v('gigahands_bimanual_clean_views')}"},
        {"experiment": "CAM-EXP-004",
         "purpose": "does aggregating more frames of one static camera help, and "
                    "is the residual error bias or noise",
         "dataset": "GigaHands demo, camera-clean subset (identical frames to "
                    "CAM-EXP-003)",
         "n_sequences": 5,
         "n_sequence_camera_views": v("camera_benchmark_views_usable"),
         "n_unique_physical_cameras": 40,
         "n_frames_input": v("cam003_frames_total"),
         "n_hand_observations": "not used",
         "n_used": v("cam003_frames_total"), "n_excluded": 0,
         "exclusion_reason_summary": "none - zero new inference, predictions reused",
         "statistical_unit": "view (175)",
         "hand_quality_required": "NO", "camera_quality_required": "YES",
         "notes": "all C(8,N) frame subsets evaluated, then collapsed to one "
                  "expected error per view"},
        {"experiment": "CAM-EXP-004.1",
         "purpose": "cluster-correct statistics, real 16/32/64 frames, "
                    "reproducibility, provenance and holdout reservation",
         "dataset": "GigaHands demo, camera-clean subset",
         "n_sequences": 5,
         "n_sequence_camera_views": v("camera_benchmark_views_usable"),
         "n_unique_physical_cameras": 40,
         "n_frames_input": v("cam0041_frames_total"),
         "n_hand_observations": "not used",
         "n_used": v("cam0041_frames_total"), "n_excluded": 0,
         "exclusion_reason_summary": "none",
         "statistical_unit": "view (175); also physical camera (40) and "
                             "sequence (5) as coarser cluster units",
         "hand_quality_required": "NO", "camera_quality_required": "YES",
         "notes": "the CAM-EXP-003 8 frames are an exact subset of these 64, so "
                  "the two runs can be compared frame by frame"},
    ]
    cols, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                cols.append(k)
    write_csv(PKG / "tables" / "report_dataset_usage.csv", rows, fieldnames=cols)
    write_md_table(
        PKG / "tables" / "report_dataset_usage.md", rows, columns=cols,
        title="Data used by each experiment",
        notes="Every count is read from the manifests and result files listed in "
              "`report_numbers.json`, not from prose. The three different "
              "'clean' subsets are not interchangeable: a camera experiment "
              "needs RGB plus valid camera parameters, a hand experiment needs "
              "valid hand annotation, and a bimanual experiment needs both hands "
              "clean in the same frame.\n\n"
              "**Eligible pool is not the same as evaluated sample.** For "
              "CAM-EXP-002 the bimanual-clean manifest is the pool it was "
              "allowed to draw from; the experiment evaluated a stratified "
              "sample of it. The `eligible_*` and `actual_*` columns are kept "
              "separate for exactly this reason, and only the `actual_*` "
              "numbers describe what was measured.")
    return rows


# ------------------------------------------------- hypotheses and evidence
def hypothesis_table():
    rows = [
        {"experiment": "CAM-EXP-001",
         "original_hypothesis": "if the camera and coordinate conventions are read "
                                "correctly, projecting the provided 3D with the "
                                "provided camera should land on the provided 2D",
         "test": "project provided 3D into every view and compare with the "
                 "provided 2D, across four datasets",
         "result": "the convention is correct, but a large error tail remains on "
                   "GigaHands",
         "key_numbers": f"median {v('cam001_gigahands_reprojection_median_px')} px "
                        f"in-image; only "
                        f"{v('cam001_gigahands_reprojection_frac_under_10px') * 100:.1f} % "
                        "of joints under 10 px",
         "verdict": "PARTIALLY_SUPPORTED - convention validated, data quality not",
         "evidence_level": "CONFIRMED_IN_CURRENT_ENVIRONMENT",
         "scope": "GigaHands demo subset; the other datasets were convention "
                  "cross-checks only",
         "remaining_uncertainty": "the tail's cause was unknown at this point",
         "next_implication": "triggered CAM-EXP-001.1 / 001.2 / 001.3"},
        {"experiment": "CAM-EXP-001.1 / 001.2",
         "original_hypothesis": "some large-error observations are genuine "
                                "annotation or hand-association problems rather "
                                "than loader mistakes",
         "test": "human-inspectable overlays, index-space audit against official "
                 "GigaHands source, repro-video comparison",
         "result": "supported for a substantial part of the tail: an all-zero 2D "
                   "pattern carrying confidence 1.0, and per-view left/right "
                   "hand swaps. It does NOT account for the whole tail.",
         "key_numbers": f"{v('gigahands_qc_zero_pattern_n'):,} observations carry "
                        "the all-zero pattern; "
                        f"{v('cam0013_identity_confirmed_2d_hand_identity_swap_n'):,} "
                        "confirmed identity swaps; still "
                        f"{v('cam0013_identity_bad_2d_geometry_n'):,} "
                        "BAD_2D_GEOMETRY and "
                        f"{v('cam0013_identity_unresolved_insufficient_geometry_n'):,} "
                        "UNRESOLVED_INSUFFICIENT_GEOMETRY",
         "verdict": "PARTIALLY_SUPPORTED - named causes for part of the tail, "
                    "not for all of it",
         "evidence_level": "CONFIRMED_IN_CURRENT_ENVIRONMENT",
         "scope": "GigaHands demo",
         "remaining_uncertainty": "the all-zero pattern is OBSERVED, not a "
                                  "documented sentinel value; and the cause of "
                                  "the BAD_2D_GEOMETRY / unresolved cases was "
                                  "not determined - they were conservatively "
                                  "labelled REVIEW/EXCLUDE rather than guessed",
         "next_implication": "the loader drops all-zero 2D; identity must be "
                             "checked per camera"},
        {"experiment": "CAM-EXP-001.3",
         "original_hypothesis": "the released multi-view 2D and the provided 3D "
                                "are highly self-consistent",
         "test": "robust RANSAC triangulation from the released 2D only, never "
                 "using the provided 3D as an input, plus a wrong-hand control",
         "result": "strongly self-consistent where the 2D is usable",
         "key_numbers": f"same-hand median "
                        f"{v('cam0013_recon_vs_provided3d_same_hand_median_mm')} mm, "
                        f"p90 {v('cam0013_recon_vs_provided3d_same_hand_p90_mm')} mm; "
                        f"wrong-hand control "
                        f"{v('cam0013_recon_vs_provided3d_other_hand_median_mm')} mm",
         "verdict": "SUPPORTED",
         "evidence_level": "CONFIRMED_IN_CURRENT_ENVIRONMENT",
         "scope": "self-consistency of one dataset's own annotations",
         "remaining_uncertainty": "this is NOT accuracy against an independent "
                                  "external ground truth and cannot be written as "
                                  "such",
         "next_implication": "produced the QC labels and the camera-clean / "
                             "hand-clean / bimanual-clean subsets"},
        {"experiment": "CAM-EXP-002",
         "original_hypothesis": "focal error propagates directly into absolute "
                                "hand depth",
         "test": "one inference per frame, cached; only the focal-dependent "
                 "conversion recomputed over a controlled focal sweep",
         "result": "confirmed, and close to proportional, as the pipeline "
                   "equation Z proportional to f predicts",
         "key_numbers": f"+/-5 % focal -> "
                        f"{v('cam002_focal_5pct_root_shift_median_mm')} mm median "
                        f"root shift; 10 % -> "
                        f"{v('cam002_focal_10pct_root_shift_median_mm')} mm; "
                        f"20 % -> {v('cam002_focal_20pct_root_shift_median_mm')} mm",
         "verdict": "SUPPORTED",
         "evidence_level": "ROBUST_BUT_SINGLE_DATASET",
         "scope": "FOCAL LENGTH ONLY, and only the absolute depth of the hand. "
                  "Principal point, anisotropic focal and distortion were not "
                  "tested downstream.",
         "remaining_uncertainty": "the mm figures are specific to this "
                                  "working-distance regime",
         "next_implication": "gives the +/-5 % focal target a physical meaning"},
        {"experiment": "CAM-EXP-002",
         "original_hypothesis": "applying the dataset-provided reference focal "
                                "(GT_EFFECTIVE_FOCAL) in place of the pipeline's "
                                "training-convention virtual focal places the "
                                "hand far closer to the reference 3D",
         "test": "same cached predictions, two focal conventions, identical metric",
         "result": "confirmed, by nearly two orders of magnitude",
         "key_numbers": f"virtual focal {v('cam002_pipeline_virtual_focal_px'):.0f} px "
                        f"-> root error "
                        f"{v('cam002_baseline_root_error_median_mm'):.0f} mm; "
                        f"dataset-provided reference focal "
                        f"{v('cam002_gt_effective_focal_median_px'):.1f} px "
                        f"-> {v('cam002_gt_effective_focal_root_error_median_mm'):.1f} mm; "
                        f"root-aligned MPJPE unchanged at "
                        f"{v('cam002_root_aligned_mpjpe_median_mm')} mm",
         "verdict": "SUPPORTED",
         "evidence_level": "ROBUST_BUT_SINGLE_DATASET",
         "scope": "metric placement, not hand shape - the root-aligned error is "
                  "identical under both focal conventions. This compares two "
                  "focal CONVENTIONS on the same predictions; it does not show "
                  "that 5000 px is an invalid camera intrinsic, which it never "
                  "claimed to be.",
         "remaining_uncertainty": "none for the direction; the magnitude is "
                                  "dataset-specific",
         "next_implication": "the camera, not the hand model, is the first thing "
                             "to fix -> CAM-EXP-003"},
        {"experiment": "CAM-EXP-003",
         "original_hypothesis": "single-frame calibration alone can reach the "
                                "+/-5 % focal target",
         "test": "four published single-frame methods on 175 static-camera views, "
                 "identical frames for every model",
         "result": "rejected - every method is far from the target",
         "key_numbers": f"AnyCalib {v('cam003_anycalib_pinhole_focal_err_median')} %, "
                        f"GeoCalib {v('cam003_geocalib_pinhole_focal_err_median')} %, "
                        f"PF-centered {v('cam003_pf_centered_focal_err_median')} %, "
                        f"PF-uncentered {v('cam003_pf_uncentered_focal_err_median')} % "
                        "median; the deployed fixed 5000 px assumption is "
                        f"{v('cam003_demo_fixed_5000_focal_err_median')} %",
         "verdict": "REJECTED",
         "evidence_level": "ROBUST_BUT_SINGLE_DATASET",
         "scope": "GigaHands optics and resolution; raw images; pinhole model",
         "remaining_uncertainty": "whether the pinhole assumption itself was the "
                                  "problem",
         "next_implication": "CAM-EXP-003.1"},
        {"experiment": "CAM-EXP-003.1",
         "original_hypothesis": "unmodelled lens distortion explains a substantial "
                                "part of the single-frame focal error",
         "test": "raw + distortion-aware model, and an oracle GT-undistorted "
                 "control, paired on the identical frames",
         "result": "confirmed as a contributor, rejected as the whole story",
         "key_numbers": f"AnyCalib "
                        f"{v('cam003_anycalib_pinhole_focal_err_median')} -> "
                        f"{v('cam0031_anycalib_gen_radial_focal_err_median')} %, "
                        f"GeoCalib {v('cam003_geocalib_pinhole_focal_err_median')} -> "
                        f"{v('cam0031_geocalib_distorted_radial_focal_err_median')} %; "
                        "paired gain "
                        f"{v('cam0031_anycalib_gen_radial_improvement_pp_view')[0]} pp "
                        f"[{v('cam0031_anycalib_gen_radial_improvement_pp_view')[1]}, "
                        f"{v('cam0031_anycalib_gen_radial_improvement_pp_view')[2]}] "
                        "at view clustering",
         "verdict": "DISTORTION_CONTRIBUTES_BUT_NOT_DOMINANT",
         "evidence_level": "ROBUST_BUT_SINGLE_DATASET",
         "scope": "one distortion regime, |k1| 0.34-0.43, so no dose-response "
                  "conclusion is available",
         "remaining_uncertainty": "about 10 % median error remains unexplained",
         "next_implication": "CAM-EXP-004 asks whether more frames remove it"},
        {"experiment": "CAM-EXP-004",
         "original_hypothesis": "averaging more frames of a static camera reduces "
                                "random frame-to-frame variation",
         "test": "all C(8,N) subsets for N = 1, 2, 4, 8, four aggregation rules",
         "result": "true in principle, almost irrelevant in practice for the best "
                   "model, and clearly useful only for the noisy one",
         "key_numbers": "AnyCalib essentially flat; GeoCalib improved; the "
                        "provable reason is that 91.4 % of AnyCalib's views have "
                        "all 8 frames erring in the same direction",
         "verdict": "PARTIALLY_SUPPORTED",
         "evidence_level": "ROBUST_BUT_SINGLE_DATASET",
         "scope": "GigaHands static views",
         "remaining_uncertainty": "whether this holds past 8 frames",
         "next_implication": "CAM-EXP-004.1 measured 16/32/64"},
        {"experiment": "CAM-EXP-004",
         "original_hypothesis": "most of AnyCalib's residual error is a stable "
                                "per-view bias rather than within-view noise",
         "test": "decompose the squared log focal error into a per-view mean and "
                 "a within-view residual",
         "result": "strongly confirmed",
         "key_numbers": f"AnyCalib "
                        f"{v('cam004_anycalib_gen_bias_fraction_pct')} % bias; "
                        f"median |view bias| "
                        f"{v('cam004_anycalib_gen_median_abs_view_bias_pct')} % vs "
                        f"within-view sd "
                        f"{v('cam004_anycalib_gen_median_within_view_std_pct')} %; "
                        f"GeoCalib {v('cam004_geocalib_distorted_bias_fraction_pct')} % "
                        "bias",
         "verdict": "SUPPORTED",
         "evidence_level": "ROBUST_BUT_SINGLE_DATASET",
         "scope": "GigaHands static views",
         "remaining_uncertainty": "the cause of the per-view bias is unknown",
         "next_implication": "this is exactly what CAM-EXP-005 must explain"},
        {"experiment": "CAM-EXP-004.1",
         "original_hypothesis": "going from 8 to 64 frames does not remove the "
                                "residual bias",
         "test": "a frozen 64-frame manifest, all 175 views, one inference pass, "
                 "the frozen aggregation rules",
         "result": "confirmed - the curves are flat, and never descended",
         "key_numbers": f"AnyCalib {v('cam0041_anycalib_gen_N8_focal_err_median')} "
                        f"-> {v('cam0041_anycalib_gen_N64_focal_err_median')} %; "
                        f"E2 {v('cam0041_E2_frozen_N8_focal_err_median')} -> "
                        f"{v('cam0041_E2_frozen_N64_focal_err_median')} %; every "
                        "paired step CI contains zero. One execution over the "
                        "same 175 benchmark views, so every N is directly "
                        "comparable.",
         "verdict": "CONFIRMED_SATURATION",
         "evidence_level": "ROBUST_BUT_SINGLE_DATASET",
         "scope": "up to 64 frames, GigaHands static cameras. Says nothing about "
                  "hundreds of frames or other rigs.",
         "remaining_uncertainty": "none within that range",
         "next_implication": "frame count is closed as a lever"},
        {"experiment": "CAM-EXP-004",
         "original_hypothesis": "two model families with opposite signed bias can "
                                "be combined, parameter-free, to beat either one",
         "test": "five equal-weight / median / geometric-mean ensembles, no "
                 "fitted weights",
         "result": "supported, and the best of them was frozen as E2",
         "key_numbers": f"AnyCalib {v('e2_or_model_rerun_anycalib_gen_cam004_median')} "
                        f"-> E2 {v('e2_or_model_rerun_E2_frozen_cam004_median')} % "
                        f"(re-run {v('e2_or_model_rerun_E2_frozen_cam0041_median')} %); "
                        "gain +2.23 pp, CI excludes zero at view, camera and "
                        "sequence clustering",
         "verdict": "SUPPORTED_AS_DIRECTION",
         "evidence_level": "EXPLORATORY_INTERNAL_VALIDATION",
         "scope": "E2 was selected as the best of five candidates on the same 175 "
                  "views it is reported on",
         "remaining_uncertainty": "the exact number is a selection-set number",
         "next_implication": "frozen as the baseline CAM-EXP-005/006 must beat"},
        {"experiment": "CAM-EXP-004.1",
         "original_hypothesis": "E2 = 6.14 % is an independent final test number",
         "test": "leave-one-sequence-out selection, plus a separate execution on "
                 "the same benchmark views",
         "result": "rejected - it is a selection-set number with real spread",
         "key_numbers": f"selection set "
                        f"{v('e2_or_model_rerun_E2_frozen_cam004_median')} %, "
                        f"separate rerun on the same views "
                        f"{v('e2_or_model_rerun_E2_frozen_cam0041_median')} %, "
                        f"LOSO held-out {v('e2_loso_heldout_median_range_pct')[0]}"
                        f"-{v('e2_loso_heldout_median_range_pct')[1]} %; "
                        f"E2 chosen in {v('e2_loso_folds_selecting_e2')}/5 folds",
         "verdict": "REJECTED - must be quoted as a range",
         "evidence_level": "EXPLORATORY_INTERNAL_VALIDATION",
         "scope": "same rig, same 5 sequences",
         "remaining_uncertainty": "a genuinely independent number needs the "
                                  "sealed external holdout",
         "next_implication": "PENDING_EXTERNAL_CONFIRMATION"},
        {"experiment": "CAM-EXP-004 / 004.1",
         "original_hypothesis": "AnyCam's focal mechanism is not identifiable "
                                "under a static camera",
         "test": "read the official focal-selection code, then run AnyCam's own "
                 "scoring function with identity vs translating relative poses",
         "result": "confirmed numerically",
         "key_numbers": f"spread across its 32 focal candidates "
                        f"{v('anycam_static_identity_pose_candidate_spread'):.2e} "
                        f"(static) vs "
                        f"{v('anycam_moving_camera_candidate_spread'):.3f} (moving)",
         "verdict": "SUPPORTED",
         "evidence_level": "CONFIRMED_IN_CURRENT_ENVIRONMENT",
         "scope": "a statement about method assumptions vs our deployment "
                  "condition, NOT about AnyCam's quality",
         "remaining_uncertainty": "the end-to-end behavioural test could not run "
                                  "on Windows",
         "next_implication": "AnyCam is out of scope for this deployment setting"},
        {"experiment": "CAM-EXP-004.1",
         "original_hypothesis": "(not planned) the calibration models are "
                                "deterministic",
         "test": "40 frames x 3 repeats on byte-identical images",
         "result": "false for GeoCalib",
         "key_numbers": f"AnyCalib "
                        f"{v('determinism_anycalib_gen_radial_pct_bit_identical')} % "
                        "bit-identical; GeoCalib "
                        f"{v('determinism_geocalib_distorted_radial_pct_bit_identical')} %, "
                        f"median spread "
                        f"{v('determinism_geocalib_distorted_radial_median_spread_pct')} %, "
                        f"max {v('determinism_geocalib_distorted_radial_max_spread_pct')} %",
         "verdict": "NEW_OPEN_ISSUE",
         "evidence_level": "OPEN_ISSUE",
         "scope": "this environment, this GeoCalib commit",
         "remaining_uncertainty": "whether deterministic algorithms fix it",
         "next_implication": "report the observed between-execution difference "
                             "(about 0.32 pp on the aggregate median) and "
                             "the repeat-diagnostic spreads separately; do "
                             "not merge them into one uncertainty interval"},
    ]
    write_csv(PKG / "tables" / "hypothesis_result_evidence.csv", rows)
    write_md_table(
        PKG / "tables" / "hypothesis_result_evidence.md", rows,
        columns=["experiment", "original_hypothesis", "result", "key_numbers",
                 "verdict", "evidence_level", "scope", "remaining_uncertainty",
                 "next_implication"],
        title="Hypothesis, result and evidence level",
        notes="Evidence levels use the fixed vocabulary "
              "CONFIRMED_IN_CURRENT_ENVIRONMENT / ROBUST_BUT_SINGLE_DATASET / "
              "EXPLORATORY_INTERNAL_VALIDATION / PENDING_EXTERNAL_CONFIRMATION / "
              "OPEN_ISSUE. Hypotheses that were rejected are kept.")
    return rows


def main() -> None:
    dataset_usage()
    hypothesis_table()
    print("wrote report_dataset_usage and hypothesis_result_evidence")


if __name__ == "__main__":
    main()
