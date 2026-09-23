"""Claim ledger, main results, limitations, future work, evaluation principles."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PKG, read_json, write_csv, write_md_table  # noqa: E402

NUM = read_json(PKG / "report_numbers.json")["numbers"]


def v(k):
    return NUM[k]["value"]


# ------------------------------------------------------------ claim ledger
def claim_ledger():
    rows = [
        {"claim_id": "C01",
         "claim_we_may_write": "The deployed pipeline assumes a fixed virtual "
                               "focal length that is far from the physical focal "
                               "length of the capture cameras, and this alone "
                               "displaces the hand by metres.",
         "supporting_experiment": "CAM-EXP-002",
         "supporting_numbers": f"{v('cam002_pipeline_virtual_focal_px'):.0f} px vs "
                               f"{v('cam002_gigahands_physical_focal_median_px'):.1f} px; "
                               f"root error "
                               f"{v('cam002_baseline_root_error_median_mm'):.0f} mm vs "
                               f"{v('cam002_physical_focal_root_error_median_mm'):.1f} mm",
         "evidence_level": "ROBUST_BUT_SINGLE_DATASET",
         "allowed_wording": "본 GigaHands 환경에서 파이프라인이 가정한 가상 초점거리는 "
                            "실제 카메라 초점거리와 크게 달랐고, 이로 인해 절대 위치가 "
                            "미터 단위로 이동했다.",
         "wording_to_avoid": "The pipeline is broken / all monocular hand pipelines "
                             "have metre-scale error.",
         "scope": "one dataset, one working-distance regime",
         "external_confirmation_needed": "yes, for the magnitude",
         "report_section": "5 (Experiment 2)"},
        {"claim_id": "C02",
         "claim_we_may_write": "Focal error propagates almost proportionally into "
                               "absolute hand depth, while leaving the hand's own "
                               "shape untouched.",
         "supporting_experiment": "CAM-EXP-002",
         "supporting_numbers": f"5 % -> {v('cam002_focal_5pct_root_shift_median_mm')} mm, "
                               f"10 % -> {v('cam002_focal_10pct_root_shift_median_mm')} mm, "
                               f"20 % -> {v('cam002_focal_20pct_root_shift_median_mm')} mm; "
                               f"root-aligned MPJPE unchanged at "
                               f"{v('cam002_root_aligned_mpjpe_median_mm')} mm",
         "evidence_level": "ROBUST_BUT_SINGLE_DATASET",
         "allowed_wording": "focal-length sensitivity of absolute hand depth; the "
                            "near-linear relation is expected from the pipeline "
                            "equation Z proportional to f.",
         "wording_to_avoid": "full camera calibration sensitivity; we discovered "
                             "that depth scales with focal length.",
         "scope": "FOCAL ONLY. Principal point, anisotropic focal and distortion "
                  "were not tested downstream.",
         "external_confirmation_needed": "no for the direction, yes for the mm",
         "report_section": "5 (Experiment 2)"},
        {"claim_id": "C03",
         "claim_we_may_write": "Reconstructing 3D from the released multi-view 2D "
                               "agrees with the provided 3D to about 4 mm, while a "
                               "wrong-hand control is ~48x worse.",
         "supporting_experiment": "CAM-EXP-001.3",
         "supporting_numbers": f"same-hand median "
                               f"{v('cam0013_recon_vs_provided3d_same_hand_median_mm')} mm, "
                               f"p90 {v('cam0013_recon_vs_provided3d_same_hand_p90_mm')} mm; "
                               f"other-hand "
                               f"{v('cam0013_recon_vs_provided3d_other_hand_median_mm')} mm",
         "evidence_level": "CONFIRMED_IN_CURRENT_ENVIRONMENT",
         "allowed_wording": "released 2D 관측으로 재구성한 3D와 제공된 3D는 약 4 mm "
                            "수준의 자기일관성(self-consistency)을 보였다.",
         "wording_to_avoid": "GigaHands 3D는 실제 손 위치에서 4 mm 정확하다 / "
                             "we validated GigaHands against ground truth.",
         "scope": "internal self-consistency of one dataset's own annotations",
         "external_confirmation_needed": "not applicable - this is a consistency "
                                         "claim, not an accuracy claim",
         "report_section": "4 (Experiment 1)"},
        {"claim_id": "C04",
         "claim_we_may_write": "Published single-frame calibration methods do not "
                               "reach the +/-5 % focal target on this data.",
         "supporting_experiment": "CAM-EXP-003",
         "supporting_numbers": f"median focal error "
                               f"{v('cam003_anycalib_pinhole_focal_err_median')} / "
                               f"{v('cam003_geocalib_pinhole_focal_err_median')} / "
                               f"{v('cam003_pf_centered_focal_err_median')} / "
                               f"{v('cam003_pf_uncentered_focal_err_median')} %, "
                               "175 views",
         "evidence_level": "ROBUST_BUT_SINGLE_DATASET",
         "allowed_wording": "본 평가 조건에서 단일 프레임 방법만으로는 목표 정확도에 "
                            "도달하지 못했다.",
         "wording_to_avoid": "AnyCalib/GeoCalib are inaccurate methods; these "
                             "models fail.",
         "scope": "GigaHands optics, 1280x720, raw images, pinhole camera model",
         "external_confirmation_needed": "yes",
         "report_section": "6 (Experiment 3)"},
        {"claim_id": "C05",
         "claim_we_may_write": "Accounting for lens distortion roughly halves the "
                               "single-frame focal error, but does not close the "
                               "gap.",
         "supporting_experiment": "CAM-EXP-003.1, re-analysed in CAM-EXP-004.1",
         "supporting_numbers": f"AnyCalib {v('cam003_anycalib_pinhole_focal_err_median')} "
                               f"-> {v('cam0031_anycalib_gen_radial_focal_err_median')} %; "
                               f"paired gain "
                               f"{v('cam0031_anycalib_gen_radial_improvement_pp_view')[0]} pp "
                               f"[{v('cam0031_anycalib_gen_radial_improvement_pp_view')[1]}, "
                               f"{v('cam0031_anycalib_gen_radial_improvement_pp_view')[2]}] "
                               "with views as the resampling unit",
         "evidence_level": "ROBUST_BUT_SINGLE_DATASET",
         "allowed_wording": "distortion-aware 모델이 pinhole 모델보다 일관되게 우수했고, "
                            "view/카메라/시퀀스 어느 군집 수준에서도 신뢰구간이 0을 "
                            "포함하지 않았다.",
         "wording_to_avoid": "distortion explains the error; quoting the frame-level "
                             "CI as if 1400 frames were independent.",
         "scope": "|k1| 0.34-0.43 only, so no dose-response conclusion",
         "external_confirmation_needed": "yes",
         "report_section": "6 (Experiment 3)"},
        {"claim_id": "C06",
         "claim_we_may_write": "The residual focal error of the best model is "
                               "dominated by a stable between-view component - a "
                               "per-(sequence, camera) static-view bias - rather "
                               "than by frame-to-frame noise.",
         "supporting_experiment": "CAM-EXP-004",
         "supporting_numbers": f"{v('cam004_anycalib_gen_bias_fraction_pct')} % of "
                               "AnyCalib's squared log focal error is the "
                               "between-view component; median |view bias| "
                               f"{v('cam004_anycalib_gen_median_abs_view_bias_pct')} % "
                               f"vs within-view sd "
                               f"{v('cam004_anycalib_gen_median_within_view_std_pct')} %",
         "evidence_level": "ROBUST_BUT_SINGLE_DATASET",
         "allowed_wording": "AnyCalib 잔여 오차는 동일 고정 관점 내 프레임 노이즈보다 "
                            "(sequence, camera) 관점별 안정적 편향에 의해 "
                            "지배되었다. 분해 단위는 고정 관점이며, "
                            "physical camera 단위의 편향을 측정한 것이 아니다.",
         "wording_to_avoid": "98.1 %의 오차가 physical-camera bias 때문이다 / "
                             "모든 고정카메라 calibration 모델의 오류는 view bias "
                             "때문이다.",
         "scope": "one rig; the decomposition unit is the (sequence, camera) "
                  "static view, NOT the physical camera. Physical camera is a "
                  "resampling unit used for robustness in CAM-EXP-004.1.",
         "external_confirmation_needed": "yes",
         "report_section": "7 (Experiment 4)"},
        {"claim_id": "C07",
         "claim_we_may_write": "Within this setting, aggregating up to 64 frames "
                               "of a static camera does not improve focal accuracy.",
         "supporting_experiment": "CAM-EXP-004 + CAM-EXP-004.1",
         "supporting_numbers": f"AnyCalib {v('cam0041_anycalib_gen_N8_focal_err_median')} "
                               f"-> {v('cam0041_anycalib_gen_N64_focal_err_median')} %; "
                               f"E2 {v('cam0041_E2_frozen_N8_focal_err_median')} -> "
                               f"{v('cam0041_E2_frozen_N64_focal_err_median')} %; "
                               "every paired step CI contains zero",
         "evidence_level": "ROBUST_BUT_SINGLE_DATASET",
         "allowed_wording": "within the evaluated GigaHands static-camera "
                            "condition, up to 64 frames, 프레임 수를 늘리는 것은 "
                            "초점거리 정확도를 개선하지 못했다.",
         "wording_to_avoid": "multi-frame aggregation never helps; 8 frames are "
                             "always enough.",
         "scope": "N <= 64, GigaHands static cameras",
         "external_confirmation_needed": "yes",
         "report_section": "7 (Experiment 4)"},
        {"claim_id": "C08",
         "claim_we_may_write": "Combining two calibration model families with "
                               "opposite signed bias is the most promising "
                               "direction found so far.",
         "supporting_experiment": "CAM-EXP-004, audited in CAM-EXP-004.1",
         "supporting_numbers": f"AnyCalib "
                               f"{v('e2_or_model_rerun_anycalib_gen_cam004_median')} % "
                               f"-> E2 {v('e2_or_model_rerun_E2_frozen_cam004_median')} % "
                               f"(re-run {v('e2_or_model_rerun_E2_frozen_cam0041_median')} %, "
                               f"LOSO {v('e2_loso_heldout_median_range_pct')[0]}-"
                               f"{v('e2_loso_heldout_median_range_pct')[1]} %); gain "
                               "+2.23 pp, CI excludes zero at all cluster levels",
         "evidence_level": "EXPLORATORY_INTERNAL_VALIDATION",
         "allowed_wording": "frozen exploratory ensemble / 현재까지 확인된 가장 "
                            "유망한 baseline / strongest baseline identified so far.",
         "wording_to_avoid": "our final method; our proposed method; E2 achieves "
                             "6.14 %.",
         "scope": "selected as the best of five candidates on the same 175 views "
                  "it is reported on",
         "external_confirmation_needed": "YES - this is the number the sealed "
                                         "holdout exists for",
         "report_section": "7 and 8"},
        {"claim_id": "C09",
         "claim_we_may_write": "AnyCam's official focal-selection mechanism is not "
                               "identifiable under a static camera.",
         "supporting_experiment": "CAM-EXP-004 (source), CAM-EXP-004.1 (numeric)",
         "supporting_numbers": f"spread across its own 32 focal candidates "
                               f"{v('anycam_static_identity_pose_candidate_spread'):.2e} "
                               f"static vs "
                               f"{v('anycam_moving_camera_candidate_spread'):.3f} moving",
         "evidence_level": "CONFIRMED_IN_CURRENT_ENVIRONMENT",
         "allowed_wording": "AnyCam의 공식 focal-selection mechanism은 "
                            "camera-induced motion을 이용하므로, 본 연구의 "
                            "static-camera condition에서는 focal candidates가 "
                            "식별되지 않았다.",
         "wording_to_avoid": "AnyCam performed poorly; AnyCam is a bad method; we "
                             "benchmarked AnyCam.",
         "scope": "method-assumption mismatch with our deployment condition",
         "external_confirmation_needed": "no; but the end-to-end behavioural test "
                                         "is still missing (Windows blocker)",
         "report_section": "6 (small subsection on external video methods)"},
        {"claim_id": "C10",
         "claim_we_may_write": "GeoCalib's output varies between runs on the same "
                               "image, but the aggregate conclusions are unchanged.",
         "supporting_experiment": "CAM-EXP-004.1",
         "supporting_numbers": f"median spread "
                               f"{v('determinism_geocalib_distorted_radial_median_spread_pct')} %, "
                               f"p90 {v('determinism_geocalib_distorted_radial_p90_spread_pct')} %, "
                               f"max {v('determinism_geocalib_distorted_radial_max_spread_pct')} %; "
                               "aggregate median shift ~0.32 pp; AnyCalib "
                               f"{v('determinism_anycalib_gen_radial_pct_bit_identical')} % "
                               "bit-identical",
         "evidence_level": "OPEN_ISSUE",
         "allowed_wording": "GeoCalib은 동일 입력에서도 실행 간 변동이 관찰되었으나, "
                            "전체 중앙값 변화는 약 0.32 %p로 주요 결론을 변경하지 "
                            "않았다.",
         "wording_to_avoid": "omitting it; or presenting any single GeoCalib "
                             "number without the tolerance.",
         "scope": "this environment and this GeoCalib commit",
         "external_confirmation_needed": "no - it needs a fix attempt, not "
                                         "confirmation",
         "report_section": "9 (Limitations), details in the appendix"},
        {"claim_id": "C11",
         "claim_we_may_write": "Different experiments use different subsets "
                               "because a camera experiment and a hand experiment "
                               "need different things to be valid.",
         "supporting_experiment": "CAM-EXP-001.3 QC + CAM-EXP-003 camera manifest",
         "supporting_numbers": f"camera-clean {v('camera_benchmark_views_usable')} "
                               f"views; bimanual-clean "
                               f"{v('gigahands_bimanual_clean_views')} views / "
                               f"{v('gigahands_bimanual_clean_frames'):,} frames",
         "evidence_level": "CONFIRMED_IN_CURRENT_ENVIRONMENT",
         "allowed_wording": "camera-clean / hand-clean / bimanual-clean subsets "
                            "are different by design.",
         "wording_to_avoid": "implying the datasets shrank because of failures, or "
                             "that hand QC removed camera views.",
         "scope": "GigaHands demo",
         "external_confirmation_needed": "no",
         "report_section": "3 (Datasets and evaluation environment)"},
        {"claim_id": "C12",
         "claim_we_may_write": "Our statistical unit is the static camera view, "
                               "not the frame.",
         "supporting_experiment": "CAM-EXP-004.1",
         "supporting_numbers": "the same point estimate with CIs ~2.5x wider under "
                               "view clustering; e.g. +6.54 pp [+5.75, +7.30] "
                               "frame-level vs [+4.57, +8.49] view-level",
         "evidence_level": "CONFIRMED_IN_CURRENT_ENVIRONMENT",
         "allowed_wording": "175 fixed-camera views, 8 frames per view, 1400 "
                            "frames in total; statistical unit = view.",
         "wording_to_avoid": "n = 1400; 1400 independent samples.",
         "scope": "all camera experiments",
         "external_confirmation_needed": "no",
         "report_section": "3 and every results section"},
    ]
    write_csv(PKG / "tables" / "claim_evidence_ledger.csv", rows)
    write_md_table(
        PKG / "tables" / "claim_evidence_ledger.md", rows,
        columns=["claim_id", "claim_we_may_write", "supporting_experiment",
                 "supporting_numbers", "evidence_level", "allowed_wording",
                 "wording_to_avoid", "scope", "external_confirmation_needed",
                 "report_section"],
        title="Claim ledger: what we may write, and how",
        notes="One row per claim the progress report is allowed to make. If a "
              "sentence in the report is not covered by a row here, it is not "
              "yet supported and should not be written.")


# ------------------------------------------------------------ main results
def main_results():
    rows = [
        {"report_section": "4 - Experiment 1", "experiment": "CAM-EXP-001.3",
         "quantity": "reconstruction vs provided 3D, same hand (self-consistency)",
         "value": f"{v('cam0013_recon_vs_provided3d_same_hand_median_mm')} mm median, "
                  f"{v('cam0013_recon_vs_provided3d_same_hand_p90_mm')} mm p90",
         "n": f"{v('cam0013_n_triangulated_observations'):,} hand observations",
         "statistical_unit": "hand observation",
         "evidence_level": "CONFIRMED_IN_CURRENT_ENVIRONMENT",
         "caveat": "self-consistency, NOT accuracy against external ground truth"},
        {"report_section": "4 - Experiment 1", "experiment": "CAM-EXP-001.3",
         "quantity": "wrong-hand control",
         "value": f"{v('cam0013_recon_vs_provided3d_other_hand_median_mm')} mm median",
         "n": f"{v('cam0013_n_triangulated_observations'):,}",
         "statistical_unit": "hand observation",
         "evidence_level": "CONFIRMED_IN_CURRENT_ENVIRONMENT",
         "caveat": "included so the same-hand agreement is not read as trivial"},
        {"report_section": "4 - Experiment 1", "experiment": "CAM-EXP-001.3",
         "quantity": "QC outcome",
         "value": f"PASS_STRICT {v('gigahands_qc_pass_strict_n'):,}, "
                  f"PASS_SINGLE_HAND {v('gigahands_qc_pass_single_hand_n'):,}, "
                  f"REVIEW {v('gigahands_qc_review_n'):,}, "
                  f"EXCLUDE {v('gigahands_qc_exclude_n'):,}",
         "n": f"{v('gigahands_qc_total_observations'):,} observations",
         "statistical_unit": "hand observation",
         "evidence_level": "CONFIRMED_IN_CURRENT_ENVIRONMENT",
         "caveat": "thresholds were set on a small clean-control set; manual "
                   "validation is still pending (OPEN_ISSUE)"},
        {"report_section": "5 - Experiment 2", "experiment": "CAM-EXP-002",
         "quantity": "virtual vs physical focal, absolute root error",
         "value": f"{v('cam002_baseline_root_error_median_mm'):.0f} mm at "
                  f"{v('cam002_pipeline_virtual_focal_px'):.0f} px -> "
                  f"{v('cam002_physical_focal_root_error_median_mm'):.1f} mm at "
                  f"{v('cam002_gigahands_physical_focal_median_px'):.1f} px",
         "n": f"{v('cam002_hands_evaluated'):,} hands over "
              f"{v('cam002_frames_evaluated'):,} evaluated frames "
              f"({v('cam002_views_evaluated')} views, "
              f"{v('cam002_unique_physical_cameras_evaluated')} physical "
              f"cameras), sampled from a "
              f"{v('gigahands_bimanual_clean_frames'):,}-frame eligible pool",
         "statistical_unit": "hand",
         "evidence_level": "ROBUST_BUT_SINGLE_DATASET",
         "caveat": "root-aligned MPJPE is identical "
                   f"({v('cam002_root_aligned_mpjpe_median_mm')} mm) - focal moves "
                   "placement, not shape"},
        {"report_section": "5 - Experiment 2", "experiment": "CAM-EXP-002",
         "quantity": "focal perturbation -> median root displacement",
         "value": f"5 % -> {v('cam002_focal_5pct_root_shift_median_mm')} mm; "
                  f"10 % -> {v('cam002_focal_10pct_root_shift_median_mm')} mm; "
                  f"20 % -> {v('cam002_focal_20pct_root_shift_median_mm')} mm",
         "n": f"{v('cam002_hands_evaluated'):,} hands over "
              f"{v('cam002_frames_evaluated'):,} evaluated frames "
              f"({v('cam002_views_evaluated')} views, "
              f"{v('cam002_unique_physical_cameras_evaluated')} physical "
              f"cameras), sampled from a "
              f"{v('gigahands_bimanual_clean_frames'):,}-frame eligible pool",
         "statistical_unit": "hand",
         "evidence_level": "ROBUST_BUT_SINGLE_DATASET",
         "caveat": "FOCAL ONLY; specific to this working-distance regime"},
        {"report_section": "6 - Experiment 3", "experiment": "CAM-EXP-003",
         "quantity": "single-frame focal error, raw image + pinhole model",
         "value": f"AnyCalib {v('cam003_anycalib_pinhole_focal_err_median')} %, "
                  f"GeoCalib {v('cam003_geocalib_pinhole_focal_err_median')} %, "
                  f"PF-centered {v('cam003_pf_centered_focal_err_median')} %, "
                  f"PF-uncentered {v('cam003_pf_uncentered_focal_err_median')} % "
                  "(median)",
         "n": "175 views, 8 frames each",
         "statistical_unit": "view",
         "evidence_level": "ROBUST_BUT_SINGLE_DATASET",
         "caveat": "the deployed fixed-5000 px assumption is "
                   f"{v('cam003_demo_fixed_5000_focal_err_median')} % for reference"},
        {"report_section": "6 - Experiment 3", "experiment": "CAM-EXP-003.1",
         "quantity": "single-frame focal error with a distortion-aware model",
         "value": f"AnyCalib {v('cam0031_anycalib_gen_radial_focal_err_median')} %, "
                  f"GeoCalib {v('cam0031_geocalib_distorted_radial_focal_err_median')} % "
                  f"(within 5 %: "
                  f"{v('cam0031_anycalib_gen_radial_focal_err_within5')} % / "
                  f"{v('cam0031_geocalib_distorted_radial_focal_err_within5')} %)",
         "n": "175 views, 8 frames each",
         "statistical_unit": "view",
         "evidence_level": "ROBUST_BUT_SINGLE_DATASET",
         "caveat": "the GT-undistorted condition is an oracle diagnostic and is "
                   "never reported as a deployable method"},
        {"report_section": "6 - Experiment 3", "experiment": "CAM-EXP-004.1",
         "quantity": "paired distortion-aware gain, view-clustered",
         "value": f"AnyCalib {v('cam0031_anycalib_gen_radial_improvement_pp_view')[0]} pp "
                  f"[{v('cam0031_anycalib_gen_radial_improvement_pp_view')[1]}, "
                  f"{v('cam0031_anycalib_gen_radial_improvement_pp_view')[2]}]; "
                  f"GeoCalib "
                  f"{v('cam0031_geocalib_distorted_radial_improvement_pp_view')[0]} pp "
                  f"[{v('cam0031_geocalib_distorted_radial_improvement_pp_view')[1]}, "
                  f"{v('cam0031_geocalib_distorted_radial_improvement_pp_view')[2]}]",
         "n": "175 view clusters, 10,000 bootstrap iterations",
         "statistical_unit": "view cluster",
         "evidence_level": "CONFIRMED_IN_CURRENT_ENVIRONMENT",
         "caveat": "use these CIs, not the frame-level ones"},
        {"report_section": "7 - Experiment 4", "experiment": "CAM-EXP-004",
         "quantity": "between-view vs within-view share of the squared log "
                     "focal error",
         "value": f"AnyCalib {v('cam004_anycalib_gen_bias_fraction_pct')} % "
                  "between-view; GeoCalib "
                  f"{v('cam004_geocalib_distorted_bias_fraction_pct')} %; "
                  f"PF {v('cam004_pf_uncentered_bias_fraction_pct')} %",
         "n": "175 views",
         "statistical_unit": "(sequence, camera) static view - the decomposition "
                             "unit, NOT the physical camera",
         "evidence_level": "ROBUST_BUT_SINGLE_DATASET",
         "caveat": "this is what makes more frames ineffective. Do not restate "
                   "it as a physical-camera-level bias: physical camera is a "
                   "resampling unit in CAM-EXP-004.1, not the unit this "
                   "decomposition was computed over."},
        {"report_section": "7 - Experiment 4",
         "experiment": "CAM-EXP-004 + 004.1",
         "quantity": "median focal error vs frame count (single re-run)",
         "value": f"AnyCalib {v('cam0041_anycalib_gen_N8_focal_err_median')} -> "
                  f"{v('cam0041_anycalib_gen_N64_focal_err_median')} %; "
                  f"GeoCalib {v('cam0041_geocalib_distorted_N8_focal_err_median')} -> "
                  f"{v('cam0041_geocalib_distorted_N64_focal_err_median')} %; "
                  f"E2 {v('cam0041_E2_frozen_N8_focal_err_median')} -> "
                  f"{v('cam0041_E2_frozen_N64_focal_err_median')} % (N=8 to 64)",
         "n": "175 views",
         "statistical_unit": "view",
         "evidence_level": "ROBUST_BUT_SINGLE_DATASET",
         "caveat": "CONFIRMED_SATURATION within N <= 64 on this rig"},
        {"report_section": "7 and 8", "experiment": "CAM-EXP-004 / 004.1",
         "quantity": "frozen E2 exploratory ensemble",
         "value": f"selection set {v('e2_or_model_rerun_E2_frozen_cam004_median')} %, "
                  f"independent re-run "
                  f"{v('e2_or_model_rerun_E2_frozen_cam0041_median')} %, "
                  f"LOSO held-out {v('e2_loso_heldout_median_range_pct')[0]}-"
                  f"{v('e2_loso_heldout_median_range_pct')[1]} %",
         "n": "175 views; 5 LOSO folds",
         "statistical_unit": "view; sequence for the folds",
         "evidence_level": "EXPLORATORY_INTERNAL_VALIDATION",
         "caveat": "always quote all three numbers together; never 6.14 % alone"},
        {"report_section": "9 - Limitations", "experiment": "CAM-EXP-004.1",
         "quantity": "GeoCalib run-to-run variation on identical input",
         "value": f"median spread "
                  f"{v('determinism_geocalib_distorted_radial_median_spread_pct')} %, "
                  f"max {v('determinism_geocalib_distorted_radial_max_spread_pct')} %; "
                  "aggregate median shift ~0.32 pp",
         "n": "40 frames x 3 repeats",
         "statistical_unit": "frame",
         "evidence_level": "OPEN_ISSUE",
         "caveat": "AnyCalib is 100 % bit-identical under the same test"},
    ]
    write_csv(PKG / "tables" / "report_main_results.csv", rows)
    write_md_table(
        PKG / "tables" / "report_main_results.md", rows,
        title="Main results for the progress report body",
        notes="Two to four headline numbers per experiment. Everything else "
              "(p90, p95, per-camera breakdowns) belongs in the appendix. Every "
              "value here is keyed to `report_numbers.json`.")


# ------------------------------------------------------------- limitations
def limitations():
    rows = [
        {"limitation": "Single GigaHands rig",
         "impact": "every camera result comes from one capture setup, one optics "
                   "family, one resolution and one distortion regime",
         "what_is_already_mitigated": "175 views over 40 physical cameras and 5 "
                                      "sequences; results re-checked with camera- "
                                      "and sequence-level clustering",
         "what_remains": "no evidence that any number transfers to another rig",
         "when_to_resolve": "final confirmatory holdout",
         "evidence_level": "PENDING_EXTERNAL_CONFIRMATION"},
        {"limitation": "Provided 3D is not an independent external ground truth",
         "impact": "CAM-EXP-001.3's ~4 mm figure is self-consistency between the "
                   "released 2D and the provided 3D",
         "what_is_already_mitigated": "reconstruction never used the provided 3D "
                                      "as an input; a wrong-hand control is "
                                      "reported alongside",
         "what_remains": "absolute hand accuracy is unknown",
         "when_to_resolve": "only with a dataset carrying independent metric GT",
         "evidence_level": "OPEN_ISSUE"},
        {"limitation": "CAM-EXP-002 is focal-only",
         "impact": "the sensitivity result covers focal length and absolute depth, "
                   "not principal point, anisotropic focal or distortion",
         "what_is_already_mitigated": "the scope is stated explicitly wherever the "
                                      "number appears",
         "what_remains": "either narrow the paper claim, or run CAM-EXP-002.1",
         "when_to_resolve": "before any broad intrinsic-sensitivity claim",
         "evidence_level": "OPEN_ISSUE"},
        {"limitation": "GeoCalib is not run-to-run deterministic",
         "impact": "any GeoCalib-derived number carries a ~+/-0.35 pp run-to-run "
                   "tolerance that no bootstrap captures",
         "what_is_already_mitigated": "quantified (40 frames x 3 repeats), the "
                                      "input was verified bit-identical, and an "
                                      "independent full re-run is reported",
         "what_remains": "try deterministic algorithms; otherwise report the "
                         "median of k repeats",
         "when_to_resolve": "before any headline number goes into a paper",
         "evidence_level": "OPEN_ISSUE"},
        {"limitation": "E2 is a selection-set number",
         "impact": "6.14 % was obtained by picking the best of five candidates on "
                   "the same views it is reported on",
         "what_is_already_mitigated": "E2 frozen to a spec file; LOSO shows the "
                                      "selection is stable (5/5) and gives a "
                                      "held-out range",
         "what_remains": "a genuinely independent number",
         "when_to_resolve": "final confirmatory holdout",
         "evidence_level": "EXPLORATORY_INTERNAL_VALIDATION"},
        {"limitation": "AnyCam end-to-end inference unavailable on Windows",
         "impact": "the behavioural question - what does it emit on static clips - "
                   "is unanswered",
         "what_is_already_mitigated": "the mechanism was tested on AnyCam's own "
                                      "unmodified scoring function; the blocker is "
                                      "documented in full",
         "what_remains": "run the kept driver script on Linux",
         "when_to_resolve": "if a Linux environment becomes available",
         "evidence_level": "OPEN_ISSUE"},
        {"limitation": "External confirmation not yet performed",
         "impact": "no claim has been tested on data that did not influence it",
         "what_is_already_mitigated": "InterHand2.6M reserved as the primary "
                                      "sealed holdout, contamination risk zero; "
                                      "HanCo reserved for single-frame claims only",
         "what_remains": "a user decision on downloading the image releases",
         "when_to_resolve": "after CAM-EXP-005/006, once",
         "evidence_level": "PENDING_EXTERNAL_CONFIRMATION"},
        {"limitation": "Hand QC thresholds not manually validated",
         "impact": "false-pass / false-exclude rates of the automatic QC are "
                   "unknown",
         "what_is_already_mitigated": "no camera result uses the hand QC at all",
         "what_remains": "human adjudication of 100-300 stratified observations",
         "when_to_resolve": "before CAM-EXP-006 (hand-aware calibration)",
         "evidence_level": "OPEN_ISSUE"},
        {"limitation": "Saturation is demonstrated only up to 64 frames",
         "impact": "says nothing about hundreds of frames",
         "what_is_already_mitigated": "measured rather than predicted, on all 175 "
                                      "views, in one run",
         "what_remains": "nothing planned - the curve never descended, so a larger "
                         "N is not a promising lever",
         "when_to_resolve": "not planned",
         "evidence_level": "ROBUST_BUT_SINGLE_DATASET"},
    ]
    write_csv(PKG / "tables" / "report_limitations.csv", rows)
    write_md_table(PKG / "tables" / "report_limitations.md", rows,
                   title="Limitations",
                   notes="Each limitation states what has already been done about "
                         "it, so the section reads as an audit rather than an "
                         "apology.")


def future_work():
    rows = [
        {"id": "CAM-EXP-005", "status": "NEXT",
         "question": "why does each camera view have its own stable focal bias?",
         "why_now": "frame count and frame selection are both closed as levers, so "
                    "the per-view bias is the binding constraint",
         "inputs": "frozen E2 as the baseline; leave-one-sequence-out for every "
                   "selection; GeoCalib numbers with the +/-0.35 pp tolerance",
         "blocked_by": "nothing",
         "out_of_scope": "hand cues"},
        {"id": "CAM-EXP-006 prerequisite", "status": "BLOCKER FOR 006",
         "question": "how reliable are the automatic hand QC labels?",
         "why_now": "hand-aware calibration would depend on them for the first time",
         "inputs": "100-300 stratified observations, pre-registered sampling, "
                   "human adjudication vs the automatic verdict",
         "blocked_by": "nothing - it needs human time",
         "out_of_scope": "re-tuning the thresholds to fit the new labels without "
                         "saying so"},
        {"id": "CAM-EXP-006", "status": "LATER",
         "question": "does hand geometry provide a usable calibration cue?",
         "why_now": "only after the RGB-only ceiling is understood",
         "inputs": "hand-clean subsets; the frozen E2 baseline to beat",
         "blocked_by": "CAM-EXP-005 and the manual QC validation",
         "out_of_scope": "-"},
        {"id": "Final external confirmation", "status": "SEALED",
         "question": "does the best method hold on data that never influenced it?",
         "why_now": "only once, at the end",
         "inputs": "InterHand2.6M images if the download is approved; otherwise "
                   "the claim stays single-dataset",
         "blocked_by": "a user decision on the ~80 GB image release",
         "out_of_scope": "looking at it before then, for any reason"},
        {"id": "CAM-EXP-002.1", "status": "CONDITIONAL",
         "question": "how do principal point, anisotropic focal and distortion "
                     "propagate downstream?",
         "why_now": "only if the report needs a broad intrinsic-sensitivity claim "
                    "rather than a focal-only one",
         "inputs": "the CAM-EXP-002 cached inference",
         "blocked_by": "nothing - it is a scope decision, not a technical one",
         "out_of_scope": "-"},
    ]
    write_csv(PKG / "tables" / "report_future_work.csv", rows)
    write_md_table(PKG / "tables" / "report_future_work.md", rows,
                   title="Planned next steps")


def principles():
    rows = [
        ("primary statistical unit",
         "one static camera view within one sequence, i.e. (sequence, camera). "
         "175 of them in the camera experiments."),
        ("frame",
         "a repeated observation of that view, NOT an independent sample. 1400 "
         "frames means 175 views x 8 frames."),
        ("coarser cluster units",
         "physical camera id (40) and sequence (5), used to check that results "
         "survive correlated structure. Five clusters is a sensitivity check, not "
         "a trustworthy confidence interval."),
        ("bias-decomposition unit vs resampling unit",
         "these are different and must not be conflated. The bias/noise "
         "decomposition is computed over the (sequence, camera) STATIC VIEW, "
         "with frames as repeated observations inside it - so its output is a "
         "between-view component, not a physical-camera-level bias. Physical "
         "camera and sequence are RESAMPLING units, used only to widen "
         "confidence intervals in the CAM-EXP-004.1 robustness check."),
        ("focal reference",
         "the dataset-provided camera intrinsics."),
        ("provided 3D",
         "a reference / self-consistency target, not an independent external "
         "ground truth."),
        ("provided 2D",
         "dataset-provided 2D observations with confidence, not curated GT. An "
         "all-zero (0,0) entry is an observed invalid pattern, not a documented "
         "sentinel."),
        ("model comparison",
         "identical input frames for every model, from a manifest frozen before "
         "inference."),
        ("inference failure",
         "logged with a reason, never silently removed."),
        ("oracle conditions",
         "anything that uses the ground truth (GT-undistorted images, oracle "
         "best-of-N frame choice) is a diagnostic and is never reported as a "
         "deployable method."),
        ("aggregation",
         "never across sequences: a deployment calibrates one video at a time."),
        ("selection protocol from CAM-EXP-005",
         "leave-one-sequence-out; selection may only see the four training-fold "
         "sequences, and all five held-out numbers are reported."),
        ("final confirmation",
         "a sealed external dataset, evaluated once, after method development is "
         "finished."),
        ("reproducibility",
         "exact repository commits and checkpoint SHA256 recorded for every "
         "external model; GeoCalib is known to vary between runs and its numbers "
         "carry a ~+/-0.35 pp tolerance."),
    ]
    md = ["# Common evaluation principles\n",
          "These rules apply to every experiment in this report and can be "
          "quoted directly in Section 3.\n",
          "| Item | Rule |", "|---|---|"]
    md += [f"| {a} | {b} |" for a, b in rows]
    (PKG / "tables" / "common_evaluation_principles.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")
    write_csv(PKG / "tables" / "common_evaluation_principles.csv",
              [{"item": a, "rule": b} for a, b in rows])


def main() -> None:
    claim_ledger()
    main_results()
    limitations()
    future_work()
    principles()
    print("wrote claim ledger, main results, limitations, future work, principles")


if __name__ == "__main__":
    main()
