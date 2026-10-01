# Experiment → report section map

The report has eleven sections. Seven experiment runs feed them. The sub-runs
(001.1, 001.2, 001.3, 003.1, 004.1) are *not* separate chapters — they are the
internal validation and robustness work inside their parent experiment.

| Report section | Experiments used | Primary artifacts |
|---|---|---|
| **1. 연구 과제 및 문제 정의** | CAM-EXP-002 (motivation only) | `figures/main/Fig01_existing_pipeline_and_camera_role.*` |
| **2. 연구 목표 및 전체 실험 설계** | all | `figures/main/Fig02_research_flow_with_findings.*` |
| **3. 데이터셋 및 공통 평가 환경** | dataset preparation, CAM-EXP-001.3 QC | `tables/report_dataset_usage.md`, `tables/common_evaluation_principles.md`, `figures/main/Fig03_gigahands_validation_and_qc_flow.*`, `experiments/manifests/external_model_provenance_v1.json` |
| **4. 실험 1. 데이터 및 좌표계 신뢰성 검증** | CAM-EXP-001 **+ 001.1 + 001.2 + 001.3 absorbed here** | `figures/main/Fig03`, `tables/report_main_results.md` rows for Experiment 1 |
| **5. 실험 2. 초점거리 오차에 따른 절대 깊이 변화** | CAM-EXP-002 | `figures/main/Fig04_focal_error_vs_absolute_depth_shift.*` |
| **6. 실험 3. 기존 카메라 파라미터 추정 방법 평가** | CAM-EXP-003 **+ 003.1 absorbed here**; CAM-EXP-004.1's corrected statistics; a short subsection on AnyCam applicability | `figures/main/Fig05_single_frame_calibration_and_distortion.*`, `tables/report_main_results.md`, `runs/CAM-EXP-004_1.../anycam_static_stress_test.md` |
| **7. 실험 4. 다중 프레임 활용 및 카메라별 편향 분석** | CAM-EXP-004 **+ 004.1 absorbed here** | `figures/main/Fig06_multiframe_1_to_64.*`, `figures/main/Fig07_bias_noise_and_ensemble_summary.*` |
| **8. 현재까지의 종합 결과 및 개선 방향** | all | `tables/hypothesis_result_evidence.md`, `tables/claim_evidence_ledger.md`, `experiments/manifests/cam_exp_004_e2_frozen_spec_v1.json` |
| **9. 연구의 한계** | CAM-EXP-004.1 audit | `tables/report_limitations.md`, `runs/CAM-EXP-004_1.../OPEN_ISSUES.md` |
| **10. 후속 실험 계획** | — | `tables/report_future_work.md`, `experiments/manifests/development_validation_split_v1.json`, `experiments/manifests/final_confirmatory_holdout_v1.json` |
| **11. 결론** | all | `report_numbers.json`, `tables/claim_evidence_ledger.md` |

## Absorption rules

* **CAM-EXP-001.1 / 001.2 / 001.3** are written as the internal validation of
  Experiment 1: how the reprojection error tail was diagnosed (all-zero 2D
  pattern, per-camera hand swaps, frame-index audit against official source) and
  how the QC subsets were produced. No separate chapter.
* **CAM-EXP-003.1** is written as the distortion/robustness check inside
  Experiment 3, immediately after the pinhole baseline numbers. No separate
  chapter.
* **CAM-EXP-004.1** is split across Sections 6, 7 and 9: its corrected clustered
  statistics belong with the results they correct, its 16/32/64-frame
  measurement belongs in Experiment 4, and its reproducibility and provenance
  findings belong in Limitations. **There is no "실험 4.1" chapter.**

## Appendix

`tables/appendix_figure_index.csv` lists the diagnostic figures from the
original runs that are worth reproducing, and which appendix subsection each
belongs to.
