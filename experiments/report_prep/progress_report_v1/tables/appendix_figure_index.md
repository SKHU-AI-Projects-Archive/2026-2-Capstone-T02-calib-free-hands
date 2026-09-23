# Appendix figure index

Figures already produced by the experiment runs. Nothing is copied into this package; the appendix should reference these paths directly.

| figure | source | purpose | recommended appendix section | keep or drop | reason |
|---|---|---|---|---|---|
| CAM_EXP_004_MAIN_EXPLANATION.png | runs/CAM-EXP-004_static_camera_multiframe_aggregation/figures/ | the seven-panel summary of the multi-frame result | A.4 Experiment 4 | DROP | its content is now split across Fig06 and Fig07, which are print-safe; keeping both would duplicate the message |
| per_view_trajectories.png | runs/CAM-EXP-004_static_camera_multiframe_aggregation/figures/ | predicted fx per frame for a good, a biased and a noisy view | A.4 Experiment 4 | KEEP | the single clearest picture of 'the camera never changed, so every offset is model error' |
| oracle_best_vs_aggregation.png | runs/CAM-EXP-004_static_camera_multiframe_aggregation/figures/ | oracle best-of-N vs deployable aggregation | A.4 Experiment 4 | KEEP | shows that even GT-assisted frame selection does not reach the target, which is why frame selection was deprioritised |
| bias_vs_noise_decomposition.png | runs/CAM-EXP-004_static_camera_multiframe_aggregation/figures/ | bias vs noise, with magnitudes | A.4 Experiment 4 | DROP | superseded by Fig07 panel A |
| model_agreement_vs_error.png | runs/CAM-EXP-004_static_camera_multiframe_aggregation/figures/ | GT-blind model disagreement vs actual error | A.4 Experiment 4 | KEEP | a weak but honest negative-ish result worth showing rather than hiding |
| CAM_EXP_003_1_MAIN_EXPLANATION.png | runs/CAM-EXP-003_1_distortion_aware_diagnostic/figures/ | distortion-aware diagnostic summary | A.3 Experiment 3 | DROP | superseded by Fig05 |
| distortion_magnitude_vs_improvement.png | runs/CAM-EXP-003_1_distortion_aware_diagnostic/figures/ | \|k1\| vs improvement, with the restricted-range caveat | A.3 Experiment 3 | KEEP | documents an inconclusive analysis that a reader would otherwise expect us to have done |
| static_camera_bias_before_after.png | runs/CAM-EXP-003_1_distortion_aware_diagnostic/figures/ | per-view bias before and after distortion handling | A.3 Experiment 3 | KEEP | links Experiment 3 to Experiment 4's bias finding |
| examples/raw_vs_undistorted_examples.png | runs/CAM-EXP-003_1_distortion_aware_diagnostic/figures/ | what the undistortion actually did to the images | A.3 Experiment 3 | KEEP | makes the oracle condition concrete and shows it was not free |
| CAM_EXP_004_1_MAIN_EXPLANATION.png | runs/CAM-EXP-004_1_pre_cam005_robustness_audit/figures/ | robustness audit summary: cluster CIs, LOSO, 8-64, determinism | A.5 Robustness and reproducibility | KEEP | the only figure that shows the statistical re-analysis and the determinism finding together |
| qc_examples/qc_status_grid.png | runs/CAM-EXP-001_3_gigahands_multiview_triangulation/figures/ | PASS_STRICT / REVIEW / EXCLUDE example crops | A.1 Experiment 1 | KEEP | lets a reader judge the QC labels instead of trusting them |
| overlay figures | runs/CAM-EXP-001_1_gigahands_bad_view_diagnosis/figures/ | bad-view overlays showing the all-zero pattern and hand swaps | A.1 Experiment 1 | KEEP | the visual evidence behind two of Experiment 1's conclusions; pick 2-3, not the whole set |
