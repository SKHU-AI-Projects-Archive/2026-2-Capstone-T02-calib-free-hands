# CAM-EXP-007 — open issues

## Raised here

| id | issue |
| --- | --- |
| `NARROW_PROBE_INSTRUMENT` | one linear probe on one mean-pooled layer. A null result from this instrument does not establish absence of camera information in the network. Deliberately narrow, so that the test is clean rather than a search. |
| `LATENT_SPATIAL_POOLING_DISCARDS_STRUCTURE` | the 1280-d latent is a global average over a 16x12 map. A perspective cue could live in the spatial arrangement and be destroyed by pooling. |
| `SINGLE_LAYER_FROZEN_BY_DESIGN` | only `vit_out` was probed. Probing several layers and reporting the best would be selection on the result, so it was not done - but it also means other layers are untested. |
| `REPRESENTATION_SIDE_NOT_EXCLUDED` | this run tested one frozen checkpoint, one layer, global-average pooling and a linear read-out. The narrow-focal rig is a major limitation, but a representation-side limitation cannot be excluded either, and neither should be presented as the single binding obstacle. |
| `HAND_COVERAGE_NOT_UNIFORM` | 15 of 175 views are hand-ineligible and they concentrate in plant (8), dog (4) and boxing (3); 13 had zero detections. The eligible set is not a uniform sample of content. |
| `TARGET_IS_ANYCALIB_SPECIFIC` | the target is AnyCalib's residual. GeoCalib was excluded for non-determinism, so no second calibration model corroborates the target. |
| `THREE_MARGINAL_ASSOCIATIONS` | 3 of 78 features have cluster-bootstrap CIs excluding zero. No formal multiple-comparison correction is applied to the feature-association table, so these are descriptive associations only, not statistically confirmed findings. None converts into unseen-camera predictive performance. |
| `NO_MULTIPLE_COMPARISON_CORRECTION` | the feature-association table reports Spearman with cluster-bootstrap CIs and no FDR control. It is a descriptive supplement and is not a success criterion. |

## Carried forward, still open

| id | issue |
| --- | --- |
| `SINGLE_FOCAL_RIG_CONFOUND` | reference focal CV 1.90 %; the constant-reference oracle scores 0.93 % and an AnyCalib-focal-only probe 1.03 %. This dominates the whole problem on GigaHands and is the single biggest obstacle to any per-camera conclusion. |
| `GEOCALIB_NONDETERMINISM` | 0 % bit-identical over 40 frames x 3 repeats. |
| `ANYCAM_BLOCKED_ON_WINDOWS` | end-to-end path still blocked. |
| `FINAL_CONFIRMATORY_HOLDOUT_UNOPENED` | InterHand2.6M / HanCo remain reserved and unopened. |
| `SINGLE_DATASET` | one rig, one dataset, one focal setting. |

## Next

| item | status |
| --- | --- |
| CAM-EXP-007.1 hand-derived correction head | **not justified by the present evidence.** No pre-registered hand feature group beats the global-bias baseline by the required margin and the shuffle controls do not degrade. This is a decision about the current evidence, not a permanent exclusion. |
| varied-focal dataset | **the most important next validation requirement.** The rig's 1.9 % focal spread provides too little dynamic range for a strong positive camera-cue claim. This does not establish that the dataset is the *only* limitation - see `REPRESENTATION_SIDE_NOT_EXCLUDED`. |
| a wider representation search | optional, and only if pre-registered as a search with multiplicity accounted for - several layers, spatial structure retained, non-linear read-out |
