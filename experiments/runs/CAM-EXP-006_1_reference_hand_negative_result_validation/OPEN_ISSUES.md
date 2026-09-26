# CAM-EXP-006.1 — open issues

## Raised here

| id | issue |
| --- | --- |
| `CAM006_DISTORTION_UNMODELLED` | CAM-EXP-006 fed raw distorted 2D to a solver assuming no distortion, on a rig whose median k1 is -0.392. Supplying the provided coefficients takes the N=16 median error from 203.13 % to 47.53 %, so unmodelled distortion was a major contributor to the inflated headline. The CAM-006 report needed a wording and attribution correction. |
| `DISTORTION_FLATNESS_INTERACTION` | a systematic reprojection bias weighs far more heavily when the objective is already flat. Ignoring the same distortion is associated with a 5.18 % error in the strong-perspective synthetic condition, against the 203 % vs 47.5 % gap on real data. The two are not separable additive contributions. Any future identifiability claim should model distortion first. |
| `REFERENCE_3D_ERROR_SENSITIVITY` | in a post-hoc sensitivity sweep, a reference-3D error of only 1-2 % of hand diameter is associated with 26-49 % focal error. This suggests the reference hand may not be accurate enough for this problem, but the sweep is explanatory and does not establish a share of the real failure. |
| `REGIME_MATCH_IS_ORDER_OF_MAGNITUDE` | the report's section 4.5 correspondence between synthetic ingredients and the real result is qualitative, not a fitted decomposition. Synthetic noise is isotropic; the real residual is partly correlated. |
| `S5_UNDERPOWERED` | the 5 % non-default-focal stress subset has 4 views and 4 physical cameras. Flagged, not interpreted. |
| `PLANAR_ENDPOINT_BACKEND` | the exactly coplanar synthetic endpoint forces IPPE/ITERATIVE instead of SQPNP; diagnostic only. |
| `B_VS_C_CUTOFF_NOT_PRE_REGISTERED` | the frozen spec supplies verdict labels A-F and four numeric materiality thresholds, but no operational rule for choosing between B and C. The R5 usability cutoff used in `final_verdict.py` was written after results were seen and is a `POST_HOC_INTERPRETIVE_VERDICT`. Future runs should pre-register the decision rule, not only the labels. |
| `MECHANISM_DIAGNOSTICS_POST_HOC` | the distance x noise sweep, the REAL_REGIME_MATCHED condition and the reference-3D perturbation sweep were all added after the primary result. They are explanatory, not confirmatory. |
| `NO_ADDITIVE_ERROR_DECOMPOSITION` | conditions R0-R5 differ in what the solver is given; they do not partition the error into independent causal components. No statement of the form "distortion caused X % of the failure" is supported. |

## Carried forward, still open

| id | issue |
| --- | --- |
| `SINGLE_FOCAL_RIG_CONFOUND` | reference focal CV 1.90 % across 175 views; a constant scores 0.85 %. Caps every per-view focal conclusion on GigaHands. |
| `REFERENCE_3D_NOT_INDEPENDENT_OF_RIG_CALIBRATION` | the reference hand uses the rig's own provided camera parameters; rig-wide error is inherited and undetectable from inside the dataset. |
| `REFERENCE_3D_COVERAGE_76PCT` | the usable subset is not a random subset of hand poses. |
| `QC_SINGLE_REVIEWER` | the 200-case manual-QC gate had one reviewer and no inter-rater agreement. |
| `QC_SAMPLE_IS_PASS_ONLY` | the audit sample was drawn from observations that already passed the existing QC filter. |
| `GEOCALIB_NONDETERMINISM` | 0 % bit-identical over 40 frames x 3 repeats. |
| `ANYCAM_BLOCKED_ON_WINDOWS` | end-to-end path still blocked. |
| `FINAL_CONFIRMATORY_HOLDOUT_UNOPENED` | InterHand2.6M / HanCo remain reserved and unopened. |
| `SINGLE_DATASET` | one rig, one dataset, one focal setting. |

## Next

| item | status |
| --- | --- |
| CAM-EXP-006 report wording correction | **done**, commit a4f40a2 |
| pre-register the verdict DECISION RULE, not only the labels | required for the next verification run |
| CAM-EXP-007 | REDESIGN — repeating the same profiled-PnP estimator with noisier predicted geometry is low priority; a redesigned experiment may still test whether learned hand-model outputs contain deployment-available camera information absent from explicit reference 3D |
| a distortion-aware reference-hand diagnostic | optional; would likely still hit the flatness limit |
