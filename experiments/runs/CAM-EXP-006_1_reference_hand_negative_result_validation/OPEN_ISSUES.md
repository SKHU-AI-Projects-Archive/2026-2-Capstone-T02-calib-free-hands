# CAM-EXP-006.1 — open issues

## Raised here

| id | issue |
| --- | --- |
| `CAM006_DISTORTION_UNMODELLED` | CAM-EXP-006 fed raw distorted 2D to a solver assuming no distortion, on a rig whose median k1 is -0.392. This inflated its headline error from about 50 % to 202 %. The CAM-006 report needs a wording and attribution correction. |
| `DISTORTION_FLATNESS_INTERACTION` | a systematic reprojection bias is far more damaging when the objective is already flat. Ignoring distortion cost 5 % in strong perspective and the difference between 203 % and 47.5 % on real data. Any future identifiability claim must model distortion first. |
| `REFERENCE_3D_ERROR_DOMINATES` | a reference-3D error of only 1-2 % of hand diameter costs 26-49 % focal error. The reference hand is not accurate enough for this problem even in principle. |
| `REGIME_MATCH_IS_ORDER_OF_MAGNITUDE` | the report's section 4.5 correspondence between synthetic ingredients and the real result is qualitative, not a fitted decomposition. Synthetic noise is isotropic; the real residual is partly correlated. |
| `S5_UNDERPOWERED` | the 5 % non-default-focal stress subset has 4 views and 4 physical cameras. Flagged, not interpreted. |
| `PLANAR_ENDPOINT_BACKEND` | the exactly coplanar synthetic endpoint forces IPPE/ITERATIVE instead of SQPNP; diagnostic only. |

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
| CAM-EXP-006 report wording correction | required; separate docs commit |
| CAM-EXP-007 | REDESIGN — the profiled-PnP-with-predicted-hand variant is low priority, but learned hand-derived camera cues are not ruled out |
| a distortion-aware reference-hand diagnostic | optional; would likely still hit the flatness limit |
