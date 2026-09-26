# CAM-EXP-009.1 — open issues

## Raised here

| id | issue |
| --- | --- |
| `BILATERAL_FRAGILE_TO_REAL_ASYMMETRY` | the bilateral objective is exact under perfect symmetry but reaches 12.0 % focal error at 2 % bilateral asymmetry and 29.3 % at 5 %, while the single-hand held-out objective stays at 0.04 %. Human hands are not symmetric to within 1 %. This is the main obstacle to using the idea. |
| `BILATERAL_MORE_NOISE_SENSITIVE` | at 1 px 2D noise the bilateral error is 8.2 % against the held-out objective's 2.7 %. A real reference articulation is noisy. |
| `DOF_GROUPING_IS_MISSPECIFIED` | the D10 and D5 groupings force bones within a group to be EQUAL in length, which no real hand satisfies. Their poor performance is model bias, not evidence about constrained anatomy. A template-scaling parameterisation was not tested. |
| `SYNTHETIC_ONLY` | nothing here has been run on GigaHands. |
| `DISTORTION_ASSUMED_KNOWN` | the coefficients are exact in synthesis; on real data they would come from the scene estimator, with its own error. |
| `EIGHT_TRIALS_PER_CONDITION` | reduced from 20 on measured cost. Effect sizes are coarse, though the 0.035 % vs 50 % contrast is far larger than that precision. |

## Resolved here

| id | resolution |
| --- | --- |
| `BILATERAL_FOCAL_SIGNAL_NOT_IDENTIFIABLE_SYNTHETICALLY` (CAM-009) | **superseded.** Caused by three solver faults, not geometry. |
| `FREE_BONE_LENGTHS_ABSORB_FOCAL_ERROR` (CAM-009) | **not supported.** With a converged, correctly-undistorted fit, 20 free bone lengths still leave the focal identifiable to 0.035 %. |
| `HELD_OUT_SPLIT_DOES_NOT_CATCH_SHARED_BIAS` (CAM-009) | **restated.** The held-out split works fine; in CAM-009 it was measuring a mismatched reprojection convention. |

## Carried forward

| id | issue |
| --- | --- |
| `SINGLE_FOCAL_RIG_CONFOUND` | GigaHands reference focal varies by ~1.7-1.9 %; even a working cue has little room to show value there. |
| `FINAL_CONFIRMATORY_HOLDOUT_UNOPENED` | InterHand2.6M / HanCo remain reserved and unopened. |

## Next

| item | status |
| --- | --- |
| asymmetry-tolerant bilateral formulation | **required before the real phase.** A side-scale nuisance or robust per-bone weighting, re-validated synthetically. |
| correctly specified low-dimensional shape model | open; template scaling rather than bone equalisation |
| GigaHands real bilateral phase | **hold** until the two above are settled; then reuse the CAM-EXP-008 paired machinery |
| varied-focal dataset | still the binding requirement for the whole line |
