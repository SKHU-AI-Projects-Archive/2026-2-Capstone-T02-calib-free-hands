# CAM-EXP-008 — open issues

## Raised here

| id | issue |
| --- | --- |
| `FULL_SEQUENCE_COMPUTE_BLOCKER` | the whole-video run (30,031 frames, 157 views) was attempted and abandoned at a measured ~8 h projection against a pre-registered 6 h bar. The run uses FULL_DURATION_64_APPROXIMATION instead, median 51 frames per unit. How much a whole video would add is unmeasured. |
| `HAND_PROFILE_SHALLOW_AND_BIASED` | over a +-50 % focal sweep the entire hand reprojection excess is a median 1.88 px, and its preferred focal sits a median 1.184x above the scene estimate with 22.7 % of views at the grid boundary. The hand term is too weak to help and points the wrong way when it does act. |
| `N64_THINLY_POPULATED` | only 23 of 154 units have 64 QC-passing frames on the grid, so the N64 row is not representative. ALL_COMMON (median 51 frames) is the pre-registered preferred primary and is used as such; the verdict is identical either way. |
| `ORACLE_NUISANCE_CONDITION_NOT_RUN` | **CLOSED.** Run in `CAM-EXP-008_closure_camera_nuisance`. Supplying the provided principal point, distortion and fy/fx ratio deepened the profile only 1.82 -> 2.86 px, left the q bias essentially unchanged (1.263 -> 1.272), and made the fused estimate marginally worse. Verdict `CAM008_NEGATIVE_ROBUST_TO_CAMERA_NUISANCE`. |
| `E2_SECONDARY_ANCHOR_NOT_RUN` | for the same reason. |
| `S2_UNDERPOWERED_S5_EMPTY` | the non-default-focal stress subsets have 6 views / 5 cameras and 0 views respectively. No conclusion drawn. |
| `SMALL_GAIN_MATCHED_BY_SHUFFLE` | S1's +0.0317 pp gain has a bootstrap CI excluding zero, yet the view-shuffled control gained more (+0.0478 pp). A CI excluding zero is therefore not evidence of hand information here — it reflects a generic effect of perturbing a robust prior. |

## Carried forward, still open

| id | issue |
| --- | --- |
| `SINGLE_FOCAL_RIG_CONFOUND` | reference focal CV 1.73 %; a constant scores 0.906 % against scene-only 8.87 %. This dominates every per-camera conclusion on GigaHands. |
| `REFERENCE_3D_NOT_INDEPENDENT_OF_RIG_CALIBRATION` | the reference hand uses the rig's own provided camera parameters. |
| `GEOCALIB_NONDETERMINISM` | 0 % bit-identical; kept out of the primary. |
| `ANYCAM_BLOCKED_ON_WINDOWS` | still blocked. |
| `FINAL_CONFIRMATORY_HOLDOUT_UNOPENED` | InterHand2.6M / HanCo remain reserved and unopened. |
| `SINGLE_DATASET` | one rig, one dataset, one focal setting. |

## Next

| item | status |
| --- | --- |
| CAM-EXP-008.1 predicted-hand practical fusion | **do not proceed for now.** The CAM-008 result gives little motivation to repeat the same explicit scene-plus-reprojection formulation with noisier predicted geometry. It does not establish that every predicted-hand formulation or learned hand-derived constraint must fail. |
| varied-focal dataset | the most important next validation requirement, for this line as for CAM-EXP-007 |
| a different fusion objective | not ruled out — this run tested one objective, so an objective-side limitation cannot be excluded |
| whole-sequence rerun | only worth doing if a cheaper hand profile is found; the current cost is dominated by 161 PnP solves per hand observation |
