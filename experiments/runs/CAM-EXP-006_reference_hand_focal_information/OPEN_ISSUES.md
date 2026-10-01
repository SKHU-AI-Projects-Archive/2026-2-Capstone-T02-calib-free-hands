# CAM-EXP-006 — open issues

Registered so they are carried forward rather than quietly forgotten. None of
these is resolved by this run.

## Carried in from earlier runs

| id | issue | status |
| --- | --- | --- |
| `SINGLE_FOCAL_RIG_CONFOUND` | the GT reference focal varies by only 1.90 % (CV) across the 175 GigaHands views, so per-view focal conclusions have almost no dynamic range to be judged against | **still open**; it is the reason the `CONSTANT_RIG_FOCAL_ORACLE` is pre-registered here |
| `GEOCALIB_NONDETERMINISM` | GeoCalib reproduced 0 % bit-identical outputs over 40 frames × 3 repeats; AnyCalib reproduced 100 % | still open; affects the E2 comparator only |
| `ANYCAM_BLOCKED_ON_WINDOWS` | the AnyCam end-to-end path is `BLOCKED_IMPLEMENTATION_ON_WINDOWS`; only the mechanistic identifiability test ran | still open |
| `FINAL_CONFIRMATORY_HOLDOUT_UNOPENED` | InterHand2.6M / HanCo remain reserved and unopened | intentionally open |

## Raised by the manual-QC gate

| id | issue |
| --- | --- |
| `QC_SINGLE_REVIEWER` | one reviewer, no inter-rater agreement measured |
| `QC_SIDE_UNVERIFIABLE_UNDER_OCCLUSION` | 3.5 % of audited cases cannot have their anatomical side confirmed from a single view |
| `QC_SAMPLE_IS_PASS_ONLY` | the audit sample is drawn from observations that already passed the existing QC filter, so it cannot estimate the failure rate of what that filter rejects |

## Raised by this run

| id | issue |
| --- | --- |
| `REFERENCE_3D_NOT_INDEPENDENT_OF_RIG_CALIBRATION` | the reference hand is triangulated using the rig's own provided camera parameters, so any error shared across the whole rig is inherited and is undetectable from inside this dataset |
| `REFERENCE_3D_COVERAGE_76PCT` | 23.7 % of (view, frame, hand) cases are unusable — hand out of frame, not annotated in this camera, or too few reconstructing cameras. The usable subset is not a random subset of hand poses, so the ceiling is measured on easier-than-average hands |
| `ORACLE_CEILING_NOT_ACHIEVABLE` | every number in this run assumes a reference 3D that does not exist at deployment time; the ceiling must never be quoted as achievable performance |
| `PLANAR_PNP_BACKEND_DIFFERS` | the `PLANARIZED` control necessarily runs through a different PnP back-end (IPPE/ITERATIVE) than the frozen primary (SQPNP), because SQPNP rejects coplanar input. The control is therefore not a perfectly matched comparison |
| `SINGLE_DATASET` | every conclusion is on GigaHands alone, on one rig, with one focal setting |

## Explicitly out of scope here

| item | where it belongs |
| --- | --- |
| predicted-hand cues from WiLoR / AnyHand | CAM-EXP-007 |
| any scene-aware focal corrector | not planned; CAM-EXP-005 found no deployable RGB cue |
| modifying E2 | E2 is frozen; it is an exploratory ensemble, never a proposed method |
