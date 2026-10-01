# CAM-EXP-012 — presentation results (early, primary only)

```
CAM-EXP-012  PRIMARY PRESENTATION ANALYSIS COMPLETE / FULL CONTROL ANALYSIS PENDING
```

The wrong-bone control has not been computed. It is deferred, not
cancelled; see `CAM_EXP_012_PRESENTATION_PROTOCOL_AMENDMENT.md`.


## Question

Does adding full-video hand-joint geometry evidence to full-video scene
focal evidence reduce focal estimation error on unseen physical
cameras? The primary comparison is `SCENE_ONLY` against
`SCENE_PLUS_CORRECT_HAND`; only the hand term differs.


## Protocol

- statistical unit: one sequence-camera video, one vote

- paired set: 153 videos scored under every compared condition

- hand curves cross-fitted over temporal blocks of 8 frames

- `alpha` from TRAIN videos only, at a fixed +/- 5 percent log-focal offset;
  it never sees a reference focal

- `lambda` selected by inner physical-camera CV inside the outer TRAIN
  set; `lambda = 0` is not a candidate

- predictions frozen and hashed before the test reference focal was
  opened

- uncertainty: paired physical-camera cluster bootstrap, 10,000 iterations


## Main result

| Model | Scene_only_median_error_pct | Scene_plus_correct_hand_median_error_pct | Gain_of_medians_pp | Paired_median_gain_pp | Paired_mean_gain_pp | Relative_gain_pct | Win_rate_pct | Tie_rate_pct | Loss_rate_pct | CI95_low | CI95_high | N_videos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AnyCalib | 16.019 | 16.447 | -0.428 | -0.459 | -0.496 | -2.67 | 9.2 | 20.3 | 70.6 | -0.492 | -0.441 | 153 |
| GeoCalib | 21.912 | 25.167 | -3.254 | 0.0 | -2.601 | -14.85 | 19.6 | 41.8 | 38.6 | 0.0 | 0.0 | 153 |
| Perspective Fields | 19.888 | 20.211 | -0.322 | 0.0 | -1.758 | -1.62 | 47.1 | 22.9 | 30.1 | 0.0 | 0.31 | 153 |


### Answer

**No.** For all 3 scene estimators the hand term left focal error
the same or worse on unseen physical cameras.
`HAND_ON_NUMERICAL_GAIN_SUPPORTED` is false for every model.


Two kinds of column must be read together. `Gain_of_medians_pp`
compares the two medians; `Paired_median_gain_pp` and
`Paired_mean_gain_pp` are the per-video paired differences, which
is what the cluster bootstrap resamples. Where a large share of
videos is unchanged the paired median can sit exactly at 0 while
the paired mean is clearly negative. That combination means most
videos are untouched and the ones that do move mostly move the
wrong way.


The inner camera CV selected the smallest available `lambda`
(0.125, the grid's lower boundary) in 12 of 15 model-by-fold
cells. Since `lambda = 0` is excluded by design, that is the
selection procedure pushing the hand term as close to off as the
grid allows. A boundary selection is normally a warning about the
grid; here it points the same way as the test result.


## Controls

| Model | Scene | Correct_Hand | Shuffled_Hand | Wrong_Bone | Correct_vs_Shuffled_advantage_pp | CI95_low | CI95_high | N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AnyCalib | 16.019 | 16.447 | 16.495 | PENDING | 0.0 | 0.0 | 0.0 | 153 |
| GeoCalib | 21.912 | 25.167 | 24.439 | PENDING | 0.0 | 0.0 | 0.0 | 153 |
| Perspective Fields | 19.888 | 20.211 | 19.558 | PENDING | 0.0 | 0.0 | 0.0 | 153 |


`Wrong_Bone` is `PENDING`. The shuffled control substitutes another
video's hand curve through a frozen donor mapping — different sequence,
different camera, different participant — reusing the same `alpha` and
the same selected `lambda`.


## How far the hand term moves the focal

| Model | Median_shift_pct | moved_gt0_pct | ge_0.25_pct | ge_0.5_pct | ge_1_pct | ge_2_pct |
| --- | --- | --- | --- | --- | --- | --- |
| AnyCalib | 0.4063 | 79.7 | 79.7 | 30.1 | 11.1 | 5.9 |
| GeoCalib | 0.4063 | 58.2 | 58.2 | 39.2 | 31.4 | 25.5 |
| Perspective Fields | 0.8142 | 77.1 | 77.1 | 56.2 | 45.1 | 35.3 |


## Selected lambda and loss balance

| model | outer_fold | D_scene | D_hand_correct | before_ratio | alpha_correct | after_ratio | selected_lambda | lambda_boundary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ANYCALIB | 0 | 6.250000000000008 | 4.355993778665483e-05 | 6.969590045864764e-06 | 143480.46203855643 | 1.0 | 0.125 | 1 |
| GEOCALIB | 0 | 1.0807286370746854 | 4.1869137298322995e-05 | 3.8741582171500807e-05 | 25812.058877028077 | 1.0 | 0.125 | 1 |
| PERSPECTIVE_FIELDS | 0 | 0.7041186474110299 | 6.009039689871984e-05 | 8.534129456685446e-05 | 11717.656792944737 | 1.0 | 0.125 | 1 |
| ANYCALIB | 1 | 6.250000000000008 | 4.1670591275065583e-05 | 6.667294604010485e-06 | 149985.87274041917 | 1.0 | 0.125 | 1 |
| GEOCALIB | 1 | 1.013929336455976 | 4.08523817992432e-05 | 4.029115277602679e-05 | 24819.344473931244 | 1.0 | 0.125 | 1 |
| PERSPECTIVE_FIELDS | 1 | 0.6956972557550357 | 5.3761433998489696e-05 | 7.727705342195544e-05 | 12940.451993423012 | 1.0 | 0.125 | 1 |
| ANYCALIB | 2 | 6.250000000000008 | 3.988632059842902e-05 | 6.381811295748635e-06 | 156695.32577156724 | 1.0 | 0.125 | 1 |
| GEOCALIB | 2 | 1.0924867433345562 | 3.7928373922391606e-05 | 3.4717468338905655e-05 | 28803.943600903745 | 1.0 | 0.125 | 1 |
| PERSPECTIVE_FIELDS | 2 | 0.7041186474110299 | 5.6864413632573996e-05 | 8.075970412324465e-05 | 12382.412873553052 | 1.0 | 2.0 | 0 |
| ANYCALIB | 3 | 6.250000000000008 | 4.2464275329942336e-05 | 6.794284052790765e-06 | 147182.5423002808 | 1.0 | 0.125 | 1 |
| GEOCALIB | 3 | 1.0031778337523174 | 3.816082196783341e-05 | 3.803993737092006e-05 | 26288.16105161251 | 1.0 | 0.125 | 1 |
| PERSPECTIVE_FIELDS | 3 | 0.7041186474110299 | 5.769142296349763e-05 | 8.193423533892038e-05 | 12204.91038774582 | 1.0 | 0.25 | 0 |
| ANYCALIB | 4 | 6.250000000000008 | 4.351361406288546e-05 | 6.962178250061665e-06 | 143633.20847051608 | 1.0 | 0.125 | 1 |
| GEOCALIB | 4 | 1.293840833759466 | 3.876218904147791e-05 | 2.9959008890489286e-05 | 33378.94132797756 | 1.0 | 0.125 | 1 |
| PERSPECTIVE_FIELDS | 4 | 0.7087072022445604 | 5.7389108568869744e-05 | 8.097717701627919e-05 | 12349.158575865958 | 1.0 | 0.25 | 0 |


## Verdicts

| Model | HAND_ON_NUMERICAL_GAIN_SUPPORTED | CORRECT_VS_SHUFFLED_SIGNAL_SUPPORTED | CORRECT_HAND_SPECIFIC_SIGNAL_SUPPORTED |
| --- | --- | --- | --- |
| AnyCalib | False | False | PENDING_WRONG_BONE |
| GeoCalib | False | False | PENDING_WRONG_BONE |
| Perspective Fields | False | False | PENDING_WRONG_BONE |


`CORRECT_HAND_SPECIFIC_SIGNAL_SUPPORTED` cannot be evaluated without
the wrong-bone control. Even if the correct condition beats both
scene-only and the shuffled control, this analysis does not establish
that the improvement comes from correct anatomical bone
correspondence: the shuffled control rules out only that any hand curve
would do, while holding the video fixed and permuting the bones is what
isolates correspondence.


## Scope

- GigaHands only, and its reference focals vary little across cameras
  (`SINGLE_FOCAL_RIG_CONFOUND`), so absolute error levels here are not
  transferable to rigs with diverse optics

- the reserved confirmatory holdout remains unopened

- reference 3D geometry is `OTHER_CAMERA_ONLY_REFERENCE_3D`; the target
  camera is excluded before reconstruction
