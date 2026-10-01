# Real evaluation protocol

## Units and eligibility

The unit is the **sequence-camera**. A unit is eligible if it has usable scene
frames, at least 8 usable LEFT hand frames and at least 8 usable RIGHT hand
frames, with hand eligibility taken from the frozen CAM-EXP-001.3 QC
(`PASS_STRICT` / `PASS_SINGLE_HAND`, triangulation successful, video
available).

Eligibility was frozen before any focal result. **No unit is ever removed for
having a large focal error, an unusual hand score, a boundary solution, a bad
absolute-3D number or a method that underperforms.** Those are results, not
eligibility criteria.

## Common set

All methods are compared on `PRIMARY_COMMON_SET`: units where M0, M1, M2 and M3
all produced a finite estimate. To stop that from hiding a method that simply
fails more often, the **full eligible set** and the per-method failure counts are
reported alongside (`n_full_eligible` in `results/summary/focal_summary.csv`).

## Frames

| | primary | nested |
| --- | ---: | --- |
| scene frames per unit | 32 | 8 / 16 / 32 / 64 |
| hand frames per side | 32 | 8 / 16 / 32 |

Frames are chosen by even spacing over the frame index — deterministic and
target-independent. Never by reference focal, hand performance or AnyCalib
error.

**N = 64 hand frames was skipped**, decided from measured runtime before any
focal result and recorded in the method spec. It is not a condition dropped
after seeing a number.

## Splits and statistics

- Outer split unit: **PHYSICAL CAMERA**, 5 folds by stable SHA-256 bucket. All
  of a camera's sequences stay in one fold.
- `lambda_fusion` selected per fold on **TRAIN cameras only**.
- Statistical cluster: **PHYSICAL CAMERA** — the independent hardware unit for a
  focal. Paired camera-cluster bootstrap, 10,000 iterations.
- Secondary sensitivity: sequence-level and participant-level aggregation, since
  scene and hand variability are sequence-level properties even though the focal
  is a camera property. With 5 sequences and 4 participants these are
  descriptive only.

## Focal metric

```
error = 100 * |f_pred - f_ref| / f_ref
```

reported as median, mean, p75, p90, and the fraction within 5 % and 10 % —
continuous with the earlier reports' convention. Scalar focal is
`sqrt(fx * fy)` throughout.

## Pre-specified success tags

Frozen before any real focal result.

**`CAM0094_FOCAL_INCREMENTAL_SIGNAL`** requires all of:
1. M3 median focal error < M0
2. relative reduction >= 10 %
3. paired median gain > 0
4. the 95 % camera-cluster bootstrap CI for the paired gain has lower bound > 0

**`CAM0094_ABSOLUTE_3D_IMPROVED`** requires all of:
1. median wrist/root error at least 10 % lower than M0
2. absolute MPJPE at least 5 % lower than M0
3. root-aligned MPJPE not worse by more than 2 mm median

**`CAM0094_SCENE_HAND_REFINEMENT_SUPPORTED`** = both of the above.

Partial outcomes have their own names rather than being argued into a win:
`FOCAL_GAIN_WITHOUT_DOWNSTREAM_GAIN`,
`DOWNSTREAM_GAIN_NOT_ATTRIBUTABLE_TO_FOCAL` (which triggers an implementation
audit rather than a claim), and `NO_INCREMENTAL_SEQUENCE_HAND_GAIN`.

## Controls

- **`C_WRONG_BONE`** — right-hand bone indices permuted within each finger,
  frozen in advance. Tests whether the hand score uses real anatomical
  correspondence. Run for M3 only, decided on runtime before results.
- **`RIG_MEDIAN_TRAIN_ONLY`** — a `DATASET_CONFOUND_DIAGNOSTIC`, not a method.
- **`REF_FOCAL_ORACLE`** — an `UPPER_DIAGNOSTIC` for absolute 3D, not an
  estimator.
- **`H_ONLY`** — hand score with no scene term; a diagnostic, not a deployment
  baseline.

A cross-video subject-swap control is deliberately **not** a primary control
here: the deployment has one worker per video, so cross-video identity
robustness is not a requirement of this task.

The method is never retuned after looking at a control.
