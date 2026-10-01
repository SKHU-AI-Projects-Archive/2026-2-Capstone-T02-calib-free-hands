# Paired evaluation protocol

## What is paired with what

For each model, `SCENE_ONLY` and `SCENE_PLUS_HAND` are compared on the **same
videos**, with the same scene frames and the same scene evidence. The `+ Hand`
condition adds hand evidence; it never removes a scene frame.

The headline table uses the `GLOBAL_6_CONDITION_COMMON_VIDEO_SET`: videos with
a result in all six conditions, so every cell of the table describes the same
population.

## Eligibility, frozen before results

- verified usable RGB video (the CAM-EXP-003 camera set)
- scene aggregate available with >= 80 % frame coverage
- >= 8 usable hand frames per side for the `+ Hand` condition

**No video is excluded for its focal error, its hand score, a boundary solution
or a method underperforming.** Those are results.

## Statistics

- **Unit**: sequence-camera video. One video, one focal, one vote.
- **Primary metric**: median relative focal error,
  `100 * |f_pred - f_ref| / f_ref`. The mean is recorded but is not primary,
  because a small number of large errors would move it disproportionately.
- **Cluster**: physical camera, since the same camera recurs across sequences.
  Paired camera-cluster bootstrap, 10,000 iterations, on `error_OFF - error_ON`.
- **Win rate**: the fraction of videos where the `+ Hand` error is lower - an
  intuitive companion to the median.

## Lambda selection

The fusion weight is chosen per model from a frozen grid
`[0, 0.25, 0.5, 1, 2, 4]` (0 is the scene-only sanity point), on **TRAIN
physical cameras only**, with the same 5 outer folds for every model. All of a
camera's sequences stay in one fold.

The **procedure** is identical for all three models. The **selected value** may
differ between them, because their scene scores are calibrated differently, and
forcing one value would handicap whichever model's score is on a different
scale.

## Leakage barrier

The reference focal is closed through frame selection, model inference, hand
scoring and aggregation. It is read for TRAIN cameras during lambda selection
only. `evaluate.py` is the first script to open the TEST focal, and it writes
`results/raw/REFERENCE_FOCAL_OPENED.txt` with the git HEAD and the hashes of
the frozen predictions, the frame manifest and the lambda selection. Nothing is
regenerated afterwards.
