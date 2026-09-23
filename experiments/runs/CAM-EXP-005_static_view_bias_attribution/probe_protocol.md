# Probe protocol

## What the probe is for

One question per feature group: **does this group carry cross-view predictive
information about the per-view bias?** It is a `LINEAR PROBE DIAGNOSTIC`. It is
not a calibration method, it is not proposed as one, and nothing here is "our
method".

## Model and preprocessing

Ridge regression. Inside every outer fold, and using the training fold only:

1. median imputation (training-fold medians),
2. standardisation (training-fold mean and sd),
3. alpha chosen from the fixed grid `{0.01, 0.1, 1, 10, 100}` by `GroupKFold`
   over physical cameras **within the training fold**.

The held-out fold contributes no statistic of any kind before prediction. Seed
20260924. The grid, the protocol and the fold membership were frozen in
`cam_exp_005_probe_spec_v1.json` and `cam_exp_005_outer_splits_v1.json` before
any performance number was inspected.

## Outer validations

| | folds | what it tests |
|---|---|---|
| `A_LOSO` | 5, one sequence held out | **content / sequence shift.** The same physical camera is usually in training too, so this is *not* a new-camera test |
| `B_LOCO` | 40, one physical camera held out | **new camera.** No view of the held-out camera is in training. This is the primary decision criterion |

## Feature sets

`P0` training mean only · `P1` scene geometry · `P2` image global · `P3` border ·
`P4` model self · `P5` model disagreement · `P6` P1+P2+P3 · `P7` all deployable.

Diagnostic-only references, never deployable: `D_CAMERA_ID` (one-hot physical
camera) and `D_SEQUENCE_ID`. Under the matching leave-one-out protocol the
held-out level is unseen, so these fall back to the training mean — which is
exactly the point: a camera-id lookup table has nothing to say about a camera it
has never seen.

## Decision thresholds, fixed before results

Primary criterion is `B_LOCO`, against the `P0` mean-bias baseline:

* **A** median relative focal error reduced by ≥ 10 % relative, or
* **B** within-±5 % rate up by ≥ 5 percentage points,

with the direction holding under the physical-camera cluster bootstrap.

## The control that decides the interpretation

`CONSTANT_RIG_FOCAL_ORACLE` ignores the image entirely and outputs the training
folds' median reference focal. It uses ground truth, so it is oracle-only — but
on a rig whose reference focal varies by 1.90 % it is a very strong predictor,
and any probe that fails to beat it has demonstrated memorisation of a constant
rather than a calibration signal.
