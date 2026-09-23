# The per-view bias target

## Definition

For each static view `v = (sequence, camera)` and each frame `t` of the frozen
64-frame nested set:

```
e_v = median_t  log( f_pred(v, t) / f_ref(v) )
```

* `f_pred` — AnyCalib `anycalib_gen` + `radial:2` predicted focal, read from the
  frozen CAM-EXP-004.1 predictions. **No new inference was run.**
* `f_ref` — the dataset-provided reference focal: `GT_EFFECTIVE_FOCAL`, which
  CAM-EXP-002's `focal_usage_audit.md` derives to equal `GT_NATIVE_FX` exactly in
  the validated original-image pixel convention.
* the median over frames, not the mean, because CAM-EXP-004 showed the
  within-view scatter is small but occasionally heavy-tailed.

Secondary targets: `|e_v|`, and the conventional relative focal error in percent.

## Why AnyCalib is the primary model

CAM-EXP-004.1 measured AnyCalib-gen as **100 % bit-identical across repeated
runs on the same image**, while GeoCalib was 0 % bit-identical (median spread
1.18 %, p90 6.85 %, max 41.50 %). Using AnyCalib keeps run-to-run variation out
of the attribution target entirely. The frozen E2 ensemble is carried as a
secondary target only; it contains GeoCalib, and its definition is not modified
here.

## What this target is and is not

It is an **analysis target for attribution**. It is not a deployment estimator:
computing it requires `f_ref`, which a deployment does not have.

Because `f_ref` appears inside the target, it is never used as a predictor. Its
descriptive association with the bias is reported in the oracle table and
flagged as partly definitional.

## Measured values

| | value |
|---|---|
| views | 175 |
| sequences | 5 |
| unique physical cameras | 40 (every one appears in more than one sequence) |
| frames per view | 64 (frozen nested set) |
| median signed bias | −0.0974 log = **−9.28 %** |
| camera-cluster bootstrap 95 % CI | [−0.1159, −0.0778] log = [−10.94 %, −7.48 %] |

## The confound that shapes every conclusion below

The reference focal of this rig is **almost constant**: across all 175 views it
spans 824.9–986.6 px with a median of 921.6 px and a coefficient of variation of
**1.90 %**. The AnyCalib prediction varies about four times as much (CV 7.92 %).

Two consequences follow, and both are load-bearing:

1. The per-view "bias" is almost entirely variation in the *prediction*, not in
   the quantity being predicted.
2. An estimator that ignores the image completely and always returns this rig's
   median focal scores **0.85–0.88 % median relative error and 97.7 % within
   ±5 %**. Any feature group containing the predicted focal can therefore reach a
   very low error by learning that one constant. That is not calibration, and the
   `CONSTANT_RIG_FOCAL_ORACLE` control exists to make the distinction impossible
   to miss.
