# Error decomposition protocol

## Per-frame quantities

With the reference focal supplied (oracle), for each valid frame and hand:

```
T_pred = [cam_t_x, cam_t_y, cam_t_z * f_ref / f_pipeline]
T_ref  = Xc[0]                      reference root in camera coordinates

dx, dy, dz  = T_pred - T_ref
root_error  = ||T_pred - T_ref||
xy_error    = sqrt(dx^2 + dy^2)
depth_error = |dz|
```

and the depth ratio `z_ref / z_pred`, reported in log space so that a
multiplicative error is additive and symmetric.

## The cached residual, and why

Writing the reference joints as `Xc_j` and the network's root-relative joints as
`kp3_j`:

```
(kp3_j + T) - Xc_j  =  R_j + e
R_j = kp3_j - (Xc_j - T_ref)     root-relative pose residual, FIXED
e   = T - T_ref                  translation error
```

`R_j` does not depend on the translation, so absolute MPJPE under **any**
corrected translation is `mean_j ||R_j + e||`. Caching `R` lets every oracle in
the hierarchy be scored without re-running the reconstruction, and guarantees
all of them are scored on exactly the same frames and the same pose.

`mean_j ||R_j||` is the **root-aligned pose floor** - the error that survives
even a perfect translation.

## FIT / EVAL split

Corrections O1-O4 have parameters. Fitting them on a frame and then scoring that
same frame would make the oracle trivially good, so each frame is assigned
`FIT` or `EVAL` by a stable SHA-256 bucket of
`(sequence, camera, hand, frame)` - deterministic, and independent of any error
or outcome.

Parameters are fitted on **FIT frames only**, frozen, and then applied to EVAL
frames. **Every number reported for O1-O6 is an EVAL-frame number.**

## Granularity

**PRIMARY: one correction per sequence-camera video**, fitted from both hands
together. That matches the deployment shape - one video, one worker, one
correction.

**SECONDARY:** left-only and right-only fits, so the two hands' required
corrections can be compared. LEFT and RIGHT need not appear in the same frames.

## Robust estimators, fixed before results

```
O1  alpha = exp( median_f [ log z_ref,f - log z_pred,f ] )
O2  alpha, beta by Huber IRLS on z_pred -> z_ref
O3  b = median_f ( T_ref,f - T_pred,f ), per axis
O4  x/y constant bias + affine z
```

Only frames with positive `z_pred` and `z_ref` enter the scale fits.

## Statistics

Frame errors are aggregated to a **sequence-camera median**, then compared
paired against O0 with a **physical-camera cluster bootstrap**, 10,000
iterations. All errors in mm; `alpha` is dimensionless. Median, p75 and p90 are
reported.

Independence is not overstated: frames are not independent, the two hands share
a person and a video, the sequence-camera is the raw diagnostic unit, the
physical camera is the bootstrap cluster, and with 4 participants no
anatomy-population claim is made.

## No p-value dependence

This is a decomposition. Direction is decided by **effect size and held-out
correction magnitude**, never by a single p-value.
