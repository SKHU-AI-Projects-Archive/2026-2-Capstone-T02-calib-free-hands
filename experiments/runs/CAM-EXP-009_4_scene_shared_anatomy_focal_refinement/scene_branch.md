# Scene branch

## Estimator

`AnyCalib` in the `anycalib_gen / radial:2` configuration - the deterministic
configuration used in CAM-EXP-003 / 003.1 / 008. The existing frozen prediction
cache from CAM-EXP-004.1 is **reused**, not re-run, so the preprocessing,
distortion convention and focal scalar definition are identical to the earlier
experiments by construction.

## Scalar focal convention

`sqrt(fx * fy)`, unchanged from CAM-EXP-003 / 005 / 008. Not redefined here.

## Static-camera sequence aggregation

The camera is fixed for the sequence, so per-unit scene frames are aggregated in
the log domain:

```
f_scene = exp( median_i log f_i )
```

which is the geometric median. This is **M0_SCENE_ONLY**, the baseline.

A robust scale comes from the same predictions:

```
sigma_scene = max( 1.4826 * MAD(log f_i), 0.02 )
```

The floor keeps a unit whose scene predictions happen to agree almost exactly
from dominating the fusion.

## Nuisance parameters

The principal point used in the candidate `K` is the scene estimator's own
prediction. **No GT `cx`/`cy`, no GT `k1`/`k2`, and no GT `fy/fx` ratio enters
inference.** CAM-EXP-008's closure experiments that used GT nuisance values were
diagnostics, not a deployment method, and that distinction is kept here.

## Scene frames are sampled separately from hand frames

Scene frames are uniformly spaced over the whole sequence; hand frames come from
pre-existing target-independent QC. The two sets may differ - deliberately, so
the scene estimator is not made to depend on the frames where hands happen to be
well visible. The comparison unit is the same for every method.
