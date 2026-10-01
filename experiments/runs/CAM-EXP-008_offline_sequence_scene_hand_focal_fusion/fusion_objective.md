# CAM-EXP-008 — the fused objective

```
L_total(f) = L_scene(f) + lambda * L_hand(f)
f_fused    = argmin_f L_total(f)
```

## 1. Candidate grid

`q = f / f_scene`, 161 log-spaced points over **[0.5, 1.5]**, frozen before any
result.

The range is wide relative to AnyCalib's known error scale and deliberately
avoids the meaningless large-focal plateau CAM-EXP-006 mapped: this run refines
a scene anchor, it does not solve for a focal from scratch. A boundary minimum
is flagged, not refined.

Secondary sensitivity [0.25, 2.0] was registered but not needed, because the
primary result is a clean null.

## 2. Refinement

Parabolic interpolation in log-q across the three grid points around an
interior minimum.

## 3. lambda

Grid {0.01, 0.03, 0.1, 0.3, 1, 3, 10}, plus lambda = 0 which *is* S0.

Selected by grouped CV over **training physical cameras only**, inside each
outer fold, against those cameras' dataset-provided reference focal. The
held-out camera never influences it.

Observed selection was bimodal — about half the folds chose 0.01 (effectively
off) and about half chose 10 — which is itself a symptom: the inner objective
could not find a lambda that reliably helped.

## 4. Validation

| protocol | folds | role |
| --- | ---: | --- |
| leave-one-physical-camera-out | 40 | **PRIMARY** |
| leave-one-sequence-out | 5 | secondary, content shift only |

LOCO is primary because deployment means a new factory and a new camera. If the
same physical camera appeared in training through a different sequence, a
camera-specific bias could be memorised.

## 5. Statistical unit

The `SEQUENCE_CAMERA_UNIT`. Frames, hands and joints are repeated observations
and are never counted as independent samples. Uncertainty is a physical-camera
cluster bootstrap, 10,000 iterations.
