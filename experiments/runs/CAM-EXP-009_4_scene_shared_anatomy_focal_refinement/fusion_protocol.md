# Fusion protocol

## Both terms dimensionless, in the log-focal domain

```
L_total(q) = L_scene(q) + lambda * L_hand(q)

L_scene(q) = ( log q / sigma_scene )^2
L_hand(q)  = heldout_reprojection(q) / image_diagonal , divided by the unit's
             own median finite score
```

where `q = f_candidate / f_scene`.

## No candidate-dependent rescaling

The hand curve is divided by a **fixed** per-unit scale (its own median finite
score), not min-max normalised. Min-max rescaling would change what `lambda`
means from unit to unit and would make the tuned value meaningless.

## Candidate grid

`q` in `[0.5, 1.5]`, 81 log-spaced points, anchored on the unit's own scene
estimate. A minimum at either end is recorded as a **boundary** outcome and
reported, never clipped or hidden.

## Lambda selection — nested, and never on the test focal

- Outer split unit is the **PHYSICAL CAMERA**, 5 folds, assigned by a stable
  SHA-256 bucket of the camera id. Every sequence of a camera lands in the same
  fold, so the same hardware never appears on both sides.
- For each outer fold, `lambda` is chosen from the frozen grid
  `[0, 0.25, 0.5, 1, 2, 4]` using the **TRAIN cameras' reference focal only**.
  `lambda = 0` is the scene-only sanity point.
- The TEST cameras' reference focal is not read during selection. After the
  predictions are written, `evaluate_focal.py` is the first script that opens
  it, and it logs the moment in `results/raw/REFERENCE_FOCAL_OPENED.txt` with
  the git HEAD and the SHA-256 of the frozen predictions.
- Predictions are **never regenerated** after that point.

## Statistics

Clustered on the **physical camera** - the independent hardware unit for a
focal. Paired camera-cluster bootstrap, 10,000 iterations, on the per-unit gain
`error(M0) - error(M3)`.

Secondary sensitivity aggregations by sequence and by participant are reported,
because the scene and hand variability are sequence-level even though the focal
is a camera property.
