# Hand branch

## The observation source is the deployable one

The hand information comes from the **target camera's monocular WiLoR output**,
the same path the demo pipeline uses, not from multi-view reconstruction. That
is what makes the method runnable in the actual deployment.

Fields used, confirmed by reading `rgb_predictor.py`:

- `keypoints_2d` (21, 2) — **original image pixels**, produced by
  `_kp2d_crop_to_full` from the normalised crop prediction. This is the 2D
  observation.
- `keypoints_3d` (21, 3) — root-relative metres, with the left-hand x-flip
  already undone.

## Bone lengths from the network are DISCARDED

Only the **unit direction** of each connected bone is taken:

```
u_f,b = (X_child - X_parent) / ||X_child - X_parent||
```

The reason is leakage of a different kind: WiLoR's predicted bone lengths encode
its own training shape prior. Using them as the anatomy target would smuggle
that prior into a model whose whole point is to estimate anatomy. So the network
supplies **articulation direction only**, and the bone-length proportions are
estimated by our own model.

Degenerate or non-finite bones are invalid, and a frame needs at least 12 usable
joints.

## One worker per video

The deployment has one worker and two hands. Per frame, the highest-scoring
detection of the requested handedness is taken — the production logic, not a
selection made using dataset identity to improve the benchmark. Frames where the
detector returns more than one hand of the same handedness are counted and
reported as an ambiguity rate rather than silently resolved.

## Caching

WiLoR runs **once per frame**, and every method consumes the same cached output.
The network is never re-run per focal condition, which is what makes any
downstream difference attributable to the focal estimate alone.
