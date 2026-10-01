# CAM-EXP-008 — reference hand provenance

## 1. Name and status

`OTHER_CAMERA_ONLY_REFERENCE_3D`, re-assembled into
`SEQUENCE_CONSISTENT_REFERENCE_HAND`.

It **is**: a hand triangulated from the other calibrated cameras of the same
sequence, with the camera under test excluded before reconstruction, then
re-expressed with sequence-median bone lengths and its absolute scale removed.

It **is not**: ground truth 3D, independent physical ground truth, or anything
a single deployed camera could obtain.

## 2. Build

| item | value |
| --- | --- |
| implementation | `CAM-EXP-001_3/src/loco.py :: reconstruct()`, reused unchanged |
| inlier threshold | 8.0 px |
| minimum inlier cameras | 4 |
| minimum paired joints | 12 of 21 |
| 2D confidence threshold | 0.5 |
| excluded 2D pattern | the all-(0,0) "not annotated" pattern |
| units built | 154 |
| hand observations | 12,860 |

The 2D are **dataset-provided 2D observations** throughout, never "GT2D".

## 3. Non-circularity

| check | result |
| --- | --- |
| camera under test excluded before reconstruction | yes, by construction |
| camera under test ever an inlier of its own reference hand | **0 of 12,860** |
| target-camera extrinsics used | no |
| GT focal used in the optimisation | no |
| GT distortion or principal point used in the primary | no |

```
NON_CIRCULARITY = PASS
```

The target camera's own 2D **is** used on the other side of the objective — as
the observation the reference hand is reprojected against. That is the intended
design and is recorded explicitly in `tables/circularity_audit.csv`.

## 4. The limit that remains

The reference hand is triangulated with the rig's own provided camera
parameters, so any error shared across the whole rig is inherited and is not
detectable from inside this dataset. The reprojection figures are internal
consistency, not external accuracy.
