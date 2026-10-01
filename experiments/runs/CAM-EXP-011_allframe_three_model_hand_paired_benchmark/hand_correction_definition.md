# The hand-structure correction

## What it is

The generic hand anatomy correction inherited from CAM-EXP-009.4, equivalent to
that run's `M1_SCENE_GENERIC_INDEPENDENT`:

- a **generic** prior: the neutral MANO hand's 20 connected bone proportions,
  with absolute size removed
- WiLoR 2D keypoints as image observations
- WiLoR root-relative 3D used only for **unit bone directions**; the network's
  predicted bone **lengths are discarded**, so its training shape prior does
  not enter the anatomy estimate
- LEFT and RIGHT anatomy fitted **independently**
- a held-out frame split inside each video, so the score is not fitted and
  scored on the same frames

## What it deliberately is not

CAM-EXP-009.4's shared-anatomy term (`M3`, the shared `p_seq`) is **not used**.
That run measured its contribution as exactly 0.000 pp under the tested
parameterisation, and the reason was a gauge freedom in the parameterisation
rather than a property of hands. Carrying it into a presentation benchmark
would add a component known to be inert.

No participant template from another video is used; nothing is carried across
recordings.

## Identical across all three models - the fairness condition

The score is evaluated on an **absolute focal grid** (400-2400 px, 81
log-spaced points), not on a grid anchored to any one model's estimate. So for
a given video the hand evidence is **one function of absolute focal**, and
`AnyCalib + Hand`, `GeoCalib + Hand` and `Perspective Fields + Hand` all read
the same function.

That makes the comparison clean: the only thing that differs between the three
`+ Hand` conditions is the scene evidence.

WiLoR runs **once per RGB frame**. The same cached observations serve all three
models; the network is never re-run per condition.

## Frames

All hand-available frames of the video enter the correction - no 8/16/32
sampling. A video needs at least 8 usable frames per side, a threshold
inherited from CAM-EXP-009.4 rather than chosen here.

Frames with no usable hand are recorded as `hand_available = 0`. They are
**not** removed from the scene benchmark: hand quality never gates scene
eligibility.

## An honest caveat

This correction is inherited from the CAM-EXP-009.4 line of work and is **not
itself claimed to be a solved hand-geometry calibration method**. CAM-EXP-009.4
found its gain on this rig small and confounded by the dataset's narrow focal
range. This run measures what it does, on all frames, across three estimators -
nothing more.
