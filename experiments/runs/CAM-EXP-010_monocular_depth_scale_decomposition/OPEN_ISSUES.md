# Open issues — CAM-EXP-010

## 1. Every correction here is unobtainable at deployment

O1-O6 are all fitted on the reference 3D. The headline
74.7 -> 26.4 mm is what three per-video constants **could** buy if someone
handed them to you. Nobody will.

`metric_scale_identifiability.md` shows why they cannot simply be estimated
instead: the global metric scale is a one-parameter family that every image
explains equally well. So the follow-up is not "build an alpha estimator" -
it is "decide what metric anchor the task can afford". This run deliberately
does not choose one.

## 2. A bug my own reproduction gate caught

The first extraction took the predicted root to be `cam_t`, assuming WiLoR's
`keypoints_3d` were root-centred. `kp3[0]` is actually ~96 mm from the local
origin, nearly all in x. Depth still matched CAM-EXP-009.4 to 1.04 mm, so a
depth-only check would have passed, while wrist/root read 120.14 mm against the
true 74.95 mm.

Fixed to `kp3[0] + cam_t`, re-extracted, gate now matches to 0.000 mm. Worth
recording because the failure mode was quiet: the decomposition would have
attributed ~96 mm of phantom error to x/y, which is exactly the quantity this
experiment exists to measure.

## 3. The required correction does not transfer between recordings

Same physical camera, different sequence: 9.2 % median disagreement in alpha -
no better than the 8.1 % between the two hands of a single video. A
camera-specific factor measured once would not carry over, which rules out the
cheapest imaginable calibration story. Consistent with CAM-EXP-009.3.1, where
p41's geometry did not transfer across sessions.

## 4. No observable predicts the scale error

Every correlate tested is weak (|rho| <= 0.12): bbox fraction, predicted depth,
`cam_t` z, detection score, hand extent, handedness. There is no obvious
deployable regressor in these features. A negative result, but a useful one.

## 5. x/y was underweighted in the programme's thinking

CAM-EXP-009.4 framed the residual as monocular depth. It is 58 mm depth **and**
42 mm x/y, and x/y is dominated by `dx`. A perfect per-frame depth oracle still
leaves 40.6 mm of root error - worse than three per-video constants. Any future
work on translation must address x/y explicitly, and the cause of the `dx`
asymmetry was not investigated here.

## 6. The 35.6 mm pose floor is untouched and unexplained

Root-relative articulation error survives a perfect translation and is the
single largest block in absolute MPJPE (35.6 of 61.6 mm). No camera or
translation work reduces it. It also contains reference-reconstruction error of
unknown size, so it is an upper bound on the true network pose error rather than
a clean measurement of it.

## 7. Scope and power

4 participants, 5 sequences, one rig. The per-video bias structure could be
specific to this capture setup. `alpha` comes from as few as 6 FIT frames in
some units. O4 is restricted to x/y bias plus affine z to stay identifiable.

## 8. Not addressed

- `ORACLE_ABSOLUTE_HAND_SCALE` (how much one known hand size would fix) was
  specified as optional and not run.
- The camera-to-workspace transform. Metric camera coordinates still need their
  own transform and anchor; this run stops at the camera frame.
- Whether the learned prior's gap narrows on subjects closer to WiLoR's training
  distribution.
