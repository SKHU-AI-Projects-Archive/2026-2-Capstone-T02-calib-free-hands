# Metric scale identifiability

## The question

Can the global metric scale of a hand trajectory be determined from the cues the
deployment is allowed to use?

Allowed: a static monocular RGB camera, one worker, that worker's two hands, one
continuous sequence, intrinsics constant within the sequence, the whole video
offline. **Not** allowed: EXIF, known table or object dimensions, an assumed
`typical human hand size`, a participant template from another video, multi-camera
input, or any external metric marker.

## The algebra

Under a pinhole camera a 3D point `X` in camera coordinates projects to

```
u = f * X / Z + c
```

Now scale every 3D point and the camera translation by one positive scalar `s`:

```
X' = sX ,  T' = sT   =>   X'_cam = s X_cam
u' = f * (sX) / (sZ) + c = f * X / Z + c = u
```

The `s` cancels exactly. **Every 2D observation is unchanged**, so no amount of
2D evidence can distinguish `s`.

The hand anatomy used throughout CAM-EXP-009.x makes this worse rather than
better, because it was deliberately made scale-free:

```
p_b = l_b / sum_b l_b
```

Multiplying every bone length by `s` leaves `p` identical. The anatomy term
therefore carries **no** absolute size information by construction.

## Numerical verification

`src/identifiability_audit.py`, scales `s = 0.5, 1, 2, 5`:

| test | what it varies | max difference | result |
| --- | --- | ---: | --- |
| `GLOBAL_SCALE_INVARIANCE_TEST` | one frame, one hand | 5.68e-14 px | **PASS** |
| `MULTI_FRAME_SCALE_TEST` | 64 frames, one consistent hand | 1.14e-13 px | **PASS** |
| `BILATERAL_SCALE_TEST` | both hands, both trajectories | 1.14e-13 px | **PASS** |
| `ANATOMY_SCALE_FREE_TEST` | sum-normalised bone proportions | 2.78e-17 | **PASS** |

All differences are at machine precision.

### Does multi-frame consistency break the tie?

**No.** Requiring the same hand across 64 frames constrains the *relative*
geometry, but scaling the hand and the entire trajectory together reproduces
every frame's 2D exactly. A consistent hand of size `L` at distance `Z` and a
consistent hand of size `sL` at distance `sZ` generate identical video.

### Does bilateral consistency break the tie?

**No.** The two hands' correspondence constrains relative shape. Scaling both
hands and both trajectories by the same `s` leaves both projections unchanged.
Without knowing the *absolute* size of at least one hand, left/right agreement
says nothing about `s`.

## Conclusion

> With the deployment's allowed cues, the global metric scale is **not uniquely
> determined by geometry**. It is a one-parameter family of solutions, all of
> which explain the images equally well.

## Two distinctions that must not be collapsed

**1. This is not the claim that `a monocular camera cannot estimate depth`.**

A network trained on metric data — WiLoR here — does produce metric-looking
output. It does so by applying a **learned statistical scale prior**: hands in
its training distribution had certain sizes, so a hand subtending a given angle
is placed at a typical distance. That is a genuine and useful estimator. It is
also a *prior*, not a measurement: its error is whatever the gap is between this
worker's hand and the training distribution, and no amount of image evidence
corrects it.

`LEARNED STATISTICAL SCALE PRIOR` ≠ `GEOMETRICALLY IDENTIFIABLE SCALE`. The
empirical part of this run measures how large that gap actually is.

**2. Correct intrinsics are not metric world scale.**

CAM-EXP-009.4 is the empirical demonstration: with the reference focal supplied
as an oracle, wrist/root error was still around 78 mm. A perfect `K` fixes the
*ray directions*; it does not fix *how far along the ray* anything is.

## What this implies for the project, without adopting it here

If metric absolute 3D is required, the scale must come from somewhere other than
monocular geometry. Candidate sources, stated as the logical options rather than
as decisions:

- an explicit metric anchor in the scene (a known dimension, a calibration
  object, a second camera, a depth sensor),
- a per-worker measurement taken once,
- or an accepted reliance on the learned prior, with its error budgeted and
  stated.

**None of these is adopted as an assumption by this experiment.** This run is a
diagnostic; choosing among them is a project decision that needs the error
budget first.

## Not addressed here

The camera-to-workspace transform. Even with metric camera-frame coordinates, a
workspace/global coordinate system needs its own transform and its own anchor.
This run stops at the camera frame.
