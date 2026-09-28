# Research question

## The question

> Does adding sequence-specific shared hand anatomy and a generic anatomical
> prior to a scene-based sequence focal estimate improve (1) focal estimation
> error, (2) absolute wrist/root 3D error and (3) absolute 3D MPJPE, compared
> with scene-only calibration?

This is the first experiment in the CAM-009 line that goes past *"is there
focal information in hand geometry?"* and asks *"does it add anything to a
camera estimate we already have?"*

## What the earlier runs established

- **CAM-EXP-009.1** — in clean symmetric synthetic geometry the bilateral cue
  recovers a known focal almost exactly, once three solver faults are fixed.
- **CAM-EXP-009.2** — with bone-level mismatch, 2D noise and articulation error
  the formulation becomes unstable, and at `COMBINED_MODERATE` a different
  synthetic subject's right hand worked as well as the matched one.
- **CAM-EXP-009.3 / 009.3.1** — with participant identity verified from the
  official GigaHands naming convention: within-session left/right structure is
  closer than across participants (ratio 0.585, separation 1.886x the
  reconstruction repeatability), but the one participant with two sessions has
  hands that are **further apart across sessions than different participants
  are**. Conclusion: `SUBJECT_SPECIFIC_PRIOR_NOT_JUSTIFIED`,
  `SEQUENCE_SPECIFIC_SHARED_ANATOMY_REMAINS_PLAUSIBLE`, and the generic
  bone-identity signal is strong (+857 % control degradation).

## What this run therefore builds

Exactly the formulation those results support, and not the one they rule out:

```
p_seq = softmax(log(p0 + eps) + a)              <- re-estimated per video
p_L   = softmax(log(p_seq + eps) + delta_L)
p_R   = softmax(log(p_seq + eps) + delta_R)
```

`p0` is a **generic** neutral-MANO prior. `a`, `delta_L`, `delta_R` are fitted
from scratch for every sequence-camera unit. **No participant template is
carried over from another recording** - that is the thing CAM-EXP-009.3.1 found
no support for.

## Deployment conditions this is designed around

One worker per video, that worker's two hands, one continuous sequence, a
static camera, fixed intrinsics within the sequence, the whole video available
offline, no real-time requirement, and **no need to recognise a person across
videos**. No absolute hand size, no known object dimensions, no EXIF, and the
reference focal is never an inference input.
