# Separating session from subject

## The confound CAM-EXP-009.3 could not resolve

A within-session left/right pair shares:

- the same participant
- the same recording session
- the same calibration solution
- the same reconstruction context
- the same capture conditions

A cross-participant pair is camera-matched but shares none of the last four.
So `D_WITHIN_SESSION < D_CROSS_PARTICIPANT` is consistent with **either** a
person-specific anatomical signature **or** a session-specific reconstruction
effect. The comparison alone cannot tell them apart.

## The one comparison that can

Only a **same participant, different session** comparison separates them, and in
this subset only `p41` has two sessions (`boxing-0021` and `plant-0004`).

| level | median |
| --- | ---: |
| `D_repeat` (same hand twice) | 0.01087 |
| `D_within_session` (participant equal weight) | 0.02887 |
| `D_cross_participant` (pair equal weight) | 0.04937 |
| **`p41` cross-session, same side (L-L, R-R)** | **0.08516** |
| **`p41` cross-session, opposite side (L-R, R-L)** | **0.07704** |

`p41`'s own within-session distance is 0.02113 — among the lowest of any
participant. Across two sessions the same participant's hands are **0.075–0.089
apart, further than two different participants (0.0457)**, and all four
camera-matched cross-session comparisons (L-L, R-R, L-R, R-L) exceed the
cross-participant level.

## What follows

The within-session advantage is **not explained by a persistent person-level
hand-shape signature**. If it were, `p41`'s templates would have transferred
between its two sessions, and they do not — even left-vs-left.

The session/reconstruction context is therefore doing substantial work in the
within-session number.

## What does not follow

- Not that human hand anatomy changes between recordings. The reference geometry
  mixes annotation, triangulation, calibration and association error with
  anatomy, and this result is equally consistent with the reconstruction being
  session-dependent.
- Not that same-person hand geometry is useless. This is **one participant**,
  two sessions, in one dataset subset, measured through one reconstruction
  pipeline.
- Not a population claim of any kind. n = 1 for the cross-session question and
  n = 4 for the participant question.

## Consequence for the deployment task

The intended deployment is one worker per video, that worker's two hands, one
continuous sequence, offline whole-video processing, with calibration estimated
per video. **Persistent person identification across videos is not required.**

So a weak cross-session person signature does **not** by itself sink the idea of
a shared hand constraint within a sequence — those are different problems.
Equally, the within-session similarity does **not** justify a person-specific
prior carried between recordings.
