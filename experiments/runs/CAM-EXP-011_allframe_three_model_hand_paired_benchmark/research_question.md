# Research question

## The question, in one line

> On the same GigaHands videos, does adding whole-video hand structure to an
> existing scene focal estimator reduce focal error?

Three scene estimators, each in two conditions:

| | |
| --- | --- |
| AnyCalib | AnyCalib + Hand |
| GeoCalib | GeoCalib + Hand |
| Perspective Fields | Perspective Fields + Hand |

The comparison of interest is **within** each model, OFF versus ON. Ranking the
three estimators against each other is not the purpose of this run, and the
results are not presented that way.

## Why all frames

CAM-EXP-003 evaluated these estimators on 8 uniformly spaced frames per view
(175 views x 8 = 1,400 frames), chosen before any model ran. CAM-EXP-004.1
extended that to 16, 32 and 64 frames per view (11,200 frames) and found the
per-video aggregation largely saturated.

Neither of those is treated here as a mistake. This run does something
different: it processes **every usable RGB frame**, so the whole-video question
can be answered directly rather than by extrapolating from a saturation curve.

## What stays constant between OFF and ON

Everything except the hand term: the same sequences, the same cameras, the same
RGB frames, the same scalar focal definition, the same aggregation, the same
reference, the same folds and the same evaluation code. The `+ Hand` condition
keeps the entire scene evidence of the scene-only condition and only adds hand
evidence.

## Evaluation unit

One **sequence-camera video** produces one focal estimate and gets one vote. A
video's frames are repeated observations of one static camera, not independent
samples, so a 381-frame sequence does not outvote a 174-frame one.
