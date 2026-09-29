# Open issues — CAM-EXP-011

## 1. The lambda ceiling is the binding constraint, and it was left alone

Moving the chosen candidate by one grid step needs lambda ~ 13.4; the
pre-registered grid stops at 4.0. So this run measures *the hand term under the
inherited CAM-EXP-009.4 fusion*, not the maximum contribution hand geometry
could make.

The grid and the sigma definition were both pre-registered. Widening lambda
after seeing the term was inert would have been tuning the protocol to the
outcome, so it was not done. But the limitation should be stated plainly rather
than presented as "hand geometry does nothing".

A properly pre-registered follow-up could ask what a wider lambda or a
differently-scaled scene term does. That is a new experiment, not a patch here.

## 2. More frames made the auxiliary cue *less* able to contribute

Slightly counter-intuitive and worth remembering. With all 381 frames of a
video the per-frame log-MAD of a good scene estimator falls to the frozen sigma
floor (0.02), which makes the scene penalty steep near q = 1. The hand score is
not flat - it varies about 2x across the candidate window - but its local slope
near the scene estimate is an order of magnitude too small to overcome that.

So "use every frame" tightened the thing the auxiliary cue would have had to
beat. An 8-frame or 64-frame comparison would have given the hand term more
room, purely because the scene aggregate is looser there.

## 3. The nested tuning turned the hand term off by itself

lambda = 0 was selected in 11 of 15 outer folds, on TRAIN cameras only. That is
the procedure working: it found no benefit on training data and disabled the
term. It also means the headline zero is not an artifact of one bad fold.

## 4. Two bugs were fixed before the focal was opened

Both found by smoke-testing the fusion, not by a result looking wrong:

- `q_grid` did not contain q = 1.0, so lambda = 0 did not reproduce scene-only
  (-0.65 % offset on every `+ Hand` estimate).
- the hand score was normalised over the whole 400-2400 px grid instead of the
  candidate window, crushing the local variation the fusion sees.

Neither changed the conclusion, but a null reported from a pipeline carrying a
-0.65 % systematic offset would not have been trustworthy.

## 5. The correction is inherited and is not itself validated here

It comes from the CAM-EXP-009.4 line. That run already found its gain on this
rig small and confounded by GigaHands' narrow focal range, where a constant
scored 0.869 %. CAM-EXP-011 measures what it does across three estimators on all
frames; it does not claim the correction is a solved method.

## 6. Dataset scope

One capture rig, 5 sequences, 4 participants, 175 videos, and a reference focal
range spanning under 2 %. A null here constrains what can be said about this
dataset, not about camera calibration in general.

## 7. 24 videos are outside the paired set

15 have fewer than 8 usable hand frames on a side and 8 have no usable hand at
all; 151 of 175 remain. They were excluded by the pre-registered hand-coverage
rule, never for performance.

## 8. Counts that must not be conflated later

`SEQUENCE_TOTAL_FRAMES`, `SCENE_INPUT_FRAMES` and `HAND_AVAILABLE_FRAMES` are
different quantities and are stored in separate columns. Tea is a 381-frame
sequence whose scene branch used all 381 per camera and whose hand branch used
about 308 per camera. Writing "Tea = 308 frames" would be wrong.

Also: the measured total is 52,423, not the historical 52,445. Frame counts vary
by +/-1-2 between cameras inside a sequence; the historical figure assumed one
count per sequence. Neither number is an error - they measure slightly different
things - but the measured one is what this run used.
