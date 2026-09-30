# Open issues — CAM-EXP-011

> A post-run accounting and wording correction was applied after commit
> `0c523d6`. **No inference was re-run and no focal number changed.** See
> `results/summary/hand_score_accounting_audit.json` and
> `tables/paired_set_accounting.csv`.

## 1. The lambda ceiling is the binding constraint, and it was left alone

Moving the chosen candidate by one grid step needs lambda ~ 13.4; the
pre-registered grid stops at 4.0. So this run measures *the hand term under the
inherited CAM-EXP-009.4 fusion*, not the maximum contribution hand geometry
could make.

The grid and the sigma definition were both pre-registered. Widening lambda
after seeing the term was inert would have been tuning the protocol to the
outcome, so it was not done. But the limitation should be stated plainly rather
than presented as "hand geometry does nothing".

## 2. In the all-frame regime, the scene term dominated the frozen fusion scale

Measured facts:

| quantity | measured |
| --- | ---: |
| `sigma_scene`, median across videos | **0.0200** (= the frozen floor) |
| scene penalty for one candidate grid step | 0.1303 |
| hand penalty change over that same step | 0.00974 |
| lambda needed to move one step | **~ 13.4** |
| frozen lambda grid maximum | 4.0 |
| normalised hand-score range across the window | ~2x (the score is NOT flat) |
| lambda = 0 selected | **11 of 15 outer folds**, TRAIN cameras only |

So in the evaluated all-frame regime the scene score operated at the frozen
sigma floor, and its local candidate penalty was substantially larger than the
hand-term penalty under the pre-specified lambda grid.

**CAM-EXP-011 does not identify frame count as the cause of this scale
imbalance.** The run did not compare an identical hand fusion at N = 8, N = 64
and all frames under otherwise matched conditions, so nothing here isolates N.
Determining whether frame count changes the relative scene/hand influence
requires a separate controlled-N experiment.

## 3. The nested tuning turned the hand term off by itself

lambda = 0 was selected in 11 of 15 outer folds, on TRAIN cameras only. That is
the procedure working: it found no benefit on training data and disabled the
term. It also means the aggregate zero is not an artifact of one bad fold.

A cheap re-verification on the frozen artefacts confirms lambda = 0 reproduces
the scene-only focal exactly: 153 videos, 0 mismatches, max relative difference
0.000e+00.

## 4. Two bugs were fixed before the focal was opened

Both found by smoke-testing the fusion, not by a result looking wrong:

- `q_grid` did not contain q = 1.0, so lambda = 0 did not reproduce scene-only
  (-0.65 % offset on every `+ Hand` estimate).
- the hand score was normalised over the whole 400-2400 px grid instead of the
  candidate window, crushing the local variation the fusion sees.

These are pipeline-integrity notes. They are recorded because a null produced
by a pipeline carrying a systematic offset would not be trustworthy; they are
not part of the main result.

## 5. The correction is inherited and is not itself validated here

It comes from the CAM-EXP-009.4 line. That run already found its gain on this
rig small and confounded by GigaHands' narrow focal range, where a constant
scored 0.869 %. CAM-EXP-011 measures what it does across three estimators on
all frames; it does not claim the correction is a solved method.

## 6. Dataset scope

One capture rig, 5 sequences, 4 participants, 175 videos, and a reference focal
range spanning under 2 %. A null here constrains what can be said about this
dataset, not about camera calibration in general.

## 7. The 24 videos outside the paired set — exact accounting

The earlier draft of this file said "15 insufficient + 8 no usable hand", which
summed to 23 and was wrong on both counts. The audited ledger
(`tables/paired_set_accounting.csv`, one row per usable video) gives:

| primary reason | videos |
| --- | ---: |
| INSUFFICIENT_LEFT_HAND_FRAMES | 7 |
| NO_USABLE_HAND | 7 |
| INSUFFICIENT_RIGHT_HAND_FRAMES | 6 |
| INSUFFICIENT_BOTH_SIDES | 2 |
| HAND_GRID_DOES_NOT_COVER_CANDIDATE_WINDOW | 2 |
| **TOTAL EXCLUDED** | **24** |
| FINAL PAIRED | 151 |
| TOTAL USABLE VIDEOS | **175** |

Each excluded video carries exactly one primary reason, and
151 + 24 = 175 is machine-checked.

The last category is a limitation of this run's design worth naming: the hand
score is computed on a fixed **absolute** grid of 400-2400 px. For two
`p41-plant-0004` videos GeoCalib's scene focal is extreme (4853 px and 252 px),
so the candidate window `f_scene x [0.5, 1.5]` falls entirely outside that grid,
the interpolation returns NaN, and GeoCalib's `+ Hand` cell is missing. The
video is then dropped from the six-condition set for **all** models equally, so
the comparison stays paired — but the exclusion is correlated with GeoCalib
producing an extreme estimate, and that should not be hidden.

## 8. 152 vs 153 — an intermediate log, not two definitions

An intermediate merge log reported "152 eligible"; the final count is **153**.
The difference is not a definition change:

- 175 usable videos; 7 have no usable hand frame at all, so 168 were attempted.
- 168 hand-score cache files exist on disk; **153** are eligible.
- The shard index files hold 167 rows (152 eligible) because **one video was
  computed by the single-threaded run before sharding began**, and its index row
  was overwritten when the shard indices were merged. Its cache file survived.

So 153 is correct and 152 was an incomplete intermediate state. The cache on
disk, not either log, is the source of truth.

## 9. Counts that must not be conflated

`SEQUENCE_TOTAL_FRAMES`, `SCENE_INPUT_FRAMES`, `HAND_AVAILABLE_FRAMES`,
`HAND_SIDE_OBSERVATIONS` and `HAND_SCORE_USED_FRAMES` are five different
quantities.

In particular **44,291 any-hand-available frames is not the same as the 42,593
frames that actually entered the hand objective** — the difference is frames
belonging to videos whose hand score was not computable. And left + right
side-observations (35,854 + 36,240) double-count the 27,803 frames with both
hands, so they are *side observations*, not frames.

Two different coverage numbers also exist and are not interchangeable:
frame-level hand coverage **44,291 / 52,423 = 84.5 %**, and video-level paired
coverage **151 / 175 = 86.3 %**.

Video lengths vary between cameras inside a sequence, so a single number per
sequence is a median, not an exact length. See
`tables/presentation_frame_accounting_corrected.csv`.
