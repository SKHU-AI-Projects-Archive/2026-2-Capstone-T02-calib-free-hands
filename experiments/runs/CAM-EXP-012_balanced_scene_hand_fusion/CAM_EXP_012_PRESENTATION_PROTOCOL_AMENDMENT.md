# CAM-EXP-012 — presentation protocol amendment

**Written BEFORE any CAM-EXP-012 test reference focal was opened.**

At the time of writing:

| check | state |
| --- | --- |
| `results/raw/REFERENCE_FOCAL_OPENED.txt` | **does not exist** |
| `results/raw/frozen_test_predictions.csv.gz` | **does not exist** |
| correct-hand curves computed | 76 of 153 (still running) |
| test reference focal read | **NO** |
| any CAM-EXP-012 focal error computed | **NO** |

Nothing in this amendment could have been informed by a result, because no
result exists yet.

## Why this amendment exists

A presentation deadline. The cross-fitted hand-curve computation is far more
expensive than projected — about 6.95 minutes per video in aggregate across 8
workers, so roughly 17.7 hours per variant. Computing both the correct and the
wrong-bone variants before the deadline is not possible.

This is a **scheduling** amendment. It does not abandon the CAM-EXP-012
protocol and it does not weaken any pre-registered rule.

## What is unchanged

The computation currently running is **not touched, not restarted and not
modified**. Every method parameter stays exactly as frozen:

- the correct-hand curve computation continues to completion, unchanged
- alpha formula unchanged: fixed `delta_log_f = 0.05`,
  `q- = exp(-0.05)`, `q+ = exp(+0.05)`, `alpha = median(D_scene) / median(D_hand)`
  over TRAIN videos only, never seeing a reference focal
- lambda grid unchanged: `{0.125, 0.25, 0.5, 1, 2, 4, 8}`, with **lambda = 0
  still excluded** from Hand-ON selection
- q-grid unchanged: 201 points containing exactly 1.0
- global absolute focal grid unchanged: 126.2–7280.2 px, 401 points
- temporal-block cross-fit unchanged: block = 8 frames, >= 4 observations per
  fold per side
- primary video set unchanged: the frozen 153-video manifest
- shuffled-hand donor mapping unchanged: already frozen, 153/153 donors, all
  different sequence, different camera, different participant
- nested physical-camera CV unchanged
- **no result-dependent selection of any kind has been or will be performed**

## What changes

**The wrong-bone control is DEFERRED, not cancelled.**

It will be computed after the presentation using the already-frozen
specification — the same within-finger bone permutation, the same global grid,
the same cross-fit, and critically the **same `alpha_correct` and the same
selected lambda** as the correct condition. Having seen the correct result
first will not change any of those, because they are all frozen now.

## What the presentation may and may not claim

The PRIMARY numerical question is unchanged:

> Does `Scene + Correct Hand` reduce focal error relative to `Scene Only`, on
> unseen physical cameras?

Two verdicts may be used at presentation time:

| tag | condition |
| --- | --- |
| `HAND_ON_NUMERICAL_GAIN_SUPPORTED` | Correct median error < Scene, paired median gain > 0, and camera-cluster bootstrap CI lower bound > 0 |
| `CORRECT_VS_SHUFFLED_SIGNAL_SUPPORTED` | Correct beats Shuffled in the paired comparison with CI lower bound > 0 |

One verdict **may not** be claimed:

| tag | state |
| --- | --- |
| `CORRECT_HAND_SPECIFIC_SIGNAL_SUPPORTED` | **PENDING** — requires the wrong-bone control |

So even if the correct condition beats both scene-only and the shuffled
control, the presentation will **not** assert that the improvement is due to
correct anatomical bone correspondence. The shuffled control rules out *some*
alternative explanations — it shows another video's hand curve does not produce
the same effect — but only the wrong-bone control isolates bone correspondence
while holding the same video's observations fixed.

## Status wording

Until the wrong-bone control is complete, the correct status is:

```
CAM-EXP-012  PRIMARY PRESENTATION ANALYSIS COMPLETE
             FULL CONTROL ANALYSIS PENDING
```

**`CAM-EXP-012 COMPLETE` must not be written** before the wrong-bone control
finishes.

## Order of operations from here

1. wait for all 153 correct-hand curves and verify their integrity
2. TRAIN-only alpha
3. nested inner camera CV for a positive lambda
4. outer TEST predictions for Scene-only, Correct Hand, Shuffled Hand
5. **freeze and hash those predictions** while the test focal is still closed
6. only then open the test reference focal
7. evaluate and produce the presentation tables

The shuffled control requires **no new solver computation** — it reuses the
correct curves through the frozen donor mapping, with the same alpha and the
same lambda. No shuffled-specific alpha and no shuffled-specific lambda tuning.
