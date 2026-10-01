# CAM-EXP-009 — real-data protocol (NOT EXECUTED)

This protocol was specified so that the gate decision would be honest: it is
recorded here as what *would* have run, and it did not run.

```
REAL GIGAHANDS PHASE: NOT RUN
reason: BILATERAL_FOCAL_SIGNAL_NOT_IDENTIFIABLE_SYNTHETICALLY
```

The GigaHands reference focal was never read by this experiment.

## Conditions that would have run

| id | term enabled |
| --- | --- |
| R0 | scene only |
| R1 | scene + temporal separate-side geometry (the CAM-EXP-008 analogue) |
| R2 | scene + bilateral |
| R3 | scene + temporal + bilateral |

with the ablations R0 vs R1, R0 vs R2, R0 vs R3 and **R1 vs R3** — the last
being the one that would have isolated whether the bilateral constraint adds
anything beyond the temporal hand term.

## Shared design

* population: bimanual-clean frames, chosen on pre-existing QC and data
  availability only
* identical frames, scene predictions, nuisance parameters and candidate grid
  across R0-R3; only the hand terms differ, machine-verified
* primary validation: leave-one-physical-camera-out, lambda selected on
  training cameras only
* controls: wrong bone correspondence, right-hand frame shuffle, right-hand
  view shuffle, fixed-length invariant control
* `CONSTANT_RIG_FOCAL_ORACLE` retained as a sanity control, since the rig's
  reference focal varies by only ~1.7-1.9 %

## Why running it anyway would have been wrong

The synthetic phase showed the objective is minimised at the search boundary in
90-100 % of trials under ideal conditions. On real data, with a noisier
articulation source and a scene anchor to fight against, the result would have
been uninterpretable: any apparent gain could not have been attributed to
bilateral structure, and any loss would have been unsurprising. The gate exists
precisely to prevent spending real-data compute on that.
