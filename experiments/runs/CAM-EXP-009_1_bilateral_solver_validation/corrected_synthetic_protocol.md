# CAM-EXP-009.1 — corrected synthetic protocol

## Order, enforced

1. G0 camera round trip
2. G1 true-geometry projection and PnP
3. G2 bone fitter at the true focal
4. G3 determinism across processes
5. G4 ideal-vs-raw scoring consistency
6. **freeze the solver configuration**
7. no-distortion focal sweep
8. realistic-distortion focal sweep
9. DOF ablation, controls

The corrected sweep runs only if G0-G4 all pass; otherwise
`FULL_SWEEP_NOT_RUN`.

## Per candidate focal

```
l(q)   fitted on FIT frames only
FIT reprojection         on the frames it was fitted to
HELD-OUT reprojection    the SAME l(q), scored on EVAL frames
BILATERAL_FIT            || l_L(q) - l_R(q) ||, from the FIT fits
BILATERAL_EVAL           from an independent refit on EVAL frames
ORACLE_TRUE_SHAPE        true shape fixed, pose only
```

The **held-out** score is the primary identifiability diagnostic, because free
bone proportions can absorb a wrong focal on the frames they were fitted to.
FIT and EVAL are a deterministic alternating split, independent of any target.

## Conditions

distance 2/4/8/16/32 diameters, asymmetry 0/1/2/5 %, noise 0/0.5/1/2/4 px, pose
diversity LOW/MED/HIGH, shape DOF D20/D10/D5, and the controls.

8 trials per condition, reduced from CAM-009's 20 on measured cost (38.4 s per
trial; 4.3 h at 20 versus 1.7 h at 8). Decided before any sweep result and
applied uniformly, so no condition has more power than another.

## Controls

| id | expectation |
| --- | --- |
| correct bone mapping | should identify the focal |
| wrong bone mapping | should degrade |
| right hand from another subject | should degrade |
| fixed reference lengths | must stay focal-invariant |

Criterion frozen in advance: correct mapping must beat both wrong mapping and
subject swap by >= 10 percentage points of absolute focal error, or by >= 20 pp
of within-5 %.
