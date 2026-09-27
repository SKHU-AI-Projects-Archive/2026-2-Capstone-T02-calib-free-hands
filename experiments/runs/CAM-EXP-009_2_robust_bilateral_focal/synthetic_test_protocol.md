# TEST protocol — the reported synthetic evidence

## What runs

Only two methods: the DEV winner and `M0_BASELINE_RAW_L1`. Running all nine
here and then quoting whichever did best would turn TEST into a second
selection set.

29 cells, 16 subjects for primary cells and 8 for secondary, 400 trials in all.
Seeds are disjoint from DEV by construction.

## The candidate grid is not centred on the answer

The grid is anchored on an arbitrary constant `F_NOMINAL_SYNTH = 1000`, while
the true synthetic focal is 900. The truth therefore sits at q = 0.9, away from
the centre of the [0.5, 1.5] search range, so a method cannot score well merely
by preferring the middle of the grid. A minimum at either end of the grid is
recorded as a **boundary** outcome and counted as a failure to estimate, not as
an estimate.

## Cells

- `TEST_CLEAN`
- `TEST_ASYM_GLOBAL_01`
- `TEST_ASYM_GLOBAL_02`
- `TEST_ASYM_GLOBAL_05`
- `TEST_ASYM_DENSE_01`
- `TEST_ASYM_DENSE_02`
- `TEST_ASYM_DENSE_05`
- `TEST_ASYM_SPARSE_01`
- `TEST_ASYM_SPARSE_02`
- `TEST_ASYM_SPARSE_05`
- `TEST_ASYM_FINGER_01`
- `TEST_ASYM_FINGER_02`
- `TEST_ASYM_FINGER_05`
- `TEST_NOISE_0.5PX`
- `TEST_NOISE_1.0PX`
- `TEST_NOISE_2.0PX`
- `TEST_ARTIC_0.5DEG`
- `TEST_ARTIC_1.0DEG`
- `TEST_ARTIC_2.0DEG`
- `TEST_MISSING_10`
- `TEST_MISSING_20`
- `TEST_VIS_75`
- `TEST_VIS_50`
- `TEST_VIS_25`
- `TEST_DIST_2X`
- `TEST_DIST_8X`
- `TEST_DIST_16X`
- `TEST_COMBINED_MODERATE`
- `TEST_COMBINED_STRONG`

## The gate, frozen before the DEV run

| gate | criterion | required |
| --- | --- | --- |
| g1 | clean cell median error <= 1.0 % | no |
| g2 | COMBINED_MODERATE median error <= 5.0 % | **yes** |
| g3 | >= 30 % relative reduction vs M0 on COMBINED_MODERATE | **yes** |
| g4 | boundary rate <= 0.10 on COMBINED_MODERATE | no |
| g5 | FIT vs EVAL median error differ by <= 5.0 pp | no |
| g6 | wrong bone mapping degrades by >= 10 pp | **yes** |
| g7 | subject swap degrades by >= 10 pp | **yes** |
| g8 | fixed-length control curve is exactly flat | no |

If the required gates do not all pass, the recorded status is
`REAL_FOCAL_PHASE_NOT_RUN` and the GigaHands phase does not run. That rule was
written before any DEV or TEST result existed.
