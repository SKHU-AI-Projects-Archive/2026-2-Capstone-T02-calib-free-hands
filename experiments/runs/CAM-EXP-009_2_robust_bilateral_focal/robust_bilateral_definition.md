# The robust bilateral score family M0-M6

## Why a robust score at all

CAM-EXP-009.1 established, after fixing three solver faults, that the
candidate-conditioned bilateral cue recovers a known synthetic focal to 0.035 %
when the two hands are *identical*. It also established that this collapses
quickly once they are not: 2 % bilateral asymmetry took the error to about
12 %, and 2D noise degraded it similarly.

Real left and right hands are not identical. **This run does not assume they
are.** The question is whether a comparison that tolerates a few strongly
mismatched bones keeps the focal information that a plain L1 distance loses.

## What every method shares

All seven methods consume the *same* candidate-conditioned bone proportions
`p_L(q)` and `p_R(q)`, fitted once per (trial, candidate focal) by the
CAM-EXP-009.1 solver, imported unchanged under `UNDISTORT_ONCE_INTERNAL` with
`N_ITER = 32` and the D20 parameterisation. No method receives an input another
method did not receive, so any difference between them is the score, not the
fit.

Both vectors are normalised to sum to 1, so absolute hand size is never a cue
and a person with larger hands is not penalised.

## The family

| method | formula | what it tolerates |
| --- | --- | --- |
| M0_BASELINE_RAW_L1 | `mean_b |p_L,b - p_R,b|` | nothing - the CAM-EXP-009.1 baseline |
| M1_LOG_L1 | `mean_b |log p_L,b - log p_R,b|` | puts large and small bones on equal footing |
| M2_LOG_MEDIAN | `median_b |d_b|` | a minority of strongly asymmetric bones |
| M3_LOG_TRIMMED | mean of the smallest 80 % of `|d_b|` | the 4 worst-matching bones of 20 |
| M4_HUBER_001/002/005 | `mean huber(d, tau)` | graded down-weighting above tau |
| M5_FINGER_INTERNAL_RATIO | `median_b |log r_L - log r_R|`, r normalised inside each finger | a per-finger length difference between the sides |
| M6_FINGER_HUBER | `mean huber(finger-internal d, 0.02)` | both of the above |

`d_b = log(p_L,b + eps) - log(p_R,b + eps)`.

M0 is included as the comparison baseline, not as a robust method. M1 is not
robust either; it is the log-space reference that isolates how much of any gain
comes from the log transform rather than from the robust loss.

## The family is closed

These nine entries were frozen in
`cam_exp_0092_robust_candidate_spec_v1.json` before the DEV run. No additional
loss, no additional `tau` and no additional trim fraction may be introduced
after a result has been seen. The trim fraction (0.20) and the three `tau`
values were fixed at the same time.

## A global LEFT/RIGHT scale parameter is redundant, not omitted

One could imagine adding a nuisance parameter `r_LR` for an overall size
difference between a person's two hands. It would do nothing here: both
proportion vectors are constrained to sum to 1, so multiplying one side by any
scalar and renormalising returns the identical vector.

This was verified numerically rather than asserted -
`audit_r_LR_redundancy()`, 200 random pairs, scale factors drawn from
[0.5, 2.0]: the largest score change across all nine methods was
**5.68e-14**. The `A_GLOBAL_SIDE_SCALE` stress family in the synthetic generator
exists to confirm the same thing end to end, through the solver.

`results/summary/r_lr_redundancy_audit.json`.
