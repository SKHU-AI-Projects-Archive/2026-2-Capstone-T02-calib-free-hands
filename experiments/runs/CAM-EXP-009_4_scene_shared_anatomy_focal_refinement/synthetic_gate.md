# Synthetic implementation gate

## Purpose

To check the new joint-anatomy solver is geometrically sound **before** any real
reference focal is opened. Passing does **not** mean the method works on real
data; it means the implementation is sane enough to justify the real evaluation.

The CAM-EXP-009.2 synthetic generator is imported unchanged and none of its
outputs are overwritten.

## Conditions and results

| condition | median focal error | boundary rate |
| --- | ---: | ---: |
| S0 CLEAN | **0.272 %** | 0.00 |
| S1 MODERATE (CAM-EXP-009.2 `COMBINED_MODERATE`) | **3.420 %** | 0.00 |
| S2 WRONG_BONE_MAPPING | 36.541 % | 0.38 |
| S3 GENERIC_PRIOR_OFF | 9.024 % | 0.00 |

`S0`'s 0.272 % is the candidate-grid quantisation floor, i.e. an exact hit at
the resolution of the frozen grid.

## Gate

| gate | criterion | result | required |
| --- | --- | --- | --- |
| G0 | projection / undistortion roundtrip clean | PASS | - |
| G1 | clean focal error <= 1 % | **PASS** | yes |
| G2 | moderate focal error <= 5 % | **PASS** | yes |
| G3 | wrong bone mapping degrades by >= 10 pp | **PASS** (+33.1 pp) | yes |
| G4 | boundary rate <= 25 % | **PASS** | - |

**GATE_PASSED — the real phase was eligible to run.**

## Two things the gate also showed

- The **generic prior earns its place**: removing it (S3) takes the moderate
  error from 3.420 % to 9.024 %.
- The score genuinely uses anatomical bone correspondence: permuting bones
  within each finger costs +33.1 pp.

## Regularisers

`lambda_generic = 1.0`, `lambda_side = 1.0`, selected here and **only** here.
They were never tuned against the real reference focal.
