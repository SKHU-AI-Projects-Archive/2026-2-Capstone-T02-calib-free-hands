# CAM-EXP-009 — synthetic protocol and gate

## 1. Purpose

Establish whether the bilateral objective can identify a KNOWN focal before any
real data is touched. A synthetic gate is cheap; an uninterpretable multi-hour
real run is not.

## 2. Generation

Deterministic, seeded. A synthetic subject has a LEFT and a RIGHT bone-length
vector that are related but not identical. Per frame: a rest pose plus
articulation jitter, a random rigid pose, projection through a known K with
realistic radial distortion.

The solver receives only the 2D observations, the bone directions and the image
size. It is never given f_true, R, T or the true bone vectors
(`tables/leakage_audit.csv`).

## 3. Sweeps

| sweep | values |
| --- | --- |
| distance / hand diameter | 2, 4, 8, 16, 32 |
| bilateral asymmetry | 0 %, 1 %, 2 %, 5 % |
| 2D noise | 0, 0.5, 1, 2, 4 px |
| pose diversity | LOW, MEDIUM, HIGH |

20 trials per condition, 12 frames per trial, 161-point candidate grid.

## 4. Over-fitting guard

Twenty free bone lengths per side is a lot of freedom. Frames are split
deterministically (alternating) into FIT and EVAL; the bone vector is estimated
on FIT frames and the reprojection is measured on both. Held-out reprojection
is reported separately as gate g3.

## 5. Controls

| control | expectation |
| --- | --- |
| `BONE_INDEX_PERMUTATION` | corresponding bones mismapped; should degrade |
| `RIGHT_HAND_SWAP` | right hand of a different subject; should degrade |
| `FIXED_REFERENCE_LENGTH_BILATERAL` | must be flat — this is the control that demonstrates why the naive idea fails |

## 6. The gate, fixed before execution

| id | criterion | required |
| --- | --- | :-: |
| g1 | bilateral objective is not focal-invariant | yes |
| g2 | true focal near the minimum, noiseless (median error <= 5 %) | yes |
| g3 | held-out reprojection identifies the focal | no |
| g4 | wrong bone correspondence degrades | yes |
| g5 | fixed-length control is flat | no |

If the gate fails, the GigaHands real-data phase is NOT run.

## 7. Outcome

g1 PASS, g2 **FAIL** (50.0 %), g3 FAIL, g4 **FAIL**, g5 PASS.

`BILATERAL_FOCAL_SIGNAL_NOT_IDENTIFIABLE_SYNTHETICALLY`. The real phase was not
run.

A note on g4: a control can only degrade performance that exists. The real
correspondence was already pinned at the grid boundary, so mismapping the bones
had nothing left to spoil. g4's failure is a consequence of g2's, not an
independent finding.
