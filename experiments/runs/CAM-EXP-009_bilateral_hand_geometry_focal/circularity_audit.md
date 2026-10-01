# CAM-EXP-009 — circularity audit

## 1. Synthetic phase (the only phase executed)

| quantity | used in generation | used by the solver |
| --- | :-: | :-: |
| f_true | YES | **NO** |
| true bone-length vectors | YES | **NO** (the fit starts uniform) |
| true per-frame pose R, T | YES | **NO** (re-estimated by PnP) |
| camera distortion | YES | supplied as a fixed nuisance, as it would be on real data |
| 2D observations | — | YES |
| bone directions | — | YES |

The synthetic true focal is used only to generate the images and to score the
recovered focal afterwards. It never enters the objective.

## 2. Real phase

Not run. The GigaHands reference focal was never read.

Had it run, the contract would have been the one inherited from CAM-EXP-008:
the target camera excluded from the reference articulation before
reconstruction, its 2D used only as the observation to be explained, no target
extrinsics, and no provided focal magnitude anywhere in the optimisation.

## 3. The distinction that matters in this run

There is a subtler circularity specific to CAM-EXP-009, and it is the reason
the primary formulation is built the way it is. If the reference bone LENGTHS
were carried into the objective, the bilateral comparison would be testing a
quantity that was already determined before the candidate focal was chosen —
and it would return the same answer at every focal. That is not circular in the
leakage sense, but it is vacuous, and the focal-dependence audit measures it as
exactly zero variation.
