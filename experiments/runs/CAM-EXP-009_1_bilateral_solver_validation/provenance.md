# CAM-EXP-009.1 — provenance

## This run

`CAM0091_PRE_SPECIFIED_BEFORE_CORRECTED_EXECUTION`.

The gates, thresholds and control criterion were fixed before the corrected
sweep executed. They are **not** pre-registered from the start of the project:
CAM-EXP-009 had already produced a synthetic negative result, and this run
exists specifically to check the solver behind it. Calling these criteria
"pre-registered" without that qualifier would be misleading.

## The one change made mid-run

The alternation count went from 8 to 32. This was:

* judged **only** against the positive control - reprojection at the TRUE
  focal from a uniform start;
* decided **before any focal sweep had been executed**;
* measured rather than guessed (8 -> 0.425 px, 16 -> 0.021, 32 -> 0.00067,
  64 -> 0.00061, 128 -> 0.00054).

After it, the solver configuration was frozen in
`cam_exp_0091_solver_validation_spec_v1.json` and nothing in it changed. No
optimiser setting was touched after seeing a focal-sweep result.

## CAM-EXP-009's own chronology

CAM-009's formal synthetic gate was written **after** a preliminary
focal-dependence audit had already revealed the boundary minimum. It was frozen
before the full distance/noise/asymmetry sweep, but not before all synthetic
evidence.

So CAM-009's report should not state that every gate was fully result-blind.
That correction is applied to CAM-009's own documents in a separate commit;
its numbers are not changed.

## Status of CAM-EXP-009's result

`SUPERSEDED_FOR_INTERPRETATION_DUE_TO_SOLVER_VALIDATION_ISSUE`.

Its raw outputs, summaries, figures and frozen manifests are untouched and
remain on record as what that code produced. What is superseded is the
*interpretation* - that the bilateral objective is geometrically
unidentifiable. It is not; three implementation faults made it look that way.
