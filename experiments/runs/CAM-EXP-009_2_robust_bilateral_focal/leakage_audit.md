# Leakage audit

## Never used as an input

| quantity | status |
| --- | --- |
| the true / reference focal | never passed to any solver call; checked at AST level by `self_audit.py` (`gt_focal_passed_to_solver`) |
| GT principal point | not a primary input |
| GT distortion | the synthetic generator's own distortion is known by construction; on real data it is not a primary input |
| any absolute hand-size prior | none; both proportion vectors are sum-normalised |
| the FINAL_CONFIRMATORY_HOLDOUT (InterHand2.6M / HanCo) | never opened |

## The one declared oracle

`C4_FIXED_LENGTH_INVARIANT` is handed the true bone lengths. It is a
**diagnostic, not a method**: it could never be run on real data, and its only
function is to reconfirm CAM-EXP-009's exact result that a score built from
candidate-independent lengths cannot vary with the candidate focal. It is
excluded from the AST leakage check by name, and the exclusion is recorded in
`self_audit.py`.

## Selection hygiene

- The method family M0-M6 was frozen before the DEV run.
- The winner was chosen on DEV seeds only, by a deterministic pre-registered
  rule, and frozen to `cam_exp_0092_selected_method_v1.json`.
- TEST seeds are disjoint from DEV seeds; asserted at freeze time and rechecked
  in the audit.
- Trial counts were fixed from measured runtime, before any result.
- No frame, cell or threshold was ever chosen by looking at a result. The frame
  FIT/EVAL split is by parity and does not depend on fit quality.

## Prior runs

CAM-EXP-001 through CAM-EXP-009.1 are read-only. The CAM-EXP-009.1 solver is
imported, never copied or edited; `self_audit.py` runs `git status` over that
run directory and fails if anything there is modified.
