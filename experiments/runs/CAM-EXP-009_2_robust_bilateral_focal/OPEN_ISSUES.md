# Open issues — CAM-EXP-009.2

## 1. `C3_TEMPORAL_SHUFFLE` was vacuous by construction (design fault)

The objective fits `p_L` from all left frames and `p_R` from all right frames
independently and compares only the resulting proportion vectors. It never
pairs a left frame with a right frame, so a temporal shuffle has nothing to
destroy and collapses exactly onto `C2_SUBJECT_SWAP` — bit-identical per
subject.

This should have been caught when the control set was designed. It is recorded
as `CONTROL_VACUOUS_BY_CONSTRUCTION` and given no evidential weight. It was not
replaced after the fact, because inventing a control once its pre-registered
form has been seen to be uninformative is a search, not a test.

*If a future run wants a genuine temporal control*, it must first make the
objective frame-paired — otherwise there is nothing to shuffle.

## 2. The C2 result is established at one stress level only

`C2_SUBJECT_SWAP` failing to degrade is the finding that decides this run, and
it was measured at `COMBINED_MODERATE` with 8 subjects. It was not run across
the stress grid, and not on the clean cell — where the cue does recover the
focal exactly and C2 might well degrade as it should. So the claim is
specifically "at moderate stress the residual preference is not bilateral", not
"the cue is never bilateral".

Running C2 across the grid would be a reasonable pre-registered follow-up. It
must be pre-registered, not run now to see whether it rescues the verdict.

## 3. The DEV selection is not stable across the stress space

`M4_HUBER_005` won the 7-cell DEV set, but on TEST it loses to the plain L1
baseline on sparse asymmetry (−218 % at the 2 % level) and on 2D noise (−53 %
at 2 px). The family members are not consistently ordered, so a different DEV
composition would plausibly have selected a different winner. The single-winner
protocol is correct for avoiding selection bias, but it means the reported
method is one draw from an unstable ranking.

## 4. The headline bootstrap intervals overlap

`COMBINED_MODERATE`: winner 10.78 % [7.02, 21.69], M0 15.58 % [10.16, 22.76].
Gate g3 passed on the point estimate (30.8 % relative reduction) while the
intervals overlap across most of their range. g3 is reported as passed because
that is what the frozen criterion said, but it should not be read as a
well-separated improvement, and the report says so.

## 5. Several TEST cells do not test what their names suggest

`MISSING`, `VISIBILITY` and `DIST` were specified with `asym = 0`, `noise = 0`,
`artic = 0`. The two hands in those cells are therefore identical and the
problem reduces to the clean case — which is why all of them return the grid
floor of 0.272 % for both methods. They are valid evidence that the cue
tolerates missing joints, sparse observation and distance *in isolation*, and
they are not evidence about those stresses combined with asymmetry. Only the
two `COMBINED` cells test combinations.

A future grid should cross the observability axes with a non-zero asymmetry
level.

## 6. The grid quantisation floor is 0.272 %, not 0

The candidate grid has a 0.689 % step and the true focal does not sit on a grid
point, so a perfect estimate reads as 0.272 % error. Gate g1 (≤ 1 %) is
therefore only about 1.5 grid steps wide. Any future run wanting to resolve
sub-0.3 % differences needs a finer grid or a local refinement step; this one
did not, because nothing came close to that floor under stress.

## 7. Not addressed by this run

- Whether a jointly constrained bilateral formulation (shared low-dimensional
  anatomical subspace, or a cross-subject prior on corresponding bones) avoids
  the degeneracy. Explicitly untested — §7 of the report.
- Anything about real GigaHands data. `REAL_FOCAL_PHASE_NOT_RUN`.
- Whether the observed synthetic behaviour transfers to real data at all. The
  generator uses the solver's own hand and camera model.
