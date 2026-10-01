# CAM-EXP-006.1 — scientific wording review

Machine-readable: `tables/scientific_claim_status.csv`.

CAM-EXP-006's **numbers are not changed by this review**. Its *claims* are
narrowed to what the evidence now supports.

## 1. Upheld

> the reference hand does not identify the focal under this formulation

R5 — the true principal point, distortion and fy/fx ratio all supplied — still
gives 49.96 % median error, 77.1 % flat profiles, 13.7 % cleanly identifiable
views, and a tracking correlation whose interval contains zero.

## 2. Reproduced but misattributed

> 202 % median relative focal error at N=16

R0 reproduces it exactly. Supplying the dataset-provided distortion
coefficients reduced the N=16 median focal error from 203.13 % to 47.53 % under
the corresponding oracle diagnostic, so unmodelled distortion was a major
contributor to the inflated headline. Both numbers should be reported, and
202 % should stop being quoted as the size of the reference-hand focal limit.

This is a comparison between conditions that differ in what the solver is
given, not an additive causal decomposition. Statements of the form "75 % of
the error was distortion" or "distortion caused exactly X % of the failure" are
**not** supported by this run.

## 3. Overstated

> the limit is geometric, not a modelling shortfall
> it will not be fixed by a better hand model

Modelling mattered a great deal — supplying the provided distortion took the
N=16 median error from 203.13 % to 47.53 %. And hand accuracy is not
irrelevant: a post-hoc sensitivity sweep shows a reference-3D error of 1 % of
hand diameter already costing about 26 % focal error (a
`POST_HOC_SENSITIVITY_DIAGNOSTIC`, not pre-registered evidence).

Replace with:

> Improving hand-geometry accuracy alone is unlikely to resolve the observed
> focal-depth ambiguity in this formulation and regime.

## 4. Logically too strong

> a predicted hand cannot carry more focal information than the oracle hand
> CAM-007 is already ruled out

A learned hand model can encode training priors, image appearance and camera
priors that are not present in explicit 3D geometry. The CAM-006 result
constrains *this profiled-PnP estimator*, not every hand-derived cue.

Replace with:

> The CAM-006 result rules out the assumption that replacing the reference
> geometry with a noisier predicted geometry will solve this same profiled-PnP
> focal problem. It does not establish that learned hand-derived features
> contain no camera information.

## 5. Weak evidence

> the planarized control beating the main condition shows the signal is absent

That control necessarily ran a different PnP back-end. Demote it to sensitivity
evidence. The synthetic planarity sweep is the clean evidence and points the
same way.

## 6. Self-contradictory label

`PRE_REGISTERED_POST_HOC_ORACLE_SANITY_CONTROL` cannot be both. The confound
was found post hoc in CAM-005 and was therefore already known when CAM-006
began, so in CAM-006 it is simply pre-registered.

Use `PRE_REGISTERED_IN_CAM006_ORACLE_SANITY_CONTROL`, motivated by the CAM-005
post-hoc finding.

## 7. Canonical final statement

> CAM-EXP-006.1 reproduces CAM-EXP-006 and confirms that the profiled-PnP
> reference-hand formulation does not reliably identify focal under the
> evaluated GigaHands regime.
>
> However, the original 202 % headline substantially overstated the failure
> magnitude, because the solver ignored substantial lens distortion. In an
> oracle diagnostic using the dataset-provided distortion coefficients, median
> error fell to roughly 48–50 %, while the focal profile remained largely flat
> and did not track the dataset-provided reference focal.
>
> The evidence therefore supports a failure of this formulation under the
> evaluated regime, not a universal absence of camera information in hands.

The real-data failure is consistent with an interaction among limited
perspective leverage, reference-3D / target-2D disagreement, and camera-model
mismatch. CAM-EXP-006.1 does not provide an additive causal decomposition of
these effects.

## 8. The scope sentence to use

> Under the evaluated GigaHands imaging regime and this profiled-PnP
> formulation, other-camera-only reference hand geometry does not provide
> enough perspective information to identify the focal reliably, even when the
> rest of the camera model is supplied exactly. This does not establish that
> hand geometry carries no camera information in other regimes, nor that
> learned hand-derived features encoding camera priors are ruled out.


## 9. Evidence level of the statements above

The upheld claim (§1) and the reproduction/attribution numbers (§2) rest on
**pre-registered** conditions R0–R5 and the frozen materiality thresholds.

The narrowing in §3 draws partly on a **post-hoc sensitivity diagnostic** (the
reference-3D perturbation sweep). That diagnostic is explanatory; the narrowing
itself is a wording judgement, not a statistical claim.

The B-vs-C verdict classification is a `POST_HOC_INTERPRETIVE_VERDICT`. See
[`analysis_provenance.md`](analysis_provenance.md) §3.
