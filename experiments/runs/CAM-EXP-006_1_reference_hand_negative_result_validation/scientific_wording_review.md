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

R0 reproduces it exactly. But most of that magnitude was unmodelled lens
distortion. The camera-model-corrected figure is about 50 %. Both numbers
should be reported, and 202 % should stop being quoted as the size of a
geometric limit.

## 3. Overstated

> the limit is geometric, not a modelling shortfall
> it will not be fixed by a better hand model

Modelling mattered a great deal — supplying the provided distortion cut the
error by 75 % in relative terms. And hand accuracy is not irrelevant: the
synthetic sweep shows a reference-3D error of 1 % of hand diameter already
costs about 26 % focal error.

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

## 7. The sentence to use

> Under the evaluated GigaHands imaging regime and this profiled-PnP
> formulation, other-camera-only reference hand geometry does not provide
> enough perspective information to identify the focal reliably, even when the
> rest of the camera model is supplied exactly. This does not establish that
> hand geometry carries no camera information in other regimes, nor that
> learned hand-derived features encoding camera priors are ruled out.
