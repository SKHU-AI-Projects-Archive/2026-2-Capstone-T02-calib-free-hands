# CAM-EXP-006.1 — analysis provenance

This document separates what was **frozen before any result was seen** from what
was **added after seeing results**. No numerical result is changed by this
document; it is a record of evidence level and time order.

Authority for "frozen": `experiments/manifests/cam_exp_0061_validation_spec_v1.json`
and `cam_exp_0061_real_conditions_v1.json`, both carrying
`created_before_results = true`. An analysis counts as pre-registered only if it
appears in those files. Anything else is post-hoc, regardless of how reasonable
it is.

**Post-hoc does not mean unreliable.** The post-hoc analyses here were added to
understand *why* the pre-registered test failed. They support mechanistic
interpretation. They are not used as confirmatory evidence for the primary
verdict.

## 1. Summary table

| analysis | planned before result | added after result | role | used for primary verdict | notes |
| --- | :-: | :-: | --- | :-: | --- |
| R0 ORIGINAL_REPRODUCTION | yes | no | CONFIRMATORY | **yes** | in `cam_exp_0061_real_conditions_v1.json` |
| R1 FRAME_BALANCED | yes | no | CONFIRMATORY | **yes** | ditto |
| R2 GT principal point | yes | no | ORACLE_DIAGNOSTIC | **yes** | ditto |
| R3 GT distortion | yes | no | ORACLE_DIAGNOSTIC | **yes** | ditto |
| R4 GT PP + distortion | yes | no | ORACLE_DIAGNOSTIC | **yes** | ditto |
| R5 full camera model except focal | yes | no | ORACLE_DIAGNOSTIC | **yes** | ditto |
| SYNTH_GENERIC_STRONG_PERSPECTIVE | yes | no | CONFIRMATORY (solver validation) | **yes** | frozen suite + frozen pass conditions |
| SYNTH_HAND_STRONG_PERSPECTIVE | yes | no | CONFIRMATORY (solver validation) | **yes** | frozen suite |
| distance sweep, noiseless (2–40×) | yes | no | MECHANISM (frozen) | no | `sweeps.distance_over_diameter` |
| planarity sweep (depth scale 1.0–0.0) | yes | no | MECHANISM (frozen) | no | `sweeps.depth_scale` |
| noise sweep (0, 0.5, 1, 2 px) | yes | no | MECHANISM (frozen) | no | `sweeps.noise_px` |
| principal-point synthetic diagnostic | yes | no | MECHANISM (frozen) | no | `sweeps.principal_point` |
| distortion synthetic diagnostic | yes | no | MECHANISM (frozen) | no | `sweeps.distortion` |
| FRAME_WEIGHTING_MATERIAL criterion | yes | no | decision threshold | **yes** | `thresholds` |
| CAMERA_MODEL_MISMATCH_MATERIAL criterion | yes | no | decision threshold | **yes** | `thresholds` |
| REPRODUCTION_MISMATCH criterion | yes | no | decision threshold | **yes** | `thresholds` |
| UNDERPOWERED_STRESS_SUBSET criterion | yes | no | decision threshold | **yes** | `thresholds` |
| S2 / S5 stress subsets | yes | no | CONFIRMATORY | **yes** | threshold frozen |
| **distance × 1 px noise sweep** | **no** | **yes** | `POST_HOC_MECHANISM_DIAGNOSTIC` | no | §2.1 |
| **REAL_REGIME_MATCHED (4.12× + 7.21 px)** | **no** | **yes** | `POST_HOC_MECHANISM_DIAGNOSTIC` | no | §2.2 |
| **reference-3D perturbation sweep** | **no** | **yes** | `POST_HOC_SENSITIVITY_DIAGNOSTIC` | no | §2.3 |
| ORACLE_PERSPECTIVE_STRENGTH_DIAGNOSTIC | partial | partial | MECHANISM | no | §2.4 |
| **B-vs-C usability cutoff** | **no** | **yes** | `POST_HOC_INTERPRETIVE_VERDICT` | **yes (interpretation)** | §3 |

## 2. Post-hoc mechanism diagnostics

### 2.1 Distance × 1 px noise sweep — `POST_HOC_MECHANISM_DIAGNOSTIC`

The frozen spec contains a distance sweep and a noise sweep as **separate**
one-dimensional sweeps. It does **not** contain their combination.

Why it was added: the frozen noiseless distance sweep returned a median focal
error of 0.12 % at *every* distance from 2× to 40×, while the profile width
grew from 0.025 to 2.996. That is the correct behaviour — an exact 2D
observation pins the minimum of even a nearly flat curve — but it meant the
frozen sweep could not show how loss of identifiability turns into focal
*error*. The combined sweep was added to make that link explicit.

It is explanatory. It is not used to support the primary verdict.

### 2.2 REAL_REGIME_MATCHED — `POST_HOC_MECHANISM_DIAGNOSTIC`

`src/regime_match.py`, at distance 4.12 × hand diameter and 2D noise 7.21 px.

Both constants were **measured from results**: 4.12 from the perspective-strength
diagnostic on real data, 7.21 px from the CAM-EXP-006 manual-QC gate. The
condition therefore could not have existed before those results, and it is not
in the frozen spec.

It was added to test whether the real regime plus a realistic residual
reproduces the observed scale of failure. It is an order-of-magnitude
correspondence, not a fit, and not confirmatory evidence.

### 2.3 Reference-3D perturbation sweep — `POST_HOC_SENSITIVITY_DIAGNOSTIC`

Perturbations of 0.5 %, 1 %, 2 % and 5 % of hand diameter. Not in the frozen
spec; added after the §2.2 result under-predicted the real failure, to ask
whether error in the reference hand itself could account for the remainder.

Sensitivity evidence only.

### 2.4 ORACLE_PERSPECTIVE_STRENGTH_DIAGNOSTIC

The task specification for this run called for a real-data perspective-strength
diagnostic, so it was planned. Its specific outputs (ΔZ/Z, distance/diameter)
were not enumerated in the frozen JSON spec. Treated as a mechanism diagnostic,
not confirmatory evidence.

## 3. The B-vs-C classification is post-hoc and interpretive

This is the most important entry in this document.

**What was frozen.** The threshold
`CAMERA_MODEL_MISMATCH_MATERIAL` — fires if the R1→R5 median error falls by
≥ 25 % relative, or identifiable share rises by ≥ 20 pp, or flat share falls by
≥ 20 pp, or tracking becomes positive with a CI excluding zero. Also frozen: the
verdict **labels** A–F with one-line descriptions.

**What was frozen: the trigger did fire.** R1 → R5 is a 75.4 % relative error
reduction, +10.9 pp identifiable, −10.9 pp flat. The criterion was met on the
error-reduction clause.

**What was not frozen.** The frozen spec contains **no operational rule** for
choosing between verdict B and verdict C. The rule actually applied, in
`src/final_verdict.py`, is

```python
r5_usable = (r5_median_error <= 10.0
             and r5_identifiable_share >= 0.50
             and r5_tracking_ci_excludes_zero)
```

Searching the frozen spec confirms the absence: no `r5_usable`, no `<= 10`, no
`0.50`, no "restored", no "identifiable >=".

**Therefore:**

```
PRIMARY NUMERIC TRIGGERS:                    PRE_REGISTERED
FINAL B-vs-C INTERPRETIVE CLASSIFICATION:    POST_HOC_INTERPRETIVE_VERDICT
```

The wording to use:

> The pre-registered `CAMERA_MODEL_MISMATCH_MATERIAL` trigger fired. The final
> B-vs-C classification is a post-hoc interpretive classification, because the
> explicit R5 usability thresholds used in `final_verdict.py` were not part of
> the frozen specification.

**This is not p-hacking and not an analysis error.** The pre-registered trigger
fired and is reported as having fired. What was added afterwards is a
*substantive scientific judgement* about what that trigger means: whether a
large improvement that still leaves 49.96 % error, 77.1 % flat profiles and no
tracking correlation should be called "the camera model explains the result".
That judgement is defensible, and it is labelled as a judgement rather than as
a pre-registered test.

Anyone applying only the frozen numeric trigger would record that
`CAMERA_MODEL_MISMATCH_MATERIAL` fired. The step from there to verdict B is
this run's interpretation.

## 4. Implementation corrections made during analysis

### 4.1 Planar PnP back-end (carried from CAM-EXP-006)

`SOLVEPNP_SQPNP` rejects coplanar input, which the planarity sweep produces by
construction. A fallback chain (IPPE, then ITERATIVE) was added. Without it the
planar condition would have failed for an implementation reason and falsely
appeared to support the hypothesis. Registered as `PLANAR_ENDPOINT_BACKEND`.

### 4.2 Reference-3D perturbation diagnostic — corrected before interpretation

An initial implementation of the reference-3D perturbation diagnostic perturbed
both the 3D input and the image-generation geometry, leaving them
self-consistent; it therefore measured nothing and returned the unperturbed
result at every perturbation level. This was detected before interpretation and
replaced by the corrected diagnostic, in which the 2D is generated from the
clean reference shape and only the 3D handed to the solver is perturbed.

The failed intermediate run is not used as a scientific result anywhere, and its
numbers do not appear in any headline, table or figure. It is recorded here for
reproducibility.

## 5. What contributes to the primary conclusion

**Confirmatory (frozen):**

* R0–R5 real-data conditions and their frozen camera models
* the two synthetic positive controls and their frozen pass conditions
* `REPRODUCTION_MISMATCH`, `FRAME_WEIGHTING_MATERIAL`,
  `CAMERA_MODEL_MISMATCH_MATERIAL`, `UNDERPOWERED_STRESS_SUBSET`
* the S2 / S5 stress subsets
* cluster bootstrap over physical cameras as the frozen uncertainty method

**Explanatory only (frozen mechanism sweeps):** noiseless distance sweep,
planarity sweep, noise sweep, synthetic principal-point and distortion
diagnostics.

**Explanatory only (post-hoc):** distance × noise sweep, REAL_REGIME_MATCHED,
reference-3D perturbation sweep, perspective-strength diagnostic.

**Interpretive:** the B-vs-C classification.

## 6. Frozen manifests are not retro-edited

`cam_exp_0061_validation_spec_v1.json`, `cam_exp_0061_real_conditions_v1.json`
and the CAM-EXP-006 frozen manifests are **deliberately left unmodified**. Their
value is as a dated record of what was committed to before results were seen.
Corrections and later additions are recorded in this document, in `report.md`
and in `scientific_wording_review.md` instead.
