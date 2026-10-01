# CAM-EXP-006.1 — validation protocol

> **Scope of this document.** Sections 1–10 describe the protocol **as frozen
> before any result was seen**. Section 11 lists the analyses added *after* the
> primary result. Nothing in sections 1–10 was added or edited after results
> existed. Full account:
> [`analysis_provenance.md`](analysis_provenance.md).

Frozen in `experiments/manifests/cam_exp_0061_validation_spec_v1.json` and
`cam_exp_0061_real_conditions_v1.json`, both with
`created_before_results = true`, before any CAM-006.1 result existed.

## 1. Purpose

Validate CAM-EXP-006's negative result. This run builds **no new calibration
method**. Every oracle condition is a diagnostic, not a proposal.

## 2. Separation of evidence

| evidence | what it establishes |
| --- | --- |
| synthetic | the implementation and the theory |
| real GigaHands | the empirical result |

They are reported in separate tables and figures. A synthetic success is never
offered as evidence that real focal recovery works, and a real failure is never
offered as evidence that the solver is broken.

## 3. CAM0061_FRAME_BALANCED

1. per hand: median over joints of the reprojection error in px
2. **per frame: median over the usable hands in that frame** — so a one-hand
   frame and a two-hand frame each count as exactly one frame
3. per view: mean, in log space, across frames — **unchanged from CAM-EXP-006**

Only step 2 differs from CAM-006. `STRICT_MEDIAN_FRAME_SENSITIVITY` (median
rather than mean at step 3) is available as secondary sensitivity only.

## 4. Real conditions

| id | principal point | distortion | focal | deployable |
| --- | --- | --- | --- | --- |
| R0 | centre | none | scalar, hand-weighted aggregation | yes |
| R1 | centre | none | scalar | yes |
| R2 | **provided** | none | scalar | no |
| R3 | centre | **provided** | scalar | no |
| R4 | **provided** | **provided** | scalar | no |
| R5 | **provided** | **provided** | fx unknown, fy = provided ratio x fx | no |

R0 and R1 share one camera model and differ only in aggregation, so their
per-hand profiles are computed once and aggregated two ways. That isolates the
hand-weighting mismatch exactly.

## 5. Distortion implementation

The coefficients are applied at **every candidate focal**, inside both
`cv2.solvePnP` and `cv2.projectPoints`, because `K(f)` changes with the
candidate. The 2D is never undistorted once and reused across candidates —
that would be wrong, and it is explicitly avoided in
`camera_model_solver.profile_hand`.

## 6. GT information

`tables/oracle_information_usage.csv` records, per condition, exactly which
provided quantities are consumed. In **every** condition:

```
uses_gt_focal_magnitude = 0
uses_gt_extrinsic       = 0
uses_target_2d_in_reference_3d = 0
```

R5 receives the fy/fx **ratio** only. The focal **magnitude** is never supplied
to any solver, in any condition.

## 7. Statistics

Statistical unit: the `(sequence, camera)` view. Uncertainty: cluster bootstrap
over physical cameras, 10 000 iterations. Frames and joints are never treated
as independent samples.

## 8. Frozen thresholds

| id | fires if |
| --- | --- |
| `FRAME_WEIGHTING_MATERIAL` | R0 to R1 changes median error, flat share or identifiable share by >= 10 pp |
| `CAMERA_MODEL_MISMATCH_MATERIAL` | R1 to R5 cuts error by >= 25 % relative, or +20 pp identifiable, or -20 pp flat, or tracking becomes positive with CI excluding zero |
| `REPRODUCTION_MISMATCH` | R0 N=16 differs from 202.16 % by > 5 pp, or flat differs from 86.86 % by > 5 pp |
| `UNDERPOWERED_STRESS_SUBSET` | fewer than 20 views or fewer than 8 physical cameras |

## 9. Verdict taxonomy

A `CAM006_NEGATIVE_RESULT_STRONGLY_VALIDATED` · B `..._VALIDATED_WITH_CAVEATS`
· C `CAM006_CAMERA_MODEL_MISMATCH_EXPLAINS_RESULT` · D
`CAM006_FRAME_WEIGHTING_MATERIALLY_AFFECTED_RESULT` · E
`CAM006_SOLVER_NOT_VALIDATED` · F `CAM006_INCONCLUSIVE`

The frozen specification supplies these **labels** and the four numeric
thresholds above. It does **not** supply an operational rule for choosing
between B and C.

The rule actually applied — that verdict C requires identifiability to be
*restored* rather than merely improved, operationalised in
`src/final_verdict.py` as R5 error <= 10 %, identifiable >= 50 % and a tracking
CI excluding zero — was written **after** the results were seen. It is
therefore a `POST_HOC_INTERPRETIVE_VERDICT`, not a pre-registered test. See
[`analysis_provenance.md`](analysis_provenance.md) §3.

## 10. Constant-rig focal oracle

Name: `CONSTANT_RIG_FOCAL_ORACLE`. Status: `ORACLE_SANITY_CONTROL`.
Registration: `PRE_REGISTERED_IN_CAM006_1_ORACLE_SANITY_CONTROL`, motivated by
the post-hoc CAM-005 finding. The confound was discovered post hoc in CAM-005
and was therefore already known when CAM-006 began, so in CAM-006 and
CAM-006.1 it is genuinely pre-registered. The self-contradictory label
`PRE_REGISTERED_POST_HOC` is not used.


## 11. Post-hoc extensions added after the primary result

These were **not** in the frozen specification. They are mechanism and
sensitivity diagnostics, added to understand why the pre-registered real-data
test failed. They are not confirmatory evidence for the primary verdict.

| addition | role | why it was added |
| --- | --- | --- |
| distance x 1 px noise sweep | `POST_HOC_MECHANISM_DIAGNOSTIC` | the frozen noiseless distance sweep returned 0.12 % at every distance, so it could not show how lost identifiability becomes focal error |
| REAL_REGIME_MATCHED (4.12x diameter, 7.21 px noise) | `POST_HOC_MECHANISM_DIAGNOSTIC` | both constants were measured from results, so the condition could not have been specified in advance |
| reference-3D perturbation sweep (0.5–5 % of hand diameter) | `POST_HOC_SENSITIVITY_DIAGNOSTIC` | added after the matched condition under-predicted the real failure |
| B-vs-C usability cutoff | `POST_HOC_INTERPRETIVE_VERDICT` | the frozen spec gave verdict labels but no operational B/C rule |

One implementation correction was made during analysis: the first reference-3D
perturbation diagnostic perturbed both the solver's 3D input and the
image-generation geometry, leaving them self-consistent. It was detected before
interpretation and replaced; its output is not used as a result anywhere. See
[`analysis_provenance.md`](analysis_provenance.md) §4.2.
