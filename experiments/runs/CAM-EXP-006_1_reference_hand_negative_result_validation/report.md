# CAM-EXP-006.1 — Validation of the Reference-Hand Focal Negative Result

```
PRIMARY VERDICT  B
CAM006_NEGATIVE_RESULT_VALIDATED_WITH_CAVEATS
```

**One sentence:** CAM-EXP-006's conclusion was right in *direction* but wrong in
*attribution* — most of its headline 202 % error came from ignoring the rig's
lens distortion, not from hand geometry, and the honest number is about 50 %;
yet even with the true camera model supplied the focal still cannot be
identified.

## 1. What was checked and what came back

| question | answer |
| --- | --- |
| A. Does the solver work when the focal *is* identifiable? | **Yes.** 0.12 % median error, 100 % of trials within 1 % |
| B. Did a two-hand frame get double weight? | **Yes**, confirmed at source — but it changed the result by only 1.0 pp |
| C. Did the centre-principal-point approximation cause it? | **No.** R2 changed nothing (203.1 → 204.8 %) |
| D. Did ignoring distortion cause it? | **Partly, and substantially.** R3 cut the error from 203 % to 47.5 % |
| E. Did the fx = fy assumption cause it? | **No.** The rig's fy/fx is 0.9987 |
| F. Was "the limit is geometric" too strong? | **Yes.** Modelling mattered a great deal; the claim is narrowed |

## 2. Synthetic solver validation

`SOLVER_VALIDATED`. All frozen pass conditions met.

| suite | median focal error | within 5 % | flat | boundary |
| --- | ---: | ---: | ---: | ---: |
| generic non-planar cloud, strong perspective | **0.12 %** | 100 % | 0 % | 0 % |
| real hand shapes, strong perspective | **0.12 %** | 100 % | 0 % | 0 % |

The solver recovers a known 900 px focal essentially exactly. CAM-EXP-006's
real-data failure is therefore **not** an implementation failure.

### 2.1 Distance sweep — the weak-perspective mechanism

| distance / hand diameter | profile width | flat | error (noiseless) | error (1 px noise) |
| ---: | ---: | ---: | ---: | ---: |
| 2× | 0.025 | 0 % | 0.12 % | 2.98 % |
| 3× | 0.075 | 0 % | 0.12 % | 6.58 % |
| 5× | 0.175 | 0 % | 0.12 % | 20.09 % |
| 10× | 0.761 | 0 % | 0.12 % | 49.29 % |
| 20× | 2.846 | 100 % | 0.12 % | 64.44 % |
| 40× | 2.996 | 100 % | 0.12 % | 101.57 % |

A subtlety worth stating plainly: with **noiseless** 2D the focal error stays
at 0.12 % at every distance, because an exact observation pins the minimum of
even a very flat curve. The loss of identifiability shows up as the **width**
of the near-optimal region, which grows 120-fold. It only turns into focal
*error* once observations carry noise. That is why the two right-hand columns
differ so much, and it is the mechanism behind the real result.

### 2.2 Planarity sweep

| depth scale | profile width | median error |
| ---: | ---: | ---: |
| 1.00 | 0.025 | 0.12 % |
| 0.50 | 0.075 | 0.12 % |
| 0.25 | 0.200 | 0.13 % |
| 0.10 | 0.499 | 0.13 % |
| 0.00 (planar) | 2.996 | 60.87 % |

Depth extent is exactly what makes the focal observable. This is the clean
replacement for CAM-006's `PLANARIZED` control, which was confounded by a
different PnP back-end.

### 2.3 Camera-model sensitivity, synthetic

| condition | median error |
| --- | ---: |
| correct model | 0.12 % |
| off-centre principal point, solver assumes centre | 2.09 % |
| true principal point supplied | 0.12 % |
| distorted image, solver assumes no distortion | **5.18 %** |
| true distortion supplied | 0.12 % |

In a *strong-perspective* regime, ignoring distortion costs only ~5 %. The real
data shows it costing far more — see §4.3.

## 3. The imaging regime GigaHands was actually in

`ORACLE_PERSPECTIVE_STRENGTH_DIAGNOSTIC`, 4271 hand observations:

| quantity | p10 | median | p90 |
| --- | ---: | ---: | ---: |
| camera distance / hand diameter | 2.55 | **4.12** | 6.50 |
| relative depth extent ΔZ / Z | 0.092 | **0.150** | 0.263 |
| hand diameter / Z | 0.154 | 0.243 | 0.393 |

This is a **moderately** weak-perspective regime — around 4 hand-diameters, not
the extreme 40× end of the sweep. That matters, and it is why weak perspective
alone does not account for the real numbers (§4.4).

## 4. Real-data reanalysis

### 4.1 R0 reproduces CAM-EXP-006 exactly

`REPRODUCTION_OK`. R0 N=16 gives **202.16 %**, identical to CAM-006's
202.16 %; flat share 86.9 % against 86.9 %. The reanalysis pipeline is
faithful.

### 4.2 The frame-weighting mismatch was real but immaterial

`FRAME_WEIGHTING_NON_MATERIAL`.

| | N=1 | N=16 | flat (N=16) | identifiable (N=16) |
| --- | ---: | ---: | ---: | ---: |
| R0 original (hand-weighted) | 109.97 % | 202.16 % | 86.9 % | 4.0 % |
| R1 frame-balanced (corrected) | 143.02 % | 203.13 % | 88.0 % | 2.9 % |

Differences at N=16: 0.96 pp error, 1.14 pp flat, 1.14 pp identifiable — all
below the 10 pp threshold frozen in advance. The bug was genuine; it did not
drive the conclusion. (At N=1 the difference is larger, 33 pp, but in the
direction of making the corrected result *worse*.)

### 4.3 The camera model was material — this is the main correction

| condition | N=16 error | flat | identifiable | what it is given |
| --- | ---: | ---: | ---: | --- |
| R0 original | 202.16 % | 86.9 % | 4.0 % | nothing |
| R1 frame-balanced | 203.13 % | 88.0 % | 2.9 % | nothing |
| R2 + true principal point | 204.75 % | 88.0 % | 2.9 % | cx, cy |
| **R3 + true distortion** | **47.53 %** | 77.1 % | 13.7 % | k1,k2,p1,p2 |
| R4 + both | 49.92 % | 76.6 % | 14.3 % | cx,cy + distortion |
| R5 + both + fy/fx ratio | 49.96 % | 77.1 % | 13.7 % | + aspect ratio |
| `CONSTANT_RIG_FOCAL_ORACLE` | **0.85 %** | — | — | the rig's median focal |

The frozen `CAMERA_MODEL_MISMATCH_MATERIAL` trigger **fired**: a 75.4 %
relative error reduction, +10.9 pp identifiable, −10.9 pp flat.

The driver is distortion alone (R3), not the principal point (R2). The rig's
median `k1` is −0.392 — substantial barrel distortion that CAM-006 modelled as
zero while feeding the solver raw distorted 2D.

**So CAM-EXP-006's headline 202 % substantially overstated the geometric
limit.** The defensible figure for "how badly does reference-hand geometry
identify the focal" is about **50 %**, not 202 %.

### 4.4 But the focal is still not identifiable

Verdict C would require identifiability to be *restored*. It was not:

* R5 median error **49.96 %** against a constant's **0.85 %** — a single
  number beats the full-oracle solver by about 59×.
* **77.1 %** of profiles are still flat; only **13.7 %** are cleanly
  identifiable.
* Tracking: Spearman **−0.025**, CI [−0.168, +0.118] — the interval contains
  zero. The estimates do not follow the true focal at all.
* The estimates scatter with CV **86.4 %** while the reference focal varies by
  CV **1.90 %**. This is noise, not collapse onto a constant.
* Figure 06 shows the R5 objective still plateauing from roughly the true focal
  onward, with no minimum to find.

On the powered stress subset **S2** (37 views, 23 physical cameras, not
underpowered), R5 gives 32.64 % against the constant's 2.58 %. **S5** has only
4 views and 4 cameras and is flagged `UNDERPOWERED_STRESS_SUBSET`; no
conclusion is drawn from it.

### 4.5 Why the real result is worse than weak perspective alone predicts

At the measured 4.12× distance with 1 px noise, synthesis predicts ~11 %
error — far short of the real 110–203 %. Two further ingredients were measured:

| synthetic condition at the real 4.12× distance | median error | flat |
| --- | ---: | ---: |
| noiseless | 0.12 % | 0 % |
| 2D noise 1 px | 11.38 % | 0 % |
| 2D noise 7.21 px (the measured reference-3D/2D residual) | 55.30 % | 34 % |
| reference-3D error 1 % of hand diameter | 25.72 % | 9 % |
| reference-3D error 2 % of hand diameter | 48.61 % | 17 % |

The real failure is therefore **not** weak perspective on its own. It is a
moderately weak-perspective geometry *combined with* a reference hand that
disagrees with the 2D by several pixels, plus — before R3 — unmodelled
distortion. Each ingredient alone is insufficient; together they reproduce the
observed scale.

This is an order-of-magnitude correspondence, not a fit. The synthetic noise is
isotropic Gaussian, whereas the real residual is partly correlated
reconstruction error.

## 5. Corrections required to CAM-EXP-006's report

Numerical results in CAM-006 are untouched. Its **wording** needs narrowing.
Full table: `tables/scientific_claim_status.csv`.

| CAM-006 claim | status | action |
| --- | --- | --- |
| the reference hand does not identify the focal | **UPHELD** | keep, scope narrowed |
| 202 % median error at N=16 | **REPRODUCED_BUT_MISATTRIBUTED** | report ~50 % as the camera-model-corrected figure |
| "the limit is geometric, not a modelling shortfall" | **OVERSTATED** | modelling mattered a great deal |
| "it will not be fixed by a better hand model" | **NOT_SUPPORTED_AS_STATED** | hand accuracy does matter — 1 % 3D error costs ~26 % focal error |
| "a predicted hand cannot carry more focal information than the oracle hand" | **LOGICALLY_TOO_STRONG** | learned models encode priors beyond explicit 3D |
| planarized control beating main "proves" absence | **WEAK_EVIDENCE** | demote to sensitivity; the synthetic sweep is the clean evidence |
| `PRE_REGISTERED_POST_HOC_ORACLE_SANITY_CONTROL` | **SELF_CONTRADICTORY_LABEL** | rename |

The accurate scope statement:

> Under the evaluated GigaHands imaging regime and this profiled-PnP
> formulation, other-camera-only reference hand geometry does not provide
> enough perspective information to identify the focal reliably, even when the
> rest of the camera model is supplied exactly. This does not establish that
> hand geometry carries no camera information in other regimes, nor that
> learned hand-derived features encoding camera priors are ruled out.

## 6. CAM-EXP-007

**Recommendation: REDESIGN** — neither proceed as planned nor abandon.

Repeating the *same profiled-PnP focal estimator* with a predicted hand is low
priority: the oracle hand already fails, and the reference-3D sweep shows a
noisier hand makes it strictly worse. But CAM-006's blanket statement was too
strong. A learned hand model can encode training priors, image appearance and
camera priors that explicit 3D geometry does not contain, so **learned
hand-derived camera cues are not ruled out**. CAM-007 should be restated around
that different question.

## 7. Limitations

1. One rig, one dataset, one focal setting. `SINGLE_FOCAL_RIG_CONFOUND` still
   caps everything: the reference focal varies by only 1.90 % (CV), so a
   constant scores 0.85 % and there is almost no dynamic range in which to
   demonstrate skill.
2. R2–R5 are `ORACLE_DIAGNOSTIC_ONLY` and not deployable.
3. The reference 3D is triangulated with the rig's own provided camera
   parameters, so rig-wide calibration error is inherited and undetectable
   from inside this dataset.
4. The §4.5 correspondence is order-of-magnitude, not a fitted decomposition.
5. The S5 stress subset is underpowered (4 views).
6. The exactly-planar synthetic endpoint uses a different PnP back-end and is
   diagnostic only.
7. The manual-QC gate is reused from CAM-006 unchanged: 200 cases, single
   reviewer, a readiness gate and not a dataset accuracy figure.

## 8. Reproduce

```
python src/audit_cam006_implementation.py
python src/freeze_validation_spec.py       # freeze BEFORE results
python src/synthetic_validation.py
python src/regime_match.py
python src/perspective_strength.py
python src/run_real_reanalysis.py
python src/evaluate_real.py                # first comparison to the reference focal
python src/final_verdict.py
python src/figures.py
```
