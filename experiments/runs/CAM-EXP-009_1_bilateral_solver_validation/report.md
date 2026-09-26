# CAM-EXP-009.1 — Bilateral Bone-Fitting Solver Validation and Corrected Re-Test

```
VERDICT  CAM009_FAILURE_WAS_IMPLEMENTATION_ARTIFACT

CAM-EXP-009's synthetic negative result is
SUPERSEDED_FOR_INTERPRETATION_DUE_TO_SOLVER_VALIDATION_ISSUE

REAL GIGAHANDS PHASE: STILL NOT RUN
```

**One sentence:** CAM-EXP-009's conclusion that the bilateral objective cannot
identify a focal was **wrong, and wrong because of three implementation faults
in its solver** — with those fixed, the same objective on the same kind of
synthetic data recovers the true focal to **0.035 %** instead of failing at the
50 % grid boundary, and both negative controls destroy that recovery exactly as
a real signal should.

## 1. Plain-language summary

Experiment 9 reported that comparing the left and right hands could not find
the camera's focal length. That conclusion has not survived checking.

Three separate bugs were found in the code behind it:

1. the lens distortion was being removed **twice**;
2. the quality score compared a no-distortion prediction against
   **with-distortion** measurements;
3. the fitting loop was stopped after 6 rounds, far too few — it had not
   finished converging even when the focal was exactly right.

Any one of these would corrupt the result. After fixing all three, the same
method finds the true focal almost exactly. And when the left–right bone
correspondence is deliberately scrambled, it fails completely — which is what
you want to see, because it shows the method is genuinely using the matching
between the two hands.

So the user's bilateral idea was **not** refuted by experiment 9. It was never
properly tested.

Two important cautions remain, and they are not small. This is synthetic data
where the hands are geometrically perfect. The moment real left–right asymmetry
is introduced the bilateral signal decays quickly, and it is also more
sensitive to measurement noise than the simpler single-hand approach. And none
of this has been run on real GigaHands data.

## 2. The three faults, confirmed at source

`tables/cam009_solver_implementation_audit.csv`.

| component | what CAM-009 did | fault | severity |
| --- | --- | --- | --- |
| `fit_bones` | `undistortPoints(uv,K,dist,P=K)`, then `solvePnP(..., K, dist)` on **those same undistorted points** | `DOUBLE_DISTORTION_CONVENTION_BUG` | high |
| `run_synthetic.eval_reproj` | distortion-aware PnP, then a **pinhole** projection compared against **raw distorted** pixels | `RAW_VS_PINHOLE_REPROJECTION_MISMATCH` | high |
| alternating loop | `n_iter = 6` | `INSUFFICIENT_ALTERNATION_ITERATIONS` — left a ~0.4 px residual at the **true** focal | high |
| condition seeding | `abs(hash(name))` — Python's hash is salted per process | `NONDETERMINISTIC_SEEDING` | medium |
| `bilateral_held` metric | numerically identical to `bilateral_fit`; nothing was held out | `MISLEADING_METRIC_NAME` | medium |

The third fault was found by measurement, not inspection:

| iterations | reprojection at the TRUE focal, uniform start |
| ---: | ---: |
| 8 | 0.425 px |
| 16 | 0.021 px |
| **32** | **0.00067 px** |
| 64 | 0.00061 px |
| 128 | 0.00054 px |

CAM-009 used 6. Its "the true focal does not give a low error" observation was
simply an unconverged fit.

## 3. Corrected convention

`UNDISTORT_ONCE_INTERNAL`. Per candidate focal, the raw pixels are undistorted
**once** with that candidate's own `K`, and everything inside the solver then
runs with `distCoeffs = None`. Scores are reported in both spaces, each
self-consistent:

* **ideal** — pinhole projection against undistorted pixels
* **raw** — `cv2.projectPoints` with distortion against raw pixels

The undistortion is recomputed for every candidate; reusing one made with a
different `K` would leak the answer.

## 4. Gates G0–G4 — all pass

| gate | result |
| --- | --- |
| G0 camera round trip | raw median **0.0 px** (max 0.0), ideal median **2.2e-09 px** (max 8.4e-04) |
| G1 true-geometry projection + PnP | ideal **3.9e-07 px**, raw **3.8e-07 px** (gate 0.05) |
| G2 bone fitter at q = 1.0 | uniform **6.7e-04 px**, multi-start **6.6e-04**, oracle-init **1.0e-05** |
| G3 determinism across processes | identical |
| G4 ideal vs raw scoring | Spearman **0.999**, both argmin at q = 1.021 |

G2's bone-vector recovery from a uniform start: L1 error 4.8e-07, cosine
0.9999. Multi-start shape spread 2.7e-04 — every initialisation converges to
the same proportions, so there is **no shape non-identifiability** here.

Only after all five passed was the corrected sweep run.

## 5. The corrected result

Moderate perspective (4× hand diameters), realistic distortion, noiseless,
8 trials, 161-point grid, true focal at q = 1.0:

| objective | recovered q | focal error | boundary rate |
| --- | ---: | ---: | ---: |
| held-out reprojection | 1.000 | **0.035 %** | 0.00 |
| **bilateral distance** | 1.000 | **0.035 %** | 0.00 |
| oracle true shape | 1.000 | 0.035 % | 0.00 |

Against CAM-009's 50 % at the boundary. The same numbers hold with no
distortion, and at every distance from 2× to 32×.

| | CAM-009 | CAM-009.1 |
| --- | ---: | ---: |
| bilateral recovered q | 1.500 (boundary) | **1.000** |
| bilateral focal error | 50.0 % | **0.035 %** |
| boundary rate | 0.95 | **0.00** |
| held-out focal error | 50.0 % | **0.035 %** |

## 6. The controls — the signal is genuinely bilateral

| condition | bilateral focal error |
| --- | ---: |
| correct bone mapping | **0.035 %** |
| wrong bone mapping | **50.0 %** |
| right hand from another subject | **50.0 %** |

Both controls degrade by ~50 percentage points, far past the 10 pp criterion
frozen in advance. The held-out reprojection is unaffected by either (0.035 %),
exactly as expected since it never uses the correspondence.

The fixed-reference-length control remains **exactly focal-invariant**, so
CAM-009's one correct finding stands: the naive version of the idea cannot
work.

## 7. But: the bilateral term is fragile where it matters

This is the part that tempers the positive result.

**Real asymmetry breaks it.** Held-out reprojection is unaffected; the
bilateral term is not:

| bilateral asymmetry | bilateral focal error | held-out focal error |
| ---: | ---: | ---: |
| 0 % | 0.04 % | 0.04 % |
| 1 % | 3.0 % | 0.04 % |
| 2 % | **12.0 %** | 0.04 % |
| 5 % | **29.3 %** | 0.04 % |

Human hands are not symmetric to within 1 %. At a plausible 2 % the bilateral
objective is already outside its own 5 % gate while the single-hand temporal
objective is untouched.

**Noise hurts it more than the single-hand objective:**

| 2D noise | held-out | bilateral |
| ---: | ---: | ---: |
| 0 px | 0.04 % | 0.04 % |
| 0.5 px | 3.1 % | 2.4 % |
| 1 px | 2.7 % | 8.2 % |
| 2 px | 13.6 % | 14.6 % |
| 4 px | 29.2 % | 36.8 % |

Pose diversity made no difference (LOW/MED/HIGH all 0.04 %) — unsurprising once
the objective is well conditioned.

## 8. DOF ablation — and a caveat that undercuts it

| shape model | held-out | bilateral |
| --- | ---: | ---: |
| D20 every bone free | 0.04 % | 0.04 % |
| D10 adjacent bones paired | 35.0 % | 48.4 % |
| D5 one per finger | 19.1 % | 32.9 % |

This looks like "fewer parameters is worse", which is the opposite of the
CAM-009 hypothesis. **It should not be read that way.** My D10 and D5 groupings
force the bones within a group to be *equal in length*, and a real hand's
metacarpal is much longer than its distal phalanx. So these are not
lower-dimensional descriptions of the same family — they are a different,
misspecified family, and the error is model bias, not a statement about
constrained anatomy.

A properly designed low-dimensional model would scale a template rather than
equalise bones. That was not tested. Registered as
`DOF_GROUPING_IS_MISSPECIFIED`.

The DOF ablation was motivated by D20 failing, which no longer happens, so it
no longer bears on the main verdict.

## 9. What this changes, and what it does not

**Changes.** CAM-EXP-009's synthetic negative result cannot be read as evidence
about the bilateral idea. It is superseded for interpretation. Its raw outputs
are untouched and remain on record as what that code produced.

**Does not change.** Nothing here has been run on GigaHands. The real-data
phase remains not run, and the reference focal has still never been read by
either experiment. A synthetic success under perfect bilateral symmetry is a
long way from a factory.

The honest position: the idea deserved a fair test, CAM-009 did not give it
one, and CAM-009.1 shows it is at least identifiable in principle — while also
showing it degrades fast under exactly the conditions a real person's hands
present.

## 10. Recommendation

**Do not jump to the GigaHands real phase yet.** Two things should be settled
first, and both are cheap:

1. **An asymmetry-tolerant formulation.** At 2 % asymmetry the objective is
   already out of tolerance. A side-scale nuisance, or a robust per-bone
   weighting, would need to be designed and re-validated synthetically before
   it is worth spending real-data compute.
2. **A correctly specified low-dimensional shape model** (template scaling,
   not bone equalisation), since §8's ablation answered a different question
   than intended.

If those hold up, the real phase becomes justified — and it should reuse the
CAM-EXP-008 paired machinery so that scene-only and scene+bilateral differ in
nothing but the bilateral term.

Also worth stating plainly: on GigaHands the reference focal varies by only
~1.7–1.9 %, so even a working bilateral cue would have very little room to
demonstrate value there. The varied-focal dataset requirement from CAM-007 and
CAM-008 has not gone away.

## 11. Limitations

1. Synthetic only.
2. 8 trials per condition, reduced from CAM-009's 20 on measured cost (38.4 s
   per trial), decided before any sweep result and applied uniformly.
3. The bone-length distribution of the generator is plausible but invented.
4. The DOF ablation is misspecified (§8).
5. Perfect knowledge of the distortion coefficients is assumed; on real data
   they would come from the scene estimator.
6. The articulation directions are exact here; a real reference reconstruction
   is noisy, and §7 shows the bilateral term is noise-sensitive.

## 12. Provenance

`CAM0091_PRE_SPECIFIED_BEFORE_CORRECTED_EXECUTION`. The gates and thresholds
were fixed before the corrected sweep ran. They are **not** pre-registered from
the start of the project — CAM-009 had already produced a negative result, and
this run exists to check the solver behind it.

One change was made during the run: the alternation count went from 8 to 32.
That was judged **only** against the positive control at the true focal, with a
uniform start, and **before any focal sweep had been executed**. The solver
configuration was then frozen (`cam_exp_0091_solver_validation_spec_v1.json`)
and nothing in it changed afterwards.

CAM-EXP-009's chronology also needs stating: its formal synthetic gate was
written **after** a preliminary focal-dependence audit had already revealed the
boundary minimum. It was frozen before the full distance/noise/asymmetry sweep,
but not before all synthetic evidence. CAM-009's report should not describe
every gate as fully result-blind.

## 13. Reproduce

```
python src/run_gates.py             # G0-G4; exits non-zero on failure
python src/run_corrected_sweep.py   # only after the gates pass
```
