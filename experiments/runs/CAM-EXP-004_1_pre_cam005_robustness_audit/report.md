# CAM-EXP-004.1 — Pre-CAM-005 robustness, statistics, reproducibility & holdout audit

No new calibration method. This run stress-tests the CAM-EXP-001…004 results
that CAM-EXP-005 is about to build on, and fixes in writing what is confirmed,
what is robust-but-single-dataset, and what is still exploratory.

**Headline: every substantive conclusion survived.** Two things changed:
uncertainty intervals roughly 2.5× wider once frames stop being counted as
independent samples, and one new defect found — **GeoCalib is not reproducible
run to run**, which puts a ±0.35 pp tolerance on every number that contains it.

---

## Q1 — Does CAM-EXP-003.1 survive removing frame-level pseudo-replication?

**Yes.** CAM-EXP-003.1 bootstrapped over 1400 frames, but 8 frames of one static
camera are not 8 independent observations of that camera's error. Re-run with
whole clusters resampled (10 000 iterations, point estimates untouched):

| Comparison | resampling unit | n clusters | median improvement | 95 % CI |
|---|---|---|---|---|
| AnyCalib-gen vs AnyCalib pinhole | frame (legacy) | 1400 | +6.54 pp | [+5.75, +7.30] |
| | **view** | **175** | +6.54 pp | **[+4.57, +8.49]** |
| | physical camera | 40 | +6.54 pp | [+3.51, +9.74] |
| | sequence *(sensitivity only)* | 5 | +6.54 pp | [+4.38, +8.20] |
| GeoCalib-distorted vs GeoCalib pinhole | frame (legacy) | 1400 | +8.53 pp | [+7.09, +10.24] |
| | **view** | **175** | +8.53 pp | **[+5.09, +12.86]** |
| | physical camera | 40 | +8.53 pp | [+3.40, +15.72] |
| | sequence *(sensitivity only)* | 5 | +8.53 pp | [+5.16, +12.96] |

The legacy frame CIs were about 2.5× too narrow, exactly as expected from
pseudo-replication. **No interval crosses zero at any clustering level**, for any
of the CAM-EXP-003.1 comparisons. The legacy numbers are retained and labelled
`LEGACY_FRAME_BOOTSTRAP` rather than deleted.

The 5-sequence intervals are reported as a sensitivity check only; with 5
clusters a bootstrap CI is not trustworthy, and it happens to come out *narrower*
than the camera-level one, which is itself a symptom of too few clusters rather
than evidence of precision.

## Q2 — Does the CAM-EXP-004 ensemble gain survive cluster structure?

**Yes, at every level.** E2 vs AnyCalib-gen, N=8, per view:

| Statistic | view (175) | physical camera (40) | sequence (5) |
|---|---|---|---|
| median error improvement | +2.23 pp [+1.32, +3.00] | +2.23 [+0.80, +3.26] | +2.23 [+0.66, +3.20] |
| Δ within-5 % rate | +18.3 pp [+9.1, +27.4] | +18.3 [+6.8, +29.3] | +18.3 [+8.8, +24.0] |
| Δ within-10 % rate | +17.1 pp [+8.6, +26.3] | +17.1 [+6.3, +27.9] | +17.1 [+7.3, +27.5] |
| signed bias, AnyCalib | −9.25 % [−10.43, −7.96] | −9.25 [−10.91, −7.63] | −9.25 [−10.49, −8.24] |
| signed bias, E2 | −2.36 % [−3.60, −0.69] | −2.36 [−4.06, −0.20] | −2.36 [−3.90, −0.87] |

CAM-EXP-004 already used the view as its unit, so nothing there was wrong; this
extends it to the two coarser structures. All intervals exclude zero.

## Q3 — Is E2 stable under leave-one-sequence-out selection?

**The selection is completely stable; the performance is not a constant.** For
each fold, the best of E1–E5 was chosen on the 4 training sequences and applied
to the held-out one:

| Held out | chosen on training | held-out median (chosen) | held-out median (fixed E2) | held-out ≤5 % |
|---|---|---|---|---|
| p36-tea-0010 | E2 | 7.08 % | 7.08 % | 32.5 % |
| p41-boxing-0021 | E2 | 5.73 % | 5.73 % | 37.5 % |
| p41-plant-0004 | E2 | 5.78 % | 5.78 % | 45.0 % |
| p44-dog-0004 | E2 | 5.81 % | 5.81 % | 45.0 % |
| p52-instrument-0034 | E2 | 8.80 % | 8.80 % | 20.0 % |

**E2 wins on the training folds in 5/5 folds, selection cost 0.00 pp.** But the
held-out median ranges **5.73 – 8.80 %** around the 6.14 % selection-set figure,
and the worst fold is the small, atypical 15-view sequence. So the right way to
quote E2 is a range, not 6.14 %.

This is labelled `RETROSPECTIVE_INTERNAL_VALIDATION`. It is the same rig, the
same two models and the same 5 sequences; it is **not** an independent test and
must never be described as one.

**E2 is now frozen**: `experiments/manifests/cam_exp_004_e2_frozen_spec_v1.json`.
Per-frame `sqrt(f_anycalib · f_geocalib)`, then median over the frames of one
`(sequence, camera)` view. No weights, no thresholds, nothing fitted. CAM-EXP-005
and 006 use it verbatim; a different rule is a new estimator with a new name.

## Q4 / Q5 — 16 / 32 / 64 frames, actually measured

CAM-EXP-004 skipped Phase B on a pre-registered stopping rule. That decision was
defensible but it was a prediction, so it was checked. A new 64-frame manifest
was frozen before inference
(`gigahands_demo_cam_exp_0041_64frames_v1.csv.gz`, sha256 `c79f9b29…`, 11 200
rows, 175 views), with the CAM-EXP-003 eight frames an **exact subset** of the 64
in 175/175 views and full 8 ⊂ 16 ⊂ 32 ⊂ 64 nesting. Frames were chosen from video
length alone — no GT, no predictions, no hand labels, no scene content. All 175
views were processed, in a pre-registered stratified order.

Median relative focal error (aggregation rules frozen from CAM-EXP-004):

| N | AnyCalib-gen | GeoCalib-distorted | E2 (frozen) |
|---|---|---|---|
| 8 | 9.890 % | 10.860 % | 6.460 % |
| 16 | 9.966 % | 10.920 % | 6.610 % |
| 32 | 9.954 % | 10.786 % | 6.703 % |
| 64 | 9.966 % | 11.283 % | 6.740 % |

Within-5 % rate over the same range: AnyCalib 20.0 → 20.6 %, GeoCalib 22.3 →
22.3 %, E2 40.0 → 40.0 %. Every paired step (8→16, 16→32, 32→64) has a bootstrap
CI containing zero for all three estimators.

**Verdict: `CONFIRMED_SATURATION`, for all three estimators.** Going from 8 to 64
frames of a static camera changes nothing — it is slightly *worse*, within noise.
CAM-EXP-004's prediction was right, and the saturation claim can now be made for
N ≤ 64 on measurement rather than on a stopping rule.

What this does **not** license: a claim about N in the hundreds, or about any
other rig. And note the curves are flat, not just flat-after-8 — there was never
a descent.

## Q6 — Reproducibility, and a defect we did not expect

Re-running the identical 8 frames through the identical environment reproduced
**AnyCalib exactly** and **GeoCalib not at all**. Before blaming anything, the
input was ruled out: a frame decoded fresh from the video is **bit-identical** to
the PNG CAM-EXP-003.1 used (max absolute pixel difference 0).

Dedicated test, 40 frames × 3 repeats on byte-identical images:

| Model | bit-identical across repeats | median spread | p90 | max |
|---|---|---|---|---|
| AnyCalib-gen + radial:2 | **100 %** | 0.00 % | 0.00 % | 0.00 % |
| GeoCalib-distorted + radial | **0 %** | 1.18 % | 6.85 % | **41.50 %** |

The mechanism is visible in GeoCalib's own logs: its Levenberg–Marquardt
optimiser repeatedly reports *"Reached maximum number of steps without
convergence"*. A non-converged iterative solve plus non-deterministic CUDA
reductions lands in different places on different runs — on one pathological
frame, repeated calls returned 539, 516, 473 and 520 px, and other calls on the
same pixels returned ~3300 px.

**Effect on our conclusions, measured on all 175 views:**

| Estimator | CAM-EXP-004 | independent re-run | difference |
|---|---|---|---|
| AnyCalib-gen | 9.890 % | 9.890 % | **0.000 pp** |
| GeoCalib-distorted | 11.184 % | 10.860 % | −0.324 pp |
| E2 (frozen) | 6.138 % | 6.460 % | +0.322 pp |

So GeoCalib-derived numbers carry a run-to-run tolerance of roughly **±0.35 pp**
that no bootstrap captures. That is an order of magnitude smaller than the
effects being claimed (the ensemble gain is +2.23 pp), so **no conclusion
changes** — but every such number should be quoted with the tolerance, and this
is registered as `OPEN_ISSUES.md` item 3 with a plan to try deterministic
algorithms and, failing that, to report the median of k repeats.

## Q7 — Is external provenance now fully pinned?

**Yes, with no unresolved cases.** `experiments/manifests/external_model_provenance_v1.json`:

| Model | exact commit | how recovered | checkpoints |
|---|---|---|---|
| AnyCalib | `027a8497d893f4b2596f23d6324c05e4b81064ed` | pip `direct_url.json` `vcs_info` | gen / dist / pinhole, SHA256 recorded |
| GeoCalib | `97b8968e7798a66bf04fcf791fb535624241bda7` | pip `direct_url.json` `vcs_info` | distorted / pinhole, SHA256 recorded |
| PerspectiveFields | `d54be737d6eacfb9d39a2b7079a494924b45bb6c` | git HEAD of the local clone it was installed from | rpf / rpfpp, SHA256 recorded |
| AnyCam | `e609cc8a9e4ee8f78cf2ce39ebeb86b35e82d10d` | git HEAD of the clone; two of its functions were executed for the identifiability test, no end-to-end inference | `fwimbauer/anycam_v1_seq8` |

All seven checkpoints are present and hashed. No
`EXACT_COMMIT_UNRESOLVED_FOR_HISTORICAL_RUN` case remained, so no re-install
equivalence check was needed for provenance reasons — and the independent re-run
above serves as one anyway. Dependencies are locked in
`requirements-calibration-frozen.txt` with the critical ones tabulated in
`tables/dependency_versions.csv` (torch 2.1.2+cu121, kornia 0.7.2 — required,
numpy 1.26.4).

## Q8 — AnyCam on static video

**Attempted for real, and blocked for a documented platform reason.** Official
repo cloned at the commit above, official checkpoint downloaded from the location
the repo's own `hubconf.py` points at, isolated venv, no system CUDA touched.

The official inference path cannot run on Windows: `fit_video.py` applies
`torch.compile` at import (fails on torch ≤ 2.5 on Windows), while AnyCam's depth
backbone needs `xformers.components`, which exists only in `xformers ≤ 0.0.28.post3`
— a version with no Windows wheel that also requires torch 2.5.1. The two
requirements are mutually exclusive here. Satisfying them would mean editing the
official source, which this project forbids, so it was not done.
`BLOCKED_IMPLEMENTATION_ON_WINDOWS`. No placeholder results were written.

The *mechanism* was tested instead, using **AnyCam's own unmodified code** — the
function `fit_video.py` uses to score focal candidates before its argmax:

| Condition | max &#124;induced flow&#124; | spread across the 32 focal candidates | distinguishable? |
|---|---|---|---|
| **static camera** (identity relative pose) | 1.79e-07 | **2.73e-09** | **no** |
| moving camera | 2.394 | 0.223 | yes |

Under a static camera the objective that picks the focal is exactly flat across
every candidate; the argmax is decided by the prior and float noise.

**Verdict: the official AnyCam formulation is not identifiable in our
static-camera deployment condition** — `METHOD_ASSUMPTION_MISMATCH`, consistent
with CAM-EXP-004, and not a statement that AnyCam is a poor method. What is still
missing is the behavioural test (what does it emit anyway?); the driver script is
kept so it can be run unchanged on Linux.

## Q9 — GigaHands development protocol, frozen

`experiments/manifests/development_validation_split_v1.json`, effective from
CAM-EXP-005: **leave-one-sequence-out**, not a fixed two-way split.

The reason is the structure: 5 sequences with 40/40/40/40/**15** views. Any fixed
split either puts a single sequence in validation — possibly the small atypical
one — or starves development. LOSO uses each sequence as validation exactly once
and reports 5 held-out numbers instead of one lucky one; the E2 audit above shows
why that matters, since those five ranged 5.73–8.80 %. Rules: selection and
tuning may only see the 4 training-fold sequences; the view is the statistical
unit inside a fold; all 5 held-out numbers get reported, not just their mean.

## Q10 — Final confirmatory holdout, reserved

`experiments/manifests/final_confirmatory_holdout_v1.json`. This turned out to be
the hardest item, and the honest answer is partly negative.

* **Primary: InterHand2.6M — `DESIGNATED_PRIMARY_HOLDOUT_PENDING_IMAGE_DOWNLOAD_APPROVAL`.**
  Genuinely static multi-camera studio rig, different optics, lighting, scene and
  resolution; the only candidate that could confirm both the single-frame and the
  static-camera multi-frame claims. We hold its annotations and camera parameters
  (1.6 GB) but **not** its ~80 GB image release, which was previously excluded.
  Contamination risk is currently zero: no image has ever been read.
* **Secondary: HanCo (tester) — `RESERVED_FINAL_CONFIRMATORY_HOLDOUT_SINGLE_FRAME`.**
  536 images, 8 cameras, 1 sequence, 224×224, per-frame K. A structural sanity
  check (logged, no model run, no visual inspection) found that the released
  `rgb/` images are **per-frame hand-centred crops**: within a single camera, `fx`
  varies by up to 96 % relative and `cx` by up to 119 px across 67 frames. So
  HanCo cannot test the static-camera premise at all, and since its images ship
  already undistorted it cannot confirm the distortion-aware result either. It
  remains usable for single-frame calibration claims.
* AssemblyHands: annotations only, no images — not a candidate.

**Decision needed from you:** confirming the CAM-EXP-004 static-camera result
needs a second rig with static cameras and full images, and nothing local
provides one. The options are (a) download the InterHand2.6M image release,
(b) download full HanCo RGB, or (c) accept the multi-frame claim as
single-dataset. All three were previously excluded on size grounds, so this is a
decision rather than something to assume.

---

## Evidence grading

**`CONFIRMED`**
- Distortion-aware beats pinhole (CAM-EXP-003.1), at frame, view, physical-camera
  and sequence clustering.
- More frames do not help a bias-dominated estimator: measured at N = 8/16/32/64,
  flat, `CONFIRMED_SATURATION` for all three estimators.
- The residual error is a stable per-camera bias, not frame noise.
- AnyCam is not identifiable under a static camera, demonstrated numerically on
  its own scoring function.
- AnyCalib is bit-reproducible; GeoCalib is not.

**`ROBUST_BUT_SINGLE_DATASET`**
- All of the above: one GigaHands rig, 5 sequences, 40 physical cameras, one
  resolution, one distortion regime.

**`EXPLORATORY`**
- E2 = 6.14 % (6.46 % on the independent re-run, 5.73–8.80 % across LOSO folds).
  Selected as the best of 5 candidates on its own evaluation set. The *direction*
  of the ensemble gain is confirmed; the *number* is a selection-set number.

**`PENDING_EXTERNAL_CONFIRMATION`**
- Every claim above. Nothing local can serve as an independent confirmation of
  the static-camera result.

**`OPEN_ISSUE`** — see `OPEN_ISSUES.md`
1. `HAND_QC_MANUAL_VALIDATION_PENDING` — blocks CAM-EXP-006, not CAM-EXP-005.
2. `CAM_EXP_002_SCOPE_OVERSTATED` — CAM-EXP-002 measured the focal sensitivity of
   absolute hand depth, not full camera-calibration sensitivity. Raw results are
   unmodified; the scope is corrected here. Either narrow the claim or run
   CAM-EXP-002.1 for principal point / anisotropic focal / distortion.
3. `GEOCALIB_RUN_TO_RUN_NONDETERMINISM` — new; ±0.35 pp tolerance on medians.
4. `MULTI_FRAME_CLAIM_IS_SINGLE_RIG` — needs the download decision above.
5. `E2_IS_A_SELECTION_SET_NUMBER`.

---

## CAM-EXP-005 readiness

| Gate | Status |
|---|---|
| 003.1 view-cluster bootstrap | done — conclusions hold |
| 004 hierarchical / cluster sensitivity | done — conclusions hold |
| frozen E2 specification | done — `cam_exp_004_e2_frozen_spec_v1.json` |
| E2 retrospective LOSO audit | done — 5/5 folds, held-out 5.73–8.80 % |
| actual 16/32/64-frame check | done — `CONFIRMED_SATURATION` |
| AnyCam static stress test *or* clear evidence it cannot run | done — blocked on Windows with documented cause, plus a positive mechanistic result |
| exact external-model provenance | done — all commits and checkpoint hashes resolved |
| dependency freeze | done — `requirements-calibration-frozen.txt` |
| GigaHands development/validation protocol | done — leave-one-sequence-out, frozen |
| final confirmatory holdout reserved | done, **with an open decision** on image downloads |
| open issues registered | done — `OPEN_ISSUES.md`, 5 items |

**CAM-EXP-005 can proceed.** It should carry in the frozen E2 spec as its
baseline, use leave-one-sequence-out for every selection, quote GeoCalib-derived
numbers with the ±0.35 pp run-to-run tolerance, and treat the per-camera bias —
not frame count and not frame selection — as the quantity to explain.

## Artifacts

`config.json`, `environment.json`, `OPEN_ISSUES.md`,
`anycam_static_stress_test.md`, `requirements-calibration-frozen.txt`;
`results/raw/` (cluster bootstraps, LOSO selection, 22 400 extended-frame
predictions, extended aggregation, determinism repeats, AnyCam candidate
diagnostics); `results/summary/` (both statistical re-analyses, selection
stability, 8/16/32/64 table and verdict, determinism check, independent re-run
comparison, AnyCam identifiability, holdout plan); `tables/` (statistical units,
provenance, dependency versions, frame-count extension, holdout candidates,
manifest nesting check, static-GT check); `figures/CAM_EXP_004_1_MAIN_EXPLANATION.png`;
new manifests `gigahands_demo_cam_exp_0041_64frames_v1.csv.gz`,
`cam_exp_004_e2_frozen_spec_v1.json`, `external_model_provenance_v1.json`,
`development_validation_split_v1.json`, `final_confirmatory_holdout_v1.json`.

`results/raw/anycam_static_predictions.csv.gz` is deliberately absent — that test
did not run, and no placeholder was fabricated. CAM-EXP-001 through CAM-EXP-004
were read only and are unmodified.
