# Open issues (registered 2026-09-23, CAM-EXP-004.1)

Things we know are unresolved. Registered so they are not quietly forgotten
between experiments. Each has an owner experiment and a blocking level.

---

## 1. `HAND_QC_MANUAL_VALIDATION_PENDING`

**Blocks:** CAM-EXP-006 (hand-aware calibration). Does not block CAM-EXP-005.

CAM-EXP-001.3's strict QC thresholds (`PASS_STRICT` / `PASS_SINGLE_HAND` /
`REVIEW` / `EXCLUDE`) were calibrated against a comparatively small clean-control
set. No human has checked how the automatic verdicts behave outside that set, so
we do not know the false-pass or false-exclude rate.

This has not affected any camera result so far, because CAM-EXP-003 onwards
deliberately use the camera-usability flag and never the hand QC. It becomes
load-bearing the moment hand geometry is used as a calibration cue.

**Plan:** before CAM-EXP-006, a human adjudicates 100-300 observations sampled
across sequence, camera, hand and QC group (including borderline cases, not only
confident ones), and the result is compared against the automatic verdict to
produce a confusion matrix and per-class rates. Sampling must be stratified and
pre-registered so the reviewer cannot be steered.

---

## 2. `CAM_EXP_002_SCOPE_OVERSTATED`

**Blocks:** nothing technically. Blocks the wording of any paper claim.

What CAM-EXP-002 actually measured is the sensitivity of **absolute hand depth
to focal length**: one intrinsic parameter, one downstream quantity, on GigaHands
geometry (5 % focal error ≈ 32.8 mm, 10 % ≈ 65.6 mm, 20 % ≈ 131.2 mm of isolated
depth displacement). Anywhere that has been paraphrased as "camera calibration
sensitivity" it is broader than the evidence.

CAM-EXP-002's raw results are **not** modified; this is a scope correction
recorded here and in the CAM-EXP-004.1 report.

**Two ways to close it, to be chosen later:**

* **A —** narrow the claim to focal sensitivity of absolute depth, and say so
  explicitly wherever the number is quoted. Costs nothing.
* **B —** run CAM-EXP-002.1 covering the parameters we never tested downstream:
  principal point, anisotropic focal (`fx != fy`), and distortion. Only worth it
  if a claim about those is actually needed.

No new experiment was run in CAM-EXP-004.1.

---

## 3. `GEOCALIB_RUN_TO_RUN_NONDETERMINISM`

**New in this run. Blocks:** nothing outright, but it changes how GeoCalib and
any GeoCalib-containing estimate must be quoted.

GeoCalib is **not reproducible run to run on a byte-identical image**. Measured
over 40 frames x 3 repeats in one process, same environment, same weights:

| Model | frames bit-identical across repeats | median spread | p90 | max |
|---|---|---|---|---|
| AnyCalib-gen + radial:2 | **100 %** | 0.00 % | 0.00 % | 0.00 % |
| GeoCalib-distorted + radial | **0 %** | 1.18 % | 6.85 % | 41.50 % |

The input was ruled out first: a frame decoded fresh from the video is
bit-identical to the PNG CAM-EXP-003.1 used (max absolute pixel difference 0).
The likely mechanism is visible in GeoCalib's own logs — its Levenberg-Marquardt
optimiser repeatedly reports *"Reached maximum number of steps without
convergence"*, and a non-converged iterative solve combined with
non-deterministic CUDA reductions lands in different places on different runs.

**Practical effect, measured on the full 175 views:** an independent re-run
shifts the reported medians by about a third of a percentage point
(GeoCalib 11.18 % → 10.86 %, E2 6.14 % → 6.46 %; AnyCalib 9.89 % → 9.89 %, exactly).
That is far smaller than the effects we are claiming, so no conclusion changes,
but it is a real uncertainty component that no bootstrap captures.

**Plan:** quote every GeoCalib-derived number with a run-to-run tolerance of
about ±0.35 pp; try `torch.use_deterministic_algorithms(True)` plus a fixed seed
and re-measure; if that does not fix it, report the median of k repeats and say
k. Do this before any headline number goes into a paper.

---

## 4. `MULTI_FRAME_CLAIM_IS_SINGLE_RIG`

**Blocks:** any general claim about static-camera aggregation.

Every static-camera result comes from one GigaHands rig. The inventory in
`tables/holdout_candidate_datasets.csv` shows nothing else locally available can
confirm it: HanCo's released images are per-frame hand-centred crops, so their
image-space intrinsics change every frame; AssemblyHands and InterHand2.6M are
annotation-only on disk.

**Plan:** a decision is needed from the user — download the InterHand2.6M image
release, download full HanCo RGB, or accept the claim as single-dataset. All
were previously excluded on size grounds, so none can be assumed.

---

## 5. `E2_IS_A_SELECTION_SET_NUMBER`

**Blocks:** calling 6.14 % a confirmatory result.

E2 was chosen as the best of five candidate ensembles on the same 175 views it is
reported on. Leave-one-sequence-out shows the *selection* is stable (E2 wins on
the training folds in 5/5 folds, selection cost 0.00 pp) and the held-out medians
are 5.73-8.80 %, but all of that is still the same rig and the same 5 sequences.

**Plan:** the frozen spec is `experiments/manifests/cam_exp_004_e2_frozen_spec_v1.json`.
It is used verbatim from CAM-EXP-005 on, and its confirmatory number comes from
the reserved holdout, once.
