# Evidence index

One entry per report section: what may be claimed, the numbers that support it,
the artifacts to use, and the caveat that has to travel with the claim.

All numbers here are keyed to `report_numbers.json`. If a number in the report
disagrees with that file, that file wins.

---

## Why the experiments use different amounts of data

This question comes up immediately when the tables are read side by side, so it
belongs near the front of the report.

Three different validity requirements produce three different subsets:

| Subset | What it requires | Size | Used by |
|---|---|---|---|
| **camera-clean** | an RGB segment that matches the annotation, plus valid provided camera parameters | 175 of 200 views, 40 physical cameras, 5 sequences | Experiments 3 and 4 |
| **hand-clean** | a usable hand annotation for that hand | `PASS_STRICT` 32,826 + `PASS_SINGLE_HAND` 16,005 observations | future hand-aware work |
| **bimanual-clean** | BOTH hands clean in the same frame | 16,413 frames over 154 views (this is the **eligible pool**, not what Experiment 2 evaluated) | Experiment 2 draws its sample from here |

The 25 views missing from the camera-clean subset were excluded because they
have no usable RGB segment matching the annotation — **not** because their hand
annotations were poor. Hand annotation quality never removes a camera-valid
view; that was a deliberate rule, because a badly annotated hand in front of a
perfectly good camera is still a perfectly good calibration sample.

---

## Section 3 — Datasets and common evaluation environment

**Claim:** every camera experiment uses the same frozen frames, the same
metric definitions, and the view — not the frame — as its statistical unit.

**Numbers:** 175 views × 8 frames = 1400 frames (Experiment 3 and 4); 175 × 64 =
11,200 frames (the frame-count extension); 40 physical cameras; 5 sequences.

**Artifacts:** `tables/report_dataset_usage.md`,
`tables/common_evaluation_principles.md`, `figures/main/Fig03`,
`experiments/manifests/external_model_provenance_v1.json`.

**Caveat:** frames are repeated observations of one static camera. Reporting
1400 as an independent sample count would make every interval about 2.5× too
narrow — measured, not assumed (CAM-EXP-004.1).

---

## Section 4 — Experiment 1: data and coordinate reliability

**Claims:**
1. Our reading of the camera and coordinate conventions is correct.
2. A substantial part of the large-error tail was traced to named data
   problems: the observed invalid all-zero 2D pattern, and left/right hand
   identity mismatches within a view. Other cases remained `BAD_2D_GEOMETRY`
   (15,611) or `UNRESOLVED_INSUFFICIENT_GEOMETRY` (10,207) because the available
   multi-view geometry was insufficient to decide, and were conservatively
   labelled REVIEW/EXCLUDE rather than explained.
3. 3D reconstructed from the released 2D agrees with the provided 3D to about
   4 mm, while a wrong-hand control is about 48× worse.

**Numbers:** same-hand median 3.9977 mm, p90 6.4816 mm; other-hand control
192.3047 mm; triangulation reprojection median 3.082 px; 107,640 observations.
QC: `PASS_STRICT` 32,826 / `PASS_SINGLE_HAND` 16,005 / `REVIEW` 10,236 /
`EXCLUDE` 49,173 out of 108,240. All-zero pattern: 26,083 observations.
Verdict classes: `MULTIVIEW_CONFIRMED_GOOD` 48,860, `NO_USABLE_2D_ANNOTATION`
25,852, `BAD_2D_GEOMETRY` 15,611, `UNRESOLVED_INSUFFICIENT_GEOMETRY` 10,207,
`CONFIRMED_2D_HAND_IDENTITY_SWAP` 7,101.

**Artifacts:** `figures/main/Fig03`, `tables/report_main_results.md`,
appendix overlays and the QC status grid.

**Caveat — the important one:** this is **self-consistency between the released
2D and the provided 3D**, not accuracy against an independent external ground
truth. The wrong-hand control is reported precisely so the agreement is not read
as trivial. The diagnosis is also partial: we did **not** explain every
large-error case, and must not write that we did. The QC thresholds themselves
have not yet been validated by a human (`OPEN_ISSUE`).

---

## Section 5 — Experiment 2: focal error and absolute depth

**Claims:**
1. The deployed pipeline assumes a focal length far from the physical one, and
   that alone displaces the hand by metres.
2. Focal error propagates nearly proportionally into absolute depth, while the
   hand's own shape is untouched.

**Evaluated sample:** 2,210 hands over
1,173 frames, 134 views,
37 physical cameras,
5 sequences. These came from a stratified sample
of 1,200 frames drawn from the
16,413-frame bimanual-clean **eligible pool**;
11 frames had no hand detected and
16 had an ambiguous hand association, and
both were dropped. The pool size is not the experiment's sample size.

**Numbers:** assumed 5000 px vs physical 922.77 px (median); root error
2881.885 mm → 78.54 mm; absolute MPJPE 2885.314 mm → 68.68 mm; root-aligned
MPJPE **identical** at 34.404 mm in both conditions. Perturbation: 5 % →
32.801 mm, 10 % → 65.602 mm, 20 % → 131.204 mm (median). 2,210 hands.

**Artifacts:** `figures/main/Fig01`, `figures/main/Fig04`.

**Caveat:** this is the **focal-length sensitivity of absolute hand depth**, not
a general camera-calibration sensitivity analysis. Principal point, anisotropic
focal and distortion were never varied downstream. The mm values belong to this
working-distance regime. The near-linear shape is expected from the pipeline
equation Z ∝ f, not a discovery.

---

## Section 6 — Experiment 3: existing calibration methods

**Claims:**
1. No published single-frame method reaches the ±5 % focal target here.
2. Modelling lens distortion roughly halves the error, and does not close the gap.
3. AnyCam's focal mechanism is not identifiable under a static camera.

**Numbers:** pinhole medians — AnyCalib 16.3102 %, GeoCalib 24.9645 %,
PF-centered 20.8152 %, PF-uncentered 23.5346 %; the deployed fixed-5000 px
assumption 442.5284 % for reference. Distortion-aware — AnyCalib 9.552 %,
GeoCalib 12.1633 %. Paired gain with views as the resampling unit: AnyCalib
+6.54 pp [+4.57, +8.49], GeoCalib +8.53 pp [+5.09, +12.86]. AnyCam candidate
spread 2.73e-09 static vs 0.223 moving.

**Artifacts:** `figures/main/Fig05`, `tables/report_main_results.md`,
`runs/CAM-EXP-004_1.../anycam_static_stress_test.md`.

**Caveats:** use the **view-clustered** intervals, not the frame-level ones. The
GT-undistorted condition is an oracle diagnostic and is never a deployable
method. |k1| spans only 0.34–0.43, so no dose-response conclusion about
distortion magnitude is available. AnyCam produced no focal-accuracy number and
must not appear in a performance table; its end-to-end inference could not be
run on Windows, and what was tested was the identifiability of its own scoring
function.

---

## Section 7 — Experiment 4: multi-frame use and per-view bias

**Claims:**
1. Within GigaHands static views, aggregating 1 → 64 frames did not improve
   focal accuracy for the best model.
2. The reason is that the residual error is dominated by a stable
   **between-view** component — a per-(sequence, camera) static-view bias — not
   by frame-to-frame noise.
3. Combining two model families with opposite signed bias is the most promising
   direction found so far.

**Numbers:** AnyCalib 9.62 % (N=1) → 9.97 % (N=64); GeoCalib 13.89 → 11.28 %;
E2 7.76 → 6.74 %, all from a single run. Between-view share of the squared log
error: AnyCalib 98.1 %, GeoCalib 53.4 %, PF 96.0 %; AnyCalib median |view bias|
9.63 % vs within-view sd 1.43 %. **The decomposition unit is the
(sequence, camera) static view**, with frames as repeated observations inside
it; physical camera is a *resampling* unit used for robustness in CAM-EXP-004.1
and is not what this 98.1 % refers to. E2 gain over AnyCalib +2.23 pp, CI excluding zero at
view, camera and sequence clustering.

**Artifacts:** `figures/main/Fig06`, `figures/main/Fig07`,
`tables/multiframe_run_agreement.csv`.

**Caveats:** one rig; the statement covers N ≤ 64 and does not imply that more
frames never help elsewhere. E2's exact value is a selection-set number — quote
6.14 % (selection set), 6.46 % (independent re-run) and 5.73–8.80 %
(leave-one-sequence-out) together, never the first alone. GeoCalib and therefore
E2 carry a ~±0.35 pp run-to-run tolerance.

---

## Section 8 — Overall results and direction

**Claim:** the binding constraint is now a stable per-view bias. Frame count
is closed as a lever (measured to 64), frame selection is low-value (a GT oracle
best-of-8 only reaches 6.85 % for AnyCalib, which the deployable ensemble
already beats), and the remaining lever is understanding why each view is biased.

**Artifacts:** `tables/hypothesis_result_evidence.md`,
`tables/claim_evidence_ledger.md`,
`experiments/manifests/cam_exp_004_e2_frozen_spec_v1.json`.

---

## Section 9 — Limitations

Use `tables/report_limitations.md` directly. The four that must not be omitted:
single rig; provided 3D is not an independent external ground truth;
CAM-EXP-002 is focal-only; GeoCalib is not run-to-run deterministic (median
spread 1.18 %, p90 6.85 %, max 41.5 % on identical input; aggregate median shift
about 0.32 pp, so conclusions are unchanged — AnyCalib is 100 % bit-identical
under the same test).

---

## Section 10 — Future work

`tables/report_future_work.md`. CAM-EXP-005 (why the per-view bias exists) is
next; CAM-EXP-006 (hand cues) is blocked behind a manual QC validation of
100–300 observations; the sealed external confirmation needs a user decision on
downloading the InterHand2.6M image release.
