# CAM-EXP-003.1 — Distortion-aware diagnostic

**Question.** CAM-EXP-003 found that every single-frame calibration baseline had a
large focal error on GigaHands (median 16–25 %). Was a substantial part of that
error caused by feeding a **pinhole** camera model images from lenses with real
radial distortion (GT `|k1| ≈ 0.39`, barrel)?

**Verdict: `DISTORTION_CONTRIBUTES_BUT_NOT_DOMINANT`.**

Accounting for distortion roughly **halves** the focal error — AnyCalib
16.31 % → 9.55 %, GeoCalib 24.96 % → 12.16 % median — and the two independent
ways of accounting for it (a distortion-aware model on the raw image, and an
oracle-undistorted image fed to the unchanged pinhole model) agree closely. But
about **10 % median error remains**, and even at the best setting 4 frames in 5
are still outside ±5 %. Distortion was a real and large confound; it was not the
whole story.

Same 1400 frames (175 views × 8 frames) as CAM-EXP-003, reused bit-identically
from its frozen manifest and PNG cache, so every comparison below is paired on
the same image.

---

## Design

| Condition | Image | Camera model | Deployable? |
|---|---|---|---|
| **A** `RAW_PINHOLE` | raw | pinhole | yes — this is CAM-EXP-003, reused verbatim, not re-run |
| **B** `RAW_DISTORTION_AWARE` | raw | official distortion-aware variant | **yes** |
| **C** `GT_UNDISTORTED_PINHOLE` | undistorted with the **GT** distortion | pinhole (unchanged) | **no — oracle** |

Condition C requires the ground-truth distortion coefficients, which a real
deployment does not have. It is a diagnostic upper bound on "what if distortion
were perfectly removed", never a reportable deployment number.

**Undistortion bookkeeping (the easiest place to fake a result).**
`cv2.undistort` with `K_new` from `cv2.getOptimalNewCameraMatrix(..., alpha=0)`,
output kept at 1280×720 so no black border changes what the model sees.
Undistorting **changes the camera**: `K_new.fx / K.fx` has median **0.7734**
(range 0.744–0.855). The ground truth for a condition-C prediction is therefore
`K_new`, not the original focal; scoring against the original `fx` would have
manufactured a ~23 % error or an equally fake improvement. Every per-frame
`K_new`, alpha and ROI is in `results/raw/gt_undistortion_bookkeeping.csv.gz`,
and each condition-C row carries `gt_fx_is_K_new=1` and `orig_gt_fx`.

Metric code (`to_row`) is **imported** from CAM-EXP-003 rather than
reimplemented, so "relative focal error" means exactly the same thing in both
experiments. Distortion-parameter conventions were audited **before** running
anything — see `distortion_model_audit.md`; AnyCalib `radial` and GeoCalib
`radial` both use the OpenCV normalised polynomial `1 + k1 r² + k2 r⁴`, so their
`k1`/`k2` are directly comparable to GT. AnyCam is out of scope per the brief.

---

## Q1 — Headline numbers

Relative focal error, 1400 frames, all conditions (`results/summary/model_summary.csv`):

| Run | Cond | median | p90 | ≤5 % | ≤10 % | ≤20 % | signed median | pred `k1` |
|---|---|---|---|---|---|---|---|---|
| `anycalib` | A | 16.31 % | 32.81 % | 11.4 % | 25.7 % | 64.0 % | +15.6 % | — |
| `anycalib_dist_radial` | B | 11.12 % | 19.88 % | 16.1 % | 41.6 % | 90.4 % | −11.1 % | −0.309 |
| **`anycalib_gen_radial`** | **B** | **9.55 %** | **18.55 %** | **20.1 %** | 53.2 % | **94.1 %** | −9.3 % | −0.312 |
| `undist_anycalib_pinhole` | C | 10.40 % | 21.92 % | 20.6 % | 47.9 % | 85.6 % | +9.8 % | — |
| `geocalib` | A | 24.96 % | 77.18 % | 11.5 % | 23.9 % | 42.9 % | +16.4 % | — |
| `geocalib_distorted_radial` | B | 12.16 % | 49.35 % | 21.8 % | 41.6 % | 70.1 % | +6.6 % | −0.369 |
| `undist_geocalib_pinhole` | C | 12.95 % | 39.59 % | 22.1 % | 41.1 % | 68.4 % | +0.0 % | — |
| `pf_centered` | A | 20.82 % | 39.15 % | 10.3 % | 21.4 % | 47.6 % | −11.6 % | — |
| `undist_pf_centered` | C | 28.84 % | 50.42 % | 5.6 % | 13.1 % | 29.9 % | −26.3 % | — |
| `pf_uncentered` | A | 23.53 % | 66.26 % | 11.1 % | 22.6 % | 43.6 % | +22.7 % | — |
| `undist_pf_uncentered` | C | 16.44 % | 37.29 % | 17.2 % | 31.5 % | 57.8 % | −13.2 % | — |

All 1400 frames succeeded in every run; no frame was dropped.

## Q2 — Does a distortion-aware model beat the pinhole model on the same frame?

Paired, per frame, 2000-resample bootstrap 95 % CI
(`results/summary/paired_improvement_summary.csv`; positive = better):

| Comparison | Δ median (pp) | 95 % CI | % frames improved |
|---|---|---|---|
| `anycalib_gen_radial` vs `anycalib` | **+6.54** | [5.86, 7.32] | 70.6 % |
| `anycalib_dist_radial` vs `anycalib` | +5.06 | [4.18, 5.66] | 65.2 % |
| `geocalib_distorted_radial` vs `geocalib` | **+8.53** | [7.03, 10.08] | 65.6 % |
| `undist_geocalib_pinhole` vs `geocalib` | +10.20 | [8.29, 11.89] | 67.6 % |
| `undist_pf_uncentered` vs `pf_uncentered` | +7.40 | [5.06, 9.56] | 60.9 % |
| `undist_anycalib_pinhole` vs `anycalib` | +5.32 | [4.78, 5.71] | 76.1 % |
| `undist_pf_centered` vs `pf_centered` | **−4.66** | [−6.39, −3.42] | 40.2 % (worse) |

Yes for six of seven, with CIs well clear of zero. The exception matters: for
PF-centered, *removing* distortion made the prediction **worse**. That is a
negative result and is kept as measured.

## Q3 — Condition B vs condition C: do they agree?

Yes, and that is the strongest evidence here. For AnyCalib, the deployable
distortion-aware variant (9.55 %) is *slightly better* than the oracle
undistortion (10.40 %); for GeoCalib they are within 0.8 pp (12.16 % vs
12.95 %). Two mechanistically different interventions — change the model, or
change the image — converge on roughly the same residual, which is what one
expects if distortion is a genuine shared confound that both remove and if what
remains is a different error source.

The deployable route is not worse than the oracle route. Removing distortion is
not gated on knowing the distortion.

## Q4 — Is the remaining error frame noise or view bias?

**View bias.** Median absolute per-view bias
(`results/summary/stability_summary.csv`):

| Model family | A | B | C |
|---|---|---|---|
| AnyCalib | 16.46 % | 9.89 % | 10.70 % |
| GeoCalib | 21.62 % | 11.18 % | 12.66 % |
| PF-centered | 20.21 % | — | 27.86 % |
| PF-uncentered | 23.88 % | — | 15.41 % |

Within-view coefficient of variation across the 8 frames of a static camera
barely moves (AnyCalib 0.0160 → 0.0152). So handling distortion cuts the
*systematic per-camera offset* roughly in half while leaving frame-to-frame
scatter essentially unchanged. The residual ~10 % is a stable per-view bias, not
noise — which is directly relevant to CAM-EXP-004, since averaging more frames
of a static camera will not remove it.

## Q5 — Does per-camera distortion magnitude predict improvement?

**No measurable relationship** (`results/summary/distortion_vs_improvement.csv`):
across the 175 views, `|k1|` vs per-view median improvement gives Pearson
|r| ≤ 0.145 and Spearman |ρ| ≤ 0.187 for every comparison, all p ≥ 0.25.

**This must not be read as "distortion does not matter."** GT `|k1|` spans only
**0.342–0.431** across the whole benchmark — a severely restricted range, all of
it substantial barrel distortion. The experiment has essentially no
low-distortion cameras to contrast against, so it cannot resolve a dose-response
relationship within that narrow band. The paired A→B/C comparisons, which
contrast *handling* vs *not handling* distortion, are the informative test; this
correlation is not, and correlation would not establish causation even if it
were significant. Status: `INCONCLUSIVE_RESTRICTED_RANGE`.

## Q6 — Do the models recover the right distortion?

Approximately. Against a GT `k1` median of ≈ −0.39: `anycalib_dist` −0.309,
`anycalib_gen` −0.312, `geocalib_distorted` −0.369. All three get barrel
distortion of about the right magnitude. GeoCalib is closest in `k1` yet has the
worse focal, so `k1` accuracy alone does not predict focal accuracy. No model
predicts tangential terms; GT `p1 ≈ 3e-3`, `p2 ≈ 7e-4` are ~100× smaller than
`k1`, so that comparison is recorded as
`SKIPPED_INCOMPATIBLE_PARAMETERIZATION` rather than approximated.

## Q7 — Why is PerspectiveFields the exception?

PF has **`NO_DISTORTION_VARIANT`** — the released
`Paramnet-360Cities-edina-{centered,uncentered}` checkpoints emit no distortion
term at all, so PF has no condition B. Nothing was substituted for it.

In condition C, PF-uncentered improves (+7.40 pp) while PF-centered **degrades**
(−4.66 pp) and its signed median swings to −26.3 %, i.e. it now systematically
under-predicts focal. A plausible reading is that PF regresses a vertical FoV
from global perspective cues, and undistortion at alpha=0 rescales the field of
view (fx × 0.77) in a way PF's centered prior mishandles; PF-uncentered, which
also estimates a principal point, absorbs more of that change. This is a
hypothesis consistent with the numbers, **not** something this experiment tested.

## Q8 — Sign of the error

Condition A over-predicts focal (`anycalib` +15.6 %, `geocalib` +16.4 %):
barrel distortion makes the image look wider at the edges than a pinhole lens,
and the models compensate the wrong way. Distortion-aware AnyCalib flips to
under-prediction (−9.3 to −11.1 %); `undist_geocalib_pinhole` lands at a signed
median of **+0.0 %** — unbiased in the median, though with p90 39.6 % its spread
is still wide. Halving the magnitude did not simply shrink a one-sided bias; it
moved the bias across zero for several runs.

## Q9 — Does this change CAM-EXP-002's conclusion?

No. CAM-EXP-002 showed the deployed pipeline's virtual 5000 px focal (vs a
physical fx ≈ 921.6 px) produces metre-scale depth error, and that 5 % focal
error ≈ 32.8 mm of depth displacement. A best-case ~9.6 % median single-frame
focal error still implies roughly 60+ mm of depth displacement at the median and
much more in the tail. Distortion-aware calibration is a large step, not a
solution. Nothing in this run measures depth.

## Q10 — What was not established

- No causal claim from the `|k1|` correlation; it is inconclusive given the
  available data (restricted range).
- The PF-centered degradation mechanism is untested.
- Only `alpha = 0` was tried for undistortion; other alphas were not swept.
- Only the `radial` model family was used; `kb`, `ucm`, `eucm`, `division` and
  GeoCalib's `simple_divisional` were not run (different parameterisations,
  which the brief forbids converting by hand).
- Single dataset (GigaHands), one distortion regime, one image resolution.
- Ray-angle metric recorded as `NOT_REQUIRED_CONVENTIONS_ALREADY_COMPATIBLE`:
  every distortion-predicting model here already shares the OpenCV radial
  convention, so no neutral re-parameterisation was needed.

---

## RECOMMENDED_VARIANTS_FOR_CAM_EXP_004

Deployable (condition B / A) only — condition C is an oracle and must not be
carried forward as a method.

1. **`anycalib_gen` + `cam_id="radial:2"` — primary.** Best median (9.55 %),
   best tail (p90 18.55 %), 94.1 % of frames within 20 %, lowest per-view bias
   (9.89 %). Consistently ahead of `anycalib_dist` on the same frames.
2. **`geocalib_distorted` + `camera_model="radial"` — secondary, different
   family.** Highest ≤5 % rate (21.8 %) and the largest paired gain (+8.53 pp),
   but a heavy tail (p90 49.35 %). Worth keeping for family diversity and
   agreement-based confidence, not as the single estimator.
3. **`pf_uncentered`, raw + pinhole — keep as-is.** PF has no distortion
   variant, and oracle undistortion is not deployable. Lowest priority: its raw
   performance (23.53 %) is the weakest of the three families.
4. **Do not** feed PF-centered undistorted images — measured degradation.

**Carry into CAM-EXP-004 as constraints, not conclusions:** the residual ~10 %
is a *stable per-view bias*, so naive multi-frame averaging over a static camera
will not reduce it (Q4). Any aggregation strategy must attack bias — cross-model
agreement, scene-content diversity, or an external metric anchor — rather than
frame count.

---

## Artifacts

`config.json`, `environment.json`, `distortion_model_audit.md`,
`tables/{model_availability,distortion_parameter_conversion}.csv`,
`results/raw/*.csv.gz` (per-frame predictions, undistortion bookkeeping, paired
comparisons), `results/summary/*.csv`,
`figures/CAM_EXP_003_1_MAIN_EXPLANATION.png` plus five supporting figures.
CAM-EXP-003's outputs were read but never modified.
