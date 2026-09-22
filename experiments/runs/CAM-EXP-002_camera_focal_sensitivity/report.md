# CAM-EXP-002 — Camera focal sensitivity of absolute hand 3D

Run date 2026-09-22 · branch `kjh` · base commit `bf74e0c` · RTX 4060 Ti.
Audit in `focal_usage_audit.md`, parameters in `config.json`, environment in
`environment.json`, log in `logs/run.log`.

## 1. Question

How wrong is the absolute 3D hand position and depth produced by the existing
AnyHand/WiLoR pipeline when the camera focal length is wrong? No calibration
method is developed here; this measures the sensitivity of what already exists.

## 2. Data

Only CAM-EXP-001.3 QC manifests decide what is used:

* frames come from `gigahands_demo_bimanual_clean_v1` (LEFT **and** RIGHT both
  `PASS_STRICT`);
* the camera must have the RGB segment that matches the annotation
  (`gigahands_demo_camera_benchmark_v1`, CAM-EXP-001.2 M6);
* `REVIEW` and `EXCLUDE` are never used.

Stratified sample, seed `20260922`: **1200 frames** over all 5 sequences and 37
cameras (p36 309, p41-boxing 292, p41-plant 256, p44-dog 241, p52-instrument
102). After inference and association: **2210 clean hand observations**,
8296 bimanual frame records, 17680 sweep rows.

Failures are recorded, not dropped silently: **11** frames with no detection
(`results/raw/model_failures.csv`, all in `brics-odroid-027_cam0`) and **16**
`AMBIGUOUS_HAND_ASSOCIATION` frames (`results/raw/hand_association_failures.csv`).

## 3. Method

Inference runs **once per frame** and is cached
(`cache/hand_inference/*.npz`: bbox, `pred_cam`-derived translation,
root-relative joints, predicted 2D, handedness, score). Every focal condition
then recomputes only the focal-dependent conversion. The detector, the crop and
the network regression are therefore bit-identical across conditions, which is
what makes the focal effect exactly isolated.

**Hand association** uses 2D only: predicted keypoint centroid vs the provided
2D annotation centroid, best assignment required to beat the runner-up by 1.5x
and to be within 150 px. 3D error is never used for association — choosing the
pairing that minimises the quantity under study would manufacture the result.
Measured separability: median matched distance **3.0 px**, and the detector's
own handedness agrees with the 2D-chosen mapping in **98.7 %** of frames.

**Reference 3D** is the GigaHands provided 3D transformed to camera space with
the CAM-EXP-001-validated `R, t`. It is a dataset-provided reference, not
external ground truth.

## 4. Q1 — what is the pipeline's focal?

**A training-convention virtual focal, expressed in original full-image pixels.**
`rgb_predictor.py:407-411`:

```python
scaled_focal = EXTRA.FOCAL_LENGTH / MODEL.IMAGE_SIZE * max(W, H)
             = 1000 / 256 * 1280 = 5000.0 px      # for a 1280x720 frame
```

It is never measured, never read from calibration, never estimated from the
image. It encodes the assumption that every image has the training
field of view. It is not a physical focal (A), not a resized-image focal (B) —
there is no resize — and not a crop focal (C).

The focal enters exactly one place (`_cam_crop_to_full`, lines 972-974):

```
tz = 2 f / (s · B)      <- proportional to f
tx, ty                  <- no focal term at all
```

and the principal point used is the image centre, never the calibrated
`cx, cy`. Full trace in `tables/focal_usage_trace.csv` (19 steps).

## 5. Q2 — can the physical focal be inserted correctly?

**Yes, and the required transform is the identity.** The pipeline focal lives in
original full-image pixels; GigaHands intrinsics are in the same 1280x720
pixels (the coordinate system in which CAM-EXP-001 validated reprojection to
~3 px); there is no resize. So

```
GT_EFFECTIVE_FOCAL = GT_NATIVE_FX      (median 922.8 px over the sample)
```

`fy` differs from `fx` by 0.1-0.3 %, and both are stored per row so the analysis
can be redone with `fy`. No mean/sqrt combination was invented.

Consequently `PIPELINE_BASELINE / GT_NATIVE_FX = 5000 / 922.8 = 5.42`: the
pipeline assumes a focal **5.4x longer** than the camera really has.

## 6. Sanity check of the formula

Predicted depth ratio `Z(alpha)/Z(alpha=1)` against `alpha`, over all 2210
observations: **maximum deviation 0.0e+00**. The relation `Z ∝ f` holds
exactly, as the source predicts, so no hidden focal-dependent term exists.

Also verified: `|root shift| - |ΔZ| = 0.0 mm` for every observation — a wrong
focal moves the hand **only along Z**, exactly as `tx, ty` having no focal term
requires. (`figures/predicted_depth_ratio_vs_focal_scale.png`)

## 7. Q3-Q5 — how much does the 3D move?

**Incremental focal effect** (the clean measure: same inference, only the focal
changed), median over 2210 hands, `results/summary/focal_sensitivity_summary.csv`:

| focal error | signed ΔZ (median) | hand displacement (median) | p90 |
|---|---|---|---|
| **−20 %** | −131.2 mm | **131.2 mm** | 188.8 mm |
| **−10 %** | −65.6 mm | **65.6 mm** | 94.4 mm |
| **−5 %** | −32.8 mm | **32.8 mm** | 47.2 mm |
| +5 % | +32.8 mm | **32.8 mm** | 47.2 mm |
| +10 % | +65.6 mm | **65.6 mm** | 94.4 mm |
| +20 % | +131.2 mm | **131.2 mm** | 188.8 mm |

So **a 5 % focal error moves the hand ~33 mm, 10 % moves it ~66 mm, and 20 %
moves it ~131 mm** — for a hand at roughly 0.65 m. The displacement is entirely
in depth.

**Absolute error** vs the provided 3D (contains hand-model and detector error
too), median:

| condition | focal | abs depth err | root 3D err | absolute MPJPE | root-aligned MPJPE |
|---|---|---|---|---|---|
| GT x0.80 | 734.7 px | 151.9 mm | 158.6 mm | 161.3 mm | 34.4 mm |
| GT x0.90 | 826.5 px | 90.6 mm | 103.8 mm | 100.9 mm | 34.4 mm |
| GT x0.95 | 872.4 px | 66.9 mm | 86.0 mm | 83.4 mm | 34.4 mm |
| **GT x1.00** | **918.4 px** | **58.9 mm** | **78.5 mm** | **68.7 mm** | 34.4 mm |
| GT x1.05 | 964.3 px | 56.2 mm | 78.3 mm | 66.3 mm | 34.4 mm |
| GT x1.10 | 1010.2 px | 62.3 mm | 82.4 mm | 69.3 mm | 34.4 mm |
| GT x1.20 | 1102.0 px | 105.5 mm | 117.3 mm | 109.8 mm | 34.4 mm |
| **PIPELINE_BASELINE** | **5000 px** | **2881.6 mm** | **2881.9 mm** | **2885.3 mm** | 34.4 mm |

## 8. Q6 — linearity

Yes, exactly, by construction and in measurement: `Z(alpha) = alpha · Z(1)` with
zero deviation, so the incremental depth shift is perfectly linear in the focal
error (−131.2 / −65.6 / −32.8 / +32.8 / +65.6 / +131.2 mm for
−20/−10/−5/+5/+10/+20 %).

The **absolute** error is *not* linear and is not minimised exactly at
alpha = 1.00 but slightly beyond it (56.2 mm at alpha = 1.05 vs 58.9 mm at
1.00). That is expected: absolute error also contains a systematic
weak-perspective/scale bias — the signed Z error at alpha = 1.00 is −20.2 mm
(median), i.e. the pipeline places hands slightly too close even with the
correct focal, and a ~5 % focal increase happens to cancel part of it. This is a
bias in the existing regression, not evidence that the true focal is wrong.

## 9. Q7 — which axis

**Only Z.** `tx, ty` contain no focal term, and the measured
`|root shift| − |ΔZ|` is exactly 0.0 mm for every observation. X and Y in metres
are untouched; what changes is the depth, and therefore the implied ray
direction.

A notable corollary from §4: since `tz ∝ f` and `tx, ty` are fixed, the
**2D reprojection is completely invariant to the focal**. A wrong focal is
invisible to any 2D overlay check — it can only be caught in metric 3D.

## 10. Q8 — absolute joint error

Absolute 21-joint MPJPE moves from 68.7 mm (correct focal) to 100.9 mm (−10 %),
161.3 mm (−20 %), 69.3 mm (+10 %), 109.8 mm (+20 %). Bimanual 42-joint absolute
MPJPE is 72.7 mm at the physical focal and 2781.0 mm at the pipeline baseline.
The left-right root distance error is −26.9 mm at the physical focal versus
+432.5 mm at the baseline: with a 5.4x focal the whole bimanual configuration is
inflated, so even the *relative* geometry between the two hands is wrong.

## 11. Q9 — root-aligned pose

**Root-aligned MPJPE is 34.404 mm in every single condition**, including the
baseline — identical to three decimals. The focal has no effect whatsoever on
the shape of the hand, only on where it is placed. This is exactly the
interpretation the brief anticipated: camera focal is a global placement and
depth problem, not a pose problem.

## 12. Q10 — baseline vs physical GT focal

| | PIPELINE_BASELINE | PHYSICAL GT focal | ratio |
|---|---|---|---|
| focal | 5000 px | 922.8 px | 5.42x |
| abs depth error | **2881.6 mm** | **58.9 mm** | 48.9x |
| root 3D error | **2881.9 mm** | **78.5 mm** | **36.7x** |
| absolute MPJPE | 2885.3 mm | 68.7 mm | 42.0x |
| bimanual 42-joint MPJPE | 2781.0 mm | 72.7 mm | 38.2x |
| root-aligned MPJPE | 34.4 mm | 34.4 mm | 1.00x |

The existing pipeline places hands **about 2.9 metres too far away** on this
data. The cause is structural and visible from the source: `tz = 2f/(sB)` with a
focal 5.4x too long gives a depth 5.4x too large; at a true depth of ~0.65 m
that is ~3.5 m, i.e. ~2.9 m of error.

Contrary to the caution in §19 of the brief, the physical focal did **not**
turn out worse than the training virtual focal here — it is better by 36.7x.
That is not automatic: it holds because the audit established that the two
focals live in the same coordinate system, so substituting one for the other is
mathematically legitimate. Had the pipeline consumed a crop-space or
resized-image focal, the substitution would have been invalid and the comparison
meaningless.

**Why the baseline is built this way**: WiLoR/HaMeR-style models regress a
weak-perspective `pred_cam` under a fixed training focal, which is fine for
*rendering an overlay* (the 2D projection is focal-invariant, §9) but is not a
metric-depth estimator. The pipeline's focal is a rendering convention that was
never intended as a calibration. For any absolute/global 3D use it must be
replaced by a real focal.

## 13. Principal point (brief §14)

Not swept, and deliberately so: `_cam_crop_to_full` uses the image centre and
the calibrated `cx, cy` is never read anywhere in the pipeline. **The current
pipeline is insensitive to the principal point because it does not use it.** No
artificial effect was manufactured. The image-centre assumption does introduce a
fixed bias in `tx, ty` (for `brics-odroid-001_cam0` the real principal point is
6.6 px and 21.3 px away from the centre), which is recorded as an observation.

## 14. Interpretation, and what these numbers do and do not show

The absolute-error column mixes four sources: hand-model pose error, detector
and bbox error, weak-perspective scale error, and focal error. Only the
**incremental** column isolates the focal, and it is the one to quote for "how
much does a focal error cost". They are never mixed in this report:

* focal error alone: **~6.6 mm of depth per 1 % of focal error** at this working
  distance (33 mm per 5 %), perfectly linear;
* everything else combined: ~59-79 mm of residual error even with the correct
  focal, of which ~34 mm is pose error that no camera parameter can fix.

So at this working distance the focal stops being the dominant error source once
it is accurate to roughly ±5 %; below that, hand-model and weak-perspective
error dominate. That is a useful target for any future calibration work.

## 15. Limitations

- One working distance (~0.65 m) and one camera rig; the mm-per-percent figure
  scales with depth and should not be transferred to other distances without
  rescaling.
- 1200 frames of the 14,671 available bimanual-clean RGB-usable frames
  (stratified, seed recorded); the effect is deterministic given the cache, so
  more frames would tighten distributions, not change the conclusion.
- The reference 3D is GigaHands' own, validated in CAM-EXP-001.3 to ~4 mm
  against an independent re-triangulation, but it is not external ground truth.
- Only the WiLoR backend was run; HaMeR uses the identical focal expression
  (`rgb_predictor.py:446-450`) so the structural conclusion carries, but the
  numbers were not measured for it.
- `fx` was used as the scalar focal; `fy` would shift the numbers by 0.1-0.3 %.
- The +5 % optimum of the absolute error is a property of this pipeline and this
  dataset, not a recommendation to scale focals by 1.05.

## 16. Next steps

1. Replace the virtual focal with a real per-camera focal wherever absolute 3D
   is used; this alone is a 36.7x improvement in root placement here.
2. Investigate the −20 mm signed depth bias that remains at the correct focal
   (weak-perspective/scale calibration of the regression).
3. Since 2D is focal-invariant, any calibration benchmark must be evaluated in
   metric 3D — a 2D overlay cannot detect this class of error.
4. CAM-EXP-003 can now quote a concrete accuracy target for focal estimation:
   ±5 % keeps the focal-induced depth error (~33 mm) below the pipeline's own
   residual error (~59 mm).

## Artifacts

```
focal_usage_audit.md                          source-level audit
tables/focal_usage_trace.csv                  19-step machine-readable trace
tables/smoke_numeric_trace.csv                per-sample numeric trace
results/raw/inference_cache_index.csv         1200 cached frames
results/raw/focal_sweep_per_hand.csv.gz       17680 rows (2210 hands x 8 conditions)
results/raw/focal_sweep_per_frame.csv.gz      8296 bimanual rows
results/raw/model_failures.csv                11 detector failures
results/raw/hand_association_failures.csv     16 ambiguous associations
results/summary/focal_sensitivity_summary.csv isolated focal effect per alpha
results/summary/per_alpha_summary.csv         all metrics per condition
results/summary/baseline_vs_gt_focal.csv      the headline comparison
results/summary/{per_sequence,per_camera,failure}_summary.csv
figures/CAM_EXP_002_MAIN_EXPLANATION.png      the whole result on one sheet
figures/*.png                                 7 supporting figures
```

Reproduce:

```bash
PYTHONPATH=<repo> .venv-anyhand/Scripts/python src/run_inference.py --n 1200   # GPU
PYTHONPATH=<repo> .venv/Scripts/python src/run_all.py                          # analysis
```
