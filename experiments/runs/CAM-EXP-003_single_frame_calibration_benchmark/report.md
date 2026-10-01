# CAM-EXP-003 — Single-frame camera calibration baseline benchmark

Run date 2026-09-23 · branch `kjh` · base commit `06cfe6d` · RTX 4060 Ti.
Convention audit in `model_output_convention_audit.md`, parameters in
`config.json`, environment and checkpoints in `environment.json`.

## 1. Question

From a single RGB frame of a static, workshop-like GigaHands view, how
accurately can existing published camera-calibration models recover the
camera's physical focal length? No new method is developed, nothing is
fine-tuned, no hand geometry is used, and no multi-frame aggregation is
performed.

**Why ±5 % matters.** CAM-EXP-002 measured on this same data that a 5 % focal
error displaces the reconstructed hand by about **32.8 mm** in depth (10 % →
65.6 mm, 20 % → 131.2 mm), and that below roughly 5 % the focal stops being the
dominant error source. That is a *target imported from CAM-EXP-002*; nothing in
CAM-EXP-003 measures a depth error. Results are reported at 5 %, 10 % and 20 %.

## 2. Data and fairness

Frames were frozen **before any model ran**:
`experiments/manifests/gigahands_demo_cam_exp_003_frames_v1.csv.gz` —
**175 sequence × camera views × 8 frames = 1400 frames**, evenly spaced over
each view's annotated range. Views come from
`gigahands_demo_camera_benchmark_v1` with `usable_for_camera_benchmark == 1`,
which excludes the cameras whose RGB segment does not match the calibration
(CAM-EXP-001.2 M6). Hand-annotation quality is deliberately **not** used as a
filter: this is a camera benchmark, so a view with poor hand labels is still a
perfectly good RGB + GT camera sample.

Fairness measures actually implemented, not merely intended:

* every model reads the **same 1400 frames** from one lossless PNG cache, so
  the pixels are bit-identical across models;
* failures are written as rows with `success=0` and never dropped;
* both **success-only** and **coverage-aware** threshold rates are reported;
* `ORACLE_GT` is a reference line only and is excluded from all rankings — its
  0.00 % error also serves as the end-to-end check that the evaluation code is
  correct.

## 3. Q1/Q2 — output conventions, and did the conversion get verified?

Full detail in `model_output_convention_audit.md` and
`tables/model_output_conversion.csv`; five traced frames per model in
`tables/adapter_numeric_trace.csv`.

| Model | Native focal convention | Conversion applied | Principal point |
|---|---|---|---|
| **AnyCalib** | `[fx, fy, cx, cy]`, **already original-image pixels** | identity | predicted |
| **GeoCalib** | `camera.f`, original-image pixels (`camera.size` = `[1280,720]`) | identity | **not predicted** (fixed at image centre) |
| **PerspectiveFields** | `pred_rel_focal` = **f / image height** | `f_px = rel_focal × H` | predicted (uncentered only) |
| DEMO_FIXED_5000 | constant 5000 px | identity | not predicted |

Two points were genuinely ambiguous and were resolved from source plus a
numerical check rather than assumed:

1. **AnyCalib's README contradicts its code.** The README says focals are
   relative to the resized network input; `AnyCalib.predict` in fact ends with
   `cam.reverse_scale_and_shift(...)`. The code wins, confirmed because the
   returned principal point lands at the centre of the **original** 1280×720
   image (638.3, 360.2), which is impossible in 420×238 network pixels. Taking
   the README at face value would have inflated AnyCalib's error threefold.
2. **PerspectiveFields' two checkpoints disagree internally.** Their official
   converter `general_vfov_to_focal(rel_cx, rel_cy, 1, gvfov)` reproduces the
   *uncentered* model's `pred_rel_focal` to seven decimals (1.2353644 vs
   1.2353643), confirming the height normalisation. For the *centered*
   checkpoint the same identity fails (1.396 from its reported vfov vs 1.068
   from its focal head), so its reported vfov is not the quantity its focal head
   produced. `pred_rel_focal` is used for both, and the vfov-implied value is
   stored beside it as `extra_fx_px_from_vfov`. No convention was chosen by
   whichever scored better against ground truth.

Hand-verified on a frame with GT fx = 921.6 px
(`tables/adapter_numeric_trace.csv`): AnyCalib 1100.38 px (+19.4 %), GeoCalib
985.23 px (+6.9 %), PF-uncentered 1.13416 × 720 = 816.60 px (−11.4 %),
PF-centered 1.31787 × 720 = 948.87 px (+3.0 %), ORACLE_GT 921.61 px (0.00 %).

## 4. Q3/Q4 — accuracy

1400 frames per model, 0 % failures everywhere
(`results/summary/model_summary.csv`):

| Model | median | p90 | p95 | ≤5 % | ≤10 % | ≤20 % | failure | median ms |
|---|---|---|---|---|---|---|---|---|
| **AnyCalib** | **16.3 %** | 32.8 % | 43.0 % | **11.4 %** | **25.7 %** | **64.0 %** | 0 % | 86 |
| PF [centered] | 20.8 % | 39.1 % | 48.2 % | 10.3 % | 21.4 % | 47.6 % | 0 % | 50 |
| PF [uncentered] | 23.5 % | 66.3 % | 81.1 % | 11.1 % | 22.6 % | 43.6 % | 0 % | 72 |
| GeoCalib | 25.0 % | 77.2 % | 102.8 % | 11.5 % | 23.9 % | 42.9 % | 0 % | 370 |
| DEMO_FIXED_5000 | 442.5 % | 453.7 % | 458.8 % | 0 % | 0 % | 0 % | 0 % | 0 |
| *ORACLE_GT (reference)* | *0.00 %* | *0.00 %* | *0.00 %* | *100 %* | *100 %* | *100 %* | *0 %* | *0* |

Because no model ever fails, the coverage-aware rates equal the success-only
rates here; both are in the summary CSV regardless.

## 5. Q5 — does anything meet the ±5 % target?

**No.** The best model, AnyCalib, puts **11.4 %** of single frames within 5 %
of the true focal — roughly one frame in nine. Its median error, 16.3 %, is
more than three times the target. Interpreted through CAM-EXP-002's
sensitivity, a 16 % focal error corresponds to roughly 100 mm of depth
displacement at this working distance, which is far larger than the pipeline's
own ~59 mm residual.

Two things are worth separating, though. The deployed baseline
`DEMO_FIXED_5000` is wrong by **442 %**, so every learned model is a very large
improvement over what the pipeline uses today — roughly 27× better in median
error. "Not good enough for the 5 % target" and "much better than the status
quo" are both true.

All four learned variants cluster at 10–12 % within 5 %, despite differing
medians. Their signed medians differ in direction — AnyCalib +15.6 %,
GeoCalib +16.4 %, PF-uncentered +22.7 %, PF-centered −11.6 % — so three of the
four systematically **over**-estimate the focal (predicting a narrower field of
view than reality) on this data.

## 6. Q6 — where do they fail

Per sequence (AnyCalib, `results/summary/per_sequence_summary.csv`) the median
error is fairly flat, 13.9 %–18.7 %, so no single scene dominates. The spread
across **cameras** is much larger
(`results/summary/per_camera_summary.csv`, `figures/per_camera_error_heatmap.png`):

| | camera | median error |
|---|---|---|
| worst | `brics-odroid-016_cam0` | 50.4 % |
| | `brics-odroid-008_cam0` | 29.9 % |
| | `brics-odroid-029_cam0` | 28.7 % |
| best | `brics-odroid-015_cam1` | **2.5 %** |
| | `brics-odroid-024_cam0` | 7.1 % |
| | `brics-odroid-025_cam1` | 7.4 % |

So the difficulty is a property of the **viewpoint**, not of the activity: some
views are solved to a few percent while others are wrong by half. Representative
good and bad frames are in `figures/examples/*_best_worst_grid.png`, with GT and
predicted focal and HFOV printed on each panel. A qualitative look at the best
cases shows views where the workspace walls and their straight edges dominate
the frame; the worst are close-ups where the scene is mostly a textured surface
with little usable perspective structure. This is an observation from the
representative frames, not a controlled ablation — scene-cue ablation is
CAM-EXP-005.

## 7. Q7 — static-camera stability

The camera does not move and its intrinsics do not change, so any spread across
the 8 frames of one view is the model's own instability
(`results/summary/single_frame_stability.csv`, `tables/stability_by_model.csv`):

| Model | CV median | CV p90 | spread over 8 frames (median) | p90 |
|---|---|---|---|---|
| **AnyCalib** | **1.6 %** | 3.7 % | **4.6 %** | 11.0 % |
| PF [uncentered] | 4.9 % | 9.6 % | 14.4 % | 28.0 % |
| GeoCalib | 6.0 % | 44.2 % | 17.7 % | **131.1 %** |
| PF [centered] | 6.2 % | 18.1 % | 18.8 % | 50.8 % |

AnyCalib is not only the most accurate, it is by far the most self-consistent:
looking at eight different moments of the *same* fixed camera moves its focal
estimate by about 4.6 % of its own mean. GeoCalib is the opposite extreme — in
the worst 10 % of views its estimate varies by more than its own mean.

## 8. Q8 — principal point

Only two variants predict one:

| Model | median PP error | normalised |
|---|---|---|
| AnyCalib | **32.1 px** | 0.038 |
| PF [uncentered] | 163.2 px | 0.192 |
| `IMAGE_CENTER_PP` (reference) | **31.5 px** | 0.038 |

This is the clearest negative result of the experiment: **AnyCalib's predicted
principal point is no better than simply assuming the image centre** (32.1 px
vs 31.5 px — the trivial baseline is marginally better). PF-uncentered is five
times worse than the trivial baseline. GeoCalib and PF-centered do not predict
it at all and are reported `NOT_PREDICTED` rather than being credited with the
image centre.

CAM-EXP-002 showed the deployed pipeline never uses a calibrated principal
point, so this does not affect it today; it is measured here as a property of
the calibration models themselves.

## 9. Q9 — failure rate and runtime

Failure rate is **0 %** for every model over all 1400 frames — no crashes, no
missing outputs, no unusable predictions. Median runtime per frame (inference
plus cache read, same machine): PF-centered 50 ms, PF-uncentered 72 ms,
AnyCalib 86 ms, GeoCalib 370 ms. AnyCalib is therefore both the most accurate
and ~4× faster than GeoCalib (`figures/runtime_vs_accuracy.png`).

## 10. Q10 — is single-frame enough, or is multi-frame worth testing?

Single-frame is **not** sufficient against the 5 % target: the best model
reaches it on 11.4 % of frames.

There is, however, concrete evidence that multi-frame aggregation is worth
testing, and the evidence points in two different directions depending on the
model:

* **AnyCalib**: spread across 8 frames of one static camera is only ~4.6 % of
  its mean, while its median error is 16.3 %. The error is therefore dominated
  by a **consistent per-view bias**, not by frame-to-frame noise — averaging
  frames will reduce the small noise term and leave the large bias. Multi-frame
  aggregation alone is unlikely to rescue AnyCalib.
* **GeoCalib and PF**: spread of 17.7–18.8 % (GeoCalib p90 131 %) is comparable
  to or larger than their median error, so a per-view median over frames could
  plausibly help them substantially.

So the honest answer is: aggregation is clearly worth a controlled test, but
the stability data already suggests it will help the *less* accurate models
more than the best one, and will not by itself close a 16 % bias. Whether a
per-view median beats the best single frame is exactly what CAM-EXP-004 should
measure; this report does not assume the outcome.

## 11. What was and was not run

`tables/model_availability.csv`:

| Model | Status |
|---|---|
| AnyCalib (`anycalib_pinhole`) | RUN_SUCCESS |
| GeoCalib (`pinhole`) | RUN_SUCCESS |
| PerspectiveFields (uncentered, centered) | RUN_SUCCESS |
| DEMO_FIXED_5000, IMAGE_CENTER_PP, ORACLE_GT | RUN_SUCCESS (references) |
| **AnyCam** | **OUT_OF_SCOPE_SINGLE_FRAME** |

AnyCam recovers poses and intrinsics **from video** using inter-frame motion.
Feeding it a single still would not be the published method, and the resulting
number would say nothing about AnyCam. It is deliberately not substituted with
an unofficial single-frame variant, and it is the natural candidate for
CAM-EXP-004.

Installation obstacles were solved rather than worked around with unofficial
code: `kornia` pinned to 0.7.2 (newer versions break the GeoCalib import under
torch 2.1.2), PerspectiveFields installed with `--no-deps` because
`albumentations` pulls `stringzilla`, which has no Windows wheel and is not
needed for inference.

## 12. Limitations

- One camera rig, one focal range. GT fx spans a narrow band (~890–1010 px), so
  this measures accuracy at one operating point, not across focal lengths.
- 8 frames per view; the stability estimate is a small sample per view (175
  views give it weight in aggregate).
- Only the pinhole variants were run. AnyCalib also ships `anycalib_gen` /
  `anycalib_dist`, and GeoCalib has radial variants; a distorted-camera model
  might do better on these lenses (GigaHands has noticeable `k1 ≈ −0.39`).
  That is a real caveat on the numbers, not a reason to discount them: the
  pinhole variant is the correct choice when the downstream pipeline is pinhole.
- Images are fed raw, as deployed. No `GT_UNDISTORTED_DIAGNOSTIC` run was made
  because no model failed outright; whether undistorting first would help is
  untested.
- The observation about which views are easy or hard is qualitative; a
  controlled scene-cue ablation is CAM-EXP-005.
- The ±5 % target is imported from CAM-EXP-002's sensitivity measurement on
  this data and this working distance; it is not a universal constant.

## 13. Next steps

1. **CAM-EXP-004**: per-view aggregation over frames, plus AnyCam as a genuine
   video method. Test explicitly whether aggregation reduces the per-view bias
   or only the noise.
2. Try the distortion-aware variants (`anycalib_gen`, GeoCalib radial) — the
   rig has real radial distortion that the pinhole variants cannot represent.
3. Since the per-camera spread is large and per-frame noise is small for the
   best model, a per-view bias correction (or any cue that resolves scale, such
   as the hand itself in CAM-EXP-006) looks more promising than more frames.

## Artifacts

```
model_output_convention_audit.md               source-level convention audit
tables/model_output_conversion.csv             machine-readable conversion table
tables/model_availability.csv                  what ran, what did not, and why
tables/adapter_numeric_trace.csv               5 hand-checkable frames per model
tables/model_ranking.csv                       ranked models (ORACLE_GT excluded)
tables/stability_by_model.csv                  static-camera instability
tables/representative_frames.csv               best / typical / worst picks
results/raw/{model}_predictions.csv.gz         per-frame predictions, 1400 rows each
results/raw/all_predictions.csv.gz             9800 rows, all models
results/raw/inference_failures.csv             every failure, with a reason
results/summary/model_summary.csv              headline metrics per model
results/summary/{per_sequence,per_camera}_summary.csv
results/summary/single_frame_stability.csv     1050 view x model records
figures/CAM_EXP_003_MAIN_EXPLANATION.png       everything on one sheet
figures/{focal_error_boxplot,focal_error_cdf,within_threshold_rates,
         predicted_vs_gt_focal,single_frame_stability,
         per_camera_error_heatmap,runtime_vs_accuracy}.png
figures/examples/*_best_worst_grid.png         best / typical / worst frames per model
```

Every summary is derived from `results/raw/all_predictions.csv.gz` alone and can
be regenerated with `python src/evaluate.py && python src/figures.py`.
