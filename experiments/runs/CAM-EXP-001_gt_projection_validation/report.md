# CAM-EXP-001 — GT Projection Validation

Run date 2026-09-22 · branch `kjh` · base commit `224ea50` · no GPU used.
Full environment in `environment.json`, exact parameters in `config.json`.

## 1. Purpose

Establish that we read each dataset's **ground-truth camera** and **ground-truth
3D joints** correctly, by reprojecting the 3D joints into the image and
comparing against the dataset's own 2D annotation.

This is a **dataset-loader / camera-convention validation**, not an evaluation
of any calibration method. No claim is made here about any algorithm. Its role
is to make every later experiment (CAM-EXP-002 onwards) trustworthy: if the
world→camera transform, the unit, the rotation direction or the distortion
model were misread, every downstream number about camera error would be wrong
in a way that is very hard to detect later.

## 2. Hypothesis

> If the official GT camera and GT 3D joints are interpreted according to each
> dataset's own convention, the projection will agree with the provided 2D
> annotation at low reprojection error.

Explicitly **not** assumed: that the same convention holds across datasets,
that all datasets provide a 2D annotation, or that any particular error value
counts as success.

## 3. Datasets

| Dataset | Subset used | Samples | Valid joints | Independent 2D? |
|---|---|---|---|---|
| GigaHands | `demo_all`, 5 sequences, up to 40 cams, 20 frames/seq/hand | 8000 | 168,000 | yes — per-view detections |
| InterHand2.6M | `human_annot/test` | 1816 | 35,676 | no — bbox only |
| HanCo | `tester`, sequence `0110`, 8 cameras | 160 | 3,360 | no |
| AssemblyHands | `demo` | 32 | 672 | yes (but see §7) |
| AssemblyHands | `test-eccv2024` | 0 | 0 | GT withheld — rejected by loader |

A sample is one `(sequence, camera, frame, hand)` observation.

## 4. Camera / coordinate convention

Each loader converts into one canonical form,
`X_cam = R · X_world + t`, then `K · distort(X_cam / z)`:

| Dataset | Stored as | Conversion | Unit | Distortion |
|---|---|---|---|---|
| GigaHands | `optim_params.txt`: COLMAP `qvec(w,x,y,z)`, `tvec` | `R = quat_to_rotmat(qvec)`, `t = tvec` | metres | OpenCV `[k1,k2,p1,p2]` — **applied** |
| InterHand2.6M | `camrot`, `campos` | `R = camrot`, `t = -camrot · campos` | millimetres | none (images undistorted) |
| HanCo | per-frame `K[8]`, `M[8]` (4×4) | `R = M[:3,:3]`, `t = M[:3,3]` | metres | none (images undistorted) |
| AssemblyHands | per-frame 3×4 `[R\|t]` | used directly | millimetres | none (rectified) |

Conventions were **established empirically, not assumed**. For GigaHands, four
candidate transforms were swept over 40 cameras:

| Variant | Median-of-medians error |
|---|---|
| `R·X + t` (COLMAP) | **30.97 px** |
| `Rᵀ·X + t` | 399.69 px |
| `R·(X − t)` | 1280.80 px |
| `Rᵀ·(X − t)` | 1185.28 px |

Applying the provided distortion then halved the error (12.26 → 5.35 px
median), so distortion is material for GigaHands and is not ignored.

Frame correspondence was likewise verified rather than assumed: the GigaHands
2D `jsonl` has 381 rows and the source video has exactly 381 frames, so the row
index is the video frame index; `chosen_frames_<hand>.json` marks the frames
with a valid 3D fit.

## 5. Method

1. Load GT camera, GT 3D joints and (where present) GT 2D per sample.
2. Project with the canonical model, applying distortion where provided.
3. Record **one row per joint** with both GT and projected coordinates,
   the error, depth, validity, in-image flags and image size.
4. Derive all statistics, tables and figures from that raw table only.

A joint enters the error statistics when it is annotated-valid **and** both the
GT point and the projection fall inside the image. This excludes joints that are
simply outside the frame, which would otherwise dominate the statistics without
saying anything about the convention.

Reproduce with `python experiments/run_cam_exp_001.py --full` followed by
`python experiments/analyze_cam_exp_001.py`.

## 6. Metrics

Mean / median / p90 / p95 / max / RMSE of the reprojection error, plus the
fraction of joints below a range of thresholds. Thresholds are reported to
**describe the shape of the distribution**, never as a pass/fail line: an
acceptable error differs per dataset depending on how its 2D was produced.
Per-camera, per-sequence, per-hand and per-joint breakdowns are in
`results/summary/reprojection_by_group.csv` and `tables/per_joint_summary.csv`.

## 7. Results

### GigaHands (the one real independent test)

157,451 comparable joints:

| median | p90 | p95 | max | RMSE |
|---|---|---|---|---|
| **13.17 px** | 687.70 | 820.97 | 1457.64 | 338.21 |

The distribution is strongly **bimodal**, which the aggregate hides. Grouping
by view (`tables/view_agreement_summary.csv`, 7824 views):

| View median error | Fraction of views |
|---|---|
| < 5 px | 17.8 % |
| < 10 px | 44.9 % |
| < 20 px | 58.5 % |
| < 50 px | **64.9 %** |
| < 500 px | 80.4 % |

Within the 5078 views below 50 px, the median is **7.21 px** and p90 is
19.88 px. The remaining ~20 % of views sit near ~690 px — roughly the distance
between the two hands in a 1280-wide frame.

Per-joint medians are uniform (10.2–18.0 px across all 21 joints, see
`tables/per_joint_summary.csv`), so the large errors are **not** a bad joint
index, a wrong skeleton order or a hand swap: entire views fail together. That
is the signature of a failed or mis-assigned per-view 2D detection, not of a
wrong camera convention.

The overlays confirm this directly (`results/summary/overlay_index.csv`): on the
same frame 0 of `p36-tea-0010`, the projection lands on the hand at **3.37 px**
in `brics-odroid-007_cam1` and **3.42 px** in `brics-odroid-007_cam0`, while
`brics-odroid-001_cam0` reports 808 px — a camera in which that hand is not
visible at all.

### AssemblyHands (`demo`)

660 joints with 2D, error **0.0000 px** (max 0.0011 px). The shipped 2D
keypoints are reproducible from the calibration to floating-point precision.

This confirms our camera maths matches the dataset authors' **exactly**, but it
is a *circular* check: the 2D was generated by this same projection, so it
cannot reveal an error we share with them. It is reported as an exact-agreement
result, not as independent evidence.

### InterHand2.6M

No reprojection error is computable (no 2D keypoints exist). Geometric checks
over 35,676 valid joints:

- positive depth: **100 %**
- projected inside image: **100 %**
- inside the annotated `bbox`: **99.97 %**

A wrong rotation direction or a mm/m confusion would not produce a 99.97 %
bbox-containment rate, so the convention is confirmed to the strength this
signal allows.

### HanCo

3360 valid joints: positive depth 100 %, inside image 100 %. Six overlays
(`figures/hanco/`) show the projected skeleton landing on the hand in every
camera of the rig, which is the strongest available check given that HanCo has
no 2D annotation.

### AssemblyHands (`test-eccv2024`)

Zero samples. Every `world_coord` and every `keypoints` value in this split is
the constant `1.0`; it is the held-out HANDS2024 benchmark split. Processed
naively it yields a 579 px "median error" over 63,000 joints that looks like a
convention bug but is purely placeholder data. The loader now rejects
degenerate records.

## 8. Problems encountered

1. First GigaHands attempt gave ~800 px. Cause: a camera that cannot see the
   queried hand. Resolved by a convention sweep plus in-image filtering, not by
   assuming the dataset was wrong.
2. A GT 2D point outside the image height suggested a resolution mismatch;
   checking the video (1280×720, 381 frames) showed the indexing was right and
   the point was a failed detection.
3. AssemblyHands camera identifiers differ between the data and calibration
   files (`HMC_84358933` vs `HMC_84358933_mono10bit`); resolved by prefix.
4. `test-eccv2024` placeholder GT, as above.
5. Three of four datasets have no independent 2D annotation, which bounds what
   this experiment can prove for them (§10).

## 9. Interpretation

The canonical camera model is correct for all four datasets. Where an
independent test exists (GigaHands), well-observed views agree at ~7 px median;
where the 2D is derived from the calibration (AssemblyHands) agreement is exact;
where only weaker signals exist (InterHand2.6M bbox, HanCo overlay) every check
passes.

The GigaHands heavy tail is a property of its **per-view 2D detections**, not of
our loader — supported by three independent observations: per-joint errors are
uniform, failures are whole-view, and low-error and high-error views coexist in
the same frame of the same sequence. Consequently GigaHands `keypoints_2d`
should be treated as a noisy per-view signal and filtered (for example by
per-view median) in any experiment that uses it as a reference.

## 10. Pass / fail for loader validation

| Dataset | Verdict | Strength of evidence |
|---|---|---|
| GigaHands | **PASS** | strong — independent 2D, 7.21 px median on 5078 clean views, visual overlay |
| AssemblyHands (`demo`) | **PASS** | exact (0.0000 px) but circular; small sample (16 images) |
| InterHand2.6M | **PASS** | moderate — 99.97 % bbox containment, no 2D keypoints available |
| HanCo | **PASS** | moderate — geometry 100 % consistent + visual overlay on 8 cameras |
| AssemblyHands (`test-eccv2024`) | **N/A** | GT withheld by the dataset authors |
| Re:InterHand | **N/A** | not downloaded by design |

Overall: the loader and camera conventions are validated. At least one dataset
(GigaHands) carries a genuine end-to-end reprojection validation, which was the
requirement for this stage.

## 11. Next step

1. Treat the canonical `Camera` model in `experiments/src/geometry/camera.py`
   as settled and build CAM-EXP-002 (effect of camera error on absolute 3D)
   on top of it.
2. For GigaHands, adopt a per-view quality filter (per-view median error) so
   downstream experiments are not polluted by failed detections.
3. To strengthen the InterHand2.6M and HanCo evidence from "consistent" to
   "measured", a subset of images would be needed — deliberately not downloaded
   here (InterHand full images ≈ 80 GB, HanCo RGB ≈ 14 GB).
4. If AssemblyHands is to carry real weight, obtain the `test-iccv2023` joint_3d
   GT (the downloaded file is a prediction template) or the ego images.

## Artifacts

```
config.json                                     exact run parameters + conventions
environment.json                                date, OS, python, git, numpy, opencv
results/raw/reprojection_per_joint.csv.gz       210,168 rows - the source of truth
results/summary/dataset_summary.csv             per-dataset statistics
results/summary/reprojection_by_group.csv       per camera / sequence / hand
results/summary/view_level_summary.csv          7848 per-view rows
results/summary/overlay_index.csv               figure <-> source frame mapping
tables/reprojection_summary.csv                 headline table
tables/per_joint_summary.csv                    per-joint breakdown
tables/view_agreement_summary.csv               view-level agreement fractions
tables/error_distribution.csv                   threshold fractions
figures/error_histogram.png                     per-joint error, log scale
figures/view_median_histogram.png               per-view median error, log scale
figures/gigahands/, figures/hanco/              GT (green) vs projected (red) overlays
logs/run.log                                    run log
```

Every table and figure is regenerable from the raw CSV alone via
`analyze_cam_exp_001.py`; no dataset access or recomputation is required.
