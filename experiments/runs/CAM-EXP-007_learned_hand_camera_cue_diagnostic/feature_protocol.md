# CAM-EXP-007 — feature protocol

Frozen in `experiments/manifests/cam_exp_007_feature_spec_v1.json`,
`created_before_target_analysis = true`.

## 1. Frame grid

The frozen CAM-EXP-006 16-frame nested grid, reused unchanged: 175 views × 16
frames = 2800 frames. Frames were ordered by farthest-point insertion in
CAM-EXP-004.1, long before this run. No frame was chosen by looking at an
AnyCalib error, a hand-model success or a reference focal.

A frame with no detection is **never replaced**. It stays in the manifest and
becomes an availability statistic.

## 2. Eligibility

A view is `HAND_FEATURE_ELIGIBLE` if at least **4 of its 16 frames** yield at
least one usable hand prediction. Otherwise `HAND_FEATURE_UNAVAILABLE`, and it
is excluded from the hand-probe common set but reported in coverage.

Result: 160 of 175 views eligible. Of the 15 excluded, 13 had zero detections.

## 3. Aggregation

```
hands  -> frame : median over hands (mean for vectors and the latent)
frames -> view  : median and IQR, plus std for the temporal group
```

A frame contributes **exactly once** whether it holds one hand or two. This is
the frame-balanced rule, adopted deliberately after CAM-EXP-006.1 found the
same mismatch in CAM-EXP-006's aggregation.

## 4. Groups

| group | what it is | n features |
| --- | --- | ---: |
| F0 | detector and crop geometry on the image plane | 20 |
| F1 | the model's weak-perspective camera head, crop space | 6 |
| F2 | root-relative predicted hand shape and pose | 42 |
| F3 | temporal dispersion across a static view | 10 |
| F4 | pooled image-encoder latent | 1280 → 32 PCA |

F0 is labelled screen/detection geometry, **not** a learned hand
representation, so that a positive result there would not be overclaimed.

## 5. Focal-free construction

Root-relative joints have the root subtracted explicitly in the feature code
rather than relying on the upstream convention. `pred_cam` is taken in crop
space and never converted to a metric translation. See `leakage_audit.md` §2.

Left hands: the wrapper already mirrors them into a canonical right-hand frame.
Only side-invariant scalars are used, and `pred_cam.tx` is mirrored with the
model's own `2*is_right - 1` convention. No extra mirror transform was invented
here.

## 6. Quality audit

Target-independent criteria only: all-NaN, constant, near-zero variance, or NaN
rate above 50 %. **78 features, 0 dropped.** Table:
`tables/hand_feature_quality_audit.csv`.

## 7. Missingness

Training-fold median imputation, fit inside each outer fold.
`availability_rate` and `n_frames_with_hand` are themselves features.
