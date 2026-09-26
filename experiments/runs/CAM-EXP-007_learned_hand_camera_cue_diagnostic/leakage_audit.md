# CAM-EXP-007 — leakage audit

Per-feature table: `tables/hand_feature_leakage_audit.csv`.

## 1. What every primary feature must satisfy

For all 78 primary hand features, and for the 1280-d latent:

```
uses_dataset_2d_annotation = 0
uses_dataset_3d_annotation = 0
uses_GT_focal              = 0
uses_GT_distortion         = 0
uses_GT_extrinsic          = 0
uses_camera_ID             = 0
uses_sequence_ID           = 0
uses_anycalib_prediction   = 0
uses_pipeline_focal        = 0
```

Automated check result: **0 primary features with a forbidden source.**

## 2. The focal-contamination rule

The hand model internally converts its crop-space camera output into a metric
translation using the pipeline focal:

```
cam_t_full = _cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal)
```

Anything downstream of that multiplication carries the camera focal, so using
it as a predictor would make the question circular. Excluded accordingly:

| excluded output | why |
| --- | --- |
| `cam_t` / `pred_cam_t` | computed with `scaled_focal` |
| `focal_length` | is `PIPELINE_BASELINE_FOCAL` itself |
| absolute vertices / absolute joints | placed by `cam_t` |

What is used instead is the model output **before** that conversion:
`pred_cam` in crop space, and root-relative 3D joints with the root subtracted
explicitly in the feature code.

## 3. The two deliberate non-hand references

Both are kept strictly outside every primary feature set.

| id | consumes | role |
| --- | --- | --- |
| `C1` | `log(f_anycalib)` | `SINGLE_FOCAL_PRIOR_CONFOUND_CONTROL`. Reaches 1.03 % median error and Spearman +0.969 - which is precisely the point: it measures the confound, not a cue. |
| `B3` | training-fold median reference focal | `ORACLE_SANITY_CONTROL`, uses training target labels, not deployable. |

AnyCalib's focal appears in the evaluation formula
`f_corrected = f_anycalib / exp(bias_hat)` for every probe. That is the
correction being diagnosed, not an input feature: no probe except `C1` ever
receives it as a predictor.

## 4. Phase separation

`src/run_hand_inference.py` and `src/extract_hand_features.py` import nothing
that touches the target, the reference focal or any AnyCalib output. The first
file to read `view_targets.csv.gz` is `src/run_probes.py`, which ran only after
every spec, split and feature had been frozen.

## 5. Preprocessing containment

Median imputation, standardisation and latent PCA are fit on **training-fold
views only**, inside each outer fold, and then applied to the held-out camera.
The ridge alpha is selected by grouped CV inside the training fold. The outer
test set never influences any fitted quantity.
