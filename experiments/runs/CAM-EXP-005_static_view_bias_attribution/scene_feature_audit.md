# Scene feature audit

## Freezing

`experiments/manifests/cam_exp_005_scene_feature_spec_v1.json`
(sha256 `f78d17c1010faddf…`) lists all 42 features and was written **before any
feature was correlated with the target**. The manifest carries
`created_before_target_analysis: true`. No feature was added afterwards because
it happened to correlate.

## Groups

| group | n | what it is | deployable |
|---|---|---|---|
| `F_SCENE_GEOM` | 16 | edges, gradients, LSD line statistics, orientation entropy, corners, vanishing-point consensus, orthogonality support | yes |
| `F_IMAGE_GLOBAL` | 7 | grayscale/saturation statistics, local contrast, Laplacian variance | yes (nuisance/condition, not a geometric claim) |
| `F_BORDER_SCENE` | 8 | the same edge/line statistics restricted to the outer 20 % border ring | yes |
| `F_MODEL_SELF` | 8 | AnyCalib's own outputs: predicted focal, k1, k2, principal-point offset, fx/fy, and within-view dispersions | yes, **but see the warning below** |
| `MODEL_DISAGREEMENT` | 3 | log ratio between the AnyCalib and GeoCalib focals | yes, needs a second model at deployment |

The border ring is a **generic image-region statistic**, not a hand
segmentation. No hand annotation, detector box or hand prediction was used
anywhere in this experiment.

## Extraction

* frames: the frozen 64-frame nested set per view, 11,200 frames total. No frame
  was chosen using the target, the ground truth or any model output.
* implementation: classical OpenCV / NumPy only, no pretrained model.
* line detector: `cv2.createLineSegmentDetector` (LSD) was available, so the
  Hough fallback was not used.
* frame → view aggregation: **median** (and IQR where the spec names one), fixed
  before any correlation was seen.

## Quality audit

`tables/scene_feature_quality_audit.csv`. Criteria are **target-independent**:
missingness above 20 %, all-NaN, zero variance, or relative variance below
1e-10. Nothing is dropped for how it correlates with the bias.

**Result: 42 declared, 42 kept, 0 dropped.** `vp_consensus_strength` produced
finite values on every view, so it is recorded as `FEATURE_IMPLEMENTED` rather
than `FEATURE_SKIPPED_UNSTABLE`.

## The warning on `F_MODEL_SELF`

`self_pred_focal_px` **is** the numerator of the target. With `f_ref` nearly
constant on this rig (CV 1.90 %), `log(f_pred/f_ref) ≈ log(f_pred) − const`, so
this feature is close to definitionally related to what it is predicting. Its
Spearman rho against the signed bias is **+0.969**, which measures that
relationship and not a scene cue.

This is why `F_MODEL_SELF` is always reported separately from the scene-only
groups, and why the `CONSTANT_RIG_FOCAL_ORACLE` control was added to the
correction diagnostic.
