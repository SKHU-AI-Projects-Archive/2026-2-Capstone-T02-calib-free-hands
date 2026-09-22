# Focal usage audit — existing AnyHand/WiLoR pipeline

Source-level trace of where a focal length enters the existing pipeline and what
coordinate system it lives in. Performed before any experiment, because the
whole design of CAM-EXP-002 depends on the answer. Nothing here is inferred from
behaviour; every claim cites a file and line.

Audited at commit `bf74e0c`, branch `kjh`.

## 1. Where the focal comes from

**`rgb_predictor.py:407-411` (WiLoR) and `rgb_predictor.py:446-450` (HaMeR)**

```python
scaled_focal = float(
    model_cfg.EXTRA.FOCAL_LENGTH      # 1000   (models/model_config_wilor.yaml)
    / model_cfg.MODEL.IMAGE_SIZE      # 256
    * max(W, H)                       # max(image width, height)
)
```

The focal is **not measured, not read from any calibration, and not estimated
from the image**. It is a constant of the training configuration rescaled by the
image size. For a GigaHands frame (1280 x 720):

```
scaled_focal = 1000 / 256 * 1280 = 5000.0 px
```

`EXTRA.FOCAL_LENGTH = 1000` and `MODEL.IMAGE_SIZE = 256` are confirmed in
`models/model_config_wilor.yaml`.

**Answer to "which kind of focal is this?"** — of the four candidates in the
brief:

| Candidate | Verdict |
|---|---|
| A. physical focal in original RGB pixels | **No** — no calibration is ever read |
| B. focal of a resized image coordinate system | **No** — there is no resize (see §3) |
| C. crop coordinate system focal | **No** — `scaled_focal` is scaled to full-image size, not crop |
| D. **training-convention virtual focal** | **Yes** — `FOCAL_LENGTH/IMAGE_SIZE` is the training normalisation, re-expressed in full-image pixels |

So it is a **virtual focal expressed in original full-image pixel units**. It
encodes the assumption that every image was taken with the same
field-of-view as the training convention, namely `f = 1000/256 * max(W,H)`,
i.e. roughly **3.9 x max(W,H)** — a very long lens / narrow field of view.

## 2. Where the focal is used

**`rgb_predictor.py:948-976`, `_cam_crop_to_full`**

```python
denom = s * box_size + 1e-9
tz    = 2.0 * focal_length / denom
tx    = tx_crop + 2.0 * (cx_box - cx_img) / denom
ty    = ty_crop + 2.0 * (cy_box - cy_img) / denom
```

with `cx_img = W/2`, `cy_img = H/2` (lines 969-970).

Three consequences follow directly from this code, and they shape the whole
experiment:

1. **`tz` is exactly proportional to the focal.** `s` (predicted weak-perspective
   scale) and `box_size` come from the network and the detector and do not
   depend on the focal, so `tz(alpha) = alpha * tz(1)` is an algebraic identity,
   not an empirical expectation.
2. **`tx` and `ty` do not contain the focal at all.** Changing the focal moves
   the hand purely along the camera Z axis; its metric X and Y are untouched.
3. **The principal point is never the calibrated one.** The code uses the image
   centre. The real `cx, cy` from `optim_params.txt` never enters the pipeline.

A fourth consequence is worth stating because it is easy to misread: since
`tz ∝ f` and `tx, ty` are constant, the *projected* 2D position is invariant to
the focal:

```
u = f * tx / tz + cx_img = f * tx * (s*B) / (2f) + cx_img = tx*s*B/2 + cx_img
```

The focal cancels. So a wrong focal produces a hand that reprojects onto exactly
the same pixels while sitting at a wrong metric depth — which is precisely why a
2D-only sanity check cannot detect it, and why this experiment is needed.

**`rgb_predictor.py:873-918`, `project_3d_to_2d`** uses the same focal and the
same image-centre principal point, and forms absolute points as
`points + cam_t`. This confirms that `HandPrediction.keypoints_3d` is
**root-relative** (the docstring on line 118 calling it "camera-space" is
misleading); absolute camera-space joints are `keypoints_3d + cam_t`.

**`rgb_predictor.py:560`** stores `focal_length = scaled_focal` on every
`HandPrediction`, and **line 783** notes it may differ per hand only if image
sizes differ.

Inside the network (`WiLoR/wilor/models/wilor.py:117, 138`) the same constant
`FOCAL_LENGTH` is used to turn `pred_cam` into a translation in the *crop*
frame, with `focal_length / MODEL.IMAGE_SIZE` normalisation
(`wilor.py:157`, `refinement_net.py:172-176`). That is the training convention
which `scaled_focal` mirrors at full-image scale.

## 3. Is there a resize?

No. `rgb_predictor.py:404-405` takes `W, H` straight from `img_bgr.shape`, and
`ViTDetDataset` receives the same `img_bgr`. `box_center` and `box_size` are
therefore in **original image pixels**, and `scaled_focal` is in the same units.
`rescale_factor` (line 204, default 2.0) enlarges the *bounding box*, not the
image, and feeds `box_size`; it is focal-independent.

## 4. The three focal quantities, kept separate

| Name | Definition | Value for a GigaHands 1280x720 frame |
|---|---|---|
| `PIPELINE_BASELINE_FOCAL` | what the code uses today, `1000/256 * max(W,H)` | **5000.0 px** |
| `GT_NATIVE_FX` / `GT_NATIVE_FY` | GigaHands `optim_params.txt` intrinsics for that camera | e.g. **921.61 / 919.36 px** (`brics-odroid-001_cam0`) |
| `GT_EFFECTIVE_FOCAL` | the physical focal expressed in the coordinate system the pipeline's focal lives in | **= `GT_NATIVE_FX`** |

`GT_EFFECTIVE_FOCAL = GT_NATIVE_FX` **exactly**, with no transform, because:

* the pipeline focal is in original full-image pixels (§1, §3);
* GigaHands intrinsics are in the same original 1280x720 pixels — the same
  coordinate system in which `keypoints_2d` lives and in which CAM-EXP-001
  validated the projection to ~3 px;
* there is no resize and no crop rescaling applied to the focal.

So the conversion required by §3 of the brief is the identity, and the
experiment is well-posed. This is derived, not assumed.

**Anisotropy.** GigaHands has separate `fx` and `fy`, but the pipeline uses a
single scalar and only in `tz = 2f/(s*B)`. `fx` is used as the scalar because it
is the first intrinsic and the one the pipeline's `u = f*x/z + cx` form
corresponds to; `fy` differs from `fx` by **0.1-0.3 %** across the demo cameras,
so this choice changes `tz` by well under a percent and is recorded per row
(`gt_native_fx`, `gt_native_fy`) so any reader can redo it with `fy`. No
mean/sqrt combination was invented, per §9 of the brief.

## 5. The headline discrepancy

```
PIPELINE_BASELINE_FOCAL / GT_NATIVE_FX  =  5000 / 921.6  =  5.43
```

The pipeline assumes a focal roughly **5.4x longer** than the camera actually
has. Because `tz ∝ f`, the predicted hand depth is expected to be about **5.4x
too large** — a metre-scale absolute placement error for a hand at ~0.6 m. This
is a property of the existing pipeline, visible from the source alone, and the
sweep measures it rather than assuming it.

This also means `PIPELINE_BASELINE` and `GT x 1.00` are very far apart, exactly
as §10 of the brief warned they might be; they are reported as separate
conditions throughout.

## 6. Principal-point sensitivity (brief §14)

The calibrated principal point is **not used anywhere** in the pipeline's
translation computation: `_cam_crop_to_full` uses `W/2, H/2`, and
`project_3d_to_2d` does the same. Therefore:

> The current pipeline is insensitive to the calibrated principal point because
> it does not use it. Substituting the real `cx, cy` would change nothing in the
> predicted camera translation.

No artificial principal-point effect is manufactured. What the image-centre
assumption *does* cause is a fixed offset between the pipeline's implied optical
axis and the real one (for `brics-odroid-001_cam0`, real `cx, cy` = 633.4, 338.7
vs assumed 640.0, 360.0 — about 6.6 px and 21.3 px), which biases `tx, ty`
through the `(cx_box - cx_img)` term. That is recorded as an observation, not
swept.

## 7. What this means for the sweep

Because the focal enters only through `tz`, the sweep can be applied to cached
inference output exactly:

```
tz(alpha)  = alpha * tz_at_GT
tx, ty     = unchanged
joints_abs = keypoints_3d + [tx, ty, tz(alpha)]
```

So a single inference per frame is not merely an efficiency measure — it makes
the focal effect *exactly* isolated, with no re-detection or re-regression noise.
The predictions `Z(alpha)/Z(1) = alpha` and "root-aligned pose is invariant" are
falsifiable checks on the implementation, and are verified in the report.

## 8. Trace table

Machine-readable version: `tables/focal_usage_trace.csv`.
