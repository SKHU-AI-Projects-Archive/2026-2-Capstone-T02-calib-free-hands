# Distortion model audit

Performed before running anything, because comparing distortion coefficients
across different parameterisations is the easiest way to produce a confident
wrong answer. Machine-readable version:
`tables/distortion_parameter_conversion.csv`.

## Ground truth

GigaHands `optim_params.txt` gives OpenCV Brown-Conrady coefficients
`[k1, k2, p1, p2]`, applied in normalised camera coordinates:

```
r² = x² + y²
x_d = x(1 + k1 r² + k2 r⁴) + 2 p1 x y + p2 (r² + 2x²)
y_d = y(1 + k1 r² + k2 r⁴) + p1 (r² + 2y²) + 2 p2 x y
u = fx x_d + cx ,  v = fy y_d + cy
```

Over the 175 benchmark views the radial term dominates: `|k1|` ranges
**0.342 – 0.431** (median ≈ 0.39, barrel), while the tangential terms are two
orders of magnitude smaller (`p1 ≈ 3e-3`, `p2 ≈ 7e-4`).

## AnyCalib

Available checkpoints (from the installed package, not from documentation):
`anycalib_pinhole`, `anycalib_dist`, `anycalib_gen`, `anycalib_edit`.
Available camera models (`anycalib.cameras.factory.CameraFactory.FACTORY`):
`pinhole`, `simple_pinhole`, `radial`, `simple_radial`, `kb`, `simple_kb`,
`ucm`, `simple_ucm`, `eucm`, `simple_eucm`, `division`, `simple_division`, `fov`.

The `radial` model's own docstring states:

```
x = fx * (X / Z) * (1 + k1 * r^2 + k2 * r^4 + ...) + cx
y = fy * (Y / Z) * (1 + k1 * r^2 + k2 * r^4 + ...) + cy
parameters: fx, fy, cx, cy, k1, k2, ...
```

This is **the same normalised radial polynomial as OpenCV**, with the same
meaning for `k1` and `k2`. `num_k` defaults to 2, so `cam_id="radial"` and
`cam_id="radial:2"` produce identical output (verified numerically).

**Verdict: `k1`/`k2` are directly comparable to the GT radial terms.** The one
real difference is that AnyCalib's `radial` has **no tangential terms**, so it
cannot represent GT `p1, p2`. Given those are ~100x smaller than `k1` here, the
radial comparison is meaningful, and that limitation is stated rather than
hidden.

Variants run: `anycalib_dist` and `anycalib_gen`, both with `cam_id="radial:2"`.

## GeoCalib

Available weights: `pinhole` and **`distorted`**. Available camera models
(`geocalib.camera.camera_models`): `pinhole`, `radial`, `simple_radial`,
`simple_divisional`. Their docstrings:

* `radial` — "The distortion model is 1 + k1 * r^2 + k2 * r^4 where r^2 = x^2 + y^2"
* `simple_radial` — "1 + k1 * r^2"
* `simple_divisional` — "(1 - sqrt(1 - 4 k1 r^2)) / (2 k1 r^2)" (a division model)

**Verdict: GeoCalib's `radial` uses the same convention as OpenCV and AnyCalib,
so its `k1`/`k2` are directly comparable too.** `simple_divisional` is a
different parameterisation and would need a conversion, so it is not used.

Variant run: `weights="distorted"`, `camera_model="radial"`.

## Perspective Fields

The installed package exposes no distortion-aware checkpoint: the released
`Paramnet-360Cities-edina-{centered,uncentered}` models output roll, pitch,
general vfov, relative focal and (uncentered) a relative principal point, with
no distortion term anywhere in the output dict.

**Status: `NO_DISTORTION_VARIANT`.** Nothing is substituted. PF still appears in
condition C, where the *image* is undistorted and the model is unchanged.

## What is and is not compared numerically

| Comparison | Status |
|---|---|
| AnyCalib `k1`,`k2` vs GT `k1`,`k2` | **VALID** — identical normalised radial polynomial |
| GeoCalib `k1`,`k2` vs GT `k1`,`k2` | **VALID** — identical convention |
| any model's tangential terms vs GT `p1`,`p2` | **SKIPPED_INCOMPATIBLE_PARAMETERIZATION** — no model predicts tangential distortion |
| `simple_divisional` vs GT | not run; division model is a different parameterisation |

Measured `k1` (median over 1400 frames) against a GT median of ≈ −0.39:

| Run | predicted `k1` median |
|---|---|
| `anycalib_dist` / radial | −0.309 |
| `anycalib_gen` / radial | −0.312 |
| `geocalib_distorted` / radial | −0.369 |

All three recover barrel distortion of about the right magnitude, GeoCalib
closest in `k1` — which, as the results show, does not translate into the best
focal.

## Model-independent ray metric

The brief allows a ray-angle comparison as a parameterisation-neutral metric,
but only "when a reliable unprojection API can be obtained" and explicitly
forbids inventing distortion equations. Here it was **not needed**: every model
that predicts distortion in this run uses the OpenCV-compatible radial
polynomial, so the coefficients are directly comparable and the focal metric is
already expressed in one common convention. Adding a ray metric would have
required re-implementing each library's unprojection, which is exactly the kind
of hand-written geometry the brief warns against. Recorded as
`NOT_REQUIRED_CONVENTIONS_ALREADY_COMPATIBLE` rather than silently skipped.

## GT-undistortion bookkeeping (condition C)

`cv2.undistort(img, K, D, None, K_new)` with
`K_new, roi = cv2.getOptimalNewCameraMatrix(K, D, (W,H), alpha=0, (W,H))`.

`alpha = 0` was chosen so the output keeps the original 1280×720 size with no
invalid black border — a border would itself change what a calibration model
sees and would confound the comparison.

**The critical consequence**: undistorting changes the camera. Over the 1400
frames `K_new.fx / K.fx` has median **0.7734** (range 0.744 – 0.855). The
ground truth for a condition-C prediction is therefore `K_new`, never the
original focal. Comparing a condition-C prediction against the original `fx`
would fabricate a ~23 % error (or, with the opposite sign, a fake improvement).
Every per-frame `K_new`, the alpha, and the valid ROI are stored in
`results/raw/gt_undistortion_bookkeeping.csv.gz`, and each condition-C
prediction row carries `gt_fx_is_K_new=1` plus `orig_gt_fx` so the choice is
auditable.
