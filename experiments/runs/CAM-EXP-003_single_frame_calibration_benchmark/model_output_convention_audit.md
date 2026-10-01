# Model output convention audit

Done **before** the benchmark, because the most dangerous error in this kind of
comparison is reading a model's focal in the wrong coordinate system — e.g.
treating a focal expressed in 420 px network pixels as if it were in 1280 px
original pixels, which would wrongly inflate the error by 3x.

Every claim below is checked twice: once by reading the official source, and
once numerically on a real GigaHands frame with known ground truth
(`brics-odroid-001_cam0`, 1280x720, GT fx = 921.61, fy = 919.36, cx = 633.37,
cy = 338.68). Machine-readable version: `tables/model_output_conversion.csv`.
Per-frame raw output and the applied conversion are kept in every prediction
row (`raw_output`, `conversion_note`), and five frames per model are traced in
`tables/adapter_numeric_trace.csv`.

## AnyCalib (ICCV 2025)

* Repo `github.com/javrtg/AnyCalib`, installed from `main`; checkpoint
  `anycalib_pinhole.pt` auto-downloaded from the official GitHub release
  `v1.0.0` (1.19 GB) into the torch hub cache.
* API: `AnyCalib(model_id="anycalib_pinhole").predict(image, cam_id="pinhole")`.
* Raw output: `intrinsics` = `[fx, fy, cx, cy]`, plus `pred_size`, `rays`,
  `fov_field`, `success`.

**Coordinate system — the one genuinely ambiguous case.** The README states
that focal lengths are "expressed relative to the resized network input
(pred_size), not the original image dimensions". The shipped code says
otherwise: `AnyCalib.predict` ends with

```python
pred["intrinsics"][i] = cam.reverse_scale_and_shift(intrins, scale_xy, shift_xy)
```

which undoes the internal resize. Two independent checks settle it in favour of
the code:

1. the returned principal point is `(638.3, 360.2)` — the centre of the
   **1280x720** image, not of the 420x238 network input;
2. the returned focal (1273 px on the probe frame) is of the right order for
   the original image; interpreted as network pixels and rescaled it would be
   ~3900 px, which is not physically plausible for this rig.

**Conversion applied: none.** `intrinsics` is already in original-image pixels.

Predicts a principal point (evaluated in §14 of the brief). The pinhole variant
predicts no distortion.

## GeoCalib (ECCV 2024)

* Repo `github.com/cvg/GeoCalib`, installed from `main`; weights
  `geocalib-pinhole.tar` auto-downloaded from the official release `v1.0`
  (111 MB).
* API: `GeoCalib().calibrate(image)` -> `{"camera", "gravity", ...}`.
* Raw output: `camera.f` = `[fx, fy]`, `camera.size` = `[W, H]`,
  `camera.c`, `camera.vfov`, plus gravity/roll/pitch and uncertainties.

**Coordinate system: original-image pixels, unambiguous.** `camera.size`
returns `[1280.0, 720.0]` for the probe frame — the original size — and the
returned `f = 937.79` satisfies `f = (H/2)/tan(vfov/2)` with `H = 720`
(`360/tan(21.0°) = 937.7`). **Conversion applied: none.**

**Principal point is not predicted.** GeoCalib fixes it at the image centre by
design (`camera.c` returns exactly `(640, 360)`), so this adapter reports
`NOT_PREDICTED` rather than passing the image centre off as an estimate.

The pinhole variant returns an isotropic focal (`fx == fy` to 4 decimals).

## Perspective Fields (CVPR 2023)

* Repo `github.com/jinlinyi/PerspectiveFields`, commit
  `d54be737d6eacfb9d39a2b7079a494924b45bb6c`; checkpoints
  `Paramnet-360Cities-edina-centered` and `-uncentered` from the official
  Hugging Face release.
* API: `PerspectiveFields(version).inference(img_bgr=...)`.
* Raw output: `pred_roll`, `pred_pitch`, `pred_general_vfov` (degrees),
  `pred_rel_focal`, `pred_rel_cx`, `pred_rel_cy`, plus dense up/latitude fields.

**Coordinate system: focal relative to image HEIGHT.** The official converter
`perspective2d/utils/utils.py::general_vfov_to_focal(rel_cx, rel_cy, h, gvfov,
degree)` solves for `focal/h`, and its docstring states the result "is relative
to the image height if h is set to 1". So

```
f_px = pred_rel_focal * image_height
```

Verified numerically with the model's own converter: for the **uncentered**
checkpoint, `general_vfov_to_focal(0.0759767, -0.3120852, 1, 41.902946, True)`
returns `1.2353644` against the model's `pred_rel_focal = 1.2353643` — a match
to seven decimals, which confirms both the relation and the height
normalisation.

**A discrepancy that is recorded rather than smoothed over.** For the
**centered** checkpoint the same identity does *not* hold: from its reported
`pred_general_vfov = 39.4117°` the official converter gives `1.3960`, while the
model outputs `pred_rel_focal = 1.0679`. The two are inconsistent, so that
checkpoint's reported vfov is not the quantity its focal head produced. The
adapter therefore uses `pred_rel_focal` (the network's own focal output) for
both checkpoints, and stores the vfov-implied focal beside it as
`extra_fx_px_from_vfov` so any reader can recompute the other way. No attempt
was made to pick whichever convention scored better against ground truth.

The uncentered checkpoint predicts a principal point
(`cx = (0.5 + rel_cx) * W`, `cy = (0.5 + rel_cy) * H`); the centered one fixes
it at the image centre and is reported as `NOT_PREDICTED`.

## AnyCam — not benchmarked, and why

`github.com/Brummi/anycam` (CVPR 2025) recovers camera **poses and intrinsics
from casual video**: its input is a sequence and its method is built on
inter-frame motion. Running it on a single still frame would not be the method
its authors describe, and any number produced that way would say nothing about
AnyCam. It is therefore recorded as **OUT_OF_SCOPE_SINGLE_FRAME** rather than
BLOCKED, and it is the natural candidate for CAM-EXP-004, where multi-frame
aggregation is the subject.

## Reference baselines

* `DEMO_FIXED_5000` — the focal the deployed AnyHand/WiLoR pipeline assumes,
  `FOCAL_LENGTH/IMAGE_SIZE * max(W,H) = 1000/256 * 1280 = 5000 px`
  (audited in CAM-EXP-002). It is a real deployed estimator, so it is ranked.
* `IMAGE_CENTER_PP` — `cx = W/2, cy = H/2`. Principal-point reference only; it
  predicts no focal and is excluded from focal rankings.
* `ORACLE_GT` — reproduces the ground truth. Reference line only, never ranked;
  its 0.00 % error is also the end-to-end sanity check that the evaluation code
  computes errors correctly.

## Ground-truth convention

GigaHands `optim_params.txt` intrinsics, in original 1280x720 pixels, as
validated in CAM-EXP-001 (reprojection to ~3 px). `fx` and `fy` differ by
0.1-0.3 %.

Because every model here returns either a single scalar focal or an
`fx == fy` pair, no scalar convention had to be invented. The primary metric is
computed against `gt_fx`, and the error against `gt_fy` is stored in every row
(`fy_relative_error_pct`); the two medians differ by at most 0.2 percentage
points for every model, so the choice does not affect any conclusion.

## Distortion

GigaHands provides OpenCV `[k1, k2, p1, p2]`. None of the three benchmarked
model variants predicts distortion in the configuration used (all are pinhole
or pinhole-equivalent), so no distortion comparison is made — there is nothing
to convert. Images are fed **raw**, undistorted by nothing: the point is to
calibrate real RGB as deployed. No `GT_UNDISTORTED_DIAGNOSTIC` run was needed,
since no model failed in a way that suggested distortion was the cause (failure
rate is 0 % for all three).
