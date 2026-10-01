# Reference geometry provenance

## Source

`OTHER_CAMERA_ONLY_REFERENCE_3D`, produced by CAM-EXP-001.3's `src/loco.py`,
imported and used **unchanged**:

- threshold 8.0 px
- minimum 4 inlier cameras
- the camera under test is removed from the observation set **before**
  reconstruction, so its own 2D annotation cannot shape the hand geometry later
  attributed to it

This is the same reconstruction CAM-EXP-006 used, with the same parameters.

`results/summary/reference_geometry_audit.json` records
`target_camera_in_reference_count`, which must be **0**.

## Eligibility

Eligibility comes only from the frozen CAM-EXP-001.3 QC
(`gigahands_demo_qc_v1.csv.gz`), accepting `PASS_STRICT` and
`PASS_SINGLE_HAND` with `triangulation_success = 1`, plus a requirement that
all 21 joints reconstruct so every bone has both endpoints.

**No frame is ever removed for having a large left/right distance, an unusual
bone, or any other metric outcome.** Frames are capped per unit at 48, evenly
spaced by frame index - deterministic, target-independent, and fixed from
measured runtime (0.122 s per reconstruction) before any distance existed.

## What this geometry is, and is not

The reference 3D mixes:

- annotation error in the per-view 2D detections
- triangulation and calibration error
- hand association error
- and actual anatomy

It is therefore called **reference geometry**, never `true anatomy`,
`physical bone truth` or `ground-truth asymmetry`. Any left/right difference
measured here contains reconstruction error of unknown size.

## Historical calibration dependency

Building `OTHER_CAMERA_ONLY_REFERENCE_3D` requires the dataset's camera
calibration, which includes intrinsics. That is a **historical reconstruction
dependency** of an artefact this run consumes.

It is not focal estimation: this run performs no focal optimisation, reads no
focal value into any analysis, and its outputs are invariant to what the focal
happens to be. The distinction is recorded in
`tables/focal_independence_audit.csv`.
