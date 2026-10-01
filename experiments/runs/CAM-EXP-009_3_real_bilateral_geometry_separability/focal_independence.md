# Focal independence

This run estimates no focal and reads none.

| component | reference focal loaded | AnyCalib loaded | candidate focal grid | allowed |
| --- | --- | --- | --- | --- |
| reference focal / `gt_fx` | NO | NO | NO | NO |
| AnyCalib inference | NO | NO | NO | NO |
| GeoCalib inference | NO | NO | NO | NO |
| E2 / scene fusion | NO | NO | NO | NO |
| candidate focal sweep | NO | NO | NO | NO |
| `OTHER_CAMERA_ONLY_REFERENCE_3D` (historical calibration dependency) | NO | NO | NO | **YES** |

## How it is enforced

The eligibility source is `gigahands_demo_qc_v1.csv.gz`, which carries **no
focal column at all**. The frame grids that do carry `gt_fx`
(`gigahands_demo_cam_exp_0041_64frames_v1.csv.gz` and the camera benchmark) are
listed in `common.FOCAL_BEARING_MANIFESTS` and are never opened; `self_audit.py`
greps the source for those names and for the tokens `gt_fx`, `gt_fy`,
`anycalib`, `geocalib`, `focal_grid` and `candidate_focal`, and fails on a hit.

## The one legitimate dependency

`OTHER_CAMERA_ONLY_REFERENCE_3D` was triangulated using the dataset's camera
calibration, which contains intrinsics. That is a property of an artefact this
run *consumes*, not an estimation this run performs. The outputs here are
distances between normalised bone-proportion vectors; no focal value enters any
of them.

A historical reconstruction dependency is not current focal estimation, and the
report keeps the two separate.

## Also not used

WiLoR or any learned hand latent, scene features, lambda fitting, the
CAM-EXP-009.2 robust loss family, and the `FINAL_CONFIRMATORY_HOLDOUT`
(InterHand2.6M / HanCo), which remains unopened.
