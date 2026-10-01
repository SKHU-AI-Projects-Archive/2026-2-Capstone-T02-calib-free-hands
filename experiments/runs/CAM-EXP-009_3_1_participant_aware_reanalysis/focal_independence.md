# Focal independence

CAM-EXP-009.3.1 estimates no focal and reads none. It is a geometry-grouping
correction.

| component | used |
| --- | --- |
| dataset reference focal / `gt_fx`, `gt_fy` | **NO** |
| AnyCalib | **NO** |
| GeoCalib | **NO** |
| E2 | **NO** |
| candidate focal grid | **NO** |
| focal estimation of any kind | **NO** |
| scene + hand fusion | **NO** |
| focal error computation | **NO** |
| external final holdout (InterHand2.6M / HanCo) | **NOT OPENED** |

## How it is enforced

This run consumes only CAM-EXP-009.3's distance and template tables, which
contain bone-proportion vectors and distances — no focal column exists in any of
them. `src/self_audit.py` greps every source file for `gt_fx`, `gt_fy`,
`anycalib`, `geocalib`, `focal_grid` and `candidate_focal` and fails on a hit.

## The historical dependency, unchanged

`OTHER_CAMERA_ONLY_REFERENCE_3D` was triangulated in CAM-EXP-001.3 using the
dataset's camera calibration, which contains intrinsics. That is a property of an
artefact this run consumes, not an estimation this run performs — the same
distinction CAM-EXP-009.3 drew. A historical reconstruction dependency is not
current focal estimation.
