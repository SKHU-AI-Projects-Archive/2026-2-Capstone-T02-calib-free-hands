# CAM-EXP-006 — provenance of the reference 3D

## 1. Name and status

The 3D hand used throughout CAM-EXP-006 is called
**`OTHER_CAMERA_ONLY_REFERENCE_3D`**.

It **is**: a hand triangulated from the dataset-provided 2D observations of
every *other* calibrated camera in the same sequence, with the camera under
test removed from the observation set before reconstruction.

It **is not**: ground truth 3D, independent physical ground truth, or a
dataset-provided 3D annotation. None of those terms may be used for it.

## 2. How it is built

| item | value |
| --- | --- |
| implementation | `CAM-EXP-001_3_gigahands_multiview_triangulation/src/loco.py :: reconstruct()`, reused **unchanged** |
| inlier threshold | 8.0 px |
| minimum inlier cameras | 4 |
| minimum paired joints required by CAM-006 | 12 of 21 |
| confidence threshold on the 2D | 0.5 |
| excluded 2D pattern | the all-`(0,0)` "not annotated" pattern |
| cache | `cache/reference_3d_v1.npz` |
| audit | `results/raw/circularity_audit.csv` (one row per case) |

The 2D inputs are **dataset-provided 2D observations** throughout. They are
never called GT2D.

## 3. Coverage

| item | value |
| --- | --- |
| views | 175 `(sequence, camera)` pairs |
| frames per view | 16, from the frozen nested grid |
| hands per frame | 2 (left, right) |
| cases attempted | 5600 |
| cases usable by the solver | 4271 (76.3 %) |
| median paired joints when usable | 21 of 21 |

The 23.7 % that are unusable are cases where one hand is out of frame, not
annotated in this camera, or reconstructed from too few cameras. They are
dropped by a rule fixed in the spec — a joint-count threshold — not by looking
at any focal result.

## 4. Non-circularity

| check | result |
| --- | --- |
| camera under test excluded before reconstruction | yes, by construction in `reconstruct(exclude_camera=...)` |
| camera under test ever appears among the reconstruction's inlier cameras | **0 of 5600** |
| dataset-provided 3D read anywhere in the build | no |
| GT focal read anywhere in the build | no |
| any earlier experiment's verdict used to pick or reject a camera | no |

```
NON_CIRCULARITY = PASS
```

This is enforced by the contract at the top of `loco.py` and re-verified per
case in `circularity_audit.csv`, whose columns `camera_under_test_excluded`,
`camera_under_test_in_inliers`, `gt_focal_used`,
`gt_extrinsics_used_in_solver_input` and `provided_3d_used` record the check
for every single case rather than once for the run.

## 5. The limit that remains

The reference 3D is built with the **provided camera parameters** of the other
cameras. It is therefore internally consistent with the rig's calibration, not
independent of it. Two consequences, both stated wherever the numbers appear:

1. Any systematic error shared by the whole rig's calibration is inherited by
   the reference hand and cannot be detected from inside this dataset.
2. The reprojection figures reported for the reference hand are **internal
   consistency** figures, not external accuracy measurements.

Related: the `SINGLE_FOCAL_RIG_CONFOUND` recorded in CAM-EXP-005 — the GT
reference focal varies by only 1.90 % (CV) across the 175 views — limits what
any per-view focal conclusion on GigaHands can establish. That is why the
`CONSTANT_RIG_FOCAL_ORACLE` is pre-registered as the control any result must
beat.
