# CAM-EXP-006 — solver protocol

Frozen in `experiments/manifests/cam_exp_006_reference_hand_spec_v1.json`
(`solver` block) before any CAM-006 solver output existed. Implemented in
`src/focal_profile_solver.py`, driven by `src/run_solver.py`.

## 1. The question the solver answers

Given a hand whose 3D shape we already know from *other* cameras, and given one
view's dataset-provided 2D observation of that same hand, how well can the
focal length of that one view be recovered?

This is deliberately an **oracle** question. At deployment time the reference 3D
does not exist — obtaining it is what calibration is for. The number this
protocol produces is therefore an **information ceiling**, not a method score.

## 2. PROFILED_PNP_FOCAL_DIAGNOSTIC

One scalar unknown, `f`, shared by `fx` and `fy`:

```
K(f) = [[f, 0, W/2],
        [0, f, H/2],
        [0, 0,   1]]
```

The principal point is **fixed at the image centre**. It is not read from the
provided calibration.

Per-hand rotation and translation are nuisance parameters. At every candidate
`f` they are re-fitted from scratch by `cv2.solvePnP`, and the objective is the
median reprojection error over the paired joints. This *profiles out* the pose,
turning a 7-parameter fit into a one-dimensional curve.

Grid: `gamma = f / max(W, H)`, 121 log-uniform points over `[0.25, 5.0]`. The
optimum is the grid argmin, refined by a parabolic fit on the three surrounding
points in log-gamma.

Combination across frames: the mean, in log space, of the per-frame objectives,
then a single argmin. Frames are combined only within one `(sequence, camera)`
view; they are never pooled across sequences.

### 2.1 PnP back-end and the planar fallback

`SOLVEPNP_SQPNP` is the frozen primary back-end. It refuses point sets whose
coordinate variance is degenerate, which the `PLANARIZED` control produces *by
construction*. A planar-capable fallback chain (`IPPE`, then `ITERATIVE`) is
therefore tried in turn.

This matters for honesty: without the fallback the planar control would fail
for an *implementation* reason rather than a geometric one, and would falsely
appear to support the hypothesis that the recovered signal is geometric.

## 3. Identifiability flags

A minimum that exists is not the same as a focal that is determined. Two flags
are recorded per solve:

| flag | meaning |
| --- | --- |
| `FLAT_PROFILE` | the objective at both grid ends is within 10 % of the minimum — the data barely constrains `f` |
| `BOUNDARY_SOLUTION` | the refined optimum landed on the first or last grid point — the true optimum is outside the searched range, or does not exist |

A result carrying either flag is reported, never silently dropped.

## 4. What the solver is not allowed to see

| quantity | used as a solver input? |
| --- | --- |
| reference 3D from other cameras | **yes** |
| dataset-provided 2D of the camera under test | **yes** |
| image width and height | **yes** |
| GT focal length | no |
| GT extrinsics | no |
| GT distortion | no |
| provided principal point | no |
| AnyCalib / GeoCalib / E2 output | no |
| bilateral apparent-size ratio between the two hands | no — explicitly forbidden as a focal estimator |

The first file in this run that reads `GT_EFFECTIVE_FOCAL` is
`src/evaluate.py`, which runs after every solver output is frozen on disk.

## 5. Negative controls

Frozen in `cam_exp_006_controls_spec_v1.json`. Each destroys the hand's 3D
shape in a different way while leaving the solver, the frames and the 2D
identical.

| control | what it destroys | why it is informative |
| --- | --- | --- |
| `JOINT_PERMUTATION` | joint correspondence, hence all shape | if the focal still comes out, the signal is not shape-based |
| `WRONG_POSE` | the match between 3D pose and 2D, while **preserving hand size** | the sharpest control: it separates "the solver reads the pose" from "the solver reads a typical hand scale" |
| `PLANARIZED` | the depth extent that makes perspective observable | a planar target cannot separate focal from depth; the profile should degrade |

## 6. Reproduce

```
python src/freeze_spec.py            # write the frozen specs (do this first)
python src/build_frame_manifest.py   # nested N = 1,2,4,8,16 grid
python src/build_reference_3d.py     # OTHER_CAMERA_ONLY_REFERENCE_3D cache
python src/run_solver.py             # all views x 4 conditions x 5 frame counts
python src/evaluate.py               # first read of GT focal
python src/figures.py
```
