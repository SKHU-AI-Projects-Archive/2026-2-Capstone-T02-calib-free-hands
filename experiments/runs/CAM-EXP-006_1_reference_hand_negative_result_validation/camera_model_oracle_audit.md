# CAM-EXP-006.1 — camera-model oracle audit

## 1. What each condition consumes

From `tables/oracle_information_usage.csv`:

| condition | focal magnitude | focal ratio | principal point | distortion | extrinsics | deployable |
| --- | :-: | :-: | :-: | :-: | :-: | :-: |
| R0 | no | no | no | no | no | yes |
| R1 | no | no | no | no | no | yes |
| R2 | no | no | **yes** | no | no | no |
| R3 | no | no | no | **yes** | no | no |
| R4 | no | no | **yes** | **yes** | no | no |
| R5 | no | **yes** | **yes** | **yes** | no | no |

`uses_gt_focal_magnitude = 0` in every row. That is the property that keeps
these diagnostics non-circular: the solver is told the *shape* of the camera
model but never the answer it is searching for.

## 2. The fy/fx ratio in R5

`r = provided_fy / provided_fx`, median 0.99867 across the rig. The solver
searches a single unknown `fx` and sets `fy = r * fx`. Neither provided
magnitude reaches the solver; only their quotient does.

This is recorded as a deliberate, bounded oracle leak: knowing the aspect ratio
of a camera is far weaker information than knowing its focal length, and on
this rig the ratio is within 0.13 % of unity, so R5 and R4 are nearly identical
(49.96 % against 49.92 %).

## 3. Result

| condition | N=16 median error | flat | identifiable |
| --- | ---: | ---: | ---: |
| R1 | 203.13 % | 88.0 % | 2.9 % |
| R2 (+ principal point) | 204.75 % | 88.0 % | 2.9 % |
| R3 (+ distortion) | **47.53 %** | 77.1 % | 13.7 % |
| R4 (+ both) | 49.92 % | 76.6 % | 14.3 % |
| R5 (+ ratio) | 49.96 % | 77.1 % | 13.7 % |

**Under this GigaHands diagnostic, supplying the provided principal point or
the fy/fx ratio produced little additional improvement beyond the
distortion-aware condition.** Supplying the distortion coefficients is what
moved the number: from 203.13 % (R1) to 47.53 % (R3).

These conditions differ in what the solver is **given**. They are not an
additive decomposition of the error, and this run does not license statements
of the form "distortion accounted for X % of the failure". The ordering is also
specific to this rig and regime, not a general claim that principal-point error
is harmless.

The result is consistent with the rig statistics: the provided principal point
sits a median 23.6 px from the image centre in y and 0.02 px in x, while the
median `k1` is -0.392, which displaces points by tens of pixels toward the frame
edge.

## 4. Why distortion matters so much more here than in synthesis

In the strong-perspective synthetic condition, ignoring the same distortion
costs only 5.18 %. On real data it costs the difference between 203 % and
47.5 %.

This is consistent with an interaction rather than a contradiction. When the
objective has a sharp minimum, a systematic reprojection bias shifts that
minimum slightly. When the objective is already nearly flat — the regime the
real hands are in — the same systematic bias has far less to push against, so
it weighs much more heavily on where the argmin falls.

Stated carefully: distortion and profile flatness are **not** separable
additive contributions here. The flatness is present before distortion is
supplied, and the two act together.

That is why distortion must be modelled before anyone claims to have measured a
geometric information limit, and it is the single most important correction
this run makes to CAM-EXP-006.

## 5. What it does not rescue

Supplying the entire camera model leaves:

* 49.96 % median focal error against a constant's 0.85 %
* 77.1 % of profiles flat
* 13.7 % of views cleanly identifiable
* Spearman tracking -0.025, CI [-0.168, +0.118]

So supplying the camera model accounts for much of the *magnitude* of
CAM-006's headline number and none of its *direction*.

All six conditions here are pre-registered in
`cam_exp_0061_real_conditions_v1.json`; see
[`analysis_provenance.md`](analysis_provenance.md).
