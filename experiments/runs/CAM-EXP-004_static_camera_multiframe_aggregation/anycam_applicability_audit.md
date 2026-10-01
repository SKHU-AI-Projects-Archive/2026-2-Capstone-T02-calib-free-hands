# AnyCam applicability audit (static camera)

CAM-EXP-003 recorded AnyCam as `OUT_OF_SCOPE_SINGLE_FRAME` because it is a video
method. CAM-EXP-004 *is* a video experiment, so the exclusion had to be
re-examined rather than inherited. The audit was done **before** any attempt to
install or run it, against the official paper and the official code.

## Sources consulted

| Source | What it is |
|---|---|
| `arxiv.org/abs/2503.23282` | AnyCam, CVPR 2025, official paper page |
| `github.com/Brummi/anycam` | official implementation, `main` |
| `anycam/models/anycam.py` | focal-length parameterisation |
| `anycam/scripts/fit_video.py` | the official inference path for a video |
| `anycam/trainer.py` → `induce_flow_dist` | how a focal hypothesis is scored |

## What the method actually is

The paper's own framing: *"Estimating camera motion and intrinsics from casual
videos …"*, *"a fast transformer model that directly estimates camera poses and
intrinsics from a dynamic video sequence"*, trained with *"pre-trained depth and
flow networks"* and an *"uncertainty-based loss"*, plus *"a lightweight
trajectory refinement step"*.

The README documents input as a video (`++input_path=/path/to/video.mp4`) or a
list of frames, and documents no static-camera mode and no way to supply known
intrinsics.

## How the focal length is chosen — the decisive detail

`anycam.py` does not regress a single focal. It uses a **candidate** grid
(`focal_parameterization = "candidates"`, `focal_num_candidates = 32`,
`focal_min = 0.1`, `focal_max = 4.0`) and produces a distribution over those
candidates. `fit_video.py` then selects one:

```python
proj_candidates = make_proj_from_focal_length(pose_result["focal_length_candidates"], w/h)
induced_flow, dist = induce_flow_dist(depths.unsqueeze(2), proj_candidates,
                                      pose_result["poses"].clone())
...
best_candidate = proj_labels.argmax()
```

`induce_flow_dist` unprojects each pixel with the candidate intrinsics, moves it
by the estimated **relative pose**, and reprojects it:

```python
unproj_pts, xy = unproject_points(depths, projs)
pts = rel_poses @ unproj_pts
proj_pts = projs @ pts[:, :3]
```

The code's own comment states the degenerate case outright:

```
# The last pose should be identity -> Induced flow should be zero
```

## Consequence for a static camera

If the camera does not move, the relative pose between frames is the identity.
Then `pts == unproj_pts`, and `projs @ unproj_pts` reprojects every pixel back to
exactly where it started — **for every focal candidate**, because the same `projs`
both unprojects and reprojects and cancels. The induced flow is identically zero
across the whole candidate grid, so the objective that picks the focal is
perfectly flat and carries **no information about the focal length at all**.

This is not an implementation gap that better engineering would close. It is the
standard structure-from-motion observability condition: a single viewpoint does
not constrain focal length by triangulation. Moving hands and objects in front of
a fixed camera do not fix it either — AnyCam's flow term is explicitly the
*camera-induced* flow, and its uncertainty formulation is designed to
**down-weight** independently moving content rather than calibrate from it.

Our deployment setting is exactly this degenerate case: monocular RGB video from
a camera that is static for the whole clip.

## Verdict

**`NOT_APPLICABLE_REQUIRES_CAMERA_MOTION`**

AnyCam is therefore **not run**, and no smoke test is performed
(the spec gates the smoke test on `APPLICABLE_STATIC_CAMERA`). This is recorded
as **`METHOD_ASSUMPTION_MISMATCH`**, not as a failure or a BLOCKED status:
AnyCam is a good method for the problem it targets, and our problem is a
different one. Running it anyway on static clips would produce numbers that
reflect only its learned prior over focal lengths, and reporting those as
"AnyCam's performance" would misrepresent the method.

## What would change this verdict

A GigaHands-like setting with a genuinely moving camera, or an AnyCam variant
that scores focal candidates on something other than camera-induced flow. If the
project ever adds handheld-capture data, the audit should be redone rather than
this conclusion carried over.
