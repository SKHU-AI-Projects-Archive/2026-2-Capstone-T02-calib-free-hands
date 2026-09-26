# CAM-EXP-008 — what counts as one sequence

## 1. Deployment definition

One already-recorded video file = one calibration unit, under these conditions:

* one physical camera, one continuous recording
* no cut or edit, constant resolution, no crop-mode change, no digital zoom
* camera intrinsics shared across the whole recording
* tripod or fixed mount; small vibration allowed, no front-to-side viewpoint
  change
* processing is **offline**: the whole video is available before any estimate
  is produced

What differs between factories, and is therefore never assumed constant: hand
size, worker, worktable, background, camera, distance, scene layout.

## 2. Development-dataset mapping

On GigaHands one `(sequence, camera)` pair is used as one offline calibration
unit, called the `SEQUENCE_CAMERA_UNIT`. It corresponds to the "static view"
unit of earlier runs, renamed here because the deployment meaning differs:
earlier runs treated it as a set of frames from a fixed camera, this run treats
it as one recorded video to be calibrated as a whole.

## 3. Shared and free parameters

| quantity | treatment |
| --- | --- |
| K, and specifically the scalar focal f | **shared** across the whole sequence |
| hand rotation and translation per frame | **free**, re-fitted at every candidate focal |
| small tripod vibration | absorbed by the per-frame hand pose, NOT modelled as an intrinsics variation |

## 4. Why focal only

The primary unknown is the focal. The fy/fx ratio, principal point and radial
distortion are taken from the scene estimator at sequence level and held
**identical** in S0 and S1, so they cannot explain any difference between the
two. Opening several parameters at once would blur the question this run
exists to answer.

## 5. Offline, not realtime

Nothing here is a causal or streaming update. The frame-count experiment is
**offline evidence accumulation**: how much of an already-recorded video needs
to be looked at. It must not be described as realtime or online refinement.
