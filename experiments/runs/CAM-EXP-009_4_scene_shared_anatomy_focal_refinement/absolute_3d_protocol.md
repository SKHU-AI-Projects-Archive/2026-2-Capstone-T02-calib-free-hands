# Absolute 3D protocol

## Why this matters more than the focal number

The end goal is robot imitation, so the question is not only whether the focal
estimate improves but whether the **hand's absolute 3D position** improves as a
result.

## Camera translation — the audited equation

From CAM-EXP-002's focal-usage audit and the demo pipeline:

```
tz = 2 f / (s * B)      =>      tz(f) = cam_t[2] * (f / f_pipeline)
```

so **only `tz` scales with the focal**; `tx` and `ty` do not. This was read out
of the code, not reconstructed from memory.

`f_pipeline = FOCAL_LENGTH / IMAGE_SIZE * max(W, H) = 1000/256 * 1280`.

## What is held fixed

The WiLoR pose, shape and 2D prediction are **identical across methods** — each
method only supplies a different focal, and `tz` is recomputed from it. The
reference 3D is never used to refit the pose.

## Reference

`OTHER_CAMERA_ONLY_REFERENCE_3D`: the target camera is excluded **before**
reconstruction, so its own annotation cannot shape the hand it is scored
against. `target_camera_in_reference_count` is re-audited and must be 0.

World reference points are mapped into the target camera frame with that
camera's **rotation and translation only** (`X_cam = R X_world + t`). The target
camera's focal is never used for the geometric transform.

## Metrics

| metric | meaning |
| --- | --- |
| `WRIST_ROOT_ERROR_MM` | predicted vs reference wrist, 3D Euclidean |
| `ROOT_DEPTH_ERROR_MM` | `|Z_pred - Z_ref|` |
| `ABSOLUTE_MPJPE_MM` | 21 joints in absolute camera coordinates |
| `ROOT_ALIGNED_MPJPE_MM` | translation removed — a **control** |

The root-aligned metric is the control: changing only the focal changes only the
camera translation, so root-aligned MPJPE should barely move. A median change
above 2 mm raises `DOWNSTREAM_IMPLEMENTATION_REVIEW`.

LEFT and RIGHT are reported separately as well as combined, so a gain on one
hand that is paid for by the other cannot hide in the average.
