# Source provenance

## Reused from CAM-EXP-009.4 (commit `1c36d92`)

| artefact | role |
| --- | --- |
| `cache/wilor/*.npz` | the monocular hand predictions - **not re-inferred** |
| `cam_exp_0094_hand_split_v1.csv.gz` | the frame set |
| `cam_exp_0094_units_v1.csv.gz` | unit metadata and participant labels |
| `cam_exp_0094_outer_camera_folds_v1.csv` | physical-camera folds |
| `tables/common_set.csv` | the `PRIMARY_COMMON_SET` population |
| `results/raw/absolute3d_frame_results.csv.gz` | the oracle-focal row this run reproduces |
| CAM-EXP-001.3 `src/loco.py` | `OTHER_CAMERA_ONLY_REFERENCE_3D`, reused unchanged |

Path, size, SHA-256 and row counts for each are in
`tables/source_artifacts.csv`.

**No new WiLoR inference was run.** Every method and oracle consumes the same
cached network output, so any difference between them is attributable to the
translation, not to the network.

## Population

The CAM-EXP-009.4 `PRIMARY_COMMON_SET` is reused as-is. No new
performance-based filtering was applied: a unit is not dropped for having a
large error, an odd scale ratio or an unhelpful correction. The only additional
requirement is enough FIT and EVAL frames to fit and then test a correction, and
that is a data-availability rule fixed before any result.

Where the actual artefacts disagree with a remembered number, the artefact wins
and the measured value is reported.

## Reference 3D

`OTHER_CAMERA_ONLY_REFERENCE_3D`: the target camera is excluded **before**
reconstruction, so its own annotation cannot shape the hand it is scored
against. `target_camera_in_reference_count` must be 0 and is re-audited.

World reference points are mapped into the target camera frame using that
camera's **rotation and translation only**. The target camera's focal is never
used for the geometric transform.

## Camera translation equation

Re-read from the code rather than recalled - CAM-EXP-002's focal-usage audit and
CAM-EXP-009.4's downstream evaluation both use:

```
tz = 2 f / (s * B)      =>      tz(f) = cam_t[2] * (f / f_pipeline)
```

so only `tz` scales with the focal; `tx` and `ty` do not.
`f_pipeline = FOCAL_LENGTH / IMAGE_SIZE * max(W, H) = 1000/256 * 1280`.

## Reproduction gate

`src/reproduce_cam0094_oracle.py` recomputes CAM-EXP-009.4's `REF_FOCAL_ORACLE`
wrist/root and root-depth medians from this run's own pipeline and compares them
at a 1 mm tolerance. If they do not match, the decomposition is resting on a
different pipeline and the run stops.
