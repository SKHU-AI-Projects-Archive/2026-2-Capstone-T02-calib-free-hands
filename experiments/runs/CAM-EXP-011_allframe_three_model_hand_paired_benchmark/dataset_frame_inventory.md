# Dataset frame inventory

## Three counts that are not the same number

| quantity | meaning |
| --- | --- |
| `SEQUENCE_TOTAL_FRAMES` | frames that exist in the video |
| `SCENE_INPUT_FRAMES` | frames fed to the scene models (here: all of them) |
| `HAND_AVAILABLE_FRAMES` | frames where hand geometry is usable - a SUBSET |

Reporting a hand-frame count as though it were the sequence length would be
wrong in both directions, so the three are kept in separate columns everywhere.

## Measured inventory

| Sequence | Total frames/video | Usable cameras | Scene frames used | Hand-usable (L / R) |
| --- | ---: | ---: | ---: | ---: |
| Tea | 381 | 40 | 15225 | 5913 / 7092 |
| Boxing | 366 | 40 | 14658 | 6076 / 6750 |
| Plant | 174 | 40 | 6962 | 3105 / 3188 |
| Dog | 337 | 40 | 13493 | 3655 / 6081 |
| Instrument | 139 | 15 | 2085 | 1510 / 1332 |

**Total scene input frames: 52,423** across **175** sequence-camera videos.

## Why 52,423 and not 52,445

The historical expectation of 52,445 comes from (frames per video x cameras)
using one frame count per sequence. The videos are not all exactly equal:

| sequence | frame counts across its usable cameras |
| --- | --- |
| Tea | 15 cameras at 380, 25 at 381 |
| Boxing | 22 cameras at 366, 18 at 367 |
| Plant | mostly 174, some 175 |
| Dog | 336-338 |
| Instrument | all 139 |

Per sequence that gives Tea -15, Boxing -22, Plant +2, Dog +13, Instrument 0,
i.e. **-22** against the uniform-count estimate. The measured value is the
source of truth; the historical figure was a good approximation, not an error.

## Why 175 videos and not 200

The verified CAM-EXP-003 camera benchmark set is reused unchanged: 40 + 40 +
40 + 40 + 15. In `p52-instrument-0034`, 25 of the 40 benchmark cameras hold an
RGB recording that is **not the same segment** as the annotation and
calibration, so they cannot be scored against that reference.

The frozen manifest records the reason explicitly, and it is worth stating
precisely: the exclusion is an **RGB recording-segment mismatch**, *not* a hand
annotation quality problem. Those 25 cameras were not re-matched by filename to
recover them.

## Frame index convention

0-based video frame index, matching the GigaHands loader. CAM-EXP-001.2
verified that the 2D row index, the 3D row index and the RGB frame index are
the same 0-based index. Every usable video contributes frames `0 .. N-1` with
no gaps; `self_audit.py` checks this.

## Exact frames

`experiments/manifests/cam_exp_011_allframe_manifest_v1.csv.gz` has one row per
`(sequence, camera, frame)` for all 52,423 scene frames, with per-frame flags
for model success and hand availability. Any question of the form "which frames
exactly?" is answerable down to the frame index.
