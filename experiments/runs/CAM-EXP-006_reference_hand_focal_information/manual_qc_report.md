# CAM-EXP-006 — Manual hand-QC readiness gate

This closes the standing blocker `HAND_QC_MANUAL_VALIDATION_PENDING`. It is a
prerequisite for every numerical result in CAM-EXP-006: the reference-hand
diagnostic is only meaningful if the dataset-provided 2D observations really sit
on the hand they claim to, and if the other-camera-only reconstruction is
geometrically plausible.

## 1. What was reviewed

| item | value |
| --- | --- |
| cases sampled | 200 |
| sampling frame | observations on the frozen CAM-EXP-004.1 64-frame grid, on cameras marked `usable_for_camera_benchmark`, with `qc_status` in {`PASS_STRICT`, `PASS_SINGLE_HAND`}, `chosen = 1`, `zero_pattern = 0` |
| strata | sequence \| anatomical side \| existing QC label \| existing triangulation-reprojection bin \| temporal position bin |
| selection rule | round-robin over strata, preferring the least-used physical camera among the next six candidates |
| sequences covered | 5 (tea 46, boxing 38, plant 45, dog 42, instrument 29) |
| distinct physical cameras covered | 39 |
| seed | 20260926 |
| manifest | `experiments/manifests/cam_exp_006_manual_hand_qc_v1.csv` |
| audit images | `manual_qc/panels/C000.png` … `C199.png`, contact sheets `manual_qc/contact_sheets/sheet_00.png` … `sheet_16.png` |

**Target independence.** No calibration result, no focal error and no solver
output took any part in choosing these cases. The strata use only dataset
metadata and the *existing* CAM-EXP-001.3 reconstruction-quality bins.

## 2. How it was reviewed

Every one of the 17 contact sheets was opened and viewed (12 cases per sheet,
each cell 640×400). Seven cases whose downscaled cell looked ambiguous were
re-opened at full panel resolution (1280×720) before being labelled. The
reviewer is recorded in the manifest as `claude_opus_5_visual_review`, with the
method in `review_method`; no case was labelled from numbers alone, and no label
was changed to make a downstream result come out a particular way.

Each panel shows, on the real RGB frame:

* **yellow** — the dataset-provided 2D observation;
* **magenta** — the leave-one-camera-out reconstruction (this camera excluded)
  reprojected into this camera using the provided camera parameters.

The magenta overlay is **AUDIT VISUALISATION ONLY**. It uses provided
calibration and is never an input to the CAM-006 focal solver.

## 3. Results

| metric | count | share of 200 |
| --- | ---: | ---: |
| cases reviewed | 200 | 100.0 % |
| `RGB_MATCH` | 200 | 100.0 % |
| `2D_ON_CORRECT_HAND` | 200 | 100.0 % |
| `HAND_SIDE_CORRECT` | 193 | 96.5 % |
| hand side `unverifiable_single_view` | 7 | 3.5 % |
| `MAJOR_2D_FAILURE` | 0 | 0.0 % |
| `MAJOR_OCCLUSION` | 7 | 3.5 % |
| `LOCO_3D_REPROJECTION_PLAUSIBLE` | 200 | 100.0 % |
| `NOT_REVIEWABLE` | 0 | 0.0 % |
| leave-one-camera-out reconstruction succeeded | 200 | 100.0 % |

Reprojection of the other-camera-only reconstruction into the held-out camera:
median **7.21 px**, p90 **13.03 px** (1280×720 frames).

Per-sequence and per-camera failure counts are in
`tables/manual_qc_breakdown.csv`: **zero** major 2D failures in any of the 5
sequences and in any of the 39 physical cameras, so there is no sequence-
localised and no camera-localised failure mode in this sample.

### 3.1 Cases opened at full resolution

| case | why it looked ambiguous | finding |
| --- | --- | --- |
| C006 | skeleton appeared to float off the hand | left hand heavily occluded behind the teapot; overlay correct |
| C079 | overlay sat over the toy | left hand resting at the toy's head; overlay correct |
| C080 | overlay at the frame edge near the LED strip | two hands at frame bottom, overlay on the correct one |
| C120 | overlay on the teapot body, visible hand elsewhere | a second hand grips the teapot from below-left; overlay correct |
| C123 | overlay above the gripping hand | right hand pinching the lid; small fingertip offset only |
| C140 | overlay on an unexpected upper-right hand | a genuine second hand is present; overlay correct |
| C185 | overlay where no hand is visible | hand occluded behind the toy's head; position geometrically consistent with the other cameras |

Every apparent anomaly resolved as **occlusion or a genuine second hand**, not
as an identity or association error.

### 3.2 On the 7 `unverifiable_single_view` labels

For these cases the hand is substantially hidden by the manipulated object, so
anatomical side cannot be confirmed from this single view alone. They are
recorded as unverifiable rather than as correct. This is a limit of single-view
review, not evidence of a labelling error.

## 4. Verdict

Pre-registered readiness conditions and their outcomes:

| condition | threshold | observed | met |
| --- | --- | --- | --- |
| catastrophic identity / association error rate | < 5 % | 0 / 200 = 0.0 % | yes |
| no systematic per-camera failure | — | 0 failures across 39 cameras | yes |
| no systematic per-sequence failure | — | 0 failures across 5 sequences | yes |
| leave-one-camera-out geometry visually plausible | — | 200 / 200 plausible | yes |

```
MANUAL_QC_READINESS_GATE = PASS
```

CAM-EXP-006 may proceed to the reference-3D build and the focal-profile solver.

## 5. What this verdict is not

* **It is not a dataset accuracy figure.** "0 / 200 catastrophic failures" is a
  readiness gate on a 200-case stratified audit sample, not a measured accuracy
  of the GigaHands annotations, and it must never be reported as one.
* **It does not validate the 2D annotations as ground truth.** The released 2D
  remain *dataset-provided 2D observations* throughout.
* **It does not validate the reconstruction as physical ground truth.** The
  leave-one-camera-out 3D remains `OTHER_CAMERA_ONLY_REFERENCE_3D`; the 7.21 px
  median reprojection is an internal consistency figure computed with provided
  camera parameters, not an external accuracy measurement.
* **It is a single-reviewer visual audit.** There is no second reviewer and no
  inter-rater agreement figure. Registered as an open issue.

## 6. Open issues raised here

| id | issue |
| --- | --- |
| `QC_SINGLE_REVIEWER` | one reviewer, no inter-rater agreement measured |
| `QC_SIDE_UNVERIFIABLE_UNDER_OCCLUSION` | 3.5 % of cases cannot have their anatomical side confirmed from a single view |
| `QC_SAMPLE_IS_PASS_ONLY` | the audit sample is drawn from observations that already passed the existing QC filter, so it cannot estimate the failure rate of the observations that filter rejects |

## 7. Reproduce

```
python src/build_manual_qc.py            # renders panels (decodes video)
python src/assemble_manual_qc.py         # manifest + contact sheets, no decode
```
