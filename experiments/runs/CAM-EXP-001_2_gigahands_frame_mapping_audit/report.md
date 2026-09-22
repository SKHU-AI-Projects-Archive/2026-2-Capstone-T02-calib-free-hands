# CAM-EXP-001.2 — GigaHands annotation / frame-mapping audit

Run date 2026-09-22 · branch `kjh` · base commit `f0f2f98` · no GPU.
Preflight and versions in `environment.json`, parameters in `config.json`,
run log in `logs/run.log`.

## 1. Purpose

Decide whether the GigaHands bad views found in CAM-EXP-001 / 001.1 come from

* **A** — the dataset's own annotation, or
* **B** — our loader mis-associating RGB frames, 2D rows, 3D rows,
  `chosen_frames` and cameras.

No calibration method is developed here. This is purely data and loader
verification.

## 2. Why this experiment was necessary

CAM-EXP-001.1 concluded that bad views were annotation-side, but it had not
verified the index semantics themselves. Its conclusion would collapse if, for
example, 3D rows were indexed by position within `chosen_frames` rather than by
frame id: every "annotation on the wrong hand" case could then be a
self-inflicted off-by-N. That possibility is now tested directly rather than
assumed away.

## 3. Official GigaHands frame / chosen_frames semantics

Source consulted: **`render_mesh_video.py`** in
[brown-ivl/GigaHands](https://github.com/brown-ivl/GigaHands), function
**`hand_pose_loader(keypoints3d_path)`**:

```python
def hand_pose_loader(keypoints3d_path):
    """Find frame indices for which both left and right hand pose data are present."""
    chosen_frames_right = set(json.load(open(.../"chosen_frames_right.json")))
    chosen_frames_left  = set(json.load(open(.../"chosen_frames_left.json")))
    chosen_hand_union_frames     = list(chosen_frames_right | chosen_frames_left)
    chosen_hand_intersect_frames = list(chosen_frames_right & chosen_frames_left)
    return chosen_hand_union_frames, chosen_hand_intersect_frames
```

and, crucially, how those frames are then used:

```python
video_indices_in_hand_iou_indices = np.asarray(
    [chosen_hand_union_frames.index(f) for f in chosen_video_frames])
mano_params_right, mano_params_left = hand_mano_loader(
    mano_main_path, video_indices_in_hand_iou_indices)
```

`chosen_hand_union_frames.index(f)` converts a **frame id** into a **position
in the union list**, and that position indexes the MANO parameters. So the
dataset uses *two different index spaces*, which is the single most important
fact of this audit.

The docstring also settles the meaning of the files: chosen_frames are the
frame indices *"for which hand pose data are present"*.

Per the project page and paper, the 2D keypoints are produced by **HaMeR**,
with **ViTPose** used in the hand-detection stage; left/right is therefore
assigned by a **per-view 2D detector**, which matters in §10.

Answers to the three required questions:

1. **What are chosen_frames?** The set of frame ids for which that hand has a
   valid 3D pose. Verified on the data: `keypoints_3d/<hand>.jsonl` rows that
   are *not* in `chosen_frames` are **100 % all-zero**, and rows that are in it
   are **100 % non-zero** (`tables/chosen_frames_semantics.csv`).
2. **Why do left and right differ?** They are per-hand validity lists. In
   `p36-tea-0010` left has 359 and right 344 chosen frames; the other four
   sequences happen to have both hands valid on every frame. The official
   loader is built around exactly this asymmetry (union vs intersection).
3. **How should our loader apply them?** As a membership filter on the frame id
   for that specific hand, before indexing `keypoints_3d` — which is what
   `experiments/src/datasets/gigahands.py` already did. Using an unchosen row
   would silently feed an all-zero pose into the projection.

## 4. Dataset structure audit

`results/raw/frame_structure_inventory.csv` (200 sequence × camera rows).

| Property | Result |
|---|---|
| `keypoints_2d` rows == RGB frames | **True for all 175 cameras whose annotated video is present** |
| timestamp lines == RGB frames | True for the same 175 |
| `keypoints_3d` rows == max(chosen)+1 | **True, all 5 sequences, both hands** |
| `params` entries == \|union\| | **True** |
| `repro_2d_vid` / `repro_3d_vid` / `mano_vid` frames == \|union\| | **True, all 5 sequences** |

Per sequence (first camera):

| Sequence | RGB | timestamps | 2D rows | 3D rows | \|union\| | \|intersect\| |
|---|---|---|---|---|---|---|
| p36-tea-0010 | 381 | 381 | 381 | 375 | 359 | 344 |
| p41-boxing-0021 | 367 | 367 | 367 | 361 | 361 | 361 |
| p41-plant-0004 | 174 | 174 | 174 | 169 | 169 | 169 |
| p44-dog-0004 | 337 | 337 | 337 | 331 | 331 | 331 |
| p52-instrument-0034 | (see below) | — | 139 | 133 | 133 | 133 |

The 3D track always stops ~5-6 frames before the video ends; those trailing
frames simply have no 3D pose.

**A real data caveat, found here:** in `p52-instrument-0034`, **25 of 40
cameras** ship an `rgb_vid/<cam>/*.mp4` whose filename timestamp differs from
the annotated take by **+13.4 s or +154 s**, and whose length differs (166
frames vs 139 annotated rows). The annotated video for those cameras is *not in
the download*. Our loader builds the video path from the exact 2D filename stem
and therefore attaches no image rather than pairing annotations with a
different segment — the safe behaviour, but it means no overlay is possible for
those 25 cameras. The other four sequences are 40/40 exact matches.

## 5. Timestamp audit

`results/raw/timestamp_audit.csv`. Each `rgb_vid/<cam>/<cam>_<ts>.txt` has one
line per video frame:

```
frame_1727030430679748_000000000000
frame_1727030430711759_000000000001
```

i.e. `frame_<microseconds>_<12-digit frame index>`. Verified:

* the embedded frame index is **0-based and contiguous** for every camera checked;
* line count == video frame count == 2D row count;
* implied frame rate **30.8–31.2 fps**, differing slightly per camera;
* the timestamp in the *filename* equals the first frame timestamp to within
  3–37 ms, so it identifies the take, not an offset to apply.

**Timestamps are not needed to map 2D/3D/RGB within a camera** — the shared
0-based index already does that. Their necessary use is the one above: checking
that the mp4 in `rgb_vid/` really belongs to the annotated take. Cross-camera
temporal alignment would need them, but that is not required here because all
cameras share the same frame numbering for a take.

## 6. Frame mapping result

`results/raw/frame_mapping_table.csv`, and `figures/frame_mapping_diagram.png`.

```
RGB frame i  ==  timestamp line i  ==  keypoints_2d row i  ==  keypoints_3d row i
                                   chosen_frames = {frame ids with valid 3D}
params / repro_2d_vid / repro_3d_vid / mano_vid  =  sorted(union).index(frame)
```

`mapping_source` is `DIRECT_INDEX` for every frame inside the 3D range and
`OUT_OF_3D_RANGE` for the trailing frames; there is no `TIMESTAMP_MATCH` or
`INFERRED` row, because none was needed.

## 7. chosen_frames result

See §3. Both the official docstring and the data agree. Our CAM-EXP-001 loader
applied them correctly: the regression test `loader emits only chosen frames`
reports **0 violations**.

## 8. Official repro video comparison

`figures/official_repro_comparison/`, index in
`results/summary/official_repro_verdict.csv`.

`repro_2d_vid` and `repro_3d_vid` are **7 × 6 montages of all 40 annotated
cameras**, one tile per camera, with the official 2D skeletons and boxes drawn.
Their frame count equals \|union\|, so a frame id must be converted with
`sorted(union).index(frame)` before seeking — using the raw frame id would
compare different moments.

The tile order is undocumented, so rather than assuming it, each tile was
matched to the real RGB frame by normalised correlation. In **10 of 10** cases
the best-matching tile index equalled the position of that camera in the sorted
camera list (scores 0.62–0.77), so tile order = sorted camera order.

**Answer to the key question:** the official repro videos show the **same
annotation we show**. Where our overlay puts the LEFT annotation on the right
hand, the official montage tile shows the same skeleton in the same place. The
mismatch is therefore present in the published GigaHands annotation itself and
is not introduced by our loader, our camera convention, or our frame seeking.

## 9. Both-hand visualisation findings

All figures in this run draw **both hands at once** in four colours
(green = LEFT 2D, red = LEFT 3D, blue = RIGHT 2D, orange = RIGHT 3D). This is
what makes the failure legible: drawing one hand alone, as CAM-EXP-001.1 did,
shows a large error but cannot show *where the annotation actually went*.

In `figures/both_hands/suspected_swap_grid.png` the signature is unmistakable:
**green pairs with orange** on one hand and **blue pairs with red** on the
other — the LEFT annotation lies on the RIGHT reprojection and vice versa,
both skeletons sitting correctly on real hands.

## 10. Previous bad-case re-audit

`results/raw/previous_cases_reaudit.csv`, all 26 CAM-EXP-001.1 cases:

| old class | new class | n |
|---|---|---|
| good | GOOD_CONFIRMED | 10 |
| good | MEDIUM_CONFIRMED | 2 |
| bad | LIKELY_HAND_IDENTITY_ERROR | 10 |
| bad | UNRESOLVED | 2 |
| zero_sentinel | INVALID_OR_MISSING_2D | 2 |

**`mapping_changed` is 0 for every case.** No bad case was repaired by
re-indexing, because the indexing was already correct.

Full census over all five sequences (`results/raw/gigahands_demo_qc_census.csv.gz`,
5600 sequence × camera × frame × hand records, frame stride 20):

| status | n | share |
|---|---|---|
| good | 2386 | 42.6 % |
| invalid_2d_zero | 1333 | 23.8 % |
| medium | 1119 | 20.0 % |
| likely_hand_identity_error | 518 | 9.3 % |
| bad_unexplained | 204 | 3.6 % |
| not_chosen | 40 | 0.7 % |

Of the 722 bad records, **518 (71.7 %) are clean hand-identity errors** and 204
(28.3 %) remain unexplained. **0 %** are explained by our mapping.

The swap is **camera-dependent**: `tables/per_camera_hand_identity.csv` shows 8
sequence × camera combinations with a swap rate above 0.8 —
`brics-odroid-026_cam0/027_cam0/027_cam1/028_cam0/029_cam0/030_cam1` — clustered
in the high-numbered part of the rig, while most cameras never swap. That is
consistent with left/right being assigned by a **per-view detector** (ViTPose,
§3): from certain viewpoints the two hands are confused, and the error is then
baked into `keypoints_2d` for that camera.

## 11. Zero-pattern investigation

Verdict: **OBSERVED_INVALID_PATTERN** (not `DOCUMENTED_INVALID_SENTINEL`).

* Pattern: all 21 joints exactly `(0, 0)` with confidence `1.0`.
* Census: strictly **all-or-nothing per hand per view** — the regression test
  found 0 partial cases out of 76 affected views.
* Prevalence: 23.8 % of census records.
* Searches of the GigaHands repository README, the DeepWiki mirror and the
  paper did **not** surface any documentation of `(0,0)`, of a sentinel value,
  or of invalid-detection handling. No official statement was found, so this is
  **not** called an official sentinel.

What is certain is that such a record contains no annotation: the images show a
hand that is simply not annotated in that view. It is therefore separated as an
**INVALID candidate** for QC, not counted as a reprojection error.

## 12. Remaining mismatches

204 records (3.6 %) are `bad_unexplained`: a large error with no clean
hand-swap explanation. Inspection (`figures/both_hands/mapping_corrected_grid.png`,
ROW 4 of the main figure) shows a mixture of partial occlusion, one hand
annotated while the other is `(0,0)`, and cases where the annotation looks
displaced but not onto the other hand. These are handed to CAM-EXP-001.3.

## 13. Conclusions

Per the classification required for this experiment:

| Category | Count (census) | Basis |
|---|---|---|
| `OUR_MAPPING_ERROR` | **0** | index semantics verified against official code and data |
| `OUR_LOADER_ERROR` | **0** for indexing; **1 defect fixed** (see below) | regression tests |
| `INVALID_OR_MISSING_2D` | 1333 | all-zero annotation, all-or-nothing |
| `LIKELY_HAND_IDENTITY_ERROR` | 518 | annotation centroid on the other hand, camera-clustered |
| `LIKELY_DATASET_ANNOTATION_ERROR` | **not asserted** | withheld pending CAM-EXP-001.3 |
| `UNRESOLVED` | 204 | no clean explanation yet |

`LIKELY_DATASET_ANNOTATION_ERROR` is deliberately **not** used even though most
of its preconditions are met (frame mapping, chosen_frames, camera mapping and
official-repro comparison all check out, and our geometry is unchanged). The
missing piece is independent adjudication of 2D versus 3D: the official repro
shows the same annotation we do, but that only proves the *annotation* is what
we think it is, not that the annotation rather than the 3D is the wrong one.
That is exactly what CAM-EXP-001.3 triangulation will settle.

**One loader defect was found and fixed** — not an indexing bug, but the
acceptance of `(0,0)` rows as data. `experiments/src/datasets/gigahands.py` now
has `is_zero_2d()` and `drop_zero_2d=True`, with `drop_zero_2d=False`
reproducing the old behaviour. CAM-EXP-001's raw outputs were **not**
regenerated. Seven regression tests in `src/tests/test_gigahands_mapping.py`
all pass.

## 14. Limitations

- The census uses a frame stride of 20, not every frame.
- The 60 px hand-swap radius and the 10/50 px bands are diagnostic choices
  carried over from CAM-EXP-001.1; they are **not** the final quality gate.
- The swap criterion needs both hands annotated and chosen; views where the
  partner hand is `(0,0)` cannot be tested this way.
- Official confirmation of `(0,0)` semantics was not found; absence of
  documentation is not proof that none exists.
- The official-repro comparison covers 10 cases, all with an available video,
  and only `p36-tea-0010`.
- Tile order was verified empirically on 10 samples, not proven from source.
- 25 p52 cameras cannot be visually checked at all, because the annotated video
  is not in the download.

## 15. Next experiment

CAM-EXP-001.3: leave-one-camera-out triangulation from `keypoints_2d` alone,
compared against the provided `keypoints_3d`, to decide whether the 2D or the
3D is at fault in the swap and unresolved cases, and to fix the final quality
gate (`PASS_STRICT` / `PASS_SINGLE_HAND` / `REVIEW` / `EXCLUDE`).

---

## WHAT IS NOW CONFIRMED

1. **RGB frame i == timestamp line i == `keypoints_2d` row i == `keypoints_3d`
   row i.** One 0-based index, verified on 175 cameras across 5 sequences.
2. **`keypoints_3d` has `max(chosen)+1` rows**, and rows outside
   `chosen_frames` are 100 % all-zero — they are placeholders, never data.
3. **`params` and the official `repro_*_vid` / `mano_vid` montages use a
   different index space**: the position within `sorted(chosen_left | chosen_right)`.
   This follows the official `render_mesh_video.py` and matches the frame counts
   exactly on all 5 sequences.
4. **`chosen_frames_<hand>` = frame ids with a valid 3D pose for that hand**,
   per the official docstring and confirmed by the zero/non-zero split.
5. **Our CAM-EXP-001 indexing was correct.** 0 of 26 previous cases changed
   class because of re-indexing; 0 % of the census is explained by our mapping.
6. **The official repro videos show the same annotation we do**, including in
   the swapped cases.
7. **The hand-identity swap is camera-clustered** (8 sequence × camera pairs
   above 0.8 swap rate, mostly cameras 026–030), consistent with per-view
   left/right assignment by a 2D detector.
8. **The `(0,0)` pattern is strictly all-or-nothing per hand per view** and
   carries confidence 1.0.

## WHAT WAS WRONG IN OUR PREVIOUS ASSUMPTION

1. CAM-EXP-001.1 called `(0,0)` a **"sentinel"** and wrote that GigaHands
   "writes (0,0) … as its undetected-hand sentinel". No official documentation
   for that was found. The correct status is **OBSERVED_INVALID_PATTERN**; the
   earlier wording implied a documented convention that has not been verified.
2. CAM-EXP-001 (and 001.1) treated `(0,0)` rows as valid measurements because
   their confidence is 1.0. That inflated the error distribution and produced
   the ~690 px mode. The loader now excludes them by default.
3. CAM-EXP-001.1 reported the "other hand" match with a **per-joint** distance
   (median 15 px vs 224 px on centroids, but only 9.1 % under 20 px per joint).
   Per-joint distance understates hand swaps because left and right skeletons
   use mirrored joint ordering; centroid distance is the right instrument.
4. We had assumed `rgb_vid/<cam>/` always contains the annotated take. It does
   not: 25 of 40 cameras in `p52-instrument-0034` hold a different segment.
5. Not previously realised: **GigaHands uses two index spaces.** Anything read
   from `params` or the repro videos must be converted through the union
   position. Our code never touched those, so nothing was wrong — but the next
   experiment that uses MANO parameters would have been wrong without this.

## WHAT IS STILL UNRESOLVED

1. Whether the **2D annotation or the 3D pose** is the incorrect side in the
   518 hand-identity cases. Both land on real hands; only independent
   triangulation can adjudicate.
2. The **204 `bad_unexplained`** records.
3. Whether `(0,0)` is an intended GigaHands convention.
4. Whether the swap rates hold outside the 5 demo sequences and the stride-20
   sample.
5. The 25 p52 cameras with no annotated video — numerically auditable, not
   visually.
6. The `medium` band (1119 records, 20 %) — ordinary noise or something systematic.

## DECISION FOR CAM-EXP-001.3

**Proceed.** The data foundation is sound: index semantics are verified against
official code, our loader reproduces them, and the remaining problems are
localised and enumerated. CAM-EXP-001.3 should:

1. Triangulate from `keypoints_2d` with leave-one-camera-out and compare to the
   provided `keypoints_3d`, to decide 2D-vs-3D fault.
2. Start from the 518 swap + 204 unresolved records listed in
   `results/raw/gigahands_demo_qc_census.csv.gz` (filter `status`).
3. Exclude `invalid_2d_zero` and `not_chosen` records from any metric.
4. Use `sorted(union).index(frame)` for anything read from `params` or the
   repro videos.
5. Fix the final quality gate, which this run deliberately did not.

## Human-facing figures, in order

1. `figures/CAM_EXP_001_2_MAIN_EXPLANATION.png` — the whole result in one sheet.
2. `figures/both_hands/suspected_swap_grid.png` — green+orange and blue+red pairing.
3. `figures/frame_mapping_diagram.png` — the two index spaces.
4. `figures/official_repro_comparison/` — official vs our overlay, 5 panels each.
5. `figures/both_hands/{good,bad,sentinel,mapping_corrected}_bimanual_grid.png`.
6. `figures/mapping_examples/same_frame_multiview.png` — the swap follows the camera.

## Reproduce

```bash
PYTHONPATH=<repo root> python src/run_all.py
PYTHONPATH=<repo root> python src/tests/test_gigahands_mapping.py
```
