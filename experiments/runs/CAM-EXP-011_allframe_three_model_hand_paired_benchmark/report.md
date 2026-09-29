# CAM-EXP-011 — All-Frame Three-Model Scene-vs-Hand Paired Focal Benchmark

```
RESULT  NO_MEASURABLE_CHANGE_FROM_HAND_STRUCTURE

  AnyCalib             16.02 %  ->  16.02 %   (+0.00 pp)
  GeoCalib             21.35 %  ->  21.35 %   (+0.00 pp)
  Perspective Fields   19.50 %  ->  19.50 %   (+0.00 pp)

  median relative focal error, 151 paired videos, every usable RGB frame
```

**One sentence.** Every usable RGB frame of all five sequences was processed by
all three scene estimators (52,423 frames each, zero failures), the same frozen
generic hand-structure correction was attached to each, and on the identical
paired video set the median focal error **did not change for any of the three**.

## 1. 쉬운 설명

5개 sequence의 **사용 가능한 모든 RGB frame**(52,423장)을 세 camera model로
다시 평가하고, 각 model에 **동일한** 손 구조 correction을 붙여 같은 영상에서
비교했습니다.

결과는 **세 모델 모두 변화 없음**입니다.

왜 그런지도 확인했습니다. 영상 한 편의 모든 frame을 쓰면 scene 추정이 매우
안정적이 되어(AnyCalib의 frame 간 흩어짐이 약 3.9 %), fusion에서 scene 항이
매우 가파르게 됩니다. 손 항이 후보 focal을 한 칸 움직이려면 λ ≈ 13.4가 필요한데,
사전에 고정한 λ 범위는 최대 4였습니다. 그리고 **TRAIN 데이터에서 λ를 고르게
했더니 15개 fold 중 11개에서 λ = 0을 선택**했습니다. 즉 절차 스스로 "손 항이
도움이 안 된다"고 판단하고 꺼버린 것입니다.

중요한 점: 손 점수가 평평해서가 아닙니다. 손 점수는 후보 구간에서 약 2배
변합니다. scene 항이 그보다 훨씬 가파른 것입니다.

## 2. Q1–Q9 — the dataset, measured not assumed

| Sequence | Total frames/video | Usable cameras | Scene frames used | WiLoR hand-usable (L / R) |
| --- | ---: | ---: | ---: | ---: |
| Tea | 381 | 40 | 15,225 | — |
| Boxing | 366 | 40 | 14,658 | — |
| Plant | 174 | 40 | 6,962 | — |
| Dog | 337 | 40 | 13,493 | — |
| Instrument | 139 | 15 | 2,085 | — |
| **TOTAL** | | **175** | **52,423** | **35,854 / 36,240** |

- **Q1** All five sequences included.
- **Q2/Q3** Frame counts and camera counts measured from the videos.
- **Q4/Q5** The 25 `p52-instrument-0034` cameras are excluded because their RGB
  recording is **not the same segment** as the annotation and calibration — an
  **RGB recording-segment mismatch**, explicitly *not* a hand-annotation quality
  problem. The frozen benchmark manifest records this, and they were not
  re-matched by filename.
- **Q6/Q7** 52,423 scene frames over 175 videos.
- **Q8/Q9** The historical expectation of **52,445 differs by 22**. Frame counts
  vary ±1–2 *between cameras within a sequence* (Boxing: 22 cameras at 366, 18
  at 367; Tea: 15 at 380, 25 at 381). The historical figure used one count per
  sequence; per sequence the deltas are Tea −15, Boxing −22, Plant +2, Dog +13,
  Instrument 0 → **−22**. The measured value is authoritative.

**These three counts are different things** and are never merged: frames that
*exist*, frames fed to the *scene* models, and frames usable by the *hand*
branch. Tea's scene branch uses all 381 frames per camera; its hand branch uses
a subset.

## 3. Q10–Q14 — scene inference

| model | attempted | valid | failure rate |
| --- | ---: | ---: | ---: |
| AnyCalib | 52,423 | **52,423** | **0 %** |
| GeoCalib | 52,423 | **52,423** | **0 %** |
| Perspective Fields | 52,423 | **52,423** | **0 %** |

157,269 scene-frame inferences in total, **zero failures**.

**Q14:** because every model succeeded on every frame,
`THREE_MODEL_COMMON_SCENE_FRAMES` = **52,423** = the full set. The `STRICT` and
`NATIVE` frame sets are therefore identical here, and **Q37** is answered
trivially: the strict-common and model-native results are the same numbers.

## 4. Q15–Q21 — hand branch

| | |
| --- | --- |
| **Q15** WiLoR attempted | 52,423 frames (6,908 reused from CAM-EXP-009.4, flagged `REUSED_CACHE`) |
| **Q16** LEFT available | 35,854 |
| **Q17** RIGHT available | 36,240 |
| **Q18** both hands | 27,803 |
| no hand at all | 8,132 |
| **Q43** any-hand coverage | **44,291 / 52,423 = 84.5 %** |
| **Q19** videos with a computable hand score | **153** (15 excluded for <8 frames/side, 8 with no hand at all) |
| **Q21** exact per-frame manifest | `cam_exp_011_allframe_manifest_v1.csv.gz`, 52,423 rows |

**Q20** Per-sequence hand-usable totals are in the table above; they are a
*subset* of the scene frames, never the sequence length.

## 5. Q22–Q23 — paired set

**Q22/Q23: 151 of 175 videos** are in the `GLOBAL_6_CONDITION_COMMON_VIDEO_SET`
— videos with a result in all six conditions. The 24 missing are hand-branch
ineligible (no hand, or fewer than 8 usable frames on a side), never removed
for performance.

## 6. Q24–Q34 — the main result

### Presentation table

| Model | Scene only | Scene + Hand | Gain | N videos |
| --- | ---: | ---: | ---: | ---: |
| AnyCalib | **16.02 %** | **16.02 %** | +0.00 | 151 |
| GeoCalib | **21.35 %** | **21.35 %** | +0.00 | 151 |
| Perspective Fields | **19.50 %** | **19.50 %** | +0.00 | 151 |

Median relative focal error. Same videos, same frames, same scene evidence; the
only difference is the hand term.

| model | paired gain | 95 % CI | win rate |
| --- | ---: | --- | ---: |
| **Q26** AnyCalib | +0.00 pp | [0.00, 0.00] | 0.7 % |
| **Q29** GeoCalib | +0.00 pp | [0.00, 0.00] | 0.0 % |
| **Q32** Perspective Fields | +0.00 pp | [0.00, 0.00] | 8.6 % |

Secondary metrics are unchanged too (mean, p75, p90, ≤5 %, ≤10 %) except for
Perspective Fields' p75, which moves from 28.14 to 28.58 — a handful of videos
shifted by one grid step in the unhelpful direction.

**Q33/Q34: all three are in the same direction — no change.** The result is not
"it helped one model and not another".

## 7. Why the change is zero — the mechanism, measured before the focal was opened

This is not a flat hand score and it is not a broken pipeline.

| quantity | measured |
| --- | ---: |
| `sigma_scene`, median across videos | **0.0200** (= the frozen floor) |
| scene penalty for one candidate grid step | 0.1303 |
| hand penalty change over that same step | 0.00974 |
| **lambda needed to move one step** | **≈ 13.4** |
| frozen lambda grid maximum | **4.0** |
| normalised hand-score range across the window | 1.99 (i.e. ~2×) |

With **all** frames of a 381-frame video, the per-frame log-MAD of a good scene
estimator falls to the σ floor, which makes the scene term very steep near
q = 1. The hand score varies about twofold across the candidate window, but its
*local slope* near the scene estimate is an order of magnitude too small to tip
the argmin within the pre-registered λ range.

**The nested tuning reached the same conclusion on its own**: λ = 0 was selected
in **11 of 15 outer folds** (all 5 GeoCalib folds, 4 of 5 AnyCalib, 2 of 5 PF).
λ = 0 reproduces scene-only exactly by construction, so the procedure explicitly
turned the hand term off after finding it did not help on TRAIN cameras.

λ was **not** widened after seeing this. The grid and the σ definition were
pre-registered; enlarging λ to manufacture movement would be tuning the protocol
to the outcome.

There is a real and slightly counter-intuitive finding here: **using more frames
narrows the room for an auxiliary cue**, because it tightens the scene aggregate
that the cue would have to overcome.

## 8. Q35–Q36 — per sequence

Median scene-only error per sequence (no condition differs, so one table):

| model | Tea | Boxing | Plant | Dog | Instrument |
| --- | ---: | ---: | ---: | ---: | ---: |
| AnyCalib | 17.8 | 14.2 | 16.4 | 13.0 | 14.4 |
| GeoCalib | 17.8 | 19.9 | 30.7 | 29.6 | 11.9 |
| Perspective Fields | 14.1 | 20.4 | 21.2 | 23.6 | 12.9 |

**Q36: no single sequence dominates.** The zero gain is uniform across all five
— it is not a pooled average hiding a sequence that improved.

## 9. Q38–Q42 — accuracy and precision, and a historical cross-check

| Model | median abs error | signed bias | within-video spread | direction | precision rank |
| --- | ---: | ---: | ---: | --- | ---: |
| AnyCalib | 16.02 % | **+14.66 %** | **3.91 %** | over-estimates | 1 |
| GeoCalib | 21.35 % | **+15.85 %** | 14.71 % | over-estimates | 2 |
| Perspective Fields | 19.50 % | **−11.24 %** | 15.81 % | under-estimates | 3 |

This all-frame run **independently reproduces CAM-EXP-003's 8-frame
characterisation**, which is a useful validation that the pipeline is measuring
the same thing:

| | CAM-EXP-003 (8 frames) | CAM-EXP-011 (all frames) |
| --- | --- | --- |
| AnyCalib bias / spread | +15.6 % / 4.6 % | +14.66 % / 3.91 % |
| GeoCalib bias / spread | +16.4 % / 17.7 % | +15.85 % / 14.71 % |
| PF-centered bias / spread | −11.6 % / 18.8 % | −11.24 % / 15.81 % |

**Q40** AnyCalib remains the high-precision, strongly-biased case: it is very
self-consistent frame to frame (3.9 % spread) yet systematically over-estimates
by ~15 %. **Q41** GeoCalib's spread is ~15 %. **Q42** Perspective Fields is the
only one biased downward.

Historical 8-frame values were **not** substituted for these; both are reported
side by side.

## 10. Q44–Q46 — correction magnitude and leakage

- **Q44** Median hand-induced focal shift: **0.00 %** — where λ = 0 was selected
  the correction is exactly inert, and where λ > 0 it moved a small minority of
  videos by at most one grid step (~1.4 %).
- **Q45/Q46** The reference focal was closed through frame selection, model
  inference, hand scoring and aggregation; read for TRAIN cameras only during λ
  selection; and opened for TEST only after predictions were frozen.
  `results/raw/REFERENCE_FOCAL_OPENED.txt` records git `8fac95e` and the frozen
  prediction hash `487a08868d18859d…`; the audit confirms nothing was
  regenerated afterwards.

## 11. Two bugs fixed before the focal was opened

Both were caught by smoke-testing the fusion rather than by the result looking
wrong, and both are recorded in `results/summary/fusion_scale_diagnostic.json`.

1. **`q_grid` did not contain q = 1.0.** `geomspace(0.5, 1.5, 81)` skips 1.0, so
   λ = 0 returned 1183.8 px instead of the scene-only 1191.5 px — a spurious
   −0.65 % offset on every `+ Hand` estimate, and a λ = 0 "sanity point" that was
   not one. The nearest element is now snapped to exactly 1.0.
2. **The hand score was normalised over the wrong range** — the median over the
   whole 400–2400 px absolute grid (which spans ~12× in score) instead of over
   the candidate window, as CAM-EXP-009.4 does. That crushed the local variation
   the fusion sees.

Neither fix changed the conclusion, but reporting a null from a pipeline with a
−0.65 % systematic offset would have been wrong.

## 12. What this does and does not establish

**Establishes.** On the evaluated 151-video GigaHands paired set, using every
usable RGB frame, attaching this frozen generic hand-structure correction to
AnyCalib, GeoCalib or Perspective Fields produced **no measurable change** in
median focal error, and the nested tuning selected λ = 0 in most folds. The
scene-only characterisation of the three estimators reproduces CAM-EXP-003.

**Does not establish** that hand geometry carries no camera information.
CAM-EXP-009.1 identified a focal almost exactly from clean synthetic bilateral
geometry, and CAM-EXP-009.3.1 found a strong generic bone-identity signal. What
this run measures is narrower: *this* correction, *this* fusion, *this* dataset,
in the all-frame regime where the scene term is at its tightest.

**Does not compare the three estimators.** The purpose was OFF vs ON within each
model. No model is described as best.

## 13. Limitations

1. **The correction is inherited, not validated here.** It comes from the
   CAM-EXP-009.4 line and is **not itself claimed to be a solved hand-geometry
   calibration method**. CAM-EXP-009.4 already found its gain on this rig small
   and confounded by the dataset's narrow focal range (a constant scored
   0.869 % there).
2. The λ grid caps at 4 while ~13.4 would be needed to move one step. That is a
   pre-registered limit, honestly reported rather than relaxed — but it means
   this run tests "the hand term under the inherited fusion", not "the maximum
   possible contribution of hand geometry".
3. One capture rig, 5 sequences, 4 participants, 175 videos. GigaHands' focal
   range is narrow.
4. The hand branch depends entirely on WiLoR's detections and articulation.
5. 24 of 175 videos lack a usable hand branch and are outside the paired set.

## 14. Reproduce

```
python src/audit_dataset_frames.py            # measured frame inventory
python src/build_allframe_manifest.py         # freeze manifests
bash   src/run_all_scene_models.sh            # 3 models x 52,423 frames
python src/cache_hand_observations.py         # (.venv-anyhand, from repo root)
bash   src/run_hand_scores_parallel.sh 8      # sharded, all hand frames
python src/aggregate_and_fuse.py
python src/tune_and_freeze.py                 # TRAIN cameras only
python src/evaluate.py                        # opens the TEST focal
python src/figures.py && python src/self_audit.py
```
