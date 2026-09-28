# CAM-EXP-009.4 — Sequence-Level Scene + Shared-Hand-Anatomy Focal Refinement

```
TAGS  NO_INCREMENTAL_SEQUENCE_HAND_GAIN
      GIGAHANDS_NARROW_FOCAL_CONFOUND_DOMINANT
      SHARED_ANATOMY_TERM_INACTIVE_AS_PARAMETERISED
      REAL_WRONG_BONE_CONTROL_FAILED
      GENERIC_PRIOR_HELPS_SMALL
      FOCAL_NOT_THE_DOMINANT_ABSOLUTE_3D_ERROR

CAM0094_FOCAL_INCREMENTAL_SIGNAL   False
CAM0094_ABSOLUTE_3D_IMPROVED       False
```

**One sentence.** Adding whole-sequence hand geometry to the scene focal estimate
produced a small, statistically detectable improvement (8.08 % → 7.74 % focal
error, paired gain +0.65 pp with a camera-cluster CI excluding zero; wrist/root
3D 109.8 → 106.8 mm) — but it fell far short of the pre-registered bars, **90 % of
its corrections point at the dataset's own rig median**, the **real-data
wrong-bone control did not degrade** (it improved), and the shared-anatomy term
turned out to be inactive as I parameterised it, so the honest reading is that
this run did **not** demonstrate incremental value from hand geometry.

## 1. 쉬운 설명 / Plain-language summary

9.1~9.3.1에서는 손 구조가 focal과 관련된 신호를 갖는지를 분리해서 확인했습니다.
9.4에서는 처음으로 그 손 정보를 실제 장면 기반 focal 추정에 **더했습니다**.

결과를 쉽게 말하면 이렇습니다. 손 정보를 넣으면 focal이 아주 조금 좋아졌습니다
(8.08 % → 7.74 %). 하지만 세 가지가 이 개선을 믿기 어렵게 만듭니다.

**첫째**, GigaHands는 모든 카메라의 focal이 거의 같습니다. 그냥 "학습 카메라들의
중앙값"이라는 상수를 쓰면 **0.87 %** 오차가 나옵니다. 우리 방법(7.74 %)보다 거의
10배 정확합니다. 그리고 우리 방법의 보정 중 **90 %가 바로 그 중앙값 쪽으로**
움직였습니다. 즉 손을 봐서 좋아졌다기보다, 데이터셋의 평균 쪽으로 끌려간 것에
가깝습니다.

**둘째**, 손의 뼈 대응을 일부러 망가뜨리는 통제 실험이 **실패했습니다**. 가상
데이터에서는 뼈를 섞으면 성능이 크게 나빠졌는데(+33 pp), 실제 데이터에서는
오히려 조금 좋아졌습니다. 실제 데이터에서 손 항이 진짜 해부학적 대응을 쓰고
있다고 말할 수 없습니다.

**셋째**, 제가 만든 "양손 공유 해부학" 항이 실제로는 작동하지 않았습니다(§6).

그리고 가장 중요한 발견 하나. **focal을 완벽하게 맞춰도** 손목 3D 오차는
109.8 mm에서 **78.3 mm까지밖에** 줄지 않습니다. 남은 78 mm는 focal이 아니라
단안 깊이 추정에서 옵니다. **focal은 이 문제의 주된 오차원이 아닙니다.**

## 2. Q1–Q10 — setup and provenance

| | |
| --- | --- |
| **Q1 PRIMARY_COMMON_SET** | **123 sequence-camera units** (127 eligible; 4 lost to insufficient usable WiLoR frames) |
| **Q2 physical cameras** | **36** in the common set (37 eligible) |
| **Q3 participants / sequences** | **4 participants, 5 sequences** |
| **Q4 frames** | 32 scene frames and 32 hand frames per side (primary) |
| **Q5 generic prior source** | neutral MANO (`mano_data/MANO_RIGHT.pkl`, betas=0, rest pose) via `smplx`, tips from `smplx.vertex_ids['mano']`, mapped to the OpenPose order in `demo/hand_topology.py` |
| **Q6 GigaHands used in the prior?** | **No.** Verified in the audit. The mapping was *checked* against WiLoR output (Spearman 0.9985) but no WiLoR or GigaHands geometry enters it |
| **Q7 WiLoR fields used** | `keypoints_2d` (original pixels) and `keypoints_3d` (root-relative) |
| **Q8 bone lengths discarded?** | **Yes** — only unit bone directions; lengths never reach the anatomy model |
| **Q9 `p_seq` refit per video?** | **Yes**, from scratch for every sequence-camera unit |
| **Q10 anatomy carried across videos?** | **No** — exactly what CAM-EXP-009.3.1 found no support for |

**Leakage barrier (Q10 cont.).** The reference focal was closed through Phases
A–C; the scene cache contains `gt_fx`, so it is read only through a reader that
strips every such column (verified: 11,200 rows, 0 `gt_` columns). `lambda_fusion`
was tuned on TRAIN cameras only. `results/raw/REFERENCE_FOCAL_OPENED.txt` records
the opening at git `221dc2c` with the frozen-prediction SHA-256
`957b7ca23b4f…`, and the audit confirms the predictions were **not** regenerated
afterwards.

## 3. Q11–Q14 — synthetic gate (PASSED)

| condition | median focal error | boundary |
| --- | ---: | ---: |
| **Q12** S0 CLEAN | **0.272 %** (grid floor — exact hit) | 0.00 |
| **Q13** S1 MODERATE | **3.42 %** (bar ≤ 5 %) | 0.00 |
| **Q14** S2 WRONG_BONE | 36.54 % → **+33.1 pp** (bar ≥ 10) | 0.38 |
| S3 GENERIC_PRIOR_OFF | 9.02 % | 0.00 |

**Q11: PASSED**, so the real phase ran. The gate did its job — it caught nothing
wrong because nothing was wrong *in synthetic*. §7 is where the real data
disagrees with it.

## 4. Q15–Q21 — focal results

| method | median err | mean | p75 | p90 | ≤5 % | ≤10 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q15 M0_SCENE_ONLY** | **8.077 %** | — | — | — | 23.6 | 56.9 |
| **Q16 M1** generic, independent sides | 7.744 % | — | — | — | 26.0 | 64.2 |
| **Q17 M2** shared, no generic prior | 8.282 % | — | — | — | 25.2 | 63.4 |
| **Q18 M3** full | **7.744 %** | — | — | — | 26.0 | 64.2 |
| **Q27 RIG_MEDIAN_TRAIN_ONLY** *(diagnostic)* | **0.869 %** | — | — | — | **99.2** | **100.0** |

- **Q19 relative reduction M3 vs M0: 4.1 %** — against a pre-registered bar of
  10 %. **Not met.**
- **Q20 paired median gain: +0.651 pp, 95 % camera-cluster CI [0.638, 0.669]**
  (123 units, 36 cameras, 10,000 iterations). Positive and excludes zero.
- **Q21 within 5 %: 23.6 → 26.0 (+2.4 pp); within 10 %: 56.9 → 64.2 (+7.3 pp).**

**`CAM0094_FOCAL_INCREMENTAL_SIGNAL = False`**: criteria 1, 3 and 4 pass
(M3 < M0, gain > 0, CI lower bound > 0) but criterion 2 fails (4.1 % < 10 %).

### Q22 — hand-only, and why it matters

`H_ONLY` (hand score, no scene term) gives **37.15 %** median error with a
**boundary rate of 0.97**: on real data the hand curve has no interior minimum in
97 % of units — it simply slides to the edge of the search range. The hand term
contributes nothing on its own; it only ever nudges an estimate the scene
already anchored.

## 5. Q23–Q26 — tuning, boundaries, frame counts

- **Q23 selected lambdas**: `lambda = 2.0` in every outer fold for M1 and M3;
  2.0/2.0/2.0/4.0/4.0 for M2. Stable across folds.
- **Q24 boundary hit rate**: 0.00 for the *fused* estimate (the scene term keeps
  it interior) but **0.97 for the hand term alone**.
- **Q25 frame-count curve**: N = 8/16/32 are available; **N = 64 was skipped**,
  decided on measured runtime before any focal result and recorded in the spec.
  The primary result is N = 32.
- **Q26 pose diversity**: recorded per unit as a secondary stratification and not
  used to select frames.

## 6. Q37–Q38 — ablations, and a design fault of mine

| comparison | median A | median B | paired gain | relative |
| --- | ---: | ---: | ---: | ---: |
| **Q37** generic prior (M3 − M2) | 7.744 | 8.282 | +0.000 pp | **+6.5 %** |
| **Q38** shared anatomy (M3 − M1) | 7.744 | 7.744 | **+0.000 pp** | **+0.0 %** |
| PRIMARY (M3 − M0) | 7.744 | 8.077 | +0.651 pp | +4.1 % |

**The shared-anatomy term did nothing, because I parameterised it wrongly.**
`p_L = softmax(log p0 + a + delta_L)` depends only on the **sum** `a + delta_L`.
With `lambda_generic = lambda_side = 1.0` the ridge penalty splits the same total
offset between the two terms instead of forcing the sides to agree through
`p_seq`. Measured: ‖a‖ = 0.055 vs ‖delta_L‖ = 0.061 — comparable, i.e. the split
is roughly even and the hands are **not** tied together. The M1 and M3 score
curves coincide to a median max difference of 5.25e-06 against a typical score
scale of 5.3e-04.

Sharing would only bite with `lambda_side ≫ lambda_generic`. **I did not re-tune
after discovering this**, because TRAIN focal errors had already been seen;
changing the weights then would be tuning on results. The fix belongs to a
separately pre-registered run.

The synthetic gate could not have caught it: the gate ran M3 only and never
compared M1 against M3.

**What the ablation therefore actually measures**: the **generic MANO prior helps
a little** (+6.5 % relative), and **sequence-shared anatomy contributed exactly
nothing** at these weights.

## 7. Q39 — the real-data control FAILED

| control | hand-only median err | score range | boundary rate |
| --- | ---: | ---: | ---: |
| C0 correct bone mapping | 37.15 % | 1.48e-03 | 0.97 |
| **C_WRONG_BONE** (within-finger permutation) | **31.66 %** | 1.88e-03 | 0.37 |

**Permuting the bones did not degrade the hand score — it improved it**, and
halved the boundary rate. This is the exact opposite of the synthetic result
(+33.1 pp degradation), and it is the most important negative finding in this
run.

Reading: on real data the hand term is **not** exploiting correct anatomical bone
correspondence. Whatever tiny signal it carries does not depend on which bone is
matched to which. Combined with the 0.97 boundary rate and the rig-median
analysis below, the +0.65 pp gain is very unlikely to be hand-geometry evidence.

The method was **not** retuned after seeing this control.

## 8. Q40, Q27–Q28 — the GigaHands confound, which dominates

| | value |
| --- | ---: |
| **Q40** test reference focal, median | 920.6 px |
| relative span (max − min) / median | **1.87 %** |
| coefficient of variation | **0.53 %** |
| **Q27** `RIG_MEDIAN_TRAIN_ONLY` median error | **0.869 %** |
| **Q28** M3 corrections moving toward the rig median | **90.2 %** |

A constant borrowed from the training cameras beats every real method by roughly
a factor of ten and is within 5 % on 99.2 % of units, because the whole rig spans
under 2 % in focal. And 90.2 % of M3's corrections point toward that constant.

**`GIGAHANDS_NARROW_FOCAL_CONFOUND_DOMINANT`.** This dataset can neither reward
nor properly test a focal-refinement method. `RIG_MEDIAN_TRAIN_ONLY` is a
`DATASET_CONFOUND_DIAGNOSTIC`, **not a deployment method** — in a real
deployment there is no rig median to borrow — and it is reported here precisely
because hiding it would make a 4 % improvement look meaningful.

## 9. Q29–Q36 — absolute 3D

Bimanual combined, per sequence-camera unit, paired, camera-clustered:

| metric | M0 | M3 | relative | paired gain | 95 % CI |
| --- | ---: | ---: | ---: | ---: | --- |
| **Q29/Q30** wrist/root (mm) | 100.74 | 97.39 | +3.32 % | +2.94 mm | [2.32, 3.54] |
| **Q31** root depth (mm) | 84.18 | 80.64 | +4.20 % | +3.60 mm | [2.81, 4.00] |
| **Q32** absolute MPJPE (mm) | 99.78 | 99.14 | +0.64 % | +3.21 mm | [2.77, 3.59] |
| **Q33** root-aligned MPJPE (mm) | 34.10 | 34.10 | 0.00 % | **0.000 mm** | [0.000, 0.000] |

**Q33 is the control and it is perfect.** Changing only the focal changes only
the camera translation, so root-aligned MPJPE must not move — and it moved by
exactly 0.000 mm. The downstream implementation is validated;
`DOWNSTREAM_IMPLEMENTATION_REVIEW` is not raised.

**Q34/Q35 left vs right** (median wrist/root, mm): M0 left 109.82 / right 109.79
→ M3 left 106.88 / right 106.63. Both hands improve together; nothing is traded
between them.

**`CAM0094_ABSOLUTE_3D_IMPROVED = False`** — the bars were ≥ 10 % on wrist/root
and ≥ 5 % on absolute MPJPE.

### Q36 — and the single most useful number in this run

| method | wrist/root (mm) | root depth (mm) | abs MPJPE (mm) |
| --- | ---: | ---: | ---: |
| M0 scene-only | 109.8 | 95.4 | 105.8 |
| M3 full | 106.8 | 92.1 | 102.6 |
| **REF_FOCAL_ORACLE** (upper diagnostic) | **78.3** | **58.0** | **68.3** |

**Even a perfect focal only takes wrist/root error from 109.8 mm to 78.3 mm.**
So of ~110 mm of absolute error, roughly 32 mm is attributable to the focal and
**~78 mm comes from somewhere else** — the monocular depth/scale in WiLoR's
`cam_t`. Focal refinement is working on the smaller half of the problem, and we
captured about 3 mm of the 32 mm available.

`FOCAL_NOT_THE_DOMINANT_ABSOLUTE_3D_ERROR`.

## 10. Q41–Q42 — what this establishes

**Facts.** On the evaluated 123-unit GigaHands common set: the full model reduced
median focal error from 8.077 % to 7.744 % (paired +0.651 pp, CI [0.638, 0.669])
and median wrist/root error from 109.8 mm to 106.8 mm. A constant train-rig
focal scored 0.869 %. The real-data wrong-bone control did not degrade. The
shared-anatomy term contributed 0.000 pp. An oracle focal leaves 78.3 mm of
wrist/root error.

**Interpretation.** The small gain is more consistent with shrinkage toward a
near-constant dataset focal (90.2 % of corrections) than with hand-geometry
evidence, and the failed real-data control argues the same way. The generic
anatomical prior contributes a little; the sequence-shared anatomy, as built,
contributes nothing.

**This does NOT mean** that `hand geometry contains no camera information` —
CAM-EXP-009.1's clean synthetic identification stands, and so does
CAM-EXP-009.3.1's strong generic bone-identity signal. What it means is narrower:
*this* formulation did not add measurable focal accuracy beyond the scene
estimator on *this* dataset.

**Q41 generalisation?** **No.** With a rig spanning under 2 % in focal, neither a
positive nor a negative result here transfers to cameras in general.

**Q42 varied-intrinsics dataset needed?** **Yes, and it is now the blocking
requirement** for any further focal-refinement work. `FINAL_CONFIRMATORY_HOLDOUT`
remains unopened.

## 11. Q43 — what to do next

Ranked by expected value given these numbers:

1. **Attack the monocular depth scale, not the focal.** §9 shows ~78 mm of the
   ~110 mm absolute error survives a perfect focal. That is where the robot task
   actually loses accuracy.
2. **Get a varied-intrinsics evaluation** — a dataset with genuinely different
   focal lengths, or controlled self-captured calibration sequences. Without it
   no focal method can be told apart from a constant on this data.
3. **If the bilateral idea is pursued at all**, fix the parameterisation first
   (`lambda_side ≫ lambda_generic`, or a hard constraint tying the sides) and
   re-validate with an M1-vs-M3 comparison *inside* the synthetic gate — the
   check this run was missing.
4. **Keep the generic anatomical prior.** It is the one component that helped in
   both synthetic (+5.6 pp) and real (+6.5 % relative) settings.

## 12. Limitations

1. **The dataset cannot test the question.** Reference focal CV 0.53 %, span
   1.87 %; a constant scores 0.869 %.
2. 4 participants, 5 sequences, 36 cameras — and the focal is a camera property,
   so the effective diversity is one rig.
3. The shared-anatomy term was inactive as parameterised (§6). The M3-vs-M0
   result is therefore really "scene + generic-prior independent-side anatomy vs
   scene".
4. The real-data wrong-bone control failed (§7), so the hand term's mechanism on
   real data is not established.
5. Everything depends on WiLoR's predicted articulation; its errors propagate
   directly into the hand score.
6. The reference 3D carries reconstruction error of unknown size, which bounds
   how small a downstream difference is measurable.
7. N = 64 hand frames was not run (decided on runtime, pre-result).

## 13. Reproduce

```
python src/audit_sources.py                       # PHASE A
python src/build_mano_prior.py                    # (.venv-anyhand)
python src/build_units.py
python src/cache_wilor_predictions.py             # (.venv-anyhand, ~22 min)
python src/run_synthetic_gate.py                  # PHASE B, ~9 min
python src/run_real_label_free.py                 # PHASE C, ~154 min
python src/tune_fusion_nested.py                  # PHASE D/E
python src/evaluate_focal.py                      # PHASE F/G - opens the focal
python src/evaluate_absolute_3d.py                # PHASE H
python src/run_controls.py                        # ~59 min
python src/evaluate_ablations.py && python src/figures.py && python src/self_audit.py
```
