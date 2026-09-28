# CAM-EXP-009.3.1 — Participant-Aware Reanalysis of Real Bilateral Geometry Separability

```
ANALYSIS TYPE  post-result provenance correction and participant-aware
               reanalysis.  NOT a preregistered confirmatory experiment.
               created_after_cam0093_results = true

TAGS  OFFICIAL_PARTICIPANT_MAPPING_VERIFIED
      PARTICIPANT_REANALYSIS_UNDERPOWERED
      WITHIN_SESSION_BILATERAL_SEPARATION_PRESENT
      CROSS_SESSION_STABILITY_NOT_SUPPORTED_FOR_P41
      GENERIC_BONE_IDENTITY_SIGNAL_STRONG
      SUBJECT_SPECIFIC_PRIOR_NOT_JUSTIFIED
      SEQUENCE_SPECIFIC_SHARED_ANATOMY_REMAINS_PLAUSIBLE

FOCAL ESTIMATED: NONE.  No reconstruction re-run.
```

**One sentence.** With participant identity verified from the official GigaHands
naming convention, a participant's own left/right hand structure is closer than
another participant's (0.0289 vs 0.0494, ratio 0.59, separation 1.9× the
measurement repeatability) — **but the one participant with two sessions has
hands that are further apart across those sessions (0.077–0.085) than two
different participants are (0.046)**, so the within-session similarity is not a
persistent person-level signature, and a strong subject-specific anatomy prior is
not justified by this data.

## 1. 쉬운 설명 / Plain-language summary

9.3에서는 로컬 파일 안에서 participant 필드를 찾지 못해서 `p41-boxing`과
`p41-plant`를 같은 사람이라고 확정하지 못했습니다. 9.3.1에서는 GigaHands 공식
문서가 디렉터리 구조를 `p<participant id>-<scene>-<squence id>/`로 명시한 것을
확인해서 `p36`, `p41`, `p44`, `p52`를 participant ID로 다시 매핑했습니다.

**중요한 점은 3D 손을 다시 계산한 것이 아니라, 9.3에서 이미 만든 동일한
geometry를 올바른 participant grouping으로 다시 분석했다는 것입니다.**

결과를 쉽게 말하면 이렇습니다. 같은 영상 안에서 한 사람의 두 손은 서로 꽤
비슷했고, 다른 사람의 손보다 확실히 더 가까웠습니다. 여기까지는 좋은 신호입니다.

그런데 두 번 촬영된 유일한 사람(`p41`)을 보니, **같은 사람의 손인데도 다른
촬영분끼리는 서로 전혀 닮지 않았습니다.** 왼손끼리 비교해도 그랬습니다. 오히려
남남의 손보다 더 멀었습니다.

즉 "같은 영상 안에서 두 손이 닮았다"는 것은 사람의 고유한 손 모양 때문이라기보다
**같은 촬영·같은 재구성 조건을 공유하기 때문**일 가능성이 큽니다. 그래서 사람별
고정 손 모양을 미리 정해두고 다른 영상에 가져다 쓰는 방식은 근거가 약합니다.

다만 우리 실제 과제는 영상마다 작업자 한 명이고 calibration도 영상마다 새로
하므로, **영상 하나 안에서 두 손을 함께 묶는 방식은 여전히 유효한 선택지**입니다.

## 2. Q1–Q4 — the identity correction

| | |
| --- | --- |
| **Q1 official convention verified?** | **YES** — `OFFICIAL_PARTICIPANT_MAPPING_VERIFIED` |
| **Q2 source** | GigaHands official repository README, `https://github.com/Kristen-Z/GigaHands`, accessed 2026-09-28, snapshot SHA-256 `7082c858e8b8b00b…` |
| **Q3 participants** | 5 sequences → **4 participants** |
| **Q4 p41 two sequences = same participant?** | **YES** |

The README's own directory specification writes the take directory as

```
p<participant id>-<scene>-<squence id>/
```

and the local layout `demo_all/raw/hand_pose/p36-tea-0010/` matches it exactly.
This is a **dataset-defined identifier whose semantics the dataset documents**,
not a pattern guessed from a filename.

| sequence | participant | scene | sequence id |
| --- | --- | --- | --- |
| p36-tea-0010 | p36 | tea | 0010 |
| p41-boxing-0021 | **p41** | boxing | 0021 |
| p41-plant-0004 | **p41** | plant | 0004 |
| p44-dog-0004 | p44 | dog | 0004 |
| p52-instrument-0034 | p52 | instrument | 0034 |

**MANO `shapes` was not used as an identity source**, exactly as in
CAM-EXP-009.3: defining identity from fitted hand shape and then measuring
hand-shape similarity would be circular.

**Why CAM-EXP-009.3 said otherwise.** Its audit searched only the locally
shipped files, which genuinely contain no participant field. That was the
correct conclusion from the evidence it had; the convention is documented
upstream. CAM-EXP-009.3 was not an analysis error, and its historical verdict
stays on the record untouched.

## 3. Q5–Q6 — reuse and reproduction

| | |
| --- | --- |
| **Q5 raw geometry reused byte-identically?** | **YES** |
| reference reconstruction re-run | **NO** — the 14,015 `loco.reconstruct` calls were not repeated |
| source run | `CAM-EXP-009_3_real_bilateral_geometry_separability`, commit `1b92dc1`, working tree clean |
| **Q6 v1 metrics reproduced?** | **YES — all 11 at tolerance 1e-9** |

Recorded facts re-checked against the actual files (not trusted from the
report): 14,015 reconstructions, 13,903 frame vectors,
`target_camera_in_reference_count = 0`, 20 bones, `sum p = 1`, ≤ 48 frames per
unit — all OK. Full hashes in `tables/source_artifact_hashes.csv`;
reproduction in `tables/cam0093_v1_reproduction_audit.csv`.

## 4. Q7–Q12 — the four distance levels

Primary distance unchanged from CAM-EXP-009.3:
`median_b |log(p_A,b + 1e-8) − log(p_B,b + 1e-8)|` over the 20 connected bones.
No new metric was introduced.

| quantity | median |
| --- | ---: |
| **Q7 `D_repeat`** same hand, measured twice | **0.01087** |
| **Q8 `D_view_repeat`** same hand, two cameras | **0.01130** |
| **Q9 `D_within_session`** — all 145 units | 0.02829 |
| **Q9 `D_within_session`** — participant equal weight | **0.02887** |
| **Q10 `D_cross_participant`** — all 947 pairs | 0.04569 |
| **Q10 `D_cross_participant`** — participant-pair equal weight | **0.04937** |
| `D_cross_participant` LEFT→RIGHT | 0.04727 |
| `D_cross_participant` RIGHT→LEFT | 0.04706 |
| **`p41` cross-session, same side** | **0.08516** |
| **`p41` cross-session, opposite side** | **0.07704** |

- **Q11 ratio (participant equal weight) = 0.585**
- **Q12 separation / repeatability = (0.04937 − 0.02887) / 0.01087 = 1.886**

Participant-clustered bootstrap (4 clusters): within [0.0169, 0.0426], cross
[0.0428, 0.0482]. **Supplemental and unstable** — 4 clusters is not enough for
these intervals to carry weight, and a narrow-looking interval here is not
strong evidence.

## 5. Q13 — per participant

| participant | sequences | `D_within_session` |
| --- | ---: | ---: |
| p36 | 1 | 0.05210 |
| **p41** | **2** | **0.02066** |
| p44 | 1 | 0.02683 |
| p52 | 1 | 0.03091 |

`p36` is above the cross-participant level (0.0494), i.e. reversed. `p41` — the
participant that *can* be checked across sessions — has the tightest
within-session pairing of the four.

## 6. Q14–Q18 — the cross-session test, and the result that decides this run

`p41-boxing-0021` vs `p41-plant-0004`, camera-matched, all four combinations:

| | comparison | cameras | median | vs cross-participant (0.04569) |
| --- | --- | ---: | ---: | --- |
| **Q14** | boxing L ↔ plant L (same side) | 25 | **0.08904** | **larger** |
| **Q15** | boxing R ↔ plant R (same side) | 23 | **0.07525** | **larger** |
| **Q16** | boxing L ↔ plant R (opposite) | 24 | **0.08104** | **larger** |
| **Q17** | boxing R ↔ plant L (opposite) | 24 | **0.07455** | **larger** |
| | pooled same side | 48 | 0.08516 | larger |
| | pooled opposite side | 48 | 0.07704 | larger |
| | *p41 within-session, for reference* | | *0.02113* | *smaller* |

**Q18: No. All four cross-session comparisons are LARGER than the
cross-participant distance**, by roughly 1.6–1.9×, and roughly 4× larger than
`p41`'s own within-session distance.

Even **left vs left** — the same participant's same hand, two sessions — does
not match. If a stable person-level hand template existed in this reference
geometry, that is the comparison that should have been tight, and it is the
loosest of all.

**Caveat, and it is a large one: this is ONE participant.** No population claim
follows from n = 1.

## 7. Q19–Q22 — identification and permutation

| | LEFT→RIGHT | RIGHT→LEFT | combined |
| --- | ---: | ---: | ---: |
| **Q19/Q20 participant-template top-1** | 50.0 % (2/4) | 50.0 % (2/4) | 50.0 % (4/8) |
| **Q21 chance** | 25.0 % | 25.0 % | 25.0 % |

Above chance in raw terms, but this is **2 correct out of 4** in each direction.
With 4 participants nothing can be concluded from it, and it is reported as a
descriptive diagnostic only.

**Q22 participant-block permutation** — exact over 4! = 24 relabelings, with
`p41`'s two sessions always moving together as one block:

| | value |
| --- | ---: |
| observed `median(cross) − median(same)` | +0.009846 |
| null median | +0.001342 |
| null max | +0.013214 |
| **exact p (one-sided)** | **0.0833** |
| floor (1/24) | 0.0417 |

**Not significant even at this coarse resolution**, and two relabelings scored
higher than the truth. Note the change from CAM-EXP-009.3's sequence-level test
(p = 0.0167): once `p41`'s two sessions are correctly bound into a single block
rather than permuted independently, the apparent effect weakens. That is the
statistical face of the same finding as §6.

This test is **supporting only**; no conclusion rests on it.

## 8. Q23–Q24 — the generic bone-identity control

| | median `D_within_session` |
| --- | ---: |
| correct bone mapping | 0.02829 |
| within-finger permuted mapping | **0.27065** |

**Q23/Q24: +857 % degradation — the generic signal is strong and survives
unchanged.** Permuting which bone is compared with which destroys the distance
by nearly an order of magnitude.

This establishes that correct **anatomical bone correspondence** carries strong
generic structure. It says nothing about subject-specific structure — those are
different claims and are kept apart throughout.

It does **not** yet mean a generic prior improves focal estimation. That is a
separate experiment and was not run here.

## 9. Q25–Q28 — the three questions, separated

**A. Generic anatomical correspondence** (index bone ↔ index bone):
**STRONG.** +857 % control degradation.

**B. Within-session bilateral consistency** (one recording, one worker's two
hands): **PRESENT.** ratio 0.585, separation 1.9× repeatability. One of four
participants (`p36`) is reversed, and the participant-block permutation is not
significant (p = 0.083), so it is present but not strongly established.

**C. Cross-session person-specific signature**: **NOT SUPPORTED.** For the only
participant testable, all four cross-session comparisons exceed the
cross-participant level.

- **Q25 subject-specific stable anatomy signal?** Not supported by this data.
- **Q26 within-session bilateral consistency?** Present.
- **Q27 how big is the session confound?** Large enough to be the leading
  explanation of B. A within-session pair shares session, calibration, capture
  conditions and reconstruction context; the cross-session result shows those
  shared factors are not carried by the participant.
- **Q28 population claim with 4 participants?** **No.** `UNDERPOWERED` stands.
  Verifying identity fixed the provenance, not the power.

## 10. Q29–Q32 — claim survival

| claim | status |
| --- | --- |
| same recording's L/R closer than a different recording's | **SUPPORTED** |
| same participant L/R closer than different participants | **UNDERPOWERED** (true at 0.585, but n=4, permutation p=0.083, not separable from session) |
| same participant geometry persists across sessions | **NOT_SUPPORTED_FOR_P41** |
| generic corresponding bone identity matters | **SUPPORTED** (+857 %) |
| **Q29/Q31 strong subject-specific shared anatomy prior justified** | **NO** |
| **Q30/Q32 sequence-specific shared anatomy remains plausible** | **YES** |

`tables/claim_survival_audit.csv`.

### What is superseded, and what is not

CAM-EXP-009.3 **remains valid** as a within-sequence vs cross-sequence analysis;
all of its numbers reproduce exactly. Its **participant-provenance
interpretation** is superseded: its `SAME_SUBJECT_PROVENANCE_UNRESOLVED` was the
right outcome of a local-file audit, and the official documentation resolves it.

Two of its specific readings change:

1. CAM-EXP-009.3 excluded the `p41` pair as "ambiguous" and noted it was the
   furthest apart of anything measured. That is now interpretable: it is a
   **same-participant cross-session** comparison, and it is the central result
   rather than an oddity.
2. Its 53.7 % top-1 figure is retained but **renamed** to
   `WITHIN_SESSION_PAIRING_TOP1` / `SEQUENCE_PAIRING_DISCRIMINATION`. It was
   never participant identification and is not reported as such.

## 11. Q33 — what this means for the deployment task, and the next experiment

The intended deployment is: **one worker per video, that worker's two hands, one
continuous sequence, offline whole-video processing, calibration estimated per
video, and no requirement to recognise a person across videos.**

That matters for how this result should be read. A weak cross-session person
signature does **not** sink the idea of a shared hand constraint *within* a
sequence — persistent identity across videos is not something the task needs.
Equally, the within-session similarity does **not** justify carrying a
person-specific prior between recordings.

**Recommended direction — OPTION B + C, not A:**

- **OPTION A — strong subject-specific prior** (a person's stable anatomy reused
  across recordings): **not recommended.** §6 is direct evidence against it, and
  §9C is the reason.
- **OPTION B — sequence-specific shared anatomy: recommended.** Estimate
  `p_seq` afresh in every new video, with
  `p_L = p_seq + delta_L`, `p_R = p_seq + delta_R`, and carry nothing over from
  a previous recording. This matches the deployment (one worker per video,
  per-video calibration) and is the formulation the evidence supports: the
  within-session bilateral relationship is the one that held.
- **OPTION C — generic human-hand anatomical prior: recommended alongside B.**
  The +857 % bone-permutation control shows correct bone identity and plausible
  proportion structure carry strong generic signal, independent of who the
  person is.

**No focal model was built here, by design.** Whether B + C actually improves a
focal estimate is the next substantive experiment, and it needs its own frozen
spec.

## 12. Limitations

1. **n = 4 participants**, against the pre-set bar of 8. Everything at
   participant level is underpowered.
2. **n = 1 for the cross-session question.** The finding that decides this run
   rests on a single participant's two sessions.
3. This is a **post-result reanalysis**. It is not preregistered, and it is
   labelled as such throughout.
4. The permutation has 24 relabelings, 4 units, and a floor of 0.0417; the
   bootstrap has 4 clusters. Both are supplemental.
5. The reference geometry mixes annotation, triangulation, calibration and
   association error with anatomy. A cross-session mismatch is equally
   consistent with a session-dependent reconstruction as with anything about
   hands.
6. One reconstruction pipeline, one QC definition, one primary metric. No
   per-bone feature selection was performed.
7. `p41`'s participant template is an average of two sessions that disagree
   strongly, so the participant-level identification diagnostic is weakened for
   exactly the participant that matters most.

## 13. Reproduce

```
python src/verify_official_identity.py     # the identity gate
python src/audit_source_artifacts.py       # integrity of reused CAM-009.3 files
python src/reproduce_cam0093_v1.py         # v1 metrics, tolerance 1e-9
python src/compute_participant_distances.py
python src/compute_cross_session_p41.py
python src/build_participant_templates.py
python src/run_participant_permutation.py
python src/evaluate_reanalysis.py
python src/figures.py
python src/self_audit.py
```
