# Presentation results

## Main table

| Model | Scene only | Scene + Hand | Gain |
| --- | ---: | ---: | ---: |
| AnyCalib | **16.02 %** | **16.02 %** | +0.00 pp |
| GeoCalib | **21.35 %** | **21.35 %** | +0.00 pp |
| Perspective Fields | **19.50 %** | **19.50 %** | +0.00 pp |

*Median relative focal error on the same **151 paired sequence-camera
videos**. Gain in percentage points (pp).*

**Median relative focal error did not improve for any of the three
estimators.** At the aggregate median level the frozen generic-hand correction
produced a 0.00 pp gain in all three cases.

That is a statement about the **median**, not about every prediction. Some
individual videos did move:

| Model | videos moved | improved | worsened | median shift among moved |
| --- | ---: | ---: | ---: | ---: |
| AnyCalib | 1 / 151 | 1 | 0 | 0.72 % |
| GeoCalib | 0 / 151 | 0 | 0 | — |
| Perspective Fields | 20 / 151 | 13 | 7 | 0.72 % (max 4.96 %) |

So GeoCalib's predictions really were unchanged, while Perspective Fields moved
20 videos — 13 better and 7 worse — which did not produce a measurable median
improvement.

## Frame table

| Sequence | Frames/video | Usable cameras | Scene RGB frames | Any-hand frames | Paired videos |
| --- | --- | ---: | ---: | ---: | ---: |
| Tea | 380–381 (med. 381) | 40 | 15,225 | 12,313 | 39 |
| Boxing | 366–367 (med. 366) | 40 | 14,658 | 12,979 | 36 |
| Plant | 174–175 (med. 174) | 40 | 6,962 | 5,197 | 27 |
| Dog | 336–338 (med. 337) | 40 | 13,493 | 11,717 | 37 |
| Instrument | 139 | 15 | 2,085 | 2,085 | 12 |
| **TOTAL** | | **175** | **52,423** | **44,291** | **151** |

**Video lengths vary between cameras inside a sequence**, so "frames/video" is
a range with a median, not a fixed length. The per-camera lengths are in
`results/summary/video_frame_counts.csv`.

"Any-hand frames" is the subset of RGB frames with at least one usable hand —
it is **not** a sequence length, and it is **not** the number that entered the
hand objective. Of the 44,291 any-hand frames, **42,593** actually entered the
hand score; the rest belong to videos whose hand score was not computable.

The experiment's real input is the directly measured **52,423** RGB
frame-video observations. The figure 52,445 that appears in earlier notes is a
historical estimate obtained by assuming one representative frame count per
sequence.

## Paired set: 175 → 151

| Reason | Videos |
| --- | ---: |
| INSUFFICIENT_LEFT_HAND_FRAMES | 7 |
| NO_USABLE_HAND | 7 |
| INSUFFICIENT_RIGHT_HAND_FRAMES | 6 |
| INSUFFICIENT_BOTH_SIDES | 2 |
| HAND_GRID_DOES_NOT_COVER_CANDIDATE_WINDOW | 2 |
| **Total excluded** | **24** |
| **Final paired** | **151** |

Every excluded video carries exactly one primary reason and 151 + 24 = 175 is
machine-checked. Full ledger: `tables/paired_set_accounting.csv`.

Two coverage numbers, not interchangeable: frame-level hand coverage
**84.5 %** (44,291 / 52,423) and video-level paired coverage **86.3 %**
(151 / 175).

## Accuracy vs precision (scene only)

| Model | median error | signed bias | within-video spread |
| --- | ---: | ---: | ---: |
| AnyCalib | 16.02 % | +14.66 % | **3.91 %** |
| GeoCalib | 21.35 % | +15.85 % | 14.71 % |
| Perspective Fields | 19.50 % | −11.24 % | 15.81 % |

AnyCalib is very consistent frame-to-frame yet systematically over-estimates;
Perspective Fields is the only one biased downward. These reproduce the earlier
8-frame characterisation closely.

This is a within-model OFF/ON comparison. No model is claimed to be best.

## What to say

> 5개 시퀀스의 175개 usable sequence-camera 영상을 대상으로, 실제 RGB
> **52,423 프레임을 모두** 세 scene estimator에 입력했습니다.
>
> 손 정보를 사용할 수 있는 조건을 동일하게 적용한 **151개 paired 영상**에서
> Scene only와 Scene + Hand를 비교했습니다.
>
> 현재 frozen generic-hand correction을 추가했을 때 세 모델 모두 median
> relative focal error의 개선은 **0.00 pp**였습니다.
>
> 이는 손 기하에 camera 정보가 없다는 뜻이 아니라, 이번 all-frame 조건과
> 현재 fusion scale에서는 추가적인 median 성능 향상을 만들지 못했다는
> 결과입니다.

### Phrases to avoid

- "프레임을 많이 쓰니까 손 정보가 필요 없어졌다" — N was not manipulated in a
  controlled way here, so frame count is not identified as the cause.
- "손 정보는 focal 예측에 도움이 안 된다" — this run tests one frozen
  correction under one fusion scale on one rig.
- "세 모델의 모든 예측이 똑같았다" — GeoCalib's were; AnyCalib moved 1 video
  and Perspective Fields moved 20.
- "175개 모두 paired evaluation에 들어갔다" — 151 did.
- "Tea는 381프레임으로 정확히 고정되어 있다" — 380–381 across its cameras.
- "Hand branch는 44,291개의 독립 sample이다" — those are frames nested inside
  videos, and 42,593 of them entered the objective.
