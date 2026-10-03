# GigaHands Split 분석

## 전체 데이터

- participant: 4
- sequence: 5
- camera condition: 245
- video: 248
- 상태: locked
- frame: 70573

## 전체 Focal 분포

- fx: min=824.945, max=1017.040, mean=924.341, median=922.734, Q25=916.312, Q75=932.027
- fy: min=828.935, max=1012.889, mean=923.091, median=922.544, Q25=914.132, Q75=931.859

## Participant별 분포

| Participant | Sequence | Camera | Video | fx median | fy median |
|---|---:|---:|---:|---:|---:|
| p36 | 1 | 49 | 49 | 924.298 | 922.836 |
| p41 | 2 | 98 | 98 | 924.298 | 922.836 |
| p44 | 1 | 49 | 52 | 927.978 | 929.265 |
| p52 | 1 | 49 | 49 | 918.639 | 914.171 |

## Calibration 중복 확인

camera name이 아닌 10개 calibration 값의 exact signature가 겹치는 개수입니다.

| Participant A | Participant B | 동일 Calibration |
|---|---|---:|
| p36 | p41 | 49 |
| p36 | p44 | 0 |
| p36 | p52 | 0 |
| p41 | p44 | 0 |
| p41 | p52 | 0 |
| p44 | p52 | 0 |

## Frame 수

- 전체: 70573
- Train: 44254 (62.7067%)
- Validation: 18652 (26.4294%)
- Test: 7667 (10.8639%)

Camera condition 기준 비율과 실제 frame 비율은 영상 길이가 달라 다를 수 있습니다. 이는 split 오류가 아니며, 학습 시 frame sampling에서 별도로 다룹니다.

## Split 기준

- 같은 participant를 서로 다른 split에 넣지 않습니다.
- frame/video random split을 사용하지 않습니다.
- 245개의 unique sequence-camera 기준으로 focal 분포를 비교합니다.
- fx와 fy를 따로 비교하고 5-bin histogram을 사용합니다.

## 분포 비교 수식

D(S) = mean(|p_S - g|)이며, Train/Validation/Test와 fx/fy의 6개 D 값 평균을 후보 score로 사용합니다.

## Histogram 비교

### fx

| Bin | Global | Train | Validation | Test |
|---|---:|---:|---:|---:|
| 824.945 - 863.364 | 0.82% | 1.36% | 0.00% | 0.00% |
| 863.364 - 901.783 | 6.12% | 6.80% | 4.08% | 6.12% |
| 901.783 - 940.202 | 79.59% | 74.83% | 85.71% | 87.76% |
| 940.202 - 978.621 | 12.65% | 15.65% | 10.20% | 6.12% |
| 978.621 - 1017.040 | 0.82% | 1.36% | 0.00% | 0.00% |
### fy

| Bin | Global | Train | Validation | Test |
|---|---:|---:|---:|---:|
| 828.935 - 865.726 | 0.82% | 1.36% | 0.00% | 0.00% |
| 865.726 - 902.517 | 7.35% | 6.80% | 4.08% | 12.24% |
| 902.517 - 939.307 | 80.41% | 76.19% | 89.80% | 83.67% |
| 939.307 - 976.098 | 10.61% | 14.29% | 6.12% | 4.08% |
| 976.098 - 1012.889 | 0.82% | 1.36% | 0.00% | 0.00% |

## 후보 비교

| Train | Validation | Test | Train cameras | Val cameras | Test cameras | Score |
|---|---|---|---:|---:|---:|---:|
| p41,p44 | p36 | p52 | 147 | 49 | 49 | 0.02757370 |
| p41,p44 | p52 | p36 | 147 | 49 | 49 | 0.02757370 |
| p41,p52 | p36 | p44 | 147 | 49 | 49 | 0.05841270 |
| p41,p52 | p44 | p36 | 147 | 49 | 49 | 0.05841270 |
| p36,p41 | p44 | p52 | 147 | 49 | 49 | 0.05986395 |
| p36,p41 | p52 | p44 | 147 | 49 | 49 | 0.05986395 |

## 확정 Split

- Train: p41, p44
- Validation: p36
- Test: p52

## 선택 이유

Train=p41,p44 조합의 fx/fy histogram score가 가장 낮았습니다. Validation/Test 교환 후보의 score가 동일하여, Train-Test exact calibration overlap이 더 작은 p52를 Test로 선택했습니다.
histogram score=0.02757370, participant overlap=0, Train-Test calibration overlap=0입니다.

## 중요한 한계

participant는 Train / Validation / Test 사이에 겹치지 않습니다. 다만 GigaHands demo에서는 같은 camera identifier가 여러 sequence에 반복됩니다. 따라서 이 split은 unseen-camera 평가를 의미하지 않습니다.

현재 demo는 participant 수가 적기 때문에 정확한 80/10/10보다 participant leakage 방지를 우선했습니다.
Validation의 p36은 Train의 p41과 동일한 calibration parameter가 반복될 수 있어 unseen-calibration 평가가 아닙니다. Test는 Train과 exact calibration overlap이 없는 구성을 우선했습니다.
