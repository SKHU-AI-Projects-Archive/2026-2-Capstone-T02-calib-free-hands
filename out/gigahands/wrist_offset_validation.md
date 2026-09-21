# A. wrist offset(keypoints_3d[WRIST]) 검증

## 배경

`translation_benchmark_report.md` 초판에서, `cam_t`(WiLoR가 예측하는 절대 translation)만으로는
절대 손목 위치가 아니라는 걸 발견하고 `pred_wrist = cam_t + keypoints_3d[WRIST]`로 보정했다.
당시엔 디버그 인스턴스 1개(left≈[-95.7,6.4,6.2]mm, right≈[+95.7,6.4,6.2]mm)만 보고 "손마다
고정 오프셋"이라고 가정했다 — 이 문서는 그 가정을 매칭된 전체 1,846개 인스턴스로 검증한다.

## 방법

`gigahands_raw_instances_cache.csv`(이번에 새로 추가 — 아래 "데이터 출처" 참고)의
`kp3d_wrist_x/y/z`(=`keypoints_3d[WRIST]`, root-relative, focal_length과 무관하게
WiLoR forward pass 단독 출력) 1,846개를 손 전체/손별(is_right)/화각 그룹별/카메라별로
분포를 확인했다.

**판정 기준**: 벡터 크기(norm)의 상대표준편차(RSD = std/mean × 100%)가 **5% 미만이면
"고정 상수로 취급 가능"**, 그 이상이면 "개별 오차 성분 포함"으로 판정한다. 5%를 고른 이유:
이번 실험이 다루는 절대오차가 수십~수백 mm 단위(전체 평균 130.9mm)인데, wrist offset의
평균 크기(96mm)의 5%는 4.8mm — 이는 전체 절대오차의 약 3.7%에 불과해, 이 정도의 흔들림은
"고정 상수로 보정"이라는 단순화가 다른(훨씬 큰) 오차원에 묻혀 실질적 영향이 없다고 볼 수
있는 임계선이다. 계측/공정관리에서 변동계수(CV) 5% 미만을 "반복성 우수"로 보는 관례도
참고했다.

## 데이터 출처

`eval_translation_gigahands.py`(이미 완료된 스크립트)는 오차만 집계해 저장하고 raw 값
(`keypoints_3d[WRIST]`, `pred_cam` 등)은 디스크에 남기지 않았다 — 이번 후속분석에 그
raw 값 자체가 필요해서, 동일한 표본(seed=0, target-per-group=500)에 대해
`capture_raw_instances_gigahands.py`로 **1회** 재실행해 raw 값을 전부 캐시로 남겼다(추론
자체를 두 번째로 돌린 게 아니라, 처음부터 저장해뒀어야 할 값을 이번에 저장한 것). 재현성은
`abs_error_gt_focal_mm`/`abs_error_hardcoded_mm`/`z_error_gt_focal_mm`/`z_error_hardcoded_mm`
4개 컬럼을 기존 `eval_translation_gigahands.csv`와 1,846행 전부 대조해 **모든 값이 소수점까지
완전히 일치**함을 확인했다(최대 절대차이 0.0) — 이번 캡처가 기존 결과의 충실한 재현임이
보장된다.

## 결과

### 1) 전체 분포 (norm = √(x²+y²+z²), mm)

| | mean | std | min | max |
|---|---:|---:|---:|---:|
| norm | 96.014 | 0.168 | 94.986 | 96.562 |

**상대표준편차 = 0.168 / 96.014 = 0.17%** — 5% 임계값 대비 압도적으로 작다.

### 2) 손(is_right)별

| side | x (mm) | y (mm) | z (mm) | norm mean±std (mm) |
|---|---:|---:|---:|---:|
| left (is_right=False) | −95.605 | +6.377 | +6.198 | 96.017 ± 0.123 |
| right (is_right=True) | +95.597 | +6.375 | +6.201 | 96.010 ± 0.207 |

x축만 부호가 반대(완벽한 미러링)이고 y/z는 거의 동일 — WiLoR가 왼손 crop을 좌우반전해
넣고(`is_right==0`일 때 flip) 오른손 기준으로 추론한 뒤 x만 되돌리는 파이프라인 구조
(`rgb_predictor.py`의 `flip = 2*is_right-1` 처리)와 정확히 부합한다.

### 3) 화각 그룹별

| group | mean (mm) | std (mm) | n |
|---|---:|---:|---:|
| 25mm | 96.076 | 0.123 | 872 |
| 26mm | 95.924 | 0.195 | 656 |
| 27mm | 96.030 | 0.126 | 318 |

### 4) 카메라별 (표본에 등장한 5대 전부)

| camera | mean (mm) | std (mm) | n |
|---|---:|---:|---:|
| brics-odroid-011_cam0 | 96.074 | 0.128 | 766 |
| brics-odroid-001_cam0 | 95.924 | 0.196 | 650 |
| brics-odroid-009_cam1 | 96.030 | 0.126 | 318 |
| brics-odroid-014_cam1 | 96.092 | 0.078 | 106 |
| brics-odroid-001_cam1 | 95.910 | 0.115 | 6 |

손·화각 그룹·카메라 어느 축으로 나눠봐도 96mm 근방 ±0.2mm 이내로, 사실상 완전히
동일하다.

## 판정

**고정 상수로 취급 가능** (RSD 0.17% ≪ 5%). 이 값은 WiLoR/AnyHand의 MANO 회귀 구조가
"root"로 쓰는 관절 위치와 OpenPose 21점 레이아웃의 손목(wrist) 랜드마크 사이의 거의
불변인 기하학적 오프셋으로 보인다 — 입력 포즈·손 모양·프레임·카메라가 전부 달라져도
거의 흔들리지 않는다는 게 이번 검증으로 확인됐다.

## 보정 전/후 절대오차 비교 (참고용 — 판정이 "상수 취급 가능"이라 재계산이 최종 결론을
바꾸지는 않지만, 요청대로 나란히 제시)

| | GT focal (mean / median) | 24mm 하드코딩 (mean / median) |
|---|---:|---:|
| **보정 전** (cam_t만 사용, 버그였던 버전) | 153.44 / 145.09 mm | 170.00 / 174.59 mm |
| **보정 후** (cam_t + keypoints_3d[WRIST]) | 130.92 / 120.35 mm | 149.88 / 141.15 mm |
| 차이 | −22.5 / −24.7 mm | −20.1 / −33.4 mm |

보정으로 절대오차 수치 자체는 20~33mm 정도 줄었지만(그래서 보정이 꼭 필요했다), GT-focal
대비 24mm 하드코딩이 더 나쁘다는 **방향성 자체는 보정 전/후 모두 동일**하다 — 오프셋이
두 focal_length 조건에 똑같이 더해지는 상수라서 GT-vs-하드코딩 비교 자체의 결론(1번
질문에 대한 결론)은 바꾸지 않는다.
