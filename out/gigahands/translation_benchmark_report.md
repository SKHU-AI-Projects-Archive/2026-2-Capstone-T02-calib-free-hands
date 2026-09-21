# GigaHands Translation / Inter-hand Distance Benchmark — GT focal vs 24mm 하드코딩

## Executive Summary (후속 분석 A/B/C 종합)

최초 버전(아래 본문) 이후 세 가지 후속 검증을 마쳤다. 상세는
`wrist_offset_validation.md`, `interhand_dz_significance.md`,
`bias_investigation_report.md` 참고.

1. **A) wrist offset(±9.5cm) 검증 — 고정 상수 맞음.** 1,846개 인스턴스 전체에서
   norm 기준 평균 96.01mm, 상대표준편차 **0.17%**(<5% 임계값) — 손·화각그룹·카메라
   어느 축으로 나눠도 사실상 불변. 보정 전(cam_t만) 대비 보정 후(cam_t+wrist)로
   절대오차가 20~33mm 줄었지만, GT-focal이 24mm 하드코딩보다 낫다는 **결론의 방향은
   바뀌지 않는다.**

2. **B) 양손 dz 차이(4.1mm) — 통계적으로 유의하지만 크기는 작다.** Paired t-test·
   Wilcoxon 둘 다 p<0.0001, 95% CI [3.33, 4.92]mm로 0을 포함하지 않아 **유의함**은
   확실하다. 다만 개별 dz_err의 산포(표준편차 ~130mm대)에 비하면 4mm는 작다 —
   "유의함"과 "크다"는 서로 다른 말이며, 이 경우 **유의하지만 작다**가 정확한 결론이다.

3. **C) 130.9mm 절대오차의 원인 — 세 가설 중 어느 것도 절대오차 크기(분산)의
   대부분을 설명하지 못한다.** bbox 오차(YOLO bbox가 이론치보다 평균 **43% 큼**)와
   실측 촬영거리(GigaHands 평균 733mm)는 편향의 **방향**(GT-focal에서도 평균
   −33.7mm 과소추정)의 약 **45%**를 합쳐서 설명하지만, 절대오차 **크기**(분산)는
   세 변수를 다 합쳐도 **2%**밖에 설명 못한다 — 즉 편향의 존재/방향은 상당 부분
   설명되나, 130mm대라는 오차 규모 자체는 이번 조사로 다 설명되지 않는 인스턴스
   단위의 큰 잡음이 대부분이다(추가 검증 필요).
   - **부록(라이브 질문에서 파생)**: 파이프라인이 실제 기본으로 쓰는 focal(모델
     config 기반 5000px)은 GigaHands 실측 focal(평균 914px)보다 5.5배 이상 커서,
     그대로 두면 절대오차가 **평균 2.84m**로 폭증한다 — GT-focal vs 24mm 하드코딩의
     차이(14.5%)는 이 기본값 대비 개선(약 20배)에 비하면 훨씬 작은 2차적 문제다.

---

**대상**: `eval_pa_mpjpe_gigahands.py` baseline과 완전히 동일한 표본 — GigaHands
`p005-tea` + `p005-noodle-fastfood`, 25/26/27mm 화각 그룹 계층 샘플링, seed=0,
**1,500프레임 / GT 인스턴스 2,804개** (실행 시 baseline JSON과 total_samples·
gt_hand_instances 값을 교차확인해 일치를 확인함). 검출·매칭 성공 인스턴스는 **1,846개**
(baseline의 `matched_instances`와 동일), 양손 동시 검출 프레임은 **696개**
(baseline의 `handedness_diagnostic.n_dual_hand_frames`와 동일).

**PA-MPJPE와의 차이**: PA-MPJPE는 Procrustes 정렬(회전+이동+스케일 제거) 후 오차라서
focal_length 오차의 영향(주로 cam_t, 특히 깊이)을 구조적으로 반영하지 못한다. 이 실험은
정렬 없이 cam_t(및 파생 지표)를 GT와 직접 비교해 그 빈틈을 메운다.

## 방법 요약

- WiLoR forward pass는 baseline과 마찬가지로 **인스턴스당 1회만** 실행. `rgb_predictor._cam_crop_to_full`
  호출 시점에 raw 입력(`pred_cam`, `box_center`, `box_size`, `img_size`)을 몽키패치로
  가로채 저장해두고, 이후 GPU 없이 이 함수를 **같은 raw 값으로 focal_length만 바꿔 2회
  재호출**해서 `cam_t_GT`, `cam_t_24mm` 를 얻었다.
- **절대 위치 비교 시 보정 사항**: `cam_t` 자체는 절대 손목 위치가 아니다 —
  `keypoints_3d[WRIST]`(root-relative)이 실측상 정확히 0이 아니라 손마다 고정 오프셋
  (~±9.5cm, focal_length와 무관)을 갖고 있음을 디버그로 확인했다. 지시서는 "cam_t를
  GT와 직접 비교"라 했으나, 그대로 하면 focal_length와 무관한 고정 오프셋이 오차에
  섞여 실험 취지(focal_length 효과 분리)를 훼손하므로, 파이프라인 기존 규약
  (`demo_hand_mano.stage3_to_camera_space`: `keypoints_3d + cam_t` = 절대 위치)대로
  `cam_t + keypoints_3d[WRIST]`를 절대 예측 위치로 사용했다.
- GT 절대 위치 = GigaHands `keypoints_3d`의 손목(index 0) world 좌표를, 해당 카메라의
  실측 extrinsic(`gigahands_optim_params_sample.txt`의 qvec/tvec, COLMAP 관례
  `X_cam = R·X_world + t`)으로 카메라 좌표계 변환. 변환 방향은 실행 시 z(깊이)가
  타당 범위(0.05~5m)에 드는지로 자동 검증(임의 샘플: z=0.224m, OK).
- GT focal = 해당 카메라의 실측 `(fx+fy)/2` (px, 이미 해당 해상도 기준). 24mm 하드코딩
  focal = `24 / 36mm(풀프레임 환산 기준) × 이 프레임의 실제 픽셀 폭`(하드코딩된 해상도
  없음, 매 프레임 캡처된 실제 폭 사용).
- 검증: (1) baseline과 표본 수·GT 인스턴스 수 교차확인 일치, (2) `matched_instances`/
  `n_dual_hand_frames` 일치, (3) 8개 인스턴스에 대해 `_cam_crop_to_full` 공식을 손으로
  재현해 스크립트 출력과 `atol=1e-4`로 일치 확인 — 모두 통과.

## 1) 절대 translation 오차 (mm)

세 가지 focal_length 조건을 나란히 비교한다: **GT focal**(카메라 실측값), **24mm
하드코딩**(35mm 환산, 카메라 파라미터를 전혀 모를 때의 대체 가정), **파이프라인 기본값**
(`model_config_wilor.yaml`의 `EXTRA.FOCAL_LENGTH=1000, MODEL.IMAGE_SIZE=256`에서 나오는
`scaled_focal = 1000/256×max(W,H) = 5000px` — 지금 이 코드가 아무 override 없이 그냥
돌아갈 때 실제로 쓰는 값).

| | GT focal | 24mm 하드코딩 | **파이프라인 기본값(5000px)** |
|---|---:|---:|---:|
| 전체 절대오차(평균) | 130.9 | 149.9 | **2,840.5** |
| 전체 절대오차(중앙값) | 120.4 | 141.2 | **3,097.5** |
| z(깊이) 오차(평균) | 94.5 | 112.5 | **2,838.6** |
| z(깊이) 오차(중앙값) | 82.7 | 112.3 | **3,096.0** |

GT-focal → 24mm 하드코딩: +19.0mm(+14.5%, 평균) / +20.8mm(+17.3%, 중앙값), z가 그
증가분의 약 95%를 차지 — "focal_length 오차는 z 방향에 집중된다"는 예상과 **일치**한다.

**그런데 파이프라인 기본값(5000px)은 차원이 다르다.** GigaHands 카메라 실측 focal(평균
913.8px)보다 **5.5배** 커서, 절대오차가 GT-focal 대비 **약 21.7배**(130.9→2,840.5mm),
24mm 하드코딩 대비로도 **약 19배**(149.9→2,840.5mm) 폭증한다. 부호(아래 C1/부록,
`bias_investigation_report.md`)도 GT/24mm 케이스(과소추정, 음수)와 반대로 **큰 폭의
과대추정(+2,838.6mm)** — focal이 실제보다 훨씬 크면 `tz=2f/denom`가 그만큼 커져 "훨씬
멀리 있다"고 보게 되는 공식과 정확히 일치한다.

화각 그룹별:

| 그룹 | 카메라 수 | 절대오차 GT(평균) | 절대오차 24mm(평균) | 절대오차 파이프라인기본값(평균) |
|---|---:|---:|---:|---:|
| 25mm | **2** | 131.1 | 172.1 | **3,621.3** |
| 26mm | **2** | 133.3 | 126.3 | **1,649.3** |
| 27mm | **1** | 125.7 | 137.5 | **3,156.7** |

→ GT-vs-24mm 비교에서는 그룹별 증가율이 "24mm에서 더 멀리 벗어난 그룹일수록 더 커진다"는
단순한 단조 패턴이 **아니었다**(26mm 그룹은 오히려 하드코딩 쪽이 더 낮음, 대표성 한계는
3번 참고). 하지만 **파이프라인 기본값은 세 그룹 전부에서, GT/24mm 두 대안보다 압도적으로
나쁘다** — 그룹별 카메라 개별 편차(26mm의 역전 현상 등)를 완전히 압도할 만큼 오차 규모
차이가 크다는 뜻이다.

## 2) 양손 간 상대 거리 오차 (mm)

| | GT focal | 24mm 하드코딩 | **파이프라인 기본값(5000px)** |
|---|---:|---:|---:|
| dx 오차(평균) | −2.60 | −2.60 | **−2.60 (동일)** |
| dy 오차(평균) | −30.67 | −30.67 | **−30.67 (동일)** |
| dz 오차(평균) | −3.99 | −8.11 | **+283.4** |
| 전체 유클리드 오차(평균) | 107.4 | 105.6 | **579.7** |

→ **dx/dy는 세 조건 전부에서 소수점까지 완전히 동일** — 예상("dx,dy는 focal_length와
거의 무관")과 **정확히 일치**하며, 파이프라인 기본값(focal이 5.5배 다른 극단적 케이스)을
포함해도 이 불변성은 깨지지 않는다. `_cam_crop_to_full` 공식이애초에 `tx,ty`에
focal_length를 포함하지 않아 **수학적으로 자명하게 보장되는 결과**임을 다시 한번
확인한다.

**dz는 파이프라인 기본값에서 완전히 다른 이야기가 된다.** GT-focal(−3.99mm)·24mm
하드코딩(−8.11mm)은 둘 다 작은 음수(약간의 과소추정 방향)인 반면, 파이프라인 기본값은
**+283.4mm**(부호도 반대, 큰 과대추정) — 전체 유클리드 오차도 107~106mm대에서
**579.7mm**로 5배 이상 뛴다. 두 손의 "상대 위치"조차 파이프라인 기본값에서는 거의 못 쓸
수준으로 어긋난다.

화각 그룹별 (dz 오차, 표본 수 참고) — GT/24mm는 기존과 동일:

| 그룹 | n | dz오차 GT(평균) | dz오차 24mm(평균) |
|---|---:|---:|---:|
| 25mm | 403 | −1.9 | −7.6 |
| 26mm | 240 | +5.5 | +7.2 |
| 27mm | 53 | −62.5 | −81.7 |

(파이프라인 기본값의 그룹별 dz는 계산했으나, 전체 평균(+283.4mm)이 이미 GT/24mm 대비
압도적으로 커서 그룹 간 비교의 실익이 낮다 — `eval_interhand_distance_gigahands.csv`의
`dz_err_pipeline_mm` 컬럼에서 직접 확인 가능.)

## 3) 표본 대표성 — 화각 그룹별 카메라 다양성

| 그룹 | 실제 사용된 카메라 종류 수 |
|---|---:|
| 25mm | 2대 |
| 26mm | 2대 |
| 27mm | **1대** |

세 그룹 다 물리적 카메라 1~2대에 표본이 몰려 있다. baseline에서 지적됐던 "27mm 그룹은
카메라 1대·장면 1개 편중" 문제가 이번 실험에서도 그대로이고, **25mm/26mm도 별로 낫지
않다(각 2대뿐)**. 위 2)의 "26mm 그룹만 하드코딩이 더 정확해지는" 역전 현상이나 1)의
"25mm 그룹이 가장 크게 나빠지는" 비단조 패턴은, 화각 그룹이라는 모집단 속성이 아니라
그 그룹을 대표하는 1~2대 카메라의 개별적 특성(개별 캘리브레이션 오차, 렌즈 왜곡 등)에서
비롯됐을 가능성이 높다 — 카메라 수를 늘리지 않는 한 그룹 간 비교는 통계적으로 약하다.

## 결론

1. **GT focal 대비 24mm 하드코딩의 평균 절대오차 증가폭**: +19.0mm (+14.5%), 중앙값
   기준 +20.8mm (+17.3%).
2. **이 증가는 주로 z축에 집중된다**: 맞다 — z오차 증가분(+18.0mm)이 전체 절대오차
   증가분(+19.0mm)의 약 95%를 차지한다.
3. **양손 간 거리 오차에서 dz만 벌어지는지, dx/dy도 영향받는지**: dx/dy는 focal_length
   선택과 **완전히 무관**(소수점까지 동일), dz만 영향을 받는다 — 예상과 정확히 일치하며,
   이는 `_cam_crop_to_full` 공식 구조상 수학적으로 보장되는 결과이기도 하다. 다만 dz의
   변화폭(~4mm)은 dz_err 자체의 잡음(표준편차 ~130mm)보다 작아, focal 선택의 실질적
   기여도는 제한적이다.
4. **화각 그룹별로 예상대로 커지는가, 표본 대표성 문제가 있는 그룹은 어디인가**: 전체
   평균으로는 이론과 일치하지만, 그룹별로는 단조 패턴이 아니다(26mm은 오히려 역전).
   세 그룹 전부(25mm 2대, 26mm 2대, 27mm 1대) 카메라 다양성이 낮아 표본 대표성이
   약하고, 특히 27mm이 가장 심하다 — 그룹별 비교는 참고용으로만 보는 게 안전하다.
5. **(추가) 파이프라인이 실제로 쓰는 기본값과 비교하면 위 1~4는 전부 2차적인 문제다**:
   `model_config_wilor.yaml` 기반 기본 focal(5000px)은 GigaHands 실측(평균 914px)보다
   5.5배 커서, 절대오차가 GT-focal 대비 약 21.7배(130.9→2,840.5mm) 폭증한다. 즉 GT-focal
   이냐 24mm 하드코딩이냐(14.5% 차이)를 고민하기 전에, **지금 코드가 기본으로 쓰고 있는
   focal 자체가 실사용 카메라에 전혀 맞지 않는다**는 게 훨씬 큰 실무적 문제다.

## 중요한 한계 (반드시 같이 읽을 것)

절대오차 자체가 GT-focal을 쓰고도 평균 130.9mm(중앙값 120.4mm)로 상당히 크다. 개별
인스턴스 디버그(부록 참고)에서, GT-focal을 정확히 넣어도 예측 깊이가 실제보다 수 미터
어긋나는 경우가 확인됐다 — 이는 focal_length 문제가 아니라 **WiLoR가 예측하는 크롭
스케일(`pred_cam`의 `s`)이 GigaHands의 실제 카메라-손 거리 분포에 대해 자체적으로
잘 보정돼 있지 않다**는 별개의 문제로 보인다. 즉 이번 실험이 측정한 "GT-vs-24mm 차이"는
실재하고 이론과도 방향이 맞지만, 그보다 훨씬 큰 절대오차(스케일 예측 자체의 편향)
위에 얹힌 상대적으로 작은 추가 효과라는 점을 함께 봐야 한다.

## 산출물

- `eval_translation_gigahands.csv` (1,846행): scene, seq, cam, frame, group, side,
  abs_error_gt_focal_mm, abs_error_hardcoded_mm, z_error_gt_focal_mm, z_error_hardcoded_mm,
  **abs_error_pipeline_mm, z_error_pipeline_mm**(이번에 병합 — 파이프라인 기본값 기준)
  (요청 스키마에 `seq`, `side` 두 컬럼을 추가 — 같은 frame에 좌/우 두 인스턴스가 있어
  구분에 필요했음)
- `eval_interhand_distance_gigahands.csv` (696행): scene, seq, cam, frame, group,
  dx/dy/dz_err_gt_focal_mm, euclid_err_gt_focal_mm, dx/dy/dz_err_hardcoded_mm,
  euclid_err_hardcoded_mm, **dx/dy/dz_err_pipeline_mm, euclid_err_pipeline_mm**(이번에 병합)
- `eval_translation_summary.json`: 위 표들의 원본 집계 (GT-focal/24mm/파이프라인 기본값
  3-way, overall + 화각 그룹별)
- `gigahands_raw_instances_cache.csv` (1,846행): raw per-instance 값(pred_cam, wrist
  offset, bbox 등) + 세 focal 조건 각각의 cam_t·오차 — A/B/C 후속분석과 이번 병합의
  공통 원천 데이터, 재추론 없이 재사용 가능
- `eval_translation_gigahands.py`, `capture_raw_instances_gigahands.py`: 재실행 가능한
  스크립트 (`--target-per-group`로 표본 크기 조절 가능, 기본 500=baseline과 동일)
- `wrist_offset_validation.md`, `interhand_dz_significance.md`,
  `bias_investigation_report.md`: A/B/C 후속분석 상세 보고서
