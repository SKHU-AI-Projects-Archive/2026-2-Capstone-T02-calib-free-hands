# 정성 데모 — 카메라 초점거리 처리가 손의 절대 위치에 미치는 영향

**가장 먼저 볼 파일: `BEST_COMPARISON.mp4`** (= `comparison/2_screw_comparison.mp4`)

이 폴더는 새로운 성능 실험이 아닙니다. CAM-EXP-001~004.1에서 수행한 카메라
캘리브레이션 연구가 실제 데모 영상에서 **눈에 어떻게 보이는지**를 확인하기 위한
정성(qualitative) 시각화입니다.

---

## 1. 무엇을 비교하는가

같은 영상, 같은 손 예측에서 **카메라 초점거리(focal)만** 바꿨을 때
카메라 좌표계에서 손의 절대 위치가 어떻게 달라지는지를 좌우로 나란히 보여줍니다.

| | 왼쪽 | 오른쪽 |
|---|---|---|
| 라벨 | **LEGACY — pipeline focal convention** | **CURRENT CALIBRATION ESTIMATE** |
| 초점거리 | 기존 파이프라인이 쓰는 값 | 본 연구의 캘리브레이션 모델 추정치 |
| 2_screw.mp4 | 7,500 px | 1,739 px (Frozen E2) |
| 1_insert.mp4 | 7,500 px | 1,662 px (Frozen E2) |

화면 구성은 기존 데모(`demo/demo_hand_mano.py`의 `stage4_visualize`)와 같은
**왼쪽 RGB + 2D 스켈레톤 / 아래 3D 스켈레톤** 배치를 그대로 따랐고,
관절 연결·손가락 색·좌우 손 색·표시 좌표 변환은 `demo/hand_topology.py`를
그대로 재사용했습니다.

## 2. LEGACY의 5000-style focal이란 무엇인가

기존 파이프라인은 초점거리를 **측정하거나 캘리브레이션에서 읽지 않습니다.**
`rgb_predictor.py:407-411`에서

```
scaled_focal = FOCAL_LENGTH / IMAGE_SIZE * max(W, H)   # 1000 / 256 * max(W,H)
```

로 계산되는 **학습 규약상의 가상 초점거리(training-convention virtual focal)**이며,
원본 전체 이미지 픽셀 단위입니다. 1280×720이면 5000 px, 이 데모의 1920×1080
영상에서는 **7500 px**가 됩니다 (하드코딩하지 않고 같은 식으로 계산).

이 숫자를 "틀린 카메라 초점거리"라고 부르지 않습니다. 이것은 약-원근
(weak-perspective) → 카메라 이동량 변환 안에서 쓰이는 **규약**입니다.

## 3. CURRENT estimate란 무엇인가

* **AnyCalib video estimate** — `anycalib_gen` + `radial:2`로 영상 전체에서
  균등 샘플링한 8프레임을 추정한 뒤 median (CAM-EXP-004에서 고정한 규칙).
* **FROZEN E2 — exploratory ensemble** — 프레임별
  `sqrt(f_AnyCalib × f_GeoCalib)`의 median.
  정의는 `experiments/manifests/cam_exp_004_e2_frozen_spec_v1.json` 그대로이며
  **바꾸지 않았습니다.**

E2는 **frozen exploratory ensemble**입니다. our method / final method /
proposed method / ground-truth calibration이 **아닙니다.**

프레임을 8장으로 정한 것은 CAM-EXP-004.1에서 GigaHands 기준 8→64프레임 추가
개선이 없었기 때문에 채택한 **연산 정책**일 뿐, 이 샘플 영상에서도 8이
최적이라는 주장은 아닙니다.

## 4. 왜 손 예측을 고정하는가

초점거리 변화의 영향만 분리해서 보기 위해서입니다. 손 추론은 **영상당 1회만**
돌리고 캐시한 뒤, 두 조건이 같은 결과를 씁니다. 파이프라인 자신의 식
(`rgb_predictor._cam_crop_to_full`)이

```
tz = 2f / (s · box_size)      → tz(f) = tz_legacy · f / f_legacy
tx, ty                        → focal이 전혀 들어가지 않음
```

이므로, 초점거리만 바꾸는 것은 재추론 없이 **정확히** 계산됩니다.

실제로 동일한지는 `metadata/<video>_focal_isolation_audit.csv`에 프레임 단위로
기록했습니다. 예 (`2_screw`, frame 0):

| field | legacy | current | verdict |
|---|---|---|---|
| bbox | [123, 0, 634, 280] | [123, 0, 634, 280] | SAME |
| detector score | 0.813449 | 0.813449 | SAME |
| root-relative joints | 2.486116409 | 2.486116409 | SAME (bit-identical) |
| kp2d | 10388.341797 | 10388.341797 | SAME |
| focal [px] | 7500.00 | 1740.60 | DIFFERENT |
| cam_t tx [m] | −0.142856 | −0.142856 | SAME (tx에 focal 없음) |
| cam_t tz [m] | 2.309239 | 0.535930 | DIFFERENT (tz ∝ focal) |
| wrist absolute Z [m] | 2.315434 | 0.542124 | DIFFERENT |

## 5. 무엇을 볼 수 있는가

**카메라 좌표계에서의 절대 깊이(Z) 배치**가 얼마나 달라지는지입니다.
3D 패널에서 흰 점이 카메라 원점이고, 점선은 카메라에서 손목까지의 시선입니다.

손목 root Z의 중앙값:

| 영상 | 손 | LEGACY | Frozen E2 | 비율 |
|---|---|---|---|---|
| 2_screw | Left | 2.200 m | 0.515 m | 4.27× |
| 2_screw | Right | 1.171 m | 0.276 m | 4.24× |
| 1_insert | Left | 0.965 m | 0.219 m | 4.41× |
| 1_insert | Right | 1.762 m | 0.395 m | 4.46× |

`<video>_root_depth.csv`와 `<video>_root_depth_comparison.png`에 시계열이
있습니다.

**2D 오버레이는 양쪽이 완전히 동일합니다.** 투영식에서 초점거리가 상쇄되기
때문입니다(CAM-EXP-002 `focal_usage_audit.md` §2). 2D만 봐서는 이 문제를
발견할 수 없다는 점이 이 데모가 보여주는 것 중 하나입니다.

## 6. 무엇을 알 수 없는가

**이 샘플 영상들에는 데이터셋이 제공하는 카메라 캘리브레이션이 없습니다.**
따라서 이 영상만으로는 **어느 쪽이 실제로 맞는지 증명할 수 없습니다.**

* "ground truth focal", "correct focal", "true focal", "corrected 3D",
  "accurate 3D" 같은 표현은 이 폴더 어디에도 쓰지 않습니다.
* 절대 깊이가 4배 줄었다는 것은 **차이**이지 **정확도 개선의 증거가 아닙니다.**
* 정량 평가는 GT 캘리브레이션이 있는 GigaHands에서 수행한
  CAM-EXP-002/003/003.1/004/004.1이며, 그 결론은 이 폴더가 아니라
  `experiments/report_prep/progress_report_v1/`에 있습니다.

또한 현재 손 파이프라인이 실제로 소비하는 캘리브레이션 파라미터는
**초점거리뿐**입니다.

| 파라미터 | 현재 파이프라인에서 |
|---|---|
| focal | 사용됨 — `tz = 2f/(s·box_size)` 에서만 |
| principal point | **사용 안 함** — `rgb_predictor`는 이미지 중심 (W/2, H/2)을 씀 |
| distortion (k1, k2) | **사용 안 함** — 캘리브레이션 모델은 예측하지만 손 3D로 전파되지 않음 |

AnyCalib/GeoCalib이 principal point와 왜곡을 예측한다고 해서 이 데모가 그것을
적용한 것은 아닙니다.

## 7. world/global 좌표계가 아닙니다

3D 패널은 **CAMERA SPACE**입니다: X = 이미지 오른쪽, Y = 이미지 아래,
Z = 카메라 전방(깊이), 원점 = 카메라, 단위 = 미터.
표시할 때만 `hand_topology.RAW_TO_VIEW`로 Z-up 화면 좌표로 회전합니다
(기존 뷰어와 동일). 이 영상에는 workspace/world/robot 좌표 변환이 없으므로
**global이나 world 좌표라고 부르지 않습니다.**

---

## 8. 샘플 영상과 카메라 움직임

정적 카메라 집계는 CAM-EXP-004가 다룬 설정이므로, 카메라가 움직이는 클립은
삭제하지 않고 범위 밖으로 기록했습니다. 판정은 균등 샘플링한 프레임 사이의
배경 특징점 이동량 중앙값으로 했습니다(`sample_probe.csv`).

| 영상 | 해상도 | 길이 | 배경 이동(중앙값) | 판정 | 처리 |
|---|---|---|---|---|---|
| 1_insert.mp4 | 1920×1080 | 31.8 s | 0.81 px | MOSTLY_STATIC | 전체 렌더링 |
| 2_screw.mp4 | 1920×1080 | 12.1 s | 0.22 px | **STATIC_CAMERA** | 전체 렌더링 |
| 3_weld.mp4 | 1080×1920 | 58.7 s | 30.41 px | MOVING_CAMERA | `OUT_OF_SCOPE_FOR_STATIC_AGGREGATION` |
| 4_assemble.mp4 | 1080×1920 | 19.0 s | 14.83 px | MOVING_CAMERA | 동일 |
| 5_fold.mp4 | 1634×908 | 16.7 s | 17.52 px | MOVING_CAMERA | 동일 |

움직이는 세 클립도 캘리브레이션 추정치는 기록했지만(`calibration/`),
영상 하나에 하나의 초점거리를 주는 것이 타당한 설정이 아니므로 렌더링하지
않았습니다.

## 9. BEST_COMPARISON.mp4를 고른 이유

`2_screw.mp4`입니다. 선정 기준은 **녹화 속성만** 사용했습니다
(`metadata/best_comparison_choice.json`):

1. `STATIC_CAMERA` — 이 연구가 다루는 설정
2. 양손이 계속 검출됨 (364/364 프레임 bimanual)
3. 12초로 짧아 전체를 볼 수 있음

**쓰지 않은 기준**: 초점거리 추정치가 얼마나 나왔는지, 깊이 변화가 얼마나
보기 좋은지, 그리고 (애초에 존재하지 않는) ground truth.

## 10. GeoCalib 재현성 주의

GeoCalib은 동일 입력에서도 실행마다 출력이 달라집니다(CAM-EXP-004.1에서
40프레임×3회 반복 측정: median spread 1.18 %, p90 6.85 %, max 41.5 %).
이 폴더의 값은 **한 번 실행한 결과를 그대로 쓴 것**이며, 여러 번 돌려 보기 좋은
실행을 고르지 않았습니다. 프레임별 예측은
`calibration/<video>_calibration_frames.csv`에 남아 있습니다.

## 11. 파일 구조

```
qualitative_research_demo/
├─ BEST_COMPARISON.mp4          ← 가장 먼저 볼 영상 (2_screw)
├─ README.md                    이 문서
├─ summary.csv                  영상별 요약 (GT가 없으므로 error 컬럼 없음)
├─ sample_probe.csv             해상도/fps/코덱/카메라 움직임 진단
├─ comparison/<v>_comparison.mp4   LEGACY | CURRENT 2×2 비교 (핵심)
├─ legacy/<v>_legacy.mp4           단일 조건
├─ anycalib/<v>_anycalib.mp4       단일 조건
├─ e2/<v>_e2.mp4                   단일 조건
├─ comparison_frames/<v>_tNNNNN.png   시작/중간/끝 정지 이미지
├─ <v>_root_depth.csv              프레임별 손목 Z (legacy/anycalib/e2)
├─ <v>_root_depth_comparison.png   위 시계열 그래프
├─ calibration/                    프레임별 추정치 + 영상 단위 집계
├─ metadata/                       영상별 메타 + focal isolation audit
├─ scripts/                        재생성용 스크립트
└─ cache/                          손 추론 캐시 (git 제외)
```

영상·정지 이미지·캐시는 용량이 커서 git에 넣지 않습니다(재생성 가능).
스크립트·CSV·JSON은 커밋됩니다.

## 12. 재생성 방법

```powershell
# 1) 샘플 조사 + 카메라 움직임 진단
experiments\.venv\Scripts\python.exe samples\qualitative_research_demo\scripts\probe_samples.py

# 2) 캘리브레이션 (AnyCalib + GeoCalib + frozen E2)
experiments\.venv-calib\Scripts\python.exe samples\qualitative_research_demo\scripts\calibrate_videos.py

# 3) 손 추론 1회 (영상당) — 기존 demo/demo_hand_mano.py 재사용
experiments\.venv-anyhand\Scripts\python.exe samples\qualitative_research_demo\scripts\run_hand_inference.py --only 2_screw

# 4) 렌더링 (재추론 없음)
experiments\.venv\Scripts\python.exe samples\qualitative_research_demo\scripts\render_demo.py --video 2_screw.mp4

# 5) 요약 + BEST_COMPARISON 선정
experiments\.venv\Scripts\python.exe samples\qualitative_research_demo\scripts\build_summary.py
```

원본 샘플 영상(`samples/*.mp4`)은 수정하지 않습니다.
`experiments/runs/*`와 `experiments/report_prep/*`의 수치 결과도 손대지
않았습니다.

## 13. 재사용한 기존 코드

| 재사용한 것 | 위치 |
|---|---|
| 모델 로딩, 검출, MANO 피팅 | `demo/demo_hand_mano.py` — `load_predictor`, `stage1_detect`, `stage2_fit_mano`, `iter_frames`, `empty_record` |
| 절대 카메라 좌표 변환 | `demo/demo_hand_mano.py` — `stage3_to_camera_space`와 동일한 `keypoints_3d + cam_t` |
| 3D 패널 구성·고정 축 범위 | `demo/demo_hand_mano.py` — `stage4_visualize`, `view_limits` 방식 |
| 21점 연결·손가락 색·좌우 손 색·표시 좌표 변환 | `demo/hand_topology.py` — `HAND_CONNECTIONS`, `joint_color`, `HAND_BASE_COLORS`, `raw_to_view` |
| 초점거리 규약 | `rgb_predictor.py:407-411`, `_cam_crop_to_full` |
| 초점거리 의미 감사 | `experiments/runs/CAM-EXP-002_camera_focal_sensitivity/focal_usage_audit.md` |
| E2 정의 | `experiments/manifests/cam_exp_004_e2_frozen_spec_v1.json` |
