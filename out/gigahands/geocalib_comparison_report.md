# GeoCalib focal 추정 — GigaHands 4-way 비교 (GT / 24mm / 파이프라인 기본값 / GeoCalib)

## 0. 환경 준비 (안전 점검 요약)

- venv: 기존 `/home/juson/project/hand-demo/.venv` 그대로 사용 (새 venv 생성 없음)
- `requirements_before_geocalib.txt`: 설치 전 스냅샷(76개 패키지) 저장 완료
- **충돌 1건 발견 후 설치 중단·보고**: GeoCalib 요구 `opencv-python`(버전 미고정)이 Python≥3.9에서
  **numpy≥2를 강제**(PyPI 메타데이터 확인) — 파이프라인이 `numpy==1.26.4`로 의도적으로 고정해둔
  것과 충돌. `opencv-python` 설치를 제외하고(이미 있는 `opencv-python-headless`로 대체) numpy/torch/
  torchvision을 constraints 파일(`constraints_geocalib.txt`)로 고정해 우회 — 사용자 승인 후 진행.
- **충돌 2건 (신규 발견, 스모크 테스트로 잡힘)**: `kornia`(요구 버전 미고정) 최신판 0.8.3이
  `torch.jit.script`로 컴파일되는 `essential.py`의 `.any(dim=(-2,-1))` 호출에서 torch 2.1.2 미지원
  오버로드로 **import 자체가 RuntimeError로 실패**. `kornia==0.7.4`로 교체해 해결(numpy/opencv 영향 없음
  확인 후 설치).
- **스모크 테스트 전부 통과**: GeoCalib GPU 0 더미 이미지 추론 정상, 기존 `rgb_predictor`/
  `AnyHandPredictor` 로드 및 캐시 CSV 재로드 정상 — 기존 numpy/torch/torchvision/opencv 버전 불변 확인.
- GPU 0 단독 사용, 실행 중 다른 GPU 사용 없음. 최고 온도 46°C(추론 중 순간)로 안전.

## 1. 고유 프레임 추출

`eval_translation_gigahands.csv`의 1,846개 인스턴스 → **고유 프레임 1,150개**(같은 프레임에 좌/우
손이 같이 있는 696개 프레임은 자연히 중복 제거됨: 1,150 + 696 = 1,846 ✓). 영상 재다운로드 없이
기존 `eval_pa_mpjpe_gigahands.py`의 `load_video_map`/`load_downloaded_video_index`/`extract_frames`를
그대로 재사용해 326개 영상에서 필요한 프레임만 디코드했다 — 경로 해석 실패 0건.

## 2. GeoCalib 추론

- 1,150개 고유 프레임 각각에 대해 1회씩 GeoCalib(`weights="pinhole"`) 추론, GPU 0.
- **총 소요시간: 304.1초** (영상 디코드 포함), 프레임당 순수 모델 추론 평균 **184.2ms**.
- 디코드 실패 0건, 이상치(음수/0/이미지폭×3 초과/NaN·Inf) **0건** — 전부 정상 범위.
- 결과: `geocalib_focal_estimates.csv` (1,150행: scene, seq, cam, frame, focal_x/y_geocalib,
  img_width/height, infer_time_s)

## 3. 절대오차 재계산 (WiLoR 재추론 없음)

기존 `capture_raw_instances_gigahands.py`가 캐시해둔 `pred_cam`/`box_center`/`box_size`에
`focal_x_geocalib`만 새로 넣어 `_cam_crop_to_full` 공식을 오프라인 재계산(같은 프레임의 좌/우
손은 같은 focal_x_geocalib 공유). 5개 인스턴스를 손으로 재현해 스크립트 출력과 `atol=1e-4`로
**전부 일치** 확인. 결과: `eval_translation_geocalib.csv` (1,846행).

## 4. GeoCalib 자체의 focal 추정 정확도

| | GeoCalib | 24mm 하드코딩 |
|---|---:|---:|
| GT 대비 상대오차(signed, 평균) | **+28.35%** | −6.86% |
| GT 대비 상대오차(signed, 중앙값) | +18.70% | (고정값이라 그룹별 동일) |
| 상대오차 표준편차 | 24.09%p | — |
| focal_px 평균 | 1,174.6 (GT 916.6) | 853.3 (GT 916.6) |

**GeoCalib이 24mm 하드코딩보다 훨씬 부정확하다** — 평균적으로 실제보다 28%나 크게(더 망원으로)
추정한다. 앞서 논의했던 "GeoCalib의 일반적 FoV 오차 ~10-12%"와 비교하면 **이번 GigaHands 실측
결과(28~29%)가 그보다 2배 이상 크다.** 이 차이의 원인은 확실히 규명하지 못했지만(추가 검증
필요), 다음이 그럴듯한 가설이다: GeoCalib은 소실점·수평선·중력 방향 같은 장면의 기하학적
단서로 캘리브레이션하는 모델인데, GigaHands는 고정 리그로 촬영된 근접 손 클로즈업이라 이런
단서가 거의 없는 구도다 — 일반 사진(GeoCalib 자체 벤치마크의 전형적 구도)과 도메인이 많이
다를 수 있다.

## 5. 화각 그룹별

| 그룹 | GeoCalib 상대오차(평균) | 절대오차 GeoCalib(평균) | 절대오차 GT(평균) | 절대오차 24mm(평균) |
|---|---:|---:|---:|---:|
| 25mm | **+50.44%** | **329.5mm** | 131.1mm | 172.1mm |
| 26mm | +9.92% | 151.1mm | 133.3mm | 126.3mm |
| 27mm | +18.17% | 180.0mm | 125.7mm | 137.5mm |

25mm 그룹에서 GeoCalib 오차가 특히 크다(+50%) — 이 그룹은 앞선 실험에서도 "카메라 2대뿐"이라는
표본 대표성 문제가 지적됐던 그룹과 동일해서, 이 수치가 25mm 화각 자체의 일반적 특성인지 그
2대 카메라의 개별 특성인지는 구분하기 어렵다.

## 6. 종합 비교표

| 조건 | focal_px 평균 | GT 대비 상대오차(%) | 절대 translation 오차 평균(mm) | z오차 평균(mm) |
|---|---:|---:|---:|---:|
| GT focal | 913.8 | 0% | 130.9 | 94.5 |
| 24mm 하드코딩 | 853.3 | −6.9% | 149.9 | 112.5 |
| 파이프라인 기본값 | 5,000.0 | +445.5% | 2,840.5 | 2,838.6 |
| **GeoCalib 추정** | **1,174.6** | **+28.3%** | **240.4** | **210.6** |

(파이프라인 기본값 상대오차는 (5000−913.8)/913.8×100 로 참고 계산)

## 결론

1. **GeoCalib이 24mm 하드코딩보다 나은가, 나쁜가**: **나쁘다.** 절대오차 240.4mm는 24mm
   하드코딩(149.9mm)보다 60% 더 크고, GT-focal(130.9mm) 대비로는 84% 더 크다. 지금 이 데이터셋·
   이 카메라 구도에서는 "24mm이라고 대충 가정하는 것"이 "이미지를 보고 추정하는 것"보다 낫다.

2. **원인 구분**: 이건 **(a) GeoCalib 자체의 추정 오차가 크기 때문**이지, **(b) "focal
   오차가 전체 오차의 14.5%만 설명한다"는 사실 때문에 개선 여지가 작아서**가 아니다.
   근거: 만약 (b)가 주된 이유라면 GeoCalib의 focal 추정이 웬만큼 정확해도(예: GT 대비
   ±10% 이내) 절대오차가 GT-focal 수준(130.9mm)과 비슷하게 나왔어야 한다. 하지만 실제로는
   GeoCalib의 focal 추정 자체가 평균 +28.3%나 벗어나 있고(24mm 하드코딩의 −6.9%보다 4배
   더 부정확), 이 큰 입력 오차가 그대로 출력 오차(240.4mm)로 이어졌다. 즉 **이번 결과가
   나쁜 건 "focal을 알아도 소용없어서"가 아니라 "GeoCalib이 준 focal 자체가 부정확해서"다.**

3. **오버헤드**: GeoCalib을 앞단에 추가하면 프레임당 평균 184.2ms(영상 디코드 포함 시
   264ms) 추가 연산이 붙는다. WiLoR 자체 처리 속도(baseline 217초/1,500프레임 ≈ 프레임당
   145ms)와 비슷한 규모라, **파이프라인에 추가하면 프레임당 처리 시간이 대략 2배로 늘어난다.**
   그런데 정확도는 오히려 24mm 하드코딩보다 나빠졌으므로, **지금 이 상태로는 이 오버헤드가
   전혀 정당화되지 않는다** — GeoCalib을 실사용하려면 이 도메인(근접 손 클로즈업, 고정 리그)에
   대한 정확도 개선(파인튜닝 등)이 먼저 필요해 보인다.

## 산출물

- `requirements_before_geocalib.txt`, `constraints_geocalib.txt`
- `geocalib_focal_estimates.csv` (1,150행)
- `eval_translation_geocalib.csv` (1,846행)
- `gigahands_raw_instances_cache.csv`, `eval_translation_gigahands.csv`: GeoCalib 컬럼
  병합 완료(`focal_x_geocalib`, `abs_error_geocalib_mm`, `z_error_geocalib_mm` 등)
- `run_geocalib_gigahands.py`: 재실행 가능 스크립트
