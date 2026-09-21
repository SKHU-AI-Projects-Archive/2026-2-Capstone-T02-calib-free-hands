# GeoCalib_영상평균 — 프레임별 지터를 영상 단위 평균으로 줄였을 때의 절대오차

같은 영상((scene,seq,cam) 하나 = 카메라 1대의 녹화 파일 하나) 안에서 샘플링된 프레임들의
GeoCalib 추정 focal 평균 하나를 그 영상 전체 인스턴스에 고정 적용한 조건. WiLoR 재추론 없음
(`gigahands_raw_instances_cache.csv`의 raw 값 + `geocalib_focal_seq_stats.csv`의 시퀀스별
평균만으로 `_cam_crop_to_full` 후처리를 재계산). 기존 5-way 비교(`geocalib_comparison_report.md`)
와 완전히 동일한 표본(1,846 인스턴스, seed=0).

## 1. 시퀀스별 GeoCalib 추정 focal 의 원래 흩어짐 (평균 내기 전)

- 영상(=`(scene,seq,cam)`) 총 326개, 그중 프레임 2개 이상(표준편차가 의미
  있는 경우) 312개.
- 영상별 focal_x_geocalib 표준편차: 평균 33.0px, 중앙값 23.8px,
  최대 420.2px, 최소 1.0px.
- 영상 단위로 평균 낸 값 하나를 쓰면, 정의상 이 프레임 간 표준편차는 0이 된다(같은 영상의
  모든 인스턴스가 동일한 focal 을 공유하게 되므로) — "지터 제거"는 focal 값 자체에는 완전히
  적용되고, 그 결과가 절대오차에 실질적으로 도움이 되는지는 아래 2절에서 본다.
- 상세: `geocalib_focal_seq_stats.csv` (326행: scene, seq, cam, n_frames,
  focal_x_geocalib_mean/std/min/max)

## 2. GeoCalib_영상평균 자체의 focal 추정 정확도

| | GeoCalib_영상평균 | GeoCalib(프레임별, 기존) |
|---|---:|---:|
| focal_px 평균 | 1191.7 | (기존 보고서: 1,174.6) |
| GT 대비 상대오차(signed, 평균) | +30.59% | (기존 보고서: +28.35%) |
| GT 대비 상대오차(signed, 중앙값) | +19.54% | (기존 보고서: +18.70%) |
| 상대오차 표준편차 | 24.47%p | (기존 보고서: 24.09%p) |

## 3. 절대 translation 오차 (mm) — 6-way 비교

`geocalib_comparison_report.md`의 종합 비교표에 이번 조건을 한 행 추가:

| 조건 | focal_px 평균 | GT 대비 상대오차(%) | 절대 translation 오차 평균(mm) | 절대오차 중앙값(mm) | z오차 평균(mm) |
|---|---:|---:|---:|---:|---:|
| GT focal | 913.8 | 0% | 130.9 | 120.4 | 94.5 |
| 24mm 하드코딩 | 853.3 | −6.9% | 149.9 | 141.2 | 112.5 |
| 파이프라인 기본값 | 5,000.0 | +445.5% | 2,840.5 | 3,097.5 | 2,838.6 |
| GeoCalib(프레임별) | 1,174.6 | +28.3% | 240.4 | (기존 보고서 미기재) | 210.6 |
| **GeoCalib_영상평균** | **1191.7** | **+30.6%** | **239.4** | **181.1** | **209.7** |

(참고용 — 이번에 재계산한 프레임별 GeoCalib 조건의 절대오차: 평균 240.4mm,
중앙값 184.5mm, 표준편차 186.0mm / 영상평균 조건:
표준편차 184.6mm — 프레임별 대비 영상평균의 절대오차 표준편차가
줄었다.)

### 화각 그룹별 절대오차 평균(mm)

| 그룹 | n | GeoCalib_영상평균 |
|---|---:|---:|
| 25mm | 872 | 327.7 |
| 26mm | 656 | 150.9 |
| 27mm | 318 | 180.1 |

## 4. 양손 간 상대거리 오차 (mm)

| | GeoCalib_영상평균 |
|---|---:|
| dx 오차(평균) | -2.60 |
| dy 오차(평균) | -30.67 |
| dz 오차(평균) | 33.22 |
| 전체 유클리드 오차(평균) | 131.2 |

(기존 조건들과 비교하려면 `translation_benchmark_report.md`/`geocalib_comparison_report.md`의
동일 표 참고 — GT −3.99mm, 24mm −8.11mm, 파이프라인 기본값 +283.4mm, 전체 유클리드는 각각
107.4/105.6/579.7mm.)

## 5. 결론

GeoCalib_영상평균의 절대오차는 평균 239.4mm — 기존 프레임별 GeoCalib(240.4mm)
보다 개선됐다. 다만 focal 추정
자체의 GT 대비 편향(+30.6% 근방)은 프레임을 평균 내도 구조적으로 남는다 —
영상 단위 평균은 "프레임 간 무작위 흔들림(분산)"만 줄일 뿐, GeoCalib이 이 도메인(GigaHands
근접 손 클로즈업)에서 갖는 **체계적 편향(bias)**은 그대로다. 즉 이 실험은 "GeoCalib을 더 안정
적으로 쓰는 방법"에 대한 답이지, "GeoCalib의 정확도 자체를 개선하는 방법"에 대한 답은 아니다
— 근본 원인(도메인 불일치로 인한 편향)은 `geocalib_comparison_report.md` 4절의 논의가 여전히
유효하다.

## 산출물

- `geocalib_focal_seq_stats.csv` (신규, 326행): (scene,seq,cam)별 GeoCalib
  focal 평균/표준편차/min/max (`run_geocalib_gigahands.py --seq-stats-only`로 재생성 가능,
  GeoCalib 재추론 없음)
- `gigahands_raw_instances_cache.csv`: `focal_geocalib_seqavg`, `cam_t_geocalib_seqavg_{x,y,z}`,
  `pred_wrist_geocalib_seqavg_{x,y,z}`, `abs_error_geocalib_seqavg_mm`, `z_error_geocalib_seqavg_mm`,
  `signed_z_err_geocalib_seqavg_mm` 컬럼 추가
- `eval_translation_gigahands.csv`: `abs_error_geocalib_seqavg_mm`, `z_error_geocalib_seqavg_mm` 컬럼 추가
- `eval_interhand_distance_gigahands.csv`: `dx/dy/dz_err_geocalib_seqavg_mm`,
  `euclid_err_geocalib_seqavg_mm` 컬럼 추가
- `eval_translation_gigahands.py --recompute-geocalib-seqavg`: 재실행 가능(GPU/재추론 불필요)
