# Camera Calibration Research

이 폴더는 제조업 고정 단안 RGB 영상의 camera calibration 및
absolute/global hand 3D reconstruction 연구 실험을 관리한다.

## Structure

```
experiments/
├─ datasets/        로컬 데이터셋 (Git 제외)
│  ├─ _downloads/   다운로드 원본 archive만 보관 (압축 해제 금지)
│  ├─ gigahands/  interhand26m/  hanco/  assemblyhands/  reinterhand/
├─ manifests/       데이터셋 구성/검증 정보 (Git 추적)
├─ src/             공통 실험 코드 (Git 추적)
│  ├─ geometry/     canonical pinhole camera model
│  ├─ datasets/     per-dataset loader (공통 Sample 타입으로 변환)
│  ├─ metrics/      reprojection error + 통계
│  └─ visualization/ GT vs projected overlay
├─ runs/            실험별 결과 (config/environment/results/figures/report)
└─ cache/           재생성 가능한 캐시 (Git 제외)
```

역할 분리 원칙: **다운로드 원본**(`datasets/_downloads/`), **압축 해제 데이터**
(`datasets/<name>/`), **manifest**(`manifests/`), **실험 결과**(`runs/`)를
절대 섞지 않는다. 데이터셋 본체와 archive는 Git에 올리지 않는다.

## Datasets

현재 준비 상태는 [`manifests/DATASET_SUMMARY.md`](manifests/DATASET_SUMMARY.md),
준비 과정의 이동/삭제/검증 기록은
[`manifests/DATA_PREPARATION_LOG.md`](manifests/DATA_PREPARATION_LOG.md) 참고.

요약: GigaHands / InterHand2.6M / HanCo / AssemblyHands 준비 완료,
Re:InterHand는 의도적으로 미다운로드.

## Environment

시스템 Python 3.11에 numpy/OpenCV가 없어 프로젝트 전용 venv를 사용한다
(Git 제외):

```bash
python -m venv experiments/.venv
experiments/.venv/Scripts/python.exe -m pip install numpy opencv-python-headless matplotlib
```

## Experiments

| ID | 목적 | 상태 |
|---|---|---|
| CAM-EXP-001 | GT camera / GT 3D joint loader & convention validation | 완료 — [report](runs/CAM-EXP-001_gt_projection_validation/report.md) |
| CAM-EXP-002 | camera error가 absolute hand 3D에 미치는 영향 | 예정 |
| CAM-EXP-003 | AnyCalib / GeoCalib / Perspective Fields baseline | 예정 |
| CAM-EXP-004 | single frame vs multi-frame video aggregation | 예정 |
| CAM-EXP-005 | background / hand / full RGB cue decomposition | 예정 |
| CAM-EXP-006 | GT hand geometry upper-bound calibration | 예정 |

### 실행

```bash
python experiments/make_manifests.py           # manifest 재생성
python experiments/run_cam_exp_001.py --full   # 실험 실행 (raw CSV 생성)
python experiments/analyze_cam_exp_001.py      # raw CSV로부터 표/그림 생성
```

`--full` 없이 실행하면 소규모 smoke test로 동작한다.

## Camera convention

모든 loader는 아래 canonical form으로 변환한 뒤 사용한다
(`src/geometry/camera.py`):

```
X_cam = R @ X_world + t
(u, v) = K @ distort(X_cam[:2] / X_cam[2])
```

데이터셋별 저장 형식과 단위는 서로 다르며, 실측으로 검증된 변환은
CAM-EXP-001 report §4에 정리되어 있다. 새 데이터셋을 추가할 때도
convention을 추측하지 말고 소규모 smoke test로 먼저 검증할 것.

## Raw result 보존 원칙

그래프/표만 저장하지 않는다. 각 run은 per-joint 단위 raw CSV를 남기고,
모든 표와 그림은 그 raw CSV만으로 재생성 가능해야 한다 (재추론 불필요).
