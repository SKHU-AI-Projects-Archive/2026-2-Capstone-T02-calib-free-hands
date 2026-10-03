# HandCalib 연구 상태

## 프로젝트 개요

HandCalib은 손-물체 상호작용 영상의 카메라 보정 방법을 연구하기 위한 작업 공간입니다. 현재는 AnyCalib 공식 코드를 연결하고 GigaHands demo 데이터를 준비하는 단계이며, HandCalib 고유 모델과 전체 학습 파이프라인은 아직 구현 중입니다.

## 현재 상태

- [x] HandCalib 기본 폴더와 AnyCalib 연결 구조 정리
- [x] 실제 데이터셋과 Python dataloader 분리
- [x] GigaHands 데이터와 다운로드 cache의 Git 제외
- [x] 공식 GigaHands 5개 demo archive 다운로드 및 검증 도구 추가
- [x] 공식 archive와 현재 local dataset의 파일 단위 비교
- [x] CAM-EXP-001 과거 검증의 목적과 raw dataset 비수정 여부 확인
- [x] Linux에서 downloader unittest와 실제 공식 archive 설치 검증
CAM-EXP-001은 제공 camera와 제공 3D joints를 2D annotation에 재투영해 camera convention, distortion, frame 대응을 확인한 과거 검증입니다. 일부 `(0, 0)` 2D detection과 per-view outlier 패턴을 관찰했지만 raw GigaHands 파일을 수정한 실험은 아닙니다.

## 실험 로드맵

- [x] 01번 실험 — 사전학습 AnyCalib 기본 성능 확인
- [ ] 02번 실험 — GigaHands를 이용한 AnyCalib 추가 학습
- [ ] 03번 실험 — 손 정보 활용을 위한 데이터 확인
- [ ] 04번 실험 — HandCalib 설계 및 구현
- [ ] 05번 실험 — HandCalib 학습 및 비교 평가

01번 결과는 `HandCalib/results/01_anycalib_pretrained.yaml`에 재현용 요약으로 고정했습니다. 원본 결과와 raw telemetry는 Git에 포함하지 않습니다.

### 01 고정 결과

- Test: `p52`, `p52-instrument-0034`, 7,667 frames, 49 pairs
- Frame success: `7,667 / 7,667` (`1.0`)
- Pair max focal error: mean `0.18497205009511505`, median `0.17258688477275663`
- Pair max principal-point error: mean `0.08199870613426849`, median `0.06982756720648872`
- Focal error within 5%: `4 / 49` (`0.08163265306122448`)

### 02-A supervision audit 계획

02번은 먼저 Train 147개와 Validation 49개 sequence-camera pair의 supervision geometry를 CPU에서 점검합니다. raw RGB를 full decode하지 않고, OpenCV Brown-Conrady `[k1, k2, p1, p2]`를 반영한 distortion-aware GT ray와 공식 AnyCalib 전처리·pinhole fitting oracle을 비교합니다. Test split은 이 감사에서 사용하지 않으며, audit가 끝나기 전 학습은 시작하지 않습니다.

## AnyCalib pretrained smoke test

가중치와 CUDA 환경이 준비되면 validation의 첫 frame 하나만 실행할 수 있습니다. `--dry-run`은 모델과 가중치를 사용하지 않고 설정·데이터 크기만 확인합니다.

```bash
python HandCalib/evaluate.py --config HandCalib/configs/01_anycalib_pretrained.yaml --dry-run
python HandCalib/evaluate.py --config HandCalib/configs/01_anycalib_pretrained.yaml --smoke
```

smoke 결과는 `HandCalib/runs/01_anycalib_pretrained/smoke_val/`에 저장되며, 전체 평가는 `--test`를 사용자가 명시적으로 실행할 때만 수행됩니다.

### 01 최종 평가 설정

- Model: `anycalib_pinhole`
- `cam_id`: `pinhole`
- Test batch size: `2`
- Test workers: `4`
- 선택 근거: Test를 보기 전에 Validation benchmark의 `bs1/nw0`, `bs1/nw2`, `bs2/nw2`, `bs4/nw2`, `bs2/nw4`를 비교해 고정했습니다.

최종 Test는 7,667 frame과 49 sequence-camera pair를 모두 사용하며, pair-level equal-weight 결과를 중심으로 해석합니다. 왜곡 계수는 이번 pinhole baseline의 primary metric에 포함하지 않습니다.

```bash
python HandCalib/evaluate.py --config HandCalib/configs/01_anycalib_pretrained.yaml --test --dry-run
CUDA_VISIBLE_DEVICES=0 python HandCalib/evaluate.py --config HandCalib/configs/01_anycalib_pretrained.yaml --test
```

최종 결과는 `HandCalib/runs/01_anycalib_pretrained/test/`에 저장됩니다. `config.yaml`은 설정 snapshot, `metadata.json`은 재현 환경 정보, `runtime.json`은 속도와 자원 사용량, `summary.txt`는 사람이 읽는 요약, `metrics.json`은 frame/pair metric, `pair_summary.csv`와 `clip_summary.csv`는 그룹별 통계, `frame_predictions.csv.gz`는 모든 frame의 source of truth, `telemetry.csv.gz`와 `batch_timings.csv.gz`는 GPU 및 batch timing 원자료입니다.

## 01 AnyCalib validation benchmark

benchmark는 최종 01 실험 결과가 아니라 throughput과 `batch_size`/`num_workers`를 비교하기 위한 Validation 첫 번째 완전한 RGB clip의 engineering 측정입니다. Test frame이나 Test inference는 사용하지 않습니다. 실제 실행은 사용자가 한 장의 GPU를 지정한 뒤 수행합니다.

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=<GPU_INDEX>
python HandCalib/evaluate.py --config HandCalib/configs/01_anycalib_pretrained.yaml --benchmark --batch-size 1 --num-workers 0 --dry-run
python HandCalib/evaluate.py --config HandCalib/configs/01_anycalib_pretrained.yaml --benchmark --batch-size 1 --num-workers 0
```

`--batch-size`와 `--num-workers`는 benchmark에서 반드시 명시하며, `1/0`은 최종 선택이 아닌 baseline engineering 설정입니다. 결과는 `HandCalib/runs/01_anycalib_pretrained/benchmark_val/bsX_nwY/`에 저장됩니다. `runtime.json`에는 load/evaluation 시간, FPS, latency percentile, loader 대기, PyTorch memory, GPU telemetry와 rough Test 시간 추정이 들어가고, `telemetry.csv.gz`는 약 1초 간격의 raw GPU 상태, `batch_timings.csv.gz`는 batch별 병목 분석 자료입니다.

## 폴더 구성

```text
HandCalib/
├── AnyCalib/       공식 AnyCalib 코드
├── configs/        실험 설정 파일
├── datasets/       실제 local 데이터셋. GitHub에는 올리지 않음
├── dataloaders/    데이터셋을 Python에서 읽는 코드
├── data/            split과 다운로드 cache
├── models/          모델 연결과 HandCalib 모델 코드
├── utils/           공통 보조 기능
├── tools/           다운로드와 데이터 검사 도구
├── runs/            local 실험 결과. 기본적으로 Git 제외
├── train.py        학습 진입점
└── evaluate.py     평가 진입점
```

## GigaHands 데이터 준비

```bash
git submodule update --init --recursive
cd HandCalib
python tools/download_gigahands_demo.py
python tools/download_gigahands_demo.py --check
```

기본 데이터 위치는 `datasets/gigahands/`이며, 다운로드 cache는 `data/downloads/gigahands/`입니다. 두 위치 모두 Git에서 제외됩니다. 다른 검증 위치가 필요하면 `--target`과 `--cache-dir`를 사용할 수 있습니다.

공식 출처: [Brown IVL GigaHands](https://github.com/brown-ivl/GigaHands)

## 확인된 데이터

- sequence 5개
- RGB MP4 248개
- camera parameter row 245개
- 전체 파일 1,385개
- 공식 archive: 760,999,592 bytes, tar entries 1,746개
- 공식 archive SHA256: `4243B1F837B85EF3F0689502E83D911798E6F634BA6A35FA27816BDD61EFBBA4`
- 현재 `datasets/gigahands/`와 공식 archive 추출본은 1,385개 파일의 경로·크기·SHA256이 모두 일치

검증 과정은 raw dataset을 수정하지 않았습니다. 알려진 invalid 2D pattern, exact video stem 연결, distortion 적용은 loader와 validation 규칙으로만 다룹니다.

## 아직 남은 일

- AnyCalib 환경과 pretrained baseline 실행
- fine-tuning protocol, split, metric 확정
- 02-A supervision audit와 fine-tuning protocol 확정
- HandCalib 모델과 학습/evaluation 코드 구현
