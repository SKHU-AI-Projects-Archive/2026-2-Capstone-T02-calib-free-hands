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

### 02-A supervision protocol 확정

02 fine-tuning은 `raw GigaHands RGB`를 입력으로 받고, GigaHands의 `fx/fy/cx/cy`로 만든 canonical pinhole target ray field를 supervision으로 사용합니다. 입력 영상은 raw distorted RGB이며, `k1/k2/p1/p2`는 02 baseline target ray 생성에 사용하지 않습니다. 모델 output과 최종 fitting camera는 `pinhole`입니다.

02-A geometry audit에서는 physical distortion-aware ray를 perfect prediction으로 가정해도 official pinhole fitting 후 focal error median이 Train 약 13.83%, Validation 약 13.94%였고, ±5% pair가 각각 `0/147`, `0/49`였습니다. 따라서 physical distortion-aware supervision은 현재 pinhole focal objective와 구조적으로 맞지 않아 baseline에서 제외하고, `raw RGB + canonical pinhole target rays`를 선택했습니다. 이 판단에 Test 결과는 사용하지 않았습니다.

Audit의 p90/p95/max는 전체 pixel을 합친 global percentile이 아니라 pair별 statistic의 pair-equal mean입니다. center/border angular mean도 함께 보존합니다. 02-B 학습 loader 구현 전 Train/Validation A-oracle과 pretrained/training state_dict strict compatibility를 통과해야 합니다.

### 02-B training smoke pipeline

02-B single-GPU training smoke: **PASS**. 확정된 canonical pinhole supervision이 official training AnyCalib, official `l1-z1` loss, backward, finite/nonzero gradients, AdamW step, parameter update, FP32 Validation pinhole fitting까지 연결됩니다. Dataset은 raw RGB를 official deterministic preprocessing으로 `238x420`으로 만들고, `0.5` pixel-center convention의 canonical ray를 생성합니다. 이 결과는 engineering smoke이며 성능 결과가 아닙니다.

02-B smoke는 사용자가 single-GPU 환경에서 성공적으로 실행했습니다. Codex는 공유 서버에서 GPU smoke와 benchmark를 자동 실행하지 않으며, benchmark는 먼저 `nvidia-smi`로 빈 GPU를 확인한 뒤 사용자가 직접 실행합니다.

```bash
python HandCalib/train.py \
  --config HandCalib/configs/02_anycalib_finetune.yaml \
  --dry-run \
  --batch-size 1 \
  --num-workers 0

CUDA_VISIBLE_DEVICES=<FREE_GPU> python HandCalib/train.py \
  --config HandCalib/configs/02_anycalib_finetune.yaml \
  --smoke \
  --batch-size 1 \
  --num-workers 0 \
  --precision bf16
```

Smoke output은 `HandCalib/runs/02_anycalib_finetune/smoke/`에 저장되며 기본적으로 기존 결과를 덮어쓰지 않습니다. 이 smoke는 성능 평가가 아니고, 한 train batch의 forward/loss/backward/optimizer step과 한 validation batch의 fitting 연결만 확인합니다.

### 02-C training benchmark

02-C는 full training이나 성능 평가가 아니라 Train-only engineering benchmark입니다. 각 실행은 fresh pretrained model, `PairBalancedSampler` epoch 0, warmup 3 steps와 measured 10 steps를 사용합니다. `batch_size`를 먼저 비교하고, 선택 후 `num_workers`, 마지막으로 `bf16`과 `fp32`를 비교합니다. 결과로 최종 설정을 자동 선택하지 않습니다.

```bash
CUDA_VISIBLE_DEVICES=<FREE_GPU> python HandCalib/train.py \
  --config HandCalib/configs/02_anycalib_finetune.yaml \
  --benchmark --batch-size 1 --num-workers 0 --precision bf16
```

Benchmark는 `HandCalib/runs/02_anycalib_finetune/benchmark_train/bsX_nwY_precision/`에 ignored 결과를 기록합니다. `steps.csv`에는 measured step별 loader wait, host-to-device, forward, loss, backward, optimizer step, total 시간이 기록되고, JSON에는 samples/sec, timing mean/median/p95, VRAM peak를 기록합니다. OOM이나 non-finite loss/gradient는 해당 설정의 실패로 기록하며 자동 batch 축소나 precision 변경을 하지 않습니다. Test와 Validation은 사용하지 않습니다.

02-C bounded engineering benchmark completed. Train-only throughput, VRAM margin, stability, loader timing을 기준으로 runtime setting을 선택했습니다: `batch_size=4`, `num_workers=4`, training `bf16`, validation `fp32`. 이는 engineering runtime setting이며 validation 성능이나 학술적 batch-size 우열을 의미하지 않습니다. Test는 사용하지 않았습니다.

### 02-D.0 fine-tuning protocol audit

02-D.0에서는 full fine-tuning을 실행하지 않고, 현재 sampler와 `batch_size=4`의 실제 epoch 길이 및 official AnyCalib recipe를 점검했습니다. `25,578` sampled frames와 `drop_last=False`에서 `6,395` optimizer steps/epoch이며 마지막 batch는 2개입니다. step-0 pretrained Validation은 p36 전체 `18,652` frames와 49 pairs를 FP32로 1회 평가했습니다.

Checkpoint primary metric은 `val_pair_max_rel_f_mean`으로 고정합니다. 각 pair의 frame prediction에서 component-wise median `[fx, fy, cx, cy]`를 만든 뒤 GT와 다시 비교하고, 49개 pair를 동일 가중치로 평균합니다. 보조 metric과 raw predictions는 `HandCalib/runs/02_anycalib_finetune/validation_step0/`에 저장했고, protocol record는 `results/02d0_finetune_protocol_audit.yaml`입니다.

Step-0 결과는 18,652/18,652 frame success, 49/49 valid pairs, primary `0.2087389594`, within 5% focal `4/49`입니다. Validation runtime은 1020.326초, 18.280 frames/sec, peak reserved 1524 MiB였습니다. 이는 pre-finetuning reference이며 성능 결론이나 Test 비교가 아닙니다.

현재 official recipe의 확인값은 AdamW, base LR `6e-5`, backbone scale `0.1`, gradient clip `1.0`, epochs `40`, warmup `1000`, milestones `10000/30000`, gamma `0.3`, best key `angular_error`입니다. optimizer options가 비어 있어 PyTorch AdamW default weight decay는 `0.01`입니다. 이 recipe가 fine-tuning 전용인지 여부는 pinned source만으로 확정하지 않았습니다. epochs, LR, scheduler, validation interval은 다음 protocol 검토에서 결정합니다.

### 02-D.1 fine-tuning protocol frozen

두 개의 bounded 1-epoch LR pilot을 fresh `anycalib_pinhole` pretrained에서 실행하고 p36 Validation으로 비교했습니다. Candidate `6e-5`의 `val_pair_max_rel_f_mean`은 `0.0131863175`, Candidate `2e-5`는 `0.0133612541`이어서 `6e-5`를 선택했습니다. 두 후보 모두 step-0 `0.2087389594`보다 개선되었으므로 `1e-5` Candidate C는 실행하지 않았습니다. raw pilot 결과는 Git에서 제외되는 `runs/02_anycalib_finetune/lr_pilot/`에 있습니다. 상세 기록은 `results/02d1_finetune_protocol.yaml`입니다.

고정 protocol은 AdamW, base LR `6e-5`, backbone LR `6e-6` (`x0.1`), weight decay `0.01`, gradient clip norm `1.0`, batch `4`, workers `4`, BF16 training, FP32 Validation입니다. 최대 `5` epochs는 최적 epoch라는 주장이 아니라 현재 GigaHands step scale에서 official `30,000`-step milestone을 포함하는 bounded maximum duration입니다. Scheduler는 official `SequentialLR` semantics의 1000-step `LinearLR` warmup (`start_factor=0.001`) 뒤 `MultiStepLR` milestones `10000/30000`, gamma `0.3`이며 optimizer step마다 진행합니다. Validation은 매 epoch, best checkpoint 기준은 lowest `val_pair_max_rel_f_mean`, early stopping은 없습니다.

Production `--train` 경로와 CPU-safe `--train-plan`을 구현했지만 이번 단계에서는 실행하지 않았습니다. Pilot 모델은 최종 모델로 재사용하지 않으며 02-D.2에서 fresh pretrained initialization으로 full 5-epoch training을 실행합니다. **No full fine-tuning was run. Test data was not used.**

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

- 02-C benchmark 사용자 실행 및 batch size/worker/precision 비교
- 02-C 결과를 바탕으로 augmentation, optimizer와 metric protocol 결정
- HandCalib 모델과 학습/evaluation 코드 구현
