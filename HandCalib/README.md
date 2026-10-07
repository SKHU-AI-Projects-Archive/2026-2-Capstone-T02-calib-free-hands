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
nvidia-smi
GPU_ID=<physical GPU number selected by the user>
CUDA_VISIBLE_DEVICES="$GPU_ID" python HandCalib/evaluate.py --config HandCalib/configs/01_anycalib_pretrained.yaml --test
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

## 02-G.1 InterHand annotation audit completion

`results/02g1_interhand_audit_completion.yaml`은 이미지 없이 수행한 CPU-only 보완 감사입니다. 현재 보관한 `InterHand2.6M.annotations.5.fps.zip`의 JSON에서 image/frame과 annotation record를 분리해 확인했고, 공식 공개 수치 `1,361,062 / 380,125 / 849,160`은 [공식 homepage](https://mks0601.github.io/InterHand2.6M/)의 5fps H+M/M frame totals임을 대조했습니다. local archive는 [GitHub v1.0 release asset](https://github.com/facebookresearch/InterHand2.6M/releases/tag/v1.0)으로 2020-11-26에 업로드되었으며, homepage의 2021-03-22 v1.0 image release와 byte-equivalence는 확인하지 않았습니다.

공식 [loader와 projection code](https://github.com/facebookresearch/InterHand2.6M/blob/main/data/InterHand2.6M/dataset.py)의 `world2cam`과 `cam2pixel` convention으로 annotation-only projection을 재현했습니다. `data.json`에는 직접 비교할 2D joint field가 없어 pixel reprojection error는 산출하지 않았지만, `width/height`, focal, principal point 좌표계와 기존 FOV 계산은 일관됩니다. GigaHands Train prior의 약 80% focal error는 **GigaHands-fine-tuned AnyCalib 모델 성능이 아니라 constant calibration prior diagnostic**입니다. 따라서 InterHand 외부 평가는 `GO WITH LIMITATIONS`, Train을 사용한 camera-generalization 주장은 공식 split overlap 때문에 filtered subset을 권장합니다. 이번 감사에서도 이미지 다운로드, model inference, training, fine-tuning은 실행하지 않았습니다.

## GigaHands 데이터 준비

```bash
git submodule update --init --recursive
cd HandCalib
python tools/download_gigahands_demo.py
python tools/download_gigahands_demo.py --check
```

기본 데이터 위치는 `datasets/gigahands/`이며, 다운로드 cache는 `data/downloads/gigahands/`입니다. 두 위치 모두 Git에서 제외됩니다. 다른 검증 위치가 필요하면 `--target`과 `--cache-dir`를 사용할 수 있습니다.

공식 출처: [Brown IVL GigaHands](https://github.com/brown-ivl/GigaHands)

## 02-H.0 InterHand external evaluation preparation

InterHand external evaluation은 Official Test만 사용하고, `(capture, camera)` 360개 calibration unit에서 metadata-only deterministic sampling으로 최대 16 frame씩 선택합니다. frozen manifest는 `data/manifests/interhand_external_test_v1.csv`이며 5,760 frames, manifest SHA256은 `af910a754c3449c258335a3baf574cb2650d8fabe6cc3458ee69cee3f429b750`입니다. 두 모델은 이 manifest, preprocessing, metric, batch 설정을 공유하고 checkpoint만 다릅니다. 02-H.1에서 image archive integration과 전체 image readiness 검증을 완료했으며 실제 inference는 실행하지 않았습니다.

Sampling estimand는 Official Test frame distribution 전반의 deterministic subsample이며 sequence-balanced estimator가 아닙니다. Calibration unit을 equal weight로 집계하고 각 unit의 frame은 representative prediction에 사용하므로, sequence imbalance만으로 frozen manifest를 교체하지 않습니다. 이번 integrity completion에서 manifest는 변경하지 않았습니다. Metadata-level intrinsic round-trip은 검증했지만 JPEG decode와 actual-image preprocessing/K consistency는 archive 준비 후 pending입니다.

Image archive를 받은 뒤의 준비 순서는 다음과 같습니다. `gh`가 설치되어 있으면 아래 primary 명령을 사용합니다.

```bash
cd /home/junghyub/2026-2-Capstone-T02-calib-free-hands
IMG_DIR="HandCalib/datasets/interhand2.6m/archives/images_5fps_v1.0"
mkdir -p "$IMG_DIR"
gh release download v1.0 \
  --repo facebookresearch/InterHand2.6M \
  --pattern 'InterHand2.6M.images.5.fps.v1.0.tar.part*' \
  --pattern 'InterHand2.6M.images.5.fps.v1.0.tar.CHECKSUM' \
  --dir "$IMG_DIR" \
  --skip-existing
```

`gh`가 없는 경우에는 exact suffix를 Python으로 생성하는 curl fallback을 사용합니다. brace expansion을 사용하지 않으며 개수·순서·중복을 확인하고 전송 실패를 반환합니다.

```bash
cd /home/junghyub/2026-2-Capstone-T02-calib-free-hands
IMG_DIR="HandCalib/datasets/interhand2.6m/archives/images_5fps_v1.0"
mkdir -p "$IMG_DIR"
mapfile -t PARTS < <(.venv/bin/python - <<'PY'
letters = "abcdefghijklmnopqrstuvwxyz"
suffixes = [f"part{a}{b}" for a in "ab" for b in letters if not (a == "b" and b > "r")]
assert len(suffixes) == 44 and suffixes[0] == "partaa" and suffixes[-1] == "partbr"
assert len(suffixes) == len(set(suffixes))
print("\n".join(suffixes))
PY
)
test "${#PARTS[@]}" -eq 44
BASE="https://github.com/facebookresearch/InterHand2.6M/releases/download/v1.0/InterHand2.6M.images.5.fps.v1.0.tar."
for suffix in "${PARTS[@]}"; do
  curl -fL --retry 5 --retry-all-errors --connect-timeout 20 --speed-time 60 --speed-limit 1024 -C - \
    -o "$IMG_DIR/InterHand2.6M.images.5.fps.v1.0.tar.$suffix" "$BASE$suffix"
done
curl -fL --retry 5 --retry-all-errors --connect-timeout 20 --speed-time 60 --speed-limit 1024 -C - \
  -o "$IMG_DIR/InterHand2.6M.images.5.fps.v1.0.tar.CHECKSUM" \
  "https://github.com/facebookresearch/InterHand2.6M/releases/download/v1.0/InterHand2.6M.images.5.fps.v1.0.tar.CHECKSUM"
```

```bash
cd /home/junghyub/2026-2-Capstone-T02-calib-free-hands
.venv/bin/python HandCalib/tools/verify_interhand_images.py \
  --archive-dir HandCalib/datasets/interhand2.6m/archives/images_5fps_v1.0 \
  --manifest HandCalib/data/manifests/interhand_external_test_v1.csv \
  --full-test-data HandCalib/datasets/interhand2.6m/raw/annotations/all/InterHand2.6M_test_data.json \
  --scan-members
.venv/bin/python HandCalib/tools/extract_interhand_subset.py \
  --archive-dir HandCalib/datasets/interhand2.6m/archives/images_5fps_v1.0 \
  --manifest HandCalib/data/manifests/interhand_external_test_v1.csv \
  --output-root HandCalib/datasets/interhand2.6m/raw/images
.venv/bin/python HandCalib/tools/check_interhand_image_readiness.py
```

## 02-H.1 InterHand image archive integration

공식 5fps archive는 44개 part, 총 `81,718,036,480` bytes이며 per-part MD5 `44/44 PASS`입니다. deadlock-safe member scan으로 frozen manifest `5,760/5,760`, Official Test annotation `352,897/352,897` mapping을 확인했고, 전체 frozen subset만 selective extraction했습니다. 실제 JPEG `5,760/5,760` decode, annotation resolution/channel/finite pixel 검사, 기존 AnyCalib preprocessing과 intrinsic transform/inverse transform도 통과했습니다. 현재 readiness는 `READY_FOR_INFERENCE`입니다.

상세 기록은 `results/02h1_interhand_image_archive_readiness.yaml`에 있습니다. 이 단계에서도 Pretrained 또는 Giga-finetuned model forward/inference는 실행하지 않았습니다.

Preprocessing은 기존 AnyCalib evaluator 경로를 재사용하며, GT `fx/fy/cx/cy`는 target metric metadata로만 사용합니다. 실제 실행 전에는 다음 dry-run이 `5760/360`을 확인해야 합니다.

```bash
.venv/bin/python HandCalib/evaluate.py --config HandCalib/configs/02h0_interhand_pretrained.yaml --test --dry-run
.venv/bin/python HandCalib/evaluate.py --config HandCalib/configs/02h0_interhand_gigahands_finetuned.yaml --test --dry-run
```

모델 평가는 사용자가 readiness 검증을 끝낸 뒤 직접 실행합니다.

## 02-H.1.1 InterHand final-Test compatibility fix

InterHand loader row의 `calibration_unit`을 공통 `camera_key`로 노출하고, `sequence`/`video_name`/`participant` alias를 추가해 기존 evaluator contract와 연결했습니다. Final Test 검증은 dataset별로 InterHand Official Test `5,760/360`과 GigaHands Test `7,667/49`를 확인하며, InterHand frozen manifest SHA256과 실제 weight source/SHA256도 dry-run metadata에 포함합니다. Summary는 config metadata를 사용하고, 결과 writer에는 기존 primary metric/aggregation을 유지한 채 precommitted secondary distribution 및 threshold statistics를 추가했습니다.

이번 수정의 preflight에서는 두 InterHand config와 기존 GigaHands Test, Validation benchmark, 02 training dry-run, synthetic result schema를 검증했습니다. model build, CUDA inference, full Test 실행은 하지 않았습니다. 상세 기록은 `results/02h1_1_interhand_final_test_compatibility.yaml`입니다.

## 02-H.1.2 Fine-tuned checkpoint inference compatibility

`checkpoint_best.pt`는 official AnyCalib 배포 weight가 아니라 HandCalib training payload이며, 공식 loader의 `.tar` suffix assertion 때문에 build 전에 실패했습니다. adapter가 payload의 `model` state dict를 직접 로드하도록 최소 수정했고, legacy `.tar` symlink 경로와 399개 state key의 tensor hash equivalence를 확인했습니다. canonical checkpoint의 missing/unexpected key는 각각 `0/0`, parameter coverage는 `100%`입니다. GigaHands validation `p36` 2-frame tiny smoke도 통과했으며, InterHand 5,760-frame fine-tuned Test inference는 실행하지 않았습니다.

상세 기록은 `results/02h1_2_finetuned_checkpoint_inference_compatibility.yaml`입니다. 현재 상태는 `FINETUNED_INTERHAND_READY_FOR_INFERENCE`입니다.

## 02-H.1.3 Config-resolved checkpoint path

Fine-tuned config의 `runs/02_anycalib_finetune/train/checkpoint_best.pt`는 `HandCalib` PROJECT_ROOT 기준으로 해석되어야 하므로, evaluator의 기존 `_path_from_config()`를 adapter 전달 전에도 적용했습니다. config-driven build에서 resolved path 존재, checkpoint SHA, epoch/global step, state-dict contract를 다시 확인했고, repository root와 `HandCalib` 내부 실행 위치에서 같은 absolute path가 생성되는 것도 확인했습니다. InterHand full Test inference는 실행하지 않았습니다.

상세 기록은 `results/02h1_3_checkpoint_path_resolution.yaml`입니다. 현재 상태는 `FINETUNED_INTERHAND_CONFIG_PATH_READY`입니다.

```bash
# Select a physical GPU after inspecting the server; no fixed GPU number is assumed.
nvidia-smi
GPU_ID=<physical GPU number selected by the user>
CUDA_VISIBLE_DEVICES="$GPU_ID" .venv/bin/python HandCalib/evaluate.py --config HandCalib/configs/02h0_interhand_pretrained.yaml --test
CUDA_VISIBLE_DEVICES="$GPU_ID" .venv/bin/python HandCalib/evaluate.py --config HandCalib/configs/02h0_interhand_gigahands_finetuned.yaml --test

# Select two different idle physical GPUs after inspecting the server.
GPU_PRE=<physical GPU number selected for pretrained>
GPU_FIN=<different physical GPU number selected for fine-tuned>
test "$GPU_PRE" != "$GPU_FIN"
CUDA_VISIBLE_DEVICES="$GPU_PRE" .venv/bin/python HandCalib/evaluate.py --config HandCalib/configs/02h0_interhand_pretrained.yaml --test &
CUDA_VISIBLE_DEVICES="$GPU_FIN" .venv/bin/python HandCalib/evaluate.py --config HandCalib/configs/02h0_interhand_gigahands_finetuned.yaml --test &
wait
```

결과는 각각 `runs/02_interhand_external_eval/pretrained/`와 `runs/02_interhand_external_eval/gigahands_finetuned/`에 저장됩니다. archive/checksum/member mapping, selective extraction, image decode, preprocessing smoke test, old GigaHands regression이 모두 통과하기 전까지는 `NOT READY`입니다.

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

## 02-D.2 AnyCalib 전체 fine-tuning

고정된 프로토콜로 pretrained AnyCalib을 5 epoch 전체 fine-tuning했습니다. p36 Validation에서 `val_pair_max_rel_f_mean`이 가장 낮은 epoch 5를 최종 checkpoint로 선택한 뒤, p52 Test를 한 번 실행했습니다. 학습은 batch size 4, workers 4, seed 42, bf16 학습과 fp32 Validation을 사용했으며 총 optimizer step은 31,975회입니다.

- Best Validation primary: `0.006481811295934264`
- Checkpoint: `runs/02_anycalib_finetune/train/checkpoint_best.pt`
- Test: 7,667/7,667 frames, 49/49 valid pairs
- Test pair max focal error: mean `0.0161718921`, median `0.0128330404`
- Test pair max principal-point error: mean `0.0406829144`, median `0.0344826592`
- Focal error within 5%: `49/49` (`1.0`)

01 pretrained 기록과 비교하면 pair max focal error mean은 `0.1849720501`에서 `0.0161718921`로 91.26% 감소했고, principal-point mean은 `0.0819987061`에서 `0.0406829144`로 50.39% 감소했습니다. Test data was not used for training or checkpoint selection. p36 Validation과 p52 Test는 서로 다른 participant이므로 이 비교는 participant overlap 주의사항을 함께 기록합니다. 상세 수치는 `results/02d2_finetune_result.yaml`에 있습니다.

## 02-E Fine-tuning gain analysis

추가 학습이나 재추론 없이 기존 p52 Test raw output을 `camera_key`로 matched-pair 분석했습니다. pretrained 대비 fine-tuned pair max focal error는 49개 중 `47개 개선`, `2개 악화`, tie `0개`였습니다. pretrained의 signed focal bias는 양수(`fx 0.163752`, `fy 0.167986`)였고 fine-tuning 후 `fx 0.010564`, `fy 0.012457`로 크게 줄었습니다.

Train-only normalized-intrinsics constant prior도 함께 계산했습니다. Train median prior의 focal mean은 `0.0153451320`으로 fine-tuned `0.0161718921`보다 약간 낮았습니다. 따라서 이번 결과는 GigaHands calibration-distribution prior가 큰 역할을 했을 가능성을 지지하지만, image-based geometry 향상이나 calibration memorization을 단독으로 증명하지 않습니다. unseen physical-camera generalization도 주장하지 않습니다. 상세 pair CSV, figures, 수치 요약은 ignored `runs/02_anycalib_finetune/analysis/`에, tracked 요약은 `results/02e_finetuning_gain_analysis.yaml`에 있습니다.
