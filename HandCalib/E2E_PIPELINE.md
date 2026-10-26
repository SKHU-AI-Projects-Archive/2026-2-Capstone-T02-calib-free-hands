# End-to-end 실행: 17 프로토콜 → 20 전환 검출을 한 명령으로

20(전환 시각 검출)은 17 실행 폴더의 검출 결과와 프레임 목록을 읽습니다. 그래서 예전에는 17을 먼저 돌리고 그 폴더 경로를 20 설정의 `sources`에 직접 적어야 했습니다. `end-to-end` 명령은 이 연결을 자동으로 합니다.

```bash
source .venv/bin/activate
# 영상 목록은 configs/e2e_pipeline.yaml 의 videos 에 적거나 --videos 로 줍니다 (input_dir 안의 파일 이름)
python HandCalib/run.py end-to-end --config HandCalib/configs/e2e_pipeline.yaml --videos IMG_7158.MOV
python HandCalib/run.py end-to-end --config HandCalib/configs/e2e_pipeline.yaml --videos IMG_7158.MOV,IMG_7180.MOV --detectors mediapipe_hands
```

## 무슨 일이 일어나는가
1. `configs/17_protocol_rerun.yaml`(수정 없음)로 17 프로토콜을 실행해 `runs/e2e_pipeline/<UTC>/17_protocol_rerun/`에 저장.
2. `configs/20_phase_segmentation.yaml`(수정 없음)을 복사해 `videos`, `detectors`, `sources`(= 방금 만든 17 폴더), `transition_method`, 출력 위치만 바꾼 `config_20_derived.yaml`을 같은 폴더에 만듦. 새 영상의 수동 전환 칸(`manual_transition_s`)은 비워 둠(자동으로 채우지 않음).
3. 그 설정으로 20을 실행해 `.../20_phase_segmentation/`에 저장.
4. `e2e_summary.md`(영상·검출기별 f, 판정, 기존·linear_d 전환 시각, 가이드), `results_17.yaml`, `results_20.yaml`, `metadata.json`, `log.txt`를 같은 폴더에 씀. **저장소의 `results/` 추적 파일은 건드리지 않습니다.**

## 설정 (`configs/e2e_pipeline.yaml`)
`config_17`, `config_20`(분석 값은 이 두 파일에 그대로 있음), `videos`, `detectors`, `anycalib`(17의 AnyCalib 값, GPU), `transition_method`, `write_results_yaml`, `output.root`.

## 메모
- GPU 검출기(wilor_yolo)와 AnyCalib는 환경변수 `CUDA_VISIBLE_DEVICES`로 쓸 GPU를 고르세요. mediapipe만 쓰면 CPU만 사용합니다.
- 17에서 결과가 나오지 않은 검출기는 20에서 자동으로 빠지고 로그에 적습니다.
- 같은 영상·검출기로 기존 공식 17·20 결과와 f, 판정, 전환 시각이 일치함을 테스트(`E2EEndToEnd`)로 확인했습니다(IMG_7158, mediapipe).
- 21, 22, 23, 25 실험도 17 실행 폴더를 `sources`로 읽습니다. 같은 방식으로 이어 붙이려면 이 명령에 단계를 더하면 됩니다(아직 하지 않음).
