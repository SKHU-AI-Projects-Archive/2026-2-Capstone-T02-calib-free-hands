# 손 템플릿 기반 초점거리 추정: origin/main 대비 변경 사항

이 브랜치(`feature/hand-template-focal`)는 **손 동작 영상 하나로 휴대폰 카메라의 초점거리 f(px)를 추정하는 실험 코드**를 `HandCalib/`에 추가한다.
origin/main에 있던 기존 파일은 **하나도 수정하지 않았다.** 추가만 했다.

## 1. 바뀐 파일 (origin/main 대비 추가 10개)

| 파일 | 내용 |
|---|---|
| `HandCalib/handfocal_estimate.py` | 초점 거리 추정 쪽 코드 전체. 영상 규격 변환, 프레임 추출, 검출 결과 읽기, 손 추적, 정면 템플릿, 기울기(θ2d), PnP 기반 f 추정(fixed_template, joint_delta, free_template, free_shape3d), 불확실도, 판정, AnyCalib 참조값, 17의 촬영 조건 점검·검출기 순위·프레임 거르기, 18의 구간 분할·체커보드 보정, 이후 실험(20~42)의 분석 함수 |
| `HandCalib/handfocal_report.py` | 결과 출력·보고서 쪽 코드. 결과 json·csv·그림·보고서(report.md) 작성과 출처 기록(git 상태, 설정 해시, 라이브러리 버전) |
| `HandCalib/handfocal.py` | 위 두 파일을 한 이름공간으로 불러오는 얇은 파일. `import handfocal as hf`는 이전과 같은 이름을 모두 제공해 run.py와 테스트는 그대로다 |
| `HandCalib/run.py` | 실행 스크립트(한 파일). 하위 명령 `protocol`, `frame-filter`, `same-session`, `overlay` |
| `HandCalib/detector_worker.py` | 2D 손 검출기 워커 3종(wilor_yolo, mediapipe_hands, rtmpose_hand). 검출기마다 다른 가상환경에서 별도 프로세스로 실행되므로 분석 코드와 분리했다 |
| `HandCalib/tests/test_handfocal.py` | 단위·통합 테스트(unittest) |
| `HandCalib/configs/17_protocol_rerun.yaml` | 실험 17 설정: 검출기 3종, 촬영 조건 문턱값, 판정 문턱값, 파이프라인 설정 |
| `HandCalib/configs/17_frame_filter_diag.yaml` | 17 진단 설정: 기울기 구간에서 θ2d ≤ 45°, 가장자리 비접촉 프레임만 사용 |
| `HandCalib/configs/18_same_session_check.yaml` | 실험 18 설정: 한 녹화 안의 손 구간 + 체커보드 구간 |
| `HandCalib/results/17_protocol_rerun.yaml` | 17 결과 요약(영상별·검출기별 f 또는 "값 없음", 판정, 촬영 조건 충족 여부, AnyCalib 중앙값) |
| `HandCalib/results/18_same_session_check.yaml` | 18 결과 요약(체커보드 기준 f, 손 방식 f, AnyCalib 중앙값) |
| `HandCalib/HAND_FOCAL_CHANGES.md` | 이 문서 |

## 2. 방법 한 줄 요약

처음 약 3초 동안 정면으로 멈춘 손의 2D 관절 모양을 "평면 손 템플릿"으로 삼는다. 이후 손을 여러 방향으로 기울인 프레임마다, f 후보 60개 각각에 대해 평면 PnP로 자세를 맞추고 재투영 오차를 합산한다. 오차가 가장 작은 f를 고른다.
가정은 핀홀, 주점 영상 중심, 왜곡 없음, 손바닥과 펴진 손가락이 한 평면이라는 것이다. 판정은 GOOD / ROUGH / REJECT이며, f가 탐색 범위 경계에 붙으면 값 대신 "값 없음"으로 적는다.

## 3. 실행 방법

```bash
source .venv/bin/activate
# 17: input_dir의 영상(또는 --videos로 지정)에 검출기 3종을 돌려 f·판정·촬영 조건·품질 순위를 낸다
python HandCalib/run.py protocol --config HandCalib/configs/17_protocol_rerun.yaml --videos IMG_XXXX.MOV
python HandCalib/run.py protocol --config HandCalib/configs/17_protocol_rerun.yaml results-yaml --run runs/17_protocol_rerun/<UTC>
# 17 진단: 큰 기울기·가장자리 프레임을 뺀 재추정
python HandCalib/run.py frame-filter --config HandCalib/configs/17_frame_filter_diag.yaml --source IMG_XXXX.MOV=runs/17_protocol_rerun/<UTC>/IMG_XXXX
# 18: 손 구간 + 체커보드 구간이 한 녹화에 있는 영상
CUDA_VISIBLE_DEVICES=<비어 있는 GPU> python HandCalib/run.py same-session --config HandCalib/configs/18_same_session_check.yaml
# 검출기별 2D 키포인트를 그린 영상 (결과 폴더 안에 저장)
python HandCalib/run.py overlay --video-dir runs/17_protocol_rerun/<UTC>/IMG_XXXX
# 테스트
python -m unittest HandCalib/tests/test_handfocal.py -v
```

결과는 `HandCalib/runs/<실험>/<UTC>/`에 저장된다. `runs/`는 git에서 제외돼 있다.

## 4. 저장소 밖에 필요한 것 (서버 로컬)

| 항목 | 위치 | 용도 |
|---|---|---|
| 입력 영상 | `/home/juson/data/hand_focal_videos/` | git에 넣지 않는다. 결과·로그에 위치(GPS)·기기 정보는 쓰지 않는다 |
| wilor_yolo | `/home/juson/models/wilor_original/detector.pt`, MonoDETR venv(ultralytics 8.4.147) | 2D 손 검출 |
| mediapipe_hands | `/home/juson/venvs/det_mediapipe`(mediapipe 0.10.21), `/home/juson/models/mediapipe/hand_landmarker.task` | 2D 손 검출(CPU) |
| rtmpose_hand | `/home/juson/venvs/det_rtmpose`(rtmlib 0.0.16, onnxruntime 1.20.1), `/home/juson/models/rtmpose/...end2end.onnx` | 2D 손 검출(CPU, YOLO 박스 입력) |
| AnyCalib | main의 `HandCalib/AnyCalib` 서브모듈과 `models/anycalib.py`(수정 없음), 가중치 `~/.cache/torch/hub/anycalib/` | 비교용 별도 값(오프라인 실행). 손 방식의 입력으로는 쓰지 않는다 |
| 실험 14 산출물 | `HandCalib/runs/14_hand_template_focal/`의 합성 연구 결과·검출 캐시·IMG_7141/7142 공식 실행 | 17·18 설정이 읽는다(예상 오차 표, 이전 영상의 프레임 목록·AnyCalib 값 재사용) |

## 5. 정리 과정

- 실험 14(파이프라인), 16(검출기 교체), 17(프로토콜 재실행), 18(같은 녹화 체커보드)의 패키지 4개와 실행 스크립트 7개를, 17·18이 쓰는 부분만 남겨 위 3개 파일로 합쳤다.
- 합치기 전 코드로 만든 공식 결과를 새 코드로 다시 만들어 비교했다(16 IMG_7142, 17 IMG_7158과 프레임 거르기, 18 IMG_7155). f, 판정, 촬영 조건, 순위, 프로파일, AnyCalib가 모두 같았다. 체커보드 f만 3×10⁻⁹ px 차이가 있었는데, OpenCV 보정의 부동소수점 차이다.
- 14·16 전용 코드·설정·결과와 03–16 실험 결과 폴더는 이 브랜치에서 지웠다. 이전 실험 브랜치(`exp/03` ~ `exp/18`)는 서버 로컬 저장소에만 있고 원격에는 올리지 않았다.

## 6. 현재 결과 상태

- 17: 영상 11개(IMG_7141, 7142, 7145, 7147, 7148, 7157, 7158, 7176, 7177, 7178, 7180) × 검출기 3종이 모두 **REJECT**다. 비용 프로파일이 거의 평평하고(날카로움 약 1.00–1.1), 검출기마다 값이 크게 갈리거나 탐색 경계에 붙는다.
- 18: IMG_7155의 체커보드 기준 f는 판정 REJECT다(화면 덮임 14%). 그래서 "기준 불확실"로 표시했다.
- 촬영 조건을 거의 지킨 영상에서도 f를 또렷이 가르지 못했다. 그래서 평면 손 모델과 2D 키포인트의 불일치(손가락 굽힘, 검출기 편향)를 다음 개선 대상으로 보고 있다.
- 세부 수치는 `HandCalib/results/17_protocol_rerun.yaml`, `HandCalib/results/18_same_session_check.yaml`과 각 실행 폴더의 `report.md`, `summary.md`에 있다.
