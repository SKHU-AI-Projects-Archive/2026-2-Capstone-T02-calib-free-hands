# HandCalib 벤치마크

HandCalib 데이터 수집 가이드(2026-11-14)에 따라 손 영상 벤치마크를 만들고 채점하는 도구 모음입니다. 가이드 내용은 [GUIDE.md](GUIDE.md)에 옮겨 두었고, 원본 HTML은 보관하지 않습니다.

`calibrator.py`와 `check_session.py`는 가이드의 코드를 수정 없이 그대로 옮긴 것입니다. 나머지 파일은 이 두 파일을 호출하기만 합니다.

| 파일 | 역할 |
|---|---|
| calibrator.py | 가이드 7장 원문. 체커보드 모드(참값)와 손 모드(f 추정) 두 가지가 있습니다. |
| check_session.py | 가이드 7장 원문. 세션 하나의 손 영상마다 f와 오차를 출력합니다. |
| session_rules.py | 가이드 규칙 중 스크립트로 확인할 수 있는 것: 이름, session.json, 체커보드 3% 규칙, 20초 이상, 해상도, 가로 영상. |
| validate_session.py | 세션 하나 또는 데이터 폴더 전체에 session_rules를 적용합니다. |
| measure_reference.py | 체커보드 영상 2개로 f_before, f_after, reference_f를 잽니다. --write를 붙이면 session.json에 저장합니다. |
| run_benchmark.py | 규약을 통과한 모든 세션의 손 영상에 Calibrator("hand")를 돌리고 reference_f 대비 지표를 냅니다. |
| session_template.json | 가이드의 모든 항목이 들어 있는 session.json 틀입니다. |
| create_session.py | 원본 카메라 파일 10개의 메타데이터로 세션 폴더와 session.json을 만듭니다. |
| qt_meta.py | exiftool 없이 QuickTime 메타데이터(기기, 렌즈, 35mm 환산 초점거리, 날짜)를 읽습니다. 위치 키는 읽지 않습니다. |
| GUIDE.md | 데이터 수집 가이드의 텍스트판입니다. |
| checkerboard_8x8.html, checkerboard_8x8.svg | 가이드의 화면용 체커보드입니다. 칸 8×8, 내부 모서리 7×7, 한 칸 너비의 흰 여백이 있습니다. |
| requirements.txt | 고정된 실행 환경입니다. `pip install --no-deps -r requirements.txt`로 설치합니다. |

설정 파일은 `configs/benchmark.json`입니다.

## 실행 환경

venv `/home/juson/venvs/handcalib_bench`를 씁니다. `requirements.txt`를 `--no-deps`로 설치한 환경입니다.

- Python 3.11
- rtmlib 0.0.15
- onnxruntime-gpu 1.22.0과 CUDA 12용 pip 라이브러리 `nvidia-*-cu12` (드라이버가 CUDA 12.4까지 지원)
- opencv-python 5.0.0
- numpy 2.4

Wholebody 모델(yolox_m, rtmw-dw-x-l, 316MB)은 `~/.cache/rtmlib`에 받아 두었습니다.

처리 속도는 RTX A5000에서 프레임당 약 0.03초입니다. 1080p 체커보드 영상은 `findChessboardCornersSB`가 모든 프레임에서 CPU로 돌기 때문에 몇 분 걸립니다.

```bash
PY=/home/juson/venvs/handcalib_bench/bin/python
cd benchmark
```

## 세션 추가

데이터 폴더는 `/public/data/handcalib_data`입니다. 수집자 ID는 아직 배정되지 않아서 당분간 s01(레포 관리자), s02(팀원)를 씁니다.

1. 원본 파일로 세션을 만듭니다. 폴더 이름과 session.json은 메타데이터에서 채웁니다.

   | 항목 | 출처 |
   |---|---|
   | date | 촬영 날짜. 메타데이터가 없으면 파일 이름(YYYYMMDD_HHMMSS) |
   | device | 기기 모델명 |
   | camera | 렌즈 모델명에서 rear/front 판단 |
   | lens | 35mm 환산 초점거리 ÷ 25를 카메라 앱 배율로 반올림 |
   | resolution | 화면에 보이는 가로×세로 |
   | fps | 영상 10개의 중앙값 |

   ```bash
   $PY create_session.py --collector s01 \
       --before IMG_a.MOV --after IMG_j.MOV \
       --twinkle IMG_b.MOV IMG_c.MOV --nod IMG_d.MOV IMG_e.MOV \
       --circle IMG_f.MOV IMG_g.MOV --mix IMG_h.MOV IMG_i.MOV \
       --stabilization off --focus-exposure-locked --shown-on "tablet screen" \
       --digital-zoom 1.0 --notes "..." [--dry-run]
   ```

   다음 경우에는 세션을 만들지 않고 멈춥니다.
   - 파일 10개의 기기, 렌즈 모델, 35mm 환산 초점거리, 해상도가 서로 다른 경우
   - 시네마틱 모드로 찍은 파일이 있는 경우

   메타데이터에 없는 항목은 null로 남기고 마지막에 목록으로 알려 줍니다. 손떨림 보정, 초점·노출 잠금, 체커보드를 보여 준 방법, 디지털 줌, notes가 여기에 해당합니다. 메타데이터에 기기 정보가 없는 영상(예: 갤럭시 영상을 다시 묶은 파일)은 `--device`, `--lens`, `--camera`로 직접 지정합니다.

   가이드가 원본 파일을 요구하므로 파일은 바이트 그대로 복사합니다. 위치 정보를 포함한 카메라 메타데이터도 그대로 남습니다.

   복사한 파일마다 파일 전체의 sha256이 원본과 같은지 확인하고, 그 값을 session.json의 `sources`에 기록합니다. 원본 파일은 바꾸지 않습니다.

2. 참값을 재서 session.json에 적습니다.

   ```bash
   $PY measure_reference.py /public/data/handcalib_data/<세션> --write
   ```

   f_before는 같은 세션의 checkerboard_before에서, f_after는 checkerboard_after에서 각각 잰 값입니다. reference_f는 두 값의 평균입니다.

   두 값이 3% 넘게 다르면 알려 주고 종료 코드 1로 끝납니다. 이 경우 세션 전체를 다시 찍어야 합니다.

3. 세션이 가이드를 지켰는지 확인합니다.

   ```bash
   $PY validate_session.py /public/data/handcalib_data/<세션>
   ```

   가이드 8장의 눈으로 확인할 항목도 체크리스트로 함께 출력합니다.

## 벤치마크 채점

```bash
$PY run_benchmark.py --config ../configs/benchmark.json
```

스크립트는 다음 순서로 동작합니다.

1. `data_root` 아래의 모든 세션 폴더를 먼저 검사합니다.
2. 가이드를 어긴 세션은 건너뛰고 요약에 표시합니다. `include_invalid: true`로 두면 그래도 채점합니다.
3. 손 영상마다 `check_session.py`와 같은 호출인 `Calibrator("hand").run()`을 돌립니다.
4. 영상이 오류로 끝나면(손이 거의 검출되지 않은 경우) `error`로 기록하고 다음 영상으로 넘어갑니다.
5. GPU 온도가 70°C 이상이면 멈추고 65°C 미만이 되면 다시 돌립니다.
6. 결과는 영상마다 바로 저장합니다. 중단된 실행은 `--resume runs/benchmark/<실행>`으로 이어서 돌립니다.

결과는 `runs/benchmark/<UTC 시각>/`에 저장됩니다.

| 파일 | 내용 |
|---|---|
| run.json | 설정, git 커밋과 수정 여부, 패키지 버전, 세션별 검사 결과 |
| results.csv | 손 영상마다 한 줄: 세션, 파일, 동작, 기기, 렌즈, 해상도, 상태, f, 프레임 수, 검출 프레임 수, 걸린 시간 |
| summary.md | 영상별 reference_f 대비 오차, 그리고 전체와 동작·세션·기기·렌즈·해상도별 집계 지표 |

집계 지표는 다음과 같습니다.

- 절대 오차의 중앙값과 평균
- 부호 있는 오차의 중앙값 (편향)
- 오차 5%, 10%, 20% 안에 든 영상의 비율

오류로 끝난 영상은 전체 영상 수에는 들어가지만 채점된 영상 수에는 들어가지 않습니다.

참값 f는 session.json과 summary.md에만 나옵니다.

## 자동으로 확인하는 규칙과 눈으로 확인할 규칙

스크립트가 확인하는 것:

- 세션 폴더 이름과 파일 이름
- 체커보드 영상 2개와 손 영상 8개(`hand_{twinkle,nod,circle,mix}_{01,02}`)가 정확히 있는지
- session.json의 모든 항목이 채워졌는지
- collector와 date가 폴더 이름과 같은지
- 손떨림 보정이 꺼져 있고 초점·노출이 잠겨 있는지
- f_before와 f_after의 차이가 3% 이내인지
- reference_f가 두 값의 평균인지
- videos[]가 실제 파일과 같은지
- 영상마다 20초 이상이고, 가로이고, session.json의 해상도와 같은지

눈으로 확인할 것 (가이드 8장):

- 오른손 하나만 보인다.
- 손가락 모양이 처음부터 끝까지 같다.
- 정면 자세로 시작하고, 손이 선처럼 보이는 순간이 없다.
- 흐린 장면이 없다.
- 손목부터 손끝까지 항상 화면 안에 있다.
- 손 길이가 화면 짧은 변의 1/3 이상이다.
- 얼굴이나 다른 사람이 찍히지 않았다.
