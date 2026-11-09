# 손 영상으로 카메라 초점거리(f) 추정: 검출기 방식과 광류(Optical flow) 방식

손을 책상 위에서 움직이는 영상 하나로 휴대폰 카메라의 초점거리 f(px)를 추정한다.
f 추정에 넣는 손 점을 어떻게 얻느냐에 따라 두 방식이 있다. 두 방식은 실행 파일이 다르다.

| 방식 | 실행 파일 | 손 점 |
|---|---|---|
| 검출기 방식 | `run.py` | 2D 손 검출기가 매 프레임 낸 관절 |
| 광류 방식 | `optical_flow/run_optical_flow.py` | 한 프레임의 관절을 광류 추적으로 다른 프레임들까지 옮긴 점 |

---

## 1. 실행 방법

아래 명령은 모두 `HandCalib` 폴더에서, 프로젝트 가상환경을 켠 뒤 실행한다.

```bash
cd HandCalib
source <프로젝트 가상환경>/bin/activate
```

입력 영상 폴더와 검출기 설정은 설정 파일이 가리키는 기본 설정(`input_dir`, `detector_settings`)을 따른다.

### 1-1. 검출기 방식

`run.py`의 하위 명령으로 실행한다.

```bash
# 검출기 관절(MediaPipe, MMPose 손 모델들) + free_shape3d
python run.py free-shape3d-detectors --config configs/51_free_shape3d_mmpose.yaml

# MediaPipe 관절 + 책상 운동 추정기 (손 경사 11° 고정, 경사 함께 추정)
python run.py detected-joints --config configs/81_detected_joints.yaml

# MediaPipe 관절 + 책상 운동 추정기, 손 경사를 여러 값으로 바꿔 가며
python run.py detected-ramp-sweep --config configs/82_detected_ramp_sweep.yaml
```

### 1-2. 광류 방식

`run.py`를 거치지 않고 이 폴더의 실행 파일을 쓴다.

```bash
python optical_flow/run_optical_flow.py --config configs/89_optical_flow_entry.yaml               # 설정의 영상 전부
python optical_flow/run_optical_flow.py --config configs/89_optical_flow_entry.yaml --videos 2    # 시험 실행: 앞의 2개 영상만
python optical_flow/run_optical_flow.py --config configs/89_optical_flow_entry.yaml --out <폴더>  # 결과 폴더 직접 지정
```

설정 파일에서 바꿀 수 있는 것:

| 키 | 뜻 |
|---|---|
| `settings_config` | 기본 설정 파일. 영상 목록, 광류 설정(`tracking`), 프레임·추정 설정, 기준 f를 여기서 읽는다 |
| `joint_set` | f 추정에 쓰는 관절 묶음(`j16` = 관절 5~20, 손목과 엄지 제외) |
| `ramps_deg` | 고정할 손 경사 목록. 경사를 함께 추정하는 결과도 같이 낸다 |
| `reference` | 추적을 시작하는 기준 프레임. 기본값 `{mode: first_pause}` = 손을 돌리기 전 첫 정지 구간의 가운데. `{mode: middle}` = 검출 범위의 한가운데(예전 규칙) |
| `videos` | 비우면 기본 설정의 영상 전부. 파일 이름 목록을 주면 그 영상만 |
| `compare_run` | 비교할 이전 실행 폴더(선택). 있으면 summary.md에 같은 추정의 f를 옆에 적는다 |
| `detection_cache_from` | MediaPipe 검출 결과를 재사용할 폴더(선택, 같은 검출기 설정일 때만) |

결과는 `runs/<설정의 output.root>/<UTC 시각>/`에 남는다.
- `results.json`: 영상별 f, 경사, 재투영 RMS 등
- `summary.md`: 기준 f와 오차
- `metadata.json`: git 상태, 설정 해시, 라이브러리 버전
- `log.txt`: 실행 로그

---

## 2. 두 방식의 차이

| 항목 | 검출기 방식 | 광류 방식 |
|---|---|---|
| 손 점을 얻는 방법 | 매 프레임 2D 손 검출기(MediaPipe 또는 MMPose)가 관절 21개를 따로 찾는다 | 기준 프레임 하나의 MediaPipe 관절을, 이웃 프레임 사이 광류로 구한 평면 변환을 이어 붙여 모든 프레임으로 옮긴다 |
| 2D 검출기의 역할 | f 추정에 들어가는 점 그 자체 | 손 위치를 알려 주는 보조: 손 영역 마스크, 기준 프레임의 관절 위치, 화면 가장자리 판정 |
| 점이 가리키는 곳 | 검출기가 추정한 관절 중심 | 손 표면과 같이 움직이는 평면 위의 점 |
| 프레임 사이 흔들림 | 프레임마다 따로 흔들린다 | 거의 없다(같은 변환으로 옮기므로) |
| 오차가 생기는 곳 | 검출기의 프레임별 오차, 보는 각도에 따라 관절 위치를 다르게 잡는 치우침 | 손 전체를 평면 하나로 옮기므로, 그 평면 밖의 점(손목 등)은 손이 기준 자세에서 많이 돌수록 어긋난다. 피부 무늬가 약하면 추적이 미끄러질 수 있다 |
| 프레임 추출 | 설정의 `detect_fps` | 30 fps (이웃 프레임 사이 움직임을 작게 하려고) |
| 사용 프레임 | 손이 검출되고 상자가 화면 가장자리에 닿지 않는 프레임 | 추적 사슬이 끊기지 않았고, 옮긴 점의 상자가 화면 가장자리에 닿지 않는 프레임 |
| f 추정기 | 같은 추정기를 쓸 수 있다(free_shape3d, 책상 운동 추정기) | 책상 운동 추정기 |

**책상 운동 추정기**는 손이 책상 위에 붙어 있다고 본다. 손은 손목 쪽이 정해진 경사만큼 들린 평면이고, 프레임마다 책상 위에서 돌고 미끄러진다. 이 모델로 f, 책상에 대한 카메라 자세, 프레임별 손의 회전과 이동을 함께 푼다. 여러 시작점에서 풀어 비용이 가장 낮은 해를 고른다.

---

## 3. 광류 방식의 추적 단계

이웃한 두 프레임마다 아래를 반복한다. 설정값은 기본 설정의 `tracking`이다.

1. **손 영역 마스크** (`kt_hand_mask`): 이전 프레임의 MediaPipe 관절 21개를 감싸는 볼록 껍질을 손 상자 긴 변의 3%만큼 넓힌다. 그 프레임에 검출이 없으면 직전 검출값을 쓴다(`kt_carry`).
2. **특징점 고르기**: 마스크 안에서 Shi-Tomasi 점수(7×7 창의 밝기 변화)가 높은 점을 최대 400개 고른다. 최고 점수의 1% 미만은 버리고, 서로 7 px 이상 떨어지게 한다. 특징점은 매 프레임 새로 뽑는다.
3. **다음 프레임 위치 찾기**: 피라미드 Lucas-Kanade(21×21 창, 3단계).
4. **되돌려서 검사**: 다음 프레임에서 거꾸로 추적해 원래 위치에서 1 px 넘게 어긋나는 점은 버린다.
5. **평면 변환**: 남은 점으로 USAC_MAGSAC 호모그래피(1.5 px)를 구한다. inlier가 20개 미만이면 그 지점에서 사슬이 끊긴다.
6. **이어 붙이기** (`kt_chain`, `track_hand`): 기준 프레임에서 앞뒤로 단계별 변환을 곱한다. 기준 프레임은 손을 돌리기 전 첫 정지 구간의 가운데다(`run_optical_flow.choose_reference_pose`). 예전 규칙(검출 범위의 한가운데, `choose_reference`)은 손 자세를 보지 않아서, 처음 자세로 돌아오지 않는 영상에서는 손을 크게 돌려 둔 순간이 기준이 됐다. 기준 프레임의 관절 21개를 이 변환으로 옮긴 것이 추적점이다(`chain_points`).

---

## 4. 이 폴더의 파일

| 파일 | 내용 |
|---|---|
| `run_optical_flow.py` | 광류 방식 실행 파일. 프레임 추출 → MediaPipe 검출·추적 → 광류 추적점 → 책상 운동 추정기 |
| `klt.py` | 광류 추적 사슬: `kt_hand_mask`, `kt_chain`, `kt_carry`, `kt_map`, `chain_points`, `choose_reference`, `track_hand` |
| `variants.py` | 추적 사슬의 변형(원인 확인용): 특징점 위치 기록, 구간마다 다시 시작, 손 몸통·손가락 균형, 손목 따로 추적, 한 단계 중간 결과, 종이 윤곽 특징점 제외 |
| `__init__.py` | `import optical_flow as of`로 위 함수들을 쓸 수 있게 묶음 |
| `test_equivalence.py` | 이 폴더의 함수가 `handfocal_report.py`, `run.py`의 같은 이름 함수와 같은 결과를 내는지 확인하는 테스트 |

- 추적 코드(`klt.py`, `variants.py`)는 numpy와 OpenCV만 쓴다.
- 실행 파일은 프레임 추출, 검출, 추정기를 검출기 방식과 공유하므로 `handfocal`을 함께 불러온다.
- `handfocal_report.py`와 `run.py`에도 같은 추적 함수가 그대로 있다. `run.py`의 기존 명령은 그쪽을 쓴다.
- 두 쪽이 같은 결과를 내는지는 아래 테스트로 확인한다.

```bash
python -m unittest optical_flow/test_equivalence.py -v
```
