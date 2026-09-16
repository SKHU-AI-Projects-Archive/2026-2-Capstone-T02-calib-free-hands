# 손 검출 → MANO 피팅 → 3D 시각화 단위 데모

단안 RGB 영상에서 손을 찾고(YOLO), MANO 손 모델을 피팅해(WiLoR, AnyHand 재학습 체크포인트)
21관절 3D 좌표·MANO 파라미터·메시를 얻고, 이를 그림·GUI 로 확인하는 코드 입니다.
파이프라인은 아래와 같습니다.

```
demo/
├── demo_hand_mano.py   추론 CLI  (GPU + torch 필요, Qt 불필요)  → results.npz, PNG, 그래프, .obj
├── viewer_qt.py        GUI 뷰어  (PyQt5 + pyqtgraph, torch 불필요) ← results.npz + 원본 영상
├── hand_topology.py    공통 상수 (21점 연결, 손가락 색, 표시 좌표 변환)
└── README.md
```

```
   영상/이미지
      │
      ▼
 ① stage1_detect            YOLO 손 검출 → bbox (N,4), is_right (N,), score (N,)
      │
      ▼
 ② stage2_fit_mano          bbox 크롭 → WiLoR → MANO pose(48)/shape(10), 메시(778), 관절(21, 손목 상대), cam_t(3)
      │
      ▼
 ③ stage3_to_camera_space   관절 + cam_t → 절대 카메라 공간 [m]   ← 깊이 지터·절대 스케일이 여기(cam_t) 에 실림
      │
      ▼
 ④ stage4_visualize         PNG (좌: 원본+2D / 우: 3D)     +  wrist_depth.png  +  results.npz
                                                              └─ viewer_qt.py 로 마우스 회전·재생
```

---

## 1. 환경 (Windows, conda, NVIDIA GPU 4 GB+)

```powershell
conda create -n anyhand python=3.10 -y
conda activate anyhand

# PyTorch CUDA 12.1 — torch < 2.6 이어야 함 (WiLoR/ultralytics 가 weights_only=False 언피클에 의존)
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# WiLoR 의존성 (+ numpy 는 마지막에 고정)
pip install opencv-python pyrender pytorch-lightning scikit-image smplx==0.1.28 yacs timm einops ultralytics==8.1.34 hydra-core rich trimesh scipy tqdm dill matplotlib
pip install numpy==1.26.4

# GUI 뷰어
pip install PyQt5 pyqtgraph
```

리눅스도 같은 패키지로 동작합니다.

## 2. 가중치·MANO

⚠️ **라이선스**: AnyHand/WiLoR 체크포인트는 CC-BY-NC-ND(비상업·연구 한정), MANO 는 MPI 라이선스상 재배포가 금지되어 있어 이 저장소에는 가중치 파일이 포함되어 있지 않습니다(`.gitignore` 처리). 아래 표의 출처에서 각자 받아 지정된 위치에 놓아야 합니다.

| 파일 | 놓을 위치 | 출처 |
|------|-----------|------|
| `anyhand_wilor.ckpt`  | `models/` | [HF chen-si-02/AnyHand-Models](https://huggingface.co/chen-si-02/AnyHand-Models) |
| `detector.pt`  | `models/` | [HF spaces/rolpotamias/WiLoR](https://huggingface.co/spaces/rolpotamias/WiLoR) `pretrained_models/` |
| `wilor_final.ckpt`  | `models/` | 위 WiLoR 스페이스 |
| `MANO_RIGHT.pkl` | `mano_data/` | [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de) (원본은 `python tools\strip_chumpy.py mano_data\MANO_RIGHT.pkl` 로 1회 변환 필요 — 동봉본은 변환 완료) |


## 3. 실행

```powershell
conda activate anyhand
cd <압축 푼 폴더>            # rgb_predictor.py 가 있는 곳

# 영상 추론 → 완료되면 GUI 뷰어가 뜸 (마우스로 3D 회전, 재생, 손목 깊이 그래프)
python demo\demo_hand_mano.py --input samples\5_fold.mp4 --out out\fold --gui

# 이미 만든 결과를 다시 보기
python demo\viewer_qt.py out\fold\results.npz

# 프레임 PNG·메시·mp4 파일까지 남기기
python demo\demo_hand_mano.py --input samples\2_screw.mp4 --out out\screw --max_frames 120 --save_mesh --make_video

# 이미지 1장
python demo\demo_hand_mano.py --input some.jpg --out out\img
```


### 주요 옵션

`demo_hand_mano.py` (추론)

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--input` | (필수) | 영상(.mp4 등) 또는 이미지 파일 |
| `--out` | (필수) | 출력 폴더 |
| `--max_frames N` | 0 (전체) | 처리할 최대 프레임 수 |
| `--stride k` | 1 | 프레임 건너뛰기 간격 |
| `--checkpoint` | `anyhand` | `anyhand` / `wilor` (config 가 짝으로 따라감) |
| `--det_conf` | 0.3 | 손 검출 확신도 임계 (낮추면 미검출↓ 오검출↑) |
| `--det_iou` | 0.3 | 검출 NMS IoU (겹친 손 처리) |
| `--rescale_factor` | 2.0 | bbox 주변 크롭 확대 배율 (클로즈업 영상은 1.2~1.5) |
| `--save_mesh` | 꺼짐 | MANO 메시를 `meshes/*.obj` 로 저장 |
| `--make_video` | 꺼짐 | 프레임 PNG 를 `demo.mp4` 로 이어붙임 |
| `--no_png` | 꺼짐 | 프레임 PNG 생략 (GUI 로만 볼 때 빠름) |
| `--gui` | 꺼짐 | 완료 후 뷰어를 별도 프로세스로 실행 |

`viewer_qt.py` (뷰어)

| 옵션 | 설명 |
|------|------|
| `results.npz` (위치 인자) | 볼 결과 파일 |
| `--video <경로>` | 원본 영상 경로 지정 (npz 에 적힌 경로가 안 맞을 때) |
| `--frame N` | 시작 프레임 번호 |
| `--screenshot <png>` | 창을 저장하고 바로 종료 (동작 확인용) |

### 뷰어 조작

| 구역 | 조작 | 동작 |
|------|------|------|
| 3D 패널 | 마우스 좌드래그 | 회전 |
| 3D 패널 | 마우스 휠 | 줌 |
| 3D 패널 | 마우스 우드래그 / 중드래그 | 이동 |
| 3D 패널 | `카메라 리셋` 버튼 | 초기 시점(촬영 카메라 쪽에서 손을 바라봄)으로 복귀 |
| 재생 | `▶ 재생` 버튼, `Space` | 재생 / 일시정지 (영상 fps) |
| 재생 | 슬라이더, `←` `→` | 프레임 이동 (한 프레임씩) |
| 깊이 그래프 | 클릭 | 그 프레임으로 점프 |
| 깊이 그래프 | 흰 세로선 드래그 | 프레임 스크럽 |
| 표시 | `Left` `Right` 체크박스 | 손별 표시 켜기/끄기 |

3D 패널의 흰 점은 촬영 카메라(원점), 바닥 격자 간격은 0.1 m 입니다. 하단 라벨에 현재 프레임 번호와 양손 손목 깊이(m)가 표시됩니다.