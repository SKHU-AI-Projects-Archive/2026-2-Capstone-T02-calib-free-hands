# CalibFreeHands 저장 영상 시연

저장소 루트에서 Windows PowerShell로 실행합니다.

```powershell
python -m capstone C:\path\to\capture.mp4 --runtime-root C:\path\to\wilor-runtime --output-dir C:\path\to\capstone-runs
```

영상 인자를 생략하면 앱에서 파일을 선택할 수 있습니다. `--headless`는 같은 추론을 실행하고 결과 경로와 요약을 출력합니다. `--estimator anycalib`는 AnyCalib 원본 모델을 설치한 환경에서 **AnyCalib 통합 검사**를 실행합니다.

## 실행 조건

- PATH의 FFmpeg와 FFprobe, CUDA 지원 PyTorch·torchvision, 기존 앱의 GUI·추론 의존성. WiLoR는 기존 `CalibFreeHands/README.md`의 모델 준비 절차를 따릅니다.
- HandCalib은 `rtmlib==0.0.15`와 ONNX Runtime을 사용합니다. 기본 CPU 경로는 `onnxruntime`, `--handcalib-device cuda`는 `onnxruntime-gpu`와 해당 CUDA 라이브러리가 필요합니다. GPU 환경에서는 두 ONNX Runtime 패키지 중 GPU용만 설치하고 RTMLib의 CPU 의존성 자동 설치를 막으려면 `pip install --no-deps rtmlib==0.0.15`를 사용합니다. RTMLib은 첫 실행 때 performance 모드의 YOLOX·RTMW 모델을 공식 배포처에서 내려받습니다.
- `--runtime-root`에는 기존 `WiLoR/wilor/`와 `rgb_predictor.py`, 공식 `WiLoR/pretrained_models/{wilor_final.ckpt,detector.pt}`, 적법하게 확보한 `mano_data/{MANO_RIGHT.pkl,mano_mean_params.npz}`가 필요합니다. 기존 WiLoR 설정 YAML을 재사용합니다.

## 처리 범위

첫 영상 타임스탬프 기준 `[0, 20)`초를 보정에 사용합니다. HandCalib의 기본 샘플링은 최대 10 Hz이며 `--calibration-hz`로 조정할 수 있습니다. 후속 5초 이내의 실제 프레임에 보정값을 적용합니다. `--work-seconds`는 후속 구간의 최대 길이를 바꿉니다. 원본은 그대로 디코딩하며 재인코딩하지 않습니다.

HandCalib은 연구 브랜치 `benchmark@8f70b58`의 촬영 가이드에 연결된 손 모드 `Calibrator`를 사용합니다. 서클은 가이드에서 제시한 촬영 동작입니다. 추정값은 원본 영상 픽셀 좌표계의 f이고, WiLoR의 기존 카메라 깊이 보정 경로에 전달됩니다.

결과 폴더의 `run.json`에 사용한 모델, 보정 샘플 시각, 후속 프레임 시각, 추정·적용 f, 검출 프레임 수를 기록합니다. `results.npz`에는 표시한 관절과 메시가 저장됩니다.
