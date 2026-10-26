# CalibFreeHands

CalibFreeHands is a Windows-oriented research desktop prototype for inspecting camera-space hand reconstruction results. The current checked-in slice provides the PySide6 shell, media queue input, and a small canonical result contract; model inference is not silently faked.

## Setup

Use a Python 3.10+ project-local environment and install the optional GUI/test dependencies:

```powershell
python -m pip install -e ".[gui,test]"
```

For an isolated Windows setup:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[gui,test]"
```

For a project-local FFmpeg binary, install the media extra:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[gui,test,media]"
```

For an isolated Windows setup:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[gui,test]"
```

Run:

```powershell
.\scripts\run_windows.ps1
```

The application currently accepts supported media files directly from a selected folder (without recursive traversal) or from the file picker. It does not modify the input media.

FFmpeg can be supplied by a Windows system installation (`ffmpeg`/`ffprobe` on `PATH`) or by the optional project-local `imageio-ffmpeg` runtime. When only the bundled `ffmpeg` executable is available, decoding works with sequential timestamps; `ffprobe` is preferred for reliable source timestamps.

FFmpeg is an external runtime dependency. Install a Windows build containing `ffprobe` and add its `bin` directory to `PATH`; missing `ffprobe` is reported explicitly.

## Data contract

Canonical numeric output is NPZ, accompanied by `run.json` provenance. Missing keypoints are represented by `NaN` and a separate `detected` mask. The intended coordinate responsibility is camera space only: X image-right, Y image-down, Z camera-forward, origin at the optical center, units in metres. This prototype does not claim ground-truth accuracy or world/robot coordinates.

AnyCalib and original pretrained WiLoR adapters, FFmpeg decoding, result viewer, exports, and the QThread-backed processing boundary are implemented. Full raw-media inference remains dependent on the separately licensed MANO model files.

## Model assets

The original WiLoR project documents `wilor_final.ckpt` and `detector.pt` under `pretrained_models/`; its published setup targets Python 3.10, PyTorch 2.0.0, and CUDA 11.7. MANO files must be obtained from the MANO website under its own terms. The repository's `anyhand_wilor.ckpt` is not the original WiLoR checkpoint.

AnyCalib's official package supports Python 3.10+ and downloads its selected pretrained model through its documented torch/Hugging Face mechanism. No checkpoint is copied into this repository.

The application uses the original WiLoR checkpoint at `../WiLoR/pretrained_models/wilor_final.ckpt`, detector at `../WiLoR/pretrained_models/detector.pt`, and the existing original config at `../models/model_config_wilor_original.yaml`. WiLoR's loader expects the licensed `mano_data/MANO_RIGHT.pkl` and `mano_data/MANO_LEFT.pkl` files relative to the repository root; the setup script intentionally does not download them.

To prepare the publicly downloadable original WiLoR checkpoint and detector in the repository's expected external `WiLoR/pretrained_models/` directory:

```powershell
.\scripts\setup_model_assets.ps1
```

This script does not download MANO assets and does not modify the protected source directories. MANO requires the user's own official account/license action.
