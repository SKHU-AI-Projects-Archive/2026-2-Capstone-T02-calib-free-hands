# CalibFreeHands

CalibFreeHands is a Windows-oriented research desktop prototype for estimating and inspecting camera-space hand reconstruction results. The PySide6 application processes queued media with the configured calibration and reconstruction backends; model inference is not silently faked.

Milestone C-1 Fix Pass has been manually approved in the Windows GUI.

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

Click **Open...** to open the Windows Common Item Dialog. Select supported files or choose **Select Current Folder** to add supported media directly inside the current folder; folder discovery is non-recursive. The dialog starts in the last input location. If the native dialog cannot be initialized, the app shows a warning before opening its Qt fallback picker. Queue items can be removed or cleared without deleting source files, and **Start Processing** runs the queue sequentially. A single successful media item opens its result viewer automatically; a multi-item batch does not open viewers automatically.

The effective output folder is shown in the main window, defaults to `Documents/CalibFreeHands`, and can be changed with **Browse...**. The selected folder persists between launches. Successful configurations are saved as one `.cfh` result bundle per source/model pair. Each bundle contains `result.npz` and `metadata.json`; source media is referenced by path and SHA256, never embedded. Intermediate session, calibration, raw-pose and preview files stay in a temporary working directory and are removed after processing.

Use **View Results...** to open one or more `.cfh` files. A single file opens the result viewer; multiple files open the Overlay/Split comparison viewer only when their source SHA256 values match. Historical canonical `results.npz` and `session.json` loading APIs remain available for development, while raw-pose NPZ files are rejected with a readable message. Automatic preview generation is disabled; the audio preference remains stored for future explicit preview export.

FFmpeg can be supplied by a Windows system installation (`ffmpeg`/`ffprobe` on `PATH`) or by the optional project-local `imageio-ffmpeg` runtime. When only the bundled `ffmpeg` executable is available, decoding works with sequential timestamps; `ffprobe` is preferred for reliable source timestamps.

FFmpeg is an external runtime dependency. Install a Windows build containing `ffprobe` and add its `bin` directory to `PATH`; missing `ffprobe` is reported explicitly.

## Data contract

The `.cfh` bundle preserves canonical NPZ numeric arrays and the original `run.json` provenance inside `metadata.json`. Missing keypoints are represented by `NaN` and a separate `detected` mask. The intended coordinate responsibility is camera space only: X image-right, Y image-down, Z camera-forward, origin at the optical center, units in metres. This prototype does not claim ground-truth accuracy or world/robot coordinates.

AnyCalib and original pretrained WiLoR adapters, FFmpeg decoding, result viewer, exports, and the QThread-backed processing boundary are implemented. Full raw-media inference remains dependent on the separately licensed MANO model files.

## Model assets

The original WiLoR project documents `wilor_final.ckpt` and `detector.pt` under `pretrained_models/`; its published setup targets Python 3.10, PyTorch 2.0.0, and CUDA 11.7. MANO files must be obtained from the MANO website under its own terms. The repository's `anyhand_wilor.ckpt` is not the original WiLoR checkpoint.

AnyCalib's official package supports Python 3.10+ and downloads its selected pretrained model through its documented torch/Hugging Face mechanism. No checkpoint is copied into this repository.

The application uses the original WiLoR checkpoint at `../WiLoR/pretrained_models/wilor_final.ckpt`, detector at `../WiLoR/pretrained_models/detector.pt`, and the existing original config at `../models/model_config_wilor_original.yaml`. WiLoR's loader expects the licensed `mano_data/MANO_RIGHT.pkl` and `mano_data/MANO_LEFT.pkl` files relative to the repository root; the setup script intentionally does not download them.

## Model selection and presets

The packaged `src/calibfreehands/app/model_config.json` lists the built-in model and preset definitions. The built-in `Original` preset selects Original AnyCalib, Original WiLoR, uniform sampling of 8 frames, and median focal aggregation. The GUI can show HandCalib and AnyHand WiLoR, but keeps them unavailable: HandCalib has no confirmed app runtime adapter, and the legacy AnyHand WiLoR checkpoint is not present or integrated with this app. User presets are stored as validated JSON under Qt's `AppConfigLocation`; **Import Preset...** and **Export Preset...** use portable `.cfhpreset` files. The selector distinguishes Built-in, User, and Custom (unsaved) configurations. **Set as default** is separate from saving a named preset and stores the startup configuration in Qt settings; theme, language, and output directory are not included in portable presets. Unknown saved IDs fall back to `Original`.

Each processing run can contain explicitly added comparison configurations. Rows may reuse the same calibration or reconstruction model; the pipeline runs each unique calibration and raw pose model once, then applies the selected focal before packaging one canonical `.cfh` file per configuration. Internal `session.json`, calibration metadata, reusable raw-pose NPZ files, per-configuration NPZ/run JSON, and previews are created only in a temporary workspace. Raw pose geometry is retained before focal correction; camera translation depth is rescaled as `tz_new = tz_source * f_target / f_source`, while x/y translation is unchanged. Checkerboard Reference accepts a focal length in pixels for comparison and includes that specified focal in its result filename; it does not represent ground-truth 3D.

The comparison viewer supports multiple same-source results in Overlay or an adaptive Split grid that retains the source media. Reference, metric selection, per-frame Left/Right metric values, confidence display, configuration colors, mesh opacity, and optional single-candidate error vectors are runtime controls over loaded arrays; they do not rerun inference or change saved results. Detach All opens a synchronized view over the same arrays. CSV and JSON metric exports use the currently selected Reference. These are reference differences, not ground-truth errors. Protocol-pending metrics stay unavailable until their evaluation protocol is verified.

To prepare the publicly downloadable original WiLoR checkpoint and detector in the repository's expected external `WiLoR/pretrained_models/` directory:

```powershell
.\scripts\setup_model_assets.ps1
```

This script does not download MANO assets and does not modify the protected source directories. MANO requires the user's own official account/license action.

## CalibFreeHands desktop application

The desktop app lives in `CalibFreeHands/` and includes a saved-result Comparison Viewer. Its comparison layout prioritizes source media and 3D viewports, with selected metrics shown in viewport overlays and per-joint details opened on demand. See `CalibFreeHands/UX_RESULT_SPEC.md` for the result-viewer contract.
