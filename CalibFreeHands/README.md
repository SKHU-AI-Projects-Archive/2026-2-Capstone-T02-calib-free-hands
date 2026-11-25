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

For project-local FFmpeg, install the optional media extra instead of relying on a system FFmpeg installation:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[gui,test,media]"
```

Run:

```powershell
.\scripts\run_windows.ps1
```

Click **Open...** to open the Windows Common Item Dialog. Select supported files or choose **Select Current Folder** to add supported media directly inside the current folder; folder discovery is non-recursive. The dialog starts in the last input location. If the native dialog cannot be initialized, the app shows a warning before opening its Qt fallback picker. Queue items can be removed or cleared without deleting source files, and **Start Processing** runs the queue sequentially. A single successful media item opens its result viewer automatically; a multi-item batch does not open viewers automatically.

The effective output folder is shown in the main window, defaults to `Documents/CalibFreeHands`, and can be changed with **Browse...**. The selected folder persists between launches. Successful configurations are saved as one `.cfh` result bundle per source/model pair. Each bundle contains `result.npz` and `metadata.json`; source media is referenced by path and SHA256, never embedded. Intermediate session, calibration, raw-pose and preview files stay in a temporary working directory and are removed after processing.

Use **View Results...** to open one or more `.cfh` files. A single file opens the result viewer; multiple files open the Overlay/Split comparison viewer only when their source SHA256 values match. Historical canonical `results.npz` and `session.json` loading APIs remain available for development, while raw-pose NPZ files are rejected with a readable message. Automatic preview generation is disabled; the audio preference remains stored for future explicit preview export.

FFmpeg is required for video decoding. Provide it either through a Windows system installation with `ffmpeg` on `PATH`, or through the optional project-local `imageio-ffmpeg` runtime. `ffprobe` is recommended, but is not required for decoding: when available, the app uses source frame timestamps; when unavailable, decoding continues with sequential timestamps calculated as `frame_index / frame_rate` when a frame rate is known. Without `ffprobe`, timestamps for variable-frame-rate (VFR) media may be less accurate.

## Capture, HandCalib and processing in one program

The window has two pages (same design as the phone app: dark surfaces, green = start/ok, red = recording/stop; the Theme switch keeps Light and System):

- **Capture**: live camera picture of the phone (streamed over SRT to a MediaMTX receiver that the page starts and stops itself), the phone QR code (opens by itself; the app scans it and streams) or of the PC's own camera (see below), and two takes:
  1. **Calibration video**: move the whole right hand in front of the phone; stopping the take starts HandCalib, which returns the focal length in pixels (`HandCalib/benchmark/calibrator.py`, hand mode: the numbers the benchmark scores). A failure (hand found in too few frames, ...) is explained and the take can be recorded again.
  2. **Work video**: recorded with that focal length. Stopping it queues the video on the Process page with **Manual Focal = the HandCalib value** and starts processing; the result viewer opens as before. "Process automatically" can be switched off.

  The focal length can also be typed in (**Manual**; then no calibration video is needed). It is reused for further work videos of the same session until **Recalibrate**. Recording is started and stopped by the user. While a take runs, the phone's exposure and focus are locked and released again. Takes are saved as `<output folder>/captures/<session>/calibration.mp4` and `work_01.mp4`, `work_02.mp4`, ... (H.264 video only, about 1 MB per second).
- **Process**: queue, models and results, as before.

### PC camera instead of the phone

The **Camera source** box on the Capture page switches between **Phone (QR)** and **PC camera**. With the PC camera a **camera box** (and **Refresh**) lists the webcams of the PC; the live view, both takes, HandCalib and the hand-over to processing work exactly as with the phone. The phone's receiver is not started while the PC camera is the source.

- The camera is opened once (OpenCV / DirectShow, 1920x1080 at 30 fps requested, MJPEG; the real size and rate are shown) and stays open between takes, so a take starts and ends at once. If the camera cannot do 1080p the video has the size it delivers; use the same camera and settings for the calibration and the work video.
- While a take runs, **focus, exposure and white balance are fixed** at what the camera had chosen (autofocus off; exposure and white balance switched to manual at their current values) and given back afterwards, like the phone's AE/AF lock. White balance is included because otherwise the colours keep drifting during the take, e.g. when the hand enters the picture. Verified on a Logitech StreamCam by reading the DirectShow controls: all three change from Auto to Manual without a jump and stay constant; on that camera the automatic exposure works through the gain, which is frozen with it. Whether another camera accepts this depends on its driver; the page says when it did not (focus/exposure) and names white balance only when it was locked too. A webcam that keeps refocusing changes its focal length between the two videos.
- Takes are H.264 `.mp4` (about 8 Mbit/s, 1 MB/s, video only) with a **constant frame rate**; a picture is held until the next one arrives, so a webcam in dim light (fewer than 30 pictures per second) still gives a file as long as the real take. `<take>.intervals.csv` lists the real arrival times (the gaps are where pictures were held).
- **A focal length belongs to one camera.** Changing the source or the camera forgets the focal length HandCalib found and asks for a new calibration video (a typed focal length stays).
- Needs `ffmpeg` (camera list, encoding) on `PATH`; the camera list is DirectShow, i.e. Windows. Close other programs that use the camera (video calls, ...), it can only be opened once.

The receiver code is `../streaming_test/pc` (see its README for the phone side); this program imports it, so there is one copy. It needs `PyYAML` and `segno` (`pip install -e ".[capture]"`) and `ffmpeg`.

### Several focal lengths of one video

Add a row with **+ Add** for every focal length (Calibration = Manual Focal, a different focal in each row) and press **Start Processing**: the reconstruction runs once and every row gets its own result file (`<video>__Manual-Focal-<f>px__<model>.cfh`); the viewer then shows all of them side by side (Overlay / Split). If some of these files exist already (the video was processed before), the **Collision = Ask** setting asks before anything runs: **Skip existing** (default; only the new focal lengths are computed and the old results of the same video open in the viewer together with the new ones), **Keep both (rename)** or **Overwrite**, or **Cancel**. The answer applies to this run only; Collision = Overwrite / Auto rename / Skip decides in advance without asking.

### Language

The **Language** box on the Process page (English / 한국어) translates the application at once, no restart: both pages, buttons, the items of the Theme / Collision / Sampling / Aggregate boxes, status and capture messages, and the **result viewer** (menus, buttons, tool tips, the frame line, detached windows, the per-joint error table; viewers that are already open follow the switch). The choice is saved. The translations are in `src/calibfreehands/app/i18n.py` (`KO` for fixed texts, `_RULES` for messages with numbers or names); a text without an entry stays English, so a new string can be added later without breaking anything.

Not translated: preset and model names (they are part of the result file names), the error metric names in the viewer (MPJPE, PA-MPVPE, ... — abbreviations, the same as in the CSV/JSON exports) and the names of the results, the HUD value lines, and error texts of libraries. The viewer is translated only when a window is built or its menus are rebuilt, never in the per-frame update (the "Frame n / m" line is the one exception: it is translated when set, and is free in English).

### Environments

HandCalib needs OpenCV 5 + numpy 2 (the benchmark environment `.venv_bench` in the repository root), this program needs numpy 1.x for torch. Measured on the same video: OpenCV 4.10 gives the same focal within 0.02 % but its `cv2.calibrateCamera` takes 52 s instead of 0.1 s per 200 frames. So the focal estimate runs in a **separate Python process of `.venv_bench`** (`calibfreehands.handcalib`); the CUDA DLL directories of that environment are put on `PATH` automatically (without them onnxruntime-gpu silently falls back to the CPU). Another interpreter can be set with `handcalib_python` in the settings.

### Memory

Videos are no longer decoded into memory (a 20 s 1080p clip would need 3.7 GB, a minute 11 GB). `calibfreehands.media.frames.VideoFrameSource` knows the frame numbers, times and size up front and decodes the video again for every processing pass; the peak memory of two full passes over 801 frames of 1080p video is 46 MB.

## Data contract

The `.cfh` bundle preserves canonical NPZ numeric arrays and the original `run.json` provenance inside `metadata.json`. Missing keypoints are represented by `NaN` and a separate `detected` mask. The intended coordinate responsibility is camera space only: X image-right, Y image-down, Z camera-forward, origin at the optical center, units in metres. This prototype does not claim ground-truth accuracy or world/robot coordinates.

AnyCalib and original pretrained WiLoR adapters, FFmpeg decoding, result viewer, exports, and the QThread-backed processing boundary are implemented. Full raw-media inference remains dependent on the separately licensed MANO model files.

## Model assets

The original WiLoR project documents `wilor_final.ckpt` and `detector.pt` under `pretrained_models/`; its published setup targets Python 3.10, PyTorch 2.0.0, and CUDA 11.7. MANO files must be obtained from the MANO website under its own terms. The repository's `anyhand_wilor.ckpt` is not the original WiLoR checkpoint.

AnyCalib's official package supports Python 3.10+ and downloads its selected pretrained model through its documented torch/Hugging Face mechanism. No checkpoint is copied into this repository.

The application uses the original WiLoR checkpoint at `../WiLoR/pretrained_models/wilor_final.ckpt`, detector at `../WiLoR/pretrained_models/detector.pt`, and the existing original config at `../models/model_config_wilor_original.yaml`. WiLoR's loader expects the licensed `mano_data/MANO_RIGHT.pkl` and `mano_data/MANO_LEFT.pkl` files relative to the repository root; the setup script intentionally does not download them.

## Model selection and presets

The packaged `src/calibfreehands/app/model_config.json` lists the built-in model and preset definitions. The built-in `Original` preset selects Original AnyCalib, Original WiLoR, all-frame focal sampling, and median focal aggregation. The GUI can show HandCalib and AnyHand WiLoR, but keeps them unavailable: HandCalib has no confirmed app runtime adapter, and the legacy AnyHand WiLoR checkpoint is not present or integrated with this app. User presets are stored as validated JSON under Qt's `AppConfigLocation`; **Import Preset...** and **Export Preset...** use portable `.cfhpreset` files. The selector distinguishes Built-in, User, and Custom (unsaved) configurations. **Set as default** is separate from saving a named preset and stores the startup configuration in Qt settings; theme, language, and output directory are not included in portable presets. Unknown saved IDs fall back to `Original`.

Each processing run can contain explicitly added comparison configurations. Rows may reuse the same calibration or reconstruction model; the pipeline runs each unique calibration and raw pose model once, then applies the selected focal before packaging one canonical `.cfh` file per configuration. Internal `session.json`, calibration metadata, reusable raw-pose NPZ files, per-configuration NPZ/run JSON, and previews are created only in a temporary workspace. Raw pose geometry is retained before focal correction; camera translation depth is rescaled as `tz_new = tz_source * f_target / f_source`, while x/y translation is unchanged. Manual Focal accepts a focal length in pixels for comparison and includes that specified focal in its result filename; it does not represent ground-truth 3D.

The comparison viewer supports multiple same-source results in Overlay or an adaptive Split grid that retains the source media. Reference, metric selection, per-frame Left/Right metric values, confidence display, configuration colors, mesh opacity, and optional single-candidate error vectors are runtime controls over loaded arrays; they do not rerun inference or change saved results. Detach All opens a synchronized view over the same arrays. CSV and JSON metric exports use the currently selected Reference. These are reference differences, not ground-truth errors. Protocol-pending metrics stay unavailable until their evaluation protocol is verified.

To prepare the publicly downloadable original WiLoR checkpoint and detector in the repository's expected external `WiLoR/pretrained_models/` directory:

```powershell
.\scripts\setup_model_assets.ps1
```

This script does not download MANO assets and does not modify the protected source directories. MANO requires the user's own official account/license action.

## CalibFreeHands desktop application

The desktop app lives in `CalibFreeHands/` and includes a unified saved-result Viewer. Single-result mode prioritizes source media and one 3D result without comparison-only metrics; comparison mode shows same-source results with selected metrics in viewport overlays and per-joint details on demand. See `CalibFreeHands/UX_RESULT_SPEC.md` for the result-viewer contract.
