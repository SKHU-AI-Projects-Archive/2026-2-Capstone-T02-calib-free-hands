# Calib-free-hands — PC side: phone → PC SRT recording (MediaMTX)

The windows are named **Calib-free-hands - Recorder** / **- Phone QR**. They, and the console, show no IP addresses, ports or error codes: networks are shown by name (`Wi-Fi`, `Wired`), failures as plain sentences (details are in the console log). `--verbose` (`python run_pc.py --verbose`) prints the addresses, ports and the receiver's full log when you need them. **This folder is also used by `CalibFreeHands/`**: its Capture page imports `recorder_control.py`, `phone_link.py`, `manual_mode.py`, `netinfo.py` and `run_pc.py` from here and starts the receiver with the same configs (see `CalibFreeHands/README.md`), so these files must stay where they are and keep their function names. This stand-alone launcher (`run_recorder.bat`) still works on its own.

Receive an SRT stream from a phone, record it without re-encoding, and check frame timing of every finished recording automatically.

## Quick start: double-click `run_pc.bat`

Requirements: Python 3 with `pyyaml` and `segno` (`pip install pyyaml segno`; `run_pc.bat` installs them if missing; `segno` is only for the phone QR code), and `ffprobe` (ffmpeg) on PATH for the auto-verify.

**Default: manual recording.** `record_mode` in `run_config_manual.yaml` is `manual`: a connected phone is only shown as *connected*, and you start/stop recording yourself in the Recorder window (section [Manual recording](#manual-recording-startstop-from-the-pc-while-the-phone-streams)). The steps below describe `record_mode: auto`, where MediaMTX records every stream from the moment the phone connects.

1. **Double-click `run_pc.bat`** (or run `python run_pc.py`).
   - If `mediamtx.exe` is missing it downloads MediaMTX (version from `run_config.yaml`) from GitHub, checks the SHA-256 of the zip against the release checksum list, and extracts `mediamtx.exe` into this folder.
   - It starts MediaMTX with `mediamtx_run.yml` and prints a short note (with `--verbose` also the phone URLs, which the QR code contains):

     ```
     Phone publish URL (same Wi-Fi/LAN as this PC):
       srt://192.168.0.10:8890?streamid=publish:cam
     Android emulator publish URL:
       srt://10.0.2.2:8890?streamid=publish:cam
     Live preview (another terminal, while the phone is streaming):
       ffplay rtsp://localhost:8554/cam
     ```

2. Put the first URL into the phone app (`stream.url` in the Android app's config), start streaming.
3. When the phone stops, the recording in `recordings/<path>/` is finished and verified automatically; the summary appears in the launcher window and `<recording>.intervals.csv` is saved next to the recording.
4. **Ctrl+C** in the launcher window stops MediaMTX cleanly. (In the `.bat` window Windows may then ask "Terminate batch job (Y/N)?" — answer either; MediaMTX is already stopped.)

A recording is only "finished" when the publisher disconnects (or after `recordSegmentDuration`, 1 h) — the verify summary appears then, not while streaming.

### `run_config.yaml`

| Key | Meaning |
|---|---|
| `mediamtx_version`, `mediamtx_download_url`, `mediamtx_checksum_url`, `download_timeout_s` | which MediaMTX to download when `mediamtx.exe` is missing (`{version}` is substituted) |
| `mediamtx_exe`, `mediamtx_config`, `verify_hook` | file names in this folder |
| `stream_path` | path name in the printed URLs and the recordings folder (any name is accepted by the receiver) |
| `srt_port` (UDP), `rtsp_port` (TCP) | listener ports; applied on top of `mediamtx_run.yml` through `MTX_*` environment variables |
| `auto_verify` | `true`: each finished recording is analyzed by `verify_hook.py`. `false`: no hook, recording only |
| `stop_timeout_s` | wait this long after the stop request before terminating MediaMTX |
| `lan_ip`, `lan_probe_address` | `auto`: the PC's connected Wi-Fi address (the phone is on Wi-Fi), else the address on the default route (found by a routing lookup towards `lan_probe_address`); set `lan_ip` to an address if the wrong adapter (VPN/WSL) is picked |
| `emulator_host` | address of the PC as seen from the Android emulator |

Frame-timing settings (`nominal_fps`, `gap_threshold`, …) stay in `verify_config.yaml`; the hook overrides only `input_path` and `output_csv`.

## Files

| File | Purpose |
|---|---|
| `run_pc.bat` / `run_pc.py` / `run_config.yaml` | one-command launcher (above) |
| `mediamtx_run.yml` | MediaMTX config used by the launcher: SRT in, record to `recordings/`, no re-encoding, `runOnRecordSegmentComplete` hook |
| `verify_hook.py` | hook: verifies one finished recording (uses `verify_recording.py` unchanged) |
| `verify_recording.py` / `verify_config.yaml` | frame-timing report from ffprobe; also usable by hand |
| `run_recorder.bat`, `run_config_manual.yaml`, `mediamtx_manual.yml`, `manual_mode.py`, `recorder_control.py`, `recorder_gui.py` | manual start/stop recording (section below); `run_recorder.bat` opens the receiver and the Recorder window |
| `phone_link.py`, `live_view.py` | exposure/focus lock on the phone while recording; live camera picture in the Recorder window |
| `qr_window.py`, `theme.py`, `netinfo.py` | phone QR code window (section below); colors shared by both windows; the PC's addresses (Wi-Fi first) |
| `mediamtx_test.yml` | earlier manual config (same, without the hook) |

## Manual recording: start/stop from the PC while the phone streams

The phone keeps streaming; you decide on the PC when to record (one file per take).

1. Double-click **`run_pc.bat`** (or `run_recorder.bat` = `run_pc.bat --manual`, which uses manual mode whatever `record_mode` says). `record_mode: auto` in `run_config_manual.yaml` brings back recording on connect, without the window.
2. MediaMTX starts with `mediamtx_manual.yml` (recording **off**, no auto-verify hook, Control API on `127.0.0.1:<api_port>`, localhost only) and the **Recorder window** opens.
3. Start the stream on the phone. The STREAM chip turns green (`CONNECTED`) and the window only says that the phone is connected; nothing is recorded yet. **Start recording** is disabled while no stream is connected.
4. Type a take name, click **Start recording**. Recording begins at the next keyframe (up to one keyframe interval later); the window shows the actual start time and how long after the click it was, and counts the elapsed time from there.
5. Click **Stop recording**. The launcher waits for the segment to close, renames it to `<take_name>_<start timestamp>.ts` in `recordings/<path>/`, copies it into `<take_name>_<start timestamp>.mp4` (`output_format: mp4` in `run_config_manual.yaml`; no re-encoding, about a second, frame times unchanged; the `.ts` is removed, `output_format: ts` keeps it; if ffmpeg is missing or fails the `.ts` stays), runs `verify_hook.py` on the result and shows it in the window (CSV next to the file). The phone sends 8 Mbps, so a take is about 1 MB per second (20 s ≈ 20 MB). The take name then counts up (`take1` → `take2`) for the next take.

The **Recorder window** has the same look and structure as the Android app screen (dark theme, header, state chips, big timer, big start/stop button, message card; palette in `recorder_gui.py` = `android/app/src/main/res/values/colors.xml`):

| Element | Meaning |
|---|---|
| LIVE VIEW (left) | the phone's camera picture, about 15 fps, while the phone streams ("Waiting for the phone ..." otherwise). ffmpeg reads the stream back from the local receiver (RTSP); it only watches, it does not touch the recording or the phone. Needs `ffmpeg` on the PATH; `live_view: false` switches it off. Delay: measured about 0.31 s from light on the phone's lens to a decoded frame on the PC (293-348 ms, SRT buffer 120 ms included), plus up to 0.03 s until the window draws it (30 fps). The ffmpeg chain is tuned for low delay (single-thread decoding, no fps filter, flush per frame); the rest is the phone's camera/encoder and the SRT buffer, which the recording shares. The picture has no sound: the phone sends video only |
| STREAM chip | `CONNECTED` (green) / `NOT CONNECTED` (grey) / `LOST` (amber, stream dropped during a take) / `NO MEDIAMTX` (red, Control API not reachable) |
| RECORDING chip | `IDLE` (grey) / `STARTING`, `STOPPING` (amber) / `REC` (red; `REC (NO STREAM)` if the stream dropped) |
| `● REC` badge, timer | red and blinking while recording; the timer counts from the actual recording start |
| Buttons | green **Start recording**, red **Stop recording**; the one that cannot be used now is grey |
| Message card | why Start is disabled while there is no stream (`Waiting for the phone to stream ...`, or how to start MediaMTX if the API is not reachable), then what is happening (start time and latency, saved file name, errors) |
| LAST TAKE | one-line verdict of the take (frames, duration, fps, gaps and estimated dropped frames; green when there are no gaps, amber otherwise). No file paths or full report on screen or in the console; start with `--verbose` (`run_recorder.bat --verbose`) to print the take details and the full verify report to the console |
| Footer | **Open folder** (the recordings folder of this path), **Always on top** |

`run_config_manual.yaml`:

| Key | Meaning |
|---|---|
| `record_mode` | `auto` or `manual` |
| `api_host`, `api_port`, `api_timeout_s` | Control API address (localhost) |
| `poll_interval_s` | how often the window polls the API |
| `lock_on_record` | `true`: **Start recording** first locks the phone's exposure and focus (so the first recorded frame is already locked), **Stop recording** releases them again. A lock the user made on the phone by long-press is left alone |
| `phone_port`, `phone_address`, `phone_timeout_s` | the phone app's remote-control port (must match `REMOTE_PORT` in the app); empty `phone_address` = taken from the receiver's SRT connection; timeout of one call |
| `live_view` | live camera picture in the window (needs ffmpeg) |
| `mediamtx_config_manual` | config used in manual mode |
| `recordings_dir` | recordings folder (relative to `pc/`), must match `recordPath` in the yml |
| `start_wait_timeout_s` | how long Start waits for the recording to begin |
| `segment_settle_s`, `segment_wait_timeout_s` | Stop: segment counts as closed when its file has not changed for `segment_settle_s` (keep it above `recordPartDuration`, 1 s) |
| `timestamp_format` | timestamp in the renamed file (actual recording start) |

How it works: `recorder_control.py` (usable from your own scripts: `Recorder(settings).start("take1")`, `.stop()`, `.status()`) toggles `record` of the path with `PATCH /v3/config/paths/patch/<path>`, reads the stream state from `GET /v3/paths/get/<path>` and the actual start of the segment from `GET /v3/recordings/get/<path>`. `recorder_gui.py` is the window; `manual_mode.py` is the glue used by `run_pc.py`.

Notes:

- If the receiver of an earlier start is still running (its window may be closed already), starting again does **not** start a second one: it opens only the Recorder window, and the running receiver keeps running (stop it with Ctrl+C in its own window). If the ports are used by something else (e.g. an earlier `record_mode: auto` receiver) it says which ports and exits.
- Closing the Recorder window while recording stops the recording first, **then the launcher stops the receiver too**, which ends the phone's stream (the phone shows `CONNECTION LOST`). Ctrl+C in the launcher stops MediaMTX (an unfinished take is left under its MediaMTX file name, not renamed or verified).
- **Exposure/focus lock** needs the PC to reach the phone (the same Wi-Fi/LAN, TCP `phone_port` 8898, no "client isolation" on the Wi-Fi). The phone only answers the PC its stream goes to, and only while it streams. If the phone cannot be reached the recording still starts and the message card says so (an old app version has no remote control: reinstall the app). Nothing is needed in the PC firewall: the PC connects to the phone, not the other way round.
- If the stream drops during a take, recording resumes by itself when it returns; those later segments are saved as `<take>_<timestamp>_part2.ts`, …
- If the stream has B-frames (e.g. an ffmpeg/x264 test source) the verify step may report one extra "dropped" frame at the end of a take: the recording is cut mid-reorder. Phone MediaCodec streams normally have no B-frames.

## Phone QR code: connect the phone without editing the config

The Android app has a **Scan QR** button: it reads the publish URL from a QR code, saves it (it replaces `stream.url` from the app config) and connects. The app opens the scanner when it starts. No rebuild when the PC's IP changes.

1. The code **opens by itself** when the Recorder window starts (unless a phone is already streaming). Open it again with the **Phone QR** button in the Recorder window, or `python qr_window.py`.
2. The first (green) network is the PC's Wi-Fi, which is the one the phone can reach. If the PC has other networks (`Wired` ...) they are listed by name too; pick another one only if the phone is in that network. No address is shown.
3. Start the app (the scanner opens by itself; otherwise tap **Scan QR**) and point the camera at the code. The app saves the address and starts streaming right away. Recording is still started by you in the Recorder window.
4. The QR window **closes by itself** when the phone connects (it watches the receiver's Control API, so this works in manual mode; a phone that is already connected when you open the window does not close it).

The code holds the URL printed by `run_pc.py`, `srt://<pc ip>:<srt_port>?streamid=publish:<stream_path>`. Needs `pip install segno`; without it the window shows the URL and the install hint.

## Phone publish URL

```
srt://<PC_IP>:8890?streamid=publish:<path>
```

`<PC_IP>` is the PC's LAN address (phone and PC on the same network); `<path>` is any name, e.g. `cam` → recordings go to `recordings/cam/`. The receiver accepts any publisher without a password (MediaMTX default `any` user) — use only on a trusted network.

## Windows firewall

SRT runs over **UDP 8890** (or your `srt_port`). On first start Windows may prompt to allow `mediamtx.exe` — allow it on **Private** networks. Otherwise add the rule yourself (admin PowerShell):

```powershell
New-NetFirewallRule -DisplayName "MediaMTX SRT" -Direction Inbound -Protocol UDP -LocalPort 8890 -Action Allow -Profile Private
# only if you want to watch via RTSP from another device:
New-NetFirewallRule -DisplayName "MediaMTX RTSP" -Direction Inbound -Protocol TCP -LocalPort 8554 -Action Allow -Profile Private
```

If the phone cannot connect, check the Wi-Fi is marked "Private" (not "Public") in Windows network settings. The Android emulator reaches the PC via `10.0.2.2` and needs no firewall rule beyond the above.

## Manual use (without the launcher)

- Receiver only: `mediamtx.exe mediamtx_run.yml` from this folder (default ports 8890/8554; hook command is `python verify_hook.py`).
- Verify any file by hand: set `input_path` in `verify_config.yaml`, then `python verify_recording.py` (or `python verify_hook.py <file>`).

Output of the verify step:

- Console: frame count (vs. expected), duration, mean fps, interval mean/std/min/max, number of gaps and estimated dropped frames (`round(interval / nominal) - 1` per gap), first `max_gaps_printed` gaps.
- CSV: one row per interval — `frame_index, pts_s, interval_s, is_gap, dropped_est`.

Timestamps are the ffprobe `best_effort_timestamp_time` of each video frame, sorted by presentation time (so B-frames do not distort intervals).
