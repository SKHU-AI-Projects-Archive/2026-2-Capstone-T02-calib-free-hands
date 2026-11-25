# Calib-free-hands — phone app (camera-control verification)

The app is called **Calib-free-hands** on screen and on the home screen. Internally it is still `com.camchk` (package, `adb logcat -s CAMCHK`, `camchk_*.csv`); only the display name changed. Note: the git remote already has a program of the same name (`CalibFreeHands/` on `main`/`capstone`); the two are separate and do not clash.

Minimal Android app (Kotlin, [RootEncoder](https://github.com/pedroSG94/RootEncoder) 2.8.2 as a Gradle dependency, library source untouched) that streams the camera to a PC and logs the **actual** `CaptureResult` values, to check whether exposure / focus / stabilization requests are really applied.

Target: Galaxy S25+ (Android 17 / API 37). One screen: preview on the left, control panel on the right.

### Screen

Landscape only, but both landscape directions work: turn the phone 180° and the screen follows (`sensorLandscape`, so it also works with the system auto-rotate switch off). The activity handles the change itself (`configChanges`), so the camera, the stream and a running connection are not restarted. The stream's own orientation is handled by RootEncoder (`autoHandleOrientation`).

The panel has the same look and structure as the PC recorder window (`../pc/recorder_gui.py`): dark theme, header, state chips (dot + caption + value), a big timer, a big start/stop button and a message card. Palette: `app/src/main/res/values/colors.xml` (same values in `recorder_gui.py`); keep the two in sync.

| Element | Meaning |
|---|---|
| STREAM chip | `PREPARING` (grey, camera starting) / `READY` (grey) / `CONNECTING` (amber) / `STREAMING` (green) / `FAILED` (red) |
| EXPOSURE / FOCUS chip | `AUTO` (grey) / `LOCKED` (amber, see long-press below) / `CONFIG` (grey: no exposure/focus module is `enabled` with `mode: auto`, so there is nothing to hold) |
| Timer | Time since the connection succeeded (green while streaming, `00:00:00` otherwise) |
| Button | Green `Start streaming`; grey `Cancel` while connecting (so an attempt never looks like a running stream); red `Stop streaming` only while really streaming; disabled while the camera is not ready |
| Header | `Calib-free-hands` and the stream settings from the config (`SRT  1920x1080 @ 30 fps`), a second line saying where the PC address comes from: `→ PC linked by QR` or `→ PC from app settings` |
| Scan QR button | Header, right (see below); enabled only while not streaming |
| Message card | Short plain messages (`Camera ready.`, `Connecting to the PC ...`, `Streaming to the PC.`, `QR code scanned. Tap Start streaming to connect.`). When the connection fails it turns into a red-framed warning: **NOT CONNECTED** (could not reach the PC, with the usual causes; the timer is hidden to make room) or **CONNECTION LOST** (it was streaming and dropped), plus a toast |

**No addresses and no error codes on screen**: IP addresses, ports, URLs, library error reasons and file names are not shown. They are in the log (`adb logcat -s CAMCHK`).
| Over the preview | `AE / AF LOCKED` badge while locked; a hint about the long-press that fades after 8 s |

## Configure

Everything is in [`app/src/main/assets/config_test_v2.yaml`](app/src/main/assets/config_test_v2.yaml) (the app loads this file; the older `config_test.yaml` is kept but unused; packaged into the APK, so rebuild after editing).

| Section | Keys |
|---|---|
| `stream` | `protocol` (`srt`\|`rtsp`), `url`, `width`, `height`, `fps`, `bitrate_kbps` (video only: the app has no microphone, the stream and the recordings carry no audio) |
| `exposure` | `enabled`, `mode` (`auto`\|`lock`\|`manual`), `exposure_time_ns`, `iso` (manual only) |
| `focus` | `enabled`, `mode` (`auto`\|`lock`\|`manual`), `focus_distance` (diopters, manual only) |
| `stabilization` | `enabled`, `eis` (`on`\|`off`), `ois` (`on`\|`off`) |
| `lock` | `wait_for_convergence` (`true`\|`false`), `lock_timeout_frames` |
| `log` | `every_n_frames` |

`enabled: false` → that module does nothing and library default behavior is kept.
Set `stream.url` to the PC receiver (see [`../pc/README.md`](../pc/README.md)), e.g. `srt://192.168.0.10:8890?streamid=publish:cam`. Or skip that and use **Scan QR** (below): the scanned address replaces `stream.url` without a rebuild.

Mode meaning: exposure `auto` = `AE_MODE_ON`; `manual` = `AE_MODE_OFF` + exposure time + ISO. Focus `auto` = continuous video AF; `manual` = `AF_MODE_OFF` + `LENS_FOCUS_DISTANCE`. Controls are applied once the camera session is running (first captured frame, via `setCustomRequest`). `lock` is described below.

### Lock section

The `lock` section is used **only when `exposure.mode` and/or `focus.mode` is `lock`** (and that module is `enabled: true`); for `auto` and `manual` it has no effect. It is shared by exposure and focus.

| Key | Meaning |
|---|---|
| `wait_for_convergence` | `true`: keep auto running and lock only once it has settled. `false`: lock on the first frame (earlier behavior). |
| `lock_timeout_frames` | With `wait_for_convergence: true`: lock after this many frames even if it never settled (fallback). Must be > 0. |

With `wait_for_convergence: true`:

- **Exposure**: normal auto-exposure runs; when `CONTROL_AE_STATE` is `CONVERGED` (or at the timeout) `CONTROL_AE_LOCK` is set.
- **Focus**: starts in `CONTINUOUS_VIDEO`; when `CONTROL_AF_STATE` is focused (`PASSIVE_FOCUSED` or `FOCUSED_LOCKED`) or at the timeout, the current `LENS_FOCUS_DISTANCE` is read from the capture result and the camera is switched to `AF_MODE_OFF` with that distance.

With `wait_for_convergence: false`: exposure sets `AE_LOCK` at once; focus uses `AF_MODE_AUTO` without a trigger (lens stays put).

### Long-press hold (exposure / focus)

Like the AE/AF lock of a camera app: **long-press the preview** to freeze the current exposure and focus, **long-press again** to go back to auto. The status text shows `locked: Exposure, Focus` / `unlocked: auto (...)`.

- Exposure holds with `CONTROL_AE_LOCK` (AE mode stays ON, the current exposure time and ISO stop changing).
- Focus reads the current `LENS_FOCUS_DISTANCE` and switches to `AF_MODE_OFF` at that distance; unlocking returns to `CONTINUOUS_VIDEO`.
- Only for modules that are `enabled: true` with `mode: auto`. `manual` is already fixed, and `lock` has its own lock logic (above), so the long-press does nothing for them.
- The hold survives starting/stopping the stream, but not restarting the app (it starts unlocked).
- Stabilization is independent: `stabilization.eis: off` and `ois: off` (the defaults) keep both off.
- The log lines `AE hold=...`, `AF hold=... locked_focus_distance=...` mark each toggle; the per-frame `ae_lock`, `af_mode`, `focus_distance` and the CSV show the effect.

### Lock from the PC while recording

When you press **Start recording** in the PC Recorder window, the PC locks the phone's exposure and focus (the same hold as the long-press) *before* the recording starts, and **Stop recording** releases it (`lock_on_record` in `../pc/run_config_manual.yaml`). The message card says `Exposure and focus locked by the PC for the recording.` / `... released by the PC.`; the LOCKED badge shows as usual. A lock you made yourself by long-press is not released by the PC, and a long-press while the PC lock is on takes it over. If the stream ends while the PC holds the lock, the phone goes back to automatic.

How: while streaming the app runs a tiny HTTP server (`RemoteControl.kt`, TCP port 8898 = `REMOTE_PORT`; `POST /hold?on=1|0`). It answers only the PC the stream goes to (the host of the scanned address); every other address gets 403. It is closed when the stream stops. The Android 17 local-network permission the app already has covers it. The PC reads the phone's address from the receiver, so nothing is configured.

### Scan QR: connect without editing the config

On the PC open the QR code (Recorder window → **Phone QR**, or `python ../pc/qr_window.py`; pick the PC address that is in the phone's Wi-Fi). **The scanner opens by itself when the app starts**: point the camera at the code. You can also tap **Scan QR** later (e.g. after cancelling, or when the PC changed).

- The scanner is Google's code scanner from Play services (`play-services-code-scanner`): no camera permission, needs Google Play services. The app releases its camera for the scan and restarts the preview afterwards, so scanning is only possible while not streaming.
- A scanned `srt://...` address is saved on the phone and **replaces `stream.url` from the config** until the next scan (the header shows `(scanned)`); to go back to the config value, clear the app data (`adb shell pm clear com.camchk`).
- A valid code **saves the address and starts streaming right away** (`QR code scanned. Connecting to the PC ...`); no extra tap. Streaming does not start recording: on the PC you start/stop takes in the Recorder window. If you cancel the scan, the preview stays and **Start streaming** uses the last scanned address.
- On the very first run Google downloads the scanner module, which needs internet once; without it the message card says so.
- **PC program closed → stream ends.** Closing the Recorder window (or Ctrl+C in `run_recorder.bat`) stops the receiver, which tells the phone: it shows `CONNECTION LOST`. If the PC just disappears (power off, Wi-Fi gone) the phone notices after about 6 s without any packet from the PC (`serverSilenceTimeout`, set by reflection in `MainActivity.enableServerSilenceCheck`; the library's own `setCheckServerAlive` pings with ICMP, which the PC firewall blocks).
- The code must use the configured protocol (`stream.protocol`, `srt`); anything else is rejected with a message.

## Build and install

1. Android Studio (current stable): **File → Open** → this `android/` folder. Let Gradle sync (needs JDK 17, Android SDK 37; Android Studio downloads Gradle 9.7.1 from `gradle-wrapper.properties`, and RootEncoder from JitPack).
2. Phone: enable Developer options → USB debugging, connect, accept the prompt.
3. Run from Android Studio, or build and install by hand:

   ```
   gradlew assembleDebug        (or Build → Build APK)
   adb install -r app/build/outputs/apk/debug/app-debug.apk
   ```

4. Grant camera (and, on Android 17, local network) permission on first launch, then tap **Start**. No microphone permission is needed.

## Read the logs

```
adb logcat -s CAMCHK
```

Logged once at camera start: hardware level, capabilities (`1` = MANUAL_SENSOR), available EIS/OIS modes, AE modes + lock availability, exposure/ISO ranges, AF modes, minimum focus distance. Then every `every_n_frames` frames the actual result: `ae_mode ae_lock exposure_time_ns iso af_mode focus_distance eis_mode ois_mode sensor_timestamp_ns ae_state af_state`.

When a lock is reached (`wait_for_convergence: true`) one line is logged for it, once:

```
AE lock: reason=converged|timeout after_frames=N exposure_time_ns=... iso=...
AF lock: reason=focused|timeout after_frames=N locked_focus_distance=... (af_state=...)
AF lock failed: no LENS_FOCUS_DISTANCE in result, staying in CONTINUOUS_VIDEO   (warning, rare)
```

`reason=timeout` means the state never reached CONVERGED / focused within `lock_timeout_frames`.

The same per-frame values go to a CSV in the app's external files dir (file name shown in the status text on launch). Columns: `frame, sensor_timestamp_ns, ae_mode, ae_lock, exposure_time_ns, iso, af_mode, focus_distance, eis_mode, ois_mode, ae_state, af_state` (`ae_state` and `af_state` are the last two columns):

```
adb shell ls /sdcard/Android/data/com.camchk/files/
adb pull /sdcard/Android/data/com.camchk/files/camchk_<timestamp>.csv .
```

Enum values: AE mode 0=OFF 1=ON; AF mode 0=OFF 1=AUTO 3=CONTINUOUS_VIDEO; EIS 0=OFF 1=ON 2=PREVIEW_STABILIZATION; OIS 0=OFF 1=ON; AE state 0=INACTIVE 1=SEARCHING 2=CONVERGED 3=LOCKED 4=FLASH_REQUIRED 5=PRECAPTURE; AF state 0=INACTIVE 1=PASSIVE_SCAN 2=PASSIVE_FOCUSED 3=ACTIVE_SCAN 4=FOCUSED_LOCKED 5=NOT_FOCUSED_LOCKED 6=PASSIVE_UNFOCUSED.

## Notes

- Not compiled or run in the environment where this was written (no Android SDK there); first build may need small fixes.
- Many devices ignore OIS while EIS is on (or the reverse) — the logged result values show what the camera actually did, which is the point of this app.
- `manual` exposure longer than the frame duration (1/fps) lowers the frame rate.
