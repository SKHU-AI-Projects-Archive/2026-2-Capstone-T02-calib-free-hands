# Calib-free-hands — PC 쪽: 폰 → PC SRT 녹화 (MediaMTX)

창 이름은 **Calib-free-hands - Recorder** / **- Phone QR**입니다. 이 창들과 콘솔에는 IP 주소, 포트, 오류 코드가 표시되지 않습니다. 네트워크는 이름(`Wi-Fi`, `Wired`)으로, 실패는 평범한 문장으로 보여 줍니다(자세한 내용은 콘솔 로그에 있습니다). 주소, 포트, 수신기의 전체 로그가 필요하면 `--verbose`(`python run_pc.py --verbose`)를 사용하세요. **이 폴더는 `CalibFreeHands/`도 사용합니다.** 그쪽 Capture 페이지가 여기의 `recorder_control.py`, `phone_link.py`, `manual_mode.py`, `netinfo.py`, `run_pc.py`를 import하고 같은 설정으로 수신기를 시작합니다(`CalibFreeHands/README.md` 참고). 따라서 이 파일들은 제자리에 두고 함수 이름도 유지해야 합니다. 단독 실행용 런처(`run_recorder.bat`)도 그대로 단독으로 동작합니다.

폰의 SRT 스트림을 받아 재인코딩 없이 녹화하고, 끝난 녹화마다 프레임 타이밍을 자동으로 검사합니다.

## 빠른 시작: `run_pc.bat` 더블클릭

필요한 것: `pyyaml`, `segno`가 설치된 Python 3(`pip install pyyaml segno`; 없으면 `run_pc.bat`이 설치합니다. `segno`는 폰 QR 코드용일 뿐입니다), 자동 검증용으로 PATH에 있는 `ffprobe`(ffmpeg).

**기본값: 수동 녹화.** `run_config_manual.yaml`의 `record_mode`가 `manual`입니다. 연결된 폰은 *connected*로만 표시되고, 녹화는 Recorder 창에서 직접 시작/정지합니다([수동 녹화](#수동-녹화-폰이-스트리밍하는-동안-pc에서-시작정지) 절). 아래 단계는 폰이 연결되는 순간부터 MediaMTX가 모든 스트림을 녹화하는 `record_mode: auto` 기준 설명입니다.

1. **`run_pc.bat`을 더블클릭**합니다(또는 `python run_pc.py` 실행).
   - `mediamtx.exe`가 없으면 `run_config.yaml`에 적힌 버전의 MediaMTX를 GitHub에서 내려받고, zip의 SHA-256을 릴리스 체크섬 목록과 대조한 뒤 `mediamtx.exe`를 이 폴더에 풉니다.
   - `mediamtx_run.yml`로 MediaMTX를 시작하고 짧은 안내를 출력합니다(`--verbose`이면 QR 코드에 들어 있는 폰 URL도 출력).

     ```
     Phone publish URL (same Wi-Fi/LAN as this PC):
       srt://192.168.0.10:8890?streamid=publish:cam
     Android emulator publish URL:
       srt://10.0.2.2:8890?streamid=publish:cam
     Live preview (another terminal, while the phone is streaming):
       ffplay rtsp://localhost:8554/cam
     ```

2. 첫 번째 URL을 폰 앱에 넣고(Android 앱 설정의 `stream.url`) 스트리밍을 시작합니다.
3. 폰이 멈추면 `recordings/<path>/`의 녹화가 끝나고 자동으로 검증됩니다. 요약은 런처 창에 나타나고, `<recording>.intervals.csv`가 녹화 파일 옆에 저장됩니다.
4. 런처 창에서 **Ctrl+C**를 누르면 MediaMTX가 깔끔하게 종료됩니다. (`.bat` 창에서는 Windows가 "Terminate batch job (Y/N)?"라고 물을 수 있습니다. 어느 쪽을 답해도 됩니다. MediaMTX는 이미 종료되었습니다.)

녹화는 발행자(publisher)의 연결이 끊어지거나(또는 `recordSegmentDuration`, 1시간 후) 되어야 "끝난" 것이 됩니다. 검증 요약도 스트리밍 중이 아니라 그때 나타납니다.

### `run_config.yaml`

| 키 | 의미 |
|---|---|
| `mediamtx_version`, `mediamtx_download_url`, `mediamtx_checksum_url`, `download_timeout_s` | `mediamtx.exe`가 없을 때 내려받을 MediaMTX (`{version}`이 치환됨) |
| `mediamtx_exe`, `mediamtx_config`, `verify_hook` | 이 폴더 안의 파일 이름 |
| `stream_path` | 출력되는 URL과 녹화 폴더에 쓰이는 경로 이름 (수신기는 어떤 이름이든 받아들임) |
| `srt_port` (UDP), `rtsp_port` (TCP) | 수신 포트. `MTX_*` 환경 변수를 통해 `mediamtx_run.yml` 위에 덮어씀 |
| `auto_verify` | `true`: 끝난 녹화마다 `verify_hook.py`가 분석. `false`: 훅 없이 녹화만 |
| `stop_timeout_s` | 정지 요청 후 MediaMTX를 강제 종료하기까지 기다리는 시간 |
| `lan_ip`, `lan_probe_address` | `auto`: PC의 연결된 Wi-Fi 주소(폰이 Wi-Fi에 있으므로), 없으면 기본 경로의 주소(`lan_probe_address`로의 라우팅 조회로 찾음). 엉뚱한 어댑터(VPN/WSL)가 잡히면 `lan_ip`에 주소를 직접 지정 |
| `emulator_host` | Android 에뮬레이터에서 본 PC의 주소 |

프레임 타이밍 설정(`nominal_fps`, `gap_threshold`, …)은 `verify_config.yaml`에 그대로 있고, 훅은 `input_path`와 `output_csv`만 덮어씁니다.

## 파일

| 파일 | 용도 |
|---|---|
| `run_pc.bat` / `run_pc.py` / `run_config.yaml` | 원 커맨드 런처 (위) |
| `mediamtx_run.yml` | 런처가 사용하는 MediaMTX 설정: SRT 입력, `recordings/`에 녹화, 재인코딩 없음, `runOnRecordSegmentComplete` 훅 |
| `verify_hook.py` | 훅: 끝난 녹화 하나를 검증 (`verify_recording.py`를 그대로 사용) |
| `verify_recording.py` / `verify_config.yaml` | ffprobe 기반 프레임 타이밍 보고서. 수동으로도 사용 가능 |
| `run_recorder.bat`, `run_config_manual.yaml`, `mediamtx_manual.yml`, `manual_mode.py`, `recorder_control.py`, `recorder_gui.py` | 수동 시작/정지 녹화 (아래 절). `run_recorder.bat`은 수신기와 Recorder 창을 엶 |
| `phone_link.py`, `live_view.py` | 녹화 중 폰의 노출/초점 고정, Recorder 창의 카메라 실시간 화면 |
| `qr_window.py`, `theme.py`, `netinfo.py` | 폰 QR 코드 창(아래 절), 두 창이 공유하는 색상, PC의 주소(Wi-Fi 우선) |
| `mediamtx_test.yml` | 이전의 수동 설정 (같지만 훅 없음) |

## 수동 녹화: 폰이 스트리밍하는 동안 PC에서 시작/정지

폰은 계속 스트리밍하고, 언제 녹화할지는 PC에서 정합니다(테이크 하나당 파일 하나).

1. **`run_pc.bat`**을 더블클릭합니다(또는 `run_recorder.bat` = `run_pc.bat --manual`: `record_mode`와 관계없이 수동 모드 사용). `run_config_manual.yaml`에서 `record_mode: auto`로 하면 창 없이 연결 시 녹화하는 방식으로 돌아갑니다.
2. MediaMTX가 `mediamtx_manual.yml`로 시작되고(녹화 **꺼짐**, 자동 검증 훅 없음, Control API는 `127.0.0.1:<api_port>`, 로컬호스트 전용) **Recorder 창**이 열립니다.
3. 폰에서 스트림을 시작합니다. STREAM 칩이 초록색(`CONNECTED`)이 되고 창에는 폰이 연결되었다고만 나타납니다. 아직 아무것도 녹화되지 않습니다. 스트림이 연결되어 있지 않으면 **Start recording**은 비활성화됩니다.
4. 테이크 이름을 입력하고 **Start recording**을 클릭합니다. 녹화는 다음 키프레임에서 시작됩니다(최대 키프레임 간격만큼 늦을 수 있음). 창에는 실제 시작 시각과 클릭 후 얼마 뒤였는지가 표시되고, 경과 시간은 그 시점부터 셉니다.
5. **Stop recording**을 클릭합니다. 런처는 세그먼트가 닫힐 때까지 기다린 뒤, `recordings/<path>/`에서 `<take_name>_<start timestamp>.ts`로 이름을 바꾸고, `<take_name>_<start timestamp>.mp4`로 복사한 다음(`run_config_manual.yaml`의 `output_format: mp4`; 재인코딩 없음, 약 1초, 프레임 시각 그대로; `.ts`는 삭제되며 `output_format: ts`이면 유지; ffmpeg가 없거나 실패하면 `.ts`가 남음), 결과에 `verify_hook.py`를 실행하고 창에 보여 줍니다(CSV는 파일 옆). 폰은 8 Mbps로 보내므로 테이크 하나는 초당 약 1 MB입니다(20초 ≈ 20 MB). 다음 테이크를 위해 테이크 이름은 자동으로 증가합니다(`take1` → `take2`).

**Recorder 창**은 Android 앱 화면과 같은 모양과 구조입니다(어두운 테마, 헤더, 상태 칩, 큰 타이머, 큰 시작/정지 버튼, 메시지 카드. 팔레트는 `recorder_gui.py` = `android/app/src/main/res/values/colors.xml`):

| 요소 | 의미 |
|---|---|
| LIVE VIEW (왼쪽) | 폰이 스트리밍하는 동안 폰 카메라의 화면, 약 15 fps (그 외에는 "Waiting for the phone ..."). ffmpeg가 로컬 수신기에서(RTSP) 스트림을 다시 읽어 옵니다. 보기만 할 뿐 녹화나 폰에는 영향을 주지 않습니다. PATH에 `ffmpeg`가 필요하며, `live_view: false`로 끌 수 있습니다. 지연: 폰 렌즈에 빛이 들어온 뒤 PC에서 프레임이 디코딩되기까지 약 0.31초 측정됨(293~348 ms, SRT 버퍼 120 ms 포함), 창이 그리기까지 최대 0.03초 추가(30 fps). ffmpeg 체인은 낮은 지연으로 튜닝되어 있고(단일 스레드 디코딩, fps 필터 없음, 프레임마다 flush), 나머지는 폰의 카메라/인코더와 녹화와 공유하는 SRT 버퍼입니다. 폰이 영상만 보내므로 화면에 소리는 없습니다 |
| STREAM 칩 | `CONNECTED` (초록) / `NOT CONNECTED` (회색) / `LOST` (주황, 테이크 중 스트림이 끊김) / `NO MEDIAMTX` (빨강, Control API에 연결할 수 없음) |
| RECORDING 칩 | `IDLE` (회색) / `STARTING`, `STOPPING` (주황) / `REC` (빨강; 스트림이 끊기면 `REC (NO STREAM)`) |
| `● REC` 배지, 타이머 | 녹화 중에는 빨갛게 깜박임. 타이머는 실제 녹화 시작부터 셈 |
| 버튼 | 초록색 **Start recording**, 빨간색 **Stop recording**. 지금 쓸 수 없는 쪽은 회색 |
| 메시지 카드 | 스트림이 없을 때 Start가 비활성인 이유(`Waiting for the phone to stream ...`, 또는 API에 연결할 수 없을 때 MediaMTX를 시작하는 방법), 그 다음에는 진행 상황(시작 시각과 지연, 저장된 파일 이름, 오류) |
| LAST TAKE | 테이크의 한 줄 판정(프레임 수, 길이, fps, 간격(gap)과 추정 드롭 프레임. gap이 없으면 초록, 있으면 주황). 화면이나 콘솔에 파일 경로나 전체 보고서는 나오지 않습니다. 테이크 상세와 전체 검증 보고서를 콘솔에 출력하려면 `--verbose`로 시작하세요(`run_recorder.bat --verbose`) |
| 푸터 | **Open folder** (이 경로의 녹화 폴더), **Always on top** |

`run_config_manual.yaml`:

| 키 | 의미 |
|---|---|
| `record_mode` | `auto` 또는 `manual` |
| `api_host`, `api_port`, `api_timeout_s` | Control API 주소 (로컬호스트) |
| `poll_interval_s` | 창이 API를 폴링하는 주기 |
| `lock_on_record` | `true`: **Start recording**이 먼저 폰의 노출과 초점을 고정하고(첫 녹화 프레임부터 이미 고정되도록), **Stop recording**이 다시 해제합니다. 사용자가 폰에서 길게 눌러 만든 고정은 건드리지 않음 |
| `phone_port`, `phone_address`, `phone_timeout_s` | 폰 앱의 원격 제어 포트(앱의 `REMOTE_PORT`와 같아야 함). `phone_address`가 비어 있으면 수신기의 SRT 연결에서 가져옴. 호출 한 번의 타임아웃 |
| `live_view` | 창의 카메라 실시간 화면 (ffmpeg 필요) |
| `mediamtx_config_manual` | 수동 모드에서 사용하는 설정 |
| `recordings_dir` | 녹화 폴더(`pc/` 기준 상대 경로). yml의 `recordPath`와 같아야 함 |
| `start_wait_timeout_s` | Start가 녹화 시작을 기다리는 시간 |
| `segment_settle_s`, `segment_wait_timeout_s` | Stop: 파일이 `segment_settle_s` 동안 변하지 않으면 세그먼트가 닫힌 것으로 간주 (`recordPartDuration`, 1초보다 크게 유지) |
| `timestamp_format` | 이름이 바뀐 파일에 들어가는 타임스탬프 (실제 녹화 시작 시각) |

동작 방식: `recorder_control.py`(직접 스크립트에서 사용 가능: `Recorder(settings).start("take1")`, `.stop()`, `.status()`)는 `PATCH /v3/config/paths/patch/<path>`로 경로의 `record`를 전환하고, `GET /v3/paths/get/<path>`로 스트림 상태를, `GET /v3/recordings/get/<path>`로 세그먼트의 실제 시작 시각을 읽습니다. `recorder_gui.py`가 창이고, `manual_mode.py`는 `run_pc.py`가 사용하는 연결 코드입니다.

참고:

- 이전에 시작한 수신기가 아직 실행 중이면(창은 이미 닫혔을 수 있음) 다시 시작해도 두 번째 수신기를 **띄우지 않습니다**. Recorder 창만 열고, 실행 중인 수신기는 계속 돕니다(자기 창에서 Ctrl+C로 종료). 포트를 다른 것이 쓰고 있으면(예: 이전의 `record_mode: auto` 수신기) 어떤 포트인지 알려 주고 종료합니다.
- 녹화 중에 Recorder 창을 닫으면 먼저 녹화를 정지하고, **그다음 런처가 수신기도 종료**하여 폰의 스트림이 끝납니다(폰에 `CONNECTION LOST` 표시). 런처에서 Ctrl+C를 누르면 MediaMTX가 종료됩니다(끝나지 않은 테이크는 MediaMTX가 붙인 파일 이름 그대로 남고, 이름 변경이나 검증은 되지 않음).
- **노출/초점 고정**을 하려면 PC가 폰에 닿을 수 있어야 합니다(같은 Wi-Fi/LAN, TCP `phone_port` 8898, Wi-Fi의 "클라이언트 격리" 없음). 폰은 자신이 스트림을 보내는 PC에만, 그리고 스트리밍 중에만 응답합니다. 폰에 닿지 못해도 녹화는 시작되며 메시지 카드가 그렇게 알려 줍니다(오래된 앱 버전에는 원격 제어가 없으므로 앱을 다시 설치하세요). PC 방화벽에서는 아무것도 필요 없습니다. PC가 폰에 연결하는 것이지 그 반대가 아닙니다.
- 테이크 중 스트림이 끊기면 돌아왔을 때 녹화가 저절로 재개됩니다. 이후 세그먼트는 `<take>_<timestamp>_part2.ts`, …로 저장됩니다.
- 스트림에 B-프레임이 있으면(예: ffmpeg/x264 테스트 소스) 검증 단계가 테이크 끝에서 "드롭" 프레임이 하나 더 있다고 보고할 수 있습니다. 재정렬 도중에 녹화가 잘리기 때문입니다. 폰의 MediaCodec 스트림에는 보통 B-프레임이 없습니다.

## 폰 QR 코드: 설정 편집 없이 폰 연결

Android 앱에는 **Scan QR** 버튼이 있습니다. QR 코드에서 publish URL을 읽어 저장하고(앱 설정의 `stream.url`을 대체) 연결합니다. 앱은 시작할 때 스캐너를 엽니다. PC의 IP가 바뀌어도 다시 빌드할 필요가 없습니다.

1. 코드는 Recorder 창이 시작될 때 **저절로 열립니다**(이미 스트리밍 중인 폰이 있으면 열리지 않음). 다시 열려면 Recorder 창의 **Phone QR** 버튼을 누르거나 `python qr_window.py`를 실행하세요.
2. 첫 번째(초록색) 네트워크가 PC의 Wi-Fi이며 폰이 닿을 수 있는 쪽입니다. PC에 다른 네트워크(`Wired` …)가 있으면 그것들도 이름으로 나열됩니다. 폰이 그 네트워크에 있을 때만 다른 것을 고르세요. 주소는 표시되지 않습니다.
3. 앱을 시작하고(스캐너가 저절로 열리며, 아니면 **Scan QR**을 탭) 카메라를 코드에 향합니다. 앱이 주소를 저장하고 바로 스트리밍을 시작합니다. 녹화는 여전히 Recorder 창에서 직접 시작합니다.
4. 폰이 연결되면 QR 창이 **저절로 닫힙니다**(수신기의 Control API를 보고 있어서 수동 모드에서도 동작합니다. 창을 열 때 이미 연결되어 있던 폰은 창을 닫지 않음).

코드에는 `run_pc.py`가 출력하는 URL, `srt://<pc ip>:<srt_port>?streamid=publish:<stream_path>`가 들어 있습니다. `pip install segno`가 필요하며, 없으면 창에 URL과 설치 안내가 표시됩니다.

## 폰 publish URL

```
srt://<PC_IP>:8890?streamid=publish:<path>
```

`<PC_IP>`는 PC의 LAN 주소(폰과 PC가 같은 네트워크)이고, `<path>`는 임의의 이름입니다. 예를 들어 `cam` → 녹화는 `recordings/cam/`에 저장됩니다. 수신기는 비밀번호 없이 모든 발행자를 받아들입니다(MediaMTX 기본 `any` 사용자). 신뢰할 수 있는 네트워크에서만 사용하세요.

## Windows 방화벽

SRT는 **UDP 8890**(또는 `srt_port`)으로 동작합니다. 처음 시작할 때 Windows가 `mediamtx.exe`를 허용할지 물을 수 있습니다. **프라이빗** 네트워크에서 허용하세요. 아니면 직접 규칙을 추가합니다(관리자 PowerShell):

```powershell
New-NetFirewallRule -DisplayName "MediaMTX SRT" -Direction Inbound -Protocol UDP -LocalPort 8890 -Action Allow -Profile Private
# 다른 기기에서 RTSP로 보려는 경우에만:
New-NetFirewallRule -DisplayName "MediaMTX RTSP" -Direction Inbound -Protocol TCP -LocalPort 8554 -Action Allow -Profile Private
```

폰이 연결되지 않으면 Windows 네트워크 설정에서 Wi-Fi가 "공용"이 아니라 "프라이빗"으로 되어 있는지 확인하세요. Android 에뮬레이터는 `10.0.2.2`로 PC에 접근하며, 위 규칙 외에 방화벽 규칙이 필요하지 않습니다.

## 수동 사용 (런처 없이)

- 수신기만: 이 폴더에서 `mediamtx.exe mediamtx_run.yml` (기본 포트 8890/8554. 훅 명령은 `python verify_hook.py`).
- 아무 파일이나 직접 검증: `verify_config.yaml`의 `input_path`를 지정한 뒤 `python verify_recording.py` (또는 `python verify_hook.py <file>`).

검증 단계의 출력:

- 콘솔: 프레임 수(예상치 대비), 길이, 평균 fps, 간격의 평균/표준편차/최소/최대, gap 수와 추정 드롭 프레임(gap마다 `round(interval / nominal) - 1`), 처음 `max_gaps_printed`개의 gap.
- CSV: 간격마다 한 행 — `frame_index, pts_s, interval_s, is_gap, dropped_est`.

타임스탬프는 각 비디오 프레임의 ffprobe `best_effort_timestamp_time`이며 표시 시각(presentation time)순으로 정렬됩니다(그래서 B-프레임이 간격을 왜곡하지 않습니다).
