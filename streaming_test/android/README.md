# Calib-free-hands — 폰 앱 (카메라 제어 검증)

앱 이름은 화면과 홈 화면에서 **Calib-free-hands**입니다. 내부적으로는 여전히 `com.camchk`입니다(패키지, `adb logcat -s CAMCHK`, `camchk_*.csv`). 표시 이름만 바뀌었습니다. 참고: git 원격에는 같은 이름의 프로그램(`main`/`capstone`의 `CalibFreeHands/`)이 이미 있습니다. 둘은 별개이며 서로 충돌하지 않습니다.

카메라를 PC로 스트리밍하고 **실제** `CaptureResult` 값을 기록하여 노출 / 초점 / 손떨림 보정 요청이 정말 적용되는지 확인하는 최소한의 Android 앱입니다(Kotlin, [RootEncoder](https://github.com/pedroSG94/RootEncoder) 2.8.2를 Gradle 의존성으로 사용, 라이브러리 소스는 수정하지 않음).

대상: Galaxy S25+ (Android 17 / API 37). 화면은 하나: 왼쪽은 프리뷰, 오른쪽은 제어 패널.

### 화면

가로 전용이지만 두 가로 방향 모두 동작합니다. 폰을 180° 돌리면 화면이 따라옵니다(`sensorLandscape`이므로 시스템 자동 회전 스위치가 꺼져 있어도 동작). 액티비티가 변경을 직접 처리하므로(`configChanges`) 카메라, 스트림, 진행 중인 연결은 재시작되지 않습니다. 스트림 자체의 방향은 RootEncoder가 처리합니다(`autoHandleOrientation`).

패널은 PC Recorder 창(`../pc/recorder_gui.py`)과 같은 모양과 구조입니다: 어두운 테마, 헤더, 상태 칩(점 + 캡션 + 값), 큰 타이머, 큰 시작/정지 버튼, 메시지 카드. 팔레트: `app/src/main/res/values/colors.xml` (`recorder_gui.py`에도 같은 값이 있음). 두 곳을 일치시켜 두세요.

| 요소 | 의미 |
|---|---|
| STREAM 칩 | `PREPARING` (회색, 카메라 시작 중) / `READY` (회색) / `CONNECTING` (주황) / `STREAMING` (초록) / `FAILED` (빨강) |
| EXPOSURE / FOCUS 칩 | `AUTO` (회색) / `LOCKED` (주황, 아래 길게 누르기 참고) / `CONFIG` (회색: `enabled`이면서 `mode: auto`인 노출/초점 모듈이 없어 고정할 것이 없음) |
| 타이머 | 연결에 성공한 뒤의 시간 (스트리밍 중에는 초록, 그 외에는 `00:00:00`) |
| 버튼 | 초록색 `Start streaming`. 연결 중에는 회색 `Cancel`(시도가 실행 중인 스트림처럼 보이지 않도록). 빨간색 `Stop streaming`은 실제로 스트리밍 중일 때만. 카메라가 준비되지 않았으면 비활성 |
| 헤더 | `Calib-free-hands`와 설정의 스트림 옵션(`SRT  1920x1080 @ 30 fps`), 그리고 PC 주소의 출처를 알려 주는 둘째 줄: `→ PC linked by QR` 또는 `→ PC from app settings` |
| Scan QR 버튼 | 헤더 오른쪽 (아래 참고). 스트리밍 중이 아닐 때만 활성 |
| 메시지 카드 | 짧고 평범한 메시지(`Camera ready.`, `Connecting to the PC ...`, `Streaming to the PC.`, `QR code scanned. Tap Start streaming to connect.`). 연결에 실패하면 빨간 테두리의 경고로 바뀝니다. **NOT CONNECTED**(PC에 닿지 못함, 흔한 원인과 함께. 공간 확보를 위해 타이머는 숨김) 또는 **CONNECTION LOST**(스트리밍 중이었는데 끊김), 그리고 토스트 |

**화면에는 주소도 오류 코드도 표시되지 않습니다**: IP 주소, 포트, URL, 라이브러리의 오류 사유, 파일 이름은 보이지 않습니다. 로그에 있습니다(`adb logcat -s CAMCHK`).

| 프리뷰 위 | 고정 중에는 `AE / AF LOCKED` 배지, 길게 누르기에 대한 안내가 8초 후 사라짐 |
|---|---|

## 설정

모든 설정은 [`app/src/main/assets/config_test_v2.yaml`](app/src/main/assets/config_test_v2.yaml)에 있습니다(앱이 이 파일을 로드합니다. 이전의 `config_test.yaml`은 남겨 두었지만 사용되지 않습니다. APK에 포함되므로 수정 후에는 다시 빌드해야 합니다).

| 섹션 | 키 |
|---|---|
| `stream` | `protocol` (`srt`\|`rtsp`), `url`, `width`, `height`, `fps`, `bitrate_kbps` (영상만: 앱에는 마이크가 없어 스트림과 녹화에 오디오가 없음) |
| `exposure` | `enabled`, `mode` (`auto`\|`lock`\|`manual`), `exposure_time_ns`, `iso` (manual에서만) |
| `focus` | `enabled`, `mode` (`auto`\|`lock`\|`manual`), `focus_distance` (디옵터, manual에서만) |
| `stabilization` | `enabled`, `eis` (`on`\|`off`), `ois` (`on`\|`off`) |
| `lock` | `wait_for_convergence` (`true`\|`false`), `lock_timeout_frames` |
| `log` | `every_n_frames` |

`enabled: false` → 그 모듈은 아무것도 하지 않고 라이브러리의 기본 동작이 유지됩니다.
`stream.url`은 PC 수신기로 설정하세요([`../pc/README.md`](../pc/README.md) 참고). 예: `srt://192.168.0.10:8890?streamid=publish:cam`. 또는 **Scan QR**(아래)을 쓰면 스캔한 주소가 다시 빌드하지 않고도 `stream.url`을 대체합니다.

모드의 의미: 노출 `auto` = `AE_MODE_ON`, `manual` = `AE_MODE_OFF` + 노출 시간 + ISO. 초점 `auto` = 연속 비디오 AF, `manual` = `AF_MODE_OFF` + `LENS_FOCUS_DISTANCE`. 제어는 카메라 세션이 실행된 뒤(첫 캡처 프레임에서, `setCustomRequest`를 통해) 적용됩니다. `lock`은 아래에서 설명합니다.

### lock 섹션

`lock` 섹션은 **`exposure.mode` 및/또는 `focus.mode`가 `lock`일 때**(그리고 그 모듈이 `enabled: true`일 때)**만** 사용됩니다. `auto`와 `manual`에서는 아무 효과가 없습니다. 노출과 초점이 함께 사용합니다.

| 키 | 의미 |
|---|---|
| `wait_for_convergence` | `true`: auto를 계속 돌리다가 안정된 뒤에 고정. `false`: 첫 프레임에서 고정 (이전 동작). |
| `lock_timeout_frames` | `wait_for_convergence: true`일 때: 안정되지 않더라도 이 프레임 수가 지나면 고정 (대비책). 0보다 커야 함. |

`wait_for_convergence: true`일 때:

- **노출**: 일반 자동 노출이 동작하다가 `CONTROL_AE_STATE`가 `CONVERGED`가 되면(또는 타임아웃에서) `CONTROL_AE_LOCK`을 설정합니다.
- **초점**: `CONTINUOUS_VIDEO`로 시작하여, `CONTROL_AF_STATE`가 초점 맞춤(`PASSIVE_FOCUSED` 또는 `FOCUSED_LOCKED`)이 되거나 타임아웃이 되면 캡처 결과에서 현재 `LENS_FOCUS_DISTANCE`를 읽고 카메라를 그 거리로 `AF_MODE_OFF`로 전환합니다.

`wait_for_convergence: false`일 때: 노출은 즉시 `AE_LOCK`을 설정하고, 초점은 트리거 없이 `AF_MODE_AUTO`를 사용합니다(렌즈가 제자리에 머묾).

### 길게 눌러 고정 (노출 / 초점)

카메라 앱의 AE/AF 잠금과 같습니다: **프리뷰를 길게 누르면** 현재 노출과 초점이 고정되고, **다시 길게 누르면** auto로 돌아갑니다. 상태 텍스트에 `locked: Exposure, Focus` / `unlocked: auto (...)`가 표시됩니다.

- 노출은 `CONTROL_AE_LOCK`으로 고정됩니다(AE 모드는 ON 유지, 현재 노출 시간과 ISO가 더 이상 변하지 않음).
- 초점은 현재 `LENS_FOCUS_DISTANCE`를 읽어 그 거리에서 `AF_MODE_OFF`로 전환합니다. 해제하면 `CONTINUOUS_VIDEO`로 돌아갑니다.
- `enabled: true`이면서 `mode: auto`인 모듈에만 적용됩니다. `manual`은 이미 고정되어 있고, `lock`은 자체 고정 로직(위)이 있으므로 길게 눌러도 아무 일도 일어나지 않습니다.
- 고정은 스트림을 시작/정지해도 유지되지만 앱을 재시작하면 유지되지 않습니다(해제 상태로 시작).
- 손떨림 보정은 독립적입니다: `stabilization.eis: off`, `ois: off`(기본값)이면 둘 다 꺼진 상태로 유지됩니다.
- 로그 줄 `AE hold=...`, `AF hold=... locked_focus_distance=...`가 전환 시점을 표시하고, 프레임별 `ae_lock`, `af_mode`, `focus_distance`와 CSV에서 그 효과를 볼 수 있습니다.

### 녹화 중 PC에서 고정

PC Recorder 창에서 **Start recording**을 누르면, 녹화가 시작되기 *전에* PC가 폰의 노출과 초점을 고정하고(길게 누르기와 같은 고정), **Stop recording**이 이를 해제합니다(`../pc/run_config_manual.yaml`의 `lock_on_record`). 메시지 카드에 `Exposure and focus locked by the PC for the recording.` / `... released by the PC.`가 표시되고, LOCKED 배지도 평소처럼 나타납니다. 사용자가 직접 길게 눌러 만든 고정은 PC가 해제하지 않으며, PC 고정이 켜진 동안 길게 누르면 그 고정을 넘겨받습니다. PC가 고정을 잡은 채 스트림이 끝나면 폰은 자동으로 돌아갑니다.

방법: 스트리밍 중에 앱이 작은 HTTP 서버를 실행합니다(`RemoteControl.kt`, TCP 포트 8898 = `REMOTE_PORT`; `POST /hold?on=1|0`). 스트림을 받는 PC(스캔한 주소의 호스트)에만 응답하고, 다른 주소에는 403을 돌려줍니다. 스트림이 멈추면 닫힙니다. 앱이 이미 가진 Android 17 로컬 네트워크 권한이 이를 포함합니다. PC는 수신기에서 폰의 주소를 읽으므로 따로 설정할 것이 없습니다.

### Scan QR: 설정 편집 없이 연결

PC에서 QR 코드를 엽니다(Recorder 창 → **Phone QR**, 또는 `python ../pc/qr_window.py`. 폰의 Wi-Fi에 속한 PC 주소를 고르세요). **앱이 시작될 때 스캐너가 저절로 열립니다**: 카메라를 코드에 향하세요. 나중에(예: 취소한 뒤나 PC가 바뀌었을 때) **Scan QR**을 탭해서 열 수도 있습니다.

- 스캐너는 Play 서비스의 Google 코드 스캐너입니다(`play-services-code-scanner`): 카메라 권한이 필요 없고 Google Play 서비스가 필요합니다. 앱은 스캔하는 동안 카메라를 놓았다가 끝나면 프리뷰를 다시 시작하므로, 스캔은 스트리밍 중이 아닐 때만 가능합니다.
- 스캔한 `srt://...` 주소는 폰에 저장되며 다음 스캔 때까지 **설정의 `stream.url`을 대체합니다**(헤더에 `(scanned)` 표시). 설정 값으로 돌아가려면 앱 데이터를 지우세요(`adb shell pm clear com.camchk`).
- 유효한 코드는 **주소를 저장하고 바로 스트리밍을 시작합니다**(`QR code scanned. Connecting to the PC ...`). 추가 탭은 필요 없습니다. 스트리밍이 녹화를 시작하지는 않습니다. 테이크 시작/정지는 PC의 Recorder 창에서 합니다. 스캔을 취소하면 프리뷰가 유지되고 **Start streaming**은 마지막으로 스캔한 주소를 사용합니다.
- 맨 처음 실행할 때 Google이 스캐너 모듈을 내려받으므로 한 번은 인터넷이 필요합니다. 없으면 메시지 카드가 그렇게 알려 줍니다.
- **PC 프로그램이 닫히면 → 스트림 종료.** Recorder 창을 닫거나(또는 `run_recorder.bat`에서 Ctrl+C) 수신기가 멈추면 폰에 알려지고 `CONNECTION LOST`가 표시됩니다. PC가 그냥 사라지면(전원 꺼짐, Wi-Fi 끊김) 폰은 PC에서 패킷이 약 6초간 오지 않을 때 알아챕니다(`serverSilenceTimeout`, `MainActivity.enableServerSilenceCheck`에서 리플렉션으로 설정. 라이브러리 자체의 `setCheckServerAlive`는 ICMP로 핑을 보내는데 PC 방화벽이 이를 막음).
- 코드는 설정된 프로토콜(`stream.protocol`, `srt`)을 써야 하며, 그 외에는 메시지와 함께 거부됩니다.

## 빌드와 설치

1. Android Studio(현재 stable): **File → Open** → 이 `android/` 폴더. Gradle 동기화를 기다립니다(JDK 17, Android SDK 37 필요. Android Studio가 `gradle-wrapper.properties`의 Gradle 9.7.1을, JitPack에서 RootEncoder를 내려받음).
2. 폰: 개발자 옵션 → USB 디버깅을 켜고 연결한 뒤 프롬프트를 수락합니다.
3. Android Studio에서 실행하거나, 직접 빌드하여 설치합니다:

   ```
   gradlew assembleDebug        (또는 Build → Build APK)
   adb install -r app/build/outputs/apk/debug/app-debug.apk
   ```

4. 처음 실행할 때 카메라(그리고 Android 17에서는 로컬 네트워크) 권한을 허용한 뒤 **Start**를 탭합니다. 마이크 권한은 필요 없습니다.

## 로그 읽기

```
adb logcat -s CAMCHK
```

카메라 시작 시 한 번 기록되는 것: 하드웨어 레벨, 기능(`1` = MANUAL_SENSOR), 사용 가능한 EIS/OIS 모드, AE 모드 + 고정 가능 여부, 노출/ISO 범위, AF 모드, 최소 초점 거리. 그 뒤로는 `every_n_frames` 프레임마다 실제 결과: `ae_mode ae_lock exposure_time_ns iso af_mode focus_distance eis_mode ois_mode sensor_timestamp_ns ae_state af_state`.

고정에 도달하면(`wait_for_convergence: true`) 그에 대한 줄이 한 번 기록됩니다:

```
AE lock: reason=converged|timeout after_frames=N exposure_time_ns=... iso=...
AF lock: reason=focused|timeout after_frames=N locked_focus_distance=... (af_state=...)
AF lock failed: no LENS_FOCUS_DISTANCE in result, staying in CONTINUOUS_VIDEO   (경고, 드물게)
```

`reason=timeout`은 상태가 `lock_timeout_frames` 안에 CONVERGED / 초점 맞춤에 도달하지 못했다는 뜻입니다.

프레임별 같은 값은 앱의 외부 파일 폴더에 있는 CSV로도 저장됩니다(파일 이름은 실행 시 상태 텍스트에 표시). 열: `frame, sensor_timestamp_ns, ae_mode, ae_lock, exposure_time_ns, iso, af_mode, focus_distance, eis_mode, ois_mode, ae_state, af_state` (`ae_state`와 `af_state`가 마지막 두 열):

```
adb shell ls /sdcard/Android/data/com.camchk/files/
adb pull /sdcard/Android/data/com.camchk/files/camchk_<timestamp>.csv .
```

열거값: AE mode 0=OFF 1=ON; AF mode 0=OFF 1=AUTO 3=CONTINUOUS_VIDEO; EIS 0=OFF 1=ON 2=PREVIEW_STABILIZATION; OIS 0=OFF 1=ON; AE state 0=INACTIVE 1=SEARCHING 2=CONVERGED 3=LOCKED 4=FLASH_REQUIRED 5=PRECAPTURE; AF state 0=INACTIVE 1=PASSIVE_SCAN 2=PASSIVE_FOCUSED 3=ACTIVE_SCAN 4=FOCUSED_LOCKED 5=NOT_FOCUSED_LOCKED 6=PASSIVE_UNFOCUSED.

## 참고

- 이 문서를 작성한 환경에서는 컴파일하거나 실행하지 않았습니다(그곳에 Android SDK가 없음). 첫 빌드에서 작은 수정이 필요할 수 있습니다.
- 많은 기기가 EIS가 켜져 있으면 OIS를 무시합니다(또는 그 반대). 기록된 결과 값이 카메라가 실제로 한 일을 보여 주며, 이 앱의 목적이 바로 그것입니다.
- 프레임 길이(1/fps)보다 긴 `manual` 노출은 프레임 레이트를 낮춥니다.
