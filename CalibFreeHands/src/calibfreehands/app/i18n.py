"""English -> Korean for the application window (Capture and Process pages).

The source text of every string stays English in the code. Two mechanisms apply a translation:

* :func:`localize` walks a window once and translates the static texts of its widgets (labels, buttons, check
  boxes, menu entries, tab names, tool tips and the items of combo boxes that opted in). It remembers the English
  original of each text, so switching the language back and forth is lossless and a text that the code changes later
  is taken as a new original.
* :func:`tr` translates a text at the moment it is shown (status messages, chip values). Messages that carry a
  number or a name are matched by the patterns in ``_RULES``.

What is not translated: file names, model names (they are part of the result file names), exception texts of
libraries, the 3D result viewers.
"""

from __future__ import annotations

import re
from typing import Callable

_language = "en"


def set_language(code: str) -> None:
    global _language
    _language = "ko" if str(code).strip().lower().startswith("ko") else "en"


def language() -> str:
    return _language


KO: dict[str, str] = {
    # ---- tabs, captions, buttons of the capture page
    "Capture": "촬영", "Process": "처리",
    "LIVE VIEW": "실시간 화면", "FOCAL LENGTH": "초점 거리", "SESSION": "세션",
    "STREAM": "스트림", "RECORDING": "녹화",
    "1  CALIBRATION VIDEO": "1  캘리브레이션 영상", "2  WORK VIDEO": "2  작업 영상",
    "HandCalib: record a calibration video first": "HandCalib: 먼저 캘리브레이션 영상을 촬영",
    "Manual: enter the focal length": "직접 입력: 초점 거리를 입력",
    "Start recording": "녹화 시작", "Stop recording": "녹화 중지",
    "Start calibration recording": "캘리브레이션 녹화 시작", "Start work recording": "작업 녹화 시작",
    "Phone QR": "폰 QR", "Open folder": "폴더 열기", "Recalibrate": "다시 캘리브레이션",
    "Process automatically after the work video": "작업 영상을 찍은 뒤 자동으로 처리",
    "Phone": "폰",
    # ---- camera source (phone or the PC's camera)
    "CAMERA SOURCE": "카메라 소스", "Phone (QR)": "폰 (QR)", "PC camera": "PC 카메라", "Refresh": "새로고침",
    "No camera found": "카메라를 찾지 못했습니다", "CAMERA": "카메라", "Hotspot": "핫스팟",  # "Hotspot": QR window
    "READY": "준비됨", "OPENING": "여는 중", "NOT OPEN": "열리지 않음",
    # ---- chip values
    "CONNECTED": "연결됨", "NOT CONNECTED": "연결 안 됨", "NO RECEIVER": "수신기 없음", "LOST": "끊김",
    "REC": "녹화 중", "REC (NO STREAM)": "녹화 중 (스트림 없음)", "IDLE": "대기",
    "STARTING": "시작 중", "STOPPING": "종료 중", "CALIBRATING": "캘리브레이션 중",
    "TO DO": "할 일", "NEXT": "다음", "WAITING": "대기 중", "NOT NEEDED": "필요 없음",
    # ---- live view
    "Waiting for the phone ...": "폰을 기다리는 중 ...", "Starting the live view ...": "실시간 화면을 시작하는 중 ...",
    "Reconnecting the live view ...": "실시간 화면을 다시 연결하는 중 ...",
    "Live view is off.": "실시간 화면이 꺼져 있습니다.",
    "Live view needs ffmpeg (not found).": "실시간 화면에는 ffmpeg가 필요합니다(찾을 수 없음).",
    "Opening the camera ...": "카메라를 여는 중 ...", "Looking for cameras ...": "카메라를 찾는 중 ...",
    "No PC camera was found.": "PC 카메라를 찾지 못했습니다.",
    # ---- capture page messages
    "Record a new calibration video: move the whole right hand in front of the phone.":
        "새 캘리브레이션 영상을 촬영하세요: 오른손 전체를 폰 앞에서 움직이세요.",
    "Waiting for the phone: start the app, it opens the QR scanner.":
        "폰을 기다리는 중: 앱을 켜세요. QR 스캐너가 열립니다.",
    "Waiting for the phone: start the app and scan the QR code (Phone QR).":
        "폰을 기다리는 중: 앱을 켜고 QR 코드를 스캔하세요(폰 QR).",
    "Start the calibration recording and move the whole right hand in front of the camera.":
        "캘리브레이션 녹화를 시작하고 오른손 전체를 카메라 앞에서 움직이세요.",
    "Start the work recording when you are ready.": "준비되면 작업 녹화를 시작하세요.",
    "Starting the recording ...": "녹화를 시작하는 중 ...",
    "Stopping and checking the recording ...": "녹화를 멈추고 확인하는 중 ...",
    "The recording could not be started. Is the phone connected?": "녹화를 시작하지 못했습니다. 폰이 연결되어 있나요?",
    "The recording could not be finished cleanly.": "녹화를 깔끔하게 마무리하지 못했습니다.",
    "The recording could not be checked.": "녹화를 확인하지 못했습니다.",
    "Exposure and focus are locked on the phone.": "폰의 노출과 초점이 고정되었습니다.",
    "The phone could not lock exposure/focus (update the app, or check the network).":
        "폰의 노출/초점을 고정하지 못했습니다(앱을 업데이트하거나 네트워크를 확인하세요).",
    "Calibration video is recording. Press Stop recording when it is done.":
        "캘리브레이션 영상을 녹화 중입니다. 끝나면 '녹화 중지'를 누르세요.",
    "Work video is recording. Press Stop recording when it is done.":
        "작업 영상을 녹화 중입니다. 끝나면 '녹화 중지'를 누르세요.",
    "Phone connected.": "폰이 연결되었습니다.",
    "No PC camera was found. Connect a webcam and press Refresh.":
        "PC 카메라를 찾지 못했습니다. 웹캠을 연결하고 '새로고침'을 누르세요.",
    "The camera could not be opened. Is another program using it?":
        "카메라를 열 수 없습니다. 다른 프로그램이 사용 중인가요?",
    "The camera does not deliver pictures.": "카메라에서 영상이 들어오지 않습니다.",
    "The camera stopped delivering pictures.": "카메라 영상이 중단되었습니다.",
    "The recording could not be started. Is the camera open?": "녹화를 시작하지 못했습니다. 카메라가 열려 있나요?",
    "Recording needs ffmpeg (not found).": "녹화에는 ffmpeg가 필요합니다(찾을 수 없음).",
    "Exposure and focus are locked on the camera.": "카메라의 노출과 초점이 고정되었습니다.",
    "Exposure, focus and white balance are locked on the camera.": "카메라의 노출, 초점, 화이트밸런스가 고정되었습니다.",
    "The camera did not accept the exposure/focus lock: it may keep adjusting by itself.":
        "카메라가 노출/초점 고정을 받아들이지 않았습니다. 스스로 계속 조절될 수 있습니다.",
    "The focal length belongs to one camera: record a new calibration video with this one.":
        "초점 거리는 카메라마다 다릅니다. 이 카메라로 캘리브레이션 영상을 새로 촬영하세요.",
    "Record a new calibration video: move the whole right hand in front of the camera.":
        "새 캘리브레이션 영상을 촬영하세요: 오른손 전체를 카메라 앞에서 움직이세요.",
    "Processing is still running. The work video is saved: add it on the Process page when the current run is finished.":
        "처리가 아직 진행 중입니다. 작업 영상은 저장되었으니, 현재 처리가 끝나면 처리 탭에서 추가하세요.",
    # ---- qr window
    "Open the app on the phone: it asks for this code when it starts (or tap Scan QR). The stream starts right after the scan.":
        "폰에서 앱을 켜세요. 시작할 때 이 코드를 스캔하라고 합니다(또는 'Scan QR'을 누르세요). 스캔하면 바로 스트리밍이 시작됩니다.",
    "The first network is the PC's Wi-Fi, the one the phone can usually reach.":
        "첫 번째 네트워크는 PC의 Wi-Fi이며, 폰이 보통 접속할 수 있는 쪽입니다.",
    "No network address of this PC was found. Connect the PC to the phone's Wi-Fi.":
        "이 PC의 네트워크 주소를 찾지 못했습니다. PC를 폰과 같은 Wi-Fi에 연결하세요.",
    "The QR code needs the 'segno' package (pip install segno).": "QR 코드를 만들려면 'segno' 패키지가 필요합니다(pip install segno).",
    # ---- process page
    "Open...": "열기...", "Remove": "제거", "Clear": "비우기", "Start Processing": "처리 시작",
    "View Results...": "결과 보기...", "Cancel": "취소",
    "Preset": "프리셋", "Preset Actions": "프리셋 메뉴", "Save Preset...": "프리셋 저장...",
    "Manage Presets...": "프리셋 관리...", "Import Preset...": "프리셋 가져오기...", "Export Preset...": "프리셋 내보내기...",
    "Name": "이름", "Calibration": "캘리브레이션", "Focal (px)": "초점 거리 (px)", "Reconstruction": "손 복원",
    "Action": "동작", "+ Add": "+ 추가", "Set as default": "기본값으로 설정",
    "Input / Queue": "입력 / 대기열", "Sampling": "샘플링", "Sample count": "샘플 수", "Aggregate": "집계",
    "Theme": "테마", "Language": "언어", "Output": "출력", "Browse...": "찾아보기...", "Collision": "파일 충돌",
    "Drag & Drop media here": "여기에 영상을 끌어다 놓으세요",
    "or click Open... to browse": "또는 '열기...'를 눌러 찾아보세요",
    "Original AnyCalib/WiLoR backends use the configured runtime assets.":
        "기본 AnyCalib/WiLoR 모델은 설정된 런타임 자산을 사용합니다.",
    # combo items (these combos keep their English-independent data values)
    "Uniform": "균등", "All frames": "모든 프레임", "Median": "중앙값", "Mean": "평균",
    "Ask": "확인", "Overwrite": "덮어쓰기", "Auto rename": "자동 이름 변경", "Skip": "건너뛰기",
    "System": "시스템", "Light": "라이트", "Dark": "다크",
    # job states, progress and dialog titles
    "Waiting": "대기 중", "Processing": "처리 중", "Completed": "완료", "Failed": "실패", "Cancelled": "취소됨",
    "Queue is empty": "대기열이 비어 있습니다",
    "Failed: queue processing returned no result": "실패: 대기열 처리가 결과를 반환하지 않았습니다",
    "Completed — no new results (all outputs skipped)": "완료 — 새 결과 없음(모든 출력을 건너뜀)",
    "Preparing media": "영상 준비", "Hashing source media": "원본 영상 해시 계산", "Reading video": "영상 읽는 중",
    "All result files skipped": "모든 결과 파일을 건너뜀",
    "Saving comparison results": "비교 결과 저장",
    "Estimating focal length (HandCalib)": "초점 거리 추정(HandCalib)",
    "Could not save preset": "프리셋을 저장하지 못했습니다", "Could not rename preset": "프리셋 이름을 바꾸지 못했습니다",
    "Could not delete preset": "프리셋을 삭제하지 못했습니다", "Could not import preset": "프리셋을 가져오지 못했습니다",
    "Could not export preset": "프리셋을 내보내지 못했습니다", "Could not open results": "결과를 열지 못했습니다",
    "Manage Presets": "프리셋 관리", "There are no User Presets to manage.": "관리할 사용자 프리셋이 없습니다.",
    "Windows media dialog unavailable": "Windows 파일 선택창을 쓸 수 없습니다",
    "Select output folder": "출력 폴더 선택", "Import Preset": "프리셋 가져오기",
    "Save As...": "다른 이름으로 저장...", "Rename...": "이름 바꾸기...", "Delete": "삭제",
    "Results already exist": "이미 결과가 있습니다",
    "Some results already exist in the output folder:": "출력 폴더에 이미 있는 결과가 있습니다:",
    "What should be done?": "어떻게 할까요?",
    "Skip existing": "기존 건너뛰기", "Keep both (rename)": "둘 다 유지(이름 변경)",
    # HandCalib messages
    "The calibration video has no readable frames.": "캘리브레이션 영상에서 읽을 수 있는 프레임이 없습니다.",
    "HandCalib took too long and was stopped.": "HandCalib가 너무 오래 걸려 중단되었습니다.",
    "HandCalib ended without a result.": "HandCalib가 결과 없이 끝났습니다.",
    # ---- result viewer (metric names such as MPJPE stay English: abbreviations, same as in the CSV/JSON exports)
    "CalibFreeHands - Comparison": "CalibFreeHands - 비교", "CalibFreeHands - Result": "CalibFreeHands - 결과",
    "Source Media": "원본 영상", "Shared Viewer controls": "공용 뷰어 컨트롤",
    "Separate Source": "원본 분리", "No visible results": "표시할 결과가 없습니다",
    # menu entries
    "All joints": "모든 관절", "Axis": "축", "Background": "배경", "Bones": "뼈대", "Camera": "카메라",
    "Detach All": "모두 분리", "Detach Source Media": "원본 영상 분리", "Detach Source": "원본 분리",
    "Reattach All": "모두 다시 붙이기", "Error (mm)": "오차 (mm)",
    "Export CSV…": "CSV 내보내기…", "Export JSON…": "JSON 내보내기…", "Export GT differences": "GT 차이 내보내기",
    "Fit All": "전체 맞춤", "Focus GT": "GT에 맞춤", "Focus Hands": "손에 맞춤", "Front": "정면",
    "GT Differences": "GT와의 차이", "Geometry": "형상", "Grid": "격자", "Hovered joint": "가리킨 관절",
    "Joint name": "관절 이름", "Joints": "관절", "Key joints": "주요 관절", "Label": "라벨",
    "Large": "큼", "Medium": "보통", "Small": "작음", "Subtle": "은은하게", "Size": "크기",
    "Left": "왼쪽", "Right": "오른쪽", "Top": "위", "Left hand": "왜손", "Right hand": "오른손",
    "Mesh": "메시", "None": "없음", "Off": "끄기", "Playback Stats": "재생 통계",
    "Pose / Mesh Shape": "자세 / 메시 형상", "Scale labels": "스케일 라벨", "Space Overview": "공간 개요",
    "Swap": "바꾸기", "Worst 5": "최악 5개",
    # buttons, labels, check boxes
    "Annotations": "주석", "Bar": "막대", "Bone": "뼈대", "Joint": "관절", "Confidence Display": "신뢰도 표시",
    "Error Types": "오차 종류", "Link Cameras": "카메라 연동", "More": "더보기", "Numeric value": "숫자 값",
    "Use as opacity": "투명도로 사용", "Error vectors": "오차 벡터", "Details": "자세히",
    "Per-Joint Error (21 joints)": "관절별 오차 (21개 관절)",
    "L hand detection confidence": "왼손 검출 신뢰도", "R hand detection confidence": "오른손 검출 신뢰도",
    "HUD opacity": "HUD 투명도", "Labels": "라벨", "Mesh opacity": "메시 투명도", "Playback": "재생",
    "Space": "공간", "Speed": "속도", "View": "보기", "N/A": "없음",
    # tool tips
    "Camera-space differences from the selected GT.": "선택한 GT와의 카메라 공간 차이.",
    "Choose candidate result color": "후보 결과의 색 선택", "GT color is fixed to #33FF33": "GT 색은 #33FF33으로 고정",
    "Decrease playback speed": "재생 속도 낮추기", "Increase playback speed": "재생 속도 높이기",
    "Detach Source Media and shared Viewer controls": "원본 영상과 공용 뷰어 컨트롤을 분리",
    "Move Source Media and shared controls into its own window": "원본 영상과 공용 컨트롤을 별도 창으로 옮기기",
    "Drag this workspace onto another Viewer window to combine them": "이 작업 공간을 다른 뷰어 창에 끌어다 놓으면 합쳐집니다",
    "Drag to reorder result panels": "끌어서 결과 패널 순서 변경", "Drag to resize · Right-click to Swap": "끌어서 크기 조절 · 우클릭으로 바꾸기",
    "Hide HUD": "HUD 숨기기", "Show HUD": "HUD 표시", "Keep controls visible": "컨트롤을 계속 표시",
    "Controls pinned": "컨트롤 고정됨", "Next frame": "다음 프레임", "Previous frame": "이전 프레임",
    "Play": "재생", "Pause": "일시정지", "View: Overlay": "보기: 겹치기", "View: Split": "보기: 나눠 보기",
    "Select error types shown in the HUD": "HUD에 표시할 오차 종류 선택",
    "Show or hide comparison results": "비교 결과 표시/숨기기", "Show or hide this result": "이 결과 표시/숨기기",
    "YOLO hand detector bounding-box confidence; not joint or reconstruction confidence.":
        "YOLO 손 검출기의 박스 신뢰도이며 관절이나 복원의 신뢰도가 아닙니다.",
    "YOLO hand detector bounding-box confidence": "YOLO 손 검출기의 박스 신뢰도",
    "Hand detector confidence unavailable": "손 검출기 신뢰도 없음",
    "Show selected GT versus candidate error vectors": "선택한 GT와 후보의 오차 벡터 표시",
    # status lines
    "Updating comparison metrics…": "비교 지표를 갱신하는 중…",
    "Physical": "물리", "Hand": "손", "On": "켜짐",
    "Left (mm)": "왼쪽 (mm)", "Right (mm)": "오른쪽 (mm)",
}


def _rule(pattern: str, template: str) -> tuple[re.Pattern, Callable[[re.Match], str]]:
    compiled = re.compile(pattern, re.S)
    return compiled, lambda m: template.format(*[tr(group) if group else "" for group in m.groups()])


_RULES: list[tuple[re.Pattern, Callable[[re.Match], str]]] = [
    # status line of the Process page: "<state>[ — <stage>] (<current>/<total>, <percent>)"
    (re.compile(r"^(Waiting|Processing|Completed|Failed|Cancelled)(?: — (.+))? \((\d+)/(\d+), (\d+%)\)$", re.S),
     lambda m: tr(m[1]) + (f" — {tr(m[2])}" if m[2] else "") + f" ({m[3]}/{m[4]}, {m[5]})"),
    # result viewer
    (re.compile(r"^Frame (\d+) / (\d+)((?: \| [\d.]+ s)?)((?: \| Source media (?:missing|changed))?)$"),
     lambda m: f"프레임 {m[1]} / {m[2]}" + (f" | {m[3][3:-2]}초" if m[3] else "")
     + {" | Source media missing": " | 원본 영상 없음", " | Source media changed": " | 원본 영상 변경됨"}.get(m[4], "")),
    _rule(r"^CalibFreeHands - (.+)$", "CalibFreeHands - {0}"),  # titles of detached viewer windows
    _rule(r"^Visible \((\d+)/(\d+)\)$", "표시 ({0}/{1})"),
    _rule(r"^Metric calculation failed: (.+)$", "지표 계산 실패: {0}"),
    _rule(r"^Saved (.+)$", "저장함: {0}"),
    _rule(r"^Export failed: (.+)$", "내보내기 실패: {0}"),
    _rule(r"^Per-Joint Error - (.+)$", "관절별 오차 - {0}"),
    _rule(r"^Error vectors: (.+)$", "오차 벡터: {0}"),
    _rule(r"^Vectors: (.+)$", "벡터: {0}"),
    _rule(r"^Show selected GT versus (.+) error vectors$", "선택한 GT와 {0}의 오차 벡터 표시"),
    _rule(r"^Space: (.+)$", "공간: {0}"),
    (re.compile(r"^Scale labels: (On|Off)$"), lambda m: "스케일 라벨: " + ("켜짐" if m[1] == "On" else "꺼짐")),
    _rule(r"^Source ([\d.]+) fps · Frame updates ([\d.]+) fps · ([\d.]+)× · Skipped (\d+)$",
          "원본 {0} fps · 프레임 갱신 {1} fps · {2}× · 건너뜀 {3}"),
    _rule(r"^Failed: (.+)$", "실패: {0}"),
    _rule(r"^Completed with errors \((\d+) failed\)$", "오류와 함께 완료({0}개 실패)"),
    _rule(r"^Completed — Viewer could not open: (.+)$", "완료 — 결과 뷰어를 열지 못했습니다: {0}"),
    _rule(r"^User presets could not be loaded: (.+)$", "사용자 프리셋을 불러오지 못했습니다: {0}"),
    _rule(r"^Estimating focal \((.+)\)$", "초점 거리 추정({0})"),
    _rule(r"^Reconstructing raw poses \((.+)\)$", "원시 손 자세 복원({0})"),
    _rule(r"^Creating preview \((.+)\)$", "미리보기 생성({0})"),
    _rule(r"^(\d+/\d+ .+?): (.+)$", "{0}: {1}"),
    # capture page
    _rule(r"^The recording could not be started \((.+)\)\.$", "녹화를 시작하지 못했습니다({0})."),
    _rule(r"^The recording could not be saved \((.+)\)\.$", "녹화를 저장하지 못했습니다({0})."),
    _rule(r"^PC camera is ready \((.+)\)\.$", "PC 카메라가 준비되었습니다({0})."),
    _rule(r"^The receiver ports are used by another program: (.+)$", "수신기 포트를 다른 프로그램이 사용 중입니다: {0}"),
    _rule(r"^The receiver could not be started \((.+)\)\.$", "수신기를 시작하지 못했습니다({0})."),
    _rule(r"^The capture page is not available: (.+)$", "촬영 탭을 쓸 수 없습니다: {0}"),
    _rule(r"^(\d+) frames  ·  ([\d.]+) s  ·  (\d+) gaps$", "{0}프레임  ·  {1}초  ·  gap {2}개"),
    _rule(r"^Calibration video saved \((.+)\)\. Estimating the focal length with HandCalib \.\.\.$",
          "캘리브레이션 영상을 저장했습니다({0}). HandCalib로 초점 거리를 추정하는 중 ..."),
    _rule(r"^Estimating the focal length with HandCalib \.\.\. (\d+)/(\d+) frames$",
          "HandCalib로 초점 거리를 추정하는 중 ... {0}/{1} 프레임"),
    _rule(r"^HandCalib: f = ([\d.]+) px\. Now record the work video\.$",
          "HandCalib: f = {0} px. 이제 작업 영상을 촬영하세요."),
    _rule(r"^Work video saved \((.+)\)\. Processing starts with f = ([\d.]+) px\.$",
          "작업 영상을 저장했습니다({0}). f = {1} px로 처리를 시작합니다."),
    _rule(r"^Work video saved \((.+)\)\. Use \"Process\" to run it, or record another one\.$",
          "작업 영상을 저장했습니다({0}). 처리 탭에서 실행하거나 다른 영상을 촬영하세요."),
    # HandCalib errors
    _rule(r"^The hand was found in only (\d+) of (\d+) frames\. Record the calibration video again: "
          r"keep the whole right hand in view and move it\.$",
          "{1}프레임 중 {0}프레임에서만 손이 인식되었습니다. 캘리브레이션 영상을 다시 촬영하세요: 오른손 전체가 보이게 하고 움직이세요."),
    _rule(r"^HandCalib failed: (.+)$", "HandCalib 실패: {0}"),
    _rule(r"^HandCalib failed \((.+)\)\.$", "HandCalib 실패({0})."),
    _rule(r"^The calibration video was not found: (.+)$", "캘리브레이션 영상을 찾을 수 없습니다: {0}"),
    _rule(r"^HandCalib returned an unusable focal length \((.+)\)\.$", "HandCalib가 사용할 수 없는 초점 거리를 반환했습니다({0})."),
    _rule(r"^The HandCalib Python environment was not found \((.+?)\)\. Set it in the settings \(handcalib_python\)\.$",
          "HandCalib의 Python 환경을 찾을 수 없습니다({0}). 설정의 handcalib_python에 지정하세요."),
    _rule(r"^HandCalib could not be started: (.+)$", "HandCalib를 시작하지 못했습니다: {0}"),
]


def tr(text: str) -> str:
    """The Korean text for *text* when the language is Korean, else *text* itself (unknown texts stay English)."""
    if _language == "en" or not text:
        return text
    hit = KO.get(text)
    if hit is not None:
        return hit
    for pattern, build in _RULES:
        match = pattern.match(text)
        if match:
            return build(match)
    return text


# ---- Qt: translate the static texts of a window ---------------------------------------------------------------

def _apply(obj, key: str, current: str, set_text: Callable[[str], None]) -> None:
    record = obj.property(key)
    source = record[0] if record and len(record) == 2 and current == record[1] else current
    new = tr(source)
    if new != current:
        set_text(new)
    obj.setProperty(key, [source, new])


def localize(root) -> None:
    """Translate (or, for English, restore) every static text below *root*; safe to call again at any time."""
    from PySide6.QtCore import QObject
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import (
        QAbstractButton, QComboBox, QGroupBox, QLabel, QLineEdit, QTabWidget, QWidget,
    )

    for obj in [root, *root.findChildren(QObject)]:
        if isinstance(obj, QLabel):
            if obj.text() and not obj.text().isspace():
                _apply(obj, "_i18n", obj.text(), obj.setText)
        elif isinstance(obj, QAbstractButton):
            if obj.text():
                _apply(obj, "_i18n", obj.text(), obj.setText)
        elif isinstance(obj, QGroupBox):
            _apply(obj, "_i18n", obj.title(), obj.setTitle)
        elif isinstance(obj, QAction):
            if obj.text():
                _apply(obj, "_i18n", obj.text(), obj.setText)
        elif isinstance(obj, QTabWidget):
            for index in range(obj.count()):
                _apply(obj, f"_i18n_tab{index}", obj.tabText(index), lambda t, i=index, o=obj: o.setTabText(i, t))
        elif isinstance(obj, QComboBox) and obj.property("i18n_items"):
            for index in range(obj.count()):
                _apply(obj, f"_i18n_item{index}", obj.itemText(index),
                       lambda t, i=index, o=obj: o.setItemText(i, t))
        if isinstance(obj, QLineEdit) and obj.placeholderText():
            _apply(obj, "_i18n_ph", obj.placeholderText(), obj.setPlaceholderText)
        if isinstance(obj, QWidget):
            if obj.toolTip():
                _apply(obj, "_i18n_tip", obj.toolTip(), obj.setToolTip)
            if obj.isWindow() and obj.windowTitle():
                _apply(obj, "_i18n_title", obj.windowTitle(), obj.setWindowTitle)
