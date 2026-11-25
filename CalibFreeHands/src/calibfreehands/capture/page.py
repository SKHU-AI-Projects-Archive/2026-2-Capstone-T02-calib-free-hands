"""The capture page: live camera picture, two takes (calibration video, work video), focal length, hand-over.

Flow (hand-motion calibration, the benchmark's ``calibrator.py``):

  1. Calibration video  - the user moves the hand in front of the phone; stopping the take starts HandCalib
                          (a few seconds to a minute) which returns the focal length in pixels.
  2. Work video         - recorded with that focal length; stopping it hands the video and the focal length to the
                          processing page (``process_requested``), which runs CalibFreeHands with Manual Focal.

The focal length can also be typed in (no calibration video then). It is reused for further work videos of the same
session until "Recalibrate". Recording is stopped by the user.

The pictures come either from the phone (QR code, SRT) or from the PC's own camera (``webcam.py``). A focal length
belongs to one camera, so changing the source or the camera forgets the one HandCalib has found.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QRadioButton, QVBoxLayout, QWidget,
)

from calibfreehands.app.i18n import tr
from calibfreehands.app.theme import current_palette
from calibfreehands.handcalib import HandCalibError, estimate_focal

from .live import LiveView

SOURCE_HANDCALIB, SOURCE_MANUAL = "handcalib", "manual"   # where the focal length comes from
SOURCE_PHONE, SOURCE_CAMERA = "phone", "camera"            # where the pictures come from
FOCAL_NOTICE = "The focal length belongs to one camera: record a new calibration video with this one."


def one_line_summary(report: str) -> str:
    """'600 frames · 20.0 s · 0 gaps' from the verify report (no file paths on screen)."""
    frames = re.findall(r"^frames\s*:\s*(\d+)", report, re.M)
    if not frames:
        return "The recording could not be checked."
    duration = sum(float(x) for x in re.findall(r"^duration\s*:\s*([\d.]+)", report, re.M))
    gaps = sum(int(x) for x in re.findall(r"\):\s*(\d+), estimated dropped frames", report))
    return f"{sum(int(x) for x in frames)} frames  ·  {duration:.1f} s  ·  {gaps} gaps"


def safe_name(name: str, fallback: str = "session") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("._")
    return cleaned or fallback


class ChipRow(QWidget):
    """dot + caption on the left, value on the right (same as the phone app and the former PC window)."""

    def __init__(self, caption: str) -> None:
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        self.dot = QLabel("●")
        self.caption = QLabel(caption)
        self.caption.setProperty("role", "caption")
        self.value = QLabel("")
        self.value.setProperty("role", "mono")
        row.addWidget(self.dot)
        row.addWidget(self.caption)
        row.addStretch(1)
        row.addWidget(self.value)
        self.set("idle", "")

    def set(self, color_name: str, text: str, dot: str = "●") -> None:
        color = getattr(current_palette(), color_name)
        self.dot.setText(dot)
        self.dot.setStyleSheet(f"color: {color}; background: transparent;")
        self.value.setText(tr(text))
        self.value.setStyleSheet(f"color: {color}; font-weight: 700; background: transparent;")


class _HandCalibThread(QThread):
    progress = Signal(int, int)
    done = Signal(float)
    failed = Signal(str)

    def __init__(self, video: Path, python: str) -> None:
        super().__init__()
        self._video, self._python = Path(video), python

    def run(self) -> None:
        try:
            result = estimate_focal(
                self._video, python=self._python or None,
                progress=lambda _stage, current, total: self.progress.emit(current, total),
                cancelled=self.isInterruptionRequested,
            )
        except InterruptedError:
            return
        except HandCalibError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"HandCalib failed ({type(exc).__name__}).")
        else:
            self.done.emit(result.focal_px)


class CapturePage(QWidget):
    process_requested = Signal(str, float)  # work video, focal length in px

    def __init__(self, settings, output_dir: Callable[[], Path], controller_factory=None, camera_factory=None,
                 parent=None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._output_dir = output_dir
        self._controller = None
        self._controller_error: str | None = None
        self._webcam = None
        self._receiver_started = False
        self._cameras: list[str] = []
        self._camera_ok = self._camera_opening = False
        self._open_camera_name = ""
        self._notice = ""  # said once with the next status message (the focal length was forgotten)
        self._phase = "idle"  # idle | starting | recording | stopping | calibrating
        self._role = "calibration"
        self._focal: float | None = None
        self._api_ok = self._stream_ok = False
        self._started_at: float | None = None
        self._calibration_thread: _HandCalibThread | None = None
        self._qr = None
        self.session_dir: Path | None = None
        self._say_parts: tuple = ()
        self._say_error = False
        self.last_work_video: Path | None = None

        try:
            from .controller import CaptureController
            self._controller = (controller_factory or CaptureController)(self)
        except Exception as exc:  # noqa: BLE001 - the page then explains what is missing instead of crashing
            self._controller_error = f"The capture page is not available: {exc}"
        try:
            if camera_factory is None:
                from .webcam import WebcamController
                camera_factory = WebcamController
            self._webcam = camera_factory(self)
        except Exception:  # noqa: BLE001 - the phone still works without the PC camera
            self._webcam = None

        self._build()
        self._load_settings()
        if self._webcam is not None:
            w = self._webcam
            w.cameras_found.connect(self._on_cameras)
            w.camera_state.connect(self._on_camera_state)
            w.frame_ready.connect(self._on_camera_frame)
            w.take_started.connect(self._on_take_started)
            w.take_saved.connect(self._on_take_saved)
            w.take_failed.connect(self._on_take_failed)
        if self._controller is not None:
            c = self._controller
            c.status.connect(self._on_status)
            c.receiver_state.connect(self._on_receiver)
            c.take_started.connect(self._on_take_started)
            c.take_saved.connect(self._on_take_saved)
            c.take_failed.connect(self._on_take_failed)
        else:
            self._say(self._controller_error or "", error=True)
        self._clock = QTimer(self)
        self._clock.setInterval(200)
        self._clock.timeout.connect(self._tick)
        self._clock.start()
        self._refresh()

    # ---- layout -----------------------------------------------------------------------------------------------
    def _build(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(22, 18, 22, 18)
        outer.setSpacing(18)

        left = QVBoxLayout()
        caption = QLabel("LIVE VIEW")
        caption.setProperty("role", "caption")
        left.addWidget(caption)
        s = self._controller.settings if self._controller else {"rtsp_port": 8554, "stream_path": "cam"}
        self.live = LiveView(s["rtsp_port"], s["stream_path"], bool(s.get("live_view", True)))
        left.addWidget(self.live)
        left.addStretch(1)
        outer.addLayout(left, 3)

        divider = QFrame()
        divider.setProperty("role", "divider")
        divider.setFixedWidth(1)
        outer.addWidget(divider)

        right = QVBoxLayout()
        right.setSpacing(10)
        title = QLabel("Calib-free-hands")
        title.setProperty("role", "title")
        sub = QLabel("Capture")
        sub.setProperty("role", "dim")
        right.addWidget(title)
        right.addWidget(sub)

        chips = QFrame()
        chips.setProperty("role", "card")
        chips_layout = QVBoxLayout(chips)
        self.stream_chip, self.rec_chip = ChipRow("STREAM"), ChipRow("RECORDING")
        chips_layout.addWidget(self.stream_chip)
        chips_layout.addWidget(self.rec_chip)
        right.addWidget(chips)

        source_caption = QLabel("CAMERA SOURCE")
        source_caption.setProperty("role", "caption")
        right.addWidget(source_caption)
        self.source_combo = QComboBox()
        self.source_combo.addItem("Phone (QR)", SOURCE_PHONE)
        self.source_combo.addItem("PC camera", SOURCE_CAMERA)
        self.source_combo.setProperty("i18n_items", True)  # item texts are translated, the data values stay
        right.addWidget(self.source_combo)
        self.camera_panel = QWidget()  # the PC camera's own controls: shown only for that source
        camera_row = QHBoxLayout(self.camera_panel)
        camera_row.setContentsMargins(0, 0, 0, 0)
        self.camera_combo = QComboBox()  # camera names are shown as they are (not translated)
        self.camera_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.camera_refresh = QPushButton("Refresh")
        camera_row.addWidget(self.camera_combo, 1)
        camera_row.addWidget(self.camera_refresh)
        right.addWidget(self.camera_panel)

        focal_caption = QLabel("FOCAL LENGTH")
        focal_caption.setProperty("role", "caption")
        right.addWidget(focal_caption)
        self.radio_handcalib = QRadioButton("HandCalib: record a calibration video first")
        self.radio_manual = QRadioButton("Manual: enter the focal length")
        self.manual_spin = QDoubleSpinBox()
        self.manual_spin.setRange(1.0, 100000.0)
        self.manual_spin.setDecimals(2)
        self.manual_spin.setSuffix(" px")
        self.manual_spin.setValue(1000.0)
        group = QButtonGroup(self)
        group.addButton(self.radio_handcalib)
        group.addButton(self.radio_manual)
        manual_row = QHBoxLayout()
        manual_row.addWidget(self.radio_manual)
        manual_row.addWidget(self.manual_spin)
        manual_row.addStretch(1)
        right.addWidget(self.radio_handcalib)
        right.addLayout(manual_row)
        self.radio_handcalib.toggled.connect(self._source_changed)
        self.manual_spin.valueChanged.connect(self._source_changed)

        steps = QFrame()
        steps.setProperty("role", "card")
        steps_layout = QVBoxLayout(steps)
        self.step_calibration, self.step_work = ChipRow("1  CALIBRATION VIDEO"), ChipRow("2  WORK VIDEO")
        steps_layout.addWidget(self.step_calibration)
        steps_layout.addWidget(self.step_work)
        right.addWidget(steps)

        session_row = QHBoxLayout()
        session_label = QLabel("SESSION")
        session_label.setProperty("role", "caption")
        self.session_edit = QLineEdit(datetime.now().strftime("%Y%m%d_%H%M"))
        session_row.addWidget(session_label)
        session_row.addWidget(self.session_edit, 1)
        right.addLayout(session_row)

        self.timer_label = QLabel("00:00:00")
        self.timer_label.setProperty("role", "timer")
        self.timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right.addWidget(self.timer_label)

        self.record_button = QPushButton("Start recording")
        self.record_button.setProperty("variant", "start")
        self.record_button.clicked.connect(self._on_record_clicked)
        right.addWidget(self.record_button)

        self.card = QFrame()
        self.card.setProperty("role", "card")
        card_layout = QVBoxLayout(self.card)
        self.message = QLabel("")
        self.message.setWordWrap(True)
        self.message.setProperty("role", "mono")
        card_layout.addWidget(self.message)
        right.addWidget(self.card)

        self.auto_check = QCheckBox("Process automatically after the work video")
        self.auto_check.setChecked(True)
        right.addWidget(self.auto_check)

        buttons = QHBoxLayout()
        self.qr_button = QPushButton("Phone QR")
        self.qr_button.clicked.connect(self.open_qr)
        self.folder_button = QPushButton("Open folder")
        self.folder_button.clicked.connect(self._open_folder)
        self.recalibrate_button = QPushButton("Recalibrate")
        self.recalibrate_button.clicked.connect(self.recalibrate)
        buttons.addWidget(self.qr_button)
        buttons.addWidget(self.folder_button)
        buttons.addWidget(self.recalibrate_button)
        buttons.addStretch(1)
        right.addLayout(buttons)
        right.addStretch(1)
        outer.addLayout(right, 2)

    # ---- settings ---------------------------------------------------------------------------------------------
    def _load_settings(self) -> None:
        manual = self._settings.get("focal_source") == SOURCE_MANUAL
        self.radio_manual.setChecked(manual)
        self.radio_handcalib.setChecked(not manual)
        self.manual_spin.setValue(float(self._settings.get("manual_focal_px")))
        saved = self._settings.get("capture_source")
        camera = saved == SOURCE_CAMERA and self._webcam is not None
        self.source_combo.setCurrentIndex(self.source_combo.findData(SOURCE_CAMERA if camera else SOURCE_PHONE))
        self.camera_panel.setVisible(camera)
        self.source_combo.currentIndexChanged.connect(self._on_capture_source_changed)
        self.camera_combo.currentIndexChanged.connect(self._on_camera_picked)
        self.camera_refresh.clicked.connect(self._on_refresh_cameras)
        if self._webcam is None:  # no PC camera support: the phone is the only source
            self.source_combo.setEnabled(False)

    def _source_changed(self, *_args) -> None:
        manual = self.radio_manual.isChecked()
        self._settings.set("focal_source", SOURCE_MANUAL if manual else SOURCE_HANDCALIB)
        self._settings.set("manual_focal_px", float(self.manual_spin.value()))
        self._settings.sync()
        if self._phase == "idle":
            self._role = "work" if self.focal_px is not None else "calibration"
        self._refresh()

    # ---- state ------------------------------------------------------------------------------------------------
    @property
    def manual(self) -> bool:
        return self.radio_manual.isChecked()

    @property
    def focal_px(self) -> float | None:
        """The focal length the next work video will be processed with, or None until HandCalib has produced one."""
        return float(self.manual_spin.value()) if self.manual else self._focal

    @property
    def _source(self) -> str:
        """Where the pictures come from: SOURCE_PHONE or SOURCE_CAMERA (the PC's webcam)."""
        return self.source_combo.currentData() or SOURCE_PHONE

    def _active_controller(self):
        return self._webcam if self._source == SOURCE_CAMERA else self._controller

    def _source_ready(self) -> bool:
        """Is there a picture source that a take could start on?"""
        if self._source == SOURCE_CAMERA:
            return self._webcam is not None and self._camera_ok
        return self._controller is not None and self._api_ok and self._stream_ok

    def recalibrate(self) -> None:
        if self._phase != "idle":
            return
        self._focal = None
        self.radio_handcalib.setChecked(True)
        self._role = "calibration"
        self._say("Record a new calibration video: move the whole right hand in front of the camera."
                  if self._source == SOURCE_CAMERA else
                  "Record a new calibration video: move the whole right hand in front of the phone.")
        self._refresh()

    def _refresh(self) -> None:
        idle = self._phase == "idle"
        self._role = "work" if self.focal_px is not None else "calibration"
        ready = self._source_ready()
        # chips
        if self._source == SOURCE_CAMERA:
            self.stream_chip.caption.setText(tr("CAMERA"))
            if self._camera_ok:
                self.stream_chip.set("ok", "READY")
            elif self._camera_opening:
                self.stream_chip.set("warn", "OPENING")
            else:
                self.stream_chip.set("idle", "NOT OPEN")
        else:
            self.stream_chip.caption.setText(tr("STREAM"))
            if self._controller is None or not self._api_ok:
                self.stream_chip.set("rec", "NO RECEIVER")
            elif self._stream_ok:
                self.stream_chip.set("ok", "CONNECTED")
            elif self._phase == "recording":
                self.stream_chip.set("warn", "LOST")
            else:
                self.stream_chip.set("idle", "NOT CONNECTED")
        if self._phase == "recording":
            self.rec_chip.set("rec", "REC" if self._source == SOURCE_CAMERA or self._stream_ok else "REC (NO STREAM)")
        elif self._phase in ("starting", "stopping"):
            self.rec_chip.set("warn", self._phase.upper())
        elif self._phase == "calibrating":
            self.rec_chip.set("warn", "CALIBRATING")
        else:
            self.rec_chip.set("idle", "IDLE")
        # steps
        if self.manual:
            self.step_calibration.set("idle", "NOT NEEDED")
        elif self._focal is not None:
            self.step_calibration.set("ok", f"f = {self._focal:.2f} px")
        elif self._phase == "calibrating":
            self.step_calibration.set("warn", "CALIBRATING")
        elif self._role == "calibration" and self._phase == "recording":
            self.step_calibration.set("rec", "RECORDING")
        else:
            self.step_calibration.set("warn", "TO DO")
        if self._role == "work" and self._phase == "recording":
            self.step_work.set("rec", "RECORDING")
        elif self.focal_px is not None:
            self.step_work.set("warn", "NEXT")
        else:
            self.step_work.set("idle", "WAITING")
        # controls
        recording = self._phase == "recording"
        label = {"calibration": "Start calibration recording", "work": "Start work recording"}[self._role]
        self.record_button.setText(tr("Stop recording" if recording else label))
        self.record_button.setProperty("variant", "stop" if recording else "start")
        self.record_button.style().unpolish(self.record_button)
        self.record_button.style().polish(self.record_button)
        self.record_button.setEnabled(recording or (idle and ready))
        for widget in (self.radio_handcalib, self.radio_manual, self.manual_spin, self.session_edit,
                       self.camera_combo, self.camera_refresh):
            widget.setEnabled(idle)
        self.source_combo.setEnabled(idle and self._webcam is not None)
        self.camera_combo.setEnabled(idle and bool(self._cameras))
        self.manual_spin.setEnabled(idle and self.manual)
        self.recalibrate_button.setEnabled(idle and not self.manual and self._focal is not None)
        self.qr_button.setEnabled(self._controller is not None and self._source == SOURCE_PHONE)
        self.timer_label.setProperty("active", recording)
        self.timer_label.style().unpolish(self.timer_label)
        self.timer_label.style().polish(self.timer_label)

    def refresh_colors(self) -> None:
        """The theme or the language changed: chips and the message are set again."""
        self._refresh()
        self.live.refresh_text()
        if not self._cameras and self.camera_combo.count():  # the placeholder is not a camera name: translate it
            self.camera_combo.setItemText(0, tr("No camera found"))
        if self._say_parts:
            self._say(*self._say_parts, error=self._say_error)

    def _say(self, *parts: str, error: bool = False) -> None:
        """Show a message (English source text, translated when it is shown). Several parts are joined by a space."""
        self._say_parts, self._say_error = parts, error
        self.message.setText(" ".join(tr(part) for part in parts if part))
        self.card.setProperty("state", "error" if error else "")
        self.card.style().unpolish(self.card)
        self.card.style().polish(self.card)

    def _tick(self) -> None:
        if self._phase == "recording" and self._started_at is not None:
            seconds = int(time.time() - self._started_at)
            self.timer_label.setText(f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}")
        elif self._phase == "idle":
            self.timer_label.setText("00:00:00")

    # ---- controller signals -----------------------------------------------------------------------------------
    def start(self) -> None:
        """Called when the page is shown for the first time: start the chosen source (receiver or PC camera)."""
        if self._source == SOURCE_CAMERA:
            self._enter_camera()
        else:
            self._start_receiver()

    def _start_receiver(self) -> None:
        """The phone's receiver is only started when the phone is the source (no MediaMTX for a PC camera alone)."""
        if self._controller is None:
            return
        if not self._receiver_started:
            self._receiver_started = True
            self._controller.begin()
        self._say("Waiting for the phone: start the app, it opens the QR scanner.", self._take_notice())

    def _take_notice(self) -> str:
        notice, self._notice = self._notice, ""
        return notice

    def _on_receiver(self, state: str, detail: str) -> None:
        if self._source != SOURCE_PHONE:
            return
        if state in ("busy", "failed"):
            self._say(detail, error=True)
        elif state == "running" and not self._stream_ok:
            self._say("Waiting for the phone: start the app and scan the QR code (Phone QR).")
            if self._qr is None:
                QTimer.singleShot(400, self.open_qr)  # like the PC window: the QR code opens by itself

    def _on_status(self, status: dict) -> None:
        was_stream = self._stream_ok
        self._api_ok, self._stream_ok = bool(status.get("api")), bool(status.get("stream"))
        if self._source == SOURCE_PHONE:
            self.live.set_stream(self._api_ok and self._stream_ok)
            if self._phase == "idle" and self._stream_ok and not was_stream:
                self._say("Phone connected.", self._take_notice(), self._next_hint())
        self._refresh()

    # ---- the PC camera ----------------------------------------------------------------------------------------
    def _on_capture_source_changed(self, _index: int = 0) -> None:
        """The source box: phone or PC camera. Not while a take runs."""
        source = self._source
        self._settings.set("capture_source", source)
        self._settings.sync()
        self._forget_focal()
        self.camera_panel.setVisible(source == SOURCE_CAMERA)
        if source == SOURCE_CAMERA:
            self._enter_camera()
        else:
            self._leave_camera()
        self._refresh()

    def _forget_focal(self) -> None:
        """A focal length belongs to one camera: the HandCalib one is dropped when the camera changes."""
        if self._focal is not None:
            self._focal = None
            self._notice = FOCAL_NOTICE

    def _enter_camera(self) -> None:
        if self._webcam is None:
            return
        self.live.use_external(True, "Looking for cameras ...")
        self._camera_ok = False
        self._say("Looking for cameras ...")  # the notice about the focal length comes with "PC camera is ready"
        self._webcam.refresh_cameras()

    def _leave_camera(self) -> None:
        if self._webcam is not None:
            self._webcam.close_camera()
        self._camera_ok = self._camera_opening = False
        self._open_camera_name = ""
        self.live.use_external(False)
        self._start_receiver()

    def _on_refresh_cameras(self) -> None:
        if self._webcam is not None and self._phase == "idle":
            self._webcam.refresh_cameras()

    def _on_cameras(self, names: list) -> None:
        """The list of PC cameras arrived: fill the box and open the saved (or the first) one."""
        self._cameras = list(names)
        blocked = self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        for name in self._cameras:
            self.camera_combo.addItem(name, name)
        if not self._cameras:
            self.camera_combo.addItem(tr("No camera found"), "")
        else:
            self.camera_combo.setCurrentIndex(max(0, self.camera_combo.findData(self._settings.get("pc_camera"))))
        self.camera_combo.blockSignals(blocked)
        if self._source != SOURCE_CAMERA:
            return
        if not self._cameras:
            self._camera_ok = self._camera_opening = False
            self.live.show_message("No PC camera was found.")
            self._say("No PC camera was found. Connect a webcam and press Refresh.", error=True)
        elif not (self._camera_ok and self.camera_combo.currentData() == self._open_camera_name):
            self._open_selected_camera()  # (a refresh while the same camera runs leaves it alone)
        self._refresh()

    def _on_camera_picked(self, _index: int = 0) -> None:
        if self._source != SOURCE_CAMERA or self._phase != "idle" or not self.camera_combo.currentData():
            return
        self._forget_focal()  # another camera: another lens
        self._open_selected_camera()
        self._refresh()

    def _open_selected_camera(self) -> None:
        name = self.camera_combo.currentData()
        if not name or self._webcam is None:
            return
        self._settings.set("pc_camera", name)
        self._settings.sync()
        self._camera_ok, self._camera_opening, self._open_camera_name = False, True, name
        self.live.show_message("Opening the camera ...")
        self._webcam.open_camera(name)

    def _on_camera_state(self, ok: bool, detail: str) -> None:
        self._camera_opening = False
        self._camera_ok = ok
        if ok:
            if self._phase == "idle":
                self._say(f"PC camera is ready ({detail}).", self._take_notice(), self._next_hint())
        else:
            if self._phase in ("starting", "recording"):  # the controller has dropped the take already
                self._phase, self._started_at = "idle", None
            self._open_camera_name = ""
            self.live.show_message(detail)
            self._say(detail, error=True)
        self._refresh()

    def _on_camera_frame(self) -> None:
        image = self._webcam.take_frame() if self._webcam is not None else None
        if image is not None and self._source == SOURCE_CAMERA:
            self.live.show_image(image)

    def _next_hint(self) -> str:
        if self._role == "calibration":
            return "Start the calibration recording and move the whole right hand in front of the camera."
        return "Start the work recording when you are ready."

    def _on_record_clicked(self) -> None:
        controller = self._active_controller()
        if controller is None:
            return
        if self._phase == "recording":
            self._phase = "stopping"
            self._say("Stopping and checking the recording ...")
            self._refresh()
            controller.stop_take(self._role, self._session_directory(), self._final_stem())
        elif self._phase == "idle":
            self._phase = "starting"
            self._say("Starting the recording ...")
            self._refresh()
            controller.start_take(self._role, self._role)

    def _on_take_started(self, role: str, info: dict) -> None:
        self._phase = "recording"
        self._started_at = time.time()
        note = self._lock_note(info)
        self._say(("Calibration" if role == "calibration" else "Work")
                  + " video is recording. Press Stop recording when it is done.", note)
        self._refresh()

    def _lock_note(self, info: dict) -> str:
        """What happened to the exposure/focus lock when the take started (the phone's or the PC camera's)."""
        if info.get("source") == "camera":
            if info.get("locked") is None:  # lock_on_record is off
                return ""
            if not info["locked"]:
                return "The camera did not accept the exposure/focus lock: it may keep adjusting by itself."
            if (info.get("lock") or {}).get("white_balance"):
                return "Exposure, focus and white balance are locked on the camera."
            return "Exposure and focus are locked on the camera."
        if info.get("locked"):
            return "Exposure and focus are locked on the phone."
        if info.get("locked") is None and self._controller.settings.get("lock_on_record"):
            return "The phone could not lock exposure/focus (update the app, or check the network)."
        return ""

    def _on_take_failed(self, role: str, reason: str) -> None:
        self._phase = "idle"
        self._started_at = None
        self._say(reason, error=True)
        self._refresh()

    def _on_take_saved(self, role: str, paths: list, report: str) -> None:
        self._started_at = None
        video = Path(paths[0])
        summary = one_line_summary(report)
        if role == "calibration":
            self._phase = "calibrating"
            self._say(f"Calibration video saved ({summary}). Estimating the focal length with HandCalib ...")
            self._refresh()
            self._run_handcalib(video)
        else:
            self._phase = "idle"
            self.last_work_video = video
            self._refresh()
            if self.auto_check.isChecked():
                self._say(f"Work video saved ({summary}). Processing starts with f = {self.focal_px:.2f} px.")
                self.process_requested.emit(str(video), float(self.focal_px))
            else:
                self._say(f"Work video saved ({summary}). Use \"Process\" to run it, or record another one.")

    # ---- HandCalib --------------------------------------------------------------------------------------------
    def _run_handcalib(self, video: Path) -> None:
        thread = _HandCalibThread(video, str(self._settings.get("handcalib_python")))
        thread.progress.connect(lambda cur, total: self._say(
            f"Estimating the focal length with HandCalib ... {cur}/{total} frames"))
        thread.done.connect(self._on_focal)
        thread.failed.connect(self._on_focal_failed)
        self._calibration_thread = thread
        thread.start()

    def _on_focal(self, focal: float) -> None:
        self._focal = float(focal)
        self._phase = "idle"
        self._say(f"HandCalib: f = {focal:.2f} px. Now record the work video.")
        self._refresh()

    def _on_focal_failed(self, message: str) -> None:
        self._phase = "idle"
        self._say(message, error=True)
        self._refresh()

    # ---- helpers ----------------------------------------------------------------------------------------------
    def _session_directory(self) -> Path:
        self.session_dir = Path(self._output_dir()) / "captures" / safe_name(self.session_edit.text())
        return self.session_dir

    def _final_stem(self) -> str:
        if self._role == "calibration":
            return "calibration"
        folder = self._session_directory()
        taken = {int(m.group(1)) for p in folder.glob("work_*") if (m := re.match(r"work_(\d+)", p.name))}
        return f"work_{max(taken, default=0) + 1:02d}"

    def open_qr(self) -> None:
        if self._controller is None:
            return
        if self._qr is not None and self._qr.isVisible():
            self._qr.raise_()
            return
        from .qr import QrDialog
        self._qr = QrDialog(self._controller, self)
        self._qr.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._qr.destroyed.connect(lambda *_: setattr(self, "_qr", None))
        self._qr.show()

    def _open_folder(self) -> None:
        folder = self._session_directory()
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def shutdown(self) -> None:
        """Window is closing: finish a running take, stop the picture, end the stream and the receiver."""
        self._clock.stop()
        if self._calibration_thread is not None and self._calibration_thread.isRunning():
            self._calibration_thread.requestInterruption()
            self._calibration_thread.wait(3000)
        self.live.stop()
        if self._webcam is not None:
            if self._phase == "recording" and self._source == SOURCE_CAMERA:
                # keep what was recorded: the take is closed and saved as <role>_interrupted
                self._webcam.abort_take(self._session_directory(), f"{self._role}_interrupted")
            self._webcam.shutdown()
        if self._controller is None:
            return
        if self._phase == "recording" and self._source == SOURCE_PHONE:
            try:  # do not leave MediaMTX recording with nobody to stop it
                self._controller.recorder.stop()
            except Exception:  # noqa: BLE001
                pass
            self._controller.pc.phone_link.set_hold(self._controller.settings, False)
        self._controller.shutdown()
