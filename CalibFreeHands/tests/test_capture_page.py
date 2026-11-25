"""Capture page and application shell, with a fake controller (no receiver, no phone, no HandCalib)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QObject, Signal

from calibfreehands.app.settings import AppSettings
from calibfreehands.capture import page as capture_page
from calibfreehands.capture.page import CapturePage, one_line_summary, safe_name

REPORT = """frames           : 600 (expected ~600 at 30 fps)
duration         : 20.000 s (first to last frame pts)
gaps (> 1.5 x nominal = 50.000 ms): 0, estimated dropped frames: 0"""


class FakeController(QObject):
    status = Signal(dict)
    receiver_state = Signal(str, str)
    take_started = Signal(str, dict)
    take_saved = Signal(str, list, str)
    take_failed = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = {"rtsp_port": 8554, "stream_path": "cam", "live_view": False, "lock_on_record": True}
        self.calls = []
        self.recorder = SimpleNamespace(stop=lambda: None)
        self.pc = SimpleNamespace(phone_link=SimpleNamespace(set_hold=lambda *_: None))

    def begin(self):
        self.calls.append("begin")

    def shutdown(self):
        self.calls.append("shutdown")

    def network_choices(self):
        return [("Wi-Fi", "10.0.0.2")]

    def phone_url(self, ip):
        return f"srt://{ip}:8890?streamid=publish:cam"

    def start_take(self, role, name):
        self.calls.append(("start", role, name))

    def stop_take(self, role, directory, stem):
        self.calls.append(("stop", role, Path(directory).name, stem))


class FakeCamera(QObject):
    """The PC camera's controller (webcam.WebcamController) without a camera."""

    cameras_found = Signal(list)
    camera_state = Signal(bool, str)
    frame_ready = Signal()
    take_started = Signal(str, dict)
    take_saved = Signal(str, list, str)
    take_failed = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.calls = []
        self.image = None

    def refresh_cameras(self):
        self.calls.append("refresh")

    def open_camera(self, name):
        self.calls.append(("open", name))

    def close_camera(self):
        self.calls.append("close")

    def take_frame(self):
        image, self.image = self.image, None
        return image

    def start_take(self, role, name):
        self.calls.append(("start", role, name))

    def stop_take(self, role, directory, stem):
        self.calls.append(("stop", role, Path(directory).name, stem))

    def abort_take(self, directory, stem):
        self.calls.append(("abort", Path(directory).name, stem))

    def shutdown(self):
        self.calls.append("shutdown")


@pytest.fixture
def page(tmp_path, qapp):
    settings = AppSettings(tmp_path / "settings.ini")
    widget = CapturePage(settings, lambda: tmp_path / "output", controller_factory=FakeController,
                         camera_factory=FakeCamera)
    widget.session_edit.setText("s1")
    yield widget
    widget._clock.stop()
    widget.live.stop()


def connect(page_, stream=True):
    page_._controller.status.emit({"api": True, "stream": stream, "record": False})


def test_helpers():
    assert one_line_summary(REPORT) == "600 frames  ·  20.0 s  ·  0 gaps"
    assert "could not be checked" in one_line_summary("nothing")
    assert safe_name(" my take/1 ") == "my_take_1" and safe_name("///") == "session"


def test_record_button_follows_the_connection_and_the_step(page):
    assert not page.record_button.isEnabled()
    connect(page)
    assert page.record_button.isEnabled() and "calibration" in page.record_button.text()
    connect(page, stream=False)
    assert not page.record_button.isEnabled()


def test_calibration_take_then_handcalib_then_work_take(page, monkeypatch, tmp_path):
    def fake_handcalib(self, video):  # the real one starts HandCalib in another Python
        assert Path(video).name == "calibration.mp4"
        self._on_focal(880.58)

    monkeypatch.setattr(CapturePage, "_run_handcalib", fake_handcalib)
    connect(page)
    page._on_record_clicked()
    assert page._controller.calls[-1] == ("start", "calibration", "calibration") and page._phase == "starting"
    page._controller.take_started.emit("calibration", {"locked": True, "start": None, "latency": 0.1})
    assert page.record_button.text() == "Stop recording" and page.record_button.isEnabled()
    page._on_record_clicked()
    assert page._controller.calls[-1] == ("stop", "calibration", "s1", "calibration")
    page._controller.take_saved.emit("calibration", [str(tmp_path / "calibration.mp4")], REPORT)

    assert page.focal_px == pytest.approx(880.58) and page._phase == "idle"
    assert "work" in page.record_button.text() and page.record_button.isEnabled()
    assert "880.58" in page.step_calibration.value.text()

    requests = []
    page.process_requested.connect(lambda video, focal: requests.append((video, focal)))
    page._on_record_clicked()
    assert page._controller.calls[-1] == ("start", "work", "work")
    page._controller.take_started.emit("work", {"locked": True, "start": None, "latency": 0.1})
    page._on_record_clicked()
    assert page._controller.calls[-1][:3] == ("stop", "work", "s1") and page._controller.calls[-1][3] == "work_01"
    page._controller.take_saved.emit("work", [str(tmp_path / "work_01.mp4")], REPORT)
    assert requests == [(str(tmp_path / "work_01.mp4"), pytest.approx(880.58))]
    assert page.focal_px == pytest.approx(880.58)  # reused for the next work video
    assert "600 frames" in page.message.text()


def test_work_videos_are_numbered_per_session(page, tmp_path):
    page._focal = 900.0
    page._refresh()
    folder = tmp_path / "output" / "captures" / "s1"
    folder.mkdir(parents=True)
    (folder / "work_01.mp4").touch()
    (folder / "work_02.mp4").touch()
    assert page._final_stem() == "work_03"


def test_handcalib_failure_keeps_the_calibration_step_and_explains(page):
    page._phase = "calibrating"
    page._on_focal_failed("The hand was found in only 2 of 90 frames. Record the calibration video again.")
    assert page._phase == "idle" and page.focal_px is None
    assert "again" in page.message.text() and page.card.property("state") == "error"


def test_manual_focal_skips_the_calibration_video(page, tmp_path):
    connect(page)
    page.radio_manual.setChecked(True)
    page.manual_spin.setValue(1234.5)
    assert page.focal_px == pytest.approx(1234.5)
    assert "work" in page.record_button.text() and page.step_calibration.value.text() == "NOT NEEDED"
    assert not page.recalibrate_button.isEnabled()
    saved = AppSettings(tmp_path / "settings.ini")
    assert saved.get("focal_source") == "manual" and saved.get("manual_focal_px") == pytest.approx(1234.5)


def test_recalibrate_forgets_the_focal_length(page):
    page._focal = 900.0
    page._refresh()
    assert page.recalibrate_button.isEnabled()
    page.recalibrate()
    assert page.focal_px is None and "calibration" in page.record_button.text()


def test_failed_take_returns_to_idle_with_a_message(page):
    connect(page)
    page._on_record_clicked()
    page._controller.take_failed.emit("calibration", "The recording could not be started. Is the phone connected?")
    assert page._phase == "idle" and "could not be started" in page.message.text()
    assert page.card.property("state") == "error"


def test_no_automatic_processing_when_the_box_is_unchecked(page, tmp_path):
    page._focal = 900.0
    page._refresh()
    page.auto_check.setChecked(False)
    requests = []
    page.process_requested.connect(lambda *args: requests.append(args))
    page._phase = "stopping"
    page._on_take_saved("work", [str(tmp_path / "work_01.mp4")], REPORT)
    assert requests == [] and page.last_work_video == tmp_path / "work_01.mp4"


def test_shell_hands_the_work_video_to_the_process_page(tmp_path, monkeypatch, qapp):
    from calibfreehands.app.shell import AppShell

    shell = AppShell(AppSettings(tmp_path / "settings.ini"), controller_factory=FakeController,
                     camera_factory=FakeCamera)
    started = []
    monkeypatch.setattr(shell.processing, "start_processing", lambda: started.append(True))
    video = tmp_path / "work_01.mp4"
    video.touch()
    shell.process_work_video(str(video), 880.58)

    rows = shell.processing.configuration_rows
    assert started == [True] and shell.tabs.currentWidget() is shell.processing
    assert [row.configuration().calibration_model_id for row in rows] == ["checkerboard_reference"]
    assert rows[0].configuration().reference_focal_px == pytest.approx(880.58)
    assert [shell.processing._queue_path(shell.processing.queue.item(i)) for i in range(shell.processing.queue.count())] \
        == [video.resolve()]
    shell.capture._clock.stop()
    shell.close()


def test_shell_does_not_start_a_second_run_while_processing(tmp_path, monkeypatch, qapp):
    from calibfreehands.app.shell import AppShell

    shell = AppShell(AppSettings(tmp_path / "settings.ini"), controller_factory=FakeController,
                     camera_factory=FakeCamera)
    shell.processing._processing = True
    monkeypatch.setattr(shell.processing, "start_processing", lambda: pytest.fail("must not start"))
    shell.process_work_video(str(tmp_path / "w.mp4"), 900.0)
    assert "still running" in shell.capture.message.text()
    shell.processing._processing = False
    shell.capture._clock.stop()
    shell.close()


# ---- the PC camera as the source -------------------------------------------------------------------------------
CAMERA_INFO = {"locked": True, "lock": {"focus": True, "exposure": True}, "source": "camera", "start": None,
               "latency": 0.1}


def choose_camera(page_, names=("Cam A", "Cam B")):
    page_.source_combo.setCurrentIndex(page_.source_combo.findData("camera"))
    page_._webcam.cameras_found.emit(list(names))


def test_pc_camera_source_opens_the_first_camera_and_enables_recording(page):
    connect(page)  # the phone is connected too: the record button must follow the chosen source
    choose_camera(page)
    camera = page._webcam
    assert camera.calls[:2] == ["refresh", ("open", "Cam A")]
    assert page.camera_panel.isVisibleTo(page) and page.camera_combo.count() == 2
    assert page.stream_chip.caption.text() == "CAMERA" and page.stream_chip.value.text() == "OPENING"
    assert not page.record_button.isEnabled()

    camera.camera_state.emit(True, "1920x1080 · 30 fps")
    assert page.stream_chip.value.text() == "READY" and page.record_button.isEnabled()
    assert "PC camera is ready (1920x1080 · 30 fps)." in page.message.text()
    assert not page.qr_button.isEnabled()
    saved = AppSettings(page._settings._settings.fileName())
    assert saved.get("capture_source") == "camera" and saved.get("pc_camera") == "Cam A"


def test_the_phone_is_not_touched_while_the_pc_camera_is_the_source(page):
    choose_camera(page)
    page._webcam.camera_state.emit(True, "1920x1080 · 30 fps")
    page._on_record_clicked()
    assert page._webcam.calls[-1] == ("start", "calibration", "calibration") and page._controller.calls == []
    page._webcam.take_started.emit("calibration", CAMERA_INFO)
    assert "locked on the camera" in page.message.text() and "phone" not in page.message.text()
    assert page.rec_chip.value.text() == "REC"
    page._on_record_clicked()
    assert page._webcam.calls[-1] == ("stop", "calibration", "s1", "calibration")
    assert page._controller.calls == []


def test_the_lock_message_names_white_balance_when_it_was_locked_too(page):
    choose_camera(page)
    page._webcam.camera_state.emit(True, "1920x1080 · 30 fps")
    page._on_record_clicked()
    lock = {"focus": True, "exposure": True, "white_balance": True}
    page._webcam.take_started.emit("calibration", {**CAMERA_INFO, "lock": lock})
    assert "Exposure, focus and white balance are locked on the camera." in page.message.text()
    page._phase = "idle"
    page._on_record_clicked()  # a camera without a white-balance control: focus and exposure still count
    page._webcam.take_started.emit("calibration", {**CAMERA_INFO, "lock": {**lock, "white_balance": False}})
    assert "Exposure and focus are locked on the camera." in page.message.text()


def test_a_camera_that_refuses_the_lock_is_said_so(page):
    choose_camera(page)
    page._webcam.camera_state.emit(True, "1920x1080 · 30 fps")
    page._on_record_clicked()
    page._webcam.take_started.emit("calibration", {**CAMERA_INFO, "locked": False})
    assert "did not accept the exposure/focus lock" in page.message.text()
    page._phase = "idle"
    page._on_record_clicked()
    page._webcam.take_started.emit("calibration", {**CAMERA_INFO, "locked": None})
    assert "lock" not in page.message.text().replace("recording", "")  # lock switched off: nothing to report


def test_pictures_of_the_camera_reach_the_live_view_and_the_phone_does_not_overwrite_them(page):
    from PySide6.QtGui import QColor, QImage

    choose_camera(page)
    image = QImage(64, 36, QImage.Format.Format_RGB888)
    image.fill(QColor("red"))
    page._webcam.image = image
    page._webcam.frame_ready.emit()
    assert page.live._has_picture and not page.live.pixmap().isNull()
    connect(page)  # a phone status arrives: the live view keeps showing the camera
    connect(page, stream=False)
    assert page.live._has_picture and not page.live.pixmap().isNull()


def test_changing_the_source_or_the_camera_forgets_the_focal_length_and_says_why(page):
    page._focal = 880.0
    page._refresh()
    choose_camera(page)
    assert page.focal_px is None and "calibration" in page.record_button.text()
    page._webcam.camera_state.emit(True, "1920x1080 · 30 fps")
    assert "belongs to one camera" in page.message.text()

    page._focal = 900.0  # HandCalib has found one for Cam A; another camera has another lens
    page.camera_combo.setCurrentIndex(1)
    assert page.focal_px is None and page._webcam.calls[-1] == ("open", "Cam B")


def test_a_typed_focal_length_survives_a_change_of_camera(page):
    page.radio_manual.setChecked(True)
    page.manual_spin.setValue(1234.5)
    choose_camera(page)
    assert page.focal_px == pytest.approx(1234.5)
    page.source_combo.setCurrentIndex(page.source_combo.findData("phone"))
    assert page.focal_px == pytest.approx(1234.5)


def test_no_camera_found_is_explained(page):
    choose_camera(page, names=())
    assert page.live.text() == "No PC camera was found."
    assert "Connect a webcam" in page.message.text() and page.card.property("state") == "error"
    assert page.camera_combo.count() == 1 and not page.camera_combo.isEnabled()
    assert not page.record_button.isEnabled()
    page.camera_refresh.click()
    assert page._webcam.calls.count("refresh") == 2


def test_refresh_does_not_reopen_the_camera_that_is_running(page):
    choose_camera(page)
    page._webcam.camera_state.emit(True, "1920x1080 · 30 fps")
    page._webcam.cameras_found.emit(["Cam A", "Cam B"])  # same list again
    assert page._webcam.calls.count(("open", "Cam A")) == 1
    page._webcam.cameras_found.emit(["Cam B"])  # Cam A was unplugged: the box changes, the next camera opens
    assert page._webcam.calls[-1] == ("open", "Cam B")


def test_a_failing_camera_ends_the_take_and_tells_the_reason(page):
    choose_camera(page)
    camera = page._webcam
    camera.camera_state.emit(True, "1920x1080 · 30 fps")
    page._on_record_clicked()
    camera.take_started.emit("calibration", CAMERA_INFO)
    camera.camera_state.emit(False, "The camera stopped delivering pictures.")
    assert page._phase == "idle" and page.stream_chip.value.text() == "NOT OPEN"
    assert page.message.text() == "The camera stopped delivering pictures." and page.card.property("state") == "error"
    assert not page.record_button.isEnabled()


def test_the_source_cannot_be_changed_while_a_take_runs(page):
    choose_camera(page)
    page._webcam.camera_state.emit(True, "1920x1080 · 30 fps")
    page._on_record_clicked()
    page._webcam.take_started.emit("calibration", CAMERA_INFO)
    assert not page.source_combo.isEnabled() and not page.camera_combo.isEnabled()
    assert not page.camera_refresh.isEnabled()


def test_a_saved_pc_camera_source_starts_the_camera_not_the_phone_receiver(tmp_path, qapp):
    settings = AppSettings(tmp_path / "settings.ini")
    settings.set("capture_source", "camera")
    widget = CapturePage(settings, lambda: tmp_path, controller_factory=FakeController, camera_factory=FakeCamera)
    try:
        widget.start()
        assert widget._webcam.calls == ["refresh"] and widget._controller.calls == []
        widget.source_combo.setCurrentIndex(widget.source_combo.findData("phone"))  # back to the phone
        assert widget._controller.calls == ["begin"] and "close" in widget._webcam.calls
        assert widget.stream_chip.caption.text() == "STREAM"
    finally:
        widget._clock.stop()
        widget.live.stop()


def test_closing_the_window_during_a_pc_camera_take_keeps_the_take(page):
    choose_camera(page)
    page._webcam.camera_state.emit(True, "1920x1080 · 30 fps")
    page._on_record_clicked()
    page._webcam.take_started.emit("calibration", CAMERA_INFO)
    page.shutdown()
    assert ("abort", "s1", "calibration_interrupted") in page._webcam.calls and page._webcam.calls[-1] == "shutdown"


def test_without_pc_camera_support_the_phone_stays_the_only_source(tmp_path, qapp):
    def broken(_parent):
        raise RuntimeError("no camera support")

    widget = CapturePage(AppSettings(tmp_path / "settings.ini"), lambda: tmp_path, controller_factory=FakeController,
                         camera_factory=broken)
    assert not widget.source_combo.isEnabled() and widget._webcam is None
    connect_page = widget._controller.status
    connect_page.emit({"api": True, "stream": True, "record": False})
    assert widget.record_button.isEnabled()
    widget._clock.stop()
    widget.live.stop()


def test_capture_page_explains_a_missing_receiver_instead_of_crashing(tmp_path, qapp):
    def broken(_parent):
        raise ImportError("The PC receiver folder was not found")

    widget = CapturePage(AppSettings(tmp_path / "settings.ini"), lambda: tmp_path, controller_factory=broken)
    assert "not available" in widget.message.text() and not widget.record_button.isEnabled()
    widget._clock.stop()
    widget.live.stop()
