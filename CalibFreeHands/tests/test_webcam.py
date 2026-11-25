"""The PC camera: device list, take encoder (real ffmpeg), exposure/focus lock, camera thread and controller.

No camera is needed: a fake capture object produces the pictures. ffmpeg is the real one (tests that need it are
skipped without it)."""

import csv
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from calibfreehands.capture import webcam
from calibfreehands.capture.page import one_line_summary
from calibfreehands.capture.webcam import (
    CameraWorker, CaptureError, TakeEncoder, WebcamController, analyze_times, lock_controls, parse_dshow_video_devices,
    report_text, unlock_controls,
)

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg is not installed")

NEW_LAYOUT = """\
[dshow @ 000001a2b3c4d5e0] "Integrated Camera" (video)
[dshow @ 000001a2b3c4d5e0]   Alternative name "@device_pnp_\\\\?\\usb#vid_0c45&pid_6a10#6&1f2e3d4c&0&2#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\\global"
[dshow @ 000001a2b3c4d5e0] "Logi C920" (video)
[dshow @ 000001a2b3c4d5e0]   Alternative name "@device_pnp_\\\\?\\usb#vid_046d&pid_082d#7&aa&0&2#{65e8773d}\\global"
[dshow @ 000001a2b3c4d5e0] "Microphone (Realtek Audio)" (audio)
[dshow @ 000001a2b3c4d5e0]   Alternative name "@device_cm_{33D9A762}\\wave_{1B2C}"
"""

OLD_LAYOUT = """\
[dshow @ 0000020b2c3d4e50] DirectShow video devices (some may be both video and audio devices)
[dshow @ 0000020b2c3d4e50]  "USB2.0 HD UVC WebCam"
[dshow @ 0000020b2c3d4e50]     Alternative name "@device_pnp_\\\\?\\usb#vid_0bda"
[dshow @ 0000020b2c3d4e50] DirectShow audio devices
[dshow @ 0000020b2c3d4e50]  "Microphone Array"
[dshow @ 0000020b2c3d4e50]     Alternative name "@device_cm_{33D9A762}"
"""


# What ffmpeg 9 really prints (a Logitech StreamCam on a Korean Windows): the lines start with [in#0 @ ...]
FFMPEG9_LAYOUT = """\
[in#0 @ 0000022a3800c840] "Logitech StreamCam" (video)
[in#0 @ 0000022a3800c840]   Alternative name "@device_pnp_\\\\?\\usb#vid_046d&pid_0893&mi_00#6&2996507b&0&0000#{65e8773d}\\global"
[in#0 @ 0000022a3800c840] "마이크(2- Logitech StreamCam)" (audio)
[in#0 @ 0000022a3800c840]   Alternative name "@device_cm_{33D9A762}\\wave_{D302B18C}"
Error opening input file dummy.
"""


def pump(condition, timeout=20.0):
    """Process Qt events (queued signals of the worker threads) until *condition()* holds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    QApplication.processEvents()
    return bool(condition())


class FakeCapture:
    """What cv2.VideoCapture gives: pictures at a steady rate, properties that remember what was set."""

    def __init__(self, width=320, height=240, fps=30.0, fail_after=None):
        self.width, self.height, self.fps = width, height, fps
        self.fail_after = fail_after
        self.count = 0
        self.sets: list[tuple] = []
        self.released = False

    def read(self):
        if self.fail_after is not None and self.count >= self.fail_after:
            time.sleep(0.005)
            return False, None
        time.sleep(1.0 / self.fps)
        self.count += 1
        frame = np.zeros((self.height, self.width, 3), np.uint8)
        frame[:, :, 1] = (self.count * 7) % 255  # the picture changes, the encoder has something to encode
        return True, frame

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_EXPOSURE: -5.0,
                cv2.CAP_PROP_WHITE_BALANCE_BLUE_U: 3270.0}.get(prop, 0.0)  # DirectShow: the colour temperature

    def set(self, prop, value):
        self.sets.append((prop, value))
        return True

    def release(self):
        self.released = True


# ---- device list -----------------------------------------------------------------------------------------------
def test_new_ffmpeg_layout_lists_video_devices_only():
    assert parse_dshow_video_devices(NEW_LAYOUT) == ["Integrated Camera", "Logi C920"]


def test_ffmpeg_9_layout_with_the_in_prefix_is_understood():
    """Regression: only the "[dshow @ ...]" prefix was stripped, so the page found no camera although one was there."""
    assert parse_dshow_video_devices(FFMPEG9_LAYOUT) == ["Logitech StreamCam"]


def test_old_ffmpeg_layout_is_understood_too():
    assert parse_dshow_video_devices(OLD_LAYOUT) == ["USB2.0 HD UVC WebCam"]


def test_no_camera_gives_an_empty_list():
    text = ('[in#0 @ 0000021691711880] Could not enumerate video devices (or none found).\n'
            '[in#0 @ 0000021691711880] Could not enumerate audio only devices (or none found).\n'
            'Error opening input file dummy.')
    assert parse_dshow_video_devices(text) == [] and parse_dshow_video_devices("") == []


# ---- opening a camera ------------------------------------------------------------------------------------------
class FakeDirectShow:
    """cv2.VideoCapture of a webcam as DirectShow behaves: setting the size drops the format back to YUY2."""

    mjpeg_at_1080 = True
    openable = True
    last = None

    def __init__(self, index, backend):
        self.size, self.fourcc, self.order, self.released = (640, 480), "YUY2", [], False
        FakeDirectShow.last = self

    def isOpened(self):
        return self.openable

    def set(self, prop, value):
        self.order.append(prop)
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            self.size, self.fourcc = (int(value), self.size[1]), "YUY2"
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            self.size, self.fourcc = (self.size[0], int(value)), "YUY2"
        elif prop == cv2.CAP_PROP_FOURCC and (self.mjpeg_at_1080 or self.size[1] <= 720):
            self.fourcc = "MJPG"
        return True

    def read(self):
        time.sleep(1 / (30.0 if self.fourcc == "MJPG" or self.size[1] <= 720 else 4.0))  # YUY2 1080p is ~4 fps
        return True, np.zeros((self.size[1], self.size[0], 3), np.uint8)

    def release(self):
        self.released = True


def test_mjpeg_is_asked_for_last_because_the_size_resets_the_format(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", FakeDirectShow)
    capture = webcam.open_capture(0)
    assert capture.size == (1920, 1080) and capture.fourcc == "MJPG"
    assert capture.order == [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS,
                             cv2.CAP_PROP_FOURCC, cv2.CAP_PROP_BUFFERSIZE]


def test_a_camera_too_slow_at_1080p_is_opened_again_at_720p(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", FakeDirectShow)
    monkeypatch.setattr(FakeDirectShow, "mjpeg_at_1080", False)  # stays YUY2 at 1080p: about 4 pictures per second
    capture = webcam.open_capture(0)
    assert capture.size == (1280, 720) and capture.fourcc == "MJPG"
    assert not capture.released  # the one that is returned; the slow first attempt was released


def test_a_camera_that_cannot_be_opened_gives_none(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", FakeDirectShow)
    monkeypatch.setattr(FakeDirectShow, "openable", False)
    assert webcam.open_capture(0) is None and FakeDirectShow.last.released


# ---- numbers of a take -----------------------------------------------------------------------------------------
def test_gaps_are_counted_like_the_phones_verify_step():
    times = [0.0, 1 / 30, 2 / 30, 3 / 30 + 0.100, 4 / 30 + 0.100]  # one 133 ms interval = 3 missing pictures
    rows, stats = analyze_times(times, 30.0)
    assert stats["frames"] == 5 and stats["gaps"] == 1 and stats["dropped_est"] == 3
    assert [r[3] for r in rows] == [0, 0, 1, 0]
    assert rows[2][1] == pytest.approx(0.2, abs=1e-6) and rows[2][2] == pytest.approx(0.1333, abs=1e-3)


def test_the_report_has_the_layout_the_capture_page_reads():
    times = [i / 30 for i in range(601)]
    _rows, stats = analyze_times(times, 30.0)
    report = report_text(stats, 30.0, (1920, 1080))
    assert one_line_summary(report) == "601 frames  ·  20.0 s  ·  0 gaps"
    assert "1920x1080" in report


def test_lock_and_unlock_write_the_camera_controls_in_a_sensible_order():
    capture = FakeCapture()
    assert lock_controls(capture) == {"focus": True, "exposure": True, "white_balance": True}
    # focus off; exposure to manual, then the exposure the camera had been using is written back (no jump);
    # white balance: the colour temperature it has is written back, which makes it manual at that value
    assert capture.sets == [(cv2.CAP_PROP_AUTOFOCUS, 0), (cv2.CAP_PROP_AUTO_EXPOSURE, 0.25),
                            (cv2.CAP_PROP_EXPOSURE, -5.0), (cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 3270.0)]
    capture.sets.clear()
    unlock_controls(capture)
    assert capture.sets == [(cv2.CAP_PROP_AUTOFOCUS, 1), (cv2.CAP_PROP_AUTO_EXPOSURE, 0.75),
                            (cv2.CAP_PROP_AUTO_WB, 1)]


def test_white_balance_falls_back_to_plain_manual_when_the_temperature_cannot_be_read():
    class NoKelvin(FakeCapture):
        def get(self, prop):
            return -1.0 if prop == cv2.CAP_PROP_WHITE_BALANCE_BLUE_U else super().get(prop)

    capture = NoKelvin()
    assert lock_controls(capture)["white_balance"] is True
    assert capture.sets[-1] == (cv2.CAP_PROP_AUTO_WB, 0)


def test_a_camera_that_refuses_the_lock_is_reported_not_fixed():
    class Stubborn(FakeCapture):
        def set(self, prop, value):
            super().set(prop, value)
            return False

    capture = Stubborn()
    assert lock_controls(capture) == {"focus": False, "exposure": False, "white_balance": False}
    assert (cv2.CAP_PROP_EXPOSURE, -5.0) not in capture.sets  # no exposure written when manual was refused


# ---- the encoder (real ffmpeg) ---------------------------------------------------------------------------------
def probe_times(path):
    out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pts_time",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    return [float(f["pts_time"]) for f in json.loads(out)["frames"]]


@needs_ffmpeg
def test_encoder_makes_a_constant_rate_mp4_as_long_as_the_real_take(tmp_path):
    """The first pictures arrive while ffmpeg is still starting: they must not all get the same time (that lost
    most of them when ffmpeg stamped the pictures itself), and a stall must stay a stall."""
    encoder = TakeEncoder(FFMPEG, tmp_path / "take.mp4", 320, 240, nominal_fps=30.0, bitrate_kbps=1000)
    capture = FakeCapture(320, 240, 50.0)  # faster than the 30 fps of the file: some pictures are not needed
    for index in range(40):
        if index == 20:
            time.sleep(0.3)  # the camera stalls: the file must hold the picture, not squeeze the pause away
        encoder.add(capture.read()[1], time.monotonic())
    span = encoder.times[-1] - encoder.times[0]
    report = encoder.close()

    expected = round(span * 30) + 1  # slots from the first to the last used picture, the last one lasts a slot
    assert abs(encoder.written - expected) <= 1 and encoder.written > 20
    video = cv2.VideoCapture(str(tmp_path / "take.mp4"))
    assert (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (320, 240)
    assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == encoder.written
    video.release()
    if FFPROBE:
        times = probe_times(tmp_path / "take.mp4")
        steps = [b - a for a, b in zip(times, times[1:])]
        assert len(times) == encoder.written and all(step == pytest.approx(1 / 30, abs=0.002) for step in steps)
    assert f"frames           : {encoder.written} " in report and "320x240" in report
    rows = list(csv.reader(open(tmp_path / "take.intervals.csv", encoding="utf-8")))
    assert rows[0] == ["frame_index", "pts_s", "interval_s", "is_gap", "dropped_est"]
    assert len(rows) == len(encoder.times)  # header + one row per interval between the used pictures
    assert any(row[3] == "1" for row in rows[1:])  # the 0.3 s pause is a gap


@needs_ffmpeg
def test_pictures_that_piled_up_before_ffmpeg_started_are_not_lost(tmp_path):
    """20 pictures are handed over at once (about what piles up while ffmpeg starts), their own times are 33 ms apart."""
    encoder = TakeEncoder(FFMPEG, tmp_path / "burst.mp4", 320, 240, nominal_fps=30.0, bitrate_kbps=1000)
    frame = np.zeros((240, 320, 3), np.uint8)
    for index in range(20):
        encoder.add(frame, 100.0 + index / 30)
    time.sleep(0.5)
    for index in range(20, 45):  # the rest arrives in real time
        encoder.add(frame, 100.0 + index / 30)
        time.sleep(0.01)
    assert encoder.queue_dropped == 0
    encoder.close()
    video = cv2.VideoCapture(str(tmp_path / "burst.mp4"))
    assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == 45
    video.release()


@needs_ffmpeg
def test_encoder_with_odd_sized_pictures_still_makes_a_video(tmp_path):
    encoder = TakeEncoder(FFMPEG, tmp_path / "odd.mp4", 321, 241, bitrate_kbps=500)
    for index in range(12):
        encoder.add(np.zeros((241, 321, 3), np.uint8), 5.0 + index / 30)
    encoder.close()
    video = cv2.VideoCapture(str(tmp_path / "odd.mp4"))
    assert (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (322, 242)
    assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
    video.release()


@needs_ffmpeg
def test_a_take_with_less_than_two_pictures_is_not_a_video(tmp_path):
    encoder = TakeEncoder(FFMPEG, tmp_path / "short.mp4", 320, 240, bitrate_kbps=500)
    encoder.add(FakeCapture().read()[1], time.monotonic())
    with pytest.raises(CaptureError):
        encoder.close()
    assert not (tmp_path / "short.mp4").exists()


@needs_ffmpeg
def test_a_full_queue_drops_pictures_instead_of_blocking_the_camera(tmp_path):
    class SlowPipe:  # a consumer that needs 0.2 s per picture
        def write(self, data):
            time.sleep(0.2)
            return len(data)

        def close(self):
            pass

    class Slow:  # stands in for the ffmpeg process
        def __init__(self, *args, **kwargs):
            self.stdin, self.stderr = SlowPipe(), None

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    encoder = TakeEncoder(FFMPEG, tmp_path / "slow.mp4", 320, 240, max_queue=2, popen=Slow)
    frame = np.zeros((240, 320, 3), np.uint8)
    started = time.monotonic()
    for index in range(30):
        encoder.add(frame, index / 30)  # a new slot each time
    assert time.monotonic() - started < 0.5          # add() never waited for the slow consumer
    assert encoder.queue_dropped > 20 and len(encoder.times) < 10
    encoder.abort()


# ---- the camera thread and the controller ----------------------------------------------------------------------
def make_controller(tmp_path, capture, names=("Cam A", "Cam B"), **kwargs):
    opened = []

    def opener(index, width, height, fps):
        opened.append(index)
        return capture

    controller = WebcamController(ffmpeg=FFMPEG, opener=opener, lister=lambda: list(names),
                                  work_dir=tmp_path / "work", **kwargs)
    controller.opened_indexes = opened
    return controller


def open_it(controller, name="Cam A"):
    states = []
    controller.camera_state.connect(lambda ok, text: states.append((ok, text)))
    controller.refresh_cameras()
    assert pump(lambda: controller.names)
    controller.open_camera(name)
    assert pump(lambda: states)
    return states


def test_camera_opens_by_its_position_in_the_list_and_shows_pictures(tmp_path, qapp):
    capture = FakeCapture(320, 240, 30.0)
    controller = make_controller(tmp_path, capture)
    frames = []
    controller.frame_ready.connect(lambda: frames.append(controller.take_frame()))
    states = open_it(controller, "Cam B")
    try:
        assert controller.opened_indexes == [1] and states == [(True, "320x240 · 30 fps")]
        assert pump(lambda: len([f for f in frames if f is not None]) >= 3)
        image = next(f for f in frames if f is not None)
        assert image.width() == 640 and image.height() == 480  # preview is 640 wide, same shape as the picture
    finally:
        controller.shutdown()
    assert capture.released


def test_unknown_camera_name_is_reported(tmp_path, qapp):
    controller = make_controller(tmp_path, FakeCapture(), names=())
    states = []
    controller.camera_state.connect(lambda ok, text: states.append((ok, text)))
    controller.refresh_cameras()
    assert pump(lambda: controller.names == [] and controller._worker is None)
    controller.open_camera("Cam A")
    assert states == [(False, "No PC camera was found. Connect a webcam and press Refresh.")]


def test_a_camera_that_cannot_be_opened_says_so(tmp_path, qapp):
    controller = WebcamController(ffmpeg=FFMPEG, opener=lambda *a: None, lister=lambda: ["Cam A"],
                                  work_dir=tmp_path)
    states = []
    controller.camera_state.connect(lambda ok, text: states.append((ok, text)))
    controller.refresh_cameras()
    assert pump(lambda: controller.names)
    controller.open_camera("Cam A")
    assert pump(lambda: states)
    assert states == [(False, "The camera could not be opened. Is another program using it?")] and not controller.is_open
    controller.shutdown()


def test_a_camera_without_pictures_is_reported_when_it_stops_delivering(tmp_path, qapp):
    capture = FakeCapture(fail_after=5)
    controller = make_controller(tmp_path, capture)
    states = open_it(controller)
    try:
        assert pump(lambda: len(states) == 2, timeout=10)
        assert states[0][0] is True and states[1] == (False, "The camera stopped delivering pictures.")
        assert not controller.is_open
    finally:
        controller.shutdown()


@needs_ffmpeg
def test_a_take_is_recorded_locked_and_moved_into_the_session_folder(tmp_path, qapp):
    capture = FakeCapture(320, 240, 30.0)
    controller = make_controller(tmp_path, capture)
    started, saved, failed = [], [], []
    controller.take_started.connect(lambda role, info: started.append((role, info)))
    controller.take_saved.connect(lambda role, paths, report: saved.append((role, paths, report)))
    controller.take_failed.connect(lambda role, reason: failed.append((role, reason)))
    open_it(controller)
    try:
        controller.start_take("calibration", "calibration")
        assert pump(lambda: started or failed) and not failed
        role, info = started[0]
        assert role == "calibration" and info["source"] == "camera" and info["locked"] is True
        assert info["lock"] == {"focus": True, "exposure": True, "white_balance": True}
        assert (cv2.CAP_PROP_AUTOFOCUS, 0) in capture.sets and (cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) in capture.sets
        assert (cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 3270.0) in capture.sets

        time.sleep(1.2)
        session = tmp_path / "output" / "captures" / "s1"
        controller.stop_take("calibration", session, "calibration")
        assert pump(lambda: saved or failed, timeout=60) and not failed
        _role, paths, report = saved[0]
        assert [Path(p).name for p in paths] == ["calibration.mp4"]
        assert (session / "calibration.mp4").is_file() and (session / "calibration.intervals.csv").is_file()
        assert not list((tmp_path / "work").glob("*"))  # nothing is left behind in the working folder
        assert one_line_summary(report).split()[0].isdigit() and int(one_line_summary(report).split()[0]) >= 20
        # the camera is given back its automatic exposure and focus
        assert capture.sets[-3:] == [(cv2.CAP_PROP_AUTOFOCUS, 1), (cv2.CAP_PROP_AUTO_EXPOSURE, 0.75),
                                     (cv2.CAP_PROP_AUTO_WB, 1)]

        controller.start_take("work", "work")  # same camera, still open: a second take starts at once
        assert pump(lambda: len(started) == 2)
        time.sleep(0.6)
        controller.stop_take("work", session, "work_01")
        assert pump(lambda: len(saved) == 2, timeout=60)
        assert (session / "work_01.mp4").is_file()
    finally:
        controller.shutdown()


@needs_ffmpeg
def test_lock_can_be_switched_off_and_is_then_not_touched(tmp_path, qapp):
    capture = FakeCapture()
    controller = make_controller(tmp_path, capture, lock_on_record=False)
    started = []
    controller.take_started.connect(lambda role, info: started.append(info))
    open_it(controller)
    try:
        controller.start_take("work", "work")
        assert pump(lambda: started)
        assert started[0]["locked"] is None and capture.sets == []
        controller._abort_encoder()
    finally:
        controller.shutdown()


def test_a_take_cannot_start_without_an_open_camera_or_ffmpeg(tmp_path, qapp):
    failures = []
    controller = WebcamController(ffmpeg=FFMPEG, lister=lambda: [], work_dir=tmp_path)
    controller.take_failed.connect(lambda role, reason: failures.append(reason))
    controller.start_take("work", "work")
    assert pump(lambda: failures)
    assert failures == ["The recording could not be started. Is the camera open?"]

    capture = FakeCapture()
    controller = make_controller(tmp_path, capture)
    controller._ffmpeg = None
    controller.take_failed.connect(lambda role, reason: failures.append(reason))
    open_it(controller)
    try:
        controller.start_take("work", "work")
        assert pump(lambda: len(failures) == 2)
        assert failures[1] == "Recording needs ffmpeg (not found)."
    finally:
        controller.shutdown()


@needs_ffmpeg
def test_the_camera_failing_during_a_take_discards_it_without_a_crash(tmp_path, qapp):
    capture = FakeCapture(fail_after=40)  # about 1.3 s of pictures, then nothing
    controller = make_controller(tmp_path, capture)
    started, states = [], []
    controller.take_started.connect(lambda role, info: started.append(info))
    states = open_it(controller)
    try:
        controller.start_take("work", "work")
        assert pump(lambda: started)
        assert pump(lambda: len(states) == 2, timeout=15)
        assert states[1][0] is False and controller._take is None
        assert not list((tmp_path / "work").glob("*.mp4"))
    finally:
        controller.shutdown()


@needs_ffmpeg
def test_capture_page_records_a_work_video_with_the_pc_camera_from_start_to_hand_over(tmp_path, qapp):
    """The real page and the real controller (threads, signals, ffmpeg); only the camera itself is a fake."""
    from calibfreehands.app.settings import AppSettings
    from calibfreehands.capture.page import CapturePage
    from test_capture_page import FakeController

    controller = make_controller(tmp_path, FakeCapture(320, 240, 30.0))
    settings = AppSettings(tmp_path / "settings.ini")
    settings.set("capture_source", "camera")
    settings.set("focal_source", "manual")
    settings.set("manual_focal_px", 900.0)
    page = CapturePage(settings, lambda: tmp_path / "out", controller_factory=FakeController,
                       camera_factory=lambda parent: controller)
    page.session_edit.setText("s1")
    requests = []
    page.process_requested.connect(lambda video, focal: requests.append((video, focal)))
    try:
        page.start()
        assert pump(lambda: page.record_button.isEnabled()) and page.stream_chip.value.text() == "READY"
        assert "work" in page.record_button.text()  # a typed focal length: no calibration video needed
        page._on_record_clicked()
        assert pump(lambda: page._phase == "recording") and "locked on the camera" in page.message.text()
        time.sleep(1.0)
        page._on_record_clicked()
        assert pump(lambda: requests, timeout=60)

        video = tmp_path / "out" / "captures" / "s1" / "work_01.mp4"
        assert requests == [(str(video), 900.0)] and video.is_file()
        capture = cv2.VideoCapture(str(video))
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) >= 20 and capture.read()[0]
        capture.release()
        assert "frames" in page.message.text() and page._phase == "idle"
        assert page.record_button.isEnabled()  # the camera stayed open: the next take can start at once
    finally:
        page.shutdown()


def test_worker_calls_run_in_the_camera_thread_and_fail_cleanly_after_it_ended(tmp_path, qapp):
    capture = FakeCapture()
    worker = CameraWorker(0, lambda *a: capture)
    worker.start()
    assert pump(lambda: worker.size != (0, 0))
    assert worker.call(lambda cap: cap is capture) is True
    with pytest.raises(ZeroDivisionError):
        worker.call(lambda cap: 1 / 0)
    worker.requestInterruption()
    worker.wait(5000)
    with pytest.raises(RuntimeError):
        worker.call(lambda cap: None)
    assert capture.released
