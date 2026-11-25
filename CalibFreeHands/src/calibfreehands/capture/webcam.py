"""The PC's own camera (a webcam) as a capture source, next to the phone.

The phone is received by MediaMTX, which also records it. A webcam has no server in between, so here the camera is
opened once (OpenCV, DirectShow) and every picture goes to two places:

  * the live view   - a small copy of the newest picture;
  * the take        - while a take runs, the full pictures go into an ffmpeg process that writes an H.264 ``.mp4``
                      (about 8 Mbit/s, like the phone, so 20 s are about 20 MB).

Keeping the camera open between takes means a take starts and ends without opening the camera again (no new
auto-exposure settling, no black frames). Exposure and focus are fixed while a take runs - the same idea as the
phone's AE/AF lock - because a lens that refocuses between the calibration video and the work video would change the
focal length in pixels. The camera driver decides whether it accepts that; the result is reported.

The video has a constant frame rate. The pictures are placed on that grid by the clock at which they arrived, not by
ffmpeg (which would stamp them when it happens to read them, so everything that piled up while it started would get
the same time): a picture is held until the next one arrives, so a webcam in dim light, which delivers fewer than 30
pictures per second, still gives a file as long as the real take. The real arrival times are in the
``.intervals.csv`` next to the video, in the columns the phone's verify step writes; its gaps say where pictures
were held.

Everything slow runs in worker threads. The controller emits Qt signals with the same shape as the phone's
``CaptureController`` (``take_started`` / ``take_saved`` / ``take_failed``), so the capture page treats both alike.
"""

from __future__ import annotations

import csv
import logging
import os
import queue
import re
import shutil
import statistics
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtGui import QImage

log = logging.getLogger("calibfreehands.capture")

WIDTH, HEIGHT, FPS = 1920, 1080, 30.0   # requested; the camera may deliver less, the real values are used
FALLBACK_SIZE = (1280, 720)               # tried when the camera is too slow at the requested size
MIN_RATE = 0.6                           # share of the requested rate the camera must deliver at a size
PREVIEW_WIDTH = 640
PREVIEW_FPS = 25.0
BITRATE_KBPS = 8000
GAP_THRESHOLD = 1.5                      # a gap is an interval longer than this many nominal intervals (as verify)


class CaptureError(RuntimeError):
    pass


# ---- finding the cameras ---------------------------------------------------------------------------------------
# ffmpeg puts the name of the part that writes the line in front: "[dshow @ 000...]", in ffmpeg 8 and later "[in#0 @ 000...]"
_PREFIX = re.compile(r"^\[[^\]@]*@\s*[0-9a-fA-Fx]+\]\s*")
_QUOTED = re.compile(r'^"([^"]+)"')


def parse_dshow_video_devices(text: str) -> list[str]:
    """Names of the video devices in the output of ``ffmpeg -list_devices true -f dshow -i dummy``.

    Two layouts exist: ffmpeg 5 and later write ``"Name" (video)`` per device, older ones list the names under a
    "DirectShow video devices" heading. Alternative (PnP path) names and audio devices are skipped.
    """
    names: list[str] = []
    section = ""
    for raw in text.splitlines():
        line = _PREFIX.sub("", raw).strip()
        lowered = line.lower()
        if lowered.startswith("directshow video devices"):
            section = "video"
        elif lowered.startswith("directshow audio devices"):
            section = "audio"
        elif lowered.startswith("alternative name"):
            continue
        else:
            match = _QUOTED.match(line)
            if not match:
                continue
            tail = line[match.end():].strip().lower()
            is_video = ("video" in tail) if tail.startswith("(") else section == "video"
            if is_video and match.group(1) not in names:
                names.append(match.group(1))
    return names


def list_cameras(ffmpeg: str | None = None, timeout: float = 20.0) -> list[str]:
    """The names of the PC's cameras in DirectShow order (Windows). Empty if there is none or ffmpeg is missing.

    DirectShow lists them in one fixed order, which is also the index OpenCV opens them by.
    """
    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg or os.name != "nt":
        return []
    flags = subprocess.CREATE_NO_WINDOW
    try:
        proc = subprocess.run([ffmpeg, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                              creationflags=flags, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        log.warning("the camera list could not be read", exc_info=True)
        return []
    return parse_dshow_video_devices(proc.stderr + "\n" + proc.stdout)


def configure_capture(capture, width: int, height: int, fps: float) -> None:
    """Ask the camera for width x height at *fps* in MJPEG (a webcam gives 1080p30 over USB only as MJPEG).

    The order matters: DirectShow drops the format back to the camera's default (YUY2: about 4 pictures per second at
    1080p) whenever the size is set, so MJPEG has to be asked for last. Measured on a Logitech StreamCam: MJPEG first
    gave YUY2 at 4 fps, size, rate, MJPEG gave MJPEG at 31 fps.
    """
    import cv2

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # a small buffer keeps the picture current


def measure_rate(capture, pictures: int = 8, timeout: float = 4.0) -> float:
    """Pictures per second the camera really delivers (the first one, which waits for the camera, is not counted)."""
    first = last = None
    count = 0
    deadline = time.monotonic() + timeout
    while count < pictures and time.monotonic() < deadline:
        ok, frame = capture.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue
        now = time.monotonic()
        if first is None:
            first = now
        else:
            last, count = now, count + 1
    return count / (last - first) if count >= 2 and last > first else 0.0


def open_capture(index: int, width: int = WIDTH, height: int = HEIGHT, fps: float = FPS):
    """Open camera *index* at width x height; if it delivers less than 60 % of *fps* there (a camera or USB port
    that cannot do 1080p in MJPEG), open it again at 1280x720. Returns None if the camera cannot be opened."""
    import cv2

    sizes = [(width, height)]
    if height > FALLBACK_SIZE[1]:
        sizes.append(FALLBACK_SIZE)
    for position, (w, h) in enumerate(sizes):
        capture = cv2.VideoCapture(index, getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY))
        if not capture.isOpened():
            capture.release()
            return None
        configure_capture(capture, w, h, fps)
        if position == len(sizes) - 1:
            return capture
        rate = measure_rate(capture)
        if rate >= MIN_RATE * fps:
            return capture
        log.warning("the camera delivers %.1f pictures per second at %dx%d: trying %dx%d", rate, w, h, *sizes[position + 1])
        capture.release()
        time.sleep(0.5)  # the camera needs a moment before it can be opened again
    return None


def lock_controls(capture) -> dict[str, bool]:
    """Fix focus, exposure and white balance at what the camera has chosen. Returns which of them the driver accepted.

    Verified on a Logitech StreamCam by reading the DirectShow controls (IAMCameraControl / IAMVideoProcAmp) while this
    ran: each goes from Auto to Manual and stays where it was; ``unlock_controls`` returns each to Auto.

    * focus: autofocus off keeps the lens where it is.
    * exposure: manual at the value the camera reports. On this camera the automatic exposure works through the
      gain, and the gain is frozen with it.
    * white balance: without it the colours keep drifting during the take (e.g. when the hand enters the picture).
      The colour temperature is read from the "blue" white-balance property, which the DirectShow backend of OpenCV
      ties to the temperature, and written back: that makes it manual *at the current value*. Switching to manual
      alone (AUTO_WB = 0) works too but jumps to the camera's default (3300 K -> 4000 K).
    """
    import cv2

    accepted = {"focus": bool(capture.set(cv2.CAP_PROP_AUTOFOCUS, 0))}
    exposure = capture.get(cv2.CAP_PROP_EXPOSURE)
    manual = bool(capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25))
    if manual and exposure == exposure:  # not NaN
        capture.set(cv2.CAP_PROP_EXPOSURE, exposure)
    accepted["exposure"] = manual
    kelvin = capture.get(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U)
    if kelvin and kelvin > 0:
        accepted["white_balance"] = bool(capture.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, kelvin))
    else:  # the temperature cannot be read: switch to manual and take the camera's default
        accepted["white_balance"] = bool(capture.set(cv2.CAP_PROP_AUTO_WB, 0))
    return accepted


def unlock_controls(capture) -> None:
    import cv2

    capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # DirectShow: 0.75 = auto
    capture.set(cv2.CAP_PROP_AUTO_WB, 1)


# ---- the take --------------------------------------------------------------------------------------------------
def analyze_times(times: list[float], nominal_fps: float) -> tuple[list[tuple], dict]:
    """(rows, stats) from the arrival times of the pictures; the same measures verify_recording.py reports."""
    nominal = 1.0 / nominal_fps
    limit = GAP_THRESHOLD * nominal
    start = times[0]
    rows, gaps, dropped = [], 0, 0
    for index, (a, b) in enumerate(zip(times, times[1:]), start=1):
        interval = b - a
        is_gap = interval > limit
        missing = max(1, round(interval / nominal) - 1) if is_gap else 0
        gaps += is_gap
        dropped += missing
        rows.append((index, b - start, interval, int(is_gap), missing))
    duration = times[-1] - start
    return rows, {
        "frames": len(times), "duration_s": duration, "gaps": gaps, "dropped_est": dropped,
        "mean_fps": (len(times) - 1) / duration if duration > 0 else float("nan"),
        "interval_mean_ms": statistics.fmean(r[2] for r in rows) * 1000,
        "interval_max_ms": max(r[2] for r in rows) * 1000,
        "gap_limit_ms": limit * 1000, "expected_frames": round(duration / nominal) + 1,
    }


def report_text(stats: dict, nominal_fps: float, size: tuple[int, int], queue_dropped: int = 0) -> str:
    """The summary of a take in the layout of the phone's verify report (the capture page reads the same lines)."""
    lines = [
        f"frames           : {stats['frames']} (expected ~{stats['expected_frames']} at {nominal_fps:g} fps)",
        f"duration         : {stats['duration_s']:.3f} s (first to last frame time)",
        f"mean fps         : {stats['mean_fps']:.3f} (nominal {nominal_fps:g})",
        f"resolution       : {size[0]}x{size[1]}",
        f"interval mean/max: {stats['interval_mean_ms']:.3f} / {stats['interval_max_ms']:.3f} ms",
        f"gaps (> {GAP_THRESHOLD:g} x nominal = {stats['gap_limit_ms']:.3f} ms): "
        f"{stats['gaps']}, estimated dropped frames: {stats['dropped_est']}",
    ]
    if queue_dropped:
        lines.append(f"not encoded in time: {queue_dropped} frames")
    return "\n".join(lines)


def write_intervals_csv(path: Path, rows: list[tuple]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "pts_s", "interval_s", "is_gap", "dropped_est"])
        writer.writerows(rows)


class TakeEncoder:
    """An ffmpeg process that encodes the pictures of one take into a constant-frame-rate H.264 .mp4.

    Output frame k belongs to the time ``start + k / fps``. A picture that arrives in slot s is written for every
    slot from the previous picture's slot up to s - 1, i.e. it is held until the next picture comes; a picture that
    arrives in a slot that is already taken (camera faster than ``fps``) is not used.

    ``add`` never blocks the camera: the pictures go into a short queue that a writer thread feeds to ffmpeg; if
    ffmpeg cannot keep up and the queue is full, the picture is skipped (the previous one is held longer).
    """

    def __init__(self, ffmpeg: str, path: Path, width: int, height: int, nominal_fps: float = FPS,
                 bitrate_kbps: int = BITRATE_KBPS, max_queue: int = 24, popen=subprocess.Popen) -> None:
        self.path = Path(path)
        self.size = (int(width), int(height))
        self.nominal_fps = float(nominal_fps) if nominal_fps and nominal_fps > 0 else FPS
        self.times: list[float] = []  # arrival times of the pictures that were used
        self.queue_dropped = 0
        self.written = 0              # frames in the file
        self._start: float | None = None
        self._held = None             # (picture, slot): waits for the next picture to know how long it lasts
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._error: Exception | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rate = f"{int(bitrate_kbps)}k"
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "rawvideo", "-pix_fmt", "bgr24", "-video_size", f"{self.size[0]}x{self.size[1]}",
                   "-framerate", f"{self.nominal_fps:g}", "-i", "-",
                   "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                   "-c:v", "libx264", "-preset", "veryfast", "-b:v", rate, "-maxrate", rate,
                   "-bufsize", f"{int(bitrate_kbps) * 2}k", "-g", "30", "-pix_fmt", "yuv420p",
                   "-fps_mode", "passthrough", "-movflags", "+faststart", str(self.path)]
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self._proc = popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           creationflags=flags)
        self._writer = threading.Thread(target=self._write_loop, name="take-encoder", daemon=True)
        self._writer.start()

    def add(self, frame, arrived: float) -> None:
        if self._error is not None:
            return
        if self._start is None:
            self._start = arrived
        slot = int((arrived - self._start) * self.nominal_fps + 0.5)
        if self._held is not None:
            held, held_slot = self._held
            if slot <= held_slot:  # this slot is taken already
                return
            try:
                self._queue.put_nowait((held, slot - held_slot))
            except queue.Full:
                self.queue_dropped += 1  # the held picture simply lasts longer
                return
        self._held = (frame, slot)
        self.times.append(arrived)

    def _write_loop(self) -> None:
        stdin = self._proc.stdin
        while True:
            item = self._queue.get()
            if item is None:
                break
            if self._error is None:  # after an error the queue is still emptied so nobody waits for it
                frame, repeat = item
                data = memoryview(frame).cast("B")
                try:
                    for _ in range(repeat):
                        stdin.write(data)
                        self.written += 1
                except (OSError, ValueError) as exc:
                    self._error = exc
        try:
            stdin.close()
        except OSError:
            pass

    def close(self) -> str:
        """Finish the file. Returns the report text; raises CaptureError if there is no usable video."""
        if self._held is not None:  # the last picture lasts one slot
            try:
                self._queue.put((self._held[0], 1), timeout=10)
            except queue.Full:
                self._proc.kill()
        try:
            self._queue.put(None, timeout=10)
        except queue.Full:  # ffmpeg hangs: end it so the writer thread can finish
            self._proc.kill()
            self._queue.put(None)
        self._writer.join(timeout=120)
        try:
            code = self._proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            code = self._proc.wait()
        detail = self._proc.stderr.read().decode("utf-8", "replace").strip()[-300:] if self._proc.stderr else ""
        if self._error is not None or code != 0 or not self.path.is_file() or self.path.stat().st_size == 0:
            self._delete_file()
            raise CaptureError(f"ffmpeg ended with code {code}: {detail}")
        if len(self.times) < 2:
            self._delete_file()
            raise CaptureError("the take has fewer than two pictures")
        rows, stats = analyze_times(self.times, self.nominal_fps)
        stats["frames"] = self.written  # what the file holds (held pictures included)
        write_intervals_csv(self.path.with_suffix(".intervals.csv"), rows)
        return report_text(stats, self.nominal_fps, self.size, self.queue_dropped)

    def abort(self) -> None:
        """Throw the take away (the camera failed): end ffmpeg and delete its file."""
        try:
            self._proc.kill()
        except OSError:
            pass
        self._queue.put(None)
        self._writer.join(timeout=10)
        try:
            self._proc.wait(timeout=10)  # Windows will not delete a file that ffmpeg still has open
        except subprocess.TimeoutExpired:
            pass
        self._delete_file()

    def _delete_file(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            log.warning("the unfinished take %s could not be deleted", self.path)


# ---- the camera thread -----------------------------------------------------------------------------------------
class CameraWorker(QThread):
    """Reads the camera until asked to stop: preview copy for the screen, full pictures for the running take."""

    frame_ready = Signal()
    opened = Signal(int, int, float)  # width, height, fps the camera really delivers
    failed = Signal(str)              # plain-language reason (English source text)

    def __init__(self, index: int, opener: Callable | None = None, width: int = WIDTH, height: int = HEIGHT,
                 fps: float = FPS, parent=None) -> None:
        super().__init__(parent)
        self._index, self._opener = index, opener or open_capture
        self._width, self._height, self._fps = width, height, fps
        self.size = (0, 0)
        self.fps = fps
        self._encoder: TakeEncoder | None = None
        self._calls: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._latest: QImage | None = None
        self._signalled = False
        self._last_preview = 0.0

    # used by the controller (GUI / worker threads)
    def take(self) -> QImage | None:
        with self._lock:
            image, self._latest, self._signalled = self._latest, None, False
        return image

    def set_encoder(self, encoder: TakeEncoder | None) -> None:
        self._encoder = encoder

    def call(self, function: Callable, timeout: float = 5.0):
        """Run ``function(capture)`` in the camera thread (an OpenCV capture is not thread-safe) and return its result."""
        if not self.isRunning():
            raise RuntimeError("the camera is not open")
        done, box = threading.Event(), {}
        self._calls.put((function, done, box))
        if not done.wait(timeout):
            raise TimeoutError("the camera did not answer")
        if "error" in box:
            raise box["error"]
        return box.get("value")

    # the thread
    def run(self) -> None:
        try:
            capture = self._opener(self._index, self._width, self._height, self._fps)
        except Exception:  # noqa: BLE001 - a driver or OpenCV failure is reported in plain words
            log.exception("opening the camera failed")
            capture = None
        if capture is None:
            self.failed.emit("The camera could not be opened. Is another program using it?")
            return
        try:
            first = self._first_picture(capture)
            if first is None:
                self.failed.emit("The camera does not deliver pictures.")
                return
            height, width = first.shape[:2]
            reported = float(capture.get(5) or 0.0)  # cv2.CAP_PROP_FPS
            self.size = (int(width), int(height))
            self.fps = reported if 1.0 <= reported <= 120.0 else self._fps
            self.opened.emit(self.size[0], self.size[1], self.fps)
            self._handle(first, time.monotonic())
            misses = 0
            while not self.isInterruptionRequested():
                self._run_calls(capture)
                ok, frame = capture.read()
                if not ok or frame is None:
                    misses += 1
                    if misses > 50:  # about a second without a picture: unplugged or taken by another program
                        self.failed.emit("The camera stopped delivering pictures.")
                        break
                    time.sleep(0.02)
                    continue
                misses = 0
                self._handle(frame, time.monotonic())
        finally:
            self._answer_waiting_calls()
            capture.release()

    def _first_picture(self, capture):
        deadline = time.monotonic() + 5.0  # cameras need a moment after opening
        while time.monotonic() < deadline and not self.isInterruptionRequested():
            ok, frame = capture.read()
            if ok and frame is not None:
                return frame
            time.sleep(0.05)
        return None

    def _handle(self, frame, arrived: float) -> None:
        encoder = self._encoder
        if encoder is not None:
            encoder.add(frame, arrived)
        if arrived - self._last_preview < 1.0 / PREVIEW_FPS:
            return
        self._last_preview = arrived
        import cv2

        height, width = frame.shape[:2]
        preview_height = max(1, round(height * PREVIEW_WIDTH / width))
        small = cv2.resize(frame, (PREVIEW_WIDTH, preview_height))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, PREVIEW_WIDTH, preview_height, PREVIEW_WIDTH * 3, QImage.Format.Format_RGB888).copy()
        with self._lock:
            self._latest = image
            notify = not self._signalled
            self._signalled = True
        if notify:  # at most one queued signal: a slow GUI skips pictures instead of falling behind
            self.frame_ready.emit()

    def _run_calls(self, capture) -> None:
        while True:
            try:
                function, done, box = self._calls.get_nowait()
            except queue.Empty:
                return
            try:
                box["value"] = function(capture)
            except Exception as exc:  # noqa: BLE001 - handed back to the caller
                box["error"] = exc
            done.set()

    def _answer_waiting_calls(self) -> None:
        while True:
            try:
                _function, done, box = self._calls.get_nowait()
            except queue.Empty:
                return
            box["error"] = RuntimeError("the camera was closed")
            done.set()


# ---- the controller --------------------------------------------------------------------------------------------
class WebcamController(QObject):
    cameras_found = Signal(list)         # names, in the order OpenCV opens them by
    camera_state = Signal(bool, str)     # open?, "1920x1080 · 30 fps" or the reason it is not
    frame_ready = Signal()               # a new preview picture is waiting (take_frame())
    take_started = Signal(str, dict)     # role, {"locked": bool|None, "lock": {...}, "start": datetime, "latency": s}
    take_saved = Signal(str, list, str)  # role, final file paths, report (same layout as the phone's verify report)
    take_failed = Signal(str, str)       # role, plain-language reason

    def __init__(self, parent: QObject | None = None, ffmpeg: str | None = None, opener: Callable | None = None,
                 lister: Callable | None = None, work_dir: Path | None = None, lock_on_record: bool = True) -> None:
        super().__init__(parent)
        self._ffmpeg = ffmpeg if ffmpeg is not None else shutil.which("ffmpeg")
        self._opener, self._lister = opener, lister or (lambda: list_cameras(self._ffmpeg))
        self._work_dir = Path(work_dir) if work_dir else Path(tempfile.gettempdir()) / "CalibFreeHands" / "takes"
        self.lock_on_record = lock_on_record
        self.names: list[str] = []
        self._worker: CameraWorker | None = None
        self._open = False
        self._take: dict | None = None  # role, encoder, locked

    # ---- cameras ---------------------------------------------------------------------------------------------
    def refresh_cameras(self) -> None:
        """Look for cameras (about a second) and announce them with cameras_found."""
        def work() -> None:
            names = list(self._lister())
            self.names = names
            self.cameras_found.emit(names)

        threading.Thread(target=work, name="camera-list", daemon=True).start()

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def size(self) -> tuple[int, int]:
        return self._worker.size if self._worker else (0, 0)

    def open_camera(self, name: str) -> None:
        """Open the camera called *name* (one of ``names``); the answer is camera_state."""
        self.close_camera()
        if name not in self.names:
            self.camera_state.emit(False, "No PC camera was found. Connect a webcam and press Refresh.")
            return
        worker = CameraWorker(self.names.index(name), self._opener)
        worker.frame_ready.connect(self.frame_ready)
        worker.opened.connect(self._on_opened)
        worker.failed.connect(self._on_failed)
        self._worker = worker
        worker.start()

    def close_camera(self) -> None:
        if self._take is not None:
            self._abort_encoder()
        worker, self._worker, self._open = self._worker, None, False
        if worker is not None:
            for signal in (worker.opened, worker.failed, worker.frame_ready):
                try:
                    signal.disconnect()
                except (RuntimeError, TypeError):
                    pass
            worker.requestInterruption()
            worker.wait(5000)

    def take_frame(self) -> QImage | None:
        return self._worker.take() if self._worker is not None else None

    def _on_opened(self, width: int, height: int, fps: float) -> None:
        self._open = True
        self.camera_state.emit(True, f"{width}x{height} · {fps:g} fps")

    def _on_failed(self, reason: str) -> None:
        self._open = False
        try:
            if self._take is not None:
                self._abort_encoder()
        except Exception:  # noqa: BLE001 - the page must still hear that the camera is gone
            log.exception("the running take could not be discarded")
        self.camera_state.emit(False, reason)

    # ---- takes -----------------------------------------------------------------------------------------------
    def start_take(self, role: str, take_name: str) -> None:
        """Fix exposure and focus, then start recording (worker thread)."""
        def work() -> None:
            worker = self._worker
            if worker is None or not self._open:
                self.take_failed.emit(role, "The recording could not be started. Is the camera open?")
                return
            if not self._ffmpeg:
                self.take_failed.emit(role, "Recording needs ffmpeg (not found).")
                return
            started = time.time()
            lock = None
            if self.lock_on_record:
                try:
                    lock = worker.call(lock_controls)  # before the first recorded picture, like the phone
                except Exception:  # noqa: BLE001 - the camera may not support it at all
                    log.warning("exposure/focus lock failed", exc_info=True)
                    lock = {"focus": False, "exposure": False, "white_balance": False}
            try:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                encoder = TakeEncoder(self._ffmpeg, self._work_dir / f"{take_name}_{stamp}.mp4", *worker.size,
                                      nominal_fps=worker.fps)
            except (OSError, ValueError) as exc:
                log.exception("the encoder could not be started")
                self.take_failed.emit(role, f"The recording could not be started ({type(exc).__name__}).")
                return
            self._take = {"role": role, "encoder": encoder, "lock": lock}
            worker.set_encoder(encoder)
            self.take_started.emit(role, {
                # focus and exposure decide; white balance is an extra that not every camera has
                "locked": None if lock is None else bool(lock["focus"] and lock["exposure"]), "lock": lock,
                "start": datetime.now().astimezone(), "latency": time.time() - started, "source": "camera"})

        threading.Thread(target=work, name="camera-take-start", daemon=True).start()

    def stop_take(self, role: str, session_dir: Path, final_stem: str) -> None:
        """Stop the take, give the camera back its automatic exposure/focus, move the file into *session_dir*."""
        def work() -> None:
            try:
                files, report = self._finish(Path(session_dir), final_stem)
            except CaptureError as exc:
                log.error("stop of the %s take failed: %s", role, exc)
                self.take_failed.emit(role, "The recording could not be finished cleanly.")
            except Exception as exc:  # noqa: BLE001
                log.exception("stop of the %s take failed", role)
                self.take_failed.emit(role, f"The recording could not be saved ({type(exc).__name__}).")
            else:
                self.take_saved.emit(role, [str(f) for f in files], report)

        threading.Thread(target=work, name="camera-take-stop", daemon=True).start()

    def abort_take(self, session_dir: Path, final_stem: str) -> None:
        """Finish a running take right now (the window is closing); the file is kept under *final_stem*."""
        try:
            self._finish(Path(session_dir), final_stem)
        except Exception:  # noqa: BLE001
            log.exception("the running take could not be saved")

    def _finish(self, session_dir: Path, final_stem: str) -> tuple[list[Path], str]:
        take, self._take = self._take, None
        if take is None:
            raise CaptureError("not recording")
        worker = self._worker
        if worker is not None:
            worker.set_encoder(None)  # no more pictures for this take
        try:
            report = take["encoder"].close()
        finally:
            if take["lock"] is not None and worker is not None:
                try:
                    worker.call(unlock_controls)
                except Exception:  # noqa: BLE001
                    log.warning("automatic exposure/focus could not be restored", exc_info=True)
        from .controller import CaptureController  # the same "never overwrite an earlier take" move as the phone's

        return CaptureController._move_into(session_dir, [take["encoder"].path], final_stem), report

    def _abort_encoder(self) -> None:
        take, self._take = self._take, None
        if self._worker is not None:
            self._worker.set_encoder(None)
        if take is not None:
            take["encoder"].abort()

    def shutdown(self) -> None:
        self.close_camera()
