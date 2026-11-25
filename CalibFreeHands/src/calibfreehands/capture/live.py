"""Live camera picture: ffmpeg reads the phone's stream back from the local receiver (RTSP on localhost) and writes
small raw frames to a pipe; the newest one is shown. It only watches, the recording and the phone are not touched.

The ffmpeg options are the low-delay set measured for the former tkinter window (about 0.31 s from the lens to a
decoded frame): single-thread decoding, no fps filter, flush after every frame, ``setsar=1`` first because the phone's
frames carry an undefined aspect ratio (sar 0/1) which made ``pad`` fail.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy

from calibfreehands.app.i18n import tr

W, H = 640, 360
RETRY_S = 3.0


def ffmpeg_command(ffmpeg: str, url: str, width: int = W, height: int = H) -> list[str]:
    return [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-fflags", "nobuffer", "-flags", "low_delay", "-threads", "1",
            "-rtsp_transport", "tcp", "-analyzeduration", "500000", "-probesize", "500000", "-i", url, "-an",
            "-vf", f"setsar=1,scale={width}:{height}:force_original_aspect_ratio=decrease,"
                   f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black",
            "-fps_mode", "passthrough", "-flush_packets", "1",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]


class _FrameReader(QThread):
    frame_ready = Signal()
    ended = Signal()

    def __init__(self, command: list[str], width: int, height: int) -> None:
        super().__init__()
        self._command, self._width, self._height = command, width, height
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._latest: QImage | None = None
        self._signalled = False

    def take(self) -> QImage | None:
        with self._lock:
            image, self._latest, self._signalled = self._latest, None, False
        return image

    def run(self) -> None:
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            self._proc = subprocess.Popen(self._command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                          stdin=subprocess.DEVNULL, creationflags=flags)
        except OSError:
            self.ended.emit()
            return
        size = self._width * self._height * 3
        try:
            while not self.isInterruptionRequested():
                buffer = self._proc.stdout.read(size)
                if len(buffer) < size:
                    break
                image = QImage(buffer, self._width, self._height, self._width * 3,
                               QImage.Format.Format_RGB888).copy()
                with self._lock:
                    self._latest = image
                    notify = not self._signalled
                    self._signalled = True
                if notify:  # at most one queued signal: a slow GUI skips frames instead of falling behind
                    self.frame_ready.emit()
        finally:
            self.stop_process()
            self.ended.emit()

    def stop_process(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.kill()


class LiveView(QLabel):
    def __init__(self, rtsp_port: int, stream_path: str, enabled: bool = True, parent=None) -> None:
        super().__init__(parent)
        self.setProperty("role", "liveview")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(480, 270)
        policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)  # always 16:9, the picture is not stretched into a tall black panel
        self.setSizePolicy(policy)
        self._url = f"rtsp://127.0.0.1:{rtsp_port}/{stream_path}"
        self._ffmpeg = shutil.which("ffmpeg")
        self._enabled = enabled and bool(self._ffmpeg)
        self._reason = "Live view is off." if not enabled else "Live view needs ffmpeg (not found)."
        self._reader: _FrameReader | None = None
        self._retry_at = 0.0
        self._has_picture = False
        self._pixmap: QPixmap | None = None
        self._text_source = ""
        self._external = False  # True: the pictures come from show_image() (the PC camera), not from the phone
        self._show_text("Waiting for the phone ...")

    # ---- the PC camera: the page hands over the pictures itself ----------------------------------------------
    def use_external(self, on: bool, text: str = "") -> None:
        """Switch between the phone's stream (False) and pictures given by show_image() (True).

        The phone's reader is stopped either way; going back to the phone, its stream is picked up again by the next
        ``set_stream`` call.
        """
        self._stop()
        self._external = on
        self._retry_at = 0.0
        self._show_text(text or ("Opening the camera ..." if on else "Waiting for the phone ..."))

    def show_image(self, image: QImage) -> None:
        self._pixmap = QPixmap.fromImage(image)
        self._has_picture = True
        self._paint()

    def show_message(self, text: str) -> None:
        """A text instead of a picture (the PC camera is opening, missing or failed)."""
        self._pixmap = None
        self._has_picture = False
        self._show_text(text)

    # ---- driven by the controller's status --------------------------------------------------------------------
    def set_stream(self, online: bool) -> None:
        if self._external:
            return
        if not self._enabled:
            self._show_text(self._reason)
            return
        running = self._reader is not None and self._reader.isRunning()
        if not online:
            if self._reader is not None:
                self._stop()
                self._show_text("Waiting for the phone ...")
            return
        if running:
            return
        if self._reader is not None:  # ffmpeg ended by itself: try again after a pause
            self._reader = None
            self._has_picture = False
            self._retry_at = time.monotonic() + RETRY_S
            self._show_text("Reconnecting the live view ...")
            return
        if time.monotonic() >= self._retry_at:
            self._start()

    def stop(self) -> None:
        self._stop()

    # ---- internals --------------------------------------------------------------------------------------------
    def _start(self) -> None:
        self._has_picture = False
        self._reader = _FrameReader(ffmpeg_command(self._ffmpeg, self._url), W, H)
        self._reader.frame_ready.connect(self._on_frame)
        self._reader.start()
        self._show_text("Starting the live view ...")

    def _stop(self) -> None:
        reader, self._reader = self._reader, None
        if reader is not None:
            reader.requestInterruption()
            reader.stop_process()
            reader.wait(2000)
        self._has_picture = False
        self._pixmap = None
        self.clear()

    def _on_frame(self) -> None:
        reader = self._reader
        image = reader.take() if reader is not None else None
        if image is None:
            return
        self._pixmap = QPixmap.fromImage(image)
        self._has_picture = True
        self._paint()

    def _paint(self) -> None:
        if self._pixmap is not None:
            self.setPixmap(self._pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                               Qt.TransformationMode.FastTransformation))

    def _show_text(self, text: str) -> None:
        self._text_source = text
        self.setPixmap(QPixmap())
        self.setText(tr(text))

    def refresh_text(self) -> None:
        """The language changed: show the placeholder text again (a running picture is left alone)."""
        if not self._has_picture:
            self._show_text(self._text_source)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt name
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt name
        return width * H // W

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt name
        super().resizeEvent(event)
        if self._has_picture:
            self._paint()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._stop()
        super().closeEvent(event)
