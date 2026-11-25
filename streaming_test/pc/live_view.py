"""Live camera picture for the recorder window.

ffmpeg reads the phone's stream back from the local receiver (RTSP, localhost) and writes small raw frames to a
pipe; this canvas shows the newest one (up to 30 fps). It only watches: the recording and the phone are not
touched, and closing the window or losing the stream stops ffmpeg. Needs ffmpeg on the PATH (the verify step
needs ffprobe from the same package anyway).

Low delay: the ffmpeg chain avoids everything that holds frames back (single-thread decoding, no fps filter,
flush after every frame) and the canvas draws every 33 ms. Measured with a screen-flash test against the
phone's camera: about 0.31 s from light on the lens to the decoded frame (0.40 s with the earlier chain, which
used frame threads, an fps filter and buffered pipe output). The rest is the phone (camera, encoder) and the
SRT receive buffer (120 ms), which the recording shares, so it is not touched here.
"""

import os
import shutil
import subprocess
import threading
import time
import tkinter as tk

from theme import OUTLINE, SANS, TEXT_DIM

W, H = 560, 315  # picture size on screen (16:9); other aspect ratios get black bars
FPS = 30         # how often the canvas looks for a new frame (the stream's own rate is the limit)
RETRY_S = 3.0    # wait before ffmpeg is started again after it ended while the stream is up


class LiveView(tk.Canvas):
    def __init__(self, parent, settings):
        super().__init__(parent, width=W, height=H, bg="#000000", highlightthickness=1,
                         highlightbackground=OUTLINE)
        self.url = f"rtsp://127.0.0.1:{settings['rtsp_port']}/{settings['stream_path']}"
        self.ffmpeg = shutil.which("ffmpeg")
        self.enabled = bool(settings.get("live_view", True)) and bool(self.ffmpeg)
        self.reason = ("Live view is off." if not settings.get("live_view", True)
                       else "Live view needs ffmpeg (not found).")
        self.proc = None
        self.frame = None          # newest frame (bytes), set by the reader thread
        self.frame_lock = threading.Lock()
        self.photo = None
        self.has_picture = False
        self.retry_at = 0.0
        self.image_id = self.create_image(0, 0, anchor="nw")
        self.text_id = self.create_text(W // 2, H // 2, fill=TEXT_DIM, font=(SANS, 11), width=W - 60,
                                        justify="center", text="")
        self._show_text("Waiting for the phone ...")
        self.after(1000 // FPS, self._render)

    # ---- called from the window's refresh (UI thread) ----
    def update_stream(self, online):
        """online: a phone is streaming (True), not (False), the receiver is unreachable (None/False)."""
        if not self.enabled:
            self._show_text(self.reason)
            return
        if not online:
            if self.proc is not None:
                self._stop()
                self._show_text("Waiting for the phone ...")
            return
        if self.proc is not None and self.proc.poll() is None:
            return  # running
        if self.proc is not None:  # ffmpeg ended by itself: try again after a pause
            self.proc = None
            self.has_picture = False
            self.retry_at = time.monotonic() + RETRY_S
            self._show_text("Reconnecting the live view ...")
            return
        if time.monotonic() >= self.retry_at:
            self._start()

    def stop(self):
        self._stop()

    # ---- ffmpeg ----
    def _start(self):
        cmd = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
               "-fflags", "nobuffer", "-flags", "low_delay",
               "-threads", "1",  # frame threads would hold back several decoded frames
               "-rtsp_transport", "tcp",
               "-analyzeduration", "500000", "-probesize", "500000", "-i", self.url, "-an",
               # Fit into W x H with black bars. setsar=1 first: the phone's frames carry an undefined aspect
               # ratio (sar 0/1), which made scale round up and the pad step fail with "Padded dimensions
               # cannot be smaller than input dimensions".
               "-vf", f"setsar=1,scale={W}:{H}:force_original_aspect_ratio=decrease,"
                      f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black",
               # No fps filter (it waits for the next frame) and flush every frame: without it the last
               # part of a frame stays in ffmpeg's output buffer until the next frame arrives.
               "-fps_mode", "passthrough", "-flush_packets", "1",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                         stdin=subprocess.DEVNULL, creationflags=flags)
        except OSError:
            self.proc = None
            self.retry_at = time.monotonic() + RETRY_S
            self._show_text("The live view could not be started.")
            return
        self.has_picture = False
        with self.frame_lock:
            self.frame = None
        self._show_text("Starting the live view ...")
        threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()

    def _stop(self):
        proc, self.proc = self.proc, None
        if proc is not None and proc.poll() is None:
            proc.kill()
        self.has_picture = False
        with self.frame_lock:
            self.frame = None
        self.itemconfigure(self.image_id, image="")
        self.photo = None

    def _read(self, proc):
        size = W * H * 3
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            if proc is not self.proc:
                break
            with self.frame_lock:
                self.frame = buf

    # ---- drawing ----
    def _show_text(self, text):
        self.itemconfigure(self.text_id, text=text, state="normal" if text else "hidden")

    def _render(self):
        with self.frame_lock:
            buf, self.frame = self.frame, None
        if buf is not None:
            self.photo = tk.PhotoImage(data=b"P6\n%d %d\n255\n" % (W, H) + buf)
            self.itemconfigure(self.image_id, image=self.photo)
            if not self.has_picture:
                self.has_picture = True
                self._show_text("")
        self.after(1000 // FPS, self._render)
