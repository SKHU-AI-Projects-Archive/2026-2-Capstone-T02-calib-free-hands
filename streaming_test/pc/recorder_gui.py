"""Small window to start/stop recording while the phone streams (uses recorder_control).

Usage: python recorder_gui.py [--config run_config.yaml] [--manual-config run_config_manual.yaml]
Normally started by run_pc.py when record_mode is manual.

Look and structure match the Android app (android/app/src/main/res/values/colors.xml, layout/activity_main.xml):
same palette (theme.py), header, state chips (dot + caption + value), big timer, big start/stop button,
message card. Keep the two in sync. The Phone QR button opens qr_window.py (QR code for the app's Scan QR).
Left: the phone's live picture (live_view.py). Start recording locks the phone's exposure/focus first and Stop
recording releases it (phone_link.py).
"""

import argparse
import logging
import os
import queue
import re
import subprocess
import threading
import time
import tkinter as tk
from pathlib import Path

import manual_mode
import phone_link
import qr_window
from live_view import LiveView
from recorder_control import Recorder, RecorderError
from theme import (BG, MONO, OK, ON_ACCENT, ON_REC, OUTLINE, PRESSED, REC, SANS, SURFACE, SURFACE_ALT, TEXT,
                   TEXT_DIM, WARN, enable_dpi_awareness, label, IDLE)

BASE = Path(__file__).resolve().parent
TICK_MS = 200  # UI refresh; the API is polled separately every poll_interval_s

# Idle hints in the message card (they explain why Start is disabled); any other message replaces them.
HINT_NO_API = "The receiver is not running. Start it with run_recorder.bat."
HINT_WAITING = "Waiting for the phone to connect ... Start recording unlocks once it is connected."
HINT_READY = "Phone connected (not recording). Enter a take name and click Start recording when you are ready."
HINTS = (HINT_NO_API, HINT_WAITING, HINT_READY)


def parse_summary(text):
    """(frames, duration_s, mean_fps, gaps, dropped) from the verify report(s) in text, or None."""
    frames = [int(x) for x in re.findall(r"^frames\s*:\s*(\d+)", text, re.M)]
    if not frames:
        return None
    duration = sum(float(x) for x in re.findall(r"^duration\s*:\s*([\d.]+)", text, re.M))
    fps = [float(x) for x in re.findall(r"^mean fps\s*:\s*([\d.]+)", text, re.M)]
    gaps = sum(int(x) for x in re.findall(r"\):\s*(\d+), estimated dropped frames", text))
    dropped = sum(int(x) for x in re.findall(r"estimated dropped frames:\s*(\d+)", text))
    return sum(frames), duration, (fps[0] if fps else float("nan")), gaps, dropped


def next_take_name(name):
    """take1 -> take2, take07 -> take08, other -> other2."""
    name = name.strip()
    m = re.match(r"^(.*?)(\d+)$", name)
    return f"{m.group(1)}{int(m.group(2)) + 1:0{len(m.group(2))}d}" if m else f"{name}2"


class BigButton(tk.Label):
    """Flat primary button: green (start) or red (stop) when enabled, grey and flat when disabled."""

    def __init__(self, parent, text, color, fg, command):
        super().__init__(parent, text=text, font=(SANS, 13, "bold"), pady=12, highlightthickness=1)
        self.color, self.fg, self.command, self.enabled = color, fg, command, False
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<ButtonRelease-1>", self._release)
        self.set_enabled(False)

    def set_enabled(self, on):
        self.enabled = on
        if on:
            self.configure(bg=self.color, fg=self.fg, highlightbackground=self.color, cursor="hand2")
        else:
            self.configure(bg=SURFACE_ALT, fg=TEXT_DIM, highlightbackground=OUTLINE, cursor="arrow")

    def _press(self, _event):
        if self.enabled:
            self.configure(bg=PRESSED[self.color])

    def _release(self, event):
        if not self.enabled:
            return
        self.configure(bg=self.color)
        if 0 <= event.x < self.winfo_width() and 0 <= event.y < self.winfo_height():
            self.command()


class RecorderApp:
    def __init__(self, root, settings):
        self.root = root
        self.settings = settings
        self.qr = None  # the Phone QR window, while open
        self.qr_auto_done = False  # the QR window opens by itself once at start (if no phone streams yet)
        self.rec = Recorder(settings)
        self.poll_s = settings["poll_interval_s"]
        self.status = {"api": False, "stream": False, "record": False}
        self.polled = False  # true after the first API poll (before that the status is unknown)
        self.state = "idle"  # idle | starting | recording | stopping
        self.started_at = None  # time.time() of the actual recording start
        self.events = queue.Queue()
        self.stopping = threading.Event()

        root.title("Calib-free-hands - Recorder")
        root.configure(bg=BG)
        root.geometry("1150x640")
        root.minsize(1000, 580)
        body = tk.Frame(root, bg=BG)
        body.pack(fill="both", expand=True)
        # Like the phone app: camera picture on the left, control panel on the right.
        left = tk.Frame(body, bg=BG, padx=22, pady=18)
        left.pack(side="left", fill="y")
        label(left, "LIVE VIEW", 9, True, TEXT_DIM, anchor="w").pack(fill="x", pady=(0, 6))
        self.live = LiveView(left, settings)
        self.live.pack(anchor="nw")
        tk.Frame(body, bg=OUTLINE, width=1).pack(side="left", fill="y")
        outer = tk.Frame(body, bg=BG, padx=22, pady=18)
        outer.pack(side="left", fill="both", expand=True)

        # Header: name, REC badge, what this window is (no paths or addresses on screen).
        head = tk.Frame(outer, bg=BG)
        head.pack(fill="x")
        label(head, "Calib-free-hands", 20, True).pack(side="left")
        self.badge = label(head, "", 10, True, padx=10, pady=2)
        self.badge.pack(side="right")
        label(outer, "Recorder", 9, fg=TEXT_DIM, mono=True, anchor="w").pack(fill="x", pady=(2, 0))

        # State chips.
        chips = tk.Frame(outer, bg=SURFACE_ALT, highlightbackground=OUTLINE, highlightthickness=1, padx=14, pady=6)
        chips.pack(fill="x", pady=(16, 0))
        self.stream_dot, self.stream_val = self._chip_row(chips, "STREAM")
        self.rec_dot, self.rec_val = self._chip_row(chips, "RECORDING")

        # Big timer.
        self.elapsed = label(outer, "00:00:00", 46, True, TEXT_DIM, mono=True)
        self.elapsed.pack(pady=(18, 10))

        # Take name + buttons.
        label(outer, "TAKE NAME", 9, True, TEXT_DIM, anchor="w").pack(fill="x")
        self.take_var = tk.StringVar(value="take1")
        self.take_entry = tk.Entry(
            outer, textvariable=self.take_var, font=(MONO, 12), bg=SURFACE_ALT, fg=TEXT, insertbackground=TEXT,
            disabledbackground=SURFACE, disabledforeground=TEXT_DIM, relief="flat",
            highlightthickness=1, highlightbackground=OUTLINE, highlightcolor=OK)
        self.take_entry.pack(fill="x", pady=(4, 12), ipady=7)
        buttons = tk.Frame(outer, bg=BG)
        buttons.pack(fill="x")
        buttons.columnconfigure((0, 1), weight=1, uniform="buttons")
        self.start_btn = BigButton(buttons, "Start recording", OK, ON_ACCENT, self.on_start)
        self.start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.stop_btn = BigButton(buttons, "Stop recording", REC, ON_REC, self.on_stop)
        self.stop_btn.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        # Message card.
        self.info_var = tk.StringVar(value=HINT_WAITING)
        info = tk.Label(outer, textvariable=self.info_var, font=(MONO, 9), fg=TEXT_DIM, bg=SURFACE_ALT,
                        anchor="w", justify="left", padx=12, pady=10,
                        highlightbackground=OUTLINE, highlightthickness=1)
        info.pack(fill="x", pady=(12, 0))
        info.bind("<Configure>", lambda e: info.configure(wraplength=max(100, e.width - 28)))

        # Last take: one-line verdict only (the full verify report, with file paths, goes to the log).
        label(outer, "LAST TAKE", 9, True, TEXT_DIM, anchor="w").pack(fill="x", pady=(16, 0))
        self.stats_label = label(outer, "No take yet", 10, True, TEXT_DIM, mono=True, anchor="w")
        self.stats_label.pack(fill="x", pady=(4, 6))
        tk.Frame(outer, bg=BG).pack(fill="both", expand=True)  # spacer: the footer stays at the bottom

        # Footer.
        foot = tk.Frame(outer, bg=BG)
        foot.pack(fill="x", pady=(10, 0))
        tk.Button(foot, text="Open folder", command=self.open_folder, font=(SANS, 9), fg=TEXT, bg=SURFACE_ALT,
                  activebackground=OUTLINE, activeforeground=TEXT, relief="flat", bd=0, padx=12, pady=5,
                  highlightthickness=1, highlightbackground=OUTLINE, cursor="hand2").pack(side="left")
        tk.Button(foot, text="Phone QR", command=self.open_qr, font=(SANS, 9), fg=TEXT, bg=SURFACE_ALT,
                  activebackground=OUTLINE, activeforeground=TEXT, relief="flat", bd=0, padx=12, pady=5,
                  highlightthickness=1, highlightbackground=OUTLINE, cursor="hand2").pack(side="left", padx=(8, 0))
        self.top_var = tk.BooleanVar(value=False)
        tk.Checkbutton(foot, text="Always on top", variable=self.top_var, font=(SANS, 9), fg=TEXT_DIM, bg=BG,
                       activebackground=BG, activeforeground=TEXT, selectcolor=SURFACE_ALT, bd=0,
                       highlightthickness=0,
                       command=lambda: root.attributes("-topmost", self.top_var.get())).pack(side="right")

        threading.Thread(target=self.poll_loop, daemon=True).start()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.tick()

    @staticmethod
    def _chip_row(parent, caption):
        row = tk.Frame(parent, bg=SURFACE_ALT)
        row.pack(fill="x", pady=4)
        dot = label(row, "●", 12, fg=IDLE, bg=SURFACE_ALT)
        dot.pack(side="left")
        label(row, caption, 9, True, TEXT_DIM, SURFACE_ALT).pack(side="left", padx=(8, 0))
        value = label(row, "", 11, True, IDLE, SURFACE_ALT, mono=True)
        value.pack(side="right")
        return dot, value

    @staticmethod
    def _set_chip(dot, value, color, text):
        dot.configure(fg=color)
        value.configure(text=text, fg=color)

    # ---- background ----
    def poll_loop(self):
        while not self.stopping.is_set():
            self.status = self.rec.status()
            self.polled = True
            self.stopping.wait(self.poll_s)

    def run_bg(self, func, done):
        def work():
            try:
                self.events.put((done, func(), None))
            except RecorderError as e:
                self.events.put((done, None, e))
        threading.Thread(target=work, daemon=True).start()

    # ---- actions ----
    def on_start(self):
        self.state = "starting"
        self.info_var.set("Starting the recording ...")
        name = self.take_var.get()  # read here: tkinter variables belong to the UI thread
        self.run_bg(lambda: self.start_take(name), self.after_start)

    def start_take(self, name):
        """Worker thread: lock the phone's exposure/focus first (so the first recorded frames are already
        locked), then start recording. result['locked']: True/False = phone answered, None = not done."""
        locked = phone_link.set_hold(self.settings, True)
        try:
            result = self.rec.start(name)
        except RecorderError:
            if locked:
                phone_link.set_hold(self.settings, False)
            raise
        result["locked"] = locked
        return result

    def after_start(self, result, error):
        if error:
            self.state = "idle"
            logging.getLogger("recorder").error("start failed: %s", error)  # details only in the console
            self.info_var.set("Could not start the recording. Check that the phone is connected.")
            return
        self.state = "recording"
        if result["locked"]:
            note = " Exposure and focus are locked on the phone."
        elif self.settings["lock_on_record"] and result["locked"] is None:
            note = " The phone could not lock exposure/focus (update the app, or check the network)."
        else:
            note = ""
        if result["start_time"]:
            self.started_at = result["start_time"].timestamp()
            self.info_var.set(f"Recording started at {result['start_time'].strftime('%H:%M:%S')} "
                              f"({result['latency_s']:.1f} s after the click).{note}")
        else:
            self.started_at = time.time()
            self.info_var.set("Recording is on, but nothing has been written yet. Check the phone's stream." + note)

    def on_stop(self):
        self.state = "stopping"
        self.info_var.set("Stopping and checking the recording ...")
        # Back to automatic exposure/focus on the phone (only what the PC locked is released).
        threading.Thread(target=phone_link.set_hold, args=(self.settings, False), daemon=True).start()
        self.run_bg(self.rec.stop, self.after_stop)

    def after_stop(self, result, error):
        self.state = "idle"
        self.started_at = None
        if error:
            logging.getLogger("recorder").error("stop failed: %s", error)
            self.info_var.set("The recording could not be finished cleanly. Look in the recordings folder.")
            return
        self.info_var.set("Saved as " + ", ".join(f.name for f in result["files"]))
        logging.getLogger("recorder").info("verify report:\n%s", result["summary"])
        self.show_stats(result["summary"])
        self.take_var.set(next_take_name(self.take_var.get()))  # ready for the next take

    def show_stats(self, summary):
        stats = parse_summary(summary)
        if stats is None:
            self.stats_label.configure(text="This recording could not be checked", fg=REC)
            return
        frames, duration, fps, gaps, dropped = stats
        self.stats_label.configure(
            text=f"{frames} frames  ·  {duration:.1f} s  ·  {fps:.2f} fps  ·  {gaps} gaps (~{dropped} dropped)",
            fg=OK if gaps == 0 else WARN)

    def open_folder(self):
        folder = self.rec.rec_dir
        try:
            folder.mkdir(parents=True, exist_ok=True)
            if os.name == "nt":
                os.startfile(folder)
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except OSError as e:
            logging.getLogger("recorder").error("cannot open %s: %s", folder, e)
            self.info_var.set("Could not open the folder.")

    def open_qr(self):
        if self.qr is not None and self.qr.winfo_exists():
            self.qr.lift()
            return
        self.qr = qr_window.open_window(self.root, self.settings)

    def on_close(self):
        if self.state == "recording":  # do not leave MediaMTX recording with nobody to stop it
            self.info_var.set("Closing: stopping the recording ...")
            self.root.update()
            try:
                self.rec.stop()
            except RecorderError as e:
                logging.getLogger("recorder").error("stop on close failed: %s", e)
            phone_link.set_hold(self.settings, False)
        self.live.stop()
        self.stopping.set()
        self.root.destroy()

    # ---- UI refresh ----
    def tick(self):
        while True:
            try:
                done, result, error = self.events.get_nowait()
            except queue.Empty:
                break
            done(result, error)

        st = self.status
        self.live.update_stream(st["api"] and st["stream"])
        if self.polled and not self.qr_auto_done:
            self.qr_auto_done = True
            if not st["stream"]:
                self.open_qr()  # closes itself when the phone connects
        recording = self.state == "recording"
        if self.polled and self.state == "idle" and self.info_var.get() in HINTS:
            self.info_var.set(HINT_NO_API if not st["api"] else HINT_READY if st["stream"] else HINT_WAITING)
        blink_on = int(time.time() * 2) % 2 == 0

        if not st["api"]:
            self._set_chip(self.stream_dot, self.stream_val, REC, "NO RECEIVER")
        elif st["stream"]:
            self._set_chip(self.stream_dot, self.stream_val, OK, "CONNECTED")
        elif recording:
            self._set_chip(self.stream_dot, self.stream_val, WARN, "LOST")
        else:
            self._set_chip(self.stream_dot, self.stream_val, IDLE, "NOT CONNECTED")

        if recording:
            self._set_chip(self.rec_dot, self.rec_val, REC, "REC" if st["stream"] else "REC (NO STREAM)")
            self.rec_dot.configure(text="●" if blink_on else "○")
        else:
            self.rec_dot.configure(text="●")
            if self.state in ("starting", "stopping"):
                self._set_chip(self.rec_dot, self.rec_val, WARN, self.state.upper())
            else:
                self._set_chip(self.rec_dot, self.rec_val, IDLE, "IDLE")
        if recording:
            self.badge.configure(text="● REC" if blink_on else "○ REC", bg=REC, fg=ON_REC)
        else:
            self.badge.configure(text="", bg=BG)

        if self.started_at and recording:
            s = int(time.time() - self.started_at)
            self.elapsed.configure(text=f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}", fg=REC)
        elif self.state == "idle":
            self.elapsed.configure(text="00:00:00", fg=TEXT_DIM)

        can_start = self.state == "idle" and st["api"] and st["stream"]
        self.start_btn.set_enabled(can_start)
        self.stop_btn.set_enabled(recording)
        self.take_entry.configure(state="normal" if self.state == "idle" else "disabled")
        self.root.after(TICK_MS, self.tick)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(BASE / "run_config.yaml"))
    parser.add_argument("--manual-config", default=str(BASE / "run_config_manual.yaml"))
    parser.add_argument("--verbose", action="store_true", help="print take details and the verify report to the console")
    args = parser.parse_args()
    # Quiet by default: no file paths or verify reports in the console, only warnings and errors.
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s [recorder] %(message)s")
    settings = manual_mode.load_settings(args.config, args.manual_config)
    enable_dpi_awareness()
    root = tk.Tk()
    RecorderApp(root, settings)
    root.mainloop()


if __name__ == "__main__":
    main()
