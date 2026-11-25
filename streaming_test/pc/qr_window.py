"""Window with a QR code of the phone publish URL, for the Android app's "Scan QR" button.

Usage: python qr_window.py [--config run_config.yaml] [--manual-config run_config_manual.yaml]
Also opened by the Recorder window (Phone QR button). Needs `pip install segno` (run_pc.bat installs it).

The code holds   srt://<pc ip>:<srt_port>?streamid=publish:<stream_path>   - the same URL run_pc.py prints.
The window closes by itself when the phone connects (a stream appears in the receiver; needs the Control API of
manual mode). A phone that is already connected when the window opens does not close it.
The first network is the Wi-Fi one (see netinfo.py); if the PC has several, pick the one the phone is on. No
addresses are shown, only the network names.
"""

import argparse
import threading
import time
import tkinter as tk
from pathlib import Path

import manual_mode
from netinfo import choices
from theme import BG, OK, ON_ACCENT, OUTLINE, REC, SANS, SURFACE_ALT, TEXT, TEXT_DIM, enable_dpi_awareness, label

BASE = Path(__file__).resolve().parent
SIZE = 400  # QR canvas size in pixels


def publish_url(settings, ip):
    return f"srt://{ip}:{settings['srt_port']}?streamid=publish:{settings['stream_path']}"


class QrWindow:
    def __init__(self, win, settings):
        self.win, self.s = win, settings
        win.title("Calib-free-hands - Phone QR")
        win.configure(bg=BG)
        win.resizable(False, False)
        outer = tk.Frame(win, bg=BG, padx=22, pady=18)
        outer.pack()

        label(outer, "Connect the phone", 18, True, anchor="w").pack(fill="x")
        label(outer, "In the Calib-free-hands app tap Scan QR and point the camera at this code, then tap Start "
                     "streaming. This window closes when the phone connects.", 9, fg=TEXT_DIM, anchor="w",
              justify="left", wraplength=SIZE).pack(fill="x", pady=(2, 0))
        nets = choices(settings)
        self.ip_of = dict(nets)  # network name -> address; the address itself is never shown
        self.buttons = {}
        if len(nets) > 1:  # only ask when there is something to choose
            label(outer, "NETWORK - pick the one the phone is on", 9, True, TEXT_DIM, anchor="w").pack(
                fill="x", pady=(14, 4))
            row = tk.Frame(outer, bg=BG)
            row.pack(anchor="w")
            for name, _ip in nets:
                b = tk.Label(row, text=name, font=(SANS, 10, "bold"), padx=12, pady=5, cursor="hand2",
                             highlightthickness=1)
                b.pack(side="left", padx=(0, 6))
                b.bind("<Button-1>", lambda _e, name=name: self.select(name))
                self.buttons[name] = b
        self.canvas = tk.Canvas(outer, width=SIZE, height=SIZE, bg=SURFACE_ALT, highlightthickness=0)
        self.canvas.pack(pady=14)
        self.note = label(outer, "", 9, fg=TEXT_DIM, anchor="w", justify="left", wraplength=SIZE)
        self.note.pack(fill="x")
        if nets:
            self.select(nets[0][0])
        else:
            self.show_message("No network connection found on this PC.")

        # Close when the phone connects: poll the receiver in the background, react in the UI thread.
        self.alive, self.online, self.was_online = True, None, None
        win.bind("<Destroy>", self._on_destroy)
        threading.Thread(target=self._poll, daemon=True).start()
        win.after(300, self._check)

    def _on_destroy(self, event):
        if event.widget is self.win:
            self.alive = False

    def _poll(self):
        while self.alive:
            self.online = manual_mode.stream_online(self.s)
            time.sleep(self.s["poll_interval_s"])

    def _check(self):
        if not self.alive:
            return
        now = self.online
        if now is not None:
            if self.was_online is False and now:  # no stream before, a stream now: the phone just connected
                self.win.destroy()
                return
            self.was_online = now
        self.win.after(300, self._check)

    def select(self, name):
        for other, b in self.buttons.items():
            on = other == name
            b.configure(bg=OK if on else SURFACE_ALT, fg=ON_ACCENT if on else TEXT,
                        highlightbackground=OK if on else OUTLINE)
        self.draw(publish_url(self.s, self.ip_of[name]))

    def show_message(self, text, color=TEXT_DIM):
        self.canvas.delete("all")
        self.canvas.configure(bg=SURFACE_ALT)  # no code to show: not a white box
        self.note.configure(text=text, fg=color)

    def draw(self, url):
        try:
            import segno  # imported here so the Recorder window works without it
        except ImportError:
            self.show_message("The QR code needs one more component.\n"
                              "Start run_recorder.bat once; it installs it automatically.", REC)
            return
        rows = segno.make(url, error="m").matrix
        n, border = len(rows), 4  # 4 modules of quiet zone, as the QR standard asks
        cell = SIZE // (n + 2 * border)
        offset = (SIZE - cell * n) // 2
        self.canvas.delete("all")
        self.canvas.configure(bg="#FFFFFF")  # dark modules on white, the quiet zone must be light
        for y, row in enumerate(rows):
            for x, dark in enumerate(row):
                if dark:
                    self.canvas.create_rectangle(offset + x * cell, offset + y * cell,
                                                 offset + (x + 1) * cell, offset + (y + 1) * cell,
                                                 fill="#000000", outline="")
        self.note.configure(text="", fg=TEXT_DIM)


def open_window(parent, settings):
    """Open the QR window over `parent` (a Tk root); returns the Toplevel."""
    win = tk.Toplevel(parent)
    QrWindow(win, settings)
    return win


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(BASE / "run_config.yaml"))
    parser.add_argument("--manual-config", default=str(BASE / "run_config_manual.yaml"))
    args = parser.parse_args()
    settings = manual_mode.load_settings(args.config, args.manual_config)
    enable_dpi_awareness()
    root = tk.Tk()
    QrWindow(root, settings)
    root.mainloop()


if __name__ == "__main__":
    main()
