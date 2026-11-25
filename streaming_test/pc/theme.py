"""Design tokens shared by the PC windows (recorder_gui.py, qr_window.py).

Same values as the Android app (android/app/src/main/res/values/colors.xml); keep the two in sync.
"""

import tkinter as tk

BG, SURFACE, SURFACE_ALT, OUTLINE = "#0F1216", "#181C22", "#20262E", "#2C333D"
TEXT, TEXT_DIM = "#E8ECF1", "#8D98A6"
OK, REC, WARN, IDLE = "#34D399", "#F43F5E", "#F5A524", "#6B7684"  # start/ok, recording/stop/error, waiting, idle
ON_ACCENT, ON_REC = "#0B1411", "#FFFFFF"
PRESSED = {OK: "#2BB583", REC: "#D63A56"}
SANS, MONO = "Segoe UI", "Consolas"


def label(parent, text="", size=10, bold=False, fg=TEXT, bg=BG, mono=False, **kw):
    font = (MONO if mono else SANS, size, "bold" if bold else "normal")
    return tk.Label(parent, text=text, font=font, fg=fg, bg=bg, **kw)


def enable_dpi_awareness():
    """Crisp text on scaled Windows displays; call before creating the Tk root."""
    import ctypes
    import sys
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
