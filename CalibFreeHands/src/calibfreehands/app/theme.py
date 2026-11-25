"""Design tokens and the application style sheet.

Same palette as the phone app (android/app/src/main/res/values/colors.xml) and the former PC recorder window
(streaming_test/pc/theme.py): dark surfaces, green = start / ok, red = recording / stop / error, amber = waiting.
A light variant keeps the Theme switch (System / Light / Dark) of the application working.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    bg: str
    surface: str
    surface_alt: str
    outline: str
    text: str
    text_dim: str
    ok: str
    rec: str
    warn: str
    idle: str
    on_accent: str
    on_rec: str
    ok_pressed: str
    rec_pressed: str


DARK = Palette(
    bg="#0F1216", surface="#181C22", surface_alt="#20262E", outline="#2C333D",
    text="#E8ECF1", text_dim="#8D98A6", ok="#34D399", rec="#F43F5E", warn="#F5A524", idle="#6B7684",
    on_accent="#0B1411", on_rec="#FFFFFF", ok_pressed="#2BB583", rec_pressed="#D63A56",
)

LIGHT = Palette(
    bg="#F4F6F8", surface="#FFFFFF", surface_alt="#EEF1F4", outline="#D3D9E0",
    text="#161B22", text_dim="#5B6572", ok="#12A06F", rec="#D6264B", warn="#C97F00", idle="#8A94A1",
    on_accent="#FFFFFF", on_rec="#FFFFFF", ok_pressed="#0E8A5F", rec_pressed="#B81E3F",
)

# Windows that react to a theme or language change (``apply_design`` / ``retranslate``). They register
# themselves; asking Qt for all top-level widgets instead would also touch half-deleted ones.
_windows: "weakref.WeakSet" = weakref.WeakSet()


def register_window(window) -> None:
    _windows.add(window)


def unregister_window(window) -> None:
    _windows.discard(window)


def call_on_windows(method: str) -> None:
    """Call ``window.<method>()`` on every registered window that is still alive and has that method."""
    from shiboken6 import isValid

    for window in list(_windows):
        if not isValid(window):
            continue
        callback = getattr(window, method, None)
        if callable(callback):
            callback()


FONT_SANS = "Segoe UI"
FONT_MONO = "Consolas"


def palette_for(dark: bool) -> Palette:
    return DARK if dark else LIGHT


def current_palette() -> Palette:
    """Palette of the theme that is applied right now (set by ``apply_application_theme``; dark by default)."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    value = app.property("cfh_dark") if app is not None else None
    return palette_for(True if value is None else bool(value))


def stylesheet(p: Palette) -> str:
    """One style sheet for every Qt widget of the application (buttons carry a ``variant`` property)."""
    return f"""
QWidget {{ background: {p.bg}; color: {p.text}; font-family: "{FONT_SANS}"; font-size: 10pt; }}
QMainWindow, QDialog {{ background: {p.bg}; }}
QToolTip {{ background: {p.surface_alt}; color: {p.text}; border: 1px solid {p.outline}; padding: 4px; }}

QLabel {{ background: transparent; }}
QLabel[role="title"] {{ font-size: 20pt; font-weight: 700; }}
QLabel[role="caption"] {{ color: {p.text_dim}; font-size: 8.5pt; font-weight: 700; }}
QLabel[role="dim"] {{ color: {p.text_dim}; }}
QLabel[role="mono"] {{ font-family: "{FONT_MONO}"; }}
QLabel[role="timer"] {{ font-family: "{FONT_MONO}"; font-size: 36pt; font-weight: 700; color: {p.text_dim}; }}
QLabel[role="timer"][active="true"] {{ color: {p.rec}; }}
QLabel[role="liveview"] {{ background: #000000; color: {p.text_dim}; border: 1px solid {p.outline}; }}

QFrame[role="card"] {{ background: {p.surface_alt}; border: 1px solid {p.outline}; border-radius: 8px; }}
QFrame[role="card"] QLabel {{ background: transparent; }}
QFrame[role="card"][state="error"] {{ border: 1px solid {p.rec}; }}
QFrame[role="divider"] {{ background: {p.outline}; border: none; }}

QPushButton {{ background: {p.surface_alt}; color: {p.text}; border: 1px solid {p.outline};
              border-radius: 8px; padding: 6px 14px; }}
QPushButton:hover {{ border-color: {p.text_dim}; }}
QPushButton:pressed {{ background: {p.surface}; }}
QPushButton:disabled {{ background: {p.surface}; color: {p.idle}; border-color: {p.outline}; }}
QPushButton[variant="start"] {{ background: {p.ok}; color: {p.on_accent}; border: none; font-weight: 700;
                                padding: 12px 18px; font-size: 11pt; }}
QPushButton[variant="start"]:hover {{ background: {p.ok}; }}
QPushButton[variant="start"]:pressed {{ background: {p.ok_pressed}; }}
QPushButton[variant="stop"] {{ background: {p.rec}; color: {p.on_rec}; border: none; font-weight: 700;
                               padding: 12px 18px; font-size: 11pt; }}
QPushButton[variant="stop"]:hover {{ background: {p.rec}; }}
QPushButton[variant="stop"]:pressed {{ background: {p.rec_pressed}; }}
QPushButton[variant="start"]:disabled, QPushButton[variant="stop"]:disabled {{
    background: {p.surface_alt}; color: {p.idle}; border: 1px solid {p.outline}; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background: {p.surface_alt}; color: {p.text}; border: 1px solid {p.outline}; border-radius: 6px;
    padding: 4px 8px; selection-background-color: {p.outline}; selection-color: {p.text}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {p.ok}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    background: {p.surface}; color: {p.idle}; }}
QLineEdit[readOnly="true"] {{ background: {p.surface}; color: {p.text_dim}; }}
QComboBox QAbstractItemView {{ background: {p.surface}; color: {p.text}; border: 1px solid {p.outline};
                               selection-background-color: {p.outline}; }}

QListWidget, QTreeView, QTableView {{ background: {p.surface}; border: 1px solid {p.outline}; border-radius: 8px;
                                      alternate-background-color: {p.surface_alt}; }}
QListWidget::item {{ padding: 6px 8px; }}
QListWidget::item:selected, QTreeView::item:selected {{ background: {p.outline}; color: {p.text}; }}

QRadioButton, QCheckBox {{ spacing: 8px; background: transparent; }}
QRadioButton::indicator, QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {p.idle};
                                                 background: {p.surface_alt}; }}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator {{ border-radius: 4px; }}
QRadioButton::indicator:checked, QCheckBox::indicator:checked {{ background: {p.ok}; border-color: {p.ok}; }}

QTabWidget::pane {{ border: none; border-top: 1px solid {p.outline}; }}
QTabBar {{ background: {p.bg}; }}
QTabBar::tab {{ background: transparent; color: {p.text_dim}; padding: 10px 22px; border: none;
                border-bottom: 2px solid transparent; font-weight: 600; }}
QTabBar::tab:selected {{ color: {p.text}; border-bottom: 2px solid {p.ok}; }}
QTabBar::tab:hover:!selected {{ color: {p.text}; }}

QMenu {{ background: {p.surface}; border: 1px solid {p.outline}; padding: 4px; }}
QMenu::item {{ padding: 6px 18px; border-radius: 4px; }}
QMenu::item:selected {{ background: {p.outline}; }}
QToolButton {{ background: {p.surface_alt}; border: 1px solid {p.outline}; border-radius: 8px; padding: 6px 12px; }}
QToolButton:hover {{ border-color: {p.text_dim}; }}
QProgressBar {{ background: {p.surface_alt}; border: 1px solid {p.outline}; border-radius: 6px; text-align: center; }}
QProgressBar::chunk {{ background: {p.ok}; border-radius: 5px; }}
QScrollBar:vertical {{ background: {p.bg}; width: 12px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {p.outline}; border-radius: 6px; min-height: 24px; }}
QScrollBar:horizontal {{ background: {p.bg}; height: 12px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {p.outline}; border-radius: 6px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QGroupBox {{ border: 1px solid {p.outline}; border-radius: 8px; margin-top: 12px; padding-top: 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; color: {p.text_dim}; }}
QHeaderView::section {{ background: {p.surface_alt}; color: {p.text_dim}; border: none; padding: 4px 8px; }}
"""
