"""Window with the QR code the phone app scans. It closes itself when the phone connects."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from calibfreehands.app.i18n import localize, tr

QUIET_ZONE = 4  # modules of white around the code (the QR standard)


def qr_pixmap(text: str, size: int = 320) -> QPixmap:
    """Black-on-white QR code of *text* (white also in the dark theme: scanners need the contrast)."""
    import segno  # PyYAML-style optional extra: only the capture page needs it

    matrix = segno.make(text, error="m", micro=False).matrix
    modules = len(matrix) + 2 * QUIET_ZONE
    cell = max(1, size // modules)
    image = QImage(modules * cell, modules * cell, QImage.Format.Format_RGB32)
    image.fill(QColor("white"))
    painter = QPainter(image)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("black"))
    for y, row in enumerate(matrix):
        for x, dark in enumerate(row):
            if dark:
                painter.drawRect((x + QUIET_ZONE) * cell, (y + QUIET_ZONE) * cell, cell, cell)
    painter.end()
    return QPixmap.fromImage(image)


class QrDialog(QDialog):
    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Calib-free-hands - Phone QR")
        self._controller = controller
        self._was_connected = False
        self._choices = controller.network_choices()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(10)
        title = QLabel("Phone")
        title.setProperty("role", "title")
        layout.addWidget(title)
        hint = QLabel("Open the app on the phone: it asks for this code when it starts "
                      "(or tap Scan QR). The stream starts right after the scan.")
        hint.setProperty("role", "dim")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._buttons: list[QPushButton] = []
        if len(self._choices) > 1:  # only when the PC is in more than one network
            row = QHBoxLayout()
            for label, ip in self._choices:
                button = QPushButton(label)
                button.setCheckable(True)
                button.clicked.connect(lambda _checked, address=ip, b=button: self._select(address, b))
                row.addWidget(button)
                self._buttons.append(button)
            row.addStretch(1)
            layout.addLayout(row)
        self._code = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self._code.setMinimumSize(340, 340)
        layout.addWidget(self._code, 1)
        self._note = QLabel("")
        self._note.setProperty("role", "dim")
        self._note.setWordWrap(True)
        layout.addWidget(self._note)

        if self._choices:
            self._select(self._choices[0][1], self._buttons[0] if self._buttons else None)
        else:
            self._note.setText(tr("No network address of this PC was found. Connect the PC to the phone's Wi-Fi."))
        controller.status.connect(self._on_status)
        localize(self)

    def _select(self, ip: str, active: QPushButton | None) -> None:
        for button in self._buttons:
            button.setChecked(button is active)
        try:
            self._code.setPixmap(qr_pixmap(self._controller.phone_url(ip)))
            self._note.setText(tr("The first network is the PC's Wi-Fi, the one the phone can usually reach.")
                               if self._buttons else "")
        except ImportError:
            self._code.setText(tr("The QR code needs the 'segno' package (pip install segno)."))

    def _on_status(self, status: dict) -> None:
        connected = bool(status.get("stream"))
        if connected and not self._was_connected:  # False -> True: the phone just connected
            self.close()
        self._was_connected = connected

    def showEvent(self, event) -> None:  # noqa: N802 - Qt name
        super().showEvent(event)
        self._was_connected = True  # a phone that is already connected does not close the window we just opened
