"""The application window: capture page + processing page (the former CalibFreeHands window) in one program.

  Capture  - phone camera (QR connection) or the PC's camera, calibration video and work video
             (see calibfreehands.capture.page)
  Process  - queue, models, results (``MainWindow``, unchanged; it is embedded here as a page)

A finished work video is handed from the first page to the second with the focal length HandCalib found
(or the one typed in), and processing starts by itself; the result viewer opens as before.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMainWindow, QTabWidget

from calibfreehands.app.i18n import localize
from calibfreehands.app.main import MainWindow
from calibfreehands.app.settings import AppSettings
from calibfreehands.app.theme import current_palette, register_window, stylesheet, unregister_window
from calibfreehands.capture.page import CapturePage


class AppShell(QMainWindow):
    def __init__(self, settings: AppSettings | None = None, controller_factory=None, camera_factory=None) -> None:
        settings = settings or AppSettings()
        super().__init__()
        self.setWindowTitle("Calib-free-hands")
        self.processing = MainWindow(settings)
        unregister_window(self.processing)  # it is a page now: this window translates it
        register_window(self)
        self.capture = CapturePage(settings, lambda: Path(self.processing.output_path.text()),
                                   controller_factory=controller_factory, camera_factory=camera_factory)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.capture, "Capture")
        self.tabs.addTab(self.processing, "Process")
        self.setCentralWidget(self.tabs)
        self.resize(1260, 780)
        self.apply_design()
        localize(self)  # Korean if the saved language is Korean
        self.capture.process_requested.connect(self.process_work_video)
        QTimer.singleShot(0, self.capture.start)  # start the receiver once the window is up

    def apply_design(self) -> None:
        """Style sheet of the shared design for this window and its pages (not for the result viewers)."""
        self.setStyleSheet(stylesheet(current_palette()))
        capture = getattr(self, "capture", None)
        if capture is not None:
            capture.refresh_colors()

    def retranslate(self) -> None:
        """The language changed (Language box on the Process page): translate both pages."""
        localize(self)
        self.processing.retranslate()
        self.capture.refresh_colors()

    def process_work_video(self, video: str, focal_px: float) -> None:
        """Queue the work video with the focal length and start processing."""
        page = self.processing
        if page.is_processing:
            self.capture._say("Processing is still running. The work video is saved: add it on the Process page "
                              "when the current run is finished.")
            return
        page._clear_queue()  # the run is for this video only
        page.add_media([video])
        name = "Manual focal" if self.capture.manual else "HandCalib focal"
        page.use_manual_focal(focal_px, name)
        self.tabs.setCurrentWidget(page)
        page.start_processing()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt name
        self.capture.shutdown()
        self.processing.close()
        super().closeEvent(event)
