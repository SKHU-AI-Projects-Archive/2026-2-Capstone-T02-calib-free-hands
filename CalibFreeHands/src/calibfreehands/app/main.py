from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QLabel, QListWidget, QMainWindow, QPushButton,
    QHBoxLayout, QVBoxLayout, QWidget, QComboBox, QLineEdit, QCheckBox,
)

from calibfreehands.media import discover_media
from calibfreehands.media.ffmpeg import FFmpegDecoder
from calibfreehands.media.probe import probe_media
from calibfreehands.app.settings import AppSettings
from calibfreehands.calibration import AnyCalibBackend
from calibfreehands.export.paths import choose_output_directory
from calibfreehands.pipeline import JobThread, ProcessingJob, process_and_save
from calibfreehands.pose import OriginalWiLoRBackend
from calibfreehands.visualization.viewer import ResultViewer


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CalibFreeHands")
        self.setMinimumSize(760, 520)
        self.queue = QListWidget()
        self.setAcceptDrops(True)
        self.settings = AppSettings()
        browse_files = QPushButton("Browse files")
        browse_folder = QPushButton("Browse folder")
        browse_files.clicked.connect(self._browse_files)
        browse_folder.clicked.connect(self._browse_folder)
        self.start_button = QPushButton("Start processing"); self.start_button.clicked.connect(self._start_selected)
        open_result = QPushButton("Open saved result"); open_result.clicked.connect(self._open_result)
        self.cancel = QPushButton("Cancel"); self.cancel.setEnabled(False); self.cancel.clicked.connect(self._cancel_job)
        self.progress = QLabel("Waiting")
        buttons = QHBoxLayout()
        buttons.addWidget(browse_files)
        buttons.addWidget(browse_folder)
        buttons.addWidget(self.start_button); buttons.addWidget(open_result); buttons.addWidget(self.cancel)
        settings_row = QHBoxLayout()
        theme = QComboBox(); theme.addItems(["System", "Light", "Dark"]); theme.setCurrentText(self.settings.get("theme").title()); theme.currentTextChanged.connect(self._set_theme)
        language = QComboBox(); language.addItems(["English", "한국어"]); language.setCurrentText("한국어" if self.settings.get("language") == "ko" else "English"); language.currentTextChanged.connect(lambda value: self._set_setting("language", "ko" if value == "한국어" else "en"))
        output = QLineEdit(str(self.settings.get("output_directory"))); output.setPlaceholderText("Output directory"); output.editingFinished.connect(lambda: self._set_setting("output_directory", output.text()))
        collision = QComboBox(); collision.addItems(["Ask", "Overwrite", "Auto rename", "Skip"]); collision.setCurrentText(str(self.settings.get("collision_policy")).replace("_", " ").title()); collision.currentTextChanged.connect(lambda value: self._set_setting("collision_policy", value.lower().replace(" ", "_")))
        sampling = QComboBox(); sampling.addItems(["Uniform", "All frames"]); sampling.setCurrentText("All frames" if self.settings.get("sampling_mode") == "all" else "Uniform"); sampling.currentTextChanged.connect(lambda value: self._set_setting("sampling_mode", "all" if value == "All frames" else "uniform"))
        sample_count = QComboBox(); sample_count.addItems(["4", "8", "16", "32"]); sample_count.setCurrentText(str(self.settings.get("sample_count"))); sample_count.currentTextChanged.connect(lambda value: self._set_setting("sample_count", int(value)))
        sample_count.setEnabled(sampling.currentText() == "Uniform")
        sampling.currentTextChanged.connect(lambda value: sample_count.setEnabled(value == "Uniform"))
        aggregation = QComboBox(); aggregation.addItems(["Median", "Mean"]); aggregation.setCurrentText(str(self.settings.get("aggregation")).title()); aggregation.currentTextChanged.connect(lambda value: self._set_setting("aggregation", value.lower()))
        preserve_audio = QCheckBox("Preserve audio"); preserve_audio.setChecked(bool(self.settings.get("preserve_audio"))); preserve_audio.toggled.connect(lambda value: self._set_setting("preserve_audio", value))
        settings_row.addWidget(QLabel("Theme")); settings_row.addWidget(theme); settings_row.addWidget(QLabel("Language")); settings_row.addWidget(language); settings_row.addWidget(output, 1)
        processing_row = QHBoxLayout()
        processing_row.setSpacing(14)
        for label, control in (("Collision", collision), ("Sampling", sampling), ("Sample count", sample_count), ("Aggregate", aggregation)):
            group = QHBoxLayout(); group.setContentsMargins(0, 0, 0, 0); group.setSpacing(4)
            group.addWidget(QLabel(label)); group.addWidget(control); processing_row.addLayout(group)
        processing_row.addWidget(preserve_audio)
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Input / Queue"))
        layout.addWidget(self.queue, 1)
        layout.addLayout(buttons)
        layout.addLayout(settings_row)
        layout.addLayout(processing_row)
        layout.addWidget(QLabel("Original AnyCalib/WiLoR backends use the configured runtime assets."))
        layout.addWidget(self.progress)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self._job_thread = None

    def _set_setting(self, key: str, value) -> None:
        self.settings.set(key, value); self.settings.sync()

    def _set_theme(self, value: str) -> None:
        self._set_setting("theme", value.lower())
        app = QApplication.instance()
        app.setStyleSheet("QWidget { background: #202124; color: #f1f3f4; }" if value == "Dark" else "")

    def _add_paths(self, paths: list[Path]) -> None:
        existing = {self.queue.item(i).text() for i in range(self.queue.count())}
        for path in paths:
            if str(path) not in existing:
                self.queue.addItem(str(path))

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        paths = []
        for url in event.mimeData().urls():
            paths.extend(discover_media(Path(url.toLocalFile())))
        self._add_paths(paths)
        event.acceptProposedAction()

    def _browse_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Select media")
        self._add_paths([Path(item) for item in files])

    def _browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select media folder")
        if folder:
            self._add_paths(discover_media(Path(folder)))

    def _start_selected(self) -> None:
        if self._job_thread is not None and self._job_thread.isRunning():
            return
        item = self.queue.currentItem()
        if item is None:
            self.progress.setText("Select a media item first")
            return
        input_path = Path(item.text())
        self._job_thread = JobThread(ProcessingJob(item.text(), operation=lambda job: self._process_media(input_path, job)))
        self._job_thread.update.connect(lambda event: self.progress.setText(f"{event.stage + ': ' if event.stage else ''}{event.state.value}: {event.current}/{event.total} ({event.progress:.0%})"))
        self._job_thread.failed.connect(lambda message: self.progress.setText(f"Failed: {message}"))
        self._job_thread.finished.connect(self._on_job_finished)
        self.start_button.setEnabled(False)
        self.cancel.setEnabled(True)
        self._job_thread.start()

    def _on_job_finished(self) -> None:
        self.start_button.setEnabled(True)
        self.cancel.setEnabled(False)

    def _process_media(self, input_path: Path, job=None) -> None:
        info = probe_media(input_path)
        if info.width is None or info.height is None:
            raise RuntimeError(f"Media dimensions are unavailable: {input_path.name}")
        root = Path(__file__).resolve().parents[4]
        repository = root / "WiLoR"
        pose = OriginalWiLoRBackend(repository, repository / "pretrained_models" / "wilor_final.ckpt", root / "models" / "model_config_wilor_original.yaml")
        pose.preflight()
        if job: job.report(1, 1, stage="Preparing media")
        frames = list(FFmpegDecoder().decode_rgb(input_path, info.width, info.height, info.frame_rate))
        output_base = Path(self.settings.get("output_directory") or input_path.parent)
        output_dir = choose_output_directory(output_base, input_path.stem, self.settings.get("collision_policy"))
        calibrator = AnyCalibBackend(device="cuda")
        settings = {**self.settings.values(), "input_path": str(input_path), "checkpoint_paths": {"wilor": str(pose.checkpoint)},
                    "preview_output": str(output_dir / "preview.mp4"), "fps": info.frame_rate or 30.0}
        def progress(stage, current, total):
            if job: job.report(current, total, stage=stage)
        process_and_save(frames, calibrator, pose, output_dir, settings, progress, lambda: bool(job and job.cancelled))

    def _cancel_job(self) -> None:
        if self._job_thread is not None:
            self._job_thread.cancel()

    def _open_result(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open saved result", "", "NumPy result (*.npz)")
        if path:
            viewer = ResultViewer(Path(path))
            viewer.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            viewer.show()
            self._result_viewer = viewer


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
