from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QLabel, QListWidget, QMainWindow, QPushButton,
    QHBoxLayout, QVBoxLayout, QWidget, QComboBox, QLineEdit, QCheckBox, QDoubleSpinBox,
)

from calibfreehands.media import discover_media
from calibfreehands.media.ffmpeg import FFmpegDecoder
from calibfreehands.media.probe import probe_media
from calibfreehands.app.configuration import ApplicationConfiguration, ModelDefinition, PresetDefinition, load_configuration
from calibfreehands.app.settings import AppSettings
from calibfreehands.export.paths import choose_output_directory
from calibfreehands.pipeline import JobThread, ProcessingJob
from calibfreehands.pipeline.comparison import PipelineConfiguration, process_comparison_session
from calibfreehands.visualization.comparison import ComparisonViewer
from calibfreehands.visualization.viewer import ResultViewer
import uuid


class ConfigurationRow(QWidget):
    def __init__(self, models: ApplicationConfiguration, name: str, calibration_id: str,
                 reconstruction_id: str, reference_focal_px: float = 1000.0, config_id: str | None = None):
        super().__init__()
        self.id = config_id or uuid.uuid4().hex
        self.name = QLineEdit(name)
        self.calibration = MainWindow._model_combo(models.calibration_models)
        self.reconstruction = MainWindow._model_combo(models.reconstruction_models)
        MainWindow._select_data(self.calibration, calibration_id)
        MainWindow._select_data(self.reconstruction, reconstruction_id)
        self.reference_focal = QDoubleSpinBox()
        self.reference_focal.setRange(1.0, 100000.0)
        self.reference_focal.setDecimals(1)
        self.reference_focal.setSuffix(" px")
        self.reference_focal.setValue(reference_focal_px)
        self.remove = QPushButton("Remove")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.name, 1)
        layout.addWidget(self.calibration, 2)
        layout.addWidget(self.reconstruction, 2)
        layout.addWidget(self.reference_focal)
        layout.addWidget(self.remove)
        self.calibration.currentIndexChanged.connect(self._update_reference)
        self._update_reference()

    def _update_reference(self, *_args):
        self.reference_focal.setEnabled(self.calibration.currentData() == "checkerboard_reference")

    def configuration(self) -> PipelineConfiguration:
        return PipelineConfiguration(
            self.id, self.name.text().strip(), self.calibration.currentData(),
            self.reconstruction.currentData(),
            self.reference_focal.value() if self.calibration.currentData() == "checkerboard_reference" else None,
        )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CalibFreeHands")
        self.setMinimumSize(760, 520)
        self.queue = QListWidget()
        self.setAcceptDrops(True)
        self.settings = AppSettings()
        self.configuration = load_configuration()
        self._updating_controls = False
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
        self.preset_combo = QComboBox()
        for preset in self.configuration.presets.values():
            self.preset_combo.addItem(preset.display_name, preset.id)
        self.configuration_rows = []
        self.rows_widget = QWidget()
        self.rows_layout = QVBoxLayout(self.rows_widget)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.set_default_button = QPushButton("Set as default")
        self.set_default_button.clicked.connect(self._save_defaults)
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Preset"))
        model_row.addWidget(self.preset_combo, 1)
        self.add_configuration_button = QPushButton("Add configuration")
        self.add_configuration_button.clicked.connect(self._add_configuration_row)
        model_row.addWidget(self.add_configuration_button)
        model_row.addWidget(self.set_default_button)
        settings_row = QHBoxLayout()
        theme = QComboBox(); theme.addItems(["System", "Light", "Dark"]); theme.setCurrentText(self.settings.get("theme").title()); theme.currentTextChanged.connect(self._set_theme)
        language = QComboBox(); language.addItems(["English", "한국어"]); language.setCurrentText("한국어" if self.settings.get("language") == "ko" else "English"); language.currentTextChanged.connect(lambda value: self._set_setting("language", "ko" if value == "한국어" else "en"))
        output = QLineEdit(str(self.settings.get("output_directory"))); output.setPlaceholderText("Output directory"); output.editingFinished.connect(lambda: self._set_setting("output_directory", output.text()))
        collision = QComboBox(); collision.addItems(["Ask", "Overwrite", "Auto rename", "Skip"]); collision.setCurrentText(str(self.settings.get("collision_policy")).replace("_", " ").title()); collision.currentTextChanged.connect(lambda value: self._set_setting("collision_policy", value.lower().replace(" ", "_")))
        sampling = QComboBox(); sampling.addItem("Uniform", "uniform"); sampling.addItem("All frames", "all"); sampling.setCurrentIndex(sampling.findData(self.settings.get("sampling_mode"))); sampling.currentTextChanged.connect(lambda value: self._set_setting("sampling_mode", "all" if value == "All frames" else "uniform"))
        sample_count = QComboBox(); sample_count.addItems(["4", "8", "16", "32"]); sample_count.setCurrentText(str(self.settings.get("sample_count"))); sample_count.currentTextChanged.connect(lambda value: self._set_setting("sample_count", int(value)))
        sample_count.setEnabled(sampling.currentText() == "Uniform")
        sampling.currentTextChanged.connect(lambda value: sample_count.setEnabled(value == "Uniform"))
        aggregation = QComboBox(); aggregation.addItem("Median", "median"); aggregation.addItem("Mean", "mean"); aggregation.setCurrentIndex(aggregation.findData(self.settings.get("aggregation"))); aggregation.currentTextChanged.connect(lambda value: self._set_setting("aggregation", value.lower()))
        self.sampling_combo = sampling
        self.sample_count_combo = sample_count
        self.aggregation_combo = aggregation
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
        layout.addLayout(model_row)
        layout.addWidget(self.rows_widget)
        layout.addLayout(settings_row)
        layout.addLayout(processing_row)
        layout.addWidget(QLabel("Original AnyCalib/WiLoR backends use the configured runtime assets."))
        layout.addWidget(self.progress)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self._job_thread = None
        self._restore_model_defaults()
        self.preset_combo.currentIndexChanged.connect(self._apply_selected_preset)
        for control in (sampling, sample_count, aggregation):
            control.currentIndexChanged.connect(self._model_or_processing_changed)

    @staticmethod
    def _model_combo(models: dict[str, ModelDefinition]) -> QComboBox:
        combo = QComboBox()
        for model in models.values():
            combo.addItem(model.display_name, model.id)
            if not model.available:
                item = combo.model().item(combo.count() - 1)
                item.setEnabled(False)
                item.setToolTip(model.unavailable_reason or "Not available")
        return combo

    @property
    def calibration_combo(self):
        return self.configuration_rows[0].calibration

    @property
    def reconstruction_combo(self):
        return self.configuration_rows[0].reconstruction

    def _add_row(self, name, calibration_id, reconstruction_id, reference_focal_px=1000.0):
        row = ConfigurationRow(self.configuration, name, calibration_id, reconstruction_id, reference_focal_px)
        row.remove.clicked.connect(lambda: self._remove_configuration_row(row))
        row.calibration.currentIndexChanged.connect(self._model_or_processing_changed)
        row.reconstruction.currentIndexChanged.connect(self._model_or_processing_changed)
        self.configuration_rows.append(row)
        self.rows_layout.addWidget(row)
        self._refresh_row_buttons()
        return row

    def _refresh_row_buttons(self):
        for row in self.configuration_rows:
            row.remove.setEnabled(len(self.configuration_rows) > 1)

    def _add_configuration_row(self):
        current = self.configuration_rows[-1].configuration()
        self._add_row(f"Configuration {len(self.configuration_rows) + 1}", current.calibration_model_id,
                      current.reconstruction_model_id, current.reference_focal_px or 1000.0)
        self._model_or_processing_changed()

    def _remove_configuration_row(self, row):
        if len(self.configuration_rows) <= 1:
            return
        self.configuration_rows.remove(row)
        self.rows_layout.removeWidget(row)
        row.deleteLater()
        self._refresh_row_buttons()
        self._model_or_processing_changed()

    def _replace_rows(self, name, calibration_id, reconstruction_id):
        for row in self.configuration_rows:
            self.rows_layout.removeWidget(row)
            row.deleteLater()
        self.configuration_rows.clear()
        self._add_row(name, calibration_id, reconstruction_id)

    def _restore_model_defaults(self) -> None:
        preset_id = self.settings.get("default_preset_id")
        calibration_id = self.settings.get("default_calibration_model_id")
        reconstruction_id = self.settings.get("default_reconstruction_model_id")
        valid_ids = (
            calibration_id in self.configuration.calibration_models
            and reconstruction_id in self.configuration.reconstruction_models
            and self.configuration.calibration_models[calibration_id].available
            and self.configuration.reconstruction_models[reconstruction_id].available
        )
        valid_custom_settings = (
            self.settings.get("sampling_mode") in {"uniform", "all"}
            and self.settings.get("sample_count") >= 1
            and self.settings.get("aggregation") in {"median", "mean"}
        )
        if preset_id == "custom" and valid_ids and valid_custom_settings:
            self.preset_combo.addItem("Custom", "custom")
            self.preset_combo.setCurrentIndex(self.preset_combo.count() - 1)
            self._replace_rows("Configuration 1", calibration_id, reconstruction_id)
            self._set_processing_controls(
                self.settings.get("sampling_mode"), self.settings.get("sample_count"),
                self.settings.get("aggregation"),
            )
        elif preset_id in self.configuration.presets and valid_ids:
            preset = self.configuration.presets[preset_id]
            self._apply_preset_values(preset)
        else:
            self._apply_preset_values(self.configuration.presets["original"])

    @staticmethod
    def _select_data(combo: QComboBox, value) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _set_processing_controls(self, sampling: str, sample_count: int, aggregation: str) -> None:
        self._select_data(self.sampling_combo, sampling)
        if self.sample_count_combo.findText(str(sample_count)) < 0:
            self.sample_count_combo.addItem(str(sample_count))
        self.sample_count_combo.setCurrentText(str(sample_count))
        self._select_data(self.aggregation_combo, aggregation)
        self.sample_count_combo.setEnabled(sampling == "uniform")

    def _apply_preset_values(self, preset: PresetDefinition) -> None:
        self._updating_controls = True
        try:
            self._select_data(self.preset_combo, preset.id)
            self._replace_rows(preset.display_name, preset.calibration_model_id, preset.reconstruction_model_id)
            self._set_processing_controls(preset.sampling, preset.sample_count, preset.aggregation)
        finally:
            self._updating_controls = False

    def _apply_selected_preset(self, _index: int) -> None:
        if self._updating_controls:
            return
        preset = self.configuration.presets.get(self.preset_combo.currentData())
        if preset is not None:
            self._apply_preset_values(preset)

    def _model_or_processing_changed(self, _index: int = 0) -> None:
        if self._updating_controls:
            return
        self.sample_count_combo.setEnabled(self.sampling_combo.currentData() == "uniform")
        first = self.configuration_rows[0].configuration()
        preset = next((preset for preset in self.configuration.presets.values() if (
            len(self.configuration_rows) == 1
            and preset.calibration_model_id == first.calibration_model_id
            and preset.reconstruction_model_id == first.reconstruction_model_id
            and preset.sampling == self.sampling_combo.currentData()
            and preset.sample_count == int(self.sample_count_combo.currentText())
            and preset.aggregation == self.aggregation_combo.currentData()
        )), None)
        if preset is not None:
            self._select_data(self.preset_combo, preset.id)
        else:
            if self.preset_combo.findData("custom") < 0:
                self.preset_combo.addItem("Custom", "custom")
            self._select_data(self.preset_combo, "custom")

    def _save_defaults(self) -> None:
        first = self.configuration_rows[0].configuration()
        for key, value in (
            ("default_preset_id", self.preset_combo.currentData()),
            ("default_calibration_model_id", first.calibration_model_id),
            ("default_reconstruction_model_id", first.reconstruction_model_id),
            ("sampling_mode", self.sampling_combo.currentData()),
            ("sample_count", int(self.sample_count_combo.currentText())),
            ("aggregation", self.aggregation_combo.currentData()),
        ):
            self.settings.set(key, value)
        self.settings.sync()

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
        selection = {
            "configurations": [row.configuration() for row in self.configuration_rows],
            "preset_id": self.preset_combo.currentData(),
        }
        self._job_thread = JobThread(ProcessingJob(
            item.text(), operation=lambda job: self._process_media(input_path, job, selection)
        ))
        self._job_thread.update.connect(lambda event: self.progress.setText(f"{event.stage + ': ' if event.stage else ''}{event.state.value}: {event.current}/{event.total} ({event.progress:.0%})"))
        self._job_thread.failed.connect(lambda message: self.progress.setText(f"Failed: {message}"))
        self._job_thread.finished.connect(self._on_job_finished)
        self.start_button.setEnabled(False)
        self.cancel.setEnabled(True)
        self._job_thread.start()

    def _on_job_finished(self) -> None:
        self.start_button.setEnabled(True)
        self.cancel.setEnabled(False)

    def _process_media(self, input_path: Path, job=None, selection: dict[str, str] | None = None) -> None:
        info = probe_media(input_path)
        if info.width is None or info.height is None:
            raise RuntimeError(f"Media dimensions are unavailable: {input_path.name}")
        root = Path(__file__).resolve().parents[4]
        selection = selection or {
            "configurations": [row.configuration() for row in self.configuration_rows],
            "preset_id": self.preset_combo.currentData(),
        }
        if job: job.report(1, 1, stage="Preparing media")
        frames = list(FFmpegDecoder().decode_rgb(input_path, info.width, info.height, info.frame_rate))
        output_base = Path(self.settings.get("output_directory") or input_path.parent)
        output_dir = choose_output_directory(output_base, input_path.stem, self.settings.get("collision_policy"))
        settings = {key: value for key, value in self.settings.values().items() if not key.startswith("default_")}
        settings.update({"input_path": str(input_path), "preset_id": selection["preset_id"],
                    "fps": info.frame_rate or 30.0, "create_preview": True}
        )
        def progress(stage, current, total):
            if job: job.report(current, total, stage=stage)
        process_comparison_session(frames, selection["configurations"], self.configuration, root,
                                   output_dir, settings, progress, lambda: bool(job and job.cancelled))

    def _cancel_job(self) -> None:
        if self._job_thread is not None:
            self._job_thread.cancel()

    def _open_result(self) -> None:
        path, selected_filter = QFileDialog.getOpenFileName(
            self, "Open saved result", "", "NumPy result (*.npz);;Comparison session (session.json)"
        )
        if path:
            viewer = ComparisonViewer(Path(path)) if selected_filter.startswith("Comparison") else ResultViewer(Path(path))
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
