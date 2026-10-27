from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt, QStandardPaths
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QLabel, QListWidget, QListWidgetItem, QMainWindow, QPushButton,
    QHBoxLayout, QVBoxLayout, QWidget, QComboBox, QLineEdit, QCheckBox, QDoubleSpinBox,
    QMenu, QAbstractItemView,
)

from calibfreehands.media import discover_media
from calibfreehands.media.discovery import SUPPORTED_EXTENSIONS
from calibfreehands.media.ffmpeg import FFmpegDecoder
from calibfreehands.media.probe import probe_media
from calibfreehands.app.configuration import (
    ApplicationConfiguration,
    ModelDefinition,
    PresetDefinition,
    load_configuration,
)
from calibfreehands.app.settings import AppSettings
from calibfreehands.export.paths import choose_output_directory
from calibfreehands.pipeline import JobState, JobThread, ProcessingJob
from calibfreehands.pipeline.comparison import (
    ComparisonSession,
    PipelineConfiguration,
    _validate_configurations,
    process_comparison_session,
)
from calibfreehands.visualization.comparison import ComparisonViewer
from calibfreehands.visualization.viewer import ResultViewer
import uuid


@dataclass(frozen=True)
class ProcessingSnapshot:
    queue_paths: tuple[Path, ...]
    configurations: tuple[PipelineConfiguration, ...]
    preset_id: str | None
    sampling_mode: str
    sample_count: int
    aggregation: str
    output_directory: Path
    collision_policy: str
    preserve_audio: bool


@dataclass(frozen=True)
class BatchResult:
    completed: tuple[tuple[Path, Path, ComparisonSession], ...]
    failures: tuple[str, ...]


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
        self.reference_focal_container = QWidget()
        focal_layout = QHBoxLayout(self.reference_focal_container)
        focal_layout.setContentsMargins(0, 0, 0, 0)
        focal_layout.addWidget(QLabel("Reference Focal"))
        focal_layout.addWidget(self.reference_focal)
        self.remove = QPushButton("Remove")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.name, 1)
        layout.addWidget(self.calibration, 2)
        layout.addWidget(self.reconstruction, 2)
        layout.addWidget(self.reference_focal_container)
        layout.addWidget(self.remove)
        self.calibration.currentIndexChanged.connect(self._update_reference)
        self._update_reference()

    def _update_reference(self, *_args):
        visible = self.calibration.currentData() == "checkerboard_reference"
        self.reference_focal_container.setVisible(visible)
        self.reference_focal.setEnabled(visible)

    def configuration(self) -> PipelineConfiguration:
        return PipelineConfiguration(
            self.id, self.name.text().strip(), self.calibration.currentData(),
            self.reconstruction.currentData(),
            self.reference_focal.value() if self.calibration.currentData() == "checkerboard_reference" else None,
        )


class MainWindow(QMainWindow):
    PATH_ROLE = Qt.ItemDataRole.UserRole
    STATE_ROLE = Qt.ItemDataRole.UserRole + 1

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CalibFreeHands")
        self.setMinimumSize(760, 520)
        self.queue = QListWidget()
        self.queue.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.queue.itemSelectionChanged.connect(self._refresh_queue_buttons)
        self.setAcceptDrops(True)
        self.settings = AppSettings()
        self.configuration = load_configuration()
        self._updating_controls = False
        self.open_button = QPushButton("Open...")
        open_menu = QMenu(self.open_button)
        self.open_files_action = open_menu.addAction("Files...")
        self.open_folder_action = open_menu.addAction("Folder...")
        self.open_files_action.triggered.connect(self._browse_files)
        self.open_folder_action.triggered.connect(self._browse_folder)
        self.open_button.setMenu(open_menu)
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self._remove_selected)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self._clear_queue)
        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self._start_selected)
        open_result = QPushButton("Open saved result")
        open_result.clicked.connect(self._open_result)
        self.cancel = QPushButton("Cancel")
        self.cancel.setEnabled(False)
        self.cancel.clicked.connect(self._cancel_job)
        self.progress = QLabel("Waiting")
        buttons = QHBoxLayout()
        buttons.addWidget(self.open_button)
        buttons.addWidget(self.remove_button)
        buttons.addWidget(self.clear_button)
        buttons.addWidget(self.start_button)
        buttons.addWidget(open_result)
        buttons.addWidget(self.cancel)
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
        self.output_path = QLineEdit()
        self.output_path.setReadOnly(True)
        saved_output = str(self.settings.get("output_directory")).strip()
        if not saved_output:
            documents = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
            saved_output = str((Path(documents) if documents else Path.home() / "Documents") / "CalibFreeHands")
            self.settings.set("output_directory", saved_output)
            self.settings.sync()
        self.output_path.setText(str(Path(saved_output).expanduser().resolve()))
        self.output_browse_button = QPushButton("Browse...")
        self.output_browse_button.clicked.connect(self._browse_output)
        collision = QComboBox(); collision.addItems(["Ask", "Overwrite", "Auto rename", "Skip"]); collision.setCurrentText(str(self.settings.get("collision_policy")).replace("_", " ").title()); collision.currentTextChanged.connect(lambda value: self._set_setting("collision_policy", value.lower().replace(" ", "_")))
        sampling = QComboBox(); sampling.addItem("Uniform", "uniform"); sampling.addItem("All frames", "all"); sampling.setCurrentIndex(sampling.findData(self.settings.get("sampling_mode"))); sampling.currentTextChanged.connect(lambda value: self._set_setting("sampling_mode", "all" if value == "All frames" else "uniform"))
        sample_count = QComboBox(); sample_count.addItems(["4", "8", "16", "32"]); sample_count.setCurrentText(str(self.settings.get("sample_count"))); sample_count.currentTextChanged.connect(lambda value: self._set_setting("sample_count", int(value)))
        sample_count.setEnabled(sampling.currentText() == "Uniform")
        sampling.currentTextChanged.connect(lambda value: sample_count.setEnabled(value == "Uniform"))
        aggregation = QComboBox(); aggregation.addItem("Median", "median"); aggregation.addItem("Mean", "mean"); aggregation.setCurrentIndex(aggregation.findData(self.settings.get("aggregation"))); aggregation.currentTextChanged.connect(lambda value: self._set_setting("aggregation", value.lower()))
        self.sampling_combo = sampling
        self.sample_count_combo = sample_count
        self.aggregation_combo = aggregation
        self.collision_combo = collision
        self.preserve_audio = QCheckBox("Preserve audio")
        self.preserve_audio.setChecked(bool(self.settings.get("preserve_audio")))
        self.preserve_audio.toggled.connect(lambda value: self._set_setting("preserve_audio", value))
        settings_row.addWidget(QLabel("Theme")); settings_row.addWidget(theme); settings_row.addWidget(QLabel("Language")); settings_row.addWidget(language)
        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output"))
        output_row.addWidget(self.output_path, 1)
        output_row.addWidget(self.output_browse_button)
        processing_row = QHBoxLayout()
        processing_row.setSpacing(14)
        for label, control in (("Collision", collision), ("Sampling", sampling), ("Sample count", sample_count), ("Aggregate", aggregation)):
            group = QHBoxLayout(); group.setContentsMargins(0, 0, 0, 0); group.setSpacing(4)
            group.addWidget(QLabel(label)); group.addWidget(control); processing_row.addLayout(group)
        processing_row.addWidget(self.preserve_audio)
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Input / Queue"))
        layout.addWidget(self.queue, 1)
        layout.addLayout(buttons)
        layout.addLayout(model_row)
        layout.addWidget(self.rows_widget)
        layout.addLayout(settings_row)
        layout.addLayout(output_row)
        layout.addLayout(processing_row)
        layout.addWidget(QLabel("Original AnyCalib/WiLoR backends use the configured runtime assets."))
        layout.addWidget(self.progress)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self._job_thread = None
        self._processing = False
        self._active_snapshot = None
        self._result_viewer = None
        self._restore_model_defaults()
        self.preset_combo.currentIndexChanged.connect(self._apply_selected_preset)
        for control in (sampling, sample_count, aggregation):
            control.currentIndexChanged.connect(self._model_or_processing_changed)
        self._refresh_queue_buttons()

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

    def _add_row(self, name, calibration_id, reconstruction_id, reference_focal_px=1000.0, config_id=None):
        row = ConfigurationRow(self.configuration, name, calibration_id, reconstruction_id, reference_focal_px, config_id)
        row.remove.clicked.connect(lambda: self._remove_configuration_row(row))
        row.calibration.currentIndexChanged.connect(self._model_or_processing_changed)
        row.reconstruction.currentIndexChanged.connect(self._model_or_processing_changed)
        row.name.textEdited.connect(self._model_or_processing_changed)
        row.reference_focal.valueChanged.connect(self._model_or_processing_changed)
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
        self._replace_configurations([PipelineConfiguration(
            uuid.uuid4().hex, name, calibration_id, reconstruction_id
        )])

    def _replace_configurations(self, configurations):
        for row in self.configuration_rows:
            self.rows_layout.removeWidget(row)
            row.deleteLater()
        self.configuration_rows.clear()
        for item in configurations:
            self._add_row(item.name, item.calibration_model_id, item.reconstruction_model_id,
                          item.reference_focal_px or 1000.0, item.id)

    def _restore_model_defaults(self) -> None:
        saved = self.settings.get("default_configurations_json")
        if saved:
            try:
                snapshot = json.loads(saved)
                configurations, preset_id, sampling, count, aggregation = self._validate_default_snapshot(snapshot)
                if preset_id != "custom" and not self._matches_preset_values(
                    self.configuration.presets[preset_id], configurations, sampling, count, aggregation
                ):
                    raise ValueError("saved configurations do not match their built-in preset")
                self._updating_controls = True
                try:
                    if preset_id == "custom":
                        self._select_preset_id("custom")
                    else:
                        self._select_data(self.preset_combo, preset_id)
                    self._replace_configurations(configurations)
                    self._set_processing_controls(sampling, count, aggregation)
                finally:
                    self._updating_controls = False
                return
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                self._apply_preset_values(self.configuration.presets["original"])
                return

        preset_id = self.settings.get("default_preset_id")
        calibration_id = self.settings.get("default_calibration_model_id")
        reconstruction_id = self.settings.get("default_reconstruction_model_id")
        valid_ids = (calibration_id in self.configuration.calibration_models
                     and reconstruction_id in self.configuration.reconstruction_models
                     and self.configuration.calibration_models[calibration_id].available
                     and self.configuration.reconstruction_models[reconstruction_id].available)
        valid_settings = (self.settings.get("sampling_mode") in {"uniform", "all"}
                          and isinstance(self.settings.get("sample_count"), int)
                          and self.settings.get("sample_count") >= 1
                          and self.settings.get("aggregation") in {"median", "mean"})
        if preset_id == "custom" and valid_ids and valid_settings:
            self._select_preset_id("custom")
            self._replace_rows("Configuration 1", calibration_id, reconstruction_id)
            self._set_processing_controls(self.settings.get("sampling_mode"), self.settings.get("sample_count"),
                                          self.settings.get("aggregation"))
        elif preset_id in self.configuration.presets:
            self._apply_preset_values(self.configuration.presets[preset_id])
        else:
            self._apply_preset_values(self.configuration.presets["original"])

    def _select_preset_id(self, preset_id):
        if self.preset_combo.findData(preset_id) < 0:
            self.preset_combo.addItem("Custom", preset_id)
        self._select_data(self.preset_combo, preset_id)

    def _validate_default_snapshot(self, snapshot):
        if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1:
            raise ValueError("unsupported saved default schema")
        preset_id = snapshot.get("preset_id")
        sampling = snapshot.get("sampling_mode")
        count = snapshot.get("sample_count")
        aggregation = snapshot.get("aggregation")
        configurations = snapshot.get("configurations")
        if (preset_id != "custom" and preset_id not in self.configuration.presets
                or sampling not in {"uniform", "all"}
                or isinstance(count, bool) or not isinstance(count, int) or count < 1
                or aggregation not in {"median", "mean"}
                or not isinstance(configurations, list) or not configurations):
            raise ValueError("invalid saved default values")
        ids = set()
        restored = []
        for item in configurations:
            if not isinstance(item, dict):
                raise ValueError("saved configuration rows must be objects")
            row_id, name = item.get("id"), item.get("name")
            calibration_id, reconstruction_id = item.get("calibration_model_id"), item.get("reconstruction_model_id")
            if (not isinstance(row_id, str) or not row_id or row_id in ids
                    or not isinstance(name, str) or not name.strip()
                    or calibration_id not in self.configuration.calibration_models
                    or reconstruction_id not in self.configuration.reconstruction_models):
                raise ValueError("invalid saved configuration row")
            calibration = self.configuration.calibration_models[calibration_id]
            reconstruction = self.configuration.reconstruction_models[reconstruction_id]
            if not calibration.available or not reconstruction.available:
                raise ValueError("saved configuration selects an unavailable model")
            focal = item.get("reference_focal_px")
            if calibration.backend == "reference_focal":
                if isinstance(focal, bool) or not isinstance(focal, (int, float)) or not math.isfinite(focal) or not 0 < focal <= 100000:
                    raise ValueError("invalid saved reference focal")
                focal = float(focal)
            elif focal is not None:
                raise ValueError("non-reference configuration contains reference focal")
            ids.add(row_id)
            restored.append(PipelineConfiguration(row_id, name.strip(), calibration_id, reconstruction_id, focal))
        return restored, preset_id, sampling, count, aggregation

    def _preset_configurations(self, preset):
        if preset.configurations:
            return [PipelineConfiguration(row.id, row.name, row.calibration_model_id,
                                          row.reconstruction_model_id, row.reference_focal_px)
                    for row in preset.configurations]
        return [PipelineConfiguration(f"{preset.id}-1", preset.display_name,
                                      preset.calibration_model_id, preset.reconstruction_model_id)]

    def _matches_preset_values(self, preset, configurations, sampling, count, aggregation):
        expected = self._preset_configurations(preset)
        return ([(c.name, c.calibration_model_id, c.reconstruction_model_id, c.reference_focal_px) for c in configurations]
                == [(c.name, c.calibration_model_id, c.reconstruction_model_id, c.reference_focal_px) for c in expected]
                and (sampling, count, aggregation) == (preset.sampling, preset.sample_count, preset.aggregation))

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
            self._replace_configurations(self._preset_configurations(preset))
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
        configurations = [row.configuration() for row in self.configuration_rows]
        preset = next((preset for preset in self.configuration.presets.values() if self._matches_preset_values(
            preset, configurations, self.sampling_combo.currentData(), int(self.sample_count_combo.currentText()),
            self.aggregation_combo.currentData()
        )), None)
        if preset is not None:
            self._select_data(self.preset_combo, preset.id)
        else:
            if self.preset_combo.findData("custom") < 0:
                self.preset_combo.addItem("Custom", "custom")
            self._select_data(self.preset_combo, "custom")

    def _save_defaults(self) -> None:
        first = self.configuration_rows[0].configuration()
        snapshot = {
            "schema_version": 1,
            "preset_id": self.preset_combo.currentData(),
            "sampling_mode": self.sampling_combo.currentData(),
            "sample_count": int(self.sample_count_combo.currentText()),
            "aggregation": self.aggregation_combo.currentData(),
            "configurations": [],
        }
        for row in self.configuration_rows:
            configuration = row.configuration()
            item = {"id": configuration.id, "name": configuration.name,
                    "calibration_model_id": configuration.calibration_model_id,
                    "reconstruction_model_id": configuration.reconstruction_model_id}
            if configuration.reference_focal_px is not None:
                item["reference_focal_px"] = configuration.reference_focal_px
            snapshot["configurations"].append(item)
        for key, value in (
            ("default_preset_id", self.preset_combo.currentData()),
            ("default_calibration_model_id", first.calibration_model_id),
            ("default_reconstruction_model_id", first.reconstruction_model_id),
            ("sampling_mode", self.sampling_combo.currentData()),
            ("sample_count", int(self.sample_count_combo.currentText())),
            ("aggregation", self.aggregation_combo.currentData()),
            ("default_configurations_json", json.dumps(snapshot, separators=(",", ":"))),
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
        existing = {os.path.normcase(str(self._queue_path(self.queue.item(i))))
                    for i in range(self.queue.count())}
        for selected in paths:
            path = Path(selected)
            supported = discover_media(path)
            for media_path in supported:
                resolved = media_path.resolve()
                identity = os.path.normcase(str(resolved))
                if identity in existing:
                    continue
                item = QListWidgetItem()
                item.setData(self.PATH_ROLE, str(resolved))
                item.setData(self.STATE_ROLE, JobState.WAITING.value)
                item.setToolTip(str(resolved))
                item.setText(f"[{JobState.WAITING.value}] {resolved.name}")
                self.queue.addItem(item)
                existing.add(identity)
        self._refresh_queue_buttons()

    def _queue_path(self, item: QListWidgetItem) -> Path:
        stored = item.data(self.PATH_ROLE)
        return Path(stored).resolve() if stored else Path(item.text()).resolve()

    def _set_queue_item_state(self, index: int, state: str, detail: str = "") -> None:
        item = self.queue.item(index)
        if item is None:
            return
        item.setData(self.STATE_ROLE, state)
        item.setText(f"[{state}] {self._queue_path(item).name}")
        item.setToolTip(f"{self._queue_path(item)}\n{detail}" if detail else str(self._queue_path(item)))
        self._refresh_queue_buttons()

    def _refresh_queue_buttons(self, *_args) -> None:
        self.remove_button.setEnabled(not self._processing and bool(self.queue.selectedItems()))
        self.clear_button.setEnabled(not self._processing and self.queue.count() > 0)

    def _remove_selected(self) -> None:
        for item in self.queue.selectedItems():
            self.queue.takeItem(self.queue.row(item))
        self._refresh_queue_buttons()

    def _clear_queue(self) -> None:
        self.queue.clear()
        self._refresh_queue_buttons()

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
        extensions = " ".join(f"*{suffix}" for suffix in sorted(SUPPORTED_EXTENSIONS))
        files, _ = QFileDialog.getOpenFileNames(self, "Select media", "", f"Supported media ({extensions})")
        self._add_paths([Path(item) for item in files])

    def _browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select media folder")
        if folder:
            self._add_paths(discover_media(Path(folder)))

    def _browse_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select output folder", self.output_path.text())
        if folder:
            selected = str(Path(folder).resolve())
            self.output_path.setText(selected)
            self.settings.set("output_directory", selected)
            self.settings.sync()

    def _capture_processing_snapshot(self) -> ProcessingSnapshot:
        configurations = tuple(row.configuration() for row in self.configuration_rows)
        output_directory = Path(self.output_path.text()).expanduser().resolve()
        if output_directory.exists() and not output_directory.is_dir():
            raise ValueError(f"Output path is a file: {output_directory}")
        _validate_configurations(list(configurations), self.configuration)
        return ProcessingSnapshot(
            tuple(self._queue_path(self.queue.item(index)) for index in range(self.queue.count())),
            configurations,
            self.preset_combo.currentData(),
            self.sampling_combo.currentData(),
            int(self.sample_count_combo.currentText()),
            self.aggregation_combo.currentData(),
            output_directory,
            str(self.collision_combo.currentText()).lower().replace(" ", "_"),
            self.preserve_audio.isChecked(),
        )

    @staticmethod
    def _prepare_output_directory(output_directory: Path) -> None:
        output_directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".cfh-write-check-", dir=output_directory) as probe_directory:
            with tempfile.TemporaryFile(dir=probe_directory):
                pass

    def _set_batch_controls(self, processing: bool) -> None:
        self._processing = processing
        controls = [
            self.open_button, self.queue, self.start_button, self.output_browse_button,
            self.preset_combo, self.add_configuration_button, self.set_default_button,
            self.sampling_combo, self.sample_count_combo, self.aggregation_combo,
            self.collision_combo, self.preserve_audio,
            *(control for row in self.configuration_rows for control in (
                row.name, row.calibration, row.reconstruction, row.reference_focal_container, row.remove
            )),
        ]
        for control in controls:
            control.setEnabled(not processing)
        self.cancel.setEnabled(processing)
        self._refresh_queue_buttons()

    def _start_selected(self) -> None:
        if self._processing or (self._job_thread is not None and self._job_thread.isRunning()):
            return
        if self.queue.count() == 0:
            self.progress.setText("Queue is empty")
            return
        try:
            snapshot = self._capture_processing_snapshot()
            self._prepare_output_directory(snapshot.output_directory)
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            self.progress.setText(f"Failed: {exc}")
            return
        self._active_snapshot = snapshot
        for index in range(self.queue.count()):
            self._set_queue_item_state(index, JobState.WAITING.value)
        job = ProcessingJob("queue", operation=lambda current_job: self._process_queue(current_job, snapshot))
        self._job_thread = JobThread(job)
        self._job_thread.update.connect(self._on_job_update)
        self._job_thread.item_state.connect(self._on_queue_item_state)
        self._job_thread.failed.connect(self._on_job_failed)
        self._job_thread.finished.connect(self._on_job_finished)
        self._set_batch_controls(True)
        self._job_thread.start()

    def _process_queue(self, job: ProcessingJob, snapshot: ProcessingSnapshot) -> BatchResult:
        completed = []
        failures = []
        for index, input_path in enumerate(snapshot.queue_paths):
            if job.cancelled:
                for cancelled_index in range(index, len(snapshot.queue_paths)):
                    job.report_item(cancelled_index, JobState.CANCELLED.value)
                raise InterruptedError("Queue processing cancelled")
            job.report_item(index, JobState.PROCESSING.value)
            try:
                output_dir, session = self._process_media(input_path, job, snapshot)
            except InterruptedError:
                for cancelled_index in range(index, len(snapshot.queue_paths)):
                    job.report_item(cancelled_index, JobState.CANCELLED.value)
                raise
            except Exception as exc:
                failures.append(f"{input_path.name}: {type(exc).__name__}: {exc}")
                job.report_item(index, JobState.FAILED.value, str(exc))
                continue
            completed.append((input_path, output_dir, session))
            job.report_item(index, JobState.COMPLETED.value)
        return BatchResult(tuple(completed), tuple(failures))

    def _on_job_update(self, event) -> None:
        stage = f" — {event.stage}" if event.stage else ""
        self.progress.setText(f"{event.state.value}{stage} ({event.current}/{event.total}, {event.progress:.0%})")

    def _on_queue_item_state(self, index: int, state: str, detail: str) -> None:
        self._set_queue_item_state(index, state, detail)

    def _on_job_failed(self, message: str) -> None:
        self.progress.setText(f"Failed: {message}")

    def _on_job_finished(self) -> None:
        self._set_batch_controls(False)
        job = self._job_thread.job
        if job.state is JobState.CANCELLED:
            self.progress.setText("Cancelled")
            return
        if job.state is JobState.FAILED:
            if not self.progress.text().startswith("Failed:"):
                self.progress.setText("Failed")
            return
        result = job.result
        if not isinstance(result, BatchResult):
            self.progress.setText("Failed: queue processing returned no result")
            return
        if result.failures:
            self.progress.setText(f"Completed with errors ({len(result.failures)} failed)")
            return
        self.progress.setText("Completed")
        if len(self._active_snapshot.queue_paths) == 1 and len(result.completed) == 1:
            _, output_dir, session = result.completed[0]
            try:
                if len(session.configurations) == 1:
                    result_path = output_dir / session.final_results[session.configurations[0].id]["path"]
                    viewer = ResultViewer(result_path)
                else:
                    viewer = ComparisonViewer(output_dir / "session.json")
                viewer.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
                viewer.show()
                self._result_viewer = viewer
            except Exception as exc:
                self.progress.setText(f"Completed — Viewer could not open: {exc}")

    def _process_media(self, input_path: Path, job=None, selection: ProcessingSnapshot | dict | None = None):
        info = probe_media(input_path)
        if info.width is None or info.height is None:
            raise RuntimeError(f"Media dimensions are unavailable: {input_path.name}")
        root = Path(__file__).resolve().parents[4]
        if isinstance(selection, ProcessingSnapshot):
            configurations, preset_id = selection.configurations, selection.preset_id
            output_base, collision_policy = selection.output_directory, selection.collision_policy
            sampling_mode, sample_count, aggregation = (
                selection.sampling_mode, selection.sample_count, selection.aggregation
            )
            preserve_audio = selection.preserve_audio
        else:
            selection = selection or {
                "configurations": tuple(row.configuration() for row in self.configuration_rows),
                "preset_id": self.preset_combo.currentData(),
            }
            configurations, preset_id = selection["configurations"], selection["preset_id"]
            output_base = Path(self.output_path.text())
            collision_policy = self.collision_combo.currentText().lower().replace(" ", "_")
            sampling_mode, sample_count, aggregation = (
                self.sampling_combo.currentData(), int(self.sample_count_combo.currentText()),
                self.aggregation_combo.currentData(),
            )
            preserve_audio = self.preserve_audio.isChecked()
        if job: job.report(1, 1, stage="Preparing media")
        frames = []
        for frame in FFmpegDecoder().decode_rgb(input_path, info.width, info.height, info.frame_rate):
            if job and job.cancelled:
                raise InterruptedError("Queue processing cancelled during media decoding")
            frames.append(frame)
        output_dir = choose_output_directory(output_base, input_path.stem, collision_policy)
        settings = {
            "sampling_mode": sampling_mode, "sample_count": sample_count, "aggregation": aggregation,
            "preserve_audio": preserve_audio, "input_path": str(input_path), "preset_id": preset_id,
            "fps": info.frame_rate or 30.0, "create_preview": True,
        }
        def progress(stage, current, total):
            if job:
                position = selection.queue_paths.index(input_path) + 1 if isinstance(selection, ProcessingSnapshot) else 1
                queue_total = len(selection.queue_paths) if isinstance(selection, ProcessingSnapshot) else 1
                job.report(current, total, stage=f"{position}/{queue_total} {input_path.name}: {stage}")
        session = process_comparison_session(
            frames, list(configurations), self.configuration, root, output_dir, settings,
            progress, lambda: bool(job and job.cancelled),
        )
        return output_dir, session

    def _cancel_job(self) -> None:
        if self._job_thread is not None:
            self._job_thread.cancel()

    def _open_result(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self, "Open saved result", "", "NumPy result (*.npz);;Comparison session (session.json)"
        )
        if path:
            result_path = Path(path)
            viewer = ComparisonViewer(result_path) if result_path.name == "session.json" else ResultViewer(result_path)
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
