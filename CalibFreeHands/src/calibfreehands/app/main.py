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
    QHBoxLayout, QVBoxLayout, QWidget, QComboBox, QLineEdit, QDoubleSpinBox,
    QAbstractItemView, QGridLayout, QMessageBox, QInputDialog, QMenu, QToolButton, QSizePolicy, QStyle,
)

from calibfreehands.media import discover_media
from calibfreehands.media.ffmpeg import FFmpegDecoder
from calibfreehands.media.probe import probe_media
from calibfreehands.app.configuration import (
    ApplicationConfiguration,
    ModelDefinition,
    PresetDefinition,
    load_configuration,
)
from calibfreehands.app.presets import (
    export_user_preset,
    import_user_preset,
    load_user_presets,
    preset_to_dict,
    save_user_presets,
    validate_user_preset,
)
from calibfreehands.app.settings import AppSettings
from calibfreehands.app.native_media_dialog import NativeMediaDialogError, select_media_paths
from calibfreehands.export.paths import choose_output_file, result_filename
from calibfreehands.pipeline import JobState, JobThread, ProcessingJob
from calibfreehands.pipeline.comparison import (
    ComparisonSession,
    PipelineConfiguration,
    _validate_configurations,
    process_comparison_session,
)
from calibfreehands.pipeline.provenance import sha256_file
from calibfreehands.pipeline.result_bundle import ResultBundle, load_result_bundle, write_result_bundle
from calibfreehands.app.media_picker import MediaOpenDialog
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
    completed: tuple[tuple[Path, tuple[Path, ...], ComparisonSession | None], ...]
    failures: tuple[str, ...]


class ConfigurationRow:
    def __init__(self, models: ApplicationConfiguration, name: str, calibration_id: str,
                 reconstruction_id: str, reference_focal_px: float = 1000.0, config_id: str | None = None):
        self.id = config_id or uuid.uuid4().hex
        self.name = QLineEdit(name)
        name_content_width = self.name.fontMetrics().horizontalAdvance("Configuration 1")
        frame_width = self.name.style().pixelMetric(QStyle.PixelMetric.PM_DefaultFrameWidth)
        self.name.setMinimumWidth(name_content_width + 2 * frame_width + 8)
        self.calibration = MainWindow._model_combo(models.calibration_models)
        self.reconstruction = MainWindow._model_combo(models.reconstruction_models)
        for combo in (self.calibration, self.reconstruction):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        MainWindow._select_data(self.calibration, calibration_id)
        MainWindow._select_data(self.reconstruction, reconstruction_id)
        self.reference_focal = QDoubleSpinBox()
        self.reference_focal.setRange(1.0, 100000.0)
        self.reference_focal.setDecimals(2)
        self.reference_focal.setSuffix(" px")
        self.reference_focal.setValue(reference_focal_px)
        self.reference_focal_container = QWidget()
        focal_layout = QHBoxLayout(self.reference_focal_container)
        focal_layout.setContentsMargins(0, 0, 0, 0)
        focal_layout.setSpacing(4)
        focal_layout.addWidget(self.reference_focal)
        self.remove = QPushButton("Remove")
        self.widgets = (self.name, self.calibration, self.reconstruction,
                        self.reference_focal_container, self.remove)
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
        self.setMinimumSize(980, 560)
        self.queue = QListWidget()
        self.queue.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.queue.itemSelectionChanged.connect(self._refresh_queue_buttons)
        self.setAcceptDrops(True)
        self.settings = AppSettings()
        self.configuration = load_configuration()
        self._preset_origin_id: str | None = None
        self.preset_store_path = Path(QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppConfigLocation
        )) / "user_presets.json"
        try:
            self.user_presets = load_user_presets(self.preset_store_path, self.configuration)
            self.preset_store_error = None
        except ValueError as exc:
            self.user_presets = []
            self.preset_store_error = str(exc)
        self._updating_controls = False
        self.open_button = QPushButton("Open...")
        self.open_button.clicked.connect(self._open_media)
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self._remove_selected)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self._clear_queue)
        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self._start_selected)
        open_result = QPushButton("View Results...")
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
            self.preset_combo.addItem(f"{preset.display_name} (Built-in)", preset.id)
        for preset in self.user_presets:
            self.preset_combo.addItem(f"{preset.display_name} (User)", preset.id)
        self.preset_combo.addItem("Custom (unsaved)", "custom")
        self.preset_actions_button = QToolButton()
        self.preset_actions_button.setText("Preset Actions")
        self.preset_actions_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        preset_menu = QMenu(self.preset_actions_button)
        self.save_preset_action = preset_menu.addAction("Save Preset...")
        self.manage_presets_action = preset_menu.addAction("Manage Presets...")
        self.import_preset_action = preset_menu.addAction("Import Preset...")
        self.export_preset_action = preset_menu.addAction("Export Preset...")
        self.preset_actions_button.setMenu(preset_menu)
        self.save_preset_action.triggered.connect(self._save_user_preset_dialog)
        self.manage_presets_action.triggered.connect(self._manage_user_presets)
        self.import_preset_action.triggered.connect(self._import_user_preset_dialog)
        self.export_preset_action.triggered.connect(self._export_user_preset_dialog)
        self.configuration_rows = []
        self.rows_widget = QWidget()
        self.rows_layout = QGridLayout(self.rows_widget)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setHorizontalSpacing(8)
        self.rows_layout.setVerticalSpacing(4)
        for column, title in enumerate(("Name", "Calibration", "Reconstruction", "Reference Focal", "Action")):
            self.rows_layout.addWidget(QLabel(title), 0, column)
        self.rows_layout.setColumnStretch(0, 2)
        self.rows_layout.setColumnStretch(1, 3)
        self.rows_layout.setColumnStretch(2, 3)
        self.rows_layout.setColumnStretch(5, 1)
        self.set_default_button = QPushButton("Set as default")
        self.set_default_button.clicked.connect(self._save_defaults)
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Preset"))
        model_row.addWidget(self.preset_combo, 1)
        model_row.addWidget(self.preset_actions_button)
        self.add_configuration_button = QPushButton("Add configuration")
        self.add_configuration_button.clicked.connect(self._add_configuration_row)
        model_row.addWidget(self.add_configuration_button)
        model_row.addWidget(self.set_default_button)
        settings_row = QHBoxLayout()
        theme = QComboBox(); theme.addItems(["System", "Light", "Dark"]); theme.setCurrentText(self.settings.get("theme").title()); theme.currentTextChanged.connect(self._set_theme)
        language = QComboBox(); language.addItems(["English", "한국어"]); language.setCurrentText("한국어" if self.settings.get("language") == "ko" else "English"); language.currentTextChanged.connect(lambda value: self._set_setting("language", "ko" if value == "한국어" else "en"))
        self.theme_combo, self.language_combo = theme, language
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
        for control in (theme, language, collision, sampling, sample_count, aggregation):
            control.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
            control.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.sampling_combo = sampling
        self.sample_count_combo = sample_count
        self.aggregation_combo = aggregation
        self.collision_combo = collision
        self.theme_group = self._label_control_group("Theme", theme)
        self.language_group = self._label_control_group("Language", language)
        settings_row.addWidget(self.theme_group)
        settings_row.addSpacing(16)
        settings_row.addWidget(self.language_group)
        settings_row.addStretch(1)
        self.settings_row = settings_row
        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output"))
        output_row.addWidget(self.output_path, 1)
        output_row.addWidget(self.output_browse_button)
        self.output_path.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        processing_row = QHBoxLayout()
        processing_row.setSpacing(14)
        self.processing_groups = []
        for label, control in (("Collision", collision), ("Sampling", sampling), ("Sample count", sample_count), ("Aggregate", aggregation)):
            group = self._label_control_group(label, control)
            self.processing_groups.append(group)
            processing_row.addWidget(group)
        processing_row.addStretch(1)
        self.processing_row = processing_row
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
        self._open_viewers = []
        self._restore_model_defaults()
        self.preset_combo.currentIndexChanged.connect(self._apply_selected_preset)
        for control in (sampling, sample_count, aggregation):
            control.currentIndexChanged.connect(self._model_or_processing_changed)
        self._refresh_queue_buttons()
        self._update_preset_actions()
        if self.preset_store_error:
            self.progress.setText(f"User presets could not be loaded: {self.preset_store_error}")

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

    @staticmethod
    def _label_control_group(text: str, control: QWidget) -> QWidget:
        group = QWidget()
        row = QHBoxLayout(group)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        label = QLabel(text)
        label.setBuddy(control)
        row.addWidget(label)
        row.addWidget(control)
        return group

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
        grid_row = len(self.configuration_rows)
        for column, widget in enumerate(row.widgets):
            self.rows_layout.addWidget(widget, grid_row, column)
        focal_width = row.reference_focal_container.sizeHint().width()
        self.rows_layout.setColumnMinimumWidth(
            3, max(self.rows_layout.columnMinimumWidth(3), focal_width)
        )
        self._refresh_row_buttons()
        return row

    def _remove_row_widgets(self, row):
        for widget in row.widgets:
            self.rows_layout.removeWidget(widget)
            widget.deleteLater()

    def _reposition_configuration_rows(self):
        for row in self.configuration_rows:
            for widget in row.widgets:
                self.rows_layout.removeWidget(widget)
        for grid_row, row in enumerate(self.configuration_rows, start=1):
            for column, widget in enumerate(row.widgets):
                self.rows_layout.addWidget(widget, grid_row, column)

    def _refresh_row_buttons(self):
        for row in self.configuration_rows:
            row.remove.setEnabled(len(self.configuration_rows) > 1)

    def _add_configuration_row(self):
        current = self.configuration_rows[-1].configuration()
        used_names = {row.name.text().strip() for row in self.configuration_rows}
        next_index = 1
        while f"Configuration {next_index}" in used_names:
            next_index += 1
        self._add_row(f"Configuration {next_index}", current.calibration_model_id,
                      current.reconstruction_model_id, current.reference_focal_px or 1000.0)
        self._model_or_processing_changed()

    def _remove_configuration_row(self, row):
        if len(self.configuration_rows) <= 1:
            return
        self.configuration_rows.remove(row)
        self._remove_row_widgets(row)
        self._reposition_configuration_rows()
        self._refresh_row_buttons()
        self._model_or_processing_changed()

    def _replace_rows(self, name, calibration_id, reconstruction_id):
        self._replace_configurations([PipelineConfiguration(
            uuid.uuid4().hex, name, calibration_id, reconstruction_id
        )])

    def _replace_configurations(self, configurations):
        for row in self.configuration_rows:
            self._remove_row_widgets(row)
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
                preset = self._preset_by_id(preset_id)
                if preset_id != "custom" and (preset is None or not self._matches_preset_values(
                    preset, configurations, sampling, count, aggregation
                )):
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
        elif self._preset_by_id(preset_id) is not None:
            self._apply_preset_values(self._preset_by_id(preset_id))
        else:
            self._apply_preset_values(self.configuration.presets["original"])

    def _select_preset_id(self, preset_id):
        if self.preset_combo.findData(preset_id) < 0:
            self.preset_combo.addItem("Custom (unsaved)", preset_id)
        self._select_data(self.preset_combo, preset_id)

    def _preset_by_id(self, preset_id):
        return self.configuration.presets.get(preset_id) or next(
            (preset for preset in self.user_presets if preset.id == preset_id), None
        )

    def _refresh_preset_combo(self, selected_id=None) -> None:
        selected_id = self.preset_combo.currentData() if selected_id is None else selected_id
        self._updating_controls = True
        try:
            self.preset_combo.clear()
            for preset in (*self.configuration.presets.values(), *self.user_presets):
                kind = "Built-in" if preset.id in self.configuration.presets else "User"
                self.preset_combo.addItem(f"{preset.display_name} ({kind})", preset.id)
            self.preset_combo.addItem("Custom (unsaved)", "custom")
            self._select_data(self.preset_combo, selected_id)
        finally:
            self._updating_controls = False
        self._update_preset_actions()

    def _update_preset_actions(self) -> None:
        selected = self.preset_combo.currentData()
        is_user = any(preset.id == selected for preset in self.user_presets)
        writable = self.preset_store_error is None
        self.save_preset_action.setEnabled(writable)
        self.manage_presets_action.setEnabled(writable and bool(self.user_presets))
        self.import_preset_action.setEnabled(writable)
        self.export_preset_action.setEnabled(writable and is_user)

    def _validate_default_snapshot(self, snapshot):
        if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1:
            raise ValueError("unsupported saved default schema")
        preset_id = snapshot.get("preset_id")
        sampling = snapshot.get("sampling_mode")
        count = snapshot.get("sample_count")
        aggregation = snapshot.get("aggregation")
        configurations = snapshot.get("configurations")
        if (preset_id != "custom" and self._preset_by_id(preset_id) is None
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
            self._preset_origin_id = preset.id
            self._replace_configurations(self._preset_configurations(preset))
            self._set_processing_controls(preset.sampling, preset.sample_count, preset.aggregation)
        finally:
            self._updating_controls = False
        self._update_preset_actions()

    def _apply_selected_preset(self, _index: int) -> None:
        if self._updating_controls:
            return
        preset = self._preset_by_id(self.preset_combo.currentData())
        if preset is not None:
            self._apply_preset_values(preset)
        else:
            self._model_or_processing_changed()

    def _model_or_processing_changed(self, _index: int = 0) -> None:
        if self._updating_controls:
            return
        self.sample_count_combo.setEnabled(self.sampling_combo.currentData() == "uniform")
        configurations = [row.configuration() for row in self.configuration_rows]
        candidates = (*self.configuration.presets.values(), *self.user_presets)
        matches = [preset for preset in candidates if self._matches_preset_values(
            preset, configurations, self.sampling_combo.currentData(), int(self.sample_count_combo.currentText()),
            self.aggregation_combo.currentData()
        )]
        current = (self._preset_by_id(self.preset_combo.currentData())
                   or self._preset_by_id(self._preset_origin_id))
        preset = current if current in matches else next(iter(matches), None)
        if preset is not None:
            custom_index = self.preset_combo.findData("custom")
            self._select_data(self.preset_combo, preset.id)
            self._preset_origin_id = preset.id
        else:
            if self.preset_combo.findData("custom") < 0:
                self.preset_combo.addItem("Custom (unsaved)", "custom")
            self._select_data(self.preset_combo, "custom")
        self._update_preset_actions()

    def _save_defaults(self) -> None:
        first = self.configuration_rows[0].configuration()
        preset_id = self.preset_combo.currentData()
        snapshot = {
            "schema_version": 1,
            "preset_id": preset_id,
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
        default_snapshot = json.dumps(snapshot, separators=(",", ":")) if preset_id == "custom" else ""
        for key, value in (
            ("default_preset_id", preset_id),
            ("default_calibration_model_id", first.calibration_model_id),
            ("default_reconstruction_model_id", first.reconstruction_model_id),
            ("sampling_mode", self.sampling_combo.currentData()),
            ("sample_count", int(self.sample_count_combo.currentText())),
            ("aggregation", self.aggregation_combo.currentData()),
            ("default_configurations_json", default_snapshot),
        ):
            self.settings.set(key, value)
        self.settings.sync()

    def _current_preset_data(self, preset_id: str, name: str) -> dict:
        rows = []
        for row in self.configuration_rows:
            item = row.configuration()
            data = {
                "id": item.id,
                "name": item.name,
                "calibration_model_id": item.calibration_model_id,
                "reconstruction_model_id": item.reconstruction_model_id,
            }
            if item.reference_focal_px is not None:
                data["reference_focal_px"] = item.reference_focal_px
            rows.append(data)
        return {
            "id": preset_id, "name": name, "configurations": rows,
            "sampling_mode": self.sampling_combo.currentData(),
            "sample_count": int(self.sample_count_combo.currentText()),
            "aggregation": self.aggregation_combo.currentData(),
        }

    def _save_user_preset(self, name: str, overwrite_id: str | None = None) -> PresetDefinition:
        existing = next((preset for preset in self.user_presets
                         if preset.display_name.casefold() == name.strip().casefold()), None)
        if existing is not None and existing.id != overwrite_id:
            raise ValueError(f"Preset already exists: {existing.display_name}")
        if overwrite_id is not None:
            original = next((preset for preset in self.user_presets if preset.id == overwrite_id), None)
            if original is None or existing is None or existing.id != overwrite_id:
                raise ValueError("The selected preset can no longer be overwritten")
            preset_id = original.id
            remaining = [preset for preset in self.user_presets if preset.id != preset_id]
        else:
            preset_id = uuid.uuid4().hex
            remaining = self.user_presets
        preset = validate_user_preset(
            self._current_preset_data(preset_id, name), self.configuration, remaining
        )
        updated = [preset if item.id == preset_id else item for item in self.user_presets]
        if not any(item.id == preset_id for item in self.user_presets):
            updated.append(preset)
        save_user_presets(self.preset_store_path, updated)
        self.user_presets = updated
        self._refresh_preset_combo(preset.id)
        self._apply_preset_values(preset)
        return preset

    def _ask_duplicate_preset_action(self, name: str) -> str | None:
        box = QMessageBox(self)
        box.setWindowTitle("Preset already exists")
        box.setText(f"A User Preset named ‘{name}’ already exists.")
        overwrite = box.addButton("Overwrite", QMessageBox.ButtonRole.AcceptRole)
        save_as = box.addButton("Save As...", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        if box.clickedButton() == overwrite:
            return "overwrite"
        if box.clickedButton() == save_as:
            return "save_as"
        return None

    def _save_user_preset_dialog(self) -> None:
        origin = next((preset for preset in self.user_presets
                       if preset.id == self._preset_origin_id), None)
        name, accepted = QInputDialog.getText(
            self, "Save Preset", "Preset name:", text=origin.display_name if origin else ""
        )
        if not accepted:
            return
        overwrite_id = None
        while True:
            duplicate = next((preset for preset in self.user_presets
                              if preset.display_name.casefold() == name.strip().casefold()), None)
            if duplicate is None:
                break
            action = self._ask_duplicate_preset_action(duplicate.display_name)
            if action == "overwrite":
                overwrite_id = duplicate.id
                break
            if action != "save_as":
                return
            name, accepted = QInputDialog.getText(
                self, "Save As", "New preset name:", text=f"{name.strip()} Copy"
            )
            if not accepted:
                return
        try:
            self._save_user_preset(name, overwrite_id)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Could not save preset", str(exc))

    def _rename_user_preset(self, preset_id: str, name: str) -> None:
        preset = next((item for item in self.user_presets if item.id == preset_id), None)
        if preset is None:
            raise ValueError("Built-in presets cannot be renamed")
        data = preset_to_dict(preset)
        data["name"] = name
        remaining = [item for item in self.user_presets if item.id != preset_id]
        renamed = validate_user_preset(data, self.configuration, remaining)
        updated = [renamed if item.id == preset_id else item for item in self.user_presets]
        save_user_presets(self.preset_store_path, updated)
        self.user_presets = updated
        self._refresh_preset_combo(preset_id)

    def _delete_user_preset(self, preset_id: str) -> None:
        if not any(item.id == preset_id for item in self.user_presets):
            raise ValueError("Built-in presets cannot be deleted")
        updated = [item for item in self.user_presets if item.id != preset_id]
        save_user_presets(self.preset_store_path, updated)
        self.user_presets = updated
        if self.settings.get("default_preset_id") == preset_id:
            self.settings.set("default_preset_id", "original")
            self.settings.set("default_configurations_json", "")
            self.settings.sync()
        if self._preset_origin_id == preset_id:
            self._preset_origin_id = None
        selected = self.preset_combo.currentData()
        if selected == preset_id:
            self._apply_preset_values(self.configuration.presets["original"])
            selected = "original"
        self._refresh_preset_combo(selected)

    def _manage_user_presets(self) -> None:
        if not self.user_presets:
            QMessageBox.information(self, "Manage Presets", "There are no User Presets to manage.")
            return
        names = [preset.display_name for preset in self.user_presets]
        name, accepted = QInputDialog.getItem(self, "Manage Presets", "User Preset:", names, 0, False)
        if not accepted:
            return
        preset = next(item for item in self.user_presets if item.display_name == name)
        box = QMessageBox(self)
        box.setWindowTitle("Manage Presets")
        box.setText(f"Manage User Preset ‘{preset.display_name}’")
        rename = box.addButton("Rename...", QMessageBox.ButtonRole.ActionRole)
        delete = box.addButton("Delete", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        if box.clickedButton() == rename:
            updated, ok = QInputDialog.getText(self, "Rename Preset", "Preset name:", text=preset.display_name)
            if ok:
                try:
                    self._rename_user_preset(preset.id, updated)
                except (OSError, ValueError) as exc:
                    QMessageBox.warning(self, "Could not rename preset", str(exc))
        elif box.clickedButton() == delete:
            try:
                self._delete_user_preset(preset.id)
            except OSError as exc:
                QMessageBox.warning(self, "Could not delete preset", str(exc))

    def _import_user_preset(self, path: Path) -> PresetDefinition:
        preset = import_user_preset(path, self.configuration, self.user_presets)
        updated = [*self.user_presets, preset]
        save_user_presets(self.preset_store_path, updated)
        self.user_presets = updated
        self._refresh_preset_combo(preset.id)
        self._apply_preset_values(preset)
        return preset

    def _import_user_preset_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import Preset", "", "CalibFreeHands Preset (*.cfhpreset)")
        if not path:
            return
        try:
            self._import_user_preset(Path(path))
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Could not import preset", str(exc))

    def _export_user_preset(self, path: Path) -> Path:
        preset = next((item for item in self.user_presets if item.id == self.preset_combo.currentData()), None)
        if preset is None:
            raise ValueError("Select a User Preset before exporting")
        return export_user_preset(path, preset)

    def _export_user_preset_dialog(self) -> None:
        preset = next((item for item in self.user_presets if item.id == self.preset_combo.currentData()), None)
        if preset is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Preset", f"{preset.display_name}.cfhpreset", "CalibFreeHands Preset (*.cfhpreset)"
        )
        if not path:
            return
        target = Path(path)
        if target.suffix.lower() != ".cfhpreset":
            target = target.with_suffix(".cfhpreset")
        try:
            self._export_user_preset(target)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Could not export preset", str(exc))

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

    def _open_media(self) -> None:
        start = Path(self.settings.get("last_input_directory") or Path.home()).expanduser()
        if not start.is_dir():
            start = Path.home()
        try:
            selected = select_media_paths(self, start)
        except NativeMediaDialogError as exc:
            QMessageBox.warning(self, "Windows media dialog unavailable", f"Using the fallback media picker.\n\n{exc}")
            dialog = MediaOpenDialog(self, start)
            if not dialog.exec():
                return
            selected = dialog.selected_paths()
        if not selected:
            return
        self._add_paths(selected)
        last = selected[-1] if selected[-1].is_dir() else selected[-1].parent
        self.settings.set("last_input_directory", str(last.resolve()))
        self.settings.sync()

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
            bool(self.settings.get("preserve_audio")),
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
            self.preset_combo, self.preset_actions_button, self.add_configuration_button, self.set_default_button,
            self.sampling_combo, self.sample_count_combo, self.aggregation_combo,
            self.collision_combo,
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
            _, bundle_paths, _session = result.completed[0]
            if not bundle_paths:
                self.progress.setText("Completed — no new results (all outputs skipped)")
                return
            try:
                bundles = [load_result_bundle(path) for path in bundle_paths]
                self._show_result_bundles(bundles)
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
            preserve_audio = bool(self.settings.get("preserve_audio"))
        if job: job.report(1, 1, stage="Preparing media")
        if job:
            job.report(1, 1, stage="Hashing source media")
        source_sha256 = sha256_file(input_path)
        frames = []
        for frame in FFmpegDecoder().decode_rgb(input_path, info.width, info.height, info.frame_rate):
            if job and job.cancelled:
                raise InterruptedError("Queue processing cancelled during media decoding")
            frames.append(frame)
        destinations = []
        reserved = set()
        for configuration in configurations:
            calibration = self.configuration.calibration_models[configuration.calibration_model_id]
            reconstruction = self.configuration.reconstruction_models[configuration.reconstruction_model_id]
            filename = result_filename(
                input_path.stem, calibration.display_name, reconstruction.display_name,
                configuration.reference_focal_px,
            )
            destination = choose_output_file(output_base, filename, collision_policy)
            if destination is not None and destination in reserved:
                index = 1
                while (output_base / f"{Path(filename).stem}_{index}{Path(filename).suffix}") in reserved or (output_base / f"{Path(filename).stem}_{index}{Path(filename).suffix}").exists():
                    index += 1
                destination = output_base / f"{Path(filename).stem}_{index}{Path(filename).suffix}"
            if destination is not None:
                reserved.add(destination)
            destinations.append(destination)
        active_configurations = [configuration for configuration, destination in zip(configurations, destinations)
                                 if destination is not None]
        if not active_configurations:
            if job:
                job.report(1, 1, stage="All result files skipped")
            return (), None
        settings = {
            "sampling_mode": sampling_mode, "sample_count": sample_count, "aggregation": aggregation,
            "preserve_audio": preserve_audio, "input_path": str(input_path), "preset_id": preset_id,
            "fps": info.frame_rate or 30.0, "create_preview": False, "source_media_sha256": source_sha256,
        }
        def progress(stage, current, total):
            if job:
                position = selection.queue_paths.index(input_path) + 1 if isinstance(selection, ProcessingSnapshot) else 1
                queue_total = len(selection.queue_paths) if isinstance(selection, ProcessingSnapshot) else 1
                job.report(current, total, stage=f"{position}/{queue_total} {input_path.name}: {stage}")
        bundle_paths = []
        mano_hashes = {}
        with tempfile.TemporaryDirectory(prefix="calibfreehands-") as working_directory:
            working_directory = Path(working_directory)
            session = process_comparison_session(
                frames, active_configurations, self.configuration, root, working_directory, settings,
                progress, lambda: bool(job and job.cancelled),
            )
            destination_by_id = {item.id: destination for item, destination in zip(configurations, destinations)}
            for item in active_configurations:
                if job and job.cancelled:
                    raise InterruptedError("Queue processing cancelled while packaging results")
                record = session.final_results[item.id]
                run_path = working_directory / record["run_json"]
                run_metadata = json.loads(run_path.read_text(encoding="utf-8"))
                calibration = session.calibration_results[record["calibration_result_id"]]
                pose = session.raw_pose_results[item.reconstruction_model_id]
                mano_provenance = {}
                for hand, asset_path in pose.get("mano_assets", {}).items():
                    asset = Path(asset_path)
                    if hand not in mano_hashes:
                        mano_hashes[hand] = sha256_file(asset) if asset.is_file() else None
                    mano_provenance[hand] = {"path": str(asset), "sha256": mano_hashes[hand]}
                model_calibration = self.configuration.calibration_models[item.calibration_model_id]
                model_reconstruction = self.configuration.reconstruction_models[item.reconstruction_model_id]
                metadata = {
                    "application_version": "0.1.0", "source_media_name": input_path.name,
                    "source_media_path": str(input_path.resolve()), "source_media_sha256": source_sha256,
                    "image_size": {"width": info.width, "height": info.height},
                    "calibration_model_id": item.calibration_model_id,
                    "calibration_display_name": model_calibration.display_name,
                    "reconstruction_model_id": item.reconstruction_model_id,
                    "reconstruction_display_name": model_reconstruction.display_name,
                    "configuration_id": item.id, "configuration_name": item.name,
                    "reference_configuration_id": session.reference_configuration_id,
                    "focal_px": float(record["focal_px"]), "sampling_mode": sampling_mode,
                    "sample_count": int(sample_count), "aggregation": aggregation,
                    "coordinate_system": run_metadata.get("coordinate_system", "camera_space"),
                    "unit": run_metadata.get("unit", "metre"),
                    "principal_point_applied": run_metadata.get("applied_camera_parameters", {}).get("principal_point_applied", False),
                    "distortion_applied": run_metadata.get("applied_camera_parameters", {}).get("distortion_applied", False),
                    "calibration_provenance": calibration.get("metadata", {}),
                    "reconstruction_checkpoint": pose.get("checkpoint"),
                    "reconstruction_checkpoint_sha256": pose.get("checkpoint_sha256"),
                    "mano_provenance": mano_provenance, "run_provenance": run_metadata,
                }
                destination = destination_by_id[item.id]
                final_path = destination
                if final_path.exists() and collision_policy != "overwrite":
                    if collision_policy == "skip":
                        continue
                    if collision_policy == "auto_rename":
                        final_path = choose_output_file(output_base, final_path.name, "auto_rename")
                    else:
                        raise FileExistsError(f"result appeared during processing; refusing to overwrite: {final_path}")
                bundle_paths.append(write_result_bundle(working_directory / record["path"], metadata, final_path))
                if job and job.cancelled:
                    raise InterruptedError("Queue processing cancelled after atomically writing a result")
        return tuple(bundle_paths), session

    def _cancel_job(self) -> None:
        if self._job_thread is not None:
            self._job_thread.cancel()

    def _show_result_bundles(self, bundles: list[ResultBundle]) -> None:
        if len(bundles) > 1:
            source_ids = {bundle.metadata["source_media_sha256"].casefold() for bundle in bundles}
            if len(source_ids) != 1:
                raise ValueError("Selected results were generated from different source media.")
            viewer = ComparisonViewer(bundles)
        else:
            viewer = ResultViewer(bundles[0])
        viewer.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        viewer.show()
        self._result_viewer = viewer
        self._open_viewers.append(viewer)

    def _open_result(self) -> None:
        start_directory = Path(self.settings.get("last_result_directory") or self.output_path.text()).expanduser()
        if not start_directory.is_dir():
            start_directory = Path(self.output_path.text())
        paths, _selected_filter = QFileDialog.getOpenFileNames(
            self, "View Results...", str(start_directory), "CalibFreeHands Result (*.cfh)"
        )
        if not paths:
            return
        result_paths = [Path(path).resolve() for path in paths]
        self.settings.set("last_result_directory", str(result_paths[-1].parent))
        self.settings.sync()
        try:
            bundles = [load_result_bundle(path) for path in result_paths]
            self._show_result_bundles(bundles)
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Could not open results", str(exc))


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
