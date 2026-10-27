from calibfreehands.app.settings import AppSettings
from PySide6.QtWidgets import QCheckBox, QComboBox, QLabel, QListWidgetItem
import importlib
import threading
import time
import pytest
from PySide6.QtCore import Qt
from pathlib import Path
from types import SimpleNamespace


def test_settings_persist(tmp_path):
    settings = AppSettings(tmp_path / "settings.ini")
    settings.set("language", "ko")
    settings.set("sample_count", 12)
    settings.sync()
    restored = AppSettings(tmp_path / "settings.ini")
    assert restored.get("language") == "ko"
    assert restored.get("sample_count") == 12


def test_model_and_preset_defaults_persist_and_missing_ids_fall_back(tmp_path, monkeypatch, qapp):
    main = importlib.import_module("calibfreehands.app.main")
    settings_path = tmp_path / "settings.ini"
    monkeypatch.setattr(main, "AppSettings", lambda: AppSettings(settings_path))
    window = main.MainWindow()

    assert window.preset_combo.currentData() == "original"
    assert len(window.configuration_rows) == 1
    assert window.calibration_combo.currentData() == "anycalib_original"
    assert window.reconstruction_combo.currentData() == "wilor_original"
    assert window.sampling_combo.currentData() == "uniform"
    assert window.sample_count_combo.currentText() == "8"
    assert window.aggregation_combo.currentData() == "median"
    handcalib_index = window.calibration_combo.findData("handcalib")
    assert handcalib_index >= 0
    assert not window.calibration_combo.model().item(handcalib_index).isEnabled()
    assert "no confirmed" in window.calibration_combo.itemData(handcalib_index, Qt.ItemDataRole.ToolTipRole)

    window._save_defaults()
    window.close()
    restored = main.MainWindow()
    assert restored.preset_combo.currentData() == "original"
    assert restored.calibration_combo.currentData() == "anycalib_original"
    assert restored.reconstruction_combo.currentData() == "wilor_original"
    restored.close()

    settings = AppSettings(settings_path)
    settings.set("default_preset_id", "removed_preset")
    settings.set("default_calibration_model_id", "removed_calibration")
    settings.set("default_reconstruction_model_id", "removed_pose")
    settings.sync()
    fallback = main.MainWindow()
    assert fallback.preset_combo.currentData() == "original"
    assert fallback.calibration_combo.currentData() == "anycalib_original"
    assert fallback.reconstruction_combo.currentData() == "wilor_original"
    fallback.close()


def test_preset_updates_controls_and_manual_change_does_not_mutate_definition(tmp_path, monkeypatch, qapp):
    main = importlib.import_module("calibfreehands.app.main")
    monkeypatch.setattr(main, "AppSettings", lambda: AppSettings(tmp_path / "settings.ini"))
    window = main.MainWindow()
    original = window.configuration.presets["original"]

    window.sampling_combo.setCurrentIndex(window.sampling_combo.findData("all"))
    window.sample_count_combo.setCurrentText("16")
    window.aggregation_combo.setCurrentText("mean")
    assert window.preset_combo.currentData() == "custom"
    assert (original.sampling, original.sample_count, original.aggregation) == ("uniform", 8, "median")

    window.preset_combo.setCurrentIndex(window.preset_combo.findData("original"))
    assert window.sampling_combo.currentData() == "uniform"
    assert window.sample_count_combo.currentText() == "8"
    assert window.aggregation_combo.currentData() == "median"
    window.close()


def test_configuration_rows_allow_duplicates_and_keep_one_row(tmp_path, monkeypatch, qapp):
    main = importlib.import_module("calibfreehands.app.main")
    monkeypatch.setattr(main, "AppSettings", lambda: AppSettings(tmp_path / "settings.ini"))
    window = main.MainWindow()
    first = window.configuration_rows[0]
    first_id = first.id
    window._add_configuration_row()
    second = window.configuration_rows[1]
    assert second.calibration.currentData() == first.calibration.currentData()
    assert second.reconstruction.currentData() == first.reconstruction.currentData()
    assert first.id == first_id and second.id != first_id
    assert first.remove.isEnabled() and second.remove.isEnabled()

    second.calibration.setCurrentIndex(second.calibration.findData("checkerboard_reference"))
    assert second.reference_focal.isEnabled()
    second.reference_focal.setValue(1234.5)
    assert second.configuration().reference_focal_px == 1234.5
    assert window.preset_combo.currentData() == "custom"
    second.remove.click()
    assert len(window.configuration_rows) == 1
    assert not window.configuration_rows[0].remove.isEnabled()
    window.close()


def test_all_configuration_rows_and_processing_settings_persist_across_restart(tmp_path, monkeypatch, qapp):
    main = importlib.import_module("calibfreehands.app.main")
    settings_path = tmp_path / "settings.ini"
    monkeypatch.setattr(main, "AppSettings", lambda: AppSettings(settings_path))
    window = main.MainWindow()
    first = window.configuration_rows[0]
    first.name.setText("Reference")
    first.calibration.setCurrentIndex(first.calibration.findData("checkerboard_reference"))
    first.reference_focal.setValue(1280.5)
    window._add_configuration_row()
    second = window.configuration_rows[-1]
    second.name.setText("Baseline")
    second.calibration.setCurrentIndex(second.calibration.findData("anycalib_original"))
    window._add_configuration_row()
    third = window.configuration_rows[-1]
    third.name.setText("Proposed")
    third.calibration.setCurrentIndex(third.calibration.findData("checkerboard_reference"))
    third.reference_focal.setValue(900.0)
    window.sampling_combo.setCurrentIndex(window.sampling_combo.findData("all"))
    window.sample_count_combo.setCurrentText("16")
    window.aggregation_combo.setCurrentIndex(window.aggregation_combo.findData("mean"))
    ids = [row.id for row in window.configuration_rows]
    window._save_defaults()
    assert window.configuration.presets["original"].calibration_model_id == "anycalib_original"
    window.close()

    restored = main.MainWindow()
    assert restored.preset_combo.currentData() == "custom"
    assert [row.id for row in restored.configuration_rows] == ids
    assert [row.name.text() for row in restored.configuration_rows] == ["Reference", "Baseline", "Proposed"]
    assert [row.calibration.currentData() for row in restored.configuration_rows] == [
        "checkerboard_reference", "anycalib_original", "checkerboard_reference"
    ]
    assert [row.reconstruction.currentData() for row in restored.configuration_rows] == ["wilor_original"] * 3
    assert [row.reference_focal.value() for row in restored.configuration_rows] == [1280.5, 1000.0, 900.0]
    assert restored.sampling_combo.currentData() == "all"
    assert restored.sample_count_combo.currentText() == "16"
    assert restored.aggregation_combo.currentData() == "mean"
    restored.close()


@pytest.mark.parametrize("saved_json", ["{broken", "[]", "{}"])
def test_malformed_or_empty_saved_configuration_falls_back_to_original(tmp_path, monkeypatch, qapp, saved_json):
    main = importlib.import_module("calibfreehands.app.main")
    settings_path = tmp_path / "settings.ini"
    settings = AppSettings(settings_path)
    settings.set("default_preset_id", "custom")
    settings.set("default_configurations_json", saved_json)
    settings.sync()
    monkeypatch.setattr(main, "AppSettings", lambda: AppSettings(settings_path))
    window = main.MainWindow()
    assert window.preset_combo.currentData() == "original"
    assert len(window.configuration_rows) == 1
    assert window.calibration_combo.currentData() == "anycalib_original"
    assert window.reconstruction_combo.currentData() == "wilor_original"
    assert window.sampling_combo.currentData() == "uniform"
    assert window.sample_count_combo.currentText() == "8"
    assert window.aggregation_combo.currentData() == "median"
    window.close()


def test_unavailable_model_in_saved_configuration_falls_back_to_original(tmp_path, monkeypatch, qapp):
    import json
    main = importlib.import_module("calibfreehands.app.main")
    settings_path = tmp_path / "settings.ini"
    settings = AppSettings(settings_path)
    settings.set("default_preset_id", "custom")
    settings.set("default_configurations_json", json.dumps({"schema_version": 1, "configurations": [
        {"id": "row", "name": "Invalid", "calibration_model_id": "handcalib",
         "reconstruction_model_id": "wilor_original"}
    ]}))
    settings.sync()
    monkeypatch.setattr(main, "AppSettings", lambda: AppSettings(settings_path))
    window = main.MainWindow()
    assert window.preset_combo.currentData() == "original"
    assert window.calibration_combo.currentData() == "anycalib_original"
    window.close()


def test_custom_default_selection_is_restored_after_restart(tmp_path, monkeypatch, qapp):
    main = importlib.import_module("calibfreehands.app.main")
    settings_path = tmp_path / "settings.ini"
    monkeypatch.setattr(main, "AppSettings", lambda: AppSettings(settings_path))
    window = main.MainWindow()
    window.sampling_combo.setCurrentIndex(window.sampling_combo.findData("all"))
    window.sample_count_combo.setCurrentText("16")
    window.aggregation_combo.setCurrentIndex(window.aggregation_combo.findData("mean"))
    window._save_defaults()
    window.close()

    restored = main.MainWindow()
    assert restored.preset_combo.currentData() == "custom"
    assert restored.sampling_combo.currentData() == "all"
    assert restored.sample_count_combo.currentText() == "16"
    assert restored.aggregation_combo.currentData() == "mean"
    restored.close()


def test_processing_path_uses_selected_registry_models_and_records_selection(tmp_path, monkeypatch, qapp):
    main = importlib.import_module("calibfreehands.app.main")
    monkeypatch.setattr(main, "AppSettings", lambda: AppSettings(tmp_path / "settings.ini"))
    window = main.MainWindow()
    source = tmp_path / "input.mp4"
    source.touch()
    monkeypatch.setattr(main, "probe_media", lambda path: SimpleNamespace(width=8, height=8, frame_rate=25.0))
    monkeypatch.setattr(main, "FFmpegDecoder", lambda: SimpleNamespace(decode_rgb=lambda *args: ["frame"]))
    monkeypatch.setattr(main, "choose_output_file", lambda *args: tmp_path / "output.cfh")
    captured = {}
    def process(frames, configurations, config, root, out, settings, *args):
        captured.update(configurations=configurations, settings=settings)
        from test_queue_workflow import _synthetic_session
        return _synthetic_session(out, configurations, settings)
    monkeypatch.setattr(main, "process_comparison_session", process)
    selection = {
        "configurations": [main.PipelineConfiguration("stable-id", "Original", "anycalib_original", "wilor_original")],
        "preset_id": "original",
    }

    window._process_media(source, selection=selection)

    assert captured["configurations"][0].calibration_model_id == "anycalib_original"
    assert captured["configurations"][0].reconstruction_model_id == "wilor_original"
    assert captured["settings"]["preset_id"] == "original"
    assert captured["settings"]["create_preview"] is False
    assert (tmp_path / "output.cfh").is_file()
    assert {path.name for path in tmp_path.iterdir()} == {"settings.ini", "input.mp4", "output.cfh"}
    window.close()


def test_main_window_groups_processing_settings_without_unused_audio_control(tmp_path, monkeypatch, qapp):
    main = importlib.import_module("calibfreehands.app.main")

    monkeypatch.setattr(main, "AppSettings", lambda: AppSettings(tmp_path / "settings.ini"))
    MainWindow = main.MainWindow

    window = MainWindow()
    labels = [label.text() for label in window.findChildren(QLabel)]
    combos = window.findChildren(QComboBox)
    processing_row = next(item.layout() for item in (window.centralWidget().layout().itemAt(i) for i in range(window.centralWidget().layout().count())) if item and item.layout() and item.layout().count() == 4 and item.layout().itemAt(0).layout() is not None and item.layout().itemAt(0).layout().itemAt(0).widget().text() == "Collision")
    sampling = next(combo for combo in combos if [combo.itemText(i) for i in range(combo.count())] == ["Uniform", "All frames"])
    sample_count = next(combo for combo in combos if [combo.itemText(i) for i in range(combo.count())] == ["4", "8", "16", "32"])
    collision = next(combo for combo in combos if [combo.itemText(i) for i in range(combo.count())] == ["Ask", "Overwrite", "Auto rename", "Skip"])
    aggregation = next(combo for combo in combos if [combo.itemText(i) for i in range(combo.count())] == ["Median", "Mean"])

    assert "N" not in labels
    assert {"Collision", "Sampling", "Sample count", "Aggregate"}.issubset(labels)
    compact_groups = [processing_row.itemAt(index).layout() for index in range(4)]
    assert all(group is not None and group.contentsMargins().left() == 0 and group.spacing() <= 6 for group in compact_groups)
    assert processing_row.spacing() > max(group.spacing() for group in compact_groups)
    assert [group.itemAt(0).widget().text() for group in compact_groups] == ["Collision", "Sampling", "Sample count", "Aggregate"]
    assert collision.currentText() == "Ask"
    assert sampling.currentText() == "Uniform"
    assert sample_count.currentText() == str(window.settings.get("sample_count"))
    assert aggregation.currentText() == "Median"
    assert not any(box.text() == "Preserve audio" for box in window.findChildren(QCheckBox))
    sampling.setCurrentText("All frames")
    assert not sample_count.isEnabled()
    sampling.setCurrentText("Uniform")
    assert sample_count.isEnabled()
    window.close()


def test_main_window_prevents_concurrent_jobs_and_restores_controls(tmp_path, monkeypatch, qapp):
    main = importlib.import_module("calibfreehands.app.main")
    monkeypatch.setattr(main, "AppSettings", lambda: AppSettings(tmp_path / "settings.ini"))
    window = main.MainWindow()
    window.queue.addItem(QListWidgetItem("sample.mp4"))
    window.queue.setCurrentRow(0)
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def process_media(path, job, selection):
        calls.append(path)
        entered.set()
        assert release.wait(2)

    monkeypatch.setattr(window, "_process_media", process_media)
    assert window.start_button.isEnabled()
    assert not window.cancel.isEnabled()

    window._start_selected()
    thread = window._job_thread
    assert entered.wait(2)
    qapp.processEvents()
    assert not window.start_button.isEnabled()
    assert window.cancel.isEnabled()

    window._start_selected()
    assert window._job_thread is thread
    assert len(calls) == 1
    assert calls[0].name == "sample.mp4"

    release.set()
    assert thread.wait(2000)
    qapp.processEvents()
    assert window.start_button.isEnabled()
    assert not window.cancel.isEnabled()
    window.close()


def test_main_window_restores_controls_after_job_failure(tmp_path, monkeypatch, qapp):
    main = importlib.import_module("calibfreehands.app.main")
    monkeypatch.setattr(main, "AppSettings", lambda: AppSettings(tmp_path / "settings.ini"))
    window = main.MainWindow()
    source = tmp_path / "sample.mp4"
    source.touch()
    window._add_paths([source])
    window.queue.setCurrentRow(0)
    monkeypatch.setattr(window, "_process_media", lambda path, job, selection: (_ for _ in ()).throw(RuntimeError("failed")))

    window._start_selected()
    thread = window._job_thread
    assert thread.wait(2000)
    qapp.processEvents()
    assert thread.job.state.value == "Completed"
    assert window.queue.item(0).data(window.STATE_ROLE) == "Failed"
    assert window.progress.text().startswith("Completed with errors")
    assert window.start_button.isEnabled()
    assert not window.cancel.isEnabled()
    window.close()


def test_main_window_restores_controls_after_cancellation(tmp_path, monkeypatch, qapp):
    main = importlib.import_module("calibfreehands.app.main")
    monkeypatch.setattr(main, "AppSettings", lambda: AppSettings(tmp_path / "settings.ini"))
    window = main.MainWindow()
    window.queue.addItem(QListWidgetItem("sample.mp4"))
    window.queue.setCurrentRow(0)
    entered = threading.Event()

    def process_media(path, job, selection):
        entered.set()
        while not job.cancelled:
            time.sleep(0.001)
        raise InterruptedError

    monkeypatch.setattr(window, "_process_media", process_media)
    window._start_selected()
    thread = window._job_thread
    assert entered.wait(2)
    window._cancel_job()
    assert thread.wait(2000)
    qapp.processEvents()
    assert thread.job.state.value == "Cancelled"
    assert window.start_button.isEnabled()
    assert not window.cancel.isEnabled()
    window.close()


@pytest.mark.parametrize(
    "filenames, expected_viewer",
    [(["result.cfh"], "result"), (["a.cfh", "b.cfh"], "comparison")],
)
def test_view_results_opens_single_bundle_or_same_source_comparison(tmp_path, monkeypatch, qapp,
                                                                    filenames, expected_viewer):
    main = importlib.import_module("calibfreehands.app.main")
    monkeypatch.setattr(main, "AppSettings", lambda: AppSettings(tmp_path / "settings.ini"))
    opened = {}

    class Viewer:
        viewer_type = "unknown"
        def __init__(self, path): opened["path"] = path; opened["viewer"] = self.viewer_type
        def setAttribute(self, *_args): pass
        def show(self): opened["shown"] = True

    class ComparisonStub(Viewer): viewer_type = "comparison"
    class ResultStub(Viewer): viewer_type = "result"

    paths = [tmp_path / filename for filename in filenames]
    monkeypatch.setattr(main.QFileDialog, "getOpenFileNames", lambda *args: ([str(path) for path in paths], ""))
    monkeypatch.setattr(main, "ComparisonViewer", ComparisonStub)
    monkeypatch.setattr(main, "ResultViewer", ResultStub)
    monkeypatch.setattr(main, "load_result_bundle", lambda path: SimpleNamespace(metadata={"source_media_sha256": "a" * 64}))
    window = main.MainWindow()
    window._open_result()
    assert opened["viewer"] == expected_viewer
    if expected_viewer == "comparison":
        assert len(opened["path"]) == 2
    else:
        assert opened["path"].metadata["source_media_sha256"] == "a" * 64
    assert opened["shown"]
    window.close()
