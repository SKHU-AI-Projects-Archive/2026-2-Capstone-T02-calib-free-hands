from calibfreehands.app.settings import AppSettings
from PySide6.QtWidgets import QCheckBox, QComboBox, QLabel, QListWidgetItem
import importlib
import threading
import time
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
    assert window.calibration_combo.currentData() == "anycalib_original"
    assert window.reconstruction_combo.currentData() == "wilor_original"
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
    second.remove.click()
    assert len(window.configuration_rows) == 1
    assert not window.configuration_rows[0].remove.isEnabled()
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
    created = []
    monkeypatch.setattr(main, "probe_media", lambda path: SimpleNamespace(width=8, height=8, frame_rate=25.0))
    monkeypatch.setattr(main, "FFmpegDecoder", lambda: SimpleNamespace(decode_rgb=lambda *args: ["frame"]))
    monkeypatch.setattr(main, "choose_output_directory", lambda *args: tmp_path)
    captured = {}
    monkeypatch.setattr(main, "process_comparison_session", lambda frames, configurations, config, root, out, settings, *args: captured.update(
        configurations=configurations, settings=settings
    ))
    selection = {
        "configurations": [main.PipelineConfiguration("stable-id", "Original", "anycalib_original", "wilor_original")],
        "preset_id": "original",
    }

    window._process_media(Path("input.mp4"), selection=selection)

    assert captured["configurations"][0].calibration_model_id == "anycalib_original"
    assert captured["configurations"][0].reconstruction_model_id == "wilor_original"
    assert captured["settings"]["preset_id"] == "original"
    window.close()


def test_main_window_groups_sample_count_and_preserves_sampling_toggle(tmp_path, monkeypatch, qapp):
    main = importlib.import_module("calibfreehands.app.main")

    monkeypatch.setattr(main, "AppSettings", lambda: AppSettings(tmp_path / "settings.ini"))
    MainWindow = main.MainWindow

    window = MainWindow()
    labels = [label.text() for label in window.findChildren(QLabel)]
    combos = window.findChildren(QComboBox)
    processing_row = next(item.layout() for item in (window.centralWidget().layout().itemAt(i) for i in range(window.centralWidget().layout().count())) if item and item.layout() and item.layout().count() == 5 and item.layout().itemAt(0).layout() is not None and item.layout().itemAt(0).layout().itemAt(0).widget().text() == "Collision")
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
    assert next(box for box in window.findChildren(QCheckBox) if box.text() == "Preserve audio").isChecked()
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
    window.queue.addItem(QListWidgetItem("sample.mp4"))
    window.queue.setCurrentRow(0)
    monkeypatch.setattr(window, "_process_media", lambda path, job, selection: (_ for _ in ()).throw(RuntimeError("failed")))

    window._start_selected()
    thread = window._job_thread
    assert thread.wait(2000)
    qapp.processEvents()
    assert thread.job.state.value == "Failed"
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
