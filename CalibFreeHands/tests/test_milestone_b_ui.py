from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QFileDialog, QPushButton

from calibfreehands.app.settings import AppSettings


def _window(tmp_path, monkeypatch):
    main = importlib.import_module("calibfreehands.app.main")
    monkeypatch.setattr(main, "AppSettings", lambda: AppSettings(tmp_path / "settings.ini"))
    return main, main.MainWindow()


def test_open_uses_native_selection_boundary_with_persistent_input_location(tmp_path, monkeypatch, qapp):
    main, window = _window(tmp_path, monkeypatch)
    source = tmp_path / "media" / "clip.mp4"
    source.parent.mkdir()
    source.touch()
    calls = []

    monkeypatch.setattr(main, "select_media_paths", lambda parent, start: calls.append((parent, start)) or [source])
    window.open_button.click()
    assert window.open_button.menu() is None
    assert len(calls) == 1
    assert calls[0][1] == Path.home()
    assert window.queue.count() == 1
    assert Path(window.settings.get("last_input_directory")) == source.parent
    window.close()


def test_native_dialog_error_shows_readable_fallback_and_keeps_open_flow(tmp_path, monkeypatch, qapp):
    from calibfreehands.app.native_media_dialog import NativeMediaDialogError

    main, window = _window(tmp_path, monkeypatch)
    source = tmp_path / "media" / "fallback.mp4"
    source.parent.mkdir()
    source.touch()
    warnings = []

    class Fallback:
        def __init__(self, *_args): pass
        def exec(self): return 1
        def selected_paths(self): return [source]

    def unavailable(*_args):
        raise NativeMediaDialogError("COM unavailable")

    monkeypatch.setattr(main, "select_media_paths", unavailable)
    monkeypatch.setattr(main.QMessageBox, "warning", lambda *args: warnings.append(args[-1]))
    monkeypatch.setattr(main, "MediaOpenDialog", Fallback)
    window._open_media()

    assert warnings and "fallback media picker" in warnings[0]
    assert window.queue.count() == 1
    assert Path(window.settings.get("last_input_directory")) == source.parent
    window.close()


def test_cancelled_native_picker_does_not_mutate_queue_or_last_directory(tmp_path, monkeypatch, qapp):
    main, window = _window(tmp_path, monkeypatch)
    initial = window.settings.get("last_input_directory")
    monkeypatch.setattr(main, "select_media_paths", lambda *_: [])

    window.open_button.click()

    assert window.queue.count() == 0
    assert window.settings.get("last_input_directory") == initial
    window.close()


def test_picker_can_select_multiple_files_or_current_folder_nonrecursively(tmp_path, qapp):
    from calibfreehands.app.media_picker import MediaOpenDialog

    folder = tmp_path / "media"
    (folder / "nested").mkdir(parents=True)
    first, second = folder / "a.mp4", folder / "b.png"
    first.touch(); second.touch(); (folder / "nested" / "c.mp4").touch()
    dialog = MediaOpenDialog(None, folder)
    qapp.processEvents()
    model = dialog.model
    selection = dialog.tree.selectionModel()
    selection.select(model.index(str(first)), selection.SelectionFlag.Select | selection.SelectionFlag.Rows)
    selection.select(model.index(str(second)), selection.SelectionFlag.Select | selection.SelectionFlag.Rows)
    dialog.add_files_button.click()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.selected_paths() == [first, second]

    folder_dialog = MediaOpenDialog(None, folder)
    folder_dialog.add_folder_button.click()
    assert folder_dialog.selected_paths() == [folder]
    folder_dialog.close()


def test_result_bundle_output_and_multiselect_result_picker(tmp_path, monkeypatch, qapp):
    main, window = _window(tmp_path, monkeypatch)
    first, second = tmp_path / "a.cfh", tmp_path / "b.cfh"
    first.touch(); second.touch()
    calls = []

    class Viewer:
        def __init__(self, paths):
            calls.append(paths)
        def setAttribute(self, *_): pass
        def show(self): pass

    monkeypatch.setattr(main.QFileDialog, "getOpenFileNames", lambda *args: (calls.append(args) or ([str(first), str(second)], "")))
    monkeypatch.setattr(main, "ComparisonViewer", Viewer)
    monkeypatch.setattr(main, "load_result_bundle", lambda path: type("Bundle", (), {"metadata": {"source_media_sha256": "a" * 64}})())
    window._open_result()

    dialog_args = calls[0]
    assert dialog_args[1] == "View Results..."
    assert dialog_args[2] == str(Path(window.output_path.text()))
    assert "CalibFreeHands Result (*.cfh)" in dialog_args[3]
    assert len(calls[1]) == 2
    window.close()


def test_view_results_rejects_different_source_hashes_without_opening_viewer(tmp_path, monkeypatch, qapp):
    main, window = _window(tmp_path, monkeypatch)
    paths = [tmp_path / "same-name-a.cfh", tmp_path / "same-name-b.cfh"]
    opened, warnings = [], []
    monkeypatch.setattr(main.QFileDialog, "getOpenFileNames", lambda *_: ([str(path) for path in paths], ""))
    monkeypatch.setattr(main, "load_result_bundle", lambda path: type("Bundle", (), {"metadata": {
        "source_media_sha256": "a" * 64 if Path(path).name.endswith("a.cfh") else "b" * 64,
    }})())
    monkeypatch.setattr(main, "ComparisonViewer", lambda *_: opened.append("comparison"))
    monkeypatch.setattr(main.QMessageBox, "warning", lambda *args: warnings.append(args[-1]))
    window._open_result()
    assert opened == []
    assert warnings == ["Selected results were generated from different source media."]
    window.close()


def test_view_results_falls_back_when_saved_result_directory_was_removed(tmp_path, monkeypatch, qapp):
    main, window = _window(tmp_path, monkeypatch)
    window.settings.set("last_result_directory", str(tmp_path / "deleted"))
    window.settings.sync()
    starts = []
    monkeypatch.setattr(main.QFileDialog, "getOpenFileNames", lambda *args: (starts.append(args[2]) or ([], "")))
    window._open_result()
    assert starts == [window.output_path.text()]
    window.close()


def test_configuration_layout_and_audio_preview_controls_match_bundle_workflow(tmp_path, monkeypatch, qapp):
    _, window = _window(tmp_path, monkeypatch)
    labels = {label.text() for label in window.findChildren(__import__("PySide6.QtWidgets", fromlist=["QLabel"]).QLabel)}
    assert {"Name", "Calibration", "Reconstruction", "Reference Focal", "Action", "Theme", "Language", "Output"} <= labels
    row = window.configuration_rows[0]
    readable_name_width = row.name.fontMetrics().horizontalAdvance("Configuration 1")
    assert row.name.minimumWidth() >= readable_name_width
    for combo in (row.calibration, row.reconstruction):
        assert combo.sizeAdjustPolicy() == combo.SizeAdjustPolicy.AdjustToContents
    assert not any(box.text() == "Preserve audio" for box in window.findChildren(__import__("PySide6.QtWidgets", fromlist=["QCheckBox"]).QCheckBox))
    assert window.minimumWidth() >= 900
    window.close()


def test_added_configuration_uses_first_free_default_number_and_keeps_custom_names(tmp_path, monkeypatch, qapp):
    _, window = _window(tmp_path, monkeypatch)
    window.configuration_rows[0].name.setText("Reference")
    window._add_configuration_row()
    assert window.configuration_rows[-1].name.text() == "Configuration 1"
    window._add_configuration_row()
    assert window.configuration_rows[-1].name.text() == "Configuration 2"
    window.configuration_rows[2].name.setText("Proposed")
    window._remove_configuration_row(window.configuration_rows[1])
    window._add_configuration_row()
    assert window.configuration_rows[-1].name.text() == "Configuration 1"
    assert window.configuration_rows[0].name.text() == "Reference"
    assert window.configuration_rows[1].name.text() == "Proposed"
    assert [window.rows_layout.getItemPosition(window.rows_layout.indexOf(row.name))[0]
            for row in window.configuration_rows] == [1, 2, 3]
    window.close()

