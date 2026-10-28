from __future__ import annotations

import importlib
import json

import pytest
from PySide6.QtCore import QStandardPaths
from PySide6.QtWidgets import QComboBox, QGridLayout, QSizePolicy

from calibfreehands.app.settings import AppSettings


def make_window(tmp_path, monkeypatch):
    main = importlib.import_module("calibfreehands.app.main")
    settings_file = tmp_path / "settings.ini"
    monkeypatch.setattr(main, "AppSettings", lambda: AppSettings(settings_file))
    monkeypatch.setattr(
        main.QStandardPaths,
        "writableLocation",
        lambda location: str(tmp_path / ("app-config" if location == QStandardPaths.StandardLocation.AppConfigLocation else "Documents")),
    )
    return main, main.MainWindow(), settings_file


def save_preset(monkeypatch, window, name="Calibration Comparison"):
    main = importlib.import_module("calibfreehands.app.main")
    monkeypatch.setattr(main.QInputDialog, "getText", lambda *_args, **_kwargs: (name, True))
    window._save_user_preset_dialog()
    return window.user_presets[-1]


def configure_comparison(window):
    window._add_configuration_row()
    row = window.configuration_rows[-1]
    row.name.setText("Reference")
    row.calibration.setCurrentIndex(row.calibration.findData("checkerboard_reference"))
    row.reference_focal.setValue(1280.25)
    window.sample_count_combo.setCurrentText("16")
    window.aggregation_combo.setCurrentIndex(window.aggregation_combo.findData("mean"))


def test_user_preset_save_restart_restore_and_excludes_ui_settings(tmp_path, monkeypatch, qapp):
    main, window, _ = make_window(tmp_path, monkeypatch)
    configure_comparison(window)
    window._set_theme("Dark")
    custom_output = tmp_path / "personal-output"
    window.output_path.setText(str(custom_output))
    window.settings.set("output_directory", str(custom_output))
    window.settings.sync()

    saved = save_preset(monkeypatch, window)
    stored = json.loads(window.preset_store_path.read_text(encoding="utf-8"))["presets"][0]
    assert window.preset_combo.currentData() == saved.id
    assert stored["configurations"][1]["reference_focal_px"] == 1280.25
    assert {"theme", "language", "output_directory", "viewer_colors"}.isdisjoint(stored)
    window.close()

    restored = main.MainWindow()
    restored._select_data(restored.preset_combo, saved.id)
    restored._apply_selected_preset(0)
    assert [row.name.text() for row in restored.configuration_rows] == ["Original", "Reference"]
    assert restored.configuration_rows[1].reference_focal.value() == 1280.25
    assert restored.sample_count_combo.currentText() == "16"
    assert restored.aggregation_combo.currentData() == "mean"
    assert restored.settings.get("theme") == "dark"
    assert restored.output_path.text() == str(custom_output)
    restored.close()


def test_user_preset_default_is_separate_from_save_and_survives_rename(tmp_path, monkeypatch, qapp):
    main, window, _ = make_window(tmp_path, monkeypatch)
    configure_comparison(window)
    preset = save_preset(monkeypatch, window)

    assert window.settings.get("default_preset_id") == "original"
    window.set_default_button.click()
    assert window.settings.get("default_preset_id") == preset.id
    assert window.settings.get("default_configurations_json") == ""
    window._rename_user_preset(preset.id, "Renamed Comparison")
    window.close()

    restored = main.MainWindow()
    assert restored.preset_combo.currentData() == preset.id
    assert restored.preset_combo.currentText() == "Renamed Comparison (User)"
    assert len(restored.configuration_rows) == 2
    restored.close()


def test_deleting_default_user_preset_falls_back_to_original(tmp_path, monkeypatch, qapp):
    main, window, _ = make_window(tmp_path, monkeypatch)
    preset = save_preset(monkeypatch, window)
    window.set_default_button.click()

    window._delete_user_preset(preset.id)

    assert window.settings.get("default_preset_id") == "original"
    assert window.preset_combo.currentData() == "original"
    window.close()
    restarted = main.MainWindow()
    assert restarted.preset_combo.currentData() == "original"
    assert len(restarted.configuration_rows) == 1
    restarted.close()


def test_import_export_user_preset_and_export_disabled_for_custom(tmp_path, monkeypatch, qapp):
    _, window, _ = make_window(tmp_path, monkeypatch)
    preset = save_preset(monkeypatch, window, "Exportable")
    exported = tmp_path / "portable.cfhpreset"

    window._export_user_preset(exported)
    window._delete_user_preset(preset.id)
    window.configuration_rows[0].name.setText("Edited")
    window._model_or_processing_changed()
    assert window.preset_combo.currentData() == "custom"
    assert not window.export_preset_action.isEnabled()
    imported = window._import_user_preset(exported)

    assert imported.id == preset.id
    assert window.preset_combo.currentData() == preset.id
    assert [item.display_name for item in window.user_presets] == ["Exportable"]
    window.close()


def test_built_in_preset_is_immutable_and_controls_are_compact(tmp_path, monkeypatch, qapp):
    _, window, _ = make_window(tmp_path, monkeypatch)
    assert window.preset_combo.currentData() == "original"
    with pytest.raises(ValueError, match="cannot be renamed"):
        window._rename_user_preset("original", "Changed")
    with pytest.raises(ValueError, match="cannot be deleted"):
        window._delete_user_preset("original")

    assert isinstance(window.rows_layout, QGridLayout)
    assert window.theme_group.layout().count() == 2
    assert window.language_group.layout().count() == 2
    assert window.settings_row.stretch(3) == 1
    assert all(isinstance(combo, QComboBox) for combo in (window.theme_combo, window.language_combo))
    assert len(window.processing_groups) == 4
    assert all(group.layout().count() == 2 for group in window.processing_groups)
    assert window.output_path.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding
    assert window.output_browse_button.sizePolicy().horizontalPolicy() != QSizePolicy.Policy.Expanding
    assert window.configuration_rows[0].reference_focal.decimals() == 2
    window._add_configuration_row()
    for row in window.configuration_rows:
        assert [window.rows_layout.getItemPosition(window.rows_layout.indexOf(widget))[1]
                for widget in row.widgets] == [0, 1, 2, 3, 4]
    window.close()


def test_custom_pseudo_item_remains_available_with_any_number_of_user_presets(tmp_path, monkeypatch, qapp):
    _, window, _ = make_window(tmp_path, monkeypatch)
    assert window.preset_combo.findData("custom") >= 0
    first = save_preset(monkeypatch, window, "First")
    assert window.preset_combo.findData("custom") >= 0
    second = save_preset(monkeypatch, window, "Second")

    before = [row.configuration() for row in window.configuration_rows]
    active_preset_id = window.preset_combo.currentData()
    window._select_data(window.preset_combo, "custom")

    assert window.preset_combo.findData("custom") >= 0
    assert window.preset_combo.currentData() == active_preset_id
    assert [row.configuration() for row in window.configuration_rows] == before
    assert {first.id, second.id}.issubset({window.preset_combo.itemData(i) for i in range(window.preset_combo.count())})
    window.close()


def test_custom_state_remembers_user_preset_origin_and_prefills_save_name(tmp_path, monkeypatch, qapp):
    main, window, _ = make_window(tmp_path, monkeypatch)
    preset = save_preset(monkeypatch, window, "KJH 1")
    window.configuration_rows[0].calibration.setCurrentIndex(
        window.configuration_rows[0].calibration.findData("checkerboard_reference")
    )
    window.configuration_rows[0].reference_focal.setValue(1200)

    assert window.preset_combo.currentData() == "custom"
    assert window._preset_origin_id == preset.id
    calls = []
    monkeypatch.setattr(main.QInputDialog, "getText", lambda *args, **kwargs: (calls.append(kwargs.get("text", "")) or "", False))
    window._save_user_preset_dialog()

    assert calls == ["KJH 1"]
    window.close()


def test_overwrite_keeps_user_preset_id_and_updates_contents(tmp_path, monkeypatch, qapp):
    main, window, _ = make_window(tmp_path, monkeypatch)
    preset = save_preset(monkeypatch, window, "KJH 1")
    window.set_default_button.click()
    window.configuration_rows[0].calibration.setCurrentIndex(
        window.configuration_rows[0].calibration.findData("checkerboard_reference")
    )
    window.configuration_rows[0].reference_focal.setValue(1200)

    updated = window._save_user_preset("KJH 1", overwrite_id=preset.id)

    assert updated.id == preset.id
    assert len([item for item in window.user_presets if item.display_name == "KJH 1"]) == 1
    assert updated.configurations[0].reference_focal_px == 1200
    assert window.preset_combo.currentData() == preset.id
    window.close()
    restarted = main.MainWindow()
    restored = next(item for item in restarted.user_presets if item.id == preset.id)
    assert restored.configurations[0].reference_focal_px == 1200
    assert restarted.preset_combo.currentData() == preset.id
    restarted.close()


def test_save_as_creates_new_id_and_leaves_origin_unchanged(tmp_path, monkeypatch, qapp):
    _, window, _ = make_window(tmp_path, monkeypatch)
    original = save_preset(monkeypatch, window, "KJH 1")
    window.configuration_rows[0].calibration.setCurrentIndex(
        window.configuration_rows[0].calibration.findData("checkerboard_reference")
    )
    window.configuration_rows[0].reference_focal.setValue(1200)

    copied = window._save_user_preset("KJH 2")

    assert copied.id != original.id
    assert original.configurations[0].reference_focal_px is None
    assert copied.configurations[0].reference_focal_px == 1200
    assert window.preset_combo.currentData() == copied.id
    window.close()


def test_save_dialog_requires_explicit_overwrite_choice(tmp_path, monkeypatch, qapp):
    main, window, _ = make_window(tmp_path, monkeypatch)
    original = save_preset(monkeypatch, window, "KJH 1")
    window.configuration_rows[0].calibration.setCurrentIndex(
        window.configuration_rows[0].calibration.findData("checkerboard_reference")
    )
    window.configuration_rows[0].reference_focal.setValue(1200)
    monkeypatch.setattr(main.QInputDialog, "getText", lambda *_args, **_kwargs: ("KJH 1", True))
    choices = []
    monkeypatch.setattr(window, "_ask_duplicate_preset_action", lambda name: choices.append(name) or "overwrite")

    window._save_user_preset_dialog()

    assert choices == ["KJH 1"]
    assert len(window.user_presets) == 1
    assert window.user_presets[0].id == original.id
    assert window.user_presets[0].configurations[0].reference_focal_px == 1200
    window.close()


def test_save_as_dialog_leaves_duplicate_origin_unchanged(tmp_path, monkeypatch, qapp):
    main, window, _ = make_window(tmp_path, monkeypatch)
    original = save_preset(monkeypatch, window, "KJH 1")
    window.configuration_rows[0].calibration.setCurrentIndex(
        window.configuration_rows[0].calibration.findData("checkerboard_reference")
    )
    window.configuration_rows[0].reference_focal.setValue(1200)
    names = iter([("KJH 1", True), ("KJH 2", True)])
    monkeypatch.setattr(main.QInputDialog, "getText", lambda *_args, **_kwargs: next(names))
    monkeypatch.setattr(window, "_ask_duplicate_preset_action", lambda _name: "save_as")

    window._save_user_preset_dialog()

    assert len(window.user_presets) == 2
    assert window.user_presets[0].id == original.id
    assert window.user_presets[0].configurations[0].reference_focal_px is None
    assert window.user_presets[1].display_name == "KJH 2"
    assert window.user_presets[1].id != original.id
    assert window.preset_combo.currentData() == window.user_presets[1].id
    window.close()


def test_builtin_preset_name_cannot_be_saved_over(tmp_path, monkeypatch, qapp):
    _, window, _ = make_window(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="reserved by a built-in"):
        window._save_user_preset("Original")
    assert len(window.user_presets) == 0
    window.close()


def test_custom_origin_can_be_builtin_and_save_creates_a_user_preset(tmp_path, monkeypatch, qapp):
    main, window, _ = make_window(tmp_path, monkeypatch)
    window.configuration_rows[0].name.setText("Modified Original")
    window._model_or_processing_changed()
    assert window.preset_combo.currentData() == "custom"
    assert window._preset_origin_id == "original"
    monkeypatch.setattr(main.QInputDialog, "getText", lambda *_args, **_kwargs: ("Original Copy", True))

    window._save_user_preset_dialog()

    assert window.configuration.presets["original"].display_name == "Original"
    assert [preset.display_name for preset in window.user_presets] == ["Original Copy"]
    assert window.preset_combo.currentData() == window.user_presets[0].id
    window.close()


def test_duplicate_non_origin_preset_requires_choice_and_preserves_other_preset(tmp_path, monkeypatch, qapp):
    main, window, _ = make_window(tmp_path, monkeypatch)
    non_origin = save_preset(monkeypatch, window, "KJH 1")
    origin = window._save_user_preset("KJH 2")
    window.configuration_rows[0].calibration.setCurrentIndex(
        window.configuration_rows[0].calibration.findData("checkerboard_reference")
    )
    window.configuration_rows[0].reference_focal.setValue(1200)
    monkeypatch.setattr(main.QInputDialog, "getText", lambda *_args, **_kwargs: ("KJH 1", True))
    choices = []
    monkeypatch.setattr(window, "_ask_duplicate_preset_action", lambda name: choices.append(name) or "overwrite")

    window._save_user_preset_dialog()

    assert choices == ["KJH 1"]
    assert window.user_presets[0].id == non_origin.id
    assert window.user_presets[0].configurations[0].reference_focal_px == 1200
    assert window.user_presets[1] == origin
    window.close()


def test_restore_exact_user_preset_match_clears_custom_and_updates_origin(tmp_path, monkeypatch, qapp):
    _, window, _ = make_window(tmp_path, monkeypatch)
    configure_comparison(window)
    preset = save_preset(monkeypatch, window, "KJH 1")
    reference = window.configuration_rows[1]
    reference.reference_focal.setValue(1300)
    reference.reference_focal.setValue(1280.25)
    window._model_or_processing_changed()

    assert window.preset_combo.currentData() == preset.id
    assert window._preset_origin_id == preset.id
    assert window.preset_combo.findData("custom") >= 0
    window.close()


def test_corrupt_user_store_is_reported_and_not_overwritten(tmp_path, monkeypatch, qapp):
    main = importlib.import_module("calibfreehands.app.main")
    store = tmp_path / "app-config" / "user_presets.json"
    store.parent.mkdir()
    store.write_text("not json", encoding="utf-8")
    monkeypatch.setattr(main, "AppSettings", lambda: AppSettings(tmp_path / "settings.ini"))
    monkeypatch.setattr(main.QStandardPaths, "writableLocation", lambda *_: str(tmp_path / "app-config"))

    window = main.MainWindow()

    assert "could not be loaded" in window.progress.text().lower()
    assert not window.save_preset_action.isEnabled()
    window.close()
    assert store.read_text(encoding="utf-8") == "not json"
