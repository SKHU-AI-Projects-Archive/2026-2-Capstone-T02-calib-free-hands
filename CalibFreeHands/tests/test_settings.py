from calibfreehands.app.settings import AppSettings
from PySide6.QtWidgets import QCheckBox, QComboBox, QLabel
import importlib


def test_settings_persist(tmp_path):
    settings = AppSettings(tmp_path / "settings.ini")
    settings.set("language", "ko")
    settings.set("sample_count", 12)
    settings.sync()
    restored = AppSettings(tmp_path / "settings.ini")
    assert restored.get("language") == "ko"
    assert restored.get("sample_count") == 12


def test_main_window_groups_sample_count_and_preserves_sampling_toggle(tmp_path, monkeypatch, qapp):
    main = importlib.import_module("calibfreehands.app.main")

    monkeypatch.setattr(main, "AppSettings", lambda: AppSettings(tmp_path / "settings.ini"))
    MainWindow = main.MainWindow

    window = MainWindow()
    labels = [label.text() for label in window.findChildren(QLabel)]
    combos = window.findChildren(QComboBox)
    processing_row = window.centralWidget().layout().itemAt(4).layout()
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
