from calibfreehands.app.settings import AppSettings
from PySide6.QtWidgets import QComboBox, QLabel
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
    sampling = next(combo for combo in combos if [combo.itemText(i) for i in range(combo.count())] == ["Uniform", "All frames"])
    sample_count = next(combo for combo in combos if [combo.itemText(i) for i in range(combo.count())] == ["4", "8", "16", "32"])

    assert "N" not in labels
    assert "Sample count" in labels
    assert sample_count.currentText() == str(window.settings.get("sample_count"))
    sampling.setCurrentText("All frames")
    assert not sample_count.isEnabled()
    sampling.setCurrentText("Uniform")
    assert sample_count.isEnabled()
    window.close()
