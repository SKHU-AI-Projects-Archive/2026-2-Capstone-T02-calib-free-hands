from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings


class AppSettings:
    defaults = {"theme": "system", "language": "en", "output_directory": "", "collision_policy": "ask",
                "preserve_audio": True, "sampling_mode": "uniform", "sample_count": 8, "aggregation": "median",
                "default_preset_id": "original", "default_calibration_model_id": "anycalib_original",
                "default_reconstruction_model_id": "wilor_original"}

    def __init__(self, path: Path | None = None) -> None:
        self._settings = QSettings(str(path), QSettings.IniFormat) if path else QSettings("CalibFreeHands", "CalibFreeHands")

    def get(self, key: str):
        return self._settings.value(key, self.defaults[key], type=type(self.defaults[key]))

    def set(self, key: str, value) -> None:
        if key not in self.defaults:
            raise KeyError(key)
        self._settings.setValue(key, value)

    def sync(self) -> None:
        self._settings.sync()

    def values(self) -> dict:
        return {key: self.get(key) for key in self.defaults}
