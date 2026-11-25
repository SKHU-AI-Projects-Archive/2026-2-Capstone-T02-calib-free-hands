from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings


class AppSettings:
    # theme: dark is the design of the phone app and the capture page; Light / System stay available.
    defaults = {"theme": "dark", "language": "en", "output_directory": "", "last_input_directory": "", "last_result_directory": "", "collision_policy": "ask",
                "preserve_audio": True, "sampling_mode": "all", "sample_count": 8, "aggregation": "median",
                "default_preset_id": "original", "default_calibration_model_id": "anycalib_original",
                "default_reconstruction_model_id": "wilor_original", "default_configurations_json": "",
                "handcalib_python": "",  # empty = <repository>/.venv_bench (HandCalib's environment)
                "focal_source": "handcalib", "manual_focal_px": 1000.0, "capture_session": "session",
                # capture_source: "phone" (QR, SRT) or "camera" (the PC's webcam); pc_camera: its name
                "capture_source": "phone", "pc_camera": ""}

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
