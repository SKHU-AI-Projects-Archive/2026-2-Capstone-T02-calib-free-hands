from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath


class ConfigError(ValueError):
    """Invalid application model or preset configuration."""


@dataclass(frozen=True)
class ModelDefinition:
    id: str
    display_name: str
    backend: str
    available: bool
    unavailable_reason: str | None = None
    model_id: str | None = None
    assets: dict[str, str] | None = None


@dataclass(frozen=True)
class PresetDefinition:
    id: str
    display_name: str
    calibration_model_id: str
    reconstruction_model_id: str
    sampling: str
    sample_count: int
    aggregation: str


@dataclass(frozen=True)
class ApplicationConfiguration:
    source_path: Path
    calibration_models: dict[str, ModelDefinition]
    reconstruction_models: dict[str, ModelDefinition]
    presets: dict[str, PresetDefinition]


def _required(entry: dict, fields: set[str], label: str) -> None:
    missing = fields - entry.keys()
    if missing:
        raise ConfigError(f"{label} is missing required keys: {', '.join(sorted(missing))}")


def _models(entries: object, label: str, supported_backends: set[str]) -> dict[str, ModelDefinition]:
    if not isinstance(entries, list):
        raise ConfigError(f"{label} must be a list")
    models = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ConfigError(f"Each {label} entry must be an object")
        _required(entry, {"id", "display_name", "backend", "available"}, label)
        model_id = entry["id"]
        if not isinstance(model_id, str) or not model_id:
            raise ConfigError(f"{label} id must be a non-empty string")
        if not isinstance(entry["display_name"], str) or not entry["display_name"]:
            raise ConfigError(f"{label} {model_id} display_name must be a non-empty string")
        if not isinstance(entry["backend"], str):
            raise ConfigError(f"{label} {model_id} backend must be a string")
        if model_id in models:
            raise ConfigError(f"Duplicate {label} id: {model_id}")
        if entry["backend"] not in supported_backends:
            raise ConfigError(f"Unknown {label} backend: {entry['backend']}")
        if not isinstance(entry["available"], bool):
            raise ConfigError(f"{label} {model_id} available must be boolean")
        reason = entry.get("unavailable_reason")
        if reason is not None and (not isinstance(reason, str) or not reason):
            raise ConfigError(f"{label} {model_id} unavailable_reason must be a non-empty string")
        if not entry["available"] and not reason:
            raise ConfigError(f"{label} {model_id} requires unavailable_reason")
        if entry.get("model_id") is not None and not isinstance(entry["model_id"], str):
            raise ConfigError(f"{label} {model_id} model_id must be a string")
        assets = entry.get("assets", {})
        if not isinstance(assets, dict) or any(not isinstance(value, str) for value in assets.values()):
            raise ConfigError(f"{label} {model_id} assets must map names to relative paths")
        for value in assets.values():
            if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
                raise ConfigError(f"{label} {model_id} asset paths must be relative")
        models[model_id] = ModelDefinition(
            model_id, entry["display_name"], entry["backend"], entry["available"],
            reason, entry.get("model_id"), assets,
        )
    if not models:
        raise ConfigError(f"{label} must contain at least one model")
    return models


def load_configuration(path: Path | None = None) -> ApplicationConfiguration:
    source = Path(path) if path is not None else Path(__file__).with_name("model_config.json")
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Could not load application config {source}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("Application config root must be an object")
    _required(data, {"schema_version", "calibration_models", "reconstruction_models", "presets"}, "Application config")
    if not isinstance(data["schema_version"], int) or isinstance(data["schema_version"], bool) or data["schema_version"] != 1:
        raise ConfigError(f"Unsupported application config schema_version: {data['schema_version']}")
    calibration_models = _models(
        data["calibration_models"], "calibration_models", {"anycalib", "handcalib", "reference_focal"}
    )
    reconstruction_models = _models(
        data["reconstruction_models"], "reconstruction_models", {"wilor", "anyhand_wilor"}
    )
    if not isinstance(data["presets"], list):
        raise ConfigError("presets must be a list")
    presets = {}
    for entry in data["presets"]:
        if not isinstance(entry, dict):
            raise ConfigError("Each preset entry must be an object")
        fields = {
            "id", "display_name", "calibration_model_id", "reconstruction_model_id",
            "sampling", "sample_count", "aggregation",
        }
        _required(entry, fields, "preset")
        preset_id = entry["id"]
        if not isinstance(preset_id, str) or not preset_id:
            raise ConfigError("preset id must be a non-empty string")
        if preset_id in presets:
            raise ConfigError(f"Duplicate preset id: {preset_id}")
        if not isinstance(entry["display_name"], str) or not entry["display_name"]:
            raise ConfigError(f"Preset {preset_id} display_name must be a non-empty string")
        if not isinstance(entry["calibration_model_id"], str) or not isinstance(entry["reconstruction_model_id"], str):
            raise ConfigError(f"Preset {preset_id} model ids must be strings")
        if entry["calibration_model_id"] not in calibration_models:
            raise ConfigError(f"Preset {preset_id} has unknown calibration model")
        if entry["reconstruction_model_id"] not in reconstruction_models:
            raise ConfigError(f"Preset {preset_id} has unknown reconstruction model")
        if not calibration_models[entry["calibration_model_id"]].available or not reconstruction_models[entry["reconstruction_model_id"]].available:
            raise ConfigError(f"Preset {preset_id} selects a model that is not available")
        if not isinstance(entry["sampling"], str) or entry["sampling"] not in {"uniform", "all"}:
            raise ConfigError(f"Preset {preset_id} has invalid sampling mode")
        if not isinstance(entry["sample_count"], int) or isinstance(entry["sample_count"], bool) or entry["sample_count"] < 1:
            raise ConfigError(f"Preset {preset_id} sample_count must be at least 1")
        if not isinstance(entry["aggregation"], str) or entry["aggregation"] not in {"median", "mean"}:
            raise ConfigError(f"Preset {preset_id} has invalid aggregation")
        presets[preset_id] = PresetDefinition(**{key: entry[key] for key in fields})
    if "original" not in presets:
        raise ConfigError("Application config must define the original preset")
    return ApplicationConfiguration(source.resolve(), calibration_models, reconstruction_models, presets)
