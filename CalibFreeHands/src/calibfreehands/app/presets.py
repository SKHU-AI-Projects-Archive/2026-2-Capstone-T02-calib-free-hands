from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

from .configuration import (
    ApplicationConfiguration,
    PresetConfiguration,
    PresetDefinition,
)

FILE_SCHEMA_VERSION = 1
_PRESET_FIELDS = {"id", "name", "configurations", "sampling_mode", "sample_count", "aggregation"}
_ROW_FIELDS = {"id", "name", "calibration_model_id", "reconstruction_model_id", "reference_focal_px"}


def preset_to_dict(preset: PresetDefinition) -> dict:
    rows = []
    for row in preset.configurations:
        data = {
            "id": row.id,
            "name": row.name,
            "calibration_model_id": row.calibration_model_id,
            "reconstruction_model_id": row.reconstruction_model_id,
        }
        if row.reference_focal_px is not None:
            data["reference_focal_px"] = row.reference_focal_px
        rows.append(data)
    return {
        "id": preset.id,
        "name": preset.display_name,
        "configurations": rows,
        "sampling_mode": preset.sampling,
        "sample_count": preset.sample_count,
        "aggregation": preset.aggregation,
    }


def validate_user_preset(
    data: object,
    configuration: ApplicationConfiguration,
    existing: tuple[PresetDefinition, ...] | list[PresetDefinition] = (),
) -> PresetDefinition:
    if not isinstance(data, dict) or set(data) != _PRESET_FIELDS:
        raise ValueError("User preset has an invalid or unexpected schema")
    preset_id, name = data["id"], data["name"]
    if not isinstance(preset_id, str) or not preset_id.strip():
        raise ValueError("Preset ID must be a non-empty string")
    if preset_id in configuration.presets:
        raise ValueError(f"Preset ID is reserved by a built-in preset: {preset_id}")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Preset name must not be empty")
    if any(preset.display_name.casefold() == name.strip().casefold()
           for preset in configuration.presets.values()):
        raise ValueError(f"Preset name is reserved by a built-in preset: {name.strip()}")
    if any(preset.id == preset_id for preset in existing):
        raise ValueError(f"Duplicate user preset ID: {preset_id}")
    if any(preset.display_name.casefold() == name.strip().casefold() for preset in existing):
        raise ValueError(f"Duplicate user preset name: {name.strip()}")

    sampling = data["sampling_mode"]
    sample_count = data["sample_count"]
    aggregation = data["aggregation"]
    if not isinstance(sampling, str) or sampling not in {"uniform", "all"}:
        raise ValueError("Preset sampling mode must be uniform or all")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
        raise ValueError("Preset sample_count must be a positive integer")
    if not isinstance(aggregation, str) or aggregation not in {"median", "mean"}:
        raise ValueError("Preset aggregation must be median or mean")
    rows_data = data["configurations"]
    if not isinstance(rows_data, list) or not rows_data:
        raise ValueError("Preset must contain at least one configuration")

    rows = []
    row_ids = set()
    for row in rows_data:
        if not isinstance(row, dict) or set(row) - _ROW_FIELDS or not _ROW_FIELDS - {"reference_focal_px"} <= set(row):
            raise ValueError("Preset configuration row has invalid or missing fields")
        row_id, row_name = row["id"], row["name"]
        calibration_id, reconstruction_id = row["calibration_model_id"], row["reconstruction_model_id"]
        if not isinstance(row_id, str) or not row_id.strip() or row_id in row_ids:
            raise ValueError("Preset configuration IDs must be non-empty and unique")
        if not isinstance(row_name, str) or not row_name.strip():
            raise ValueError("Preset configuration name must not be empty")
        if not isinstance(calibration_id, str) or calibration_id not in configuration.calibration_models:
            raise ValueError(f"Preset requires unknown calibration model: {calibration_id}")
        if not isinstance(reconstruction_id, str) or reconstruction_id not in configuration.reconstruction_models:
            raise ValueError(f"Preset requires unknown reconstruction model: {reconstruction_id}")
        calibration = configuration.calibration_models[calibration_id]
        reconstruction = configuration.reconstruction_models[reconstruction_id]
        if not calibration.available:
            raise ValueError(f"Preset requires unavailable calibration model: {calibration_id}")
        if not reconstruction.available:
            raise ValueError(f"Preset requires unavailable reconstruction model: {reconstruction_id}")
        focal = row.get("reference_focal_px")
        if calibration.backend == "reference_focal":
            if isinstance(focal, bool) or not isinstance(focal, (float, int)) or not math.isfinite(focal) or not 0 < focal <= 100000:
                raise ValueError("Preset reference focal must be finite and positive")
            focal = float(focal)
        elif focal is not None:
            raise ValueError("Preset reference focal is only valid for Checkerboard Reference")
        rows.append(PresetConfiguration(row_id, row_name.strip(), calibration_id, reconstruction_id, focal))
        row_ids.add(row_id)

    first = rows[0]
    return PresetDefinition(
        preset_id, name.strip(), first.calibration_model_id, first.reconstruction_model_id,
        sampling, sample_count, aggregation, tuple(rows),
    )


def _validate_envelope(data: object) -> list:
    if not isinstance(data, dict) or set(data) != {"file_schema_version", "presets"}:
        raise ValueError("User preset store has an invalid schema")
    if isinstance(data["file_schema_version"], bool) or data["file_schema_version"] != FILE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported user preset schema: {data['file_schema_version']}")
    if not isinstance(data["presets"], list):
        raise ValueError("User preset store presets must be a list")
    return data["presets"]


def load_user_presets(path: Path, configuration: ApplicationConfiguration) -> list[PresetDefinition]:
    path = Path(path)
    if not path.exists():
        return []
    try:
        presets_data = _validate_envelope(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read user presets: {exc}") from exc
    presets = []
    for data in presets_data:
        presets.append(validate_user_preset(data, configuration, presets))
    return presets


def save_user_presets(path: Path, presets: list[PresetDefinition]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"file_schema_version": FILE_SCHEMA_VERSION, "presets": [preset_to_dict(preset) for preset in presets]}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_user_preset(path: Path, preset: PresetDefinition) -> Path:
    path = Path(path)
    payload = {"file_schema_version": FILE_SCHEMA_VERSION, "preset": preset_to_dict(preset)}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return path


def import_user_preset(
    path: Path,
    configuration: ApplicationConfiguration,
    existing: tuple[PresetDefinition, ...] | list[PresetDefinition] = (),
) -> PresetDefinition:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read preset file: {exc}") from exc
    if not isinstance(data, dict) or set(data) != {"file_schema_version", "preset"}:
        raise ValueError("Preset file has an invalid schema")
    if isinstance(data["file_schema_version"], bool) or data["file_schema_version"] != FILE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported preset file schema: {data['file_schema_version']}")
    return validate_user_preset(data["preset"], configuration, existing)
