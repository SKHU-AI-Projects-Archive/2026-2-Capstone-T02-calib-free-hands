from __future__ import annotations

import json

import pytest

from calibfreehands.app.configuration import load_configuration
from calibfreehands.app.presets import (
    export_user_preset,
    import_user_preset,
    load_user_presets,
    save_user_presets,
    validate_user_preset,
)


def valid_preset():
    return {
        "id": "user-calibration-comparison",
        "name": "Calibration Comparison",
        "configurations": [
            {"id": "baseline", "name": "Original", "calibration_model_id": "anycalib_original",
             "reconstruction_model_id": "wilor_original"},
            {"id": "reference", "name": "Reference", "calibration_model_id": "checkerboard_reference",
             "reconstruction_model_id": "wilor_original", "reference_focal_px": 1200.0},
        ],
        "sampling_mode": "uniform", "sample_count": 8, "aggregation": "median",
    }


def test_user_preset_round_trip_persists_configuration_rows(tmp_path):
    configuration = load_configuration()
    preset = validate_user_preset(valid_preset(), configuration)
    path = tmp_path / "user_presets.json"

    save_user_presets(path, [preset])

    loaded = load_user_presets(path, configuration)
    assert loaded == [preset]
    assert loaded[0].configurations[1].reference_focal_px == 1200.0


def test_export_then_import_user_preset(tmp_path):
    configuration = load_configuration()
    preset = validate_user_preset(valid_preset(), configuration)
    exported = tmp_path / "comparison.cfhpreset"

    export_user_preset(exported, preset)
    imported = import_user_preset(exported, configuration)

    assert imported.display_name == preset.display_name
    assert imported.configurations == preset.configurations


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda preset: preset.update(name="  "), "name"),
        (lambda preset: preset.update(name="Original"), "reserved by a built-in"),
        (lambda preset: preset["configurations"].clear(), "at least one"),
        (lambda preset: preset["configurations"][0].update(calibration_model_id="missing"), "unknown calibration model"),
        (lambda preset: preset["configurations"][0].update(calibration_model_id="handcalib"), "unavailable calibration model"),
        (lambda preset: preset["configurations"][1].update(reference_focal_px=float("nan")), "reference focal"),
        (lambda preset: preset.update(sampling_mode="random"), "sampling"),
        (lambda preset: preset.update(sample_count=0), "sample_count"),
        (lambda preset: preset.update(aggregation="mode"), "aggregation"),
        (lambda preset: preset.update(output_directory="C:/private"), "unexpected"),
    ],
)
def test_invalid_user_preset_is_rejected(change, message):
    candidate = valid_preset()
    change(candidate)
    with pytest.raises(ValueError, match=message):
        validate_user_preset(candidate, load_configuration())


def test_duplicate_user_preset_name_and_id_rejected(tmp_path):
    configuration = load_configuration()
    first = validate_user_preset(valid_preset(), configuration)
    duplicate_name = valid_preset()
    duplicate_name["id"] = "another-id"
    payload = {"file_schema_version": 1, "presets": [valid_preset(), duplicate_name]}
    path = tmp_path / "presets.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="(?i)duplicate user preset name"):
        load_user_presets(path, configuration)

    duplicate_id = valid_preset()
    payload["presets"] = [valid_preset(), duplicate_id]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)duplicate user preset ID"):
        load_user_presets(path, configuration)


def test_import_rejects_builtin_id_and_duplicate_name(tmp_path):
    configuration = load_configuration()
    preset = valid_preset()
    preset["id"] = "original"
    path = tmp_path / "builtin.cfhpreset"
    path.write_text(json.dumps({"file_schema_version": 1, "preset": preset}), encoding="utf-8")

    with pytest.raises(ValueError, match="reserved"):
        import_user_preset(path, configuration)

    path.write_text(json.dumps({"file_schema_version": 9, "preset": valid_preset()}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        import_user_preset(path, configuration)

    path.write_text(json.dumps({"file_schema_version": True, "preset": valid_preset()}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        import_user_preset(path, configuration)


def test_user_preset_store_does_not_write_partial_file_on_replace_error(tmp_path, monkeypatch):
    from calibfreehands.app import presets

    path = tmp_path / "user_presets.json"
    path.write_text("previous", encoding="utf-8")
    monkeypatch.setattr(presets.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("locked")))

    with pytest.raises(OSError, match="locked"):
        save_user_presets(path, [validate_user_preset(valid_preset(), load_configuration())])

    assert path.read_text(encoding="utf-8") == "previous"
    assert list(tmp_path.iterdir()) == [path]
