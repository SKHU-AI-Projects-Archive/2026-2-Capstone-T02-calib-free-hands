import json

import pytest

from calibfreehands.app.configuration import ConfigError, load_configuration
from calibfreehands.calibration import AnyCalibBackend
from calibfreehands.calibration.registry import create_calibration_backend
from calibfreehands.pose import OriginalWiLoRBackend
from calibfreehands.pose.registry import create_pose_backend


def test_built_in_configuration_exposes_original_default_and_unavailable_models():
    config = load_configuration()
    original = config.presets["original"]

    assert original.calibration_model_id == "anycalib_original"
    assert original.reconstruction_model_id == "wilor_original"
    assert (original.sampling, original.sample_count, original.aggregation) == ("uniform", 8, "median")
    assert config.calibration_models["anycalib_original"].available
    assert not config.calibration_models["handcalib"].available
    assert not config.reconstruction_models["anyhand_wilor"].available
    assert original.configurations == ()


def test_preset_may_optionally_define_multiple_named_configurations(tmp_path):
    data = json.loads(load_configuration().source_path.read_text(encoding="utf-8"))
    data["presets"][0]["configurations"] = [
        {"id": "reference", "name": "Reference", "calibration_model_id": "checkerboard_reference",
         "reconstruction_model_id": "wilor_original", "reference_focal_px": 1280.5},
        {"id": "baseline", "name": "Baseline", "calibration_model_id": "anycalib_original",
         "reconstruction_model_id": "wilor_original"},
    ]
    path = tmp_path / "models.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    preset = load_configuration(path).presets["original"].configurations
    assert [(row.id, row.name) for row in preset] == [("reference", "Reference"), ("baseline", "Baseline")]
    assert preset[0].reference_focal_px == 1280.5
    assert preset[1].reference_focal_px is None


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda data: data["calibration_models"].append(data["calibration_models"][0]), "(?i)duplicate"),
        (lambda data: data["calibration_models"][0].update(backend="unknown"), "backend"),
        (lambda data: data["presets"][0].update(calibration_model_id="missing"), "unknown calibration"),
        (lambda data: data["presets"][0].update(calibration_model_id="handcalib"), "not available"),
        (lambda data: data["presets"][0].update(sampling="random"), "sampling"),
        (lambda data: data["presets"][0].update(sample_count=0), "sample_count"),
        (lambda data: data["presets"][0].update(aggregation="mode"), "aggregation"),
        (lambda data: data["presets"][0].pop("display_name"), "display_name"),
        (lambda data: data["reconstruction_models"][0]["assets"].update(checkpoint="D:/weights/model.ckpt"), "relative"),
    ],
)
def test_invalid_configuration_is_rejected(tmp_path, mutate, message):
    data = json.loads(load_configuration().source_path.read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "models.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_configuration(path)


def test_registries_construct_original_backends_without_loading_assets(tmp_path):
    config = load_configuration()

    calibration = create_calibration_backend("anycalib_original", config)
    reconstruction = create_pose_backend("wilor_original", config, tmp_path)

    assert isinstance(calibration, AnyCalibBackend)
    assert calibration.model_id == "anycalib_pinhole"
    assert isinstance(reconstruction, OriginalWiLoRBackend)
    assert reconstruction.repository == (tmp_path / "WiLoR").resolve()
    assert reconstruction.checkpoint == (tmp_path / "WiLoR/pretrained_models/wilor_final.ckpt").resolve()
    assert reconstruction.config == (tmp_path / "models/model_config_wilor_original.yaml").resolve()


@pytest.mark.parametrize("model_id", ["handcalib"])
def test_unavailable_calibration_backend_fails_with_reason(model_id):
    config = load_configuration()
    with pytest.raises(RuntimeError, match="not available"):
        create_calibration_backend(model_id, config)


def test_unavailable_reconstruction_variant_fails_with_reason():
    config = load_configuration()
    with pytest.raises(RuntimeError, match="not available"):
        create_pose_backend("anyhand_wilor", config, project_root=None)
