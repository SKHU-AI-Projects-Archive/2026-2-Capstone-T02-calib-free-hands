"""공식 AnyCalib pretrained model을 HandCalib 평가 코드에 연결한다."""

from pathlib import Path
import hashlib
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "AnyCalib"))

from anycalib import AnyCalib
from omegaconf import OmegaConf

from siclib.models.networks.anycalib_net import AnyCalib as TrainingAnyCalib


EXPECTED_TRAINING_PARAMETER_COUNT = 320_259_771
EXPECTED_PRETRAINED_SHA256 = "e73b174563bcb90dc9a0e348ee64fbb898901d2a1786623e62fc0e3b176a1a38"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_training_checkpoint(model, checkpoint_path):
    """Load a HandCalib training checkpoint with the official loader semantics."""
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise RuntimeError(f"Invalid AnyCalib training checkpoint: {checkpoint_path}")
    state_dict = payload["model"]
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Checkpoint model state_dict is not dict-like: {checkpoint_path}")

    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_keys = sorted(incompatible.missing_keys)
    unexpected_keys = sorted(incompatible.unexpected_keys)
    parameter_keys = {name for name, _ in model.named_parameters()}
    loaded_parameter_keys = parameter_keys.intersection(state_dict)
    missing_parameter_keys = sorted(parameter_keys - set(state_dict))
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "AnyCalib checkpoint state_dict mismatch: "
            f"missing={missing_keys[:10]}, unexpected={unexpected_keys[:10]}"
        )

    return {
        "checkpoint_format": "handcalib_training_checkpoint",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_global_step": payload.get("global_step"),
        "state_dict_key_count": len(state_dict),
        "state_dict_missing_keys": missing_keys,
        "state_dict_unexpected_keys": unexpected_keys,
        "missing_parameter_keys": missing_parameter_keys,
        "loaded_parameter_count": len(loaded_parameter_keys),
        "model_parameter_count": len(parameter_keys),
        "loaded_parameter_coverage": len(loaded_parameter_keys) / len(parameter_keys),
    }


def build_training_anycalib(weight_path=None, device="cpu"):
    """공식 training AnyCalib를 pretrained weight와 strict하게 연결한다."""
    weight_path = Path(weight_path or Path(torch.hub.get_dir()) / "anycalib" / "anycalib_pinhole.pt")
    if not weight_path.exists():
        raise FileNotFoundError(f"AnyCalib pretrained weight does not exist: {weight_path}")
    weight_sha256 = _sha256(weight_path)
    if weight_sha256 != EXPECTED_PRETRAINED_SHA256:
        raise RuntimeError(f"Unexpected AnyCalib weight SHA256: {weight_sha256}")

    model_config_path = Path(__file__).resolve().parent.parent / "AnyCalib/siclib/configs/model/anycalib.yaml"
    model = TrainingAnyCalib(OmegaConf.load(model_config_path))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_TRAINING_PARAMETER_COUNT:
        raise RuntimeError(f"Unexpected AnyCalib parameter count: {parameter_count}")

    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)
    return model.to(torch.device(device)), {
        "parameter_count": parameter_count,
        "pretrained_weight_sha256": weight_sha256,
        "architecture_config": str(model_config_path),
    }


class AnyCalibAdapter:
    """Keep HandCalib's evaluation contract thin around the official API."""

    def __init__(self, model_id="anycalib_pinhole", cam_id="pinhole", checkpoint=None, device="cuda"):
        self.model_id = model_id
        self.cam_id = cam_id
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.device = torch.device(device)
        self.model = None
        self.load_info = {}

    def build(self):
        if self.checkpoint is not None:
            model = AnyCalib(model_id=None)
            self.load_info = _load_training_checkpoint(model, self.checkpoint)
        else:
            model = AnyCalib(model_id=self.model_id)
            self.load_info = {"checkpoint_format": "official_pretrained"}
        self.model = model.to(self.device).eval()
        return self

    @torch.inference_mode()
    def predict(self, image):
        if self.model is None:
            raise RuntimeError("Call build() before predict().")
        return self.model.predict(image.to(self.device), cam_id=self.cam_id)
