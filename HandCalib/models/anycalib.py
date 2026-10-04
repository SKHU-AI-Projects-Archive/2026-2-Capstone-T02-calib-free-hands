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

    def build(self):
        if self.checkpoint is not None:
            model = AnyCalib(model_id=None)
            model.load_weights_from_ckpt(str(self.checkpoint))
        else:
            model = AnyCalib(model_id=self.model_id)
        self.model = model.to(self.device).eval()
        return self

    @torch.inference_mode()
    def predict(self, image):
        if self.model is None:
            raise RuntimeError("Call build() before predict().")
        return self.model.predict(image.to(self.device), cam_id=self.cam_id)
