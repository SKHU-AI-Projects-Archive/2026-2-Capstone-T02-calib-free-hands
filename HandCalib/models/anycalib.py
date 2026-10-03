"""공식 AnyCalib pretrained model을 HandCalib 평가 코드에 연결한다."""

from pathlib import Path

import torch
from anycalib import AnyCalib


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
