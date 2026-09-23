"""AnyCalib adapter (ICCV 2025, github.com/javrtg/AnyCalib).

Output convention, established from source and confirmed numerically:

``AnyCalib.predict`` resizes internally to ``pred_size`` but then calls
``cam.reverse_scale_and_shift(intrins, scale_xy, shift_xy)`` before returning,
so the ``intrinsics`` vector is **already in original-image pixels**. The
README's remark that focals are relative to the resized input does not match
the shipped code; the check that settles it is that the returned principal
point lands at the image centre of the ORIGINAL image (638.3, 360.2 for a
1280x720 frame), which could not happen in 420x238 network pixels.

For ``cam_id='pinhole'`` the 4-vector is ``[fx, fy, cx, cy]``.
"""
from __future__ import annotations

import numpy as np

from .base import CalibrationAdapter, CalibrationPrediction

MODEL_ID = "anycalib_pinhole"
CAM_ID = "pinhole"

# Distortion-aware use (CAM-EXP-003.1). AnyCalib's "radial" model is
#     x = fx * (X/Z) * (1 + k1 r^2 + k2 r^4 + ...) + cx
# i.e. the same normalised radial polynomial as OpenCV's Brown-Conrady; it has
# no tangential terms. num_k defaults to 2, so "radial" == "radial:2".
DIST_MODEL_ID = "anycalib_dist"
DIST_CAM_ID = "radial:2"


class AnyCalibAdapter(CalibrationAdapter):
    name = "AnyCalib"
    predicts_principal_point = True
    predicts_distortion = False           # set True by the radial variants

    def __init__(self, device: str = "cuda", model_id: str = MODEL_ID,
                 cam_id: str = CAM_ID):
        super().__init__(device)
        self.model_id = model_id
        self.cam_id = cam_id
        self.predicts_distortion = not cam_id.startswith("pinhole")
        self.name = f"AnyCalib[{model_id}/{cam_id}]"

    def _load(self):
        import torch
        from anycalib import AnyCalib
        return AnyCalib(model_id=self.model_id).to(torch.device(self.device))

    def _predict(self, img_bgr, meta: dict) -> CalibrationPrediction:
        import torch
        rgb = img_bgr[:, :, ::-1].copy()
        im = torch.tensor(rgb, dtype=torch.float32,
                          device=torch.device(self.device)).permute(2, 0, 1) / 255.0
        out = self._model.predict(im, cam_id=self.cam_id)
        intr = np.asarray(out["intrinsics"].detach().cpu().numpy(), dtype=float).ravel()
        pred_size = out.get("pred_size")
        ok = bool(out.get("success", True))
        if intr.size < 4:
            return CalibrationPrediction(model=self.name, success=False,
                                         raw_output=str(intr.tolist()),
                                         failure_reason="unexpected intrinsics length")
        fx, fy, cx, cy = (float(v) for v in intr[:4])
        dist = intr[4:] if intr.size > 4 else np.zeros(0)
        return CalibrationPrediction(
            model=self.name, fx_px=fx, fy_px=fy, cx_px=cx, cy_px=cy,
            distortion=",".join(f"{v:.8g}" for v in dist),
            raw_output=f"intrinsics={intr[:4].tolist()} pred_size={tuple(pred_size)}",
            conversion_note=("none needed: predict() already applies "
                             "reverse_scale_and_shift back to original pixels"),
            failure_reason="" if ok else "model reported success=False",
            extra={"pred_size": str(tuple(pred_size)), "model_success_flag": ok,
                   "cam_id": self.cam_id,
                   "k1": float(dist[0]) if dist.size > 0 else "",
                   "k2": float(dist[1]) if dist.size > 1 else ""})
