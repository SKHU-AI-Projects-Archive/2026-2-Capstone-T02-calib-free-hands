"""GeoCalib adapter (ECCV 2024, github.com/cvg/GeoCalib).

Output convention, established from source and confirmed numerically:

``GeoCalib.calibrate`` returns a camera object whose ``f`` is already in
**original-image pixels** — the returned ``size`` field reads ``[1280, 720]``
for a 1280x720 input, and ``f`` satisfies ``f = (H/2)/tan(vfov/2)`` with the
original height. The principal point is **not predicted**: GeoCalib fixes it at
the image centre, so ``cx, cy`` are reported as not predicted rather than as an
estimate.

The pinhole variant returns a single isotropic focal (fx == fy).
"""
from __future__ import annotations

import numpy as np

from .base import CalibrationAdapter, CalibrationPrediction


class GeoCalibAdapter(CalibrationAdapter):
    name = "GeoCalib"
    predicts_principal_point = False      # fixed at the image centre by design
    predicts_distortion = False           # pinhole variant

    def __init__(self, device: str = "cuda", weights: str = "pinhole",
                 camera_model: str = "pinhole"):
        super().__init__(device)
        self.weights = weights
        self.camera_model = camera_model
        # GeoCalib's "radial" model is 1 + k1 r^2 + k2 r^4 with r^2 = x^2 + y^2
        # in normalised coordinates - the same convention as OpenCV's radial
        # terms (CAM-EXP-003.1 audit).
        self.predicts_distortion = camera_model != "pinhole"
        self.name = (f"GeoCalib[{weights}]" if camera_model == "pinhole"
                     else f"GeoCalib[{weights}/{camera_model}]")

    def _load(self):
        import torch
        from geocalib import GeoCalib
        return GeoCalib(weights=self.weights).to(torch.device(self.device))

    def _predict(self, img_bgr, meta: dict) -> CalibrationPrediction:
        import torch
        rgb = img_bgr[:, :, ::-1].copy()
        im = torch.tensor(rgb, dtype=torch.float32,
                          device=torch.device(self.device)).permute(2, 0, 1) / 255.0
        res = self._model.calibrate(im, camera_model=self.camera_model)
        cam = res["camera"]
        f = np.asarray(cam.f.detach().cpu().numpy(), dtype=float).ravel()
        size = np.asarray(cam.size.detach().cpu().numpy(), dtype=float).ravel()
        fx = float(f[0])
        fy = float(f[1]) if f.size > 1 else fx
        vfov = None
        try:
            vfov = float(np.degrees(cam.vfov.detach().cpu().numpy().ravel()[0]))
        except Exception:
            pass
        k1 = k2 = ""
        for attr in ("k1", "k2"):
            v = getattr(cam, attr, None)
            if v is not None:
                try:
                    val = float(np.asarray(v.detach().cpu()).ravel()[0])
                except Exception:
                    continue
                if attr == "k1":
                    k1 = val
                else:
                    k2 = val
        unc = res.get("focal_uncertainty")
        return CalibrationPrediction(
            model=self.name, fx_px=fx, fy_px=fy, cx_px=None, cy_px=None,
            vfov_deg=vfov,
            distortion=",".join(str(v) for v in (k1, k2) if v != ""),
            raw_output=f"f={f.tolist()} size={size.tolist()} k1={k1} k2={k2}",
            conversion_note=("none needed: camera.f is already in original pixels "
                             "(camera.size reports the original W,H)"),
            extra={"principal_point": "NOT_PREDICTED (fixed at image centre)",
                   "camera_model": self.camera_model, "k1": k1, "k2": k2,
                   "focal_uncertainty": (float(np.asarray(unc.detach().cpu()).ravel()[0])
                                         if unc is not None else ""),
                   "reported_size": str(size.tolist())})
