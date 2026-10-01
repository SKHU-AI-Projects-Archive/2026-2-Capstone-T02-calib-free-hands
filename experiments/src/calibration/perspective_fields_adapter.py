"""Perspective Fields adapter (CVPR 2023, github.com/jinlinyi/PerspectiveFields).

Output convention, established from source and confirmed numerically:

``inference`` returns ``pred_rel_focal``, ``pred_general_vfov`` (degrees),
``pred_rel_cx``, ``pred_rel_cy``. In ``perspective2d/utils/utils.py`` the
official converter ``general_vfov_to_focal(rel_cx, rel_cy, h, gvfov, degree)``
solves for ``focal/h``, and its docstring states the result "is relative to the
image height if h is set to 1". So

    f_px = pred_rel_focal * image_height

Checked numerically against the model's own output: for the uncentered
checkpoint, ``general_vfov_to_focal(rel_cx, rel_cy, 1, pred_general_vfov)``
reproduces ``pred_rel_focal`` to seven decimals (1.2353644 vs 1.2353643), which
confirms both the relation and the height normalisation.

Caveat recorded rather than hidden: for the *centered* checkpoint the same
identity does NOT hold (official converter gives 1.396 from its reported vfov
while the model outputs 1.068), i.e. that checkpoint's reported vfov field is
not the quantity its focal head produced. ``pred_rel_focal`` is used as the
focal in both cases because it is the network's own focal output, and the
vfov-implied focal is stored beside it as ``fx_px_from_vfov`` so a reader can
recompute either way.

``pred_rel_cx``/``pred_rel_cy`` are offsets from the image centre; per the
converter's docstring they are normalised by width and height respectively.
"""
from __future__ import annotations

import numpy as np

from .base import CalibrationAdapter, CalibrationPrediction

CENTERED = "Paramnet-360Cities-edina-centered"
UNCENTERED = "Paramnet-360Cities-edina-uncentered"


class PerspectiveFieldsAdapter(CalibrationAdapter):
    name = "PerspectiveFields"

    def __init__(self, device: str = "cuda", version: str = UNCENTERED):
        super().__init__(device)
        self.version = version
        self.name = f"PerspectiveFields[{version.split('-')[-1]}]"
        self.predicts_principal_point = version == UNCENTERED
        self.predicts_distortion = False

    def _load(self):
        from perspective2d import PerspectiveFields
        m = PerspectiveFields(self.version).eval()
        return m.cuda() if self.device == "cuda" else m

    def _predict(self, img_bgr, meta: dict) -> CalibrationPrediction:
        from perspective2d.utils.utils import general_vfov_to_focal

        p = self._model.inference(img_bgr=img_bgr)
        H = int(meta.get("image_height", img_bgr.shape[0]))
        W = int(meta.get("image_width", img_bgr.shape[1]))

        def val(key):
            v = p.get(key)
            if v is None:
                return None
            try:
                return float(v.item() if hasattr(v, "item") else v)
            except Exception:
                return None

        rel_focal = val("pred_rel_focal")
        gvfov = val("pred_general_vfov")
        if gvfov is None:
            gvfov = val("pred_vfov")
        rel_cx, rel_cy = val("pred_rel_cx"), val("pred_rel_cy")
        roll, pitch = val("pred_roll"), val("pred_pitch")

        if rel_focal is None or rel_focal <= 0:
            return CalibrationPrediction(
                model=self.name, success=False,
                raw_output=f"rel_focal={rel_focal} gvfov={gvfov}",
                failure_reason="model did not return a usable pred_rel_focal")

        f_px = rel_focal * H                       # rel_focal is f / image_height
        f_from_vfov = ""
        if gvfov is not None:
            try:
                f_from_vfov = float(general_vfov_to_focal(
                    rel_cx or 0.0, rel_cy or 0.0, 1, gvfov, True)) * H
            except Exception:
                f_from_vfov = ""

        cx = cy = None
        if self.predicts_principal_point and rel_cx is not None and rel_cy is not None:
            # offsets from the image centre, normalised by width / height
            cx = (0.5 + rel_cx) * W
            cy = (0.5 + rel_cy) * H

        return CalibrationPrediction(
            model=self.name, fx_px=f_px, fy_px=f_px, cx_px=cx, cy_px=cy,
            vfov_deg=gvfov,
            raw_output=(f"rel_focal={rel_focal} general_vfov={gvfov} "
                        f"rel_cx={rel_cx} rel_cy={rel_cy} roll={roll} pitch={pitch}"),
            conversion_note="f_px = pred_rel_focal * image_height (isotropic)",
            extra={"fx_px_from_vfov": f_from_vfov,
                   "pred_roll_deg": roll, "pred_pitch_deg": pitch,
                   "rel_focal": rel_focal,
                   "principal_point": ("predicted" if self.predicts_principal_point
                                       else "NOT_PREDICTED (centered model)")})
