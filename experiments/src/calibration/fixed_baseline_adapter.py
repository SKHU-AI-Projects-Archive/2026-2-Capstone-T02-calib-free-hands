"""Non-learned reference baselines.

DEMO_FIXED_5000 is the focal the existing AnyHand/WiLoR pipeline assumes
(CAM-EXP-002 audit): ``FOCAL_LENGTH / IMAGE_SIZE * max(W, H)``. It is a real
estimator in the sense that it is what the deployed system uses today, so it is
ranked alongside the models.

ORACLE_GT reproduces the ground truth exactly. It exists only as a reference
line in tables and figures and must never be ranked as an estimator.
"""
from __future__ import annotations

from .base import CalibrationAdapter, CalibrationPrediction

CFG_FOCAL_LENGTH = 1000.0
CFG_IMAGE_SIZE = 256.0


class DemoFixedFocalAdapter(CalibrationAdapter):
    name = "DEMO_FIXED_5000"
    predicts_principal_point = False
    predicts_distortion = False

    def _load(self):
        return object()

    def _predict(self, img_bgr, meta: dict) -> CalibrationPrediction:
        W = int(meta.get("image_width", img_bgr.shape[1]))
        H = int(meta.get("image_height", img_bgr.shape[0]))
        f = CFG_FOCAL_LENGTH / CFG_IMAGE_SIZE * max(W, H)
        return CalibrationPrediction(
            model=self.name, fx_px=f, fy_px=f, cx_px=None, cy_px=None,
            raw_output=f"FOCAL_LENGTH/IMAGE_SIZE*max(W,H) = "
                       f"{CFG_FOCAL_LENGTH}/{CFG_IMAGE_SIZE}*{max(W, H)}",
            conversion_note="constant, already in original pixels",
            extra={"principal_point": "NOT_PREDICTED (pipeline uses image centre)"})


class ImageCenterPrincipalPointAdapter(CalibrationAdapter):
    """Principal-point-only reference: cx = W/2, cy = H/2."""

    name = "IMAGE_CENTER_PP"
    predicts_principal_point = True
    predicts_distortion = False

    def _load(self):
        return object()

    def _predict(self, img_bgr, meta: dict) -> CalibrationPrediction:
        W = int(meta.get("image_width", img_bgr.shape[1]))
        H = int(meta.get("image_height", img_bgr.shape[0]))
        return CalibrationPrediction(
            model=self.name, fx_px=None, fy_px=None, cx_px=W / 2.0, cy_px=H / 2.0,
            raw_output="cx=W/2, cy=H/2",
            conversion_note="constant, already in original pixels",
            failure_reason="focal NOT_PREDICTED (principal-point reference only)")


class OracleGTAdapter(CalibrationAdapter):
    """Reference line only - never ranked as an estimator."""

    name = "ORACLE_GT"
    predicts_principal_point = True
    predicts_distortion = False

    def _load(self):
        return object()

    def _predict(self, img_bgr, meta: dict) -> CalibrationPrediction:
        return CalibrationPrediction(
            model=self.name,
            fx_px=float(meta["gt_fx"]), fy_px=float(meta["gt_fy"]),
            cx_px=float(meta["gt_cx"]), cy_px=float(meta["gt_cy"]),
            raw_output="ground truth", conversion_note="identity")
