from __future__ import annotations

import sys
import os
import inspect
from pathlib import Path

import numpy as np


def _prepare_legacy_mano_imports() -> None:
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    for name, value in (("bool", bool), ("int", int), ("float", float), ("complex", complex),
                        ("object", object), ("unicode", str), ("str", str)):
        if name not in np.__dict__:
            setattr(np, name, value)


class OriginalWiLoRBackend:
    """Adapter for the original WiLoR source tree and checkpoint pairing."""

    name = "WiLoR original pretrained"

    def __init__(self, repository: Path, checkpoint: Path, config: Path, device: str = "cuda") -> None:
        self.repository = Path(repository).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.config = Path(config).resolve()
        self.device = device
        self.model = None
        self._predictor = None
        self.mano_paths = {
            "left": self.repository.parent / "mano_data" / "MANO_LEFT.pkl",
            "right": self.repository.parent / "mano_data" / "MANO_RIGHT.pkl",
        }

    def load(self) -> None:
        missing = [str(path) for path in (self.repository, self.checkpoint, self.config) if not path.exists()]
        if missing:
            raise FileNotFoundError("Original WiLoR assets are missing: " + ", ".join(missing))
        # MANO's licensed pickle imports legacy chumpy on Python 3.11+/NumPy 2.x.
        _prepare_legacy_mano_imports()
        repository = self.repository.resolve()
        checkpoint = self.checkpoint.resolve()
        config = self.config.resolve()
        sys.path.insert(0, str(repository))
        if os.name == "nt":
            os.environ.setdefault("PYOPENGL_PLATFORM", "win32")
        try:
            from wilor.models import load_wilor
        except ImportError as exc:
            raise RuntimeError(f"Original WiLoR import failed in {sys.executable}: {exc}") from exc
        # WiLoR's published loader resolves MANO paths relative to CWD.
        # Keep that legacy assumption inside this adapter only.
        previous_cwd = Path.cwd()
        try:
            os.chdir(repository.parent)
            self.model, _ = load_wilor(str(checkpoint), str(config), init_renderer=False)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"WiLoR model asset is missing while loading from {repository.parent}: {exc}"
            ) from exc
        finally:
            os.chdir(previous_cwd)
        self.model.to(self.device).eval()

    def preflight(self) -> dict[str, object]:
        paths = (self.repository, self.checkpoint, self.config, *self.mano_paths.values(),
                 self.repository / "pretrained_models" / "detector.pt")
        missing = [str(path) for path in paths if not path.is_file() and path != self.repository]
        if not self.repository.is_dir():
            missing.insert(0, str(self.repository))
        if missing:
            raise FileNotFoundError("WiLoR preflight missing assets: " + ", ".join(missing))
        import torch
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable in {sys.executable}")
        return {"python": sys.executable, "torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available())}

    def predict(self, *args, **kwargs):
        if len(args) < 2:
            raise TypeError("predict expects image_rgb and focal_px")
        image_rgb, focal_px = np.asarray(args[0]), float(args[1])
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb must have shape (height, width, 3)")
        repository = self.repository.resolve()
        detector = repository / "pretrained_models" / "detector.pt"
        if not detector.is_file():
            raise FileNotFoundError(f"WiLoR detector asset is missing: {detector}")
        if self._predictor is None:
            root = repository.parent
            checkpoint = self.checkpoint.resolve()
            config = self.config.resolve()
            project_local_config = Path.cwd()
            if os.name == "nt":
                os.environ.setdefault("PYOPENGL_PLATFORM", "win32")
            os.environ.setdefault("YOLO_CONFIG_DIR", str(project_local_config))
            sys.path.insert(0, str(root))
            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                _prepare_legacy_mano_imports()
                from rgb_predictor import AnyHandPredictor
                try:
                    self._predictor = AnyHandPredictor(
                        backend="wilor", device=self.device,
                        wilor_ckpt=str(checkpoint), wilor_cfg=str(config), detector_pt=str(detector),
                    )
                except ImportError as exc:
                    raise RuntimeError(f"WiLoR runtime import failed in {sys.executable}: {exc}") from exc
            finally:
                os.chdir(previous_cwd)
        predictions = self._predictor.predict(image_rgb[:, :, ::-1].copy(), backend="wilor")
        from calibfreehands.calibration.focal import rescale_camera_translation
        from calibfreehands.pipeline.service import HandEstimate
        estimates = []
        mano_faces = np.asarray(getattr(getattr(self._predictor, "_wilor_model", None), "mano", None).faces, dtype=np.int32) if getattr(getattr(self._predictor, "_wilor_model", None), "mano", None) is not None else None
        for prediction in predictions:
            source_focal = float(prediction.focal_length)
            translation = rescale_camera_translation(prediction.cam_t, source_focal, focal_px)
            estimates.append(HandEstimate(
                is_right=bool(prediction.is_right),
                keypoints_2d=np.asarray(prediction.keypoints_2d, dtype=np.float32),
                keypoints_3d_relative=np.asarray(prediction.keypoints_3d, dtype=np.float32),
                camera_translation=translation,
                vertices_relative=np.asarray(prediction.vertices, dtype=np.float32),
                bbox=np.asarray(prediction.bbox, dtype=np.float32),
                confidence=float(prediction.score),
                mano_pose=np.asarray(prediction.mano_pose, dtype=np.float32),
                mano_shape=np.asarray(prediction.mano_shape, dtype=np.float32),
                mano_faces=mano_faces,
            ))
        return estimates
