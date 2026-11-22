from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile

from calibfreehands.pipeline.provenance import sha256_file
from calibfreehands.pipeline.result import save_result
from calibfreehands.pose.wilor_backend import OriginalWiLoRBackend

from .calibration import AnyCalibIntegration, HandCalib
from .pipeline import run_video
from .video import Recording


def process(path: Path, options, progress=None, focal_ready=None, cancelled=None):
    if progress:
        progress("Reading video timestamps", 0, 0)
    path = path.resolve(strict=True)
    fingerprint = sha256_file(path)
    recording = Recording.open(path)
    if progress:
        progress("Loading focal estimator", 0, 0)
    if options.estimator == "handcalib":
        calibration = HandCalib(options.handcalib_device)
        default_hz = 10.0
    else:
        calibration = AnyCalibIntegration(options.device)
        default_hz = 1.0
    calibration_hz = default_hz if options.calibration_hz is None else options.calibration_hz

    root = options.runtime_root
    if "YOLO_CONFIG_DIR" not in os.environ:
        detector_config = options.output_dir / ".detector-config"
        detector_config.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(detector_config)
    pose = OriginalWiLoRBackend(
        root / "WiLoR", root / "WiLoR/pretrained_models/wilor_final.ckpt",
        options.project_root / "models/model_config_wilor_original.yaml", options.device,
    )
    result, report = run_video(
        recording, calibration, pose, work_seconds=options.work_seconds, calibration_hz=calibration_hz,
        progress=progress, focal_ready=focal_ready, cancelled=cancelled,
    )
    if sha256_file(recording.path) != fingerprint:
        raise ValueError("Input file changed during processing")
    report.update(
        input_path=str(recording.path), input_sha256=fingerprint, coordinate_system="camera_space",
        calibration_target_hz=calibration_hz,
        handcalib_device=options.handcalib_device if options.estimator == "handcalib" else None,
        pose_device=options.device, unit="metre",
    )
    options.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
    output = Path(tempfile.mkdtemp(prefix=prefix, dir=options.output_dir))
    result_path, _ = save_result(result, output, report)
    return result_path, report
