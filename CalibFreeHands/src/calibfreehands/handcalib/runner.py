"""Focal length (px) of a calibration video, from HandCalib's ``benchmark/calibrator.py`` in hand mode.

HandCalib needs OpenCV 5 + numpy 2 (the benchmark environment, ``.venv_bench``), while this application needs
numpy 1.x for torch. OpenCV 4.10 gives the same focal within 0.02 %, but its ``calibrateCamera`` is ~500x slower on
these inputs. So the estimate runs in a separate Python process of the HandCalib environment; this module starts it,
reads its progress and result, and turns every failure into a readable :class:`HandCalibError`.

The worker does exactly what ``Calibrator.run(video)`` does (read every frame, ``add`` it, ``estimate``), only with
progress lines, so the numbers are the ones the benchmark scores.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

MIN_OBSERVATIONS = 5  # frames with a confidently detected hand needed for a focal estimate

_WORKER = r'''
import json, sys
import cv2
sys.path.insert(0, sys.argv[2])
from calibrator import Calibrator

def emit(payload):
    print("RESULT " + json.dumps(payload), flush=True)

try:
    calibrator = Calibrator("hand")
    capture = cv2.VideoCapture(sys.argv[1])
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        calibrator.add(frame)
        frames += 1
        if frames % 10 == 0:
            print("PROGRESS %d %d" % (frames, total), flush=True)
    capture.release()
    observations = len(calibrator.observations)
    if frames == 0:
        emit({"error": "no_frames"})
    elif observations < int(sys.argv[3]):
        emit({"error": "hand_not_detected", "frames": frames, "observations": observations})
    else:
        emit({"focal_px": float(calibrator.estimate()), "frames": frames, "observations": observations})
except Exception as exc:
    emit({"error": "failed", "detail": "%s: %s" % (type(exc).__name__, exc)})
'''


class HandCalibError(RuntimeError):
    """The estimate could not be made. ``kind`` is machine-readable, ``str(error)`` is a message for the user."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class HandCalibResult:
    focal_px: float
    frames: int
    observations: int
    seconds: float


def repo_root() -> Path:
    # .../CalibFreeHands/src/calibfreehands/handcalib/runner.py -> repository root (same convention as the app)
    return Path(__file__).resolve().parents[4]


def default_python(root: Path | None = None) -> Path:
    """Python of the HandCalib environment (``.venv_bench`` in the repository root)."""
    root = root or repo_root()
    return root / ".venv_bench" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _cuda_path_dirs(python: Path) -> list[str]:
    """onnxruntime-gpu silently falls back to the CPU (7x slower, slightly different f) unless the pip CUDA DLL
    directories of the environment are on PATH."""
    site = python.parent.parent / "Lib" / "site-packages" / "nvidia"
    if not site.is_dir():
        return []
    return [str(path / "bin") for path in sorted(site.iterdir()) if (path / "bin").is_dir()]


def estimate_focal(
    video: Path,
    *,
    python: Path | str | None = None,
    calibrator_dir: Path | str | None = None,
    progress: Callable[[str, int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    timeout_s: float = 1800.0,
    worker_code: str | None = None,  # tests only
) -> HandCalibResult:
    """Focal length in pixels of the resolution of *video* (a hand-motion calibration video)."""
    video = Path(video)
    if not video.is_file():
        raise HandCalibError("missing_video", f"The calibration video was not found: {video.name}")
    root = repo_root()
    python = Path(python) if python else default_python(root)
    calibrator_dir = Path(calibrator_dir) if calibrator_dir else root / "HandCalib" / "benchmark"
    if not python.is_file():
        raise HandCalibError("missing_environment",
                             f"The HandCalib Python environment was not found ({python}). "
                             "Set it in the settings (handcalib_python).")
    if worker_code is None and not (calibrator_dir / "calibrator.py").is_file():
        raise HandCalibError("missing_calibrator", f"HandCalib's calibrator.py was not found in {calibrator_dir}.")

    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([*_cuda_path_dirs(python), env.get("PATH", "")])
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    command = [str(python), "-c", worker_code or _WORKER, str(video), str(calibrator_dir), str(MIN_OBSERVATIONS)]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    started = time.monotonic()
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   encoding="utf-8", errors="replace", env=env, creationflags=flags)
    except OSError as exc:
        raise HandCalibError("start_failed", f"HandCalib could not be started: {exc}") from exc

    result: dict | None = None
    try:
        assert process.stdout is not None
        for line in process.stdout:
            if cancelled and cancelled():
                raise InterruptedError("HandCalib cancelled")
            if time.monotonic() - started > timeout_s:
                raise HandCalibError("timeout", "HandCalib took too long and was stopped.")
            line = line.strip()
            if line.startswith("PROGRESS "):
                try:
                    _, current, total = line.split()
                    if progress:
                        progress("Estimating focal length (HandCalib)", int(current), max(int(total), int(current)))
                except ValueError:
                    pass
            elif line.startswith("RESULT "):
                result = json.loads(line[len("RESULT "):])
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()

    if result is None:
        raise HandCalibError("no_result", "HandCalib ended without a result.")
    if "error" in result:
        kind = result["error"]
        if kind == "hand_not_detected":
            raise HandCalibError(
                kind, f"The hand was found in only {result.get('observations', 0)} of {result.get('frames', 0)} frames. "
                      "Record the calibration video again: keep the whole right hand in view and move it.")
        if kind == "no_frames":
            raise HandCalibError(kind, "The calibration video has no readable frames.")
        raise HandCalibError(kind, f"HandCalib failed: {result.get('detail', 'unknown error')}")
    focal = float(result["focal_px"])
    if not (focal == focal and 0 < focal < 1e6):  # NaN, zero or absurd
        raise HandCalibError("invalid_focal", f"HandCalib returned an unusable focal length ({focal}).")
    return HandCalibResult(focal, int(result["frames"]), int(result["observations"]), time.monotonic() - started)
