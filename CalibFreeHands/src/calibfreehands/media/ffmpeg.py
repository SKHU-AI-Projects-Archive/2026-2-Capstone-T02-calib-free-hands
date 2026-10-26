from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DecodedFrame:
    index: int
    timestamp_s: float
    rgb: np.ndarray


class FFmpegDecoder:
    """Small FFmpeg subprocess boundary; it never edits the source media."""

    def __init__(self, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> None:
        self.ffmpeg, self.ffprobe = ffmpeg, ffprobe

    def _executable(self, name: str) -> str:
        path = shutil.which(name)
        if path is None and name == "ffmpeg":
            try:
                import imageio_ffmpeg
                path = imageio_ffmpeg.get_ffmpeg_exe() if name == "ffmpeg" else None
            except ImportError:
                path = None
        if path is None:
            raise RuntimeError(f"{name} was not found. Install FFmpeg or the project optional media dependency.")
        return path

    def timestamps(self, path: Path) -> list[float]:
        executable = self._executable(self.ffprobe)
        command = [executable, "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(path)]
        try:
            output = subprocess.run(command, check=True, capture_output=True, text=True)
            frames = json.loads(output.stdout).get("frames", [])
            return [float(frame["best_effort_timestamp_time"]) for frame in frames if "best_effort_timestamp_time" in frame]
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(f"Could not read frame timestamps for {path.name}: {exc}") from exc

    def decode_rgb(self, path: Path, width: int, height: int, frame_rate: float | None = None):
        executable = self._executable(self.ffmpeg)
        command = [executable, "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        frame_bytes = width * height * 3
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            timestamps = self.timestamps(path)
        except RuntimeError as exc:
            if "ffprobe was not found" not in str(exc):
                raise
            timestamps = []
        index = 0
        try:
            while True:
                raw = process.stdout.read(frame_bytes)
                if len(raw) != frame_bytes:
                    break
                timestamp = timestamps[index] if index < len(timestamps) else (
                    float(index) / frame_rate if frame_rate and frame_rate > 0 else float("nan")
                )
                yield DecodedFrame(index, timestamp, np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3))
                index += 1
        finally:
            process.stdout.close()
            process.wait()
            if process.returncode not in (0, None):
                detail = process.stderr.read().decode(errors="replace")
                raise RuntimeError(f"FFmpeg could not decode {path.name}: {detail.strip()}")
