from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache
def _passthrough_args(executable: str) -> tuple[str, str]:
    help_text = subprocess.run([executable, "-hide_banner", "-h", "full"],
                               capture_output=True, text=True, errors="replace", check=True)
    if "-fps_mode" in help_text.stdout:
        return "-fps_mode", "passthrough"
    return "-vsync", "0"


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

    def _probe_video_frames(self, path: Path, entries: str) -> dict:
        executable = self._executable(self.ffprobe)
        command = [executable, "-v", "error", "-select_streams", "v:0",
                   "-show_entries", entries, "-of", "json", str(path)]
        try:
            output = subprocess.run(command, check=True, capture_output=True, text=True)
            return json.loads(output.stdout)
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not read frame timestamps for {path.name}: {exc}") from exc

    def timestamps(self, path: Path) -> list[float]:
        frames = self._probe_video_frames(path, "frame=best_effort_timestamp_time").get("frames", [])
        try:
            return [float(frame["best_effort_timestamp_time"]) if "best_effort_timestamp_time" in frame
                    else float("nan") for frame in frames]
        except ValueError as exc:
            raise RuntimeError(f"Could not read frame timestamps for {path.name}: {exc}") from exc

    def exact_timestamps(self, path: Path) -> tuple[Fraction, ...]:
        payload = self._probe_video_frames(path, "stream=time_base:frame=best_effort_timestamp")
        try:
            time_base = Fraction(payload["streams"][0]["time_base"])
            if time_base <= 0:
                raise ValueError("invalid time base")
            return tuple(int(frame["best_effort_timestamp"]) * time_base for frame in payload["frames"])
        except (IndexError, KeyError, ValueError, ZeroDivisionError) as exc:
            raise RuntimeError(f"Exact decoded timestamps are unavailable for {path.name}") from exc

    def decode_rgb(self, path: Path, width: int, height: int, frame_rate: float | None = None,
                   *, timestamps: list[float] | tuple[float, ...] | None = None):
        executable = self._executable(self.ffmpeg)
        command = [executable, "-v", "error", "-i", str(path), "-map", "0:v:0",
                   *_passthrough_args(executable), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        frame_bytes = width * height * 3
        if timestamps is None:
            try:
                timestamps = self.timestamps(path)
            except RuntimeError as exc:
                if "ffprobe was not found" not in str(exc):
                    raise
                timestamps = []
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        index = 0
        reached_end = False
        try:
            while True:
                raw = process.stdout.read(frame_bytes)
                if len(raw) != frame_bytes:
                    reached_end = True
                    break
                timestamp = timestamps[index] if index < len(timestamps) and np.isfinite(timestamps[index]) else (
                    float(index) / frame_rate if frame_rate and frame_rate > 0 else float("nan")
                )
                yield DecodedFrame(index, timestamp, np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3))
                index += 1
        finally:
            process.stdout.close()
            if not reached_end and process.poll() is None:
                process.terminate()
            process.wait()
            if reached_end and process.returncode not in (0, None):
                detail = process.stderr.read().decode(errors="replace")
                raise RuntimeError(f"FFmpeg could not decode {path.name}: {detail.strip()}")
