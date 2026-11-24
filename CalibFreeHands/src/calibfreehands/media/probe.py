from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    width: int | None
    height: int | None
    frame_rate: float | None
    duration_s: float | None
    frames: int | None


def probe_media(path: Path, ffprobe: str = "ffprobe") -> MediaInfo:
    executable = shutil.which(ffprobe)
    if executable is None:
        if ffprobe != "ffprobe":
            raise RuntimeError("ffprobe was not found. Install FFmpeg and add its bin directory to PATH.")
        try:
            import imageio_ffmpeg
            executable = imageio_ffmpeg.get_ffmpeg_exe()
            completed = subprocess.run([executable, "-hide_banner", "-i", str(path)], capture_output=True, text=True)
            text = completed.stderr
            match = re.search(r"(\d{2,5})x(\d{2,5})", text)
            if not match:
                raise RuntimeError(f"Could not determine video dimensions for {path.name}")
            rate = re.search(r"(\d+(?:\.\d+)?)\s*fps", text)
            duration = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?),", text)
            duration_s = None
            if duration:
                duration_s = int(duration.group(1)) * 3600 + int(duration.group(2)) * 60 + float(duration.group(3))
            return MediaInfo(Path(path), int(match.group(1)), int(match.group(2)), float(rate.group(1)) if rate else None, duration_s, None)
        except (ImportError, OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("ffprobe was not found and bundled FFmpeg metadata fallback is unavailable.") from exc
    command = [executable, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read media metadata for {path.name}: {exc}") from exc
    stream = next((item for item in payload.get("streams", []) if item.get("codec_type") == "video"), {})
    try:
        numerator, denominator = stream.get("r_frame_rate", "").split("/", 1)
        frame_rate = float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        frame_rate = None
    duration = stream.get("duration", payload.get("format", {}).get("duration"))
    frame_count = stream.get("nb_frames")
    return MediaInfo(Path(path), stream.get("width"), stream.get("height"), frame_rate,
                     float(duration) if duration is not None else None,
                     int(frame_count) if frame_count and frame_count != "N/A" else None)
