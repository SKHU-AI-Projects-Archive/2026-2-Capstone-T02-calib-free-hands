from __future__ import annotations

import shutil
import subprocess
from itertools import chain
from pathlib import Path
from typing import Iterable

import numpy as np


HAND_CONNECTIONS = ((0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
                    (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8),
                    (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16),
                    (17, 18), (18, 19), (19, 20))


def overlay_keypoints(frames: Iterable[np.ndarray], keypoints_2d: np.ndarray, detected: np.ndarray):
    """Yield RGB frames with the canonical 2D hand skeleton overlaid."""
    import cv2
    for index, frame in enumerate(frames):
        output = frame.copy()
        for hand in range(keypoints_2d.shape[1]):
            if not detected[index, hand]:
                continue
            points = np.rint(keypoints_2d[index, hand]).astype(np.int32)
            color = (0, 220, 255) if hand == 0 else (255, 220, 0)
            for a, b in HAND_CONNECTIONS:
                cv2.line(output, tuple(points[a]), tuple(points[b]), color, 2)
            for point in points:
                cv2.circle(output, tuple(point), 3, color, -1)
        yield output


def encode_result_preview(frames: Iterable[np.ndarray], result, output: Path, fps: float,
                          audio_source: Path | None = None, ffmpeg: str = "ffmpeg") -> Path:
    return encode_preview(overlay_keypoints(frames, result.keypoints_2d, result.detected), output, fps, audio_source, ffmpeg)


def encode_preview(frames: Iterable[np.ndarray], output: Path, fps: float, audio_source: Path | None = None,
                   ffmpeg: str = "ffmpeg") -> Path:
    """Encode RGB frames to a new MP4; optional audio is mapped from the untouched source."""
    executable = shutil.which(ffmpeg)
    if executable is None and ffmpeg == "ffmpeg":
        try:
            import imageio_ffmpeg
            executable = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            executable = None
    if executable is None:
        raise RuntimeError("ffmpeg was not found. Install FFmpeg or the project optional media dependency.")
    frame_iter = iter(frames)
    first = next(frame_iter, None)
    if first is None:
        raise ValueError("at least one preview frame is required")
    height, width = first.shape[:2]
    command = [executable, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-"]
    if audio_source is not None:
        command += ["-i", str(audio_source), "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest"]
    else:
        command += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    command.append(str(output))
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in chain((first,), frame_iter):
            if frame.shape != first.shape or frame.dtype != np.uint8:
                raise ValueError("preview frames must have identical uint8 RGB shapes")
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode(errors="replace")
        code = process.wait()
    except BrokenPipeError as exc:
        detail = process.stderr.read().decode(errors="replace")
        process.wait()
        raise RuntimeError(f"FFmpeg preview input failed: {detail.strip()}") from exc
    except Exception:
        process.kill(); process.wait()
        raise
    if code != 0:
        raise RuntimeError(f"FFmpeg preview encoding failed: {stderr.strip()}")
    return Path(output)
