"""Exact frame cut of the normalised (lossless) video: frames with t0 <= t < t1, re-encoded losslessly, timestamps reset."""
from __future__ import annotations

import subprocess
from pathlib import Path


def cut(src: Path, dst: Path, t0: float, t1: float, fps: float) -> dict:
    f0 = int(round(t0 * fps))
    f1 = int(round(t1 * fps))
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-vf",
                    f"select=between(n\\,{f0}\\,{f1 - 1}),setpts=N/FRAME_RATE/TB", "-vsync", "0", "-an", "-map_metadata", "-1",
                    "-c:v", "ffv1", str(dst)], check=True)
    return {"first_frame": f0, "end_frame": f1, "n_frames": f1 - f0, "t0": f0 / fps, "t1": f1 / fps}
