"""Frame selection (detect_fps) from the normalised video into lossless PNG files with a frame / time / hash list."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def extract(video: Path, out_dir: Path, fps: float, n_frames: int | None, detect_fps: float) -> list:
    """Every k-th frame (k = round(fps / detect_fps), at least 1). Returns [{index, frame, time_s, png, sha256}]."""
    out_dir.mkdir(parents=True, exist_ok=True)
    k = max(1, int(round(fps / detect_fps)))
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-vf", f"select=not(mod(n\\,{k}))",
                    "-vsync", "0", "-start_number", "0", str(out_dir / "f_%06d.png")], check=True)
    files = sorted(out_dir.glob("f_*.png"))
    rows = []
    for i, p in enumerate(files):
        fr = i * k
        rows.append({"index": i, "frame": fr, "time_s": fr / fps, "png": str(p), "sha256": sha256(p)})
    return rows, k
