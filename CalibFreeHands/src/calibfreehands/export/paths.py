from __future__ import annotations

from pathlib import Path


def choose_output_directory(base: Path, stem: str, policy: str = "ask") -> Path:
    base = Path(base)
    candidate = base / stem
    if not candidate.exists():
        return candidate
    if policy == "overwrite":
        return candidate
    if policy == "skip":
        raise FileExistsError(f"output already exists: {candidate}")
    if policy == "auto_rename":
        index = 1
        while (base / f"{stem}_{index}").exists():
            index += 1
        return base / f"{stem}_{index}"
    raise FileExistsError(f"output already exists; confirmation required: {candidate}")
