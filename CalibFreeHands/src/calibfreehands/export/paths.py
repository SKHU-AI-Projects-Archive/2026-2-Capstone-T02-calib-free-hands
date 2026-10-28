from __future__ import annotations

from pathlib import Path
import re


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


def result_filename(source_stem: str, calibration_name: str, reconstruction_name: str,
                    reference_focal_px: float | None = None) -> str:
    def safe(value: str) -> str:
        cleaned = re.sub(r'[<>:"/\\|?*]', "_", value).rstrip(" .") or "result"
        cleaned = re.sub(r"\s+", "-", cleaned)
        if cleaned.split(".", 1)[0].upper() in {
            "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10)),
        }:
            cleaned = f"_{cleaned}"
        return cleaned

    calibration = calibration_name
    if calibration_name == "Checkerboard Reference" and reference_focal_px is not None:
        value = format(float(reference_focal_px), ".15f").rstrip("0").rstrip(".")
        calibration = f"{calibration_name}-{value}px"
    return f"{safe(source_stem)}__{safe(calibration)}__{safe(reconstruction_name)}.cfh"


def choose_output_file(base: Path, filename: str, policy: str = "ask") -> Path | None:
    candidate = Path(base) / filename
    if not candidate.exists() or policy == "overwrite":
        return candidate
    if policy == "skip":
        return None
    if policy == "auto_rename":
        index = 1
        while candidate.with_name(f"{candidate.stem}_{index}{candidate.suffix}").exists():
            index += 1
        return candidate.with_name(f"{candidate.stem}_{index}{candidate.suffix}")
    if policy == "ask":
        raise FileExistsError(f"result already exists; confirmation required: {candidate}")
    raise ValueError(f"unknown output collision policy: {policy}")
