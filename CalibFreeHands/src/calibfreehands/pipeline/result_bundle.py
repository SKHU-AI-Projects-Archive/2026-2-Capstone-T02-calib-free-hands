from __future__ import annotations

import io
import json
import math
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .result import CanonicalResult, validate_result

BUNDLE_SCHEMA_VERSION = 1
_MEMBERS = {"result.npz", "metadata.json"}
_REQUIRED_METADATA = {
    "application_version", "source_media_name", "source_media_path", "source_media_sha256",
    "calibration_model_id", "calibration_display_name", "reconstruction_model_id",
    "reconstruction_display_name", "coordinate_system", "unit", "focal_px", "sampling_mode",
    "sample_count", "aggregation", "principal_point_applied", "distortion_applied", "run_provenance",
}
_REQUIRED_ARRAYS = {
    "frame_indices", "timestamps", "keypoints_2d", "keypoints_3d", "detected", "focal_px", "image_size",
}


@dataclass(frozen=True)
class ResultBundle:
    result: CanonicalResult
    metadata: dict


def _validate_metadata(metadata: object) -> dict:
    if not isinstance(metadata, dict):
        raise ValueError("Result bundle metadata must be a JSON object")
    version = metadata.get("bundle_schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version != BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported result bundle schema: {metadata.get('bundle_schema_version')}")
    missing = _REQUIRED_METADATA - metadata.keys()
    if missing:
        raise ValueError(f"Result bundle metadata is missing required fields: {', '.join(sorted(missing))}")
    string_fields = _REQUIRED_METADATA - {"focal_px", "sample_count", "principal_point_applied", "distortion_applied", "run_provenance"}
    for name in string_fields:
        if not isinstance(metadata[name], str) or not metadata[name].strip():
            raise ValueError(f"Result bundle metadata field {name} must be a non-empty string")
    digest = metadata["source_media_sha256"]
    if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise ValueError("Result bundle source_media_sha256 must be a 64-character SHA256")
    focal = metadata["focal_px"]
    if isinstance(focal, bool) or not isinstance(focal, (int, float)) or not math.isfinite(focal) or focal <= 0:
        raise ValueError("Result bundle focal_px must be finite and positive")
    count = metadata["sample_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("Result bundle sample_count must be a positive integer")
    if metadata["sampling_mode"] not in {"uniform", "all"} or metadata["aggregation"] not in {"median", "mean"}:
        raise ValueError("Result bundle sampling metadata is invalid")
    if any(not isinstance(metadata[name], bool) for name in ("principal_point_applied", "distortion_applied")):
        raise ValueError("Result bundle camera-parameter flags must be boolean")
    if not isinstance(metadata["run_provenance"], dict):
        raise ValueError("Result bundle run_provenance must be a JSON object")
    return metadata


def _canonical_result(npz_bytes: bytes) -> CanonicalResult:
    try:
        with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as arrays:
            missing = _REQUIRED_ARRAYS - set(arrays.files)
            if missing:
                if "keypoints_3d" not in arrays.files and "keypoints_3d_relative" in arrays.files:
                    raise ValueError("This NPZ is an internal raw-pose cache, not a final CalibFreeHands result.")
                raise ValueError(f"Result NPZ is missing canonical arrays: {', '.join(sorted(missing))}")
            optional = lambda name: arrays[name].copy() if name in arrays.files else None
            result = CanonicalResult(
                arrays["frame_indices"].copy(), arrays["timestamps"].copy(), arrays["keypoints_2d"].copy(),
                arrays["keypoints_3d"].copy(), arrays["detected"].copy(), arrays["focal_px"].copy(),
                tuple(int(value) for value in arrays["image_size"]), optional("mano_vertices"),
                optional("mano_faces"), optional("confidence"), optional("bbox"), optional("mano_pose"),
                optional("mano_shape"), optional("camera_translation"),
                int(arrays["schema_version"]) if "schema_version" in arrays.files else 1,
                float(arrays["video_focal_px"]) if "video_focal_px" in arrays.files else None,
            )
    except ValueError:
        raise
    except (KeyError, OSError, TypeError, OverflowError) as exc:
        raise ValueError(f"Invalid canonical result NPZ: {exc}") from exc
    validate_result(result)
    return result


def write_result_bundle(result_npz: Path, metadata: dict, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = {"bundle_schema_version": BUNDLE_SCHEMA_VERSION, **metadata}
    _validate_metadata(values)
    result_bytes = Path(result_npz).read_bytes()
    _canonical_result(result_bytes)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp", delete=False) as stream:
            temporary_path = Path(stream.name)
        with zipfile.ZipFile(temporary_path, "w") as archive:
            archive.writestr("result.npz", result_bytes, compress_type=zipfile.ZIP_STORED)
            archive.writestr("metadata.json", json.dumps(values, indent=2, sort_keys=True, allow_nan=False),
                             compress_type=zipfile.ZIP_DEFLATED)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output_path


def load_result_bundle(path: Path) -> ResultBundle:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(_MEMBERS) or set(names) != _MEMBERS:
                raise ValueError("Result bundle must contain exactly result.npz and metadata.json")
            try:
                raw_metadata = json.loads(archive.read("metadata.json"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid result bundle metadata JSON: {exc}") from exc
            metadata = _validate_metadata(raw_metadata)
            result = _canonical_result(archive.read("result.npz"))
    except ValueError:
        raise
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError, KeyError) as exc:
        raise ValueError(f"Could not read CalibFreeHands result bundle: {exc}") from exc
    return ResultBundle(result, metadata)
