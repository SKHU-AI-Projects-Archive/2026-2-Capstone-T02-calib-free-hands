from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from calibfreehands.pipeline.result import CanonicalResult

JOINT_NAMES = tuple(["wrist"] + [f"joint_{i}" for i in range(1, 21)])


def _rows(result: CanonicalResult):
    for frame_pos, frame_index in enumerate(result.frame_indices):
        for hand_pos, hand in enumerate(("left", "right")):
            confidence = float(result.confidence[frame_pos, hand_pos]) if result.confidence is not None else None
            for joint_index, point in enumerate(result.keypoints_3d[frame_pos, hand_pos]):
                yield {"frame_index": int(frame_index), "timestamp": float(result.timestamps[frame_pos]),
                       "hand": hand, "joint_index": joint_index, "joint_name": JOINT_NAMES[joint_index],
                       "x_m": _json_float(point[0]), "y_m": _json_float(point[1]), "z_m": _json_float(point[2]),
                       "detected": bool(result.detected[frame_pos, hand_pos]), "confidence": _json_float(confidence)}


def _json_float(value):
    value = float(value) if value is not None else None
    return value if value is not None and math.isfinite(value) else None


def export_csv(result: CanonicalResult, path: Path) -> Path:
    fields = ["frame_index", "timestamp", "hand", "joint_index", "joint_name", "x_m", "y_m", "z_m", "detected", "confidence"]
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_rows(result))
    return path


def export_json(result: CanonicalResult, path: Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(list(_rows(result)), indent=2, allow_nan=False), encoding="utf-8")
    return path
