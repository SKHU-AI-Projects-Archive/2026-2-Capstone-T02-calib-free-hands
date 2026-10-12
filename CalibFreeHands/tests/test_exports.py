import json
import numpy as np

from calibfreehands.export import export_csv, export_json
from calibfreehands.pipeline import CanonicalResult


def _result():
    keypoints_2d = np.full((1, 2, 21, 2), np.nan); keypoints_2d[0, 0] = 0
    keypoints_3d = np.full((1, 2, 21, 3), np.nan); keypoints_3d[0, 0] = 0
    return CanonicalResult(
        np.array([0]), np.array([0.25]), keypoints_2d, keypoints_3d, np.array([[True, False]]), np.array([500.0]), (640, 480),
        confidence=np.array([[0.9, np.nan]]),
    )


def test_exports_include_long_form_joint_fields(tmp_path):
    csv_path = export_csv(_result(), tmp_path / "points.csv")
    json_path = export_json(_result(), tmp_path / "points.json")
    assert "frame_index,timestamp,hand,joint_index,joint_name,x_m,y_m,z_m,detected,confidence" in csv_path.read_text()
    payload = json.loads(json_path.read_text())
    assert payload[0]["hand"] == "left"
    assert payload[-1]["detected"] is False
