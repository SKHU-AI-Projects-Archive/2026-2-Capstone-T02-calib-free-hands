import sys
from pathlib import Path

import pytest

from calibfreehands.handcalib import HandCalibError, estimate_focal

# A stand-in for the real worker: no HandCalib, no GPU, only the line protocol.
FAKE_OK = r'''
import json
print("noise from a library")
print("PROGRESS 10 20", flush=True)
print("RESULT " + json.dumps({"focal_px": 880.5, "frames": 20, "observations": 18}), flush=True)
'''


def _video(tmp_path: Path) -> Path:
    path = tmp_path / "calibration.mp4"
    path.write_bytes(b"x")
    return path


def run(tmp_path, code, **kwargs):
    return estimate_focal(_video(tmp_path), python=sys.executable, calibrator_dir=tmp_path, worker_code=code, **kwargs)


def test_parses_result_and_progress(tmp_path):
    seen = []
    result = run(tmp_path, FAKE_OK, progress=lambda stage, current, total: seen.append((current, total)))
    assert result.focal_px == 880.5 and result.frames == 20 and result.observations == 18
    assert seen == [(10, 20)]


def test_hand_not_detected_is_readable(tmp_path):
    code = 'import json; print("RESULT " + json.dumps({"error": "hand_not_detected", "frames": 90, "observations": 2}))'
    with pytest.raises(HandCalibError) as info:
        run(tmp_path, code)
    assert info.value.kind == "hand_not_detected" and "again" in str(info.value) and "2 of 90" in str(info.value)


def test_worker_crash_without_result(tmp_path):
    with pytest.raises(HandCalibError) as info:
        run(tmp_path, "raise SystemExit(3)")
    assert info.value.kind == "no_result"


def test_unusable_focal_is_rejected(tmp_path):
    code = 'import json; print("RESULT " + json.dumps({"focal_px": 0.0, "frames": 5, "observations": 5}))'
    with pytest.raises(HandCalibError) as info:
        run(tmp_path, code)
    assert info.value.kind == "invalid_focal"


def test_cancel_stops_the_worker(tmp_path):
    code = 'import time\nfor i in range(1, 100):\n    print("PROGRESS %d 100" % i, flush=True)\n    time.sleep(0.2)'
    with pytest.raises(InterruptedError):
        run(tmp_path, code, cancelled=lambda: True)


def test_missing_environment_and_video(tmp_path):
    with pytest.raises(HandCalibError) as info:
        estimate_focal(_video(tmp_path), python=tmp_path / "nope.exe", calibrator_dir=tmp_path)
    assert info.value.kind == "missing_environment"
    with pytest.raises(HandCalibError) as info:
        estimate_focal(tmp_path / "missing.mp4")
    assert info.value.kind == "missing_video"
