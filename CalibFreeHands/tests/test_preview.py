import numpy as np
import pytest

from calibfreehands.export.preview import encode_preview


def test_preview_reports_missing_ffmpeg(tmp_path):
    with pytest.raises(RuntimeError, match="ffmpeg was not found"):
        encode_preview([np.zeros((4, 4, 3), np.uint8)], tmp_path / "preview.mp4", 30, ffmpeg="missing-ffmpeg")
