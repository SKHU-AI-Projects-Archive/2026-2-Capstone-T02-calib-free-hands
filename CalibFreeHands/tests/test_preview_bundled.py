import numpy as np
import pytest

from calibfreehands.export.preview import encode_preview


def test_bundled_ffmpeg_can_encode_preview(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    output = encode_preview([np.zeros((8, 8, 3), np.uint8), np.ones((8, 8, 3), np.uint8)], tmp_path / "preview.mp4", 2)
    assert output.exists() and output.stat().st_size > 0
