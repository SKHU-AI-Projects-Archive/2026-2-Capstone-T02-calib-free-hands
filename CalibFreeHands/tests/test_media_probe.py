import pytest

from calibfreehands.media.probe import probe_media


def test_probe_reports_missing_ffprobe(tmp_path):
    with pytest.raises(RuntimeError, match="ffprobe was not found"):
        probe_media(tmp_path / "sample.mp4", ffprobe="definitely-missing-ffprobe")
