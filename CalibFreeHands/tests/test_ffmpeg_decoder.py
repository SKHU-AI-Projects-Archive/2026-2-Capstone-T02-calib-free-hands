import pytest

from calibfreehands.media.ffmpeg import FFmpegDecoder


def test_decoder_reports_missing_ffmpeg(tmp_path):
    decoder = FFmpegDecoder(ffmpeg="missing-ffmpeg", ffprobe="missing-ffprobe")
    with pytest.raises(RuntimeError, match="missing-ffprobe was not found"):
        decoder.timestamps(tmp_path / "sample.mp4")
