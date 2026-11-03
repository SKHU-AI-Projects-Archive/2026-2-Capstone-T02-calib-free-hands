import time
import numpy as np
from calibfreehands.media.provider import MediaFrameProvider


def _wait(provider, index, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame = provider.frame(index)
        if frame is not None:
            return frame
        time.sleep(0.005)
    raise AssertionError(f"frame {index} did not become ready")


def test_provider_decodes_on_request_and_bounds_lru_cache(tmp_path):
    calls = []
    provider = MediaFrameProvider(tmp_path / "clip.mp4", cache_budget=24,
                                  frame_loader=lambda index: calls.append(index) or np.full((2, 2, 3), index, np.uint8))
    try:
        assert provider.frame(0) is None
        first = _wait(provider, 0)
        assert first.shape == (2, 2, 3) and not first.flags.writeable
        assert provider.frame(0) is first
        _wait(provider, 1)
        _wait(provider, 2)
        assert provider.cache_bytes <= 24
        assert provider.evictions == 1
        assert provider.frame(0) is None
        _wait(provider, 0)
        assert provider.evictions == 2
        assert calls == [0, 1, 2, 0]
        assert provider.cache_hits >= 1
    finally:
        provider.close()


def test_application_shutdown_closes_registered_media_providers(tmp_path, monkeypatch):
    import calibfreehands.media.provider as media_provider

    worker = MediaFrameProvider(tmp_path / "clip.mp4", frame_loader=lambda index: np.zeros((1, 1, 3), np.uint8))
    registry = {(str(worker.path), None): worker}
    monkeypatch.setattr(media_provider, "_registry", registry)

    media_provider.close_shared_media_providers()

    assert worker._closed
    assert registry == {}


def test_provider_latest_seek_request_supersedes_stale_request(tmp_path):
    requested = []
    def load(index):
        requested.append(index)
        time.sleep(0.02)
        return np.full((1, 1, 3), index, np.uint8)
    provider = MediaFrameProvider(tmp_path / "clip.mp4", frame_loader=load)
    try:
        provider.frame(1)
        provider.frame(2)
        assert np.all(_wait(provider, 2) == 2)
        assert requested[-1] == 2
    finally:
        provider.close()


def test_shared_provider_validates_path_and_reuses_identity(tmp_path):
    from calibfreehands.media.provider import shared_media_provider
    missing, status = shared_media_provider(tmp_path / "missing.mp4")
    assert missing is None and status == "missing"
    video = tmp_path / "shared.mp4"
    video.write_bytes(b"not a video")
    first, first_status = shared_media_provider(video)
    second, second_status = shared_media_provider(video)
    assert first_status == second_status == "unavailable"
    assert first is second is None


def test_provider_keeps_qimage_source_buffer_alive(qapp, tmp_path):
    from PySide6.QtGui import QImage, QColor
    from calibfreehands.media.provider import MediaFrameProvider
    image_path = tmp_path / "still.png"
    image = QImage(3, 2, QImage.Format.Format_RGB888)
    image.fill(QColor("red"))
    assert image.save(str(image_path))
    provider = MediaFrameProvider(image_path, is_image=True, cache_budget=1024)
    try:
        frame = provider.frame(99)
        assert frame is not None and frame.shape == (2, 3, 3)
        assert np.all(frame[:, :, 0] == 255)
        assert np.all(frame[:, :, 1:] == 0)
    finally:
        provider.close()


def test_same_source_identity_reuses_one_provider_and_hash_mismatch_is_rejected(qapp, tmp_path):
    import hashlib
    from PySide6.QtGui import QImage
    from calibfreehands.media.provider import shared_media_provider
    path = tmp_path / "shared.png"
    image = QImage(2, 2, QImage.Format.Format_RGB888)
    image.fill(0x123456)
    assert image.save(str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    first, first_status = shared_media_provider(path, digest)
    second, second_status = shared_media_provider(path, digest)
    assert first is second
    assert first_status == second_status == "available"
    changed, changed_status = shared_media_provider(path, "0" * 64)
    assert changed is None and changed_status == "changed"
    first.close()

def test_provider_sequential_decode_is_lazy_and_random_access_restarts(tmp_path):
    from dataclasses import dataclass
    decoded = []
    @dataclass
    class Frame:
        index: int
        rgb: np.ndarray
    def frames():
        for index in range(5):
            decoded.append(index)
            yield Frame(index, np.full((1, 1, 3), index, np.uint8))
    provider = MediaFrameProvider(tmp_path / "video.mp4", sequential_frames=frames)
    try:
        assert decoded == []
        provider.frame(2)
        assert np.all(_wait(provider, 2) == 2)
        assert decoded == [0, 1, 2]
        provider.frame(0)
        assert np.all(_wait(provider, 0) == 0)
        assert decoded == [0, 1, 2]
        provider.frame(4)
        assert np.all(_wait(provider, 4) == 4)
        assert decoded[-2:] == [3, 4]
    finally:
        provider.close()
