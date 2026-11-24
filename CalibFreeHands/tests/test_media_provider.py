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
    import calibfreehands.media.provider as media_provider
    path = tmp_path / "shared.png"
    image = QImage(2, 2, QImage.Format.Format_RGB888)
    image.fill(0x123456)
    assert image.save(str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    first, first_status = media_provider.shared_media_provider(path, digest)
    second, second_status = media_provider.shared_media_provider(path, digest)
    assert first is second
    assert first_status == second_status == "available"
    reference_count = getattr(media_provider, "media_provider_reference_count", None)
    release = getattr(media_provider, "release_shared_media_provider", None)
    assert callable(reference_count), "provider reference-count query is missing"
    assert callable(release), "provider release operation is missing"
    assert reference_count(first) == 2
    changed, changed_status = media_provider.shared_media_provider(path, "0" * 64)
    assert changed is None and changed_status == "changed"
    release(first)
    assert not first._closed
    release(second)
    assert first._closed


def test_shared_provider_release_closes_only_after_last_owner(tmp_path, monkeypatch):
    import calibfreehands.media.provider as media_provider

    path = tmp_path / "shared.mp4"
    path.write_bytes(b"video")
    provider = MediaFrameProvider(
        path, frame_loader=lambda index: np.full((1, 1, 3), index, np.uint8)
    )
    monkeypatch.setattr(media_provider, "MediaFrameProvider", lambda *_args, **_kwargs: provider)

    first, _ = media_provider.shared_media_provider(path)
    second, _ = media_provider.shared_media_provider(path)
    assert first is second is provider
    reference_count = getattr(media_provider, "media_provider_reference_count", None)
    release = getattr(media_provider, "release_shared_media_provider", None)
    assert callable(reference_count), "provider reference-count query is missing"
    assert callable(release), "provider release operation is missing"
    assert reference_count(provider) == 2

    release(first)
    assert reference_count(provider) == 1
    assert not provider._closed

    release(second)
    assert reference_count(provider) == 0
    assert provider._closed
    assert not provider._worker.is_alive()
    assert provider.cache_bytes == 0
    assert media_provider._registry == {}


def test_provider_close_is_idempotent_and_releases_cached_frames(tmp_path):
    from PySide6.QtGui import QImage

    image_path = tmp_path / "still.png"
    image = QImage(3, 2, QImage.Format.Format_RGB888)
    image.fill(0x123456)
    assert image.save(str(image_path))
    provider = MediaFrameProvider(image_path, is_image=True, cache_budget=1024)
    assert provider.cache_bytes > 0

    provider.close()
    provider.close()

    assert provider._closed
    assert not provider._worker.is_alive()
    assert provider.cache_bytes == 0


def test_unrelated_source_providers_release_independently_after_last_owner(tmp_path, qapp):
    import calibfreehands.media.provider as media_provider
    from PySide6.QtGui import QImage

    paths = []
    for name in ("source-a.png", "source-b.png", "source-c.png"):
        path = tmp_path / name
        image = QImage(4, 3, QImage.Format.Format_RGB888)
        image.fill(0x336699)
        assert image.save(str(path))
        paths.append(path)
    providers = [media_provider.shared_media_provider(path)[0] for path in paths]

    assert all(providers)
    assert len(media_provider._registry) == 3
    for provider in providers:
        media_provider.release_shared_media_provider(provider)
        assert provider._closed
        assert not provider._worker.is_alive()
        assert provider.cache_bytes == 0
    assert media_provider._registry == {}
    assert media_provider._reference_counts == {}

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
