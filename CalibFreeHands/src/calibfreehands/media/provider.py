from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from threading import Condition, Thread
from typing import Callable, Iterator

import numpy as np
from PySide6.QtCore import QObject, Signal

from .ffmpeg import FFmpegDecoder
from .probe import probe_media

DEFAULT_CACHE_BUDGET = 384 * 1024 * 1024


class _ProviderSignals(QObject):
    frame_ready = Signal(int)
    failed = Signal(str)


class MediaFrameProvider:
    """Asynchronously decodes requested RGB frames into a byte-bounded LRU cache."""

    def __init__(self, path: Path, *, cache_budget: int = DEFAULT_CACHE_BUDGET,
                 frame_loader: Callable[[int], np.ndarray] | None = None,
                 sequential_frames: Callable[[], Iterator] | None = None,
                 frame_size: tuple[int, int] | None = None, frame_rate: float | None = None,
                 is_image: bool = False) -> None:
        if cache_budget <= 0:
            raise ValueError("cache_budget must be positive")
        self.path = Path(path)
        self.cache_budget = cache_budget
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_bytes = 0
        self._condition = Condition()
        self._requested: int | None = None
        self._active_target: int | None = None
        self._closed = False
        self._loader = frame_loader
        self._sequential_frames = sequential_frames
        self._frame_size = frame_size
        self._frame_rate = frame_rate
        self._iterator = None
        self._next_index = 0
        self._is_image = is_image
        self.signals = _ProviderSignals()
        self.decoded_frames = 0
        self.cache_hits = 0
        self.evictions = 0
        if is_image:
            from PySide6.QtGui import QImage
            image = QImage(str(self.path)).convertToFormat(QImage.Format.Format_RGB888)
            if image.isNull():
                raise ValueError(f"Could not load image source: {self.path}")
            rgb = np.frombuffer(image.bits(), dtype=np.uint8).reshape(image.height(), image.bytesPerLine())[:, :image.width() * 3]
            rgb = np.ascontiguousarray(rgb.reshape(image.height(), image.width(), 3))
            self._store(0, rgb)
        elif self._loader is None and self._sequential_frames is None:
            info = probe_media(self.path)
            if not info.width or not info.height:
                raise ValueError(f"Could not determine video frame size: {self.path}")
            self._frame_size = (info.width, info.height)
            self._frame_rate = info.frame_rate
            self._sequential_frames = lambda: FFmpegDecoder().decode_rgb(
                self.path, info.width, info.height, info.frame_rate)
        self._worker = Thread(target=self._work, name=f"media-decode-{self.path.name}", daemon=True)
        self._worker.start()

    @property
    def cache_bytes(self) -> int:
        with self._condition:
            return self._cache_bytes

    def frame(self, index: int) -> np.ndarray | None:
        cached = self.cached_frame(index)
        if cached is not None:
            return cached
        index = 0 if self._is_image else int(index)
        with self._condition:
            if self._closed:
                return None
            if index == self._requested or index == self._active_target:
                return None
            self._requested = index
            self._condition.notify()
            return None

    def cached_frame(self, index: int) -> np.ndarray | None:
        index = 0 if self._is_image else int(index)
        with self._condition:
            cached = self._cache.get(index)
            if cached is None:
                return None
            self._cache.move_to_end(index)
            self.cache_hits += 1
            return cached

    def _store(self, index: int, frame: np.ndarray) -> None:
        array = np.ascontiguousarray(frame)
        if array.nbytes > self.cache_budget:
            return
        array.setflags(write=False)
        with self._condition:
            old = self._cache.pop(index, None)
            if old is not None:
                self._cache_bytes -= old.nbytes
            while self._cache and self._cache_bytes + array.nbytes > self.cache_budget:
                _, evicted = self._cache.popitem(last=False)
                self._cache_bytes -= evicted.nbytes
                self.evictions += 1
            self._cache[index] = array
            self._cache_bytes += array.nbytes
            self.decoded_frames += 1

    def _decode_until(self, target: int) -> None:
        if self._loader is not None:
            frame = self._loader(target)
            self._store(target, frame)
            self.signals.frame_ready.emit(target)
            return
        if self._sequential_frames is None:
            return
        if self._iterator is None or target < self._next_index:
            if self._iterator is not None:
                self._iterator.close()
            self._iterator = iter(self._sequential_frames())
            self._next_index = 0
        while self._next_index <= target:
            with self._condition:
                if self._closed:
                    return
                latest = self._requested
            if latest is not None and latest < self._next_index:
                return
            item = next(self._iterator, None)
            if item is None:
                return
            index = int(item.index)
            self._store(index, item.rgb)
            self._next_index = index + 1
            self.signals.frame_ready.emit(index)

    def _work(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._requested is not None or self._closed)
                if self._closed:
                    if self._iterator is not None:
                        self._iterator.close()
                    return
                target = self._requested
                self._requested = None
                self._active_target = target
            if target is None:
                continue
            try:
                self._decode_until(target)
            except (OSError, RuntimeError, ValueError, StopIteration) as exc:
                self.signals.failed.emit(str(exc))
            finally:
                with self._condition:
                    self._active_target = None

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._requested = None
            self._condition.notify_all()
        if self._worker.is_alive():
            self._worker.join(timeout=1.0)


_registry: dict[tuple[str, str | None], MediaFrameProvider] = {}


def shared_media_provider(path: Path, expected_sha256: str | None = None) -> tuple[MediaFrameProvider | None, str]:
    path = Path(path).resolve()
    if not path.is_file():
        return None, "missing"
    try:
        if expected_sha256:
            from calibfreehands.pipeline.provenance import sha256_file
            if sha256_file(path).casefold() != expected_sha256.casefold():
                return None, "changed"
        key = (str(path), expected_sha256.casefold() if expected_sha256 else None)
        provider = _registry.get(key)
        if provider is None:
            suffix = path.suffix.casefold()
            is_image = suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
            provider = MediaFrameProvider(path, is_image=is_image)
            _registry[key] = provider
        return provider, "available"
    except (OSError, ValueError, RuntimeError):
        return None, "unavailable"


def close_shared_media_providers() -> None:
    providers = tuple(_registry.values())
    _registry.clear()
    for provider in providers:
        provider.close()
