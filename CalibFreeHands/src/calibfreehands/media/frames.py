"""Frames of one video without holding them in memory.

A 1080p frame is ~6 MB as RGB, so keeping every frame of a 20 s clip takes ~3.7 GB and a minute ~11 GB. The processing
steps only need (a) every frame once, in order (hand reconstruction), (b) a few sampled frames or each frame once
(focal estimation) and (c) the frame numbers, times and size. A :class:`FrameSource` gives exactly that: it knows the
meta data up front and decodes the video again for every pass, so memory stays at about one frame whatever the length.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from .ffmpeg import DecodedFrame, FFmpegDecoder


class FrameSource:
    """Re-iterable frames plus their meta data."""

    indices: np.ndarray
    timestamps: np.ndarray
    image_size: tuple[int, int]  # (width, height)

    def __len__(self) -> int:
        return int(len(self.indices))

    def iter_frames(self) -> Iterator[DecodedFrame]:  # pragma: no cover - interface
        raise NotImplementedError


class ListFrameSource(FrameSource):
    """Frames that are already in memory (tests, short clips)."""

    def __init__(self, frames: Sequence[DecodedFrame]) -> None:
        self._frames = list(frames)
        self.indices = np.asarray([frame.index for frame in self._frames])
        self.timestamps = np.asarray([frame.timestamp_s for frame in self._frames], dtype=np.float64)
        self.image_size = ((self._frames[0].rgb.shape[1], self._frames[0].rgb.shape[0]) if self._frames else (0, 0))

    def iter_frames(self) -> Iterator[DecodedFrame]:
        return iter(self._frames)


class VideoFrameSource(FrameSource):
    """A video file decoded with ffmpeg on every pass."""

    def __init__(self, path: Path, width: int, height: int, frame_rate: float | None = None,
                 decoder: FFmpegDecoder | None = None) -> None:
        self.path, self.width, self.height, self.frame_rate = Path(path), int(width), int(height), frame_rate
        self._decoder = decoder or FFmpegDecoder()
        self.image_size = (self.width, self.height)
        try:
            times = self._decoder.timestamps(self.path)
        except RuntimeError as exc:
            if "ffprobe was not found" not in str(exc):
                raise
            times = []
        self._probe_times = times or None
        if not times:  # no ffprobe: count by decoding once (still O(1) memory)
            count = sum(1 for _ in self._decoder.decode_rgb(self.path, self.width, self.height, frame_rate, []))
            times = [self._fallback_time(i) for i in range(count)]
        self.indices = np.arange(len(times))
        self.timestamps = np.asarray(times, dtype=np.float64)

    def _fallback_time(self, index: int) -> float:
        return float(index) / self.frame_rate if self.frame_rate and self.frame_rate > 0 else float("nan")

    def iter_frames(self) -> Iterator[DecodedFrame]:
        decoded = 0
        for frame in self._decoder.decode_rgb(self.path, self.width, self.height, self.frame_rate,
                                              self._probe_times if self._probe_times is not None else []):
            decoded += 1
            yield frame
        if decoded != len(self.indices):  # ffprobe and ffmpeg disagree by a frame or two: follow the decoder
            self.indices = np.arange(decoded)
            times = list(self.timestamps[:decoded])
            times += [self._fallback_time(i) for i in range(len(times), decoded)]
            self.timestamps = np.asarray(times, dtype=np.float64)


def as_frame_source(frames: "FrameSource | Sequence[DecodedFrame]") -> FrameSource:
    return frames if isinstance(frames, FrameSource) else ListFrameSource(frames)


def frame_meta(frames: "FrameSource | Sequence[DecodedFrame]") -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """(frame indices, timestamps in s, (width, height)) of a source or a list of decoded frames."""
    source = as_frame_source(frames)
    return source.indices, source.timestamps, source.image_size
