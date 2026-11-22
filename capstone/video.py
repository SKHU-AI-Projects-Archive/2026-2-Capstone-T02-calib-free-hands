from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

from calibfreehands.media.ffmpeg import DecodedFrame, FFmpegDecoder
from calibfreehands.media.probe import probe_media


@dataclass(frozen=True)
class Recording:
    path: Path
    size: tuple[int, int]
    pts: tuple[Fraction, ...]
    decode: Callable[[], Iterator[DecodedFrame]]

    @classmethod
    def open(cls, path: Path) -> Recording:
        path = path.resolve(strict=True)
        decoder = FFmpegDecoder()
        info = probe_media(path)
        if not info.width or not info.height:
            raise ValueError("Could not determine video dimensions")
        points = decoder.exact_timestamps(path)
        if len(points) < 2:
            raise ValueError("Video requires decoded frame timestamps from FFprobe")
        if any(right < left for left, right in zip(points, points[1:])):
            raise ValueError("Decoded video timestamps are not monotonic")
        return cls(path, (info.width, info.height), points, lambda: decoder.decode_rgb(
            path, info.width, info.height, timestamps=tuple(map(float, points)),
        ))

    @property
    def times(self) -> tuple[Fraction, ...]:
        return tuple(t - self.pts[0] for t in self.pts)

    def frames(self) -> Iterator[DecodedFrame]:
        raw = self.decode()
        try:
            for index, frame in enumerate(raw):
                if index >= len(self.pts) or frame.index != index:
                    raise ValueError("Decoded frames do not match the timestamp index")
                if frame.rgb.shape != (self.size[1], self.size[0], 3):
                    raise ValueError("Decoded frame dimensions changed")
                yield replace(frame, timestamp_s=float(self.pts[index] - self.pts[0]))
        finally:
            raw.close()
