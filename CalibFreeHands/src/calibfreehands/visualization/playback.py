from __future__ import annotations

from collections.abc import Callable

import numpy as np


class PlaybackClock:
    """Map monotonic wall time onto a canonical result's source timestamps."""

    SPEEDS = (0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)

    def __init__(self, timestamps, fps: float, monotonic: Callable[[], float]):
        values = np.asarray(timestamps, dtype=np.float64)
        if values.ndim != 1 or not values.size:
            raise ValueError("playback requires at least one timestamp")
        safe_fps = fps if np.isfinite(fps) and fps > 0 else 30.0
        valid = np.isfinite(values).all() and (values.size == 1 or np.all(np.diff(values) > 0))
        if valid:
            self.timestamps = values.copy()
            differences = np.diff(values)
            self.frame_duration = float(np.median(differences)) if differences.size else 1.0 / safe_fps
        else:
            differences = np.diff(values)
            usable = differences[np.isfinite(differences) & (differences > 0)]
            self.frame_duration = float(np.median(usable)) if usable.size else 1.0 / safe_fps
            origin = float(values[0]) if np.isfinite(values[0]) else 0.0
            self.timestamps = origin + np.arange(values.size, dtype=np.float64) * self.frame_duration
        self._monotonic = monotonic
        self.speed = 1.0
        self.playing = False
        self._wall_origin = 0.0
        self._source_origin = float(self.timestamps[0])

    @property
    def duration(self) -> float:
        return max(self.frame_duration, float(self.timestamps[-1] - self.timestamps[0] + self.frame_duration))

    def play(self, frame: int) -> None:
        self._source_origin = float(self.timestamps[min(max(int(frame), 0), len(self.timestamps) - 1)])
        self._wall_origin = self._monotonic()
        self.playing = True

    def pause(self, frame: int | None = None) -> None:
        if self.playing:
            self._source_origin = (float(self.timestamps[min(max(int(frame), 0), len(self.timestamps) - 1)])
                                   if frame is not None else self.source_time_at())
            self.playing = False

    def resume(self) -> None:
        if not self.playing:
            self._wall_origin = self._monotonic()
            self.playing = True

    def set_speed(self, speed: float) -> None:
        if speed not in self.SPEEDS:
            raise ValueError(f"unsupported playback speed: {speed}")
        if self.playing:
            self._source_origin = self.source_time_at()
            self._wall_origin = self._monotonic()
        self.speed = float(speed)

    def source_time_at(self) -> float:
        if not self.playing:
            return self._source_origin
        elapsed = max(0.0, self._monotonic() - self._wall_origin)
        offset = (self._source_origin - self.timestamps[0] + elapsed * self.speed) % self.duration
        return float(self.timestamps[0] + offset)

    def frame_at(self) -> int:
        source_time = self.source_time_at()
        frame = int(np.searchsorted(self.timestamps, source_time, side="right") - 1)
        return min(max(frame, 0), len(self.timestamps) - 1)
