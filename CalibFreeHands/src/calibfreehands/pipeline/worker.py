from __future__ import annotations

import time
import inspect
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class JobState(Enum):
    WAITING = "Waiting"
    PROCESSING = "Processing"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


@dataclass(frozen=True)
class JobUpdate:
    state: JobState
    progress: float
    current: int
    total: int
    elapsed_s: float
    stage: str = ""


class ProcessingJob:
    def __init__(self, input_name: str, steps: int = 1, operation: Callable[["ProcessingJob"], object] | None = None) -> None:
        self.input_name, self.steps = input_name, max(1, steps)
        self.operation = operation
        self.state = JobState.WAITING
        self._cancelled = False
        self._report = None

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def report(self, current: int, total: int, elapsed_s: float | None = None, stage: str = "") -> None:
        if self._report is not None:
            self._report(JobUpdate(self.state, current / max(1, total), current, total, elapsed_s or 0.0, stage))

    def run(self):
        started = time.monotonic()
        self.state = JobState.PROCESSING
        if self.operation is not None:
            yield JobUpdate(self.state, 0.0, 0, 1, time.monotonic() - started)
            if len(inspect.signature(self.operation).parameters) == 0:
                self.operation()
            else:
                self.operation(self)
            self.state = JobState.COMPLETED
            yield JobUpdate(self.state, 1.0, 1, 1, time.monotonic() - started)
            return
        for current in range(1, self.steps + 1):
            if self._cancelled:
                self.state = JobState.CANCELLED
                yield JobUpdate(self.state, (current - 1) / self.steps, current - 1, self.steps, time.monotonic() - started)
                return
            time.sleep(0)
            yield JobUpdate(self.state, current / self.steps, current, self.steps, time.monotonic() - started)
        self.state = JobState.COMPLETED
        yield JobUpdate(self.state, 1.0, self.steps, self.steps, time.monotonic() - started)
