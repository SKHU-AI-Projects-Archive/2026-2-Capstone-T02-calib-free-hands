from __future__ import annotations

import time
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


class ProcessingJob:
    def __init__(self, input_name: str, steps: int = 1, operation: Callable[[], object] | None = None) -> None:
        self.input_name, self.steps = input_name, max(1, steps)
        self.operation = operation
        self.state = JobState.WAITING
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self):
        started = time.monotonic()
        self.state = JobState.PROCESSING
        if self.operation is not None:
            yield JobUpdate(self.state, 0.0, 0, 1, time.monotonic() - started)
            self.operation()
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
