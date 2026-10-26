from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from .worker import JobState, JobUpdate, ProcessingJob


class JobThread(QThread):
    update = Signal(object)
    failed = Signal(str)

    def __init__(self, job: ProcessingJob) -> None:
        super().__init__()
        self.job = job

    def run(self) -> None:
        try:
            self.job._report = self.update.emit
            for event in self.job.run():
                self.update.emit(event)
        except InterruptedError:
            self.job.state = JobState.CANCELLED
            self.update.emit(JobUpdate(JobState.CANCELLED, 0.0, 0, 1, 0.0))
        except Exception as exc:
            self.job.state = JobState.FAILED
            self.update.emit(JobUpdate(JobState.FAILED, 0.0, 0, 1, 0.0))
            self.failed.emit(str(exc))

    def cancel(self) -> None:
        self.job.cancel()
