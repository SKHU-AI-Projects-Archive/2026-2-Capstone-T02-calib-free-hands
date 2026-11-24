from calibfreehands.pipeline.worker import JobState, ProcessingJob
from calibfreehands.pipeline.qt_worker import JobThread


def test_job_can_complete_and_cancel():
    job = ProcessingJob("sample.mp4", steps=3)
    updates = list(job.run())
    assert job.state is JobState.COMPLETED
    assert updates[-1].progress == 1.0
    cancelled = ProcessingJob("sample.mp4", steps=3)
    cancelled.cancel()
    list(cancelled.run())
    assert cancelled.state is JobState.CANCELLED


def test_job_operation_runs_once():
    calls = []
    updates = list(ProcessingJob("sample.mp4", operation=lambda: calls.append("processed")).run())
    assert calls == ["processed"]
    assert updates[0].progress == 0.0
    assert updates[-1].state is JobState.COMPLETED


def test_job_operation_result_is_available_after_run():
    result = object()
    job = ProcessingJob("sample.mp4", operation=lambda: result)

    list(job.run())

    assert job.result is result


def test_qt_job_emits_failed_state(qapp):
    thread = JobThread(ProcessingJob("sample.mp4", operation=lambda: (_ for _ in ()).throw(RuntimeError("boom"))))
    updates = []
    thread.update.connect(updates.append)
    thread.start()
    assert thread.wait(2000)
    qapp.processEvents()
    assert thread.job.state is JobState.FAILED
    assert updates[-1].state is JobState.FAILED
