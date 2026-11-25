"""\"Ask\" for results that already exist: the question comes before the run, the run never stops with an error.

Regression: adding a focal length to a video that was processed before aborted the whole run with
"result already exists; confirmation required" (the old results of the other focal lengths were in the way).
"""

import importlib
from types import SimpleNamespace

import pytest

from calibfreehands.app.settings import AppSettings
from calibfreehands.export.paths import result_filename


class FakeSignal:
    def connect(self, _slot):
        pass


class FakeThread:
    instances = []

    def __init__(self, job):
        self.job = job
        self.started = False
        self.update = self.item_state = self.failed = self.finished = FakeSignal()
        FakeThread.instances.append(self)

    def start(self):
        self.started = True

    def isRunning(self):
        return False


@pytest.fixture
def setup(tmp_path, monkeypatch, qapp):
    main = importlib.import_module("calibfreehands.app.main")
    FakeThread.instances.clear()
    monkeypatch.setattr(main, "JobThread", FakeThread)
    window = main.MainWindow(AppSettings(tmp_path / "settings.ini"))
    out = tmp_path / "out"
    out.mkdir()
    window.output_path.setText(str(out))
    clip = tmp_path / "clip.mp4"
    clip.touch()
    window.add_media([clip])
    window.use_manual_focal(900.0, "A 900")
    window._add_configuration_row()
    second = window.configuration_rows[1]
    second.name.setText("B 1300")
    second.reference_focal.setValue(1300.0)
    window._model_or_processing_changed()
    yield SimpleNamespace(main=main, window=window, out=out, clip=clip)
    window.close()


def existing_900(env):
    path = env.out / result_filename("clip", "Manual Focal", "Original WiLoR", 900.0)
    path.write_bytes(b"an earlier result")
    return path


def test_default_policy_is_ask(setup):
    assert setup.window.collision_combo.currentData() == "ask"


def test_only_results_this_run_would_write_are_listed(setup):
    old = existing_900(setup)
    (setup.out / "unrelated.cfh").write_bytes(b"x")
    snapshot = setup.window._capture_processing_snapshot()
    assert setup.window._existing_results(snapshot) == [old]


def test_run_with_old_and_new_focal_asks_and_skips_the_old_one(setup):
    old = existing_900(setup)
    asked = []
    setup.window._ask_collision_action = lambda existing: asked.append(existing) or "skip"

    setup.window._start_selected()

    assert asked == [[old]]
    assert setup.window._active_snapshot.collision_policy == "skip"  # for this run only
    assert setup.window._skipped_existing == [old]
    assert setup.window.collision_combo.currentData() == "ask"  # the setting itself is untouched
    assert len(FakeThread.instances) == 1 and FakeThread.instances[0].started


@pytest.mark.parametrize("answer", ["overwrite", "auto_rename"])
def test_other_answers_become_the_policy_of_the_run(setup, answer):
    existing_900(setup)
    setup.window._ask_collision_action = lambda existing: answer
    setup.window._start_selected()
    assert setup.window._active_snapshot.collision_policy == answer
    assert setup.window._skipped_existing == []


def test_cancel_starts_nothing(setup):
    existing_900(setup)
    setup.window._ask_collision_action = lambda existing: None
    setup.window._start_selected()
    assert FakeThread.instances == [] and not setup.window.is_processing
    assert setup.window.progress.text() == "Cancelled"


def test_no_question_without_existing_results_or_with_an_explicit_policy(setup):
    setup.window._ask_collision_action = lambda existing: pytest.fail("nothing exists: no question")
    setup.window._start_selected()
    assert FakeThread.instances[-1].started and setup.window._active_snapshot.collision_policy == "ask"

    existing_900(setup)
    setup.window._job_thread = None
    setup.window._processing = False
    setup.window.collision_combo.setCurrentIndex(setup.window.collision_combo.findData("overwrite"))
    setup.window._start_selected()  # the policy was chosen in advance: no question either
    assert setup.window._active_snapshot.collision_policy == "overwrite"


def test_kept_results_of_the_same_video_open_together_with_the_new_ones(setup, monkeypatch, tmp_path):
    kept_same, kept_other = tmp_path / "same.cfh", tmp_path / "other.cfh"
    setup.window._skipped_existing = [kept_same, kept_other]
    bundles = {
        kept_same: SimpleNamespace(metadata={"source_media_sha256": "ABC"}),
        kept_other: SimpleNamespace(metadata={"source_media_sha256": "different"}),
    }
    monkeypatch.setattr(setup.main, "load_result_bundle", lambda path: bundles[path])
    new = [SimpleNamespace(metadata={"source_media_sha256": "abc"})]

    assert setup.window._same_source_existing(new) == [bundles[kept_same]]
    assert setup.window._same_source_existing([]) == []  # nothing new: nothing to add to
