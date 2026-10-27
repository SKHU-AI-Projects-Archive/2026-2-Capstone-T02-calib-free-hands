from __future__ import annotations

import importlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QListWidgetItem, QPushButton

from calibfreehands.app.settings import AppSettings
from calibfreehands.pipeline import CanonicalResult, save_result


def make_window(tmp_path, monkeypatch):
    main = importlib.import_module("calibfreehands.app.main")
    settings_path = tmp_path / "settings.ini"
    monkeypatch.setattr(main, "AppSettings", lambda: AppSettings(settings_path))
    return main, main.MainWindow(), settings_path


def wait_for_job(window, qapp):
    thread = window._job_thread
    assert thread is not None
    assert thread.wait(5000)
    qapp.processEvents()
    return thread


def queue_path(item):
    return Path(item.data(Qt.ItemDataRole.UserRole))


def _synthetic_session(working_directory, configurations, settings):
    final_results, calibrations, poses = {}, {}, {}
    for config in configurations:
        points2d = np.full((1, 2, 21, 2), np.nan, np.float32)
        points3d = np.full((1, 2, 21, 3), np.nan, np.float32)
        points2d[0, 0] = 0; points3d[0, 0] = 1
        result = CanonicalResult(np.array([0]), np.array([0.]), points2d, points3d,
                                 np.array([[True, False]]), np.array([500.]), (8, 8))
        result_dir = working_directory / config.id
        provenance = {"input_path": settings["input_path"], "input_sha256": settings["source_media_sha256"],
                      "coordinate_system": "camera_space", "unit": "metre",
                      "applied_camera_parameters": {"principal_point_applied": False, "distortion_applied": False}}
        npz_path, run_path = save_result(result, result_dir, provenance)
        calibration_id = f"calibration-{config.id}"
        calibrations[calibration_id] = {"metadata": {"backend": config.calibration_model_id}}
        poses[config.reconstruction_model_id] = {"checkpoint_sha256": "b" * 64, "mano_assets": {}}
        final_results[config.id] = {
            "calibration_result_id": calibration_id,
            "path": npz_path.relative_to(working_directory).as_posix(),
            "run_json": run_path.relative_to(working_directory).as_posix(), "focal_px": 500.0,
        }
    return SimpleNamespace(configurations=list(configurations), final_results=final_results,
                           calibration_results=calibrations, raw_pose_results=poses,
                           reference_configuration_id=None)


def test_main_surface_has_one_open_button_without_dropdown(tmp_path, monkeypatch, qapp):
    main, window, _ = make_window(tmp_path, monkeypatch)
    buttons = window.findChildren(QPushButton)
    labels = [button.text() for button in buttons]
    assert labels.count("Open...") == 1
    assert "Browse files" not in labels
    assert "Browse folder" not in labels
    assert window.open_button.menu() is None
    assert "View Results..." in labels
    window.close()


def test_open_files_adds_supported_paths_once_and_path_survives_state_text(tmp_path, monkeypatch, qapp):
    main, window, _ = make_window(tmp_path, monkeypatch)
    supported = tmp_path / "clip.mp4"
    unsupported = tmp_path / "notes.txt"
    supported.touch()
    unsupported.touch()
    class Picker:
        def __init__(self, *_): pass
        def exec(self): return 1
        def selected_paths(self): return [supported, unsupported]
    monkeypatch.setattr(main, "MediaOpenDialog", Picker)
    window._open_media()
    window._add_paths([supported.parent / "." / supported.name])

    assert window.queue.count() == 1
    item = window.queue.item(0)
    assert queue_path(item) == supported.resolve()
    assert item.text() == "[Waiting] clip.mp4"
    assert item.toolTip() == str(supported.resolve())
    window.close()


def test_open_folder_is_nonrecursive_and_skips_duplicate_input(tmp_path, monkeypatch, qapp):
    main, window, _ = make_window(tmp_path, monkeypatch)
    folder = tmp_path / "media"
    nested = folder / "nested"
    nested.mkdir(parents=True)
    direct = folder / "direct.mp4"
    nested_file = nested / "nested.mp4"
    direct.touch()
    nested_file.touch()
    class Picker:
        def __init__(self, *_): pass
        def exec(self): return 1
        def selected_paths(self): return [folder]
    monkeypatch.setattr(main, "MediaOpenDialog", Picker)
    window._open_media()
    window._add_paths([direct])

    assert [queue_path(window.queue.item(i)).name for i in range(window.queue.count())] == ["direct.mp4"]
    window.close()


def test_remove_and_clear_never_delete_source_files(tmp_path, monkeypatch, qapp):
    _, window, _ = make_window(tmp_path, monkeypatch)
    sources = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    for source in sources:
        source.touch()
    window._add_paths(sources)
    window.queue.item(0).setSelected(True)
    window.queue.item(1).setSelected(True)
    window.remove_button.click()
    assert window.queue.count() == 0
    assert sources[0].is_file() and sources[1].is_file()
    window._add_paths(sources)
    window.clear_button.click()
    assert window.queue.count() == 0
    assert all(source.is_file() for source in sources)
    window.close()


def fake_session(snapshot, output_dir):
    configurations = list(snapshot.configurations)
    return SimpleNamespace(
        state="completed",
        configurations=configurations,
        final_results={item.id: {"path": f"results-{item.id}.npz"} for item in configurations},
    )


def test_queue_runs_in_order_continues_after_failure_and_reports_states(tmp_path, monkeypatch, qapp):
    _, window, _ = make_window(tmp_path, monkeypatch)
    sources = [tmp_path / f"{name}.mp4" for name in "ABC"]
    for source in sources: source.touch()
    window._add_paths(sources)
    calls = []

    def process(path, job, snapshot):
        calls.append(path.name)
        if path.name == "B.mp4":
            raise RuntimeError("B failed")
        return tmp_path, fake_session(snapshot, tmp_path)

    monkeypatch.setattr(window, "_process_media", process)
    window._start_selected()
    wait_for_job(window, qapp)

    assert calls == ["A.mp4", "B.mp4", "C.mp4"]
    assert [window.queue.item(i).data(Qt.ItemDataRole.UserRole + 1) for i in range(3)] == [
        "Completed", "Failed", "Completed"
    ]
    assert "B failed" in window.queue.item(1).toolTip()
    assert window.progress.text().startswith("Completed with errors")
    window.close()


def test_cancel_marks_current_and_remaining_items_cancelled_without_processing_next(tmp_path, monkeypatch, qapp):
    _, window, _ = make_window(tmp_path, monkeypatch)
    sources = [tmp_path / f"{name}.mp4" for name in "ABC"]
    for source in sources: source.touch()
    window._add_paths(sources)
    calls = []

    def process(path, job, snapshot):
        calls.append(path.name)
        if path.name == "B.mp4":
            job.cancel()
            raise InterruptedError("cancelled in B")
        return tmp_path, fake_session(snapshot, tmp_path)

    monkeypatch.setattr(window, "_process_media", process)
    window._start_selected()
    wait_for_job(window, qapp)

    assert calls == ["A.mp4", "B.mp4"]
    assert [window.queue.item(i).data(Qt.ItemDataRole.UserRole + 1) for i in range(3)] == [
        "Completed", "Cancelled", "Cancelled"
    ]
    assert window.progress.text() == "Cancelled"
    window.close()


def test_processing_disables_queue_and_configuration_controls_and_restores_them(tmp_path, monkeypatch, qapp):
    _, window, _ = make_window(tmp_path, monkeypatch)
    source = tmp_path / "a.mp4"
    source.touch()
    window._add_paths([source])
    entered = threading.Event()
    release = threading.Event()
    snapshots = []

    def process(path, job, snapshot):
        snapshots.append(snapshot)
        entered.set()
        assert release.wait(3)
        return tmp_path, fake_session(snapshot, tmp_path)

    monkeypatch.setattr(window, "_process_media", process)
    window._start_selected()
    assert entered.wait(2)
    qapp.processEvents()
    assert not window.open_button.isEnabled()
    assert not window.remove_button.isEnabled()
    assert not window.clear_button.isEnabled()
    assert not window.remove_button.isEnabled()
    assert not window.clear_button.isEnabled()
    assert not window.start_button.isEnabled()
    assert window.cancel.isEnabled()
    assert not window.output_browse_button.isEnabled()
    assert not window.preset_combo.isEnabled()
    assert not window.sampling_combo.isEnabled()
    assert not window.configuration_rows[0].name.isEnabled()
    window.sampling_combo.setCurrentIndex(window.sampling_combo.findData("all"))
    release.set()
    wait_for_job(window, qapp)
    assert snapshots[0].sampling_mode == "uniform"
    assert snapshots[0].queue_paths == (source.resolve(),)
    assert window.open_button.isEnabled()
    assert window.start_button.isEnabled()
    assert window.clear_button.isEnabled()
    assert not window.cancel.isEnabled()
    window.close()


def test_media_progress_includes_queue_position_without_changing_stage_progress(tmp_path, monkeypatch, qapp):
    main, window, _ = make_window(tmp_path, monkeypatch)
    source = tmp_path / "sample.mp4"
    configuration = window.configuration_rows[0].configuration()
    snapshot = main.ProcessingSnapshot(
        (source,), (configuration,), "original", "uniform", 8, "median",
        tmp_path / "output", "auto_rename", False,
    )
    reports = []

    class Job:
        cancelled = False
        def report(self, current, total, stage=""):
            reports.append((current, total, stage))

    monkeypatch.setattr(main, "probe_media", lambda *_: SimpleNamespace(width=8, height=8, frame_rate=30.0))
    monkeypatch.setattr(main, "FFmpegDecoder", lambda: SimpleNamespace(decode_rgb=lambda *_: ["frame"]))
    source.touch()
    monkeypatch.setattr(main, "choose_output_file", lambda *_: tmp_path / "output" / "sample.cfh")

    def process(_frames, _configs, _config, _root, _output, _settings, progress, _cancelled):
        progress("Estimating focal", 2, 4)
        return _synthetic_session(_output, _configs, _settings)

    monkeypatch.setattr(main, "process_comparison_session", process)
    window._process_media(source, Job(), snapshot)

    assert reports[-1] == (2, 4, "1/1 sample.mp4: Estimating focal")
    window.close()


def test_media_processing_exposes_only_cfh_and_cleans_internal_workspace(tmp_path, monkeypatch, qapp):
    main, window, _ = make_window(tmp_path, monkeypatch)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"local source bytes")
    output = tmp_path / "output"
    output.mkdir()
    window.output_path.setText(str(output))
    window._add_configuration_row()
    working_paths, captured = [], {}
    monkeypatch.setattr(main, "probe_media", lambda *_: SimpleNamespace(width=8, height=8, frame_rate=30.0))
    monkeypatch.setattr(main, "FFmpegDecoder", lambda: SimpleNamespace(decode_rgb=lambda *_: ["frame"]))

    def process(_frames, configurations, _config, _root, working, settings, *_args):
        working_paths.append(working)
        captured.update(settings=settings, configurations=configurations)
        return _synthetic_session(working, configurations, settings)

    monkeypatch.setattr(main, "process_comparison_session", process)
    paths, session = window._process_media(source)

    assert len(paths) == 2 and session is not None
    assert all(path.suffix == ".cfh" and path.is_file() for path in paths)
    assert {path.name for path in output.iterdir()} == {path.name for path in paths}
    assert all(not path.exists() for path in working_paths)
    assert captured["settings"]["create_preview"] is False
    assert len({config.reconstruction_model_id for config in captured["configurations"]}) == 1
    from calibfreehands.pipeline.result_bundle import load_result_bundle
    assert all(load_result_bundle(path).metadata["source_media_sha256"] == main.sha256_file(source) for path in paths)
    window.close()


@pytest.mark.parametrize("policy", ["ask", "skip"])
def test_existing_result_collision_never_overwrites_silently_or_runs_inference(tmp_path, monkeypatch, qapp, policy):
    main, window, _ = make_window(tmp_path, monkeypatch)
    source = tmp_path / "clip.mp4"; source.touch()
    output = tmp_path / "output"; output.mkdir()
    window.output_path.setText(str(output))
    config = window.configuration_rows[0].configuration()
    filename = main.result_filename("clip", "Original AnyCalib", "Original WiLoR")
    existing = output / filename; existing.write_bytes(b"preserve existing result")
    snapshot = main.ProcessingSnapshot((source,), (config,), "original", "uniform", 8, "median",
                                       output, policy, False)
    monkeypatch.setattr(main, "probe_media", lambda *_: SimpleNamespace(width=8, height=8, frame_rate=30.0))
    monkeypatch.setattr(main, "FFmpegDecoder", lambda: SimpleNamespace(decode_rgb=lambda *_: ["frame"]))
    monkeypatch.setattr(main, "process_comparison_session", lambda *_: pytest.fail("collision must be resolved before inference"))
    if policy == "ask":
        with pytest.raises(FileExistsError, match="confirmation"):
            window._process_media(source, selection=snapshot)
    else:
        assert window._process_media(source, selection=snapshot) == ((), None)
    assert existing.read_bytes() == b"preserve existing result"
    window.close()


def test_cancellation_is_checked_while_decoding_media(tmp_path, monkeypatch, qapp):
    main, window, _ = make_window(tmp_path, monkeypatch)
    source = tmp_path / "sample.mp4"
    source.touch()
    yielded = []

    class Job:
        cancelled = False
        def report(self, *_args, **_kwargs): pass

    job = Job()

    def decode(*_):
        for value in range(3):
            yielded.append(value)
            if value == 1:
                job.cancelled = True
            yield value

    monkeypatch.setattr(main, "probe_media", lambda *_: SimpleNamespace(width=8, height=8, frame_rate=30.0))
    monkeypatch.setattr(main, "FFmpegDecoder", lambda: SimpleNamespace(decode_rgb=decode))
    monkeypatch.setattr(main, "process_comparison_session", lambda *_: pytest.fail("inference must not start after cancellation"))

    with pytest.raises(InterruptedError, match="cancelled during media decoding"):
        window._process_media(source, job)

    assert yielded == [0, 1]
    window.close()


@pytest.mark.parametrize("configuration_count, expected", [(1, "result"), (2, "comparison")])
def test_single_media_success_auto_opens_matching_viewer(tmp_path, monkeypatch, qapp, configuration_count, expected):
    main, window, _ = make_window(tmp_path, monkeypatch)
    source = tmp_path / "a.mp4"
    source.touch()
    window._add_paths([source])
    if configuration_count == 2:
        window._add_configuration_row()
    opened = []

    class Viewer:
        viewer_type = "unknown"
        def __init__(self, value):
            opened.append((self.viewer_type, value, QThread.currentThread() == qapp.thread()))
        def setAttribute(self, *_): pass
        def show(self): pass

    class ResultStub(Viewer): viewer_type = "result"
    class ComparisonStub(Viewer): viewer_type = "comparison"
    monkeypatch.setattr(main, "ResultViewer", ResultStub)
    monkeypatch.setattr(main, "ComparisonViewer", ComparisonStub)
    monkeypatch.setattr(main, "load_result_bundle", lambda path: SimpleNamespace(metadata={"source_media_sha256": "a" * 64}))
    monkeypatch.setattr(window, "_process_media", lambda path, job, snapshot: (
        (tmp_path / "out" / "a.cfh",) if configuration_count == 1 else
        (tmp_path / "out" / "a.cfh", tmp_path / "out" / "b.cfh"), fake_session(snapshot, tmp_path / "out")
    ))

    window._start_selected()
    wait_for_job(window, qapp)

    assert len(opened) == 1
    assert opened[0][0] == expected
    assert opened[0][2]
    if expected == "result":
        assert not isinstance(opened[0][1], list)
    else:
        assert len(opened[0][1]) == 2
    assert window.progress.text() == "Completed"
    window.close()


def test_multi_media_success_does_not_auto_open_a_viewer(tmp_path, monkeypatch, qapp):
    main, window, _ = make_window(tmp_path, monkeypatch)
    sources = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    for source in sources: source.touch()
    window._add_paths(sources)
    opened = []
    monkeypatch.setattr(main, "ResultViewer", lambda path: opened.append(path))
    monkeypatch.setattr(main, "ComparisonViewer", lambda path: opened.append(path))
    monkeypatch.setattr(window, "_process_media", lambda path, job, snapshot: (
        (tmp_path / path.stem / "result.cfh",), fake_session(snapshot, tmp_path / path.stem)
    ))

    window._start_selected()
    wait_for_job(window, qapp)

    assert opened == []
    assert window.progress.text() == "Completed"
    window.close()


def test_output_path_is_concrete_browse_persists_and_file_path_blocks_processing(tmp_path, monkeypatch, qapp):
    main, window, settings_path = make_window(tmp_path, monkeypatch)
    assert window.output_path.text()
    assert window.output_path.isReadOnly()
    selected = tmp_path / "chosen output"
    selected.mkdir()
    monkeypatch.setattr(main.QFileDialog, "getExistingDirectory", lambda *_: str(selected))
    window.output_browse_button.click()
    assert window.output_path.text() == str(selected)
    assert AppSettings(settings_path).get("output_directory") == str(selected)
    window.close()

    restored_main, restored, _ = make_window(tmp_path, monkeypatch)
    assert restored.output_path.text() == str(selected)
    invalid = tmp_path / "not-a-folder"
    invalid.touch()
    restored.output_path.setText(str(invalid))
    source = tmp_path / "clip.mp4"
    source.touch()
    restored._add_paths([source])
    restored._start_selected()
    assert restored._job_thread is None
    assert restored.progress.text().startswith("Failed: Output path is a file")
    restored.close()


def test_unusable_output_base_fails_batch_before_any_media_starts(tmp_path, monkeypatch, qapp):
    _, window, _ = make_window(tmp_path, monkeypatch)
    source = tmp_path / "clip.mp4"
    source.touch()
    window._add_paths([source])
    calls = []

    def reject_output(_path):
        raise PermissionError("output directory is not writable")

    monkeypatch.setattr(window, "_prepare_output_directory", reject_output, raising=False)
    monkeypatch.setattr(window, "_process_media", lambda *args: calls.append(args))

    window._start_selected()

    assert window._job_thread is None
    assert calls == []
    assert window.progress.text().startswith("Failed: output directory is not writable")
    assert window.queue.item(0).data(Qt.ItemDataRole.UserRole + 1) == "Waiting"
    window.close()


def test_reference_focal_container_visibility_and_value_are_preserved(tmp_path, monkeypatch, qapp):
    _, window, _ = make_window(tmp_path, monkeypatch)
    row = window.configuration_rows[0]
    assert not row.reference_focal_controls.isVisibleTo(window)
    row.calibration.setCurrentIndex(row.calibration.findData("checkerboard_reference"))
    assert row.reference_focal_controls.isVisibleTo(window)
    assert row.reference_focal.isEnabled()
    row.reference_focal.setValue(1400.0)
    row.calibration.setCurrentIndex(row.calibration.findData("anycalib_original"))
    assert not row.reference_focal_controls.isVisibleTo(window)
    row.calibration.setCurrentIndex(row.calibration.findData("checkerboard_reference"))
    assert row.reference_focal_controls.isVisibleTo(window)
    assert row.reference_focal.value() == 1400.0
    window.close()
