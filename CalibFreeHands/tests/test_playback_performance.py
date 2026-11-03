import numpy as np
import pytest
from PySide6.QtWidgets import QWidget


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


def test_playback_clock_tracks_source_timestamps_and_drops_stale_frames():
    from calibfreehands.visualization.playback import PlaybackClock

    now = FakeClock()
    playback = PlaybackClock(np.arange(60) / 30, fps=30, monotonic=now)
    playback.play(0)
    now.value = 0.5
    assert playback.frame_at() == 15
    now.value = 1.0
    assert playback.frame_at() == 30


def test_playback_clock_handles_irregular_and_invalid_timestamps():
    from calibfreehands.visualization.playback import PlaybackClock

    now = FakeClock()
    playback = PlaybackClock(np.array([2.0, 2.1, 2.5, 2.6]), fps=30, monotonic=now)
    playback.play(0)
    now.value = 0.55
    assert playback.frame_at() == 2

    fallback = PlaybackClock(np.array([0.0, 0.1, 0.1, np.nan]), fps=10, monotonic=now)
    assert np.diff(fallback.timestamps).tolist() == pytest.approx([0.1, 0.1, 0.1])
    all_invalid = PlaybackClock(np.full(3, np.nan), fps=12, monotonic=now)
    assert all_invalid.frame_duration == pytest.approx(1 / 12)
    single_invalid_fps = PlaybackClock(np.array([0.0]), fps=np.nan, monotonic=now)
    assert single_invalid_fps.frame_duration == pytest.approx(1 / 30)


def test_playback_pause_resume_and_speed_change_are_continuous():
    from calibfreehands.visualization.playback import PlaybackClock

    now = FakeClock()
    playback = PlaybackClock(np.arange(300) / 30, fps=30, monotonic=now)
    playback.play(0)
    now.value = 2.0
    playback.set_speed(0.5)
    assert playback.source_time_at() == pytest.approx(2.0)
    now.value = 4.0
    assert playback.frame_at() == 90
    playback.pause()
    now.value = 20.0
    assert playback.frame_at() == 90
    playback.resume()
    now.value = 22.0
    assert playback.frame_at() == 120


def test_comparison_frame_updates_reuse_hud_and_update_only_active_renderer(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer
    from tests.test_comparison_viewer import _session

    viewer = ComparisonViewer(_session(tmp_path))
    panel = viewer.overlay_panel
    children = panel.body.findChildren(QWidget)
    identity = tuple(map(id, children))
    calls = {"overlay": 0, "split": 0}
    overlay_set_frame = viewer.overlay_view.set_frame
    split_set_frame = viewer.split_views[0].set_frame
    viewer.overlay_view.set_frame = lambda frame: (calls.__setitem__("overlay", calls["overlay"] + 1), overlay_set_frame(frame))
    viewer.split_views[0].set_frame = lambda frame: (calls.__setitem__("split", calls["split"] + 1), split_set_frame(frame))

    for frame in range(100):
        viewer.slider.setValue((frame + 1) % (viewer.slider.maximum() + 1))
    assert tuple(map(id, panel.body.findChildren(QWidget))) == identity
    assert calls["overlay"] == 100
    assert calls["split"] == 0

    viewer.set_mode("split")
    calls.update(overlay=0, split=0)
    viewer.slider.setValue((viewer.slider.value() + 1) % (viewer.slider.maximum() + 1))
    assert calls["overlay"] == 0
    assert calls["split"] == 1
    viewer.close()


def test_disabled_annotations_and_error_vectors_skip_projection_and_vector_work(tmp_path, qapp, monkeypatch):
    from calibfreehands.visualization.comparison import ComparisonViewer
    from tests.test_comparison_viewer import _session

    viewer = ComparisonViewer(_session(tmp_path))
    projection_calls = []
    monkeypatch.setattr(viewer.overlay_annotations, "projected_annotations", lambda: projection_calls.append(1) or [])
    annotation_updates = []
    monkeypatch.setattr(viewer.overlay_annotations, "update", lambda: annotation_updates.append(1))
    vector_calls = []
    monkeypatch.setattr(viewer.overlay_view, "_update_error_vectors", lambda: vector_calls.append(1))
    viewer._set_frame(1)
    assert not projection_calls
    assert not annotation_updates
    assert not vector_calls
    viewer.close()


def test_playback_timer_drops_frames_to_clock_target_and_speed_choices_are_available(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer
    from tests.test_comparison_viewer import _session

    viewer = ComparisonViewer(_session(tmp_path))
    now = FakeClock()
    viewer.playback_clock._monotonic = now
    viewer.playback_clock.play(0)
    viewer.timer.start()
    now.value = 0.25
    viewer._advance()
    assert viewer.slider.value() == 2
    assert viewer._stats_skipped == 1
    assert viewer.speed_value_label.text() == "1.0×"
    assert viewer.speed_down_button.isEnabled()
    assert viewer.speed_up_button.isEnabled()
    viewer._play()
    viewer.close()


def test_hidden_split_result_does_not_receive_frame_or_hud_updates(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer
    from tests.test_comparison_viewer import _session

    viewer = ComparisonViewer(_session(tmp_path))
    viewer.set_mode("split")
    calls = [0, 0, 0]
    hud_calls = [0, 0, 0]
    for index, view in enumerate(viewer.split_views):
        original = view.set_frame
        view.set_frame = lambda frame, i=index, fn=original: (calls.__setitem__(i, calls[i] + 1), fn(frame))
        panel = viewer.split_panels[index]
        original_update = panel.update_values
        panel.update_values = lambda owner, i=index, fn=original_update: (hud_calls.__setitem__(i, hud_calls[i] + 1), fn(owner))
    viewer.visibility_checks[1].setChecked(False)
    calls[:], hud_calls[:] = [0, 0, 0], [0, 0, 0]
    viewer.slider.setValue(1)
    assert calls == [1, 0, 1]
    assert hud_calls == [1, 0, 1]
    viewer.close()


def test_disabled_all_geometry_skips_coordinate_transform(tmp_path, qapp, monkeypatch):
    import calibfreehands.visualization.viewer as viewer_module
    from calibfreehands.visualization.comparison import ComparisonViewer
    from tests.test_comparison_viewer import _session

    viewer = ComparisonViewer(_session(tmp_path))
    view = viewer.split_views[0]
    view.show_joints = view.show_bones = view.show_mesh = False

    class NoTransform:
        @property
        def T(self):
            raise AssertionError("disabled geometry must not transform points")

    monkeypatch.setattr(viewer_module, "RAW_TO_VIEW", NoTransform())
    view.set_frame(1)
    viewer.close()


def test_overlay_result_visibility_restores_reused_geometry(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer
    from tests.test_comparison_viewer import _session

    viewer = ComparisonViewer(_session(tmp_path))
    items = viewer.overlay_view.extra_items[0][0]
    viewer.visibility_checks[1].setChecked(False)
    assert not any(item.visible() for item in items)
    viewer.visibility_checks[1].setChecked(True)
    assert items[0].visible()
    viewer.close()


def test_hud_frame_value_updates_keep_label_widgets_reused(tmp_path, qapp):
    from PySide6.QtWidgets import QScrollArea
    from calibfreehands.visualization.comparison import ComparisonViewer
    from tests.test_comparison_viewer import _session

    viewer = ComparisonViewer(_session(tmp_path, count=5))
    viewer.show()
    qapp.processEvents()
    panel = viewer.overlay_panel
    labels = dict(panel.metric_rows)
    viewer.slider.setValue(1)
    assert panel.metric_rows == labels
    viewer.confidence_numeric_checkbox.setChecked(True)
    qapp.processEvents()
    assert not panel.findChildren(QScrollArea)
    viewer.close()


def test_hud_resize_positioning_does_not_reapply_stylesheet(tmp_path, qapp):
    from calibfreehands.visualization.comparison import ComparisonViewer
    from tests.test_comparison_viewer import _session

    viewer = ComparisonViewer(_session(tmp_path))
    panel = viewer.overlay_panel
    calls = []
    panel.card.setStyleSheet = lambda _style: calls.append(1)
    for _ in range(10):
        panel._position()
    assert not calls
    viewer.close()
