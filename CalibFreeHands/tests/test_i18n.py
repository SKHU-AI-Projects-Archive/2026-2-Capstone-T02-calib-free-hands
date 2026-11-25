"""Korean user interface: the Language box translates both pages at once, nothing breaks when switching."""

from types import SimpleNamespace

import pytest

from calibfreehands.app import i18n
from calibfreehands.app.settings import AppSettings
from test_capture_page import FakeCamera, FakeController


@pytest.fixture(autouse=True)
def english_afterwards():
    yield
    i18n.set_language("en")


@pytest.fixture
def shell(tmp_path, qapp):
    from calibfreehands.app.shell import AppShell

    window = AppShell(AppSettings(tmp_path / "settings.ini"), controller_factory=FakeController,
                      camera_factory=FakeCamera)
    yield window
    window.capture._clock.stop()
    window.close()


def switch(shell, text):
    box = shell.processing.language_combo
    box.setCurrentText(text)


def test_tr_exact_and_pattern_texts():
    assert i18n.tr("Start Processing") == "Start Processing"  # English is the default
    i18n.set_language("ko")
    assert i18n.tr("Start Processing") == "처리 시작"
    assert i18n.tr("Completed with errors (2 failed)") == "오류와 함께 완료(2개 실패)"
    assert i18n.tr("Processing — 1/1 clip.mp4: Reading video (1/1, 100%)") == \
        "처리 중 — 1/1 clip.mp4: 영상 읽는 중 (1/1, 100%)"
    assert i18n.tr("600 frames  ·  20.0 s  ·  0 gaps") == "600프레임  ·  20.0초  ·  gap 0개"
    assert i18n.tr("The hand was found in only 2 of 90 frames. Record the calibration video again: keep the whole "
                   "right hand in view and move it.").startswith("90프레임 중 2프레임에서만")
    assert i18n.tr("a text that has no translation") == "a text that has no translation"
    assert i18n.tr("") == ""


def test_switching_to_korean_and_back_translates_both_pages(shell):
    process, capture = shell.processing, shell.capture
    assert process.start_button.text() == "Start Processing" and shell.tabs.tabText(0) == "Capture"

    switch(shell, "한국어")
    assert shell.processing.settings.get("language") == "ko"
    assert process.start_button.text() == "처리 시작" and process.open_button.text() == "열기..."
    assert shell.tabs.tabText(0) == "촬영" and shell.tabs.tabText(1) == "처리"
    assert capture.qr_button.text() == "폰 QR" and capture.radio_manual.text().startswith("직접 입력")
    assert capture.stream_chip.value.text() == "연결 안 됨" or capture.stream_chip.value.text() == "수신기 없음"
    assert process.progress.text() == "대기 중"

    switch(shell, "English")
    assert process.start_button.text() == "Start Processing" and shell.tabs.tabText(0) == "Capture"
    assert capture.qr_button.text() == "Phone QR" and process.progress.text() == "Waiting"


def test_pc_camera_controls_and_messages_follow_the_language(shell):
    capture = shell.capture
    capture.source_combo.setCurrentIndex(capture.source_combo.findData("camera"))
    capture._webcam.cameras_found.emit(["Cam A"])
    capture._webcam.camera_state.emit(True, "1920x1080 · 30 fps")
    assert capture.stream_chip.caption.text() == "CAMERA" and capture.camera_refresh.text() == "Refresh"

    switch(shell, "한국어")
    assert [capture.source_combo.itemText(i) for i in (0, 1)] == ["폰 (QR)", "PC 카메라"]
    assert capture.source_combo.currentData() == "camera"  # the data value does not change with the language
    assert capture.stream_chip.caption.text() == "카메라" and capture.stream_chip.value.text() == "준비됨"
    assert capture.camera_refresh.text() == "새로고침"
    assert capture.message.text().startswith("PC 카메라가 준비되었습니다(1920x1080 · 30 fps).")
    assert capture.camera_combo.itemText(0) == "Cam A"  # camera names stay as the driver reports them

    capture._on_cameras([])  # the placeholder text is translated too
    assert capture.camera_combo.itemText(0) == "카메라를 찾지 못했습니다"

    switch(shell, "English")
    assert [capture.source_combo.itemText(i) for i in (0, 1)] == ["Phone (QR)", "PC camera"]
    assert capture.stream_chip.caption.text() == "CAMERA" and capture.camera_refresh.text() == "Refresh"
    assert capture.camera_combo.itemText(0) == "No camera found"


def test_combo_boxes_keep_working_on_their_data_when_translated(shell):
    process = shell.processing
    switch(shell, "한국어")
    assert process.collision_combo.itemText(0) == "확인" and process.sampling_combo.itemText(1) == "모든 프레임"
    assert process.theme_combo.itemText(2) == "다크" and process.aggregation_combo.itemText(0) == "중앙값"
    assert process.collision_combo.currentData() == "ask"

    process.collision_combo.setCurrentIndex(process.collision_combo.findData("overwrite"))
    process.sampling_combo.setCurrentIndex(process.sampling_combo.findData("uniform"))
    process.aggregation_combo.setCurrentIndex(process.aggregation_combo.findData("mean"))
    assert process.settings.get("collision_policy") == "overwrite"
    assert process.settings.get("sampling_mode") == "uniform" and process.sample_count_combo.isEnabled()
    assert process.settings.get("aggregation") == "mean"
    process.theme_combo.setCurrentIndex(process.theme_combo.findData("light"))
    assert process.settings.get("theme") == "light"
    process.theme_combo.setCurrentIndex(process.theme_combo.findData("dark"))

    switch(shell, "English")
    assert process.collision_combo.itemText(1) == "Overwrite" and process.collision_combo.currentData() == "overwrite"


def test_new_rows_status_and_queue_follow_the_language(shell, tmp_path):
    process = shell.processing
    switch(shell, "한국어")
    process._add_configuration_row()
    assert process.configuration_rows[-1].remove.text() == "제거"  # created after the switch

    process.progress.setText("Completed with errors (1 failed)")
    assert process.progress.text() == "오류와 함께 완료(1개 실패)"

    video = tmp_path / "clip.mp4"
    video.touch()
    process.add_media([video])
    assert process.queue.item(0).text() == "[대기 중] clip.mp4"
    switch(shell, "English")
    assert process.queue.item(0).text() == "[Waiting] clip.mp4"


def test_capture_messages_are_translated_when_shown_and_when_the_language_changes(shell):
    capture = shell.capture
    capture._say("Starting the recording ...")
    assert capture.message.text() == "Starting the recording ..."
    switch(shell, "한국어")
    assert capture.message.text() == "녹화를 시작하는 중 ..."
    capture._say("Phone connected.", "Start the work recording when you are ready.")
    assert capture.message.text() == "폰이 연결되었습니다. 준비되면 작업 녹화를 시작하세요."
    switch(shell, "English")
    assert capture.message.text() == "Phone connected. Start the work recording when you are ready."


def test_saved_language_is_applied_when_the_window_opens(tmp_path, qapp):
    from calibfreehands.app.shell import AppShell

    settings = AppSettings(tmp_path / "settings.ini")
    settings.set("language", "ko")
    window = AppShell(settings, controller_factory=FakeController, camera_factory=FakeCamera)
    try:
        assert window.processing.start_button.text() == "처리 시작"
        assert window.processing.language_combo.currentText() == "한국어"
        assert window.tabs.tabText(0) == "촬영"
    finally:
        window.capture._clock.stop()
        window.close()


def test_result_viewer_is_translated_and_restored(tmp_path, qapp):
    from PySide6.QtWidgets import QAbstractButton
    from test_comparison_viewer import _session
    from calibfreehands.visualization.comparison import ComparisonViewer

    i18n.set_language("ko")
    viewer = ComparisonViewer(_session(tmp_path))
    try:
        buttons = lambda: {b.text() for b in viewer.findChildren(QAbstractButton)}
        assert "카메라 연동" in buttons() and "Link Cameras" not in buttons()
        assert "MPJPE" in " ".join(buttons())  # metric abbreviations stay as they are in the exports
        assert viewer.status.text().startswith("프레임 1 / 3")
        assert viewer.windowTitle() == "CalibFreeHands - 비교"
        assert viewer.mode_switch.toolTip() == "보기: 겹치기"

        i18n.set_language("en")
        viewer.retranslate()
        assert "Link Cameras" in buttons() and "카메라 연동" not in buttons()
        assert viewer.status.text().startswith("Frame 1 / 3")
        assert viewer.windowTitle() == "CalibFreeHands - Comparison"
    finally:
        viewer.close()


def test_viewer_widgets_built_later_are_translated_too(tmp_path, qapp):
    from PySide6.QtTest import QTest
    from test_comparison_viewer import _session
    from calibfreehands.visualization.comparison import ComparisonViewer

    i18n.set_language("ko")
    viewer = ComparisonViewer(_session(tmp_path))
    try:
        viewer.set_mode("split")
        QTest.qWait(30)  # the translation of rebuilt menus is coalesced into the next event-loop turn
        assert viewer.results_menu_button.text() == "표시 (3/3)"
        assert viewer.mode_switch.toolTip() == "보기: 나눠 보기"
    finally:
        viewer.close()
