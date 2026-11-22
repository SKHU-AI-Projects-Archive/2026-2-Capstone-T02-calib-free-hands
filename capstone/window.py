from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QPushButton, QProgressBar, QVBoxLayout, QWidget,
)

from calibfreehands.visualization.viewer import ResultViewer

from .app_run import process


class ProcessingJob(QThread):
    progress = Signal(str, int, int)
    focal_ready = Signal(float)
    completed = Signal(object, object)
    failed = Signal(str)

    def __init__(self, path, options, parent):
        super().__init__(parent)
        self.path = path
        self.options = options

    def run(self):
        try:
            result_path, report = process(
                self.path, self.options, self.progress.emit,
                self.focal_ready.emit, self.isInterruptionRequested,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.completed.emit(result_path, report)


class DemoWindow(QWidget):
    def __init__(self, options):
        super().__init__()
        self.options = options
        self.job = None
        self.viewer = None
        self.report = None
        self.closing = False
        self.setWindowTitle("CalibFreeHands · 저장 영상 시연")
        self.resize(1240, 820)
        self.choose = QPushButton("영상 열기")
        self.choose.clicked.connect(self._choose_video)
        self.source = QLabel("영상 선택 대기")
        self.source.setTextFormat(Qt.TextFormat.PlainText)
        self.focal = QLabel("f (px)  —")
        self.focal.setStyleSheet("font-size: 20px; font-weight: 600;")
        self.status = QLabel("영상을 열면 처음 20초로 보정하고 후속 프레임에 적용합니다.")
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.hide()
        self.placeholder = QLabel("보정과 3D 처리가 끝나면 원본 영상과 손 결과가 표시됩니다.")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header = QHBoxLayout()
        header.addWidget(self.choose)
        header.addWidget(self.source, 1)
        header.addWidget(self.focal)
        self.layout = QVBoxLayout(self)
        self.layout.addLayout(header)
        self.layout.addWidget(self.status)
        self.layout.addWidget(self.progress)
        self.layout.addWidget(self.placeholder, 1)

    def _choose_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "저장 영상 열기", "",
                                              "동영상 (*.mp4 *.mov *.mkv *.avi);;모든 파일 (*)")
        if path:
            self.open_video(Path(path))

    def open_video(self, path: Path):
        if self.job is not None:
            return
        if self.viewer is not None:
            self.viewer.close()
            self.layout.removeWidget(self.viewer)
            self.viewer.deleteLater()
            self.viewer = None
        self.report = None
        self.source.setText(path.name)
        self.focal.setText("f (px)  —")
        self.status.setText("영상을 여는 중…")
        self.progress.setRange(0, 0)
        self.progress.show()
        self.placeholder.show()
        self.choose.setEnabled(False)
        self.job = ProcessingJob(path, self.options, self)
        self.job.progress.connect(self._on_progress)
        self.job.focal_ready.connect(self._show_focal)
        self.job.completed.connect(self._show_result)
        self.job.failed.connect(self._show_error)
        self.job.finished.connect(self._on_finished)
        self.job.start()

    def _show_focal(self, value):
        self.focal.setText(f"f (px)  {value:,.2f}")

    def _on_progress(self, stage, done, total):
        self.status.setText({
            "Reading video timestamps": "영상 타임스탬프 읽는 중",
            "Loading focal estimator": "보정 모델 준비 중",
            "Calibrating": "처음 20초의 손 영상으로 초점거리 계산 중",
            "Reconstructing hands": "후속 프레임에 f를 적용해 3D 손 계산 중",
        }.get(stage, stage))
        self.progress.setRange(0, total)
        self.progress.setValue(done)

    def _show_result(self, result_path, report):
        if self.closing:
            return
        try:
            viewer = ResultViewer(result_path, preserve_source_aspect=True)
        except Exception as exc:
            self._show_error(f"결과 파일 {result_path.name}을 열 수 없습니다. {exc}")
            return
        self.report = report
        self.viewer = viewer
        self._show_focal(report["focal_px"])
        start, end = report["work_frame_times_s"][0], report["work_frame_times_s"][-1]
        detected, total = report["frames_with_hands"], report["work_frames"]
        self.status.setText(
            f"{report['estimator']}  |  보정 0–20초  |  후속 {start:.3f}–{end:.3f}초  |  "
            f"3D 손 검출 {detected}/{total} 프레임"
        )
        self.placeholder.hide()
        self.layout.addWidget(viewer, 1)
        viewer.mesh_checkbox.setChecked(viewer.loaded.has_mesh_topology)
        visible = np.flatnonzero(viewer.loaded.detected.any(axis=1))
        selected = int(visible[0]) if len(visible) else 0
        viewer.slider.setValue(selected)
        viewer._set_frame(selected)

    def _show_error(self, message):
        self.status.setText(f"처리 중 오류가 발생했습니다. {message}")

    def _on_finished(self):
        self.job.deleteLater()
        self.job = None
        self.progress.hide()
        self.choose.setEnabled(True)
        if self.closing:
            QTimer.singleShot(0, self.close)

    def closeEvent(self, event):
        if self.job is not None:
            self.closing = True
            self.job.requestInterruption()
            event.ignore()
            return
        if self.viewer is not None:
            self.viewer.close()
        super().closeEvent(event)
