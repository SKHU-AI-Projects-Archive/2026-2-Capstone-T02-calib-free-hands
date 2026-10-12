from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtGui import QImage
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QPen
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget

from calibfreehands.pipeline.result import CanonicalResult
from calibfreehands.media.ffmpeg import FFmpegDecoder
from calibfreehands.media.probe import probe_media


@dataclass
class LoadedResult:
    result: CanonicalResult
    media_frames: list[np.ndarray] | None = None

    @property
    def frame_count(self) -> int:
        return len(self.result.frame_indices)

    @property
    def has_mesh(self) -> bool:
        return self.result.mano_vertices is not None

    @property
    def detected(self) -> np.ndarray:
        return self.result.detected


def load_npz_result(path: Path) -> LoadedResult:
    data = np.load(path)
    def optional(name):
        return data[name] if name in data else None
    result = CanonicalResult(data["frame_indices"], data["timestamps"], data["keypoints_2d"], data["keypoints_3d"],
                             data["detected"], data["focal_px"], tuple(data["image_size"].tolist()), optional("mano_vertices"),
                             optional("confidence"), optional("bbox"), optional("mano_pose"), optional("mano_shape"), optional("camera_translation"))
    media_frames = None
    run_path = Path(path).with_name("run.json")
    if run_path.is_file():
        try:
            import json
            input_path = json.loads(run_path.read_text(encoding="utf-8")).get("input_path")
            if input_path and Path(input_path).is_file():
                info = probe_media(Path(input_path))
                media_frames = [frame.rgb for frame in FFmpegDecoder().decode_rgb(Path(input_path), info.width, info.height)]
        except (OSError, ValueError, RuntimeError):
            media_frames = None
    return LoadedResult(result, media_frames)


class _Canvas(QWidget):
    bones = tuple((i - 1, i) for i in range(1, 21))

    def __init__(self, loaded: LoadedResult):
        super().__init__()
        self.loaded, self.frame = loaded, 0
        self.show_joints, self.show_bones = True, True
        self.show_mesh = False
        self.show_left, self.show_right, self.show_grid, self.show_axis = True, True, True, True
        self.zoom, self.yaw, self.pitch = 1.0, 0.0, 0.0
        self.pan = np.zeros(2)
        self._drag = None

    def set_frame(self, frame: int):
        self.frame = max(0, min(frame, self.loaded.frame_count - 1)); self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton: self._drag = event.position()

    def mouseMoveEvent(self, event):
        if self._drag is not None:
            delta = event.position() - self._drag; self.yaw += delta.x() * .5; self.pitch += delta.y() * .5; self.pan += np.array([delta.x(), delta.y()]); self._drag = event.position(); self.update()

    def mouseReleaseEvent(self, event): self._drag = None

    def wheelEvent(self, event):
        self.zoom = max(.2, min(5.0, self.zoom * (1.1 if event.angleDelta().y() > 0 else .9))); self.update()

    def reset_camera(self): self.zoom, self.yaw, self.pitch, self.pan = 1.0, 0.0, 0.0, np.zeros(2); self.update()

    def paintEvent(self, event):
        painter = QPainter(self); painter.fillRect(self.rect(), Qt.GlobalColor.black)
        split = self.width() // 2
        if self.loaded.media_frames and self.frame < len(self.loaded.media_frames):
            rgb = self.loaded.media_frames[self.frame]
            image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888)
            painter.drawImage(self.rect().adjusted(0, 0, -split, 0), image)
            sx, sy = split / rgb.shape[1], self.height() / rgb.shape[0]
            for hand_i, hand in enumerate(self.loaded.result.keypoints_2d[self.frame]):
                if (hand_i == 0 and not self.show_left) or (hand_i == 1 and not self.show_right) or not self.loaded.detected[self.frame, hand_i]: continue
                painter.setPen(QPen(Qt.GlobalColor.cyan if hand_i == 0 else Qt.GlobalColor.yellow, 2))
                if self.show_bones:
                    for a, b in self.bones: painter.drawLine(int(hand[a, 0] * sx), int(hand[a, 1] * sy), int(hand[b, 0] * sx), int(hand[b, 1] * sy))
                if self.show_joints:
                    for point in hand: painter.drawEllipse(int(point[0] * sx - 3), int(point[1] * sy - 3), 6, 6)
        painter.setPen(QPen(Qt.GlobalColor.gray, 1)); painter.drawLine(split, 0, split, self.height())
        if self.show_grid:
            painter.setPen(QPen(Qt.GlobalColor.darkGray))
            for x in range(0, self.width(), 40): painter.drawLine(x, self.height() // 2, x, self.height())
        points = self.loaded.result.keypoints_3d[self.frame]
        center = np.array([split + (self.width() - split) / 2, self.height() / 2]) + self.pan
        projected = []
        for hand in points:
            projected.append([center + np.array([p[0] * 300, -p[1] * 300]) * self.zoom for p in hand])
        for hand_i, hand in enumerate(projected):
            if (hand_i == 0 and not self.show_left) or (hand_i == 1 and not self.show_right) or not self.loaded.detected[self.frame, hand_i]: continue
            painter.setPen(QPen(Qt.GlobalColor.cyan if hand_i == 0 else Qt.GlobalColor.yellow, 2))
            if self.show_bones:
                for a, b in self.bones: painter.drawLine(hand[a][0], hand[a][1], hand[b][0], hand[b][1])
            if self.show_joints:
                for point in hand: painter.drawEllipse(int(point[0] - 3), int(point[1] - 3), 6, 6)
            if self.show_mesh and self.loaded.result.mano_vertices is not None:
                painter.setPen(QPen(Qt.GlobalColor.magenta, 1))
                for vertex in self.loaded.result.mano_vertices[self.frame, hand_i]:
                    point = center + np.array([vertex[0] * 300, -vertex[1] * 300]) * self.zoom
                    painter.drawPoint(int(point[0]), int(point[1]))


class ResultViewer(QWidget):
    def __init__(self, result_path: Path):
        super().__init__(); self.loaded = load_npz_result(result_path); self.setWindowTitle("CalibFreeHands — Result")
        self.canvas = _Canvas(self.loaded); self.slider = QSlider(Qt.Orientation.Horizontal); self.slider.setRange(0, max(0, self.loaded.frame_count - 1)); self.slider.valueChanged.connect(self.canvas.set_frame); self.slider.valueChanged.connect(lambda _: self._update_status())
        self.status = QLabel(); self.timer = QTimer(self); self.timer.timeout.connect(self._next)
        controls = QHBoxLayout()
        for label, attr in (("Joint", "show_joints"), ("Bone", "show_bones"), ("Left", "show_left"), ("Right", "show_right"), ("Grid", "show_grid"), ("Axis", "show_axis")):
            check = QCheckBox(label); check.setChecked(True); check.toggled.connect(lambda value, a=attr: (setattr(self.canvas, a, value), self.canvas.update())); controls.addWidget(check)
        mesh = QCheckBox("Mesh"); mesh.setEnabled(self.loaded.has_mesh); mesh.toggled.connect(lambda value: (setattr(self.canvas, "show_mesh", value), self.canvas.update())); controls.addWidget(mesh)
        previous = QPushButton("Previous"); previous.clicked.connect(lambda: self.slider.setValue(max(0, self.slider.value() - 1))); next_button = QPushButton("Next"); next_button.clicked.connect(lambda: self.slider.setValue(min(self.loaded.frame_count - 1, self.slider.value() + 1)))
        play = QPushButton("Play / Pause"); play.clicked.connect(self._toggle); reset = QPushButton("Reset Camera"); reset.clicked.connect(self.canvas.reset_camera); controls.addWidget(previous); controls.addWidget(play); controls.addWidget(next_button); controls.addWidget(reset)
        layout = QVBoxLayout(self); layout.addWidget(self.canvas, 1); layout.addWidget(self.slider); layout.addLayout(controls); layout.addWidget(self.status); self._update_status()

    def _toggle(self): self.timer.stop() if self.timer.isActive() else self.timer.start(100)
    def _next(self): self.slider.setValue((self.slider.value() + 1) % max(1, self.loaded.frame_count)); self._update_status()
    def _update_status(self): self.status.setText(f"Frame {self.slider.value() + 1} / {self.loaded.frame_count}")
