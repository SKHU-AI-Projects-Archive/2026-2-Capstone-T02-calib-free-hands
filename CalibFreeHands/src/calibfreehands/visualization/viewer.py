from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.opengl import shaders
from pyqtgraph.opengl.items.GLLinePlotItem import GLLinePlotItem
from pyqtgraph.opengl.items.GLScatterPlotItem import GLScatterPlotItem
from PySide6.QtCore import Qt, QTimer, QPointF, QLineF
from PySide6.QtGui import QImage, QPainter, QPen, QKeySequence, QShortcut
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QSlider, QSplitter, QVBoxLayout, QWidget

from calibfreehands.media.ffmpeg import FFmpegDecoder
from calibfreehands.media.probe import probe_media
from calibfreehands.pipeline.result import CanonicalResult
from .topology import HAND_CONNECTIONS, RAW_TO_VIEW


def _reset_gl_shader_cache() -> None:
    """Invalidate pyqtgraph's process-global shader handles before a new Qt GL context."""
    GLLinePlotItem._shaderProgram = None
    GLScatterPlotItem._shaderProgram = None
    for program in shaders.ShaderProgram.names.values():
        program.prog = None
        for shader in program.shaders:
            shader.compiled = None


@dataclass
class LoadedResult:
    result: CanonicalResult
    media_frames: list[np.ndarray] | None = None

    @property
    def frame_count(self) -> int: return len(self.result.frame_indices)
    @property
    def has_mesh(self) -> bool: return self.result.mano_vertices is not None
    @property
    def has_mesh_topology(self) -> bool: return self.has_mesh and self.result.mano_faces is not None
    @property
    def detected(self) -> np.ndarray: return self.result.detected


def load_npz_result(path: Path) -> LoadedResult:
    with np.load(path) as data:
        optional = lambda name: data[name] if name in data else None
        result = CanonicalResult(
            frame_indices=data["frame_indices"], timestamps=data["timestamps"], keypoints_2d=data["keypoints_2d"],
            keypoints_3d=data["keypoints_3d"], detected=data["detected"], focal_px=data["focal_px"],
            image_size=tuple(data["image_size"].tolist()), mano_vertices=optional("mano_vertices"),
            mano_faces=optional("mano_faces"), confidence=optional("confidence"), bbox=optional("bbox"),
            mano_pose=optional("mano_pose"), mano_shape=optional("mano_shape"), camera_translation=optional("camera_translation"))
    media_frames = None
    run_path = Path(path).with_name("run.json")
    if run_path.is_file():
        try:
            input_path = json.loads(run_path.read_text(encoding="utf-8")).get("input_path")
            if input_path and Path(input_path).is_file():
                info = probe_media(Path(input_path))
                media_frames = [frame.rgb for frame in FFmpegDecoder().decode_rgb(Path(input_path), info.width, info.height, info.frame_rate)]
        except (OSError, ValueError, RuntimeError): media_frames = None
    return LoadedResult(result, media_frames)


class _OverlayCanvas(QWidget):
    def __init__(self, loaded):
        super().__init__(); self.loaded = loaded; self.frame = 0
        self.show_joints = self.show_bones = self.show_left = self.show_right = True
    def set_frame(self, frame): self.frame = max(0, min(frame, self.loaded.frame_count - 1)); self.update()
    def paintEvent(self, _event):
        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), Qt.GlobalColor.black)
            if not self.loaded.media_frames or self.frame >= len(self.loaded.media_frames): return
            rgb = self.loaded.media_frames[self.frame]; image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888)
            painter.drawImage(self.rect(), image); sx, sy = self.width() / rgb.shape[1], self.height() / rgb.shape[0]
            for hand_i, hand in enumerate(self.loaded.result.keypoints_2d[self.frame]):
                if not self.loaded.detected[self.frame, hand_i] or (hand_i == 0 and not self.show_left) or (hand_i == 1 and not self.show_right): continue
                painter.setPen(QPen(Qt.GlobalColor.cyan if hand_i == 0 else Qt.GlobalColor.yellow, 2))
                if self.show_bones:
                    for a, b in HAND_CONNECTIONS:
                        if not np.isfinite(hand[[a, b]]).all(): continue
                        p1 = QPointF(float(hand[a, 0] * sx), float(hand[a, 1] * sy)); p2 = QPointF(float(hand[b, 0] * sx), float(hand[b, 1] * sy))
                        painter.drawLine(QLineF(p1, p2))
                if self.show_joints:
                    for point in hand:
                        if not np.isfinite(point).all(): continue
                        x, y = float(point[0] * sx), float(point[1] * sy)
                        painter.drawEllipse(QPointF(x, y), 3.0, 3.0)
        finally:
            if painter.isActive(): painter.end()


class _GLResultView(gl.GLViewWidget):
    def __init__(self, loaded):
        _reset_gl_shader_cache(); super().__init__(); self.loaded = loaded; self.frame = 0
        self.show_joints = self.show_bones = self.show_mesh = self.show_left = self.show_right = self.show_grid = self.show_axis = True
        self.grid = gl.GLGridItem(); self.grid.setSize(1, 1); self.grid.setSpacing(0.1, 0.1); self.addItem(self.grid)
        self.axis = gl.GLAxisItem(); self.axis.setSize(0.5, 0.5, 0.5); self.addItem(self.axis); self.setCameraPosition(distance=1.2)
        self.joints = [gl.GLScatterPlotItem(), gl.GLScatterPlotItem()]
        self.bones = [gl.GLLinePlotItem(mode="lines", width=2, antialias=True), gl.GLLinePlotItem(mode="lines", width=2, antialias=True)]
        self.meshes = [gl.GLMeshItem(smooth=False, drawEdges=False), gl.GLMeshItem(smooth=False, drawEdges=False)]
        for item in (*self.joints, *self.bones, *self.meshes): self.addItem(item); item.setVisible(False)
        self.set_frame(0)
    def reset_camera(self): self._frame_camera(); self.update()
    def _frame_camera(self):
        result = self.loaded.result; points = []
        visible = result.detected[self.frame]
        for hand_i in range(2):
            if not visible[hand_i] or (hand_i == 0 and not self.show_left) or (hand_i == 1 and not self.show_right): continue
            points.append(np.asarray(result.keypoints_3d[self.frame, hand_i]) @ RAW_TO_VIEW.T)
            if self.loaded.has_mesh_topology and result.mano_vertices is not None: points.append(np.asarray(result.mano_vertices[self.frame, hand_i]) @ RAW_TO_VIEW.T)
        finite = np.concatenate(points) if points else np.zeros((1, 3), dtype=np.float32); finite = finite[np.isfinite(finite).all(axis=1)]
        if not len(finite): return
        center = finite.mean(axis=0); extent = max(float(np.ptp(finite, axis=0).max()), 0.1)
        self.setCameraPosition(distance=max(0.3, extent * 2.5), elevation=20, azimuth=45, pos=pg.Vector(*center))
    def set_frame(self, frame):
        self.frame = max(0, min(frame, self.loaded.frame_count - 1)); result = self.loaded.result; colors = ((0.1, 0.9, 1, 1), (1, 0.8, 0.1, 1))
        for hand_i in range(2):
            visible = bool(result.detected[self.frame, hand_i]) and not (hand_i == 0 and not self.show_left) and not (hand_i == 1 and not self.show_right)
            for item in (self.joints[hand_i], self.bones[hand_i], self.meshes[hand_i]): item.setVisible(False)
            if not visible: continue
            points = np.asarray(result.keypoints_3d[self.frame, hand_i]) @ RAW_TO_VIEW.T
            if not np.isfinite(points).all(): continue
            if self.show_joints:
                self.joints[hand_i].setData(pos=np.ascontiguousarray(points, dtype=np.float32), size=8, color=colors[hand_i]); self.joints[hand_i].setVisible(True)
            if self.show_bones:
                bone_points = np.ascontiguousarray(np.concatenate([points[[a, b]] for a, b in HAND_CONNECTIONS]), dtype=np.float32)
                self.bones[hand_i].setData(pos=bone_points, color=colors[hand_i], mode="lines"); self.bones[hand_i].setVisible(True)
            if self.show_mesh and self.loaded.has_mesh_topology:
                vertices = np.asarray(result.mano_vertices[self.frame, hand_i]) @ RAW_TO_VIEW.T; faces = result.mano_faces[hand_i] if result.mano_faces.ndim == 3 else result.mano_faces
                vertices = np.ascontiguousarray(vertices, dtype=np.float32); faces = np.ascontiguousarray(faces, dtype=np.int32)
                if np.isfinite(vertices).all() and faces.ndim == 2 and faces.shape[1] == 3 and faces.min() >= 0 and faces.max() < len(vertices):
                    self.meshes[hand_i].setMeshData(meshdata=gl.MeshData(vertexes=vertices, faces=faces)); self.meshes[hand_i].setColor(colors[hand_i]); self.meshes[hand_i].setVisible(True)
        self.grid.setVisible(self.show_grid); self.axis.setVisible(self.show_axis); self.update()
        if self.frame == 0: self._frame_camera()


class ResultViewer(QWidget):
    def __init__(self, result_path: Path):
        super().__init__(); self.loaded = load_npz_result(result_path); self.setWindowTitle("CalibFreeHands — Result")
        self.overlay = _OverlayCanvas(self.loaded); self.gl_view = _GLResultView(self.loaded); split = QSplitter(Qt.Orientation.Horizontal); split.addWidget(self.overlay); split.addWidget(self.gl_view); split.setSizes([1, 1])
        self.slider = QSlider(Qt.Orientation.Horizontal); self.slider.setRange(0, max(0, self.loaded.frame_count - 1)); self.slider.valueChanged.connect(self._set_frame)
        self.status = QLabel(); self.timer = QTimer(self); self.timer.timeout.connect(self._next); controls = QHBoxLayout()
        for label, attr in (("Joint", "show_joints"), ("Bone", "show_bones"), ("Left Hand", "show_left"), ("Right Hand", "show_right"), ("Grid", "show_grid"), ("Axis", "show_axis")):
            check = QCheckBox(label); check.setChecked(True); check.toggled.connect(lambda value, a=attr: self._toggle(a, value)); controls.addWidget(check)
        mesh = QCheckBox("Mesh"); mesh.setEnabled(self.loaded.has_mesh_topology); mesh.setToolTip("MANO faces are not present in this legacy NPZ" if not self.loaded.has_mesh_topology else ""); mesh.toggled.connect(lambda value: self._toggle("show_mesh", value)); controls.addWidget(mesh)
        previous = QPushButton("Previous"); previous.clicked.connect(lambda: self.slider.setValue(max(0, self.slider.value() - 1))); play = QPushButton("Play / Pause"); play.clicked.connect(self._play); nxt = QPushButton("Next"); nxt.clicked.connect(lambda: self.slider.setValue(min(self.loaded.frame_count - 1, self.slider.value() + 1))); reset = QPushButton("Reset Camera"); reset.clicked.connect(self.gl_view.reset_camera)
        controls.addWidget(previous); controls.addWidget(play); controls.addWidget(nxt); controls.addWidget(reset); layout = QVBoxLayout(self); layout.addWidget(split, 1); layout.addWidget(self.slider); layout.addLayout(controls); layout.addWidget(self.status)
        for key, callback in (("Space", self._play), ("Left", lambda: self.slider.setValue(max(0, self.slider.value() - 1))), ("Right", lambda: self.slider.setValue(min(self.loaded.frame_count - 1, self.slider.value() + 1)))): QShortcut(QKeySequence(key), self, activated=callback)
        self._update_status()
    def _toggle(self, attr, value):
        if hasattr(self.overlay, attr): setattr(self.overlay, attr, value)
        setattr(self.gl_view, attr, value); self.overlay.update(); self.gl_view.set_frame(self.slider.value())
    def _set_frame(self, frame): self.overlay.set_frame(frame); self.gl_view.set_frame(frame); self._update_status()
    def _play(self): self.timer.stop() if self.timer.isActive() else self.timer.start(100)
    def _next(self): self.slider.setValue((self.slider.value() + 1) % max(1, self.loaded.frame_count))
    def _update_status(self): self.status.setText(f"Frame {self.slider.value() + 1} / {self.loaded.frame_count}")
