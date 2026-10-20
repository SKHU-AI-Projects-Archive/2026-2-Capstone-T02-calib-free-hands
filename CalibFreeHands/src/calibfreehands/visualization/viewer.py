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
from PySide6.QtCore import QEvent, Qt, QTimer, QPointF, QLineF
from PySide6.QtGui import QGuiApplication, QImage, QPainter, QPen
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
        self.show_joints = self.show_bones = self.show_left = self.show_right = self.show_grid = self.show_axis = True
        self.show_mesh = loaded.has_mesh_topology
        self.mesh_opacity = 0.6
        self.hand_colors = ((0.1, 0.9, 1), (1, 0.8, 0.1))
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.grids = [gl.GLGridItem(color=(120, 140, 160, 55)) for _ in range(3)]
        for grid in self.grids: self.addItem(grid)
        self.axis = gl.GLLinePlotItem(mode="lines", width=2, antialias=True)
        self.axis_labels = [gl.GLTextItem(text=label) for label in ("X", "Y", "Z")]
        self.addItem(self.axis)
        for label in self.axis_labels: self.addItem(label)
        self.fit_center = None; self.fit_distance = 1.2; self.setCameraPosition(distance=self.fit_distance)
        self.joints = [gl.GLScatterPlotItem(), gl.GLScatterPlotItem()]
        self.bones = [gl.GLLinePlotItem(mode="lines", width=2, antialias=True), gl.GLLinePlotItem(mode="lines", width=2, antialias=True)]
        self.meshes = [gl.GLMeshItem(smooth=False, drawEdges=False, glOptions="translucent"), gl.GLMeshItem(smooth=False, drawEdges=False, glOptions="translucent")]
        for item in (*self.joints, *self.bones, *self.meshes): self.addItem(item); item.setVisible(False)
        self.set_mesh_opacity(self.mesh_opacity)
        self.set_frame(0)
        self._frame_camera(self._first_detected_frame(), reset_orientation=True)
    def mousePressEvent(self, event):
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        super().mousePressEvent(event)
    def set_mesh_opacity(self, opacity):
        self.mesh_opacity = float(np.clip(opacity, 0.1, 1.0))
        for hand_i, mesh in enumerate(self.meshes): mesh.setColor((*self.hand_colors[hand_i], self.mesh_opacity))
        self.update()
    def _first_detected_frame(self):
        detected = self.loaded.result.detected
        return int(np.flatnonzero(detected.any(axis=1))[0]) if detected.any() else self.frame
    def _geometry(self, frame):
        result = self.loaded.result; points = []
        visible = result.detected[frame]
        for hand_i in range(2):
            if not visible[hand_i] or (hand_i == 0 and not self.show_left) or (hand_i == 1 and not self.show_right): continue
            points.append(np.asarray(result.keypoints_3d[frame, hand_i]) @ RAW_TO_VIEW.T)
            if self.loaded.has_mesh_topology and result.mano_vertices is not None:
                points.append(np.asarray(result.mano_vertices[frame, hand_i]) @ RAW_TO_VIEW.T)
        finite = np.concatenate(points) if points else np.empty((0, 3), dtype=np.float32)
        return finite[np.isfinite(finite).all(axis=1)]
    def _update_reference(self, center, extent):
        size = max(extent * 1.5, 0.4); spacing = max(size / 10, 0.05)
        # Local XY grids become display XY (raw XZ), XZ (raw XY), and YZ (raw YZ).
        for grid, rotation in zip(self.grids, (None, (90, 1, 0, 0), (90, 0, 1, 0))):
            grid.resetTransform(); grid.setSize(size, size, 1); grid.setSpacing(spacing, spacing, 1)
            if rotation: grid.rotate(*rotation)
            grid.translate(*center)
        directions = np.eye(3, dtype=np.float32) @ RAW_TO_VIEW.T
        axis_size = size * 0.65
        positions = np.concatenate([np.stack((center, center + direction * axis_size)) for direction in directions])
        colors = np.repeat(np.asarray(((1, 0.2, 0.2, 1), (0.2, 1, 0.2, 1), (0.3, 0.5, 1, 1)), dtype=np.float32), 2, axis=0)
        self.axis.setData(pos=np.ascontiguousarray(positions, dtype=np.float32), color=colors, mode="lines")
        for label, direction in zip(self.axis_labels, directions): label.setData(pos=center + direction * axis_size * 1.08)
    def _frame_camera(self, frame=None, reset_orientation=False):
        finite = self._geometry(self.frame if frame is None else frame)
        if not len(finite): return False
        center = (finite.min(axis=0) + finite.max(axis=0)) / 2
        extent = max(float(np.ptp(finite, axis=0).max()), 0.1)
        self.fit_center = center; self.fit_distance = max(0.3, extent * 2.5); self._update_reference(center, extent)
        options = {"distance": self.fit_distance, "pos": pg.Vector(*center)}
        if reset_orientation: options.update(elevation=20, azimuth=45)
        self.setCameraPosition(**options)
        self._update_reference_visibility(); return True
    def fit_hands(self): self._frame_camera()
    def reset_camera(self): self._frame_camera(reset_orientation=True); self.update()
    def _preset(self, raw_direction):
        if self.fit_center is None: self._frame_camera()
        if self.fit_center is None: return
        direction = np.asarray(raw_direction, dtype=np.float32) @ RAW_TO_VIEW.T
        azimuth = float(np.degrees(np.arctan2(direction[1], direction[0])))
        elevation = float(np.degrees(np.arctan2(direction[2], np.hypot(direction[0], direction[1]))))
        self.setCameraPosition(pos=pg.Vector(*self.fit_center), distance=self.fit_distance, azimuth=azimuth, elevation=elevation)
    def front_view(self): self._preset((0, 0, -1))
    def side_view(self): self._preset((-1, 0, 0))
    def top_view(self): self._preset((0, -1, 0))
    def _update_reference_visibility(self):
        for grid in self.grids: grid.setVisible(self.show_grid)
        self.axis.setVisible(self.show_axis)
        for label in self.axis_labels: label.setVisible(self.show_axis)
    def set_frame(self, frame):
        self.frame = max(0, min(frame, self.loaded.frame_count - 1)); result = self.loaded.result; colors = tuple((*color, 1) for color in self.hand_colors)
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
                    self.meshes[hand_i].setMeshData(meshdata=gl.MeshData(vertexes=vertices, faces=faces)); self.meshes[hand_i].setColor((*self.hand_colors[hand_i], self.mesh_opacity)); self.meshes[hand_i].setVisible(True)
        self._update_reference_visibility(); self.update()


class ResultViewer(QWidget):
    def __init__(self, result_path: Path):
        super().__init__(); self.loaded = load_npz_result(result_path); self.setWindowTitle("CalibFreeHands — Result"); self.setMinimumSize(960, 640)
        screen = QGuiApplication.primaryScreen().availableGeometry(); self.resize(min(1400, int(screen.width() * .8)), min(900, int(screen.height() * .8)))
        self._arrow_mode = "frame"
        self.overlay = _OverlayCanvas(self.loaded); self.gl_view = _GLResultView(self.loaded); split = QSplitter(Qt.Orientation.Horizontal); split.addWidget(self.overlay); split.addWidget(self.gl_view); split.setSizes([600, 600])
        self.slider = QSlider(Qt.Orientation.Horizontal); self.slider.setRange(0, max(0, self.loaded.frame_count - 1)); self.slider.valueChanged.connect(self._set_frame)
        self.status = QLabel(); self.coordinate_info = QLabel("Camera space · metres   X: image right   Y: image down   Z: forward   ·   View centered on hands · raw coordinates unchanged")
        self.timer = QTimer(self); self.timer.timeout.connect(self._next); visibility = QHBoxLayout(); view = QHBoxLayout(); playback = QHBoxLayout()
        for label, attr in (("Joint", "show_joints"), ("Bone", "show_bones"), ("Left Hand", "show_left"), ("Right Hand", "show_right"), ("Grid", "show_grid"), ("Axis", "show_axis")):
            check = QCheckBox(label); check.setChecked(getattr(self.gl_view, attr)); check.toggled.connect(lambda value, a=attr: self._toggle(a, value)); visibility.addWidget(check)
        mesh = QCheckBox("Mesh"); mesh.setChecked(self.gl_view.show_mesh); mesh.setEnabled(self.loaded.has_mesh_topology); mesh.setToolTip("MANO mesh topology is unavailable for this result" if not self.loaded.has_mesh_topology else ""); mesh.toggled.connect(lambda value: self._toggle("show_mesh", value)); visibility.addWidget(mesh)
        opacity_label = QLabel("Opacity")
        self.mesh_opacity_slider = QSlider(Qt.Orientation.Horizontal); self.mesh_opacity_slider.setRange(10, 100); self.mesh_opacity_slider.setValue(60); self.mesh_opacity_slider.setEnabled(self.loaded.has_mesh_topology); self.mesh_opacity_slider.setMaximumWidth(120)
        self.mesh_opacity_value = QLabel("60%")
        self.mesh_opacity_slider.valueChanged.connect(self._set_mesh_opacity)
        opacity_label.setEnabled(self.loaded.has_mesh_topology); self.mesh_opacity_value.setEnabled(self.loaded.has_mesh_topology)
        visibility.addWidget(opacity_label); visibility.addWidget(self.mesh_opacity_slider); visibility.addWidget(self.mesh_opacity_value)
        previous = QPushButton("Previous"); previous.clicked.connect(self._previous); play = QPushButton("Play / Pause"); play.clicked.connect(self._play); nxt = QPushButton("Next"); nxt.clicked.connect(self._next_frame)
        for text, callback in (("Front", self.gl_view.front_view), ("Side", self.gl_view.side_view), ("Top", self.gl_view.top_view), ("Fit Hands", self.gl_view.fit_hands), ("Reset View", self.gl_view.reset_camera)):
            button = QPushButton(text); button.clicked.connect(callback); view.addWidget(button)
        playback.addWidget(previous); playback.addWidget(play); playback.addWidget(nxt)
        layout = QVBoxLayout(self); layout.addWidget(split, 1); layout.addWidget(self.slider); layout.addLayout(visibility); layout.addLayout(view); layout.addLayout(playback); layout.addWidget(self.status); layout.addWidget(self.coordinate_info)
        self.installEventFilter(self)
        for widget in self.findChildren(QWidget): widget.installEventFilter(self)
        self._update_status()
    def _toggle(self, attr, value):
        if hasattr(self.overlay, attr): setattr(self.overlay, attr, value)
        setattr(self.gl_view, attr, value); self.overlay.update(); self.gl_view.set_frame(self.slider.value())
    def _set_frame(self, frame): self.overlay.set_frame(frame); self.gl_view.set_frame(frame); self._update_status()
    def _set_mesh_opacity(self, percent):
        self.mesh_opacity_value.setText(f"{percent}%"); self.gl_view.set_mesh_opacity(percent / 100)
    def _previous(self): self.slider.setValue(max(0, self.slider.value() - 1))
    def _next_frame(self): self.slider.setValue(min(self.loaded.frame_count - 1, self.slider.value() + 1))
    def _is_gl_viewport(self, widget):
        while widget is not None:
            if widget is self.gl_view: return True
            widget = widget.parentWidget()
        return False
    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.MouseButtonPress:
            self._arrow_mode = "orbit" if self._is_gl_viewport(watched) else "frame"
        elif event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key == Qt.Key.Key_Space:
                self._play(); event.accept(); return True
            if watched is self.slider and key in (Qt.Key.Key_Left, Qt.Key.Key_Right): return False
            if self._arrow_mode == "orbit" and key in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
                azimuth, elevation = {Qt.Key.Key_Left: (-3, 0), Qt.Key.Key_Right: (3, 0), Qt.Key.Key_Up: (0, -3), Qt.Key.Key_Down: (0, 3)}[key]
                self.gl_view.orbit(azimuth, elevation); event.accept(); return True
            if self._arrow_mode == "frame":
                if key == Qt.Key.Key_Left: self._previous(); event.accept(); return True
                if key == Qt.Key.Key_Right: self._next_frame(); event.accept(); return True
                if key in (Qt.Key.Key_Up, Qt.Key.Key_Down): event.accept(); return True
        return super().eventFilter(watched, event)
    def _play(self): self.timer.stop() if self.timer.isActive() else self.timer.start(100)
    def _next(self): self.slider.setValue((self.slider.value() + 1) % max(1, self.loaded.frame_count))
    def _update_status(self):
        timestamp = self.loaded.result.timestamps[self.slider.value()]
        suffix = f" · {float(timestamp):.3f} s" if np.isfinite(timestamp) else ""
        self.status.setText(f"Frame {self.slider.value() + 1} / {self.loaded.frame_count}{suffix}")
