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
from PySide6.QtCore import QEvent, Qt, QTimer, QPointF, QLineF, QSignalBlocker
from PySide6.QtGui import QFont, QGuiApplication, QImage, QPainter, QPen
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSlider, QSplitter, QVBoxLayout, QWidget

from calibfreehands.media.provider import MediaFrameProvider, release_shared_media_provider, shared_media_provider
from calibfreehands.pipeline.result import CanonicalResult
from calibfreehands.pipeline.result_bundle import ResultBundle, load_result_bundle
from calibfreehands.joint_schema import JOINT_NAMES
from .topology import HAND_CONNECTIONS, RAW_TO_VIEW

WORKSPACE_BOUNDS_M = ((-1.0, 1.0), (-1.0, 1.0), (0.0, 2.0))
GRID_MINOR_SPACING_M = 0.1
GRID_MAJOR_SPACING_M = 0.5
WORKSPACE_CENTER_RAW = (0.0, 0.0, 1.0)
DEFAULT_WORKSPACE_ASPECT = 4 / 3
DEFAULT_CAMERA_MARGIN = 1.1
MIN_HAND_VIEW_RADIUS_M = 0.15
HAND_SPACE_MIN_SPAN_M = 0.2
HAND_SPACE_PADDING_FRACTION = 0.15
HAND_SPACE_MIN_PADDING_M = 0.05
HAND_SPACE_OUTLIER_LIMIT_M = 0.25
WORKSPACE_LABEL_OFFSET_FRACTION = 0.06
WORKSPACE_AXIS_LABEL_OFFSET_FRACTION = 0.08


def _view_point_from_raw(point):
    return np.asarray(point, dtype=np.float64) @ RAW_TO_VIEW.T


def metric_workspace_center_display():
    return tuple(_view_point_from_raw(WORKSPACE_CENTER_RAW))


def _workspace_radius():
    corners = np.asarray([
        (x, y, z)
        for x in WORKSPACE_BOUNDS_M[0]
        for y in WORKSPACE_BOUNDS_M[1]
        for z in WORKSPACE_BOUNDS_M[2]
    ])
    center = np.asarray(WORKSPACE_CENTER_RAW)
    return float(np.linalg.norm(corners - center, axis=1).max())


def _camera_distance_for_radius(radius, fov_degrees=60.0, aspect=DEFAULT_WORKSPACE_ASPECT, margin=DEFAULT_CAMERA_MARGIN):
    aspect = max(float(aspect), 1e-6)
    half_horizontal = np.deg2rad(float(fov_degrees)) / 2
    half_vertical = np.arctan(np.tan(half_horizontal) / aspect)
    half_fov = min(half_horizontal, half_vertical)
    return float(max(radius, 0.05) * margin / np.sin(half_fov))


def default_metric_camera_distance(fov_degrees=60.0, aspect=DEFAULT_WORKSPACE_ASPECT):
    return _camera_distance_for_radius(_workspace_radius(), fov_degrees, aspect)


@dataclass(frozen=True)
class WorkspaceGeometry:
    """One metric cube definition for grid, axes, labels, and camera framing."""

    space_mode: str
    center_raw: tuple[float, float, float]
    side_length_m: float
    bounds_raw: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    major_spacing_m: float
    minor_spacing_m: float

    @property
    def bounds_array(self) -> np.ndarray:
        return np.asarray(self.bounds_raw, dtype=np.float64)

    @property
    def display_center(self) -> np.ndarray:
        return _view_point_from_raw(self.center_raw)

    @property
    def display_corners(self) -> np.ndarray:
        return np.asarray([
            (x, y, z)
            for x in self.bounds_raw[0]
            for y in self.bounds_raw[1]
            for z in self.bounds_raw[2]
        ], dtype=np.float64) @ RAW_TO_VIEW.T


def workspace_geometry(space_mode: str, bounds_m: np.ndarray | None = None) -> WorkspaceGeometry:
    if space_mode not in {"physical", "hand"}:
        raise ValueError(f"unknown 3D space mode: {space_mode}")
    bounds = np.asarray(WORKSPACE_BOUNDS_M if space_mode == "physical" or bounds_m is None else bounds_m,
                        dtype=np.float64)
    if bounds.shape != (3, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("workspace bounds must be finite increasing [axis, lower/upper] pairs")
    if space_mode == "hand" and np.array_equal(bounds, np.asarray(WORKSPACE_BOUNDS_M)):
        return workspace_geometry("physical")
    if space_mode == "physical":
        bounds = np.asarray(WORKSPACE_BOUNDS_M, dtype=np.float64)
        side = 2.0
        major, minor = GRID_MAJOR_SPACING_M, GRID_MINOR_SPACING_M
    else:
        center = bounds.mean(axis=1)
        side = float(np.max(np.ptp(bounds, axis=1)))
        if side <= 0 or not np.isfinite(side):
            return workspace_geometry("physical")
        if center[2] - side / 2 < 0 and bounds[2, 0] >= 0:
            center[2] = side / 2
        bounds = np.column_stack((center - side / 2, center + side / 2))
        major = nice_grid_spacing(side)
        minor = major / 5.0
    center = bounds.mean(axis=1)
    return WorkspaceGeometry(
        space_mode="physical" if space_mode == "physical" else "hand",
        center_raw=tuple(float(value) for value in center),
        side_length_m=side,
        bounds_raw=tuple(tuple(float(value) for value in pair) for pair in bounds),
        major_spacing_m=major,
        minor_spacing_m=minor,
    )


def hand_space_bounds(keypoints_3d: np.ndarray, detected: np.ndarray,
                      mano_vertices: np.ndarray | None = None) -> np.ndarray:
    """Return padded robust raw-camera-space bounds for detected whole-video hand geometry."""
    joints = np.asarray(keypoints_3d, dtype=np.float64)
    detected = np.asarray(detected, dtype=bool)
    vertices = None if mano_vertices is None else np.asarray(mano_vertices, dtype=np.float64)
    frame_points: list[np.ndarray] = []
    frame_centers: list[np.ndarray] = []

    for frame in range(len(joints)):
        pieces = []
        centers = []
        for hand in range(joints.shape[1]):
            if not detected[frame, hand]:
                continue
            hand_joints = joints[frame, hand]
            finite_joints = hand_joints[np.isfinite(hand_joints).all(axis=1)]
            if finite_joints.size:
                pieces.append(finite_joints)
                wrist = hand_joints[0]
                centers.append(wrist if np.isfinite(wrist).all() else np.median(finite_joints, axis=0))
            if vertices is not None:
                hand_vertices = vertices[frame, hand]
                finite_vertices = hand_vertices[np.isfinite(hand_vertices).all(axis=1)]
                if finite_vertices.size:
                    pieces.append(finite_vertices)
                    if not finite_joints.size:
                        centers.append(np.median(finite_vertices, axis=0))
        if pieces:
            frame_points.append(np.concatenate(pieces))
            frame_centers.append(np.median(np.asarray(centers), axis=0))

    if not frame_points:
        return np.asarray(WORKSPACE_BOUNDS_M, dtype=np.float64)

    if len(frame_points) >= 3:
        centers = np.asarray(frame_centers)
        center = np.median(centers, axis=0)
        radial = np.linalg.norm(centers - center, axis=1)
        median_radius = float(np.median(radial))
        mad = float(np.median(np.abs(radial - median_radius)))
        cutoff = max(HAND_SPACE_OUTLIER_LIMIT_M, median_radius + 3.0 * 1.4826 * mad)
        inlier_frames = radial <= cutoff
        if inlier_frames.any():
            frame_points = [points for points, keep in zip(frame_points, inlier_frames) if keep]

    finite_points = np.concatenate(frame_points)
    robust_bounds = np.percentile(finite_points, (2.5, 97.5), axis=0).T
    spans = robust_bounds[:, 1] - robust_bounds[:, 0]
    padding = np.maximum(spans * HAND_SPACE_PADDING_FRACTION, HAND_SPACE_MIN_PADDING_M)
    bounds = robust_bounds + np.column_stack((-padding, padding))
    spans = bounds[:, 1] - bounds[:, 0]
    centers = bounds.mean(axis=1)
    too_small = spans < HAND_SPACE_MIN_SPAN_M
    bounds[too_small, 0] = centers[too_small] - HAND_SPACE_MIN_SPAN_M / 2
    bounds[too_small, 1] = centers[too_small] + HAND_SPACE_MIN_SPAN_M / 2
    cube_side = float(np.max(bounds[:, 1] - bounds[:, 0]))
    cube_center = bounds.mean(axis=1)
    if cube_center[2] - cube_side / 2 < 0 and bounds[2, 0] >= 0:
        cube_center[2] = cube_side / 2
    return np.column_stack((cube_center - cube_side / 2, cube_center + cube_side / 2))


def nice_grid_spacing(span_m: float, target_major_intervals: int = 5) -> float:
    """Choose a nearby 1/2/5 × 10ⁿ real-metre major grid step."""
    target = max(float(span_m) / max(target_major_intervals, 1), 1e-3)
    exponent = int(np.floor(np.log10(target)))
    choices = [multiplier * 10.0 ** power
               for power in range(exponent - 1, exponent + 2)
               for multiplier in (1.0, 2.0, 5.0)]
    return min(choices, key=lambda step: abs(np.log(step / target)))


def whole_video_hand_framing(keypoints_3d: np.ndarray, detected: np.ndarray,
                             fov_degrees: float = 60.0,
                             aspect: float = DEFAULT_WORKSPACE_ASPECT) -> tuple[np.ndarray, float]:
    """Return a robust whole-sequence hand target and camera distance in display space."""
    keypoints_3d = np.asarray(keypoints_3d)
    detected = np.asarray(detected, dtype=bool)
    wrist_points = keypoints_3d[:, :, JOINT_NAMES.index("wrist"), :]
    valid_wrists = detected & np.isfinite(wrist_points).all(axis=-1)
    detected_joints = keypoints_3d[detected]
    finite_joints = detected_joints[np.isfinite(detected_joints).all(axis=-1)]

    if valid_wrists.any():
        target_raw = np.median(wrist_points[valid_wrists], axis=0)
    elif len(finite_joints):
        target_raw = np.median(finite_joints, axis=0)
    else:
        return np.asarray(metric_workspace_center_display()), default_metric_camera_distance(fov_degrees, aspect)

    if not len(finite_joints):
        radius = MIN_HAND_VIEW_RADIUS_M
    else:
        bounds = np.percentile(finite_joints, (2.5, 97.5), axis=0)
        corners = np.asarray(np.meshgrid(*bounds.T)).T.reshape(-1, 3)
        radius = float(np.linalg.norm(corners - target_raw, axis=1).max())
        radius = max(radius, MIN_HAND_VIEW_RADIUS_M)
    distance = _camera_distance_for_radius(radius, fov_degrees, aspect)
    return target_raw @ RAW_TO_VIEW.T, distance


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
    source_status: str | None = None
    metadata: dict | None = None

    @property
    def frame_count(self) -> int: return len(self.result.frame_indices)
    @property
    def has_mesh(self) -> bool: return self.result.mano_vertices is not None
    @property
    def has_mesh_topology(self) -> bool: return self.has_mesh and self.result.mano_faces is not None
    @property
    def detected(self) -> np.ndarray: return self.result.detected


def load_npz_result(path: Path, decode_media: bool = True) -> LoadedResult:
    with np.load(path, allow_pickle=False) as data:
        required = {"frame_indices", "timestamps", "keypoints_2d", "keypoints_3d", "detected", "focal_px", "image_size"}
        missing = required - set(data.files)
        if missing:
            if "keypoints_3d" not in data.files and "keypoints_3d_relative" in data.files:
                raise ValueError("This NPZ is an internal raw-pose cache, not a final CalibFreeHands result.")
            raise ValueError(f"Result NPZ is missing canonical arrays: {', '.join(sorted(missing))}")
        optional = lambda name: data[name] if name in data else None
        result = CanonicalResult(
            frame_indices=data["frame_indices"], timestamps=data["timestamps"], keypoints_2d=data["keypoints_2d"],
            keypoints_3d=data["keypoints_3d"], detected=data["detected"], focal_px=data["focal_px"],
            image_size=tuple(data["image_size"].tolist()), mano_vertices=optional("mano_vertices"),
            mano_faces=optional("mano_faces"), confidence=optional("confidence"), bbox=optional("bbox"),
            mano_pose=optional("mano_pose"), mano_shape=optional("mano_shape"), camera_translation=optional("camera_translation"))
    media_frames = None
    source_status = "unavailable"
    run_path = Path(path).with_name("run.json")
    if decode_media and run_path.is_file():
        try:
            provenance = json.loads(run_path.read_text(encoding="utf-8"))
            input_path = provenance.get("input_path")
            if input_path:
                media_frames, source_status = _decode_source(Path(input_path), provenance.get("input_sha256"))
        except (OSError, ValueError, RuntimeError):
            source_status = "unavailable"
    return LoadedResult(result, media_frames, source_status)


def _decode_source(path: Path, expected_sha256: str | None) -> tuple[MediaFrameProvider | None, str]:
    return shared_media_provider(path, expected_sha256)


def load_bundle_result(bundle: ResultBundle, decode_media: bool = True) -> LoadedResult:
    metadata = bundle.metadata
    media, status = None, "unavailable"
    source_path = metadata.get("source_media_path", "")
    if decode_media and source_path:
        media, status = _decode_source(Path(source_path), metadata.get("source_media_sha256"))
    return LoadedResult(bundle.result, media, status, metadata)


class _OverlayCanvas(QWidget):
    def __init__(self, loaded):
        super().__init__(); self.loaded = loaded; self.frame = 0; self._display_image = QImage(); self._requested_source_index = None
        provider = loaded.media_frames
        if isinstance(provider, MediaFrameProvider):
            provider.signals.frame_ready.connect(self._frame_ready)
            self.set_frame(0)
        self.show_joints = self.show_bones = self.show_left = self.show_right = True
    def set_frame(self, frame, source_image=None):
        self.frame = max(0, min(frame, self.loaded.frame_count - 1))
        if source_image is not None:
            self._display_image = source_image
        provider = self.loaded.media_frames
        if isinstance(provider, MediaFrameProvider):
            source_index = int(self.loaded.result.frame_indices[self.frame])
            if provider._is_image:
                source_index = 0
            if source_index != self._requested_source_index:
                self._requested_source_index = source_index
                provider.frame(source_index)
        self.update()

    def _frame_ready(self, source_index):
        expected = int(self.loaded.result.frame_indices[self.frame])
        provider = self.loaded.media_frames
        if isinstance(provider, MediaFrameProvider) and provider._is_image:
            expected = 0
        if source_index == expected:
            self.update()
    def paintEvent(self, _event):
        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), Qt.GlobalColor.black)
            provider = self.loaded.media_frames
            if provider is None:
                message = "Source media is missing" if self.loaded.source_status == "missing" else "Source media is unavailable"
                painter.setPen(Qt.GlobalColor.lightGray)
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, message)
                return
            source_index = int(self.loaded.result.frame_indices[self.frame])
            if isinstance(provider, MediaFrameProvider):
                source_index = 0 if provider._is_image else source_index
                rgb = provider.cached_frame(source_index)
            else:
                rgb = provider[self.frame] if self.frame < len(provider) else None
            if rgb is None:
                if self._display_image.isNull():
                    painter.setPen(Qt.GlobalColor.lightGray)
                    painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Loading source frame...")
                    return
                image = self._display_image
                image_width, image_height = image.width(), image.height()
            else:
                image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
                self._display_image = image
                image_width, image_height = rgb.shape[1], rgb.shape[0]
            sx, sy = self.width() / image_width, self.height() / image_height
            painter.drawImage(self.rect(), image)
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
        self.show_labels = True
        self.show_mesh = False
        self.mesh_opacity = 1.0
        self.hand_colors = ((0.1, 0.9, 1), (1, 0.8, 0.1))
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.grids = [gl.GLGridItem(color=(120, 140, 160, 55)) for _ in range(3)]
        for grid in self.grids: self.addItem(grid)
        self.axis = gl.GLLinePlotItem(mode="lines", width=2, antialias=True)
        self.axis_labels = [gl.GLTextItem(text=label) for label in ("X", "Y", "Z")]
        self.numeric_label_color = (190, 196, 204, 255)
        self.numeric_labels = []
        self.addItem(self.axis)
        for label in self.axis_labels: self.addItem(label)
        self.workspace = workspace_geometry("physical")
        self._configure_workspace()
        self.metric_center = np.asarray(metric_workspace_center_display(), dtype=np.float64)
        self.default_metric_distance = default_metric_camera_distance(self.opts["fov"])
        self.hand_view_target, self.hand_view_distance = whole_video_hand_framing(
            self.loaded.result.keypoints_3d, self.loaded.result.detected, self.opts["fov"]
        )
        self.fit_center = np.asarray(self.hand_view_target, dtype=np.float64)
        self.fit_distance = self.hand_view_distance
        self.setCameraPosition(pos=pg.Vector(*self.fit_center), distance=self.fit_distance, elevation=20, azimuth=45)
        self.joints = [gl.GLScatterPlotItem(), gl.GLScatterPlotItem()]
        self.bones = [gl.GLLinePlotItem(mode="lines", width=2, antialias=True), gl.GLLinePlotItem(mode="lines", width=2, antialias=True)]
        self.meshes = [gl.GLMeshItem(smooth=False, drawEdges=False), gl.GLMeshItem(smooth=False, drawEdges=False)]
        for item in (*self.joints, *self.bones, *self.meshes): self.addItem(item); item.setVisible(False)
        self.set_mesh_opacity(self.mesh_opacity)
        self.set_frame(0)

    @property
    def workspace_space_mode(self) -> str:
        return self.workspace.space_mode

    @property
    def workspace_bounds_m(self) -> np.ndarray:
        return self.workspace.bounds_array

    @property
    def workspace_center_raw(self) -> np.ndarray:
        return np.asarray(self.workspace.center_raw, dtype=np.float64)

    @property
    def workspace_side_length_m(self) -> float:
        return self.workspace.side_length_m

    @property
    def workspace_major_spacing_m(self) -> float:
        return self.workspace.major_spacing_m

    @property
    def workspace_minor_spacing_m(self) -> float:
        return self.workspace.minor_spacing_m
    def mousePressEvent(self, event):
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        super().mousePressEvent(event)
    def set_mesh_opacity(self, opacity):
        self.mesh_opacity = float(np.clip(opacity, 0.1, 1.0))
        gl_options = "opaque" if self.mesh_opacity >= 1.0 else "translucent"
        for hand_i, mesh in enumerate(self.meshes):
            mesh.setGLOptions(gl_options); mesh.setColor((*self.hand_colors[hand_i], self.mesh_opacity))
        self.update()
    def _geometry(self, frame):
        result = self.loaded.result; points = []
        visible = result.detected[frame]
        for hand_i in range(2):
            if not visible[hand_i] or (hand_i == 0 and not self.show_left) or (hand_i == 1 and not self.show_right): continue
            if self.show_joints or self.show_bones:
                points.append(np.asarray(result.keypoints_3d[frame, hand_i]) @ RAW_TO_VIEW.T)
            if self.show_mesh and self.loaded.has_mesh_topology and result.mano_vertices is not None:
                points.append(np.asarray(result.mano_vertices[frame, hand_i]) @ RAW_TO_VIEW.T)
        finite = np.concatenate(points) if points else np.empty((0, 3), dtype=np.float32)
        return finite[np.isfinite(finite).all(axis=1)]

    def _configure_workspace(self) -> None:
        for label in self.numeric_labels:
            self.removeItem(label)
        self.numeric_labels.clear()

        workspace = self.workspace
        side = workspace.side_length_m
        raw_center = np.asarray(workspace.center_raw)
        bounds = workspace.bounds_array
        if workspace.space_mode == "physical":
            plane_centers_raw = (
                (raw_center[0], raw_center[1], bounds[2, 0]),
                (raw_center[0], raw_center[1], raw_center[2]),
                (raw_center[0], raw_center[1], raw_center[2]),
            )
            axis_center_raw = np.zeros(3, dtype=np.float64)
            raw_segments = np.asarray((
                ((bounds[0, 0], 0.0, 0.0), (bounds[0, 1], 0.0, 0.0)),
                ((0.0, bounds[1, 1], 0.0), (0.0, bounds[1, 0], 0.0)),
                ((0.0, 0.0, bounds[2, 0]), (0.0, 0.0, bounds[2, 1])),
            ), dtype=np.float32)
            axis_endpoints = (
                (bounds[0, 1] + min(side * WORKSPACE_AXIS_LABEL_OFFSET_FRACTION,
                                    self.workspace_major_spacing_m * 0.5), 0.0, 0.0),
                (0.0, bounds[1, 0] - min(side * WORKSPACE_AXIS_LABEL_OFFSET_FRACTION,
                                         self.workspace_major_spacing_m * 0.5), 0.0),
                (0.0, 0.0, bounds[2, 1] + min(side * WORKSPACE_AXIS_LABEL_OFFSET_FRACTION,
                                              self.workspace_major_spacing_m * 0.5)),
            )
        else:
            plane_centers_raw = (workspace.center_raw,) * 3
            axis_center_raw = raw_center
            half_side = side / 2.0
            raw_segments = np.asarray((
                ((raw_center[0] - half_side, raw_center[1], raw_center[2]),
                 (raw_center[0] + half_side, raw_center[1], raw_center[2])),
                ((raw_center[0], raw_center[1] + half_side, raw_center[2]),
                 (raw_center[0], raw_center[1] - half_side, raw_center[2])),
                ((raw_center[0], raw_center[1], raw_center[2] - half_side),
                 (raw_center[0], raw_center[1], raw_center[2] + half_side)),
            ), dtype=np.float32)
            axis_label_offset = min(side * WORKSPACE_AXIS_LABEL_OFFSET_FRACTION,
                                    self.workspace_major_spacing_m * 0.5)
            axis_endpoints = (
                (raw_center[0] + half_side + axis_label_offset, raw_center[1], raw_center[2]),
                (raw_center[0], raw_center[1] - half_side - axis_label_offset, raw_center[2]),
                (raw_center[0], raw_center[1], raw_center[2] + half_side + axis_label_offset),
            )
        grid_specs = (
            (90, 1, 0, 0),
            None,
            (90, 0, 1, 0),
        )
        for grid, rotation, plane_center_raw in zip(self.grids, grid_specs, plane_centers_raw):
            grid.resetTransform()
            grid.setSize(side, side, 1.0)
            grid.setSpacing(self.workspace_minor_spacing_m, self.workspace_minor_spacing_m, 1.0)
            if rotation:
                grid.rotate(*rotation)
            grid.translate(*_view_point_from_raw(plane_center_raw))

        positions = (raw_segments @ RAW_TO_VIEW.T).reshape(-1, 3)
        colors = np.repeat(np.asarray(((1, 0.2, 0.2, 1), (0.2, 1, 0.2, 1),
                                       (0.3, 0.5, 1, 1)), dtype=np.float32), 2, axis=0)
        self.axis.setData(pos=np.ascontiguousarray(positions), color=colors, mode="lines")

        for label, endpoint in zip(self.axis_labels, axis_endpoints):
            label.setData(pos=_view_point_from_raw(endpoint), color=self.numeric_label_color,
                          font=QFont("Helvetica", 12))

        ticks = []
        bounds = workspace.bounds_array
        label_offset = min(side * WORKSPACE_LABEL_OFFSET_FRACTION,
                           self.workspace_major_spacing_m * 0.35)
        for axis, (low, high) in enumerate(bounds):
            step = workspace.major_spacing_m
            first = np.ceil((low - 1e-10) / step) * step
            last = np.floor((high + 1e-10) / step) * step
            for value in np.arange(first, last + step * 0.5, step):
                raw_point = axis_center_raw.copy()
                raw_point[axis] = value
                if axis == 0:
                    raw_point[1] -= label_offset
                elif axis == 1:
                    raw_point[0] += label_offset
                else:
                    raw_point[0] -= label_offset
                display_value = -value if axis == 1 else value
                if abs(display_value) < step * 1e-8:
                    display_value = 0.0
                ticks.append((f"{display_value * 100:g} cm", raw_point))
        for text, raw_point in ticks:
            label = gl.GLTextItem(text=text, pos=_view_point_from_raw(raw_point), color=self.numeric_label_color,
                                  font=QFont("Helvetica", 9),
                                  alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
            self.numeric_labels.append(label)
            self.addItem(label)
        self._update_reference_visibility()

    def set_workspace(self, space_mode: str, bounds_m: np.ndarray | None = None) -> None:
        self.workspace = workspace_geometry(space_mode, bounds_m)
        self._configure_workspace()

    def set_labels_visible(self, visible: bool) -> None:
        self.show_labels = bool(visible)
        self._update_reference_visibility()

    def space_overview(self) -> bool:
        if self.workspace_space_mode == "physical":
            self.fit_center = self.workspace.display_center
            self.fit_distance = self.default_metric_distance
            self.setCameraPosition(pos=pg.Vector(*self.fit_center), distance=self.fit_distance,
                                   elevation=20, azimuth=45)
            return True
        return self.fit_camera_to_points(self.workspace.display_corners, reset_orientation=True)

    def _frame_camera(self, frame=None, reset_orientation=False):
        finite = self._geometry(self.frame if frame is None else frame)
        return self.fit_camera_to_points(finite, reset_orientation=reset_orientation)

    def fit_camera_to_points(self, points, aspect=None, reset_orientation=False):
        points = np.asarray(points)
        if points.size == 0:
            return False
        finite = points[np.isfinite(points).all(axis=1)]
        if not len(finite):
            return False
        center = (finite.min(axis=0) + finite.max(axis=0)) / 2
        radius = float(np.linalg.norm(finite.max(axis=0) - finite.min(axis=0)) / 2)
        if aspect is None:
            aspect = self.width() / self.height() if self.width() > 0 and self.height() > 0 else DEFAULT_WORKSPACE_ASPECT
        distance = _camera_distance_for_radius(radius, self.opts["fov"], aspect)
        self.fit_center = center
        self.fit_distance = distance
        options = {"distance": distance, "pos": pg.Vector(*center)}
        if reset_orientation:
            options.update(elevation=20, azimuth=45)
        self.setCameraPosition(**options)
        return True

    def fit_all(self):
        return self.fit_camera_to_points(self._geometry(self.frame))

    def fit_hands(self):
        return self.fit_all()

    def apply_hand_view(self, target, distance):
        self.fit_center = np.asarray(target, dtype=np.float64)
        self.fit_distance = float(distance)
        self.setCameraPosition(pos=pg.Vector(*self.fit_center), distance=self.fit_distance,
                               elevation=20, azimuth=45)
        self.update()

    def hand_view(self):
        self.apply_hand_view(self.hand_view_target, self.hand_view_distance)

    def metric_overview(self):
        self.space_overview()

    def reset_camera(self):
        self.hand_view()

    def _preset(self, raw_direction):
        direction = np.asarray(raw_direction, dtype=np.float32) @ RAW_TO_VIEW.T
        azimuth = float(np.degrees(np.arctan2(direction[1], direction[0])))
        elevation = float(np.degrees(np.arctan2(direction[2], np.hypot(direction[0], direction[1]))))
        current = self.cameraParams()
        center = np.asarray(tuple(current["center"]), dtype=np.float64)
        self.setCameraPosition(pos=pg.Vector(*center), distance=current["distance"],
                               azimuth=azimuth, elevation=elevation)
    def front_view(self): self._preset((0, 0, -1))
    def side_view(self): self._preset((-1, 0, 0))
    def top_view(self): self._preset((0, -1, 0))
    def _update_reference_visibility(self):
        for grid in self.grids: grid.setVisible(self.show_grid)
        for label in self.numeric_labels: label.setVisible(self.show_labels)
        self.axis.setVisible(self.show_axis)
        for label in self.axis_labels: label.setVisible(self.show_axis)
    def set_frame(self, frame):
        self.frame = max(0, min(frame, self.loaded.frame_count - 1)); result = self.loaded.result; colors = tuple((*color, 1) for color in self.hand_colors)
        for hand_i in range(2):
            visible = bool(result.detected[self.frame, hand_i]) and not (hand_i == 0 and not self.show_left) and not (hand_i == 1 and not self.show_right)
            for item in (self.joints[hand_i], self.bones[hand_i], self.meshes[hand_i]): item.setVisible(False)
            has_visible_geometry = (self.show_joints or self.show_bones
                                    or (self.show_mesh and self.loaded.has_mesh_topology))
            if not visible or not has_visible_geometry: continue
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
    def __init__(self, result_path: Path | ResultBundle):
        super().__init__()
        if isinstance(result_path, ResultBundle):
            self.loaded = load_bundle_result(result_path)
        elif Path(result_path).suffix.lower() == ".cfh":
            self.loaded = load_bundle_result(load_result_bundle(result_path))
        else:
            self.loaded = load_npz_result(result_path)
        self._media_provider_released = False
        self.setWindowTitle("CalibFreeHands — Result"); self.setMinimumSize(960, 640)
        screen = QGuiApplication.primaryScreen().availableGeometry(); self.resize(min(1400, int(screen.width() * .8)), min(900, int(screen.height() * .8)))
        self._arrow_mode = "frame"
        self.overlay = _OverlayCanvas(self.loaded); self.gl_view = _GLResultView(self.loaded); split = QSplitter(Qt.Orientation.Horizontal); split.addWidget(self.overlay); split.addWidget(self.gl_view); split.setSizes([600, 600])
        self.slider = QSlider(Qt.Orientation.Horizontal); self.slider.setRange(0, max(0, self.loaded.frame_count - 1)); self.slider.valueChanged.connect(self._set_frame)
        self.status = QLabel(); self.coordinate_info = QLabel("Camera space · metres   X: image right   Y: image down   Z: forward   ·   Fixed metric grid · raw coordinates unchanged")
        self.timer = QTimer(self); self.timer.timeout.connect(self._next); visibility = QHBoxLayout(); view = QHBoxLayout(); playback = QHBoxLayout()
        self.joint_checkbox = QCheckBox("Joint"); self.joint_checkbox.setChecked(self.gl_view.show_joints); self.joint_checkbox.toggled.connect(self._on_joint_toggled); visibility.addWidget(self.joint_checkbox)
        self.bone_checkbox = QCheckBox("Bone"); self.bone_checkbox.setChecked(self.gl_view.show_bones); self.bone_checkbox.toggled.connect(self._on_bone_toggled); visibility.addWidget(self.bone_checkbox)
        for label, attr in (("Left Hand", "show_left"), ("Right Hand", "show_right"), ("Grid", "show_grid"), ("Axis", "show_axis")):
            check = QCheckBox(label); check.setChecked(getattr(self.gl_view, attr)); check.toggled.connect(lambda value, a=attr: self._toggle(a, value)); visibility.addWidget(check)
        self.mesh_checkbox = QCheckBox("Mesh"); self.mesh_checkbox.setChecked(self.gl_view.show_mesh); self.mesh_checkbox.setEnabled(self.loaded.has_mesh_topology); self.mesh_checkbox.setToolTip("MANO mesh topology is unavailable for this result" if not self.loaded.has_mesh_topology else ""); self.mesh_checkbox.toggled.connect(self._on_mesh_toggled); visibility.addWidget(self.mesh_checkbox)
        opacity_label = QLabel("Opacity")
        self.mesh_opacity_slider = QSlider(Qt.Orientation.Horizontal); self.mesh_opacity_slider.setRange(10, 100); self.mesh_opacity_slider.setValue(100); self.mesh_opacity_slider.setEnabled(self.loaded.has_mesh_topology); self.mesh_opacity_slider.setFixedWidth(120)
        self.mesh_opacity_value = QLabel("100%")
        self.mesh_opacity_value.setFixedWidth(self.mesh_opacity_value.fontMetrics().horizontalAdvance("100%") + 4)
        self.mesh_opacity_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.mesh_opacity_slider.valueChanged.connect(self._set_mesh_opacity)
        opacity_label.setEnabled(self.loaded.has_mesh_topology); self.mesh_opacity_value.setEnabled(self.loaded.has_mesh_topology)
        self.opacity_container = QWidget(); self.opacity_container.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self.opacity_group = QHBoxLayout(self.opacity_container); self.opacity_group.setContentsMargins(0, 0, 0, 0); self.opacity_group.setSpacing(4)
        self.opacity_group.addWidget(opacity_label); self.opacity_group.addWidget(self.mesh_opacity_slider); self.opacity_group.addWidget(self.mesh_opacity_value)
        visibility.addWidget(self.opacity_container)
        previous = QPushButton("Previous"); previous.clicked.connect(self._previous); play = QPushButton("Play / Pause"); play.clicked.connect(self._play); nxt = QPushButton("Next"); nxt.clicked.connect(self._next_frame)
        for text, callback in (("Front", self.gl_view.front_view), ("Side", self.gl_view.side_view), ("Top", self.gl_view.top_view), ("Hand View", self.gl_view.hand_view), ("Metric Overview", self.gl_view.metric_overview), ("Fit All", self.gl_view.fit_all)):
            button = QPushButton(text); button.clicked.connect(callback); view.addWidget(button)
        playback.addWidget(previous); playback.addWidget(play); playback.addWidget(nxt)
        layout = QVBoxLayout(self); layout.addWidget(split, 1); layout.addWidget(self.slider); layout.addLayout(visibility); layout.addLayout(view); layout.addLayout(playback); layout.addWidget(self.status); layout.addWidget(self.coordinate_info)
        self.installEventFilter(self)
        for widget in self.findChildren(QWidget): widget.installEventFilter(self)
        self._update_status()

    def closeEvent(self, event):
        if not self._media_provider_released:
            self._media_provider_released = True
            if isinstance(self.loaded.media_frames, MediaFrameProvider):
                release_shared_media_provider(self.loaded.media_frames)
        super().closeEvent(event)

    def _toggle(self, attr, value):
        if hasattr(self.overlay, attr): setattr(self.overlay, attr, value)
        setattr(self.gl_view, attr, value); self.overlay.update(); self.gl_view.set_frame(self.slider.value())
        if attr == "show_grid":
            self.gl_view.set_labels_visible(value)
    @staticmethod
    def _set_checked_silently(checkbox, value):
        blocker = QSignalBlocker(checkbox)
        checkbox.setChecked(value)
        del blocker
    def _on_joint_toggled(self, value):
        if value and self.mesh_checkbox.isChecked():
            self._set_checked_silently(self.mesh_checkbox, False); self._toggle("show_mesh", False)
        self._toggle("show_joints", value)
    def _on_bone_toggled(self, value):
        if value and self.mesh_checkbox.isChecked():
            self._set_checked_silently(self.mesh_checkbox, False); self._toggle("show_mesh", False)
        self._toggle("show_bones", value)
    def _on_mesh_toggled(self, value):
        if value:
            self._set_checked_silently(self.joint_checkbox, False); self._set_checked_silently(self.bone_checkbox, False)
            self._toggle("show_joints", False); self._toggle("show_bones", False)
        self._toggle("show_mesh", value)
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
        source = " · Source media missing" if self.loaded.source_status == "missing" else (" · Source media changed" if self.loaded.source_status == "changed" else "")
        self.status.setText(f"Frame {self.slider.value() + 1} / {self.loaded.frame_count}{suffix}{source}")
