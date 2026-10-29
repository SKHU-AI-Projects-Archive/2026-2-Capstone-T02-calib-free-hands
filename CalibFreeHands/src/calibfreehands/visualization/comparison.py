from __future__ import annotations

import colorsys
import csv
import json
from pathlib import Path

import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtCore import QEvent, Qt, QTimer, QSignalBlocker
from PySide6.QtWidgets import (
    QCheckBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QSplitter, QStackedWidget, QVBoxLayout, QWidget,
)

from calibfreehands.media.ffmpeg import FFmpegDecoder
from calibfreehands.media.probe import probe_media
from .topology import HAND_CONNECTIONS, RAW_TO_VIEW
from calibfreehands.pipeline.result_bundle import ResultBundle
from calibfreehands.pipeline.metrics import compare_results, metric_registry
from calibfreehands.joint_schema import JOINT_NAMES
from .viewer import _GLResultView, _OverlayCanvas, load_bundle_result, load_npz_result

METRIC_FIELDS = ("source_identity", "reference_result_id", "candidate_result_id", "candidate_name", "metric_id",
                 "metric_name", "category", "protocol_version", "frame_index", "timestamp_s", "hand",
                 "joint_index", "joint_name", "value", "unit", "valid_status")


def _finite(value):
    return float(value) if np.isfinite(value) else None


def _session_results(session_path: Path):
    session_path = Path(session_path).resolve()
    data = json.loads(session_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError(f"Unsupported comparison session schema: {data.get('schema_version')}")
    configurations = data["configurations"]
    if not configurations:
        raise ValueError("comparison session has no configurations")
    results = []
    for config in configurations:
        entry = data["final_results"].get(config["id"])
        if entry is None:
            continue
        result_path = (session_path.parent / entry["path"]).resolve()
        if session_path.parent not in result_path.parents:
            raise ValueError("comparison result path escapes the session directory")
        results.append((config, load_npz_result(result_path, decode_media=False)))
    if not results:
        raise ValueError("comparison session contains no saved results")
    source = data.get("source_media") or ""
    media = None
    if source and Path(source).is_file():
        try:
            info = probe_media(Path(source))
            media = [frame.rgb for frame in FFmpegDecoder().decode_rgb(Path(source), info.width, info.height, info.frame_rate)]
        except (OSError, ValueError, RuntimeError):
            media = None
    for _, loaded in results:
        loaded.media_frames = media
    return data, results


def _bundle_results(bundles: list[ResultBundle]):
    if not bundles:
        raise ValueError("no result bundles were selected")
    source_ids = {bundle.metadata.get("source_media_sha256", "").casefold() for bundle in bundles}
    if "" in source_ids or len(source_ids) != 1:
        raise ValueError("Selected results were generated from different source media.")
    results, configurations = [], []
    for index, bundle in enumerate(bundles):
        metadata = bundle.metadata
        config = {
            "id": metadata.get("configuration_id", f"bundle-{index + 1}"),
            "name": metadata.get("configuration_name") or
                    f"{metadata['calibration_display_name']} + {metadata['reconstruction_display_name']}",
            "calibration_model_id": metadata["calibration_model_id"],
            "reconstruction_model_id": metadata["reconstruction_model_id"],
        }
        configurations.append(config)
        results.append((config, load_bundle_result(bundle, decode_media=False)))
    media_loaded = load_bundle_result(bundles[0], decode_media=True)
    for _, loaded in results:
        loaded.media_frames = media_loaded.media_frames
        loaded.source_status = media_loaded.source_status
    session = {
        "schema_version": 1, "configurations": configurations, "final_results": {},
        "source_media": bundles[0].metadata.get("source_media_path", ""),
        "source_media_sha256": bundles[0].metadata["source_media_sha256"],
        "reference_configuration_id": bundles[0].metadata.get("reference_configuration_id"),
    }
    return session, results


def _comparison_results(source):
    return _bundle_results(source) if isinstance(source, (list, tuple)) else _session_results(source)


def _color_for_index(index: int):
    palette = ((0.91, 0.30, 0.24), (0.20, 0.56, 0.86), (0.20, 0.68, 0.42),
               (0.63, 0.42, 0.78), (0.93, 0.61, 0.18), (0.17, 0.70, 0.70))
    if index < len(palette):
        return palette[index]
    return colorsys.hsv_to_rgb((index * 0.61803398875) % 1, 0.72, 0.9)


def calculate_reference_metrics(session_path: Path | list[ResultBundle]) -> list[dict]:
    data, entries = _comparison_results(session_path)
    reference_id = data.get("reference_configuration_id")
    reference = next((loaded.result for config, loaded in entries if config["id"] == reference_id), None)
    if reference is None:
        raise ValueError("session reference configuration has no saved result")
    rows = []
    for config, loaded in entries:
        if config["id"] == reference_id:
            continue
        result = loaded.result
        for metrics in compare_results(reference, result):
            joint_errors_m = metrics.joint_errors_mm / 1000.0
            finite_errors = joint_errors_m[np.isfinite(joint_errors_m)]
            depth_error_m = (result.keypoints_3d[np.flatnonzero(result.frame_indices == metrics.frame_index)[0],
                                                 ("left", "right").index(metrics.hand), :, 2] -
                             reference.keypoints_3d[np.flatnonzero(reference.frame_indices == metrics.frame_index)[0],
                                                    ("left", "right").index(metrics.hand), :, 2])
            depth_error_m = depth_error_m[np.isfinite(depth_error_m)]
            metric_values = {key: value for key, value in metrics.metrics.items()}
            metric_status = {key: value.value for key, value in metrics.metric_status.items()}
            row = {"configuration_id": config["id"], "configuration_name": config["name"],
                   "reference_configuration_id": reference_id, "frame_index": metrics.frame_index,
                   "timestamp_s": _finite(metrics.timestamp_s), "hand": metrics.hand,
                   "joint_count": metrics.valid_joint_count,
                   "joint_distance_mean_m": float(finite_errors.mean()),
                   "joint_distance_median_m": float(np.median(finite_errors)),
                   "joint_distance_rmse_m": float(np.sqrt(np.mean(finite_errors ** 2))),
                   "depth_error_mean_m": float(np.abs(depth_error_m).mean()),
                   "depth_error_rmse_m": float(np.sqrt(np.mean(depth_error_m ** 2))),
                   "translation_distance_m": (metric_values["translation_error"] / 1000
                                              if metric_values["translation_error"] is not None else None),
                   "translation_depth_error_m": (metric_values["translation_z_error"] / 1000
                                                 if metric_values["translation_z_error"] is not None else None),
                   "metric_values": metric_values, "metric_status": metric_status,
                   "per_joint_error_mm": [float(value) if np.isfinite(value) else None
                                          for value in metrics.joint_errors_mm]}
            rows.append(row)
    return rows


def export_metrics_json(session_path: Path | list[ResultBundle], output_path: Path) -> Path:
    rows = _metric_export_rows(session_path)
    output_path = Path(output_path)
    output_path.write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
    return output_path


def export_metrics_csv(session_path: Path | list[ResultBundle], output_path: Path) -> Path:
    rows = _metric_export_rows(session_path)
    output_path = Path(output_path)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _metric_export_rows(session_path: Path | list[ResultBundle]) -> list[dict]:
    data, entries = _comparison_results(session_path)
    reference_id = data.get("reference_configuration_id")
    reference_entry = next(((config, loaded.result) for config, loaded in entries
                            if config["id"] == reference_id), None)
    if reference_entry is None:
        raise ValueError("session reference configuration has no saved result")
    _, reference = reference_entry
    source_identity = data.get("source_media_sha256") or data.get("source_media") or "unknown"
    definitions = metric_registry()
    records = []
    for config, loaded in entries:
        if config["id"] == reference_id:
            continue
        for pair in compare_results(reference, loaded.result):
            for definition in definitions:
                status = pair.metric_status[definition.metric_id]
                value = pair.metrics.get(definition.metric_id)
                if definition.metric_id == "per_joint_error":
                    for joint_index, joint_value in enumerate(value):
                        joint_status = status.value if joint_value is not None else "UNAVAILABLE_FOR_DATA"
                        records.append({
                            "source_identity": source_identity, "reference_result_id": reference_id,
                            "candidate_result_id": config["id"], "candidate_name": config["name"],
                            "metric_id": definition.metric_id, "metric_name": definition.name,
                            "category": definition.category, "protocol_version": definition.protocol_version,
                            "frame_index": pair.frame_index, "timestamp_s": _finite(pair.timestamp_s),
                            "hand": pair.hand, "joint_index": joint_index,
                            "joint_name": JOINT_NAMES[joint_index], "value": joint_value,
                            "unit": definition.unit, "valid_status": joint_status,
                        })
                    continue
                records.append({
                    "source_identity": source_identity, "reference_result_id": reference_id,
                    "candidate_result_id": config["id"], "candidate_name": config["name"],
                    "metric_id": definition.metric_id, "metric_name": definition.name,
                    "category": definition.category, "protocol_version": definition.protocol_version,
                    "frame_index": pair.frame_index, "timestamp_s": _finite(pair.timestamp_s),
                    "hand": pair.hand, "joint_index": None, "joint_name": None,
                    "value": value, "unit": definition.unit, "valid_status": status.value,
                })
    return records


class _LinkedResultView(_GLResultView):
    def __init__(self, loaded, color, changed):
        self.configuration_color = color
        self.camera_changed = changed
        super().__init__(loaded)
        self.hand_colors = (color, color)
        self.set_frame(self.frame)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self.camera_changed(self)

    def wheelEvent(self, event):
        super().wheelEvent(event)
        self.camera_changed(self)

    def set_mesh_opacity(self, opacity):
        super().set_mesh_opacity(opacity)
        if not hasattr(self, "loaded") or not hasattr(self, "configuration_color"):
            return
        for hand in range(2):
            alpha = _confidence_alpha(self.loaded.result, self.frame, hand)
            _set_mesh_color(self.meshes[hand], self.configuration_color, alpha * self.mesh_opacity)

    def set_frame(self, frame):
        super().set_frame(frame)
        if not hasattr(self, "loaded"):
            return
        for hand_index in range(2):
            alpha = _confidence_alpha(self.loaded.result, self.frame, hand_index)
            color = (*self.configuration_color, alpha)
            if self.joints[hand_index].visible():
                self.joints[hand_index].setData(color=color)
            if self.bones[hand_index].visible():
                self.bones[hand_index].setData(color=color)
            if self.meshes[hand_index].visible():
                _set_mesh_color(self.meshes[hand_index], self.configuration_color, alpha * self.mesh_opacity)


def _confidence_alpha(result, frame, hand, confidence=None):
    if confidence is None:
        confidence = result.confidence[frame, hand] if result.confidence is not None else np.nan
    if not np.isfinite(confidence):
        return 0.65
    return float(0.2 + 0.8 * np.clip(confidence, 0.0, 1.0))


def _set_mesh_color(mesh, color, alpha):
    mesh.setGLOptions("opaque" if alpha >= 1.0 else "translucent")
    mesh.setColor((*color, alpha))


class _OverlayComparisonView(_GLResultView):
    missing_confidence_alpha = 0.65

    def __init__(self, loaded_results, colors):
        self.loaded_results = loaded_results
        self.configuration_colors = colors
        self.configuration_visible = [True] * len(loaded_results)
        self.extra_items = []
        super().__init__(loaded_results[0])
        self.hand_colors = (colors[0], colors[0])
        for index, loaded in enumerate(loaded_results[1:], 1):
            items = []
            for _hand in range(2):
                triplet = (gl.GLScatterPlotItem(), gl.GLLinePlotItem(mode="lines", width=2, antialias=True),
                           gl.GLMeshItem(smooth=False, drawEdges=False))
                for item in triplet:
                    self.addItem(item)
                    item.setVisible(False)
                items.append(triplet)
            self.extra_items.append(items)
        self.set_frame(self.frame)

    def confidence_alpha(self, frame, hand, confidence=None):
        return _confidence_alpha(self.loaded_results[0].result, frame, hand, confidence)

    def _geometry(self, frame):
        if not hasattr(self, "loaded_results"):
            return super()._geometry(frame)
        points = []
        for index, loaded in enumerate(self.loaded_results):
            if not self.configuration_visible[index]:
                continue
            result = loaded.result
            position = min(frame, loaded.frame_count - 1)
            for hand in range(2):
                if result.detected[position, hand] and not (hand == 0 and not self.show_left) and not (hand == 1 and not self.show_right):
                    points.append(result.keypoints_3d[position, hand] @ RAW_TO_VIEW.T)
                    if loaded.has_mesh_topology and result.mano_vertices is not None:
                        points.append(result.mano_vertices[position, hand] @ RAW_TO_VIEW.T)
        finite = np.concatenate(points) if points else np.empty((0, 3), dtype=np.float32)
        return finite[np.isfinite(finite).all(axis=1)]

    def set_configuration_visible(self, index, visible):
        self.configuration_visible[index] = bool(visible)
        self.set_frame(self.frame)

    def set_frame(self, frame):
        super().set_frame(frame)
        if not hasattr(self, "extra_items"):
            return
        for hand in range(2):
            alpha = _confidence_alpha(self.loaded.result, self.frame, hand)
            rgba = (*self.configuration_colors[0], alpha)
            if self.joints[hand].visible(): self.joints[hand].setData(color=rgba)
            if self.bones[hand].visible(): self.bones[hand].setData(color=rgba)
            if self.meshes[hand].visible(): _set_mesh_color(self.meshes[hand], self.configuration_colors[0], alpha * self.mesh_opacity)
        for item in (*self.joints, *self.bones, *self.meshes):
            item.setVisible(item.visible() and self.configuration_visible[0])
        for config_index, (loaded, hands) in enumerate(zip(self.loaded_results[1:], self.extra_items), 1):
            position = min(self.frame, loaded.frame_count - 1)
            result = loaded.result
            color = self.configuration_colors[config_index]
            for hand, (joints, bones, mesh) in enumerate(hands):
                for item in (joints, bones, mesh):
                    item.setVisible(False)
                visible = self.configuration_visible[config_index] and bool(result.detected[position, hand])
                visible &= not (hand == 0 and not self.show_left) and not (hand == 1 and not self.show_right)
                if not visible:
                    continue
                alpha = _confidence_alpha(result, position, hand)
                points = np.asarray(result.keypoints_3d[position, hand]) @ RAW_TO_VIEW.T
                if not np.isfinite(points).all():
                    continue
                rgba = (*color, alpha)
                if self.show_joints:
                    joints.setData(pos=np.ascontiguousarray(points, dtype=np.float32), size=8, color=rgba)
                    joints.setVisible(True)
                if self.show_bones:
                    bone_points = np.ascontiguousarray(np.concatenate([points[[a, b]] for a, b in HAND_CONNECTIONS]), dtype=np.float32)
                    bones.setData(pos=bone_points, color=rgba, mode="lines")
                    bones.setVisible(True)
                if self.show_mesh and loaded.has_mesh_topology:
                    vertices = np.asarray(result.mano_vertices[position, hand]) @ RAW_TO_VIEW.T
                    faces = result.mano_faces[hand] if result.mano_faces.ndim == 3 else result.mano_faces
                    if np.isfinite(vertices).all() and faces.min() >= 0 and faces.max() < len(vertices):
                        mesh.setMeshData(meshdata=gl.MeshData(vertexes=np.ascontiguousarray(vertices, dtype=np.float32), faces=np.ascontiguousarray(faces, dtype=np.int32)))
                        _set_mesh_color(mesh, color, alpha * self.mesh_opacity)
                        mesh.setVisible(True)

    def set_mesh_opacity(self, opacity):
        super().set_mesh_opacity(opacity)
        if not hasattr(self, "loaded_results"):
            return
        for hand in range(2):
            alpha = _confidence_alpha(self.loaded_results[0].result, self.frame, hand)
            _set_mesh_color(self.meshes[hand], self.configuration_colors[0], alpha * self.mesh_opacity)
        for config_index, (loaded, hands) in enumerate(zip(self.loaded_results[1:], self.extra_items), 1):
            for hand, (_, _, mesh) in enumerate(hands):
                alpha = _confidence_alpha(loaded.result, min(self.frame, loaded.frame_count - 1), hand)
                _set_mesh_color(mesh, self.configuration_colors[config_index], alpha * self.mesh_opacity)


class ComparisonViewer(QWidget):
    def __init__(self, session_path: Path | list[ResultBundle]):
        super().__init__()
        self.bundle_results = list(session_path) if isinstance(session_path, (list, tuple)) else None
        self.session_path = None if self.bundle_results is not None else Path(session_path).resolve()
        self.session, entries = _comparison_results(session_path)
        self.configurations = [config for config, _ in entries]
        self.loaded_results = [loaded for _, loaded in entries]
        self.configuration_colors = [_color_for_index(i) for i in range(len(entries))]
        self.mode = "overlay"
        self._syncing_camera = False
        self._arrow_mode = "frame"
        frame_counts = {loaded.frame_count for loaded in self.loaded_results}
        frame_indices = [loaded.result.frame_indices for loaded in self.loaded_results]
        timestamps = [loaded.result.timestamps for loaded in self.loaded_results]
        if (len(frame_counts) != 1 or any(not np.array_equal(frame_indices[0], frames) for frames in frame_indices[1:])
                or any(not np.allclose(timestamps[0], values, equal_nan=True) for values in timestamps[1:])):
            raise ValueError("comparison results must have matching frame indices, timestamps, and counts")
        self.setWindowTitle("CalibFreeHands - Comparison")
        self.setMinimumSize(1000, 680)
        self.source_overlay = _OverlayCanvas(self.loaded_results[0])
        self.overlay_view = _OverlayComparisonView(self.loaded_results, self.configuration_colors)
        self.split_views = [_LinkedResultView(loaded, color, self._sync_camera) for loaded, color in zip(self.loaded_results, self.configuration_colors)]
        self.views = QStackedWidget()
        overlay_splitter = QSplitter(Qt.Orientation.Horizontal)
        overlay_splitter.addWidget(self.source_overlay)
        overlay_splitter.addWidget(self.overlay_view)
        self.views.addWidget(overlay_splitter)
        split_widget = QWidget()
        split_layout = QGridLayout(split_widget)
        split_layout.setContentsMargins(0, 0, 0, 0)
        for index, (config, view) in enumerate(zip(self.configurations, self.split_views)):
            column_count = min(3, len(self.split_views))
            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.addWidget(QLabel(config["name"]))
            cell_layout.addWidget(view, 1)
            split_layout.addWidget(cell, index // column_count, index % column_count)
        self.views.addWidget(split_widget)
        self.mode_buttons = {}
        top = QHBoxLayout()
        top.addWidget(QLabel("View mode"))
        for mode in ("overlay", "split"):
            button = QPushButton(mode.title())
            button.setCheckable(True)
            button.setChecked(mode == self.mode)
            button.clicked.connect(lambda _checked=False, m=mode: self.set_mode(m))
            top.addWidget(button)
            self.mode_buttons[mode] = button
        top.addWidget(QLabel("Results"))
        self.visibility_checks = []
        for index, config in enumerate(self.configurations):
            check = QCheckBox(config["name"])
            check.setToolTip("Configuration color identifies the method; opacity reflects frame/hand confidence. Missing confidence uses a neutral opacity.")
            check.setStyleSheet(f"color: rgb({int(self.configuration_colors[index][0]*255)}, {int(self.configuration_colors[index][1]*255)}, {int(self.configuration_colors[index][2]*255)})")
            check.setChecked(True)
            check.toggled.connect(lambda value, i=index: self._set_configuration_visible(i, value))
            top.addWidget(check)
            self.visibility_checks.append(check)
        self.visibility_controls = top
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.loaded_results[0].frame_count - 1)
        self.slider.valueChanged.connect(self._set_frame)
        self.status = QLabel()
        self.joint_checkbox = QCheckBox("Joint"); self.joint_checkbox.setChecked(True); self.joint_checkbox.toggled.connect(self._set_joint)
        self.bone_checkbox = QCheckBox("Bone"); self.bone_checkbox.setChecked(True); self.bone_checkbox.toggled.connect(self._set_bone)
        self.mesh_checkbox = QCheckBox("Mesh"); self.mesh_checkbox.setEnabled(any(loaded.has_mesh_topology for loaded in self.loaded_results)); self.mesh_checkbox.toggled.connect(self._set_mesh)
        self.mesh_opacity_slider = QSlider(Qt.Orientation.Horizontal); self.mesh_opacity_slider.setRange(10, 100); self.mesh_opacity_slider.setValue(100); self.mesh_opacity_slider.setEnabled(self.mesh_checkbox.isEnabled()); self.mesh_opacity_slider.setFixedWidth(120); self.mesh_opacity_slider.valueChanged.connect(self._set_opacity)
        geometry = QHBoxLayout()
        for control in (self.joint_checkbox, self.bone_checkbox, self.mesh_checkbox, QLabel("Opacity"), self.mesh_opacity_slider): geometry.addWidget(control)
        for text, attr in (("Left Hand", "show_left"), ("Right Hand", "show_right"), ("Grid", "show_grid"), ("Axis", "show_axis")):
            check = QCheckBox(text); check.setChecked(True); check.toggled.connect(lambda value, name=attr: self._set_geometry(name, value)); geometry.addWidget(check)
        playback = QHBoxLayout()
        for text, callback in (("Previous", self._previous), ("Play / Pause", self._play), ("Next", self._next_frame)):
            button = QPushButton(text); button.clicked.connect(callback); playback.addWidget(button)
        self.timer = QTimer(self); self.timer.timeout.connect(self._advance)
        self.view_buttons = QHBoxLayout()
        for text, method in (("Front", "front_view"), ("Side", "side_view"), ("Top", "top_view"), ("Fit Hands", "fit_hands"), ("Reset View", "reset_camera")):
            button = QPushButton(text); button.clicked.connect(lambda _checked=False, m=method: self._apply_camera_preset(m)); self.view_buttons.addWidget(button)
        exports = QHBoxLayout()
        self.export_csv_button = QPushButton("Export metrics CSV"); self.export_csv_button.clicked.connect(self._export_csv)
        self.export_json_button = QPushButton("Export metrics JSON"); self.export_json_button.clicked.connect(self._export_json)
        self.metric_status = QLabel()
        self.metric_status.setToolTip("Camera-space differences from the selected Reference configuration; these are not ground-truth errors.")
        exports.addWidget(self.export_csv_button); exports.addWidget(self.export_json_button); exports.addWidget(self.metric_status, 1)
        has_reference = any(config["id"] == self.session.get("reference_configuration_id") for config in self.configurations)
        self.export_csv_button.setEnabled(has_reference); self.export_json_button.setEnabled(has_reference)
        if not has_reference:
            self.metric_status.setText("Select a saved reference configuration to export reference metrics")
        layout = QVBoxLayout(self); layout.addLayout(top); layout.addWidget(self.views, 1); layout.addWidget(self.slider); layout.addLayout(geometry); layout.addLayout(self.view_buttons); layout.addLayout(playback); layout.addWidget(self.status); layout.addLayout(exports)
        self.installEventFilter(self)
        for widget in self.findChildren(QWidget): widget.installEventFilter(self)
        self._set_frame(0)
        self._sync_camera(self.split_views[0])

    def set_mode(self, mode):
        if mode not in ("overlay", "split"):
            raise ValueError(f"unknown comparison view mode: {mode}")
        self.mode = mode
        self.views.setCurrentIndex(0 if mode == "overlay" else 1)
        for key, button in self.mode_buttons.items():
            blocker = QSignalBlocker(button); button.setChecked(key == mode); del blocker
        self._set_frame(self.slider.value())

    def _set_configuration_visible(self, index, visible):
        self.overlay_view.set_configuration_visible(index, visible)
        self.split_views[index].setVisible(visible)

    def _set_geometry(self, attr, value):
        setattr(self.source_overlay, attr, value) if hasattr(self.source_overlay, attr) else None
        self.source_overlay.update()
        for view in [self.overlay_view, *self.split_views]:
            setattr(view, attr, value)
            view.set_frame(self.slider.value())

    def _set_mesh(self, value):
        if value:
            for check in (self.joint_checkbox, self.bone_checkbox):
                blocker = QSignalBlocker(check); check.setChecked(False); del blocker
                self._set_geometry("show_joints" if check is self.joint_checkbox else "show_bones", False)
        for view in [self.overlay_view, *self.split_views]:
            view.show_mesh = value
            view.set_frame(self.slider.value())

    def _set_joint(self, value):
        if value and self.mesh_checkbox.isChecked():
            blocker = QSignalBlocker(self.mesh_checkbox); self.mesh_checkbox.setChecked(False); del blocker
            self._set_mesh(False)
        self._set_geometry("show_joints", value)

    def _set_bone(self, value):
        if value and self.mesh_checkbox.isChecked():
            blocker = QSignalBlocker(self.mesh_checkbox); self.mesh_checkbox.setChecked(False); del blocker
            self._set_mesh(False)
        self._set_geometry("show_bones", value)

    def _set_opacity(self, value):
        for view in [self.overlay_view, *self.split_views]: view.set_mesh_opacity(value / 100)

    def _set_frame(self, frame):
        self.source_overlay.set_frame(frame)
        self.overlay_view.set_frame(frame)
        for view in self.split_views: view.set_frame(frame)
        timestamp = self.loaded_results[0].result.timestamps[frame]
        suffix = f" · {float(timestamp):.3f} s" if np.isfinite(timestamp) else ""
        source = " · Source media missing" if self.loaded_results[0].source_status == "missing" else (" · Source media changed" if self.loaded_results[0].source_status == "changed" else "")
        self.status.setText(f"Frame {frame + 1} / {self.loaded_results[0].frame_count}{suffix}{source}")

    def _sync_camera(self, source):
        if self._syncing_camera or not self.split_views:
            return
        self._syncing_camera = True
        try:
            params = source.cameraParams()
            for view in self.split_views:
                if view is not source: view.setCameraParams(**params)
        finally:
            self._syncing_camera = False

    def _apply_camera_preset(self, method):
        views = [self.overlay_view] if self.mode == "overlay" else self.split_views
        if not views: return
        getattr(views[0], method)()
        if self.mode == "split": self._sync_camera(views[0])

    def _previous(self): self.slider.setValue(max(0, self.slider.value() - 1))
    def _next_frame(self): self.slider.setValue(min(self.slider.maximum(), self.slider.value() + 1))
    def _advance(self): self.slider.setValue((self.slider.value() + 1) % max(1, self.slider.maximum() + 1))
    def _play(self): self.timer.stop() if self.timer.isActive() else self.timer.start(100)

    def _export_csv(self): self._export("csv")
    def _export_json(self): self._export("json")
    def _export(self, kind):
        if self.session_path is not None:
            initial = self.session_path.with_suffix(f".{kind}")
        else:
            source_name = self.bundle_results[0].metadata.get("source_media_name", "comparison")
            initial = Path(self.bundle_results[0].metadata["source_media_path"]).with_name(f"{Path(source_name).stem}-metrics.{kind}")
        path, _ = QFileDialog.getSaveFileName(self, "Export reference metrics", str(initial), f"{kind.upper()} (*.{kind})")
        if not path: return
        try:
            (export_metrics_csv if kind == "csv" else export_metrics_json)(self.bundle_results or self.session_path, Path(path))
            self.metric_status.setText(f"Saved {Path(path).name}")
        except (OSError, ValueError) as exc:
            self.metric_status.setText(f"Export failed: {exc}")

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.MouseButtonPress:
            self._arrow_mode = "orbit" if any(watched is view or view.isAncestorOf(watched) for view in self.split_views + [self.overlay_view]) else "frame"
        elif event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Space:
                self._play(); event.accept(); return True
            if self._arrow_mode == "frame" and event.key() == Qt.Key.Key_Left: self._previous(); event.accept(); return True
            if self._arrow_mode == "frame" and event.key() == Qt.Key.Key_Right: self._next_frame(); event.accept(); return True
            if self._arrow_mode == "orbit" and event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
                azim, elev = {Qt.Key.Key_Left: (-3, 0), Qt.Key.Key_Right: (3, 0), Qt.Key.Key_Up: (0, -3), Qt.Key.Key_Down: (0, 3)}[event.key()]
                target = self.split_views[0] if self.mode == "split" else self.overlay_view
                target.orbit(azim, elev)
                if self.mode == "split": self._sync_camera(target)
                event.accept(); return True
        return super().eventFilter(watched, event)
