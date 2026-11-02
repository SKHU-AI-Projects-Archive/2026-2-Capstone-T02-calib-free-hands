from __future__ import annotations

import colorsys
import csv
import json
from pathlib import Path
from time import monotonic

import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtCore import QEvent, QPoint, Qt, QTimer, QSignalBlocker
from PySide6.QtGui import QAction, QActionGroup, QImage
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QColorDialog, QComboBox, QDialog, QFileDialog, QGridLayout,
    QHBoxLayout, QLabel, QMainWindow, QMenu, QPushButton, QScrollArea, QSizePolicy,
    QSlider, QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem, QToolButton,
    QVBoxLayout, QWidget, QWidgetAction,
)

from calibfreehands.media.provider import MediaFrameProvider, shared_media_provider
from .topology import HAND_CONNECTIONS, RAW_TO_VIEW
from calibfreehands.pipeline.result_bundle import ResultBundle
from calibfreehands.pipeline.metrics import compare_results, metric_registry
from calibfreehands.joint_schema import JOINT_NAMES
from .viewer import _GLResultView, _OverlayCanvas, load_bundle_result, load_npz_result
from .comparison_presentation import _JointAnnotationLayer, _ViewportPanel
from .playback import PlaybackClock

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
        loaded = load_npz_result(result_path, decode_media=False)
        run_relative = entry.get("run_json")
        if run_relative:
            run_path = (session_path.parent / run_relative).resolve()
            if session_path.parent not in run_path.parents:
                raise ValueError("comparison provenance path escapes the session directory")
            try:
                loaded.metadata = {"run_provenance": json.loads(run_path.read_text(encoding="utf-8"))}
            except (OSError, ValueError):
                pass
        results.append((config, loaded))
    if not results:
        raise ValueError("comparison session contains no saved results")
    source = data.get("source_media") or ""
    media, source_status = shared_media_provider(Path(source), data.get("source_media_sha256")) if source else (None, "missing")
    for _, loaded in results:
        loaded.media_frames = media
        loaded.source_status = source_status
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
        base_id = metadata.get("configuration_id", f"bundle-{index + 1}")
        config_id = base_id
        duplicate = 2
        while any(config["id"] == config_id for config in configurations):
            config_id = f"{base_id}#{duplicate}"
            duplicate += 1
        config = {
            "id": config_id,
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


def _hand_variant(color, hand):
    r, g, b = color
    return tuple(min(1.0, channel * 1.18 + 0.08) for channel in (r, g, b)) if hand == 0 else tuple(channel * 0.78 for channel in (r, g, b))


def _confidence_color(confidence):
    value = float(np.clip(confidence, 0, 1))
    return (value, 0.25 + 0.55 * (1 - value), 1 - value)


def calculate_reference_metrics(session_path: Path | list[ResultBundle], reference_id=None) -> list[dict]:
    data, entries = _comparison_results(session_path)
    reference_id = reference_id or data.get("reference_configuration_id")
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


def export_metrics_json(session_path: Path | list[ResultBundle], output_path: Path, reference_id=None) -> Path:
    rows = _metric_export_rows(session_path, reference_id)
    output_path = Path(output_path)
    output_path.write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
    return output_path


def export_metrics_csv(session_path: Path | list[ResultBundle], output_path: Path, reference_id=None) -> Path:
    rows = _metric_export_rows(session_path, reference_id)
    output_path = Path(output_path)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _metric_export_rows(session_path: Path | list[ResultBundle], reference_id=None) -> list[dict]:
    data, entries = _comparison_results(session_path)
    reference_id = reference_id or data.get("reference_configuration_id")
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
        self.confidence_opacity_enabled = True
        self.confidence_color_mode = "configuration"
        super().__init__(loaded)
        self.hand_colors = (_hand_variant(color, 0), _hand_variant(color, 1))
        self.set_frame(self.frame)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self.camera_changed(self)

    def wheelEvent(self, event):
        super().wheelEvent(event)
        self.camera_changed(self)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            self.setStyleSheet("border: 2px solid palette(highlight);")
            event.acceptProposedAction()
        else: event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet("")
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        self.setStyleSheet("")
        if event.mimeData().hasText() and hasattr(self, "viewer"):
            self.viewer.reorder_result(event.mimeData().text(), self.configuration_id)
            event.acceptProposedAction()
        else: event.ignore()

    def set_mesh_opacity(self, opacity):
        super().set_mesh_opacity(opacity)
        if not hasattr(self, "loaded") or not hasattr(self, "configuration_color"):
            return
        for hand in range(2):
            alpha = _confidence_alpha(self.loaded.result, self.frame, hand) if self.confidence_opacity_enabled else 1.0
            _set_mesh_color(self.meshes[hand], self.hand_colors[hand], alpha * self.mesh_opacity)

    def set_frame(self, frame):
        super().set_frame(frame)
        if not hasattr(self, "loaded"):
            return
        for hand_index in range(2):
            alpha = _confidence_alpha(self.loaded.result, self.frame, hand_index) if self.confidence_opacity_enabled else 1.0
            base = self.hand_colors[hand_index]
            if self.confidence_color_mode == "confidence" and self.loaded.result.confidence is not None:
                confidence = self.loaded.result.confidence[self.frame, hand_index]
                base = _confidence_color(confidence) if np.isfinite(confidence) else (0.55, 0.55, 0.55)
            color = (*base, alpha)
            if self.joints[hand_index].visible():
                self.joints[hand_index].setData(color=color)
            if self.bones[hand_index].visible():
                self.bones[hand_index].setData(color=color)
            if self.meshes[hand_index].visible():
                _set_mesh_color(self.meshes[hand_index], base, alpha * self.mesh_opacity)


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
        self.confidence_color_mode = "configuration"
        self.confidence_opacity_enabled = True
        self.error_vectors_enabled = False
        self.error_candidate = 1 if len(loaded_results) > 1 else 0
        self.reference_index = 0
        self.error_items = []
        self.extra_items = []
        super().__init__(loaded_results[0])
        self.error_items = [gl.GLLinePlotItem(mode="lines", width=2, antialias=True, color=(1, 1, 0, 1))
                            for _ in range(2)]
        for item in self.error_items:
            self.addItem(item)
            item.setVisible(False)
        self.hand_colors = (_hand_variant(colors[0], 0), _hand_variant(colors[0], 1))
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
        if index == 0:
            for item in (*self.joints, *self.bones, *self.meshes):
                item.setVisible(item.visible() and self.configuration_visible[0])
        else:
            for item in self.extra_items[index - 1]:
                for visual in item:
                    if not visible:
                        visual.setVisible(False)

    def set_frame(self, frame):
        super().set_frame(frame)
        if not hasattr(self, "extra_items"):
            return
        for hand in range(2):
            alpha = _confidence_alpha(self.loaded.result, self.frame, hand) if self.confidence_opacity_enabled else 1.0
            base = self.configuration_colors[0]
            if self.confidence_color_mode == "confidence" and self.loaded.result.confidence is not None:
                confidence = self.loaded.result.confidence[self.frame, hand]
                base = _confidence_color(confidence) if np.isfinite(confidence) else (0.55, 0.55, 0.55)
            else:
                base = _hand_variant(base, hand)
            rgba = (*base, alpha)
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
                has_visible_geometry = (self.show_joints or self.show_bones
                                        or (self.show_mesh and loaded.has_mesh_topology))
                if not visible or not has_visible_geometry:
                    continue
                alpha = _confidence_alpha(result, position, hand) if self.confidence_opacity_enabled else 1.0
                hand_color = _hand_variant(color, hand)
                if self.confidence_color_mode == "confidence" and result.confidence is not None:
                    confidence = result.confidence[position, hand]
                    hand_color = _confidence_color(confidence) if np.isfinite(confidence) else (0.55, 0.55, 0.55)
                points = np.asarray(result.keypoints_3d[position, hand]) @ RAW_TO_VIEW.T
                if not np.isfinite(points).all():
                    continue
                rgba = (*hand_color, alpha)
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
                        _set_mesh_color(mesh, hand_color, alpha * self.mesh_opacity)
                        mesh.setVisible(True)
        if self.error_vectors_enabled:
            self._update_error_vectors()

    def _update_error_vectors(self):
        if not self.error_items:
            return
        if (not self.error_vectors_enabled or self.error_candidate == self.reference_index
                or not 0 <= self.error_candidate < len(self.loaded_results)):
            for item in self.error_items:
                item.setVisible(False)
            return
        ref = self.loaded_results[self.reference_index].result
        candidate = self.loaded_results[self.error_candidate].result
        for item in self.error_items:
            item.setVisible(False)
        ref_pos = np.flatnonzero(ref.frame_indices == ref.frame_indices[min(self.frame, len(ref.frame_indices)-1)])
        candidate_pos = min(self.frame, len(candidate.frame_indices)-1)
        ref_pos = int(ref_pos[0])
        for hand in range(2):
            if (hand == 0 and not self.show_left) or (hand == 1 and not self.show_right):
                continue
            if not ref.detected[ref_pos, hand] or not candidate.detected[candidate_pos, hand]:
                continue
            a, b = ref.keypoints_3d[ref_pos, hand], candidate.keypoints_3d[candidate_pos, hand]
            valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
            if valid.any():
                points = np.stack((a[valid] @ RAW_TO_VIEW.T, b[valid] @ RAW_TO_VIEW.T), axis=1).reshape(-1, 3)
                self.error_items[hand].setData(pos=np.asarray(points, dtype=np.float32), color=(1, 1, 0, 1), mode="lines")
                self.error_items[hand].setVisible(True)

    def set_mesh_opacity(self, opacity):
        super().set_mesh_opacity(opacity)
        if not hasattr(self, "loaded_results"):
            return
        for hand in range(2):
            alpha = _confidence_alpha(self.loaded_results[0].result, self.frame, hand) if self.confidence_opacity_enabled else 1.0
            _set_mesh_color(self.meshes[hand], _hand_variant(self.configuration_colors[0], hand), alpha * self.mesh_opacity)
        for config_index, (loaded, hands) in enumerate(zip(self.loaded_results[1:], self.extra_items), 1):
            for hand, (_, _, mesh) in enumerate(hands):
                alpha = _confidence_alpha(loaded.result, min(self.frame, loaded.frame_count - 1), hand) if self.confidence_opacity_enabled else 1.0
                _set_mesh_color(mesh, _hand_variant(self.configuration_colors[config_index], hand), alpha * self.mesh_opacity)


class ComparisonViewer(QWidget):
    def __init__(self, session_path: Path | list[ResultBundle]):
        super().__init__()
        self.bundle_results = list(session_path) if isinstance(session_path, (list, tuple)) else None
        self.session_path = None if self.bundle_results is not None else Path(session_path).resolve()
        self.session, entries = _comparison_results(session_path)
        self.configurations = [config for config, _ in entries]
        self.loaded_results = [loaded for _, loaded in entries]
        for config, loaded in zip(self.configurations, self.loaded_results):
            if config.get("calibration_model_id") == "checkerboard_reference" and "px" not in config["name"].lower():
                focal = loaded.result.video_focal_px
                if focal is None:
                    finite = loaded.result.focal_px[np.isfinite(loaded.result.focal_px)]
                    focal = float(np.median(finite)) if finite.size else None
                if focal is not None:
                    config["name"] = f"Checkerboard Reference {focal:g} px"
        names = [config["name"] for config in self.configurations]
        for index, config in enumerate(self.configurations):
            if names.count(config["name"]) > 1:
                ordinal = sum(name == config["name"] for name in names[:index + 1]) + 1
                self.configurations[index] = dict(config, name=f"{config['name']} ({ordinal})")
        self.configuration_colors = [_color_for_index(i) for i in range(len(entries))]
        self.mode = "overlay"
        self._syncing_camera = False
        self._arrow_mode = "frame"
        self._detached_windows = []
        frame_counts = {loaded.frame_count for loaded in self.loaded_results}
        frame_indices = [loaded.result.frame_indices for loaded in self.loaded_results]
        timestamps = [loaded.result.timestamps for loaded in self.loaded_results]
        if (len(frame_counts) != 1 or any(not np.array_equal(frame_indices[0], frames) for frames in frame_indices[1:])
                or any(not np.allclose(timestamps[0], values, equal_nan=True) for values in timestamps[1:])):
            raise ValueError("comparison results must have matching frame indices, timestamps, and counts")
        self._reference_fallback = False
        self.reference_id = self._default_reference_id()
        self.metric_definitions = metric_registry()
        self.selected_metric_ids = {"wrist_error", "mpjpe", "wrist_depth_error"}
        self._metric_cache = {}
        self.setWindowTitle("CalibFreeHands - Comparison")
        self.setMinimumSize(1280, 720)

        self.source_overlay = _OverlayCanvas(self.loaded_results[0])
        self.source_overlay.setMinimumSize(360, 300)
        self.overlay_view = _OverlayComparisonView(self.loaded_results, self.configuration_colors)
        self.overlay_view.setMinimumSize(340, 260)
        self.overlay_panel = _ViewportPanel(self.overlay_view, list(range(len(entries))))
        self.overlay_view.hud = self.overlay_panel
        self.split_views = [_LinkedResultView(loaded, color, self._sync_camera)
                            for loaded, color in zip(self.loaded_results, self.configuration_colors)]
        for index, view in enumerate(self.split_views):
            view.setMinimumSize(340, 260); view.setAcceptDrops(True)
            view.viewer = self; view.configuration_id = self.configurations[index]["id"]
        self.split_panels = [_ViewportPanel(view, [index]) for index, view in enumerate(self.split_views)]
        for view, panel in zip(self.split_views, self.split_panels): view.hud = panel
        self.split_result_order = list(range(len(entries)))
        self.annotation_mode = "off"
        self.annotation_content = "error"
        self.overlay_annotations = _JointAnnotationLayer(self, self.overlay_view, list(range(len(entries))))
        self.split_annotations = [_JointAnnotationLayer(self, view, [index])
                                  for index, view in enumerate(self.split_views)]
        self.views = QStackedWidget()
        overlay_splitter = QSplitter(Qt.Orientation.Horizontal)
        overlay_splitter.addWidget(self.source_overlay)
        overlay_splitter.addWidget(self.overlay_view)
        overlay_splitter.setStretchFactor(0, 1)
        overlay_splitter.setStretchFactor(1, 2)
        overlay_splitter.setSizes([560, 1120])
        self.views.addWidget(overlay_splitter)

        panel_count = len(entries) + 1
        self.split_columns = min(3, max(1, int(np.ceil(np.sqrt(panel_count * 1.5)))))
        self.split_rows = int(np.ceil(panel_count / self.split_columns))
        self.split_widget = QWidget()
        self.split_grid = QGridLayout(self.split_widget)
        self.split_grid.setContentsMargins(4, 4, 4, 4)
        self.split_grid.setSpacing(6)
        self.split_source_overlay = _OverlayCanvas(self.loaded_results[0])
        source_cell = QWidget()
        source_cell.setMinimumSize(340, 260)
        source_layout = QVBoxLayout(source_cell)
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.addWidget(QLabel("Original media"))
        source_layout.addWidget(self.split_source_overlay, 1)
        self.split_grid.addWidget(source_cell, 0, 0)
        for order, index in enumerate(self.split_result_order, 1):
            self.split_grid.addWidget(self.split_views[index], order // self.split_columns, order % self.split_columns)
        for column in range(self.split_columns):
            self.split_grid.setColumnStretch(column, 1)
        for row in range(self.split_rows):
            self.split_grid.setRowStretch(row, 1)
        self.split_widget.setMinimumSize(self.split_columns * 340, self.split_rows * 260)
        self.split_scroll = QScrollArea()
        self.split_scroll.setWidgetResizable(True)
        self.split_scroll.setWidget(self.split_widget)
        self.views.addWidget(self.split_scroll)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.loaded_results[0].frame_count - 1)
        self.slider.setTracking(False)
        self.slider.valueChanged.connect(self._set_frame)
        self.status = QLabel()
        self.metric_status = QLabel()
        self.metric_status.setToolTip("Camera-space differences from the selected Reference; these are not ground-truth errors.")

        self.mode_selector = QWidget()
        mode_layout = QHBoxLayout(self.mode_selector); mode_layout.setContentsMargins(0, 0, 0, 0); mode_layout.setSpacing(2)
        self.mode_buttons = {}
        for text, mode in (("Overlay", "overlay"), ("Split", "split")):
            button = QToolButton(); button.setText(text); button.setCheckable(True)
            button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            button.clicked.connect(lambda checked=False, value=mode: self.set_mode(value))
            mode_layout.addWidget(button); self.mode_buttons[mode] = button
        self.mode_buttons["overlay"].setChecked(True)

        self.results_menu = QMenu(self)
        self.visibility_checks = []
        self.color_buttons = []
        for index, config in enumerate(self.configurations):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(6, 2, 8, 2)
            check = QCheckBox(config["name"])
            check.setToolTip("Show or hide this result")
            check.setChecked(True)
            check.toggled.connect(lambda value, i=index: self._set_configuration_visible(i, value))
            swatch = QToolButton()
            swatch.setFixedSize(20, 20)
            self._style_swatch(swatch, self.configuration_colors[index])
            swatch.clicked.connect(lambda _=False, i=index: self._choose_color(i))
            row_layout.addWidget(check, 1)
            row_layout.addWidget(swatch)
            action = QWidgetAction(self.results_menu)
            action.setDefaultWidget(row)
            self.results_menu.addAction(action)
            self.visibility_checks.append(check)
            self.color_buttons.append(swatch)
        self.results_menu_button = QToolButton()
        self.results_menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.results_menu_button.setMenu(self.results_menu)
        self._update_results_button()

        self.reference_label = QLabel("Reference" + (" ? fallback" if self._reference_fallback else ""))
        self.reference_selector = QComboBox()
        for config in self.configurations:
            self.reference_selector.addItem(config["name"], config["id"])
        self.reference_selector.setCurrentIndex(next((i for i, config in enumerate(self.configurations)
                                                      if config["id"] == self.reference_id), 0))
        self.reference_selector.setMaximumWidth(220)
        self.reference_selector.setMinimumWidth(120)
        self.reference_selector.setToolTip(self.reference_selector.currentText())
        self.reference_selector.currentIndexChanged.connect(self._reference_changed)

        self.metric_menu = QMenu(self)
        self.metric_checks = {}
        category_menus = {}
        for definition in self.metric_definitions:
            category = category_menus.get(definition.category)
            if category is None:
                category = self.metric_menu.addMenu(definition.category)
                category_menus[definition.category] = category
            check = QCheckBox(definition.name)
            check.setChecked(definition.metric_id in self.selected_metric_ids)
            available = self._metric_has_data(definition.metric_id)
            check.setEnabled(definition.status.value == "READY" and available)
            if not check.isEnabled():
                check.setToolTip("Protocol pending: this metric is not defined for this prototype" if definition.status.value != "READY"
                                 else "Unavailable: compatible result data is missing")
            check.toggled.connect(lambda value, key=definition.metric_id: self._set_metric_selected(key, value))
            action = QWidgetAction(category)
            action.setDefaultWidget(check)
            category.addAction(action)
            self.metric_checks[definition.metric_id] = check
        self.metric_menu_button = QToolButton()
        self.metric_menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.metric_menu_button.setMenu(self.metric_menu)
        self._update_metrics_button()

        self.confidence_menu = QMenu(self)
        self.confidence_opacity_checkbox = self._menu_checkbox(self.confidence_menu, "Use as opacity", checked=True)
        self.confidence_numeric_checkbox = self._menu_checkbox(self.confidence_menu, "Numeric value")
        self.confidence_bar_checkbox = self._menu_checkbox(self.confidence_menu, "Bar")
        self.confidence_opacity_checkbox.toggled.connect(self._confidence_opacity_changed)
        self.confidence_numeric_checkbox.toggled.connect(self._refresh_metric_panels)
        self.confidence_bar_checkbox.toggled.connect(self._refresh_metric_panels)
        self.confidence_menu_button = QToolButton()
        self.confidence_menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.confidence_menu_button.setMenu(self.confidence_menu)
        self.confidence_menu_button.setText("Detection")
        self._update_confidence_button()
        self.confidence_color_mode = QComboBox()
        self.confidence_color_mode.addItem("Result", "configuration")
        self.confidence_color_mode.addItem("Detection", "confidence")
        self.confidence_color_mode.currentIndexChanged.connect(self._confidence_color_changed)

        self.error_vectors_checkbox = QCheckBox("Error vectors")
        self.error_candidate_selector = QComboBox()
        for config in self.configurations:
            if config["id"] != self.reference_id:
                self.error_candidate_selector.addItem(config["name"], config["id"])
        self.error_candidate_selector.setMaximumWidth(190)
        self.error_candidate_selector.setEnabled(self.error_candidate_selector.count() > 0)
        self.error_vectors_checkbox.setEnabled(self.error_candidate_selector.count() > 0)
        self.error_vectors_checkbox.toggled.connect(self._set_error_vectors)
        self.error_candidate_selector.currentIndexChanged.connect(lambda: self._set_error_vectors(self.error_vectors_checkbox.isChecked()))
        self.error_candidate_selector.hide()
        self.error_vectors_menu = QMenu(self)
        self.error_vectors_group = QActionGroup(self); self.error_vectors_group.setExclusive(True)
        self.error_vectors_off = QAction("Off", self); self.error_vectors_off.setCheckable(True); self.error_vectors_off.setChecked(True)
        self.error_vectors_group.addAction(self.error_vectors_off); self.error_vectors_menu.addAction(self.error_vectors_off)
        self.error_vector_actions = {}
        for config in self.configurations:
            if config["id"] == self.reference_id: continue
            action = QAction(config["name"], self); action.setCheckable(True)
            action.triggered.connect(lambda checked, cid=config["id"]: self._select_error_candidate(cid) if checked else None)
            self.error_vectors_group.addAction(action); self.error_vectors_menu.addAction(action)
            self.error_vector_actions[config["id"]] = action
        self.error_vectors_off.triggered.connect(lambda: self._set_error_vectors(False))
        self.error_vectors_button = QToolButton(); self.error_vectors_button.setText("Error vectors: Off")
        self.error_vectors_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.error_vectors_button.setMenu(self.error_vectors_menu)

        self.annotation_menu = QMenu(self)
        self.annotation_group = QActionGroup(self); self.annotation_group.setExclusive(True)
        self.annotation_actions = {}
        for text, mode in (("Off", "off"), ("Hovered joint", "hovered"),
                           ("Key joints", "key_joints"), ("Worst 5", "worst5"), ("All joints", "all")):
            action = QAction(text, self); action.setCheckable(True); action.setChecked(mode == "off")
            action.triggered.connect(lambda checked, value=mode: self.set_annotation_mode(value) if checked else None)
            self.annotation_group.addAction(action); self.annotation_menu.addAction(action)
            self.annotation_actions[mode] = action
        self.annotation_button = QToolButton(); self.annotation_button.setText("Joint annotations")
        self.annotation_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.annotation_button.setMenu(self.annotation_menu)
        self.annotation_button.setText("Joints")
        self.annotation_content_menu = self.annotation_menu.addMenu("Label")
        self.annotation_content_group = QActionGroup(self); self.annotation_content_group.setExclusive(True)
        self.annotation_content_actions = {}
        for label, content in (("Error (mm)", "error"), ("Joint name", "name")):
            action = QAction(label, self); action.setCheckable(True); action.setChecked(content == self.annotation_content)
            action.triggered.connect(lambda checked, value=content: self._set_annotation_content(value) if checked else None)
            self.annotation_content_group.addAction(action); self.annotation_content_menu.addAction(action)
            self.annotation_content_actions[content] = action
        self.hud_menu = QMenu(self)
        for label, anchor in (("Top left", "top-left"), ("Top right", "top-right"),
                              ("Bottom left", "bottom-left"), ("Bottom right", "bottom-right")):
            action = QAction(label, self); action.triggered.connect(lambda _=False, value=anchor: self._set_hud(anchor=value))
            self.hud_menu.addAction(action)
        size_menu = self.hud_menu.addMenu("Size")
        for size in ("small", "medium", "large"):
            action = QAction(size.title(), self); action.triggered.connect(lambda _=False, value=size: self._set_hud(scale=value))
            size_menu.addAction(action)
        opacity_widget = QWidget(); opacity_layout = QHBoxLayout(opacity_widget); opacity_layout.setContentsMargins(8, 2, 8, 2)
        opacity_layout.addWidget(QLabel("HUD opacity")); self.hud_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.hud_opacity_slider.setRange(20, 100); self.hud_opacity_slider.setValue(100); self.hud_opacity_slider.setFixedWidth(100)
        self.hud_opacity_slider.valueChanged.connect(lambda value: self._set_hud(opacity=value / 100))
        opacity_layout.addWidget(self.hud_opacity_slider); opacity_action = QWidgetAction(self.hud_menu)
        opacity_action.setDefaultWidget(opacity_widget); self.hud_menu.addAction(opacity_action)
        self.hud_background_menu = self.hud_menu.addMenu("Background")
        self.hud_background_group = QActionGroup(self); self.hud_background_group.setExclusive(True)
        self.hud_background_actions = {}
        for label, style in (("None", "none"), ("Subtle", "subtle")):
            action = QAction(label, self); action.setCheckable(True); action.setChecked(style == "none")
            action.triggered.connect(lambda checked, value=style: self._set_hud_background(value) if checked else None)
            self.hud_background_group.addAction(action); self.hud_background_menu.addAction(action)
            self.hud_background_actions[style] = action
        self.hud_button = QToolButton(); self.hud_button.setText("HUD")
        self.hud_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup); self.hud_button.setMenu(self.hud_menu)

        self.detach_button = QPushButton("Detach")
        self.detach_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.detach_button.clicked.connect(self.detach_all)
        self.more_menu = QMenu(self)
        self.more_menu.addAction("Detach panels", self.detach_all)
        self.performance_stats_action = self.more_menu.addAction("Performance stats")
        self.performance_stats_action.setCheckable(True)
        self.performance_stats_action.toggled.connect(self._set_performance_stats)
        self.auto_hide_action = self.more_menu.addAction("Auto-hide controls while playing")
        self.auto_hide_action.setCheckable(True); self.auto_hide_action.setChecked(True)
        self.more_button = QToolButton(); self.more_button.setText("More")
        self.more_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup); self.more_button.setMenu(self.more_menu)
        self.export_csv_button = QPushButton("CSV")
        self.export_json_button = QPushButton("JSON")
        for button, callback in ((self.export_csv_button, self._export_csv), (self.export_json_button, self._export_json)):
            button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            button.clicked.connect(callback)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(14, 8, 14, 6)
        toolbar.setSpacing(7)
        toolbar.addWidget(self.mode_selector)
        toolbar.addWidget(self.results_menu_button)
        toolbar.addWidget(self.reference_label); toolbar.addWidget(self.reference_selector)
        toolbar.addWidget(self.metric_menu_button)
        toolbar.addWidget(self.confidence_color_mode)
        toolbar.addWidget(self.confidence_menu_button)
        toolbar.addWidget(self.error_vectors_button)
        self.error_vectors_button.setText("Vectors: Off")
        self.error_vectors_button.setToolTip("Show selected Reference → Candidate error vectors")
        toolbar.addWidget(self.annotation_button); toolbar.addWidget(self.hud_button)
        self.pin_controls_button = QToolButton(); self.pin_controls_button.setText("Pin")
        self.pin_controls_button.setCheckable(True); self.pin_controls_button.setToolTip("Keep controls visible")
        self.pin_controls_button.setStyleSheet("QToolButton:checked { background: palette(highlight); color: palette(highlighted-text); }")
        toolbar.addWidget(self.pin_controls_button); toolbar.addWidget(self.more_button)
        toolbar.addStretch(1)

        self.joint_checkbox = QCheckBox("Joint")
        self.joint_checkbox.setChecked(True)
        self.joint_checkbox.toggled.connect(self._set_joint)
        self.bone_checkbox = QCheckBox("Bone")
        self.bone_checkbox.setChecked(True)
        self.bone_checkbox.toggled.connect(self._set_bone)
        self.mesh_checkbox = QCheckBox("Mesh")
        self.mesh_checkbox.setEnabled(any(loaded.has_mesh_topology for loaded in self.loaded_results))
        self.mesh_checkbox.toggled.connect(self._set_mesh)
        self.mesh_opacity_label = QLabel("Mesh opacity")
        self.mesh_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.mesh_opacity_slider.setRange(10, 100)
        self.mesh_opacity_slider.setValue(100)
        self.mesh_opacity_slider.setFixedWidth(110)
        self.mesh_opacity_slider.setEnabled(self.mesh_checkbox.isEnabled() and self.mesh_checkbox.isChecked())
        self.mesh_checkbox.toggled.connect(self.mesh_opacity_slider.setEnabled)
        self.mesh_opacity_value = QLabel("100%")
        self.mesh_opacity_value.setFixedWidth(self.mesh_opacity_value.fontMetrics().horizontalAdvance("100%") + 4)
        self.mesh_opacity_slider.valueChanged.connect(self._set_opacity)
        self.mesh_opacity_group = QWidget()
        opacity_layout = QHBoxLayout(self.mesh_opacity_group)
        opacity_layout.setContentsMargins(0, 0, 0, 0)
        opacity_layout.setSpacing(4)
        opacity_layout.addWidget(self.mesh_opacity_label)
        opacity_layout.addWidget(self.mesh_opacity_slider)
        opacity_layout.addWidget(self.mesh_opacity_value)
        geometry = QHBoxLayout()
        for control in (self.joint_checkbox, self.bone_checkbox, self.mesh_checkbox, self.mesh_opacity_group):
            geometry.addWidget(control)
        for text, attr in (("Left", "show_left"), ("Right", "show_right"), ("Grid", "show_grid"), ("Axis", "show_axis")):
            check = QCheckBox(text)
            check.setChecked(True)
            check.toggled.connect(lambda value, name=attr: self._set_geometry(name, value))
            geometry.addWidget(check)
            if not hasattr(self, "geometry_checks"):
                self.geometry_checks = {}
            self.geometry_checks[attr] = check
        geometry.addStretch(1)

        self.view_buttons = []
        view_toolbar = QHBoxLayout()
        view_toolbar.addWidget(QLabel("Camera"))
        for text, method in (("Front", "front_view"), ("Side", "side_view"), ("Top", "top_view"),
                             ("Fit", "fit_hands"), ("Reset", "reset_camera")):
            button = QPushButton(text)
            button.setFixedWidth(64)
            button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            button.clicked.connect(lambda _=False, name=method: self._apply_camera_preset(name))
            view_toolbar.addWidget(button)
            self.view_buttons.append(button)
        view_toolbar.addStretch(1)

        geometry_menu = self.more_menu.addMenu("Geometry")
        for label, check in (("Joints", self.joint_checkbox), ("Bones", self.bone_checkbox),
                             ("Mesh", self.mesh_checkbox),
                             *[(name, self.geometry_checks[attr]) for name, attr in
                               (("Left hand", "show_left"), ("Right hand", "show_right"),
                                ("Grid", "show_grid"), ("Axis", "show_axis"))]):
            action = geometry_menu.addAction(label)
            action.setCheckable(True); action.setChecked(check.isChecked()); action.setEnabled(check.isEnabled())
            action.toggled.connect(check.setChecked); check.toggled.connect(action.setChecked)
        mesh_opacity_action = QWidgetAction(geometry_menu)
        mesh_opacity_action.setDefaultWidget(self.mesh_opacity_group)
        geometry_menu.addAction(mesh_opacity_action)
        for widget in (self.joint_checkbox, self.bone_checkbox, self.mesh_checkbox,
                       *self.geometry_checks.values(), *self.view_buttons):
            widget.setParent(self)
            widget.hide()
        camera_menu = self.more_menu.addMenu("Camera")
        for text, method in (("Front", "front_view"), ("Side", "side_view"), ("Top", "top_view"),
                             ("Fit", "fit_hands"), ("Reset", "reset_camera")):
            camera_menu.addAction(text, lambda _=False, name=method: self._apply_camera_preset(name))
        self.more_menu.addSeparator()
        self.more_menu.addAction("Export CSV…", self._export_csv)
        self.more_menu.addAction("Export JSON…", self._export_json)

        playback_fps = 30.0
        metadata = self.loaded_results[0].metadata or {}
        provenance = metadata.get("run_provenance", {})
        media_info = provenance.get("media_metadata", {}) if isinstance(provenance, dict) else {}
        inference_info = provenance.get("inference_settings", {}) if isinstance(provenance, dict) else {}
        for candidate_fps in (media_info.get("frame_rate"), inference_info.get("fps"), metadata.get("fps"), 30.0):
            try:
                if np.isfinite(float(candidate_fps)) and float(candidate_fps) > 0:
                    playback_fps = float(candidate_fps)
                    break
            except (TypeError, ValueError):
                continue
        self.playback_clock = PlaybackClock(self.loaded_results[0].result.timestamps, fps=playback_fps, monotonic=monotonic)
        self._clock_driven_update = False
        self._last_rendered_frame = None
        self._requested_frame = None
        self.requested_frame = None
        self.presented_frame = None
        self._presented_source_image = QImage()
        self._stats_frames = 0
        self._stats_skipped = 0
        self._stats_started = monotonic()
        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._advance)
        self.playback_buttons = []
        playback = QHBoxLayout()
        playback.addWidget(QLabel("Playback"))
        for text, callback, width in (("Previous", self._previous, 82), ("Play / Pause", self._play, 100), ("Next", self._next_frame, 70)):
            button = QPushButton(text)
            button.setFixedWidth(width)
            button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            button.clicked.connect(callback)
            playback.addWidget(button)
            self.playback_buttons.append(button)
        playback.addWidget(QLabel("Speed"))
        self.speed_selector = QComboBox()
        for speed in PlaybackClock.SPEEDS:
            self.speed_selector.addItem("1.0×" if speed == 1.0 else f"{speed:g}×", speed)
        self.speed_selector.setCurrentIndex(PlaybackClock.SPEEDS.index(1.0))
        self.speed_selector.setMaximumWidth(78)
        self.speed_selector.currentIndexChanged.connect(self._set_playback_speed)
        playback.addStretch(1)
        playback.addWidget(self.speed_selector)
        self.performance_stats_label = QLabel()
        self.performance_stats_label.hide()
        self.performance_stats_timer = QTimer(self)
        self.performance_stats_timer.setInterval(1000)
        self.performance_stats_timer.timeout.connect(self._update_performance_stats)

        exports = QHBoxLayout()
        exports.addWidget(self.metric_status, 1)
        exports.addWidget(self.export_csv_button)
        exports.addWidget(self.export_json_button)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.views, 0, 0)
        self.chrome = QWidget(); chrome_layout = QVBoxLayout(self.chrome); chrome_layout.setContentsMargins(0, 8, 0, 12); chrome_layout.setSpacing(6)
        chrome_layout.addLayout(toolbar); chrome_layout.addWidget(self.slider)
        playback.addWidget(self.status)
        playback.addWidget(self.metric_status, 1)
        chrome_layout.addLayout(playback); chrome_layout.addWidget(self.performance_stats_label)
        layout.addWidget(self.chrome, 0, 0, Qt.AlignmentFlag.AlignBottom)
        self.auto_hide_timer = QTimer(self); self.auto_hide_timer.setSingleShot(True); self.auto_hide_timer.setInterval(2500)
        self.auto_hide_timer.timeout.connect(self._auto_hide_controls)
        self.pin_controls_button.toggled.connect(lambda pinned: self.auto_hide_timer.stop() if pinned else self._show_controls())
        self.pin_controls_button.toggled.connect(self._set_pin_label)
        self._panel_reorder_active = False
        self._hover_menu_target = None
        self._hover_button_entered = False
        self._opening_menu_by_hover = False
        self._hover_menu_latched = False
        self._last_pointer_position = None
        self._last_pointer_widget = None
        self.hover_menu_timer = QTimer(self); self.hover_menu_timer.setSingleShot(True); self.hover_menu_timer.setInterval(0)
        self.hover_menu_timer.timeout.connect(self._open_hover_menu)
        self.hover_close_timer = QTimer(self); self.hover_close_timer.setSingleShot(True); self.hover_close_timer.setInterval(0)
        self.hover_close_timer.timeout.connect(self._close_hover_menu)
        self.setMouseTracking(True)
        for widget in self.findChildren(QWidget):
            widget.setMouseTracking(True)
        for menu in self.findChildren(QMenu):
            menu.aboutToShow.connect(lambda m=menu: self._menu_about_to_show(m))
            menu.aboutToHide.connect(lambda m=menu: self._menu_about_to_hide(m))
        source_provider = self._source_provider()
        if source_provider is not None:
            self._watch_source_provider(source_provider)
        QApplication.instance().installEventFilter(self)
        self._compute_metrics()
        self._refresh_metric_panels()
        self._refresh_annotations()
        self._set_frame(0)
        self._sync_camera(self.split_views[0])

    def _refresh_annotations(self):
        if self.annotation_mode != "off":
            layers = [self.overlay_annotations] if self.mode == "overlay" else self.split_annotations
            for layer in layers:
                layer.refresh()
        else:
            self.overlay_annotations.hide()
            for layer in self.split_annotations:
                layer.hide()
        panels = [self.overlay_panel] if self.mode == "overlay" else self.split_panels
        for panel in panels:
            panel._position()
            panel.raise_()

    def _set_hud(self, anchor=None, scale=None, opacity=None):
        for hud in (self.overlay_panel, *self.split_panels): hud.set_presentation(anchor, scale, opacity)

    def _set_hud_background(self, style):
        for hud in (self.overlay_panel, *self.split_panels): hud.set_background(style)

    def _select_error_candidate(self, candidate_id):
        index = self.error_candidate_selector.findData(candidate_id)
        if index >= 0: self.error_candidate_selector.setCurrentIndex(index)
        self.error_vectors_checkbox.setChecked(True)
        name = next(config["name"] for config in self.configurations if config["id"] == candidate_id)
        self.error_vectors_button.setText("Vectors: Candidate")
        self.error_vectors_button.setToolTip(f"Show selected Reference → {name} error vectors")

    def _show_controls(self):
        self.chrome.show()
        if self.timer.isActive() and self.auto_hide_action.isChecked(): self.auto_hide_timer.start()

    def _set_pin_label(self, pinned):
        self.pin_controls_button.setText("Pinned" if pinned else "Pin")
        self.pin_controls_button.setToolTip("Controls stay visible while pinned" if pinned else "Keep controls visible")

    def _menu_about_to_show(self, menu):
        if not self._opening_menu_by_hover and menu is getattr(self._hover_menu_target, "menu", lambda: None)():
            self._hover_menu_latched = True

    def _menu_about_to_hide(self, menu):
        button = self._hover_menu_target
        if isinstance(button, QToolButton) and button.menu() is menu:
            self._hover_menu_latched = False

    def _auto_hide_controls(self):
        if (not self.timer.isActive() or not self.auto_hide_action.isChecked()
                or self.pin_controls_button.isChecked() or self._panel_reorder_active): return
        popup_open = any(menu.isVisible() for menu in self.findChildren(QMenu))
        combo_open = any(combo.view().isVisible() for combo in self.findChildren(QComboBox))
        modal = QApplication.activeModalWidget()
        dialog_open = modal is not None and (modal is self or self.isAncestorOf(modal))
        dragging = bool(QApplication.mouseButtons() & Qt.MouseButton.LeftButton)
        if popup_open or combo_open or dialog_open or self.slider.isSliderDown() or dragging: self.auto_hide_timer.start()
        else: self.chrome.hide()

    def reorder_result(self, source_id, target_id):
        source = next(i for i, config in enumerate(self.configurations) if config["id"] == source_id)
        target = next(i for i, config in enumerate(self.configurations) if config["id"] == target_id)
        self.split_result_order.remove(source)
        self.split_result_order.insert(self.split_result_order.index(target), source)
        for position, index in enumerate(self.split_result_order, 1):
            self.split_grid.addWidget(self.split_views[index], position // self.split_columns, position % self.split_columns)
        for panel in (self.overlay_panel, *self.split_panels):
            panel.configuration_indices = sorted(panel.configuration_indices,
                key=lambda i: self.split_result_order.index(i) if i in self.split_result_order else i)
            panel.refresh(self)
        self._rebuild_results_menu()
        self._refresh_annotations()

    def _rebuild_results_menu(self):
        self.results_menu.clear(); self.visibility_checks = [None] * len(self.configurations)
        self.color_buttons = [None] * len(self.configurations)
        for index in self.split_result_order:
            config = self.configurations[index]
            row = QWidget(); row_layout = QHBoxLayout(row); row_layout.setContentsMargins(6, 2, 8, 2)
            check = QCheckBox(config["name"]); check.setChecked(self.overlay_view.configuration_visible[index])
            check.toggled.connect(lambda value, i=index: self._set_configuration_visible(i, value))
            swatch = QToolButton(); swatch.setFixedSize(20, 20); self._style_swatch(swatch, self.configuration_colors[index])
            swatch.clicked.connect(lambda _=False, i=index: self._choose_color(i))
            row_layout.addWidget(check, 1); row_layout.addWidget(swatch)
            action = QWidgetAction(self.results_menu); action.setDefaultWidget(row); self.results_menu.addAction(action)
            self.visibility_checks[index] = check; self.color_buttons[index] = swatch
        self._update_results_button()

    def set_annotation_mode(self, mode):
        if mode not in ("off", "hovered", "key_joints", "worst5", "all"): raise ValueError(mode)
        self.annotation_mode = mode
        self._refresh_annotations()

    def _set_annotation_content(self, content):
        if content not in ("error", "name"):
            raise ValueError(content)
        self.annotation_content = content
        self._refresh_annotations()

    @staticmethod
    def _menu_checkbox(menu, text, checked=False):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(8, 2, 12, 2)
        checkbox = QCheckBox(text)
        checkbox.setChecked(checked)
        layout.addWidget(checkbox)
        action = QWidgetAction(menu)
        action.setDefaultWidget(row)
        menu.addAction(action)
        return checkbox

    def _update_results_button(self):
        visible = sum(check.isChecked() for check in self.visibility_checks)
        self.results_menu_button.setText(f"Results ({visible}/{len(self.visibility_checks)})")

    def _update_metrics_button(self):
        self.metric_menu_button.setText("Metrics")

    def _update_confidence_button(self):
        active = sum(check.isChecked() for check in (self.confidence_opacity_checkbox,
                                                       self.confidence_numeric_checkbox,
                                                       self.confidence_bar_checkbox))
        self.confidence_menu_button.setText("Detection")

    def _default_reference_id(self):
        checkerboards = [c for c in self.configurations
                         if c.get("calibration_model_id") == "checkerboard_reference"]
        if len(checkerboards) == 1:
            return checkerboards[0]["id"]
        explicit = self.session.get("reference_configuration_id")
        if any(c["id"] == explicit for c in self.configurations):
            return explicit
        self._reference_fallback = True
        return self.configurations[0]["id"]

    def _metric_has_data(self, metric_id):
        if metric_id not in ("translation_error", "translation_z_error", "translation_xy_error", "mpvpe", "pa_mpvpe"):
            return True
        reference = self.loaded_results[next(i for i, c in enumerate(self.configurations)
                                              if c["id"] == self.reference_id)].result
        for config, loaded in zip(self.configurations, self.loaded_results):
            if config["id"] == self.reference_id:
                continue
            candidate = loaded.result
            if metric_id.startswith("translation_") and reference.camera_translation is not None and candidate.camera_translation is not None:
                if reference.camera_translation.shape == candidate.camera_translation.shape:
                    return True
            if metric_id in ("mpvpe", "pa_mpvpe") and reference.mano_vertices is not None and candidate.mano_vertices is not None:
                if reference.mano_faces is not None and np.array_equal(reference.mano_faces, candidate.mano_faces):
                    return True
        return False

    @staticmethod
    def _style_swatch(button, color):
        rgb = tuple(round(channel * 255) for channel in color)
        button.setStyleSheet(f"QToolButton {{ background-color: rgb{rgb}; border: 1px solid palette(mid); }}")
        button.setToolTip("Choose result color")

    def _choose_color(self, index):
        from PySide6.QtGui import QColor
        current = QColor(*(round(channel * 255) for channel in self.configuration_colors[index]))
        chosen = QColorDialog.getColor(current, self, "Result color")
        if not chosen.isValid():
            return
        self._apply_configuration_color(index, (chosen.redF(), chosen.greenF(), chosen.blueF()))

    def _apply_configuration_color(self, index, color):
        self.configuration_colors[index] = color
        self._style_swatch(self.color_buttons[index], color)
        self.overlay_view.configuration_colors[index] = color
        self.split_views[index].configuration_color = color
        self.split_views[index].hand_colors = (_hand_variant(color, 0), _hand_variant(color, 1))
        self.overlay_view.set_frame(self.slider.value())
        self.split_views[index].set_frame(self.slider.value())
        self._refresh_metric_panels()

    def _reference_changed(self, _index):
        self.reference_id = self.reference_selector.currentData()
        self.reference_selector.setToolTip(self.reference_selector.currentText())
        self.reference_label.setText("Reference")
        for definition in self.metric_definitions:
            if definition.metric_id in self.metric_checks:
                self.metric_checks[definition.metric_id].setEnabled(
                    definition.status.value == "READY" and self._metric_has_data(definition.metric_id))
        self.error_candidate_selector.clear()
        for config in self.configurations:
            if config["id"] != self.reference_id:
                self.error_candidate_selector.addItem(config["name"], config["id"])
        self.error_vectors_checkbox.setEnabled(bool(self.error_candidate_selector.count()))
        self.error_candidate_selector.setEnabled(bool(self.error_candidate_selector.count()))
        self.error_vectors_checkbox.setChecked(False)
        if not self.error_candidate_selector.count():
            self.error_vectors_checkbox.setChecked(False)
        self._compute_metrics()
        self._refresh_metric_panels()
        self.error_vectors_button.setText("Error vectors: Off")
        self._rebuild_error_vector_menu()

    def _rebuild_error_vector_menu(self):
        self.error_vectors_menu.clear(); self.error_vector_actions.clear()
        self.error_vectors_group = QActionGroup(self); self.error_vectors_group.setExclusive(True)
        self.error_vectors_off = QAction("Off", self); self.error_vectors_off.setCheckable(True)
        self.error_vectors_off.setChecked(True); self.error_vectors_off.triggered.connect(lambda: self._set_error_vectors(False))
        self.error_vectors_group.addAction(self.error_vectors_off); self.error_vectors_menu.addAction(self.error_vectors_off)
        for config in self.configurations:
            if config["id"] == self.reference_id: continue
            action = QAction(config["name"], self); action.setCheckable(True)
            action.triggered.connect(lambda checked, cid=config["id"]: self._select_error_candidate(cid) if checked else None)
            self.error_vectors_group.addAction(action); self.error_vectors_menu.addAction(action)
            self.error_vector_actions[config["id"]] = action

    def _compute_metrics(self):
        reference_index = next(i for i, config in enumerate(self.configurations) if config["id"] == self.reference_id)
        reference = self.loaded_results[reference_index].result
        cache = {}
        for index, loaded in enumerate(self.loaded_results):
            if index != reference_index:
                cache[self.configurations[index]["id"]] = {
                    (row.frame_index, row.hand): row for row in compare_results(reference, loaded.result)
                }
        self._metric_cache = cache

    def _set_metric_selected(self, metric_id, selected):
        if selected:
            self.selected_metric_ids.add(metric_id)
        else:
            self.selected_metric_ids.discard(metric_id)
        self._refresh_metric_panels()

    def _refresh_metric_panels(self, *_):
        self._update_metrics_button()
        self._update_confidence_button()
        self.overlay_panel.refresh(self)
        for panel in self.split_panels:
            panel.refresh(self)
        self._refresh_annotations()

    def _confidence_color_changed(self, _index):
        mode = self.confidence_color_mode.currentData()
        for view in [self.overlay_view, *self.split_views]:
            view.confidence_color_mode = mode
        self._render_active_frame()
        self._refresh_metric_panels()

    def _confidence_opacity_changed(self, enabled):
        self._update_confidence_button()
        for view in [self.overlay_view, *self.split_views]:
            view.confidence_opacity_enabled = enabled
        for hand, mesh in enumerate(self.overlay_view.meshes):
            alpha = _confidence_alpha(self.overlay_view.loaded.result, self.overlay_view.frame, hand) if enabled else 1.0
            _set_mesh_color(mesh, _hand_variant(self.configuration_colors[0], hand), alpha * self.overlay_view.mesh_opacity)
        for config_index, hands in enumerate(self.overlay_view.extra_items, 1):
            result = self.loaded_results[config_index].result
            frame = min(self.overlay_view.frame, self.loaded_results[config_index].frame_count - 1)
            for hand, (_, _, mesh) in enumerate(hands):
                alpha = _confidence_alpha(result, frame, hand) if enabled else 1.0
                _set_mesh_color(mesh, _hand_variant(self.configuration_colors[config_index], hand), alpha * self.overlay_view.mesh_opacity)
        for view in self.split_views:
            for hand, mesh in enumerate(view.meshes):
                alpha = _confidence_alpha(view.loaded.result, view.frame, hand) if enabled else 1.0
                _set_mesh_color(mesh, view.hand_colors[hand], alpha * view.mesh_opacity)

    def _set_error_vectors(self, enabled):
        self.error_candidate_selector.setVisible(enabled and self.error_candidate_selector.count() > 0)
        self.overlay_view.error_vectors_enabled = enabled
        self.overlay_view.reference_index = next((i for i, config in enumerate(self.configurations)
                                                  if config["id"] == self.reference_id), 0)
        self.overlay_view.error_candidate = next((i for i, c in enumerate(self.configurations)
                                                   if c["id"] == self.error_candidate_selector.currentData()), -1)
        if enabled and self.mode == "overlay":
            self.overlay_view.set_frame(self.slider.value())
        elif not enabled:
            for item in self.overlay_view.error_items:
                item.setVisible(False)
        if not enabled: self.error_vectors_button.setText("Error vectors: Off")

    def _show_joint_table(self, candidate_id):
        candidate_name = next(config["name"] for config in self.configurations if config["id"] == candidate_id)
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Per-Joint Error - {candidate_name}")
        layout = QVBoxLayout(dialog)
        table = QTableWidget(21, 3, dialog)
        table.setHorizontalHeaderLabels(["Joint", "Left (mm)", "Right (mm)"])
        frame = int(self.loaded_results[0].result.frame_indices[self.slider.value()])
        for index, name in enumerate(JOINT_NAMES):
            table.setItem(index, 0, QTableWidgetItem(name))
            for column, hand in enumerate(("left", "right"), 1):
                pair = self._metric_cache.get(candidate_id, {}).get((frame, hand))
                error = pair.joint_errors_mm[index] if pair is not None else np.nan
                table.setItem(index, column, QTableWidgetItem(f"{error:.2f}" if np.isfinite(error) else "N/A"))
        table.resizeColumnsToContents()
        layout.addWidget(table)
        dialog.resize(420, 560)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.open()
        return dialog

    def detach_all(self):
        window = QMainWindow(self, Qt.WindowType.Window)
        window.setWindowTitle("CalibFreeHands - Detached Comparison")
        content = QWidget(window)
        detached_grid = QGridLayout(content)
        detached_source = _OverlayCanvas(self.loaded_results[0])
        detached_grid.addWidget(detached_source, 0, 0)
        window.setCentralWidget(content)
        detached_views = [_LinkedResultView(loaded, color, lambda *_: None)
                          for loaded, color in zip(self.loaded_results, self.configuration_colors)]
        for index, (config, view) in enumerate(zip(self.configurations, detached_views), 1):
            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.addWidget(QLabel(config["name"]))
            cell_layout.addWidget(view, 1)
            detached_grid.addWidget(cell, index // self.split_columns, index % self.split_columns)
        self._detached_windows.append((window, detached_source, detached_views))
        for view in detached_views:
            view.camera_changed = lambda source, group=detached_views: self._sync_detached(group, source)
        window.show()

    @staticmethod
    def _sync_detached(views, source):
        params = source.cameraParams()
        for view in views:
            if view is not source:
                view.setCameraParams(**params)

    def set_mode(self, mode):
        if mode not in ("overlay", "split"):
            raise ValueError(f"unknown comparison view mode: {mode}")
        self.mode = mode
        self.views.setCurrentIndex(0 if mode == "overlay" else 1)
        for name, button in self.mode_buttons.items(): button.setChecked(name == mode)
        self._render_active_frame()
        self._refresh_metric_panels()
        self._refresh_annotations()

    def _set_configuration_visible(self, index, visible):
        self.overlay_view.set_configuration_visible(index, visible)
        if visible and self.mode == "overlay":
            self._render_active_frame()
        self.split_views[index].setVisible(visible)
        if visible and self.mode == "split":
            self.split_views[index].set_frame(self.slider.value())
        self._update_results_button()
        self._refresh_metric_panels()

    def _set_geometry(self, attr, value):
        source = self.source_overlay if self.mode == "overlay" else self.split_source_overlay
        if hasattr(source, attr):
            setattr(source, attr, value)
        source.update()
        for view in [self.overlay_view, *self.split_views]:
            setattr(view, attr, value)
        self._render_active_frame()

    def _set_mesh(self, value):
        self.mesh_opacity_slider.setEnabled(bool(value) and self.mesh_checkbox.isEnabled())
        if value:
            for check in (self.joint_checkbox, self.bone_checkbox):
                blocker = QSignalBlocker(check); check.setChecked(False); del blocker
                attr = "show_joints" if check is self.joint_checkbox else "show_bones"
                for view in [self.overlay_view, *self.split_views]:
                    setattr(view, attr, False)
        for view in [self.overlay_view, *self.split_views]:
            view.show_mesh = value
        self._render_active_frame()

    def _set_joint(self, value):
        if value and self.mesh_checkbox.isChecked():
            blocker = QSignalBlocker(self.mesh_checkbox); self.mesh_checkbox.setChecked(False); del blocker
            self.mesh_opacity_slider.setEnabled(False)
            for view in [self.overlay_view, *self.split_views]:
                view.show_mesh = False
        self._set_geometry("show_joints", value)

    def _set_bone(self, value):
        if value and self.mesh_checkbox.isChecked():
            blocker = QSignalBlocker(self.mesh_checkbox); self.mesh_checkbox.setChecked(False); del blocker
            self.mesh_opacity_slider.setEnabled(False)
            for view in [self.overlay_view, *self.split_views]:
                view.show_mesh = False
        self._set_geometry("show_bones", value)

    def _set_opacity(self, value):
        self.mesh_opacity_value.setText(f"{value}%")
        for view in [self.overlay_view, *self.split_views]:
            view.mesh_opacity = value / 100
        self._render_active_frame()

    def _set_frame(self, frame):
        frame = int(frame)
        if frame == self._last_rendered_frame and frame == self._requested_frame:
            return
        self._requested_frame = frame
        self.requested_frame = frame
        if self.timer.isActive() and not self._clock_driven_update:
            self.playback_clock.play(frame)
        provider = self._source_provider()
        if provider is not None:
            source_index = self._source_index(frame, provider)
            rgb = provider.cached_frame(source_index)
            if rgb is None:
                provider.frame(source_index)
                if self._last_rendered_frame is not None and self.slider.value() != self._last_rendered_frame:
                    blocker = QSignalBlocker(self.slider)
                    self.slider.setValue(self._last_rendered_frame)
                    del blocker
                return
        self._present_frame(frame, source_rgb=rgb if provider is not None else None)

    def _source_provider(self):
        provider = self.loaded_results[0].media_frames
        return provider if isinstance(provider, MediaFrameProvider) else None

    def _source_index(self, frame, provider):
        if provider._is_image:
            return 0
        return int(self.loaded_results[0].result.frame_indices[frame])

    def _watch_source_provider(self, provider):
        provider.signals.frame_ready.connect(self._source_frame_ready)

    def _source_frame_ready(self, source_index):
        frame = self._requested_frame
        provider = self._source_provider()
        if frame is None or provider is None or self._source_index(frame, provider) != source_index:
            return
        rgb = provider.cached_frame(source_index)
        if rgb is None:
            return
        self._present_frame(frame, source_rgb=rgb)

    def _present_frame(self, frame, *, force=False, source_rgb=None):
        if frame == self._last_rendered_frame and not force:
            return
        previous = self._last_rendered_frame
        if self.slider.value() != frame:
            blocker = QSignalBlocker(self.slider)
            self.slider.setValue(frame)
            del blocker
        if source_rgb is not None:
            self._presented_source_image = QImage(
                source_rgb.data, source_rgb.shape[1], source_rgb.shape[0], source_rgb.strides[0],
                QImage.Format.Format_RGB888).copy()
        source_image = self._presented_source_image if not self._presented_source_image.isNull() else None
        if self.mode == "overlay":
            self.source_overlay.set_frame(frame, source_image)
            self.overlay_view.set_frame(frame)
        else:
            self.split_source_overlay.set_frame(frame, source_image)
            for index, view in enumerate(self.split_views):
                if self.visibility_checks[index].isChecked():
                    view.set_frame(frame)
        for _window, detached_source, views in self._detached_windows:
            if not _window.isVisible():
                continue
            detached_source.set_frame(frame, source_image)
            for view in views:
                if not view.isHidden():
                    view.set_frame(frame)
        panels = [self.overlay_panel] if self.mode == "overlay" else self.split_panels
        for panel in panels:
            if (self.mode == "split" and not self.visibility_checks[panel.configuration_indices[0]].isChecked()):
                continue
            panel.update_values(self)
        if self.annotation_mode != "off":
            layers = [self.overlay_annotations] if self.mode == "overlay" else self.split_annotations
            for layer in layers:
                layer.update()
        timestamp = self.loaded_results[0].result.timestamps[frame]
        suffix = f" | {float(timestamp):.3f} s" if np.isfinite(timestamp) else ""
        source = " | Source media missing" if self.loaded_results[0].source_status == "missing" else (" | Source media changed" if self.loaded_results[0].source_status == "changed" else "")
        self.status.setText(f"Frame {frame + 1} / {self.loaded_results[0].frame_count}{suffix}{source}")
        self._last_rendered_frame = frame
        self.presented_frame = frame
        if previous is not None and self.timer.isActive() and self._clock_driven_update:
            self._stats_frames += 1

    def _render_active_frame(self):
        if self.presented_frame is not None:
            self._present_frame(self.presented_frame, force=True)
        else:
            self._set_frame(self.slider.value())

    def _set_playback_speed(self, index):
        self.playback_clock.set_speed(self.speed_selector.itemData(index))

    def _set_performance_stats(self, enabled):
        self.performance_stats_label.setVisible(enabled)
        self._stats_started = monotonic()
        self._stats_frames = 0
        self._stats_skipped = 0
        if enabled:
            self.performance_stats_timer.start()
            self._update_performance_stats()
        else:
            self.performance_stats_timer.stop()
            self.performance_stats_label.clear()

    def _update_performance_stats(self):
        elapsed = max(1e-6, monotonic() - self._stats_started)
        source_fps = 1.0 / self.playback_clock.frame_duration
        self.performance_stats_label.setText(
            f"Source {source_fps:.1f} fps · Frame updates {self._stats_frames / elapsed:.1f} fps · "
            f"{self.playback_clock.speed:.2f}× · Skipped {self._stats_skipped}")
        self._stats_started = monotonic()
        self._stats_frames = 0

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
        self._refresh_annotations()

    def _apply_camera_preset(self, method):
        views = [self.overlay_view] if self.mode == "overlay" else self.split_views
        if not views: return
        getattr(views[0], method)()
        if self.mode == "split": self._sync_camera(views[0])
        self._refresh_annotations()

    def _previous(self): self.slider.setValue(max(0, self.slider.value() - 1))
    def _next_frame(self): self.slider.setValue(min(self.slider.maximum(), self.slider.value() + 1))
    def _advance(self):
        frame_count = max(1, self.slider.maximum() + 1)
        target = self.playback_clock.frame_at()
        previous = self._requested_frame if self._requested_frame is not None else self.slider.value()
        if target == previous:
            return
        distance = (target - previous) % frame_count
        self._stats_skipped += max(0, distance - 1)
        self._clock_driven_update = True
        try:
            self.slider.setValue(target)
        finally:
            self._clock_driven_update = False
    def _play(self):
        if self.timer.isActive():
            pause_frame = self._last_rendered_frame if self._last_rendered_frame is not None else self.slider.value()
            self.timer.stop(); self.playback_clock.pause(pause_frame); self._requested_frame = pause_frame
            self.requested_frame = pause_frame
            self.auto_hide_timer.stop(); self.chrome.show()
        else:
            self.playback_clock.play(self.slider.value()); self.timer.start(); self._show_controls()

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
            (export_metrics_csv if kind == "csv" else export_metrics_json)(
                self.bundle_results or self.session_path, Path(path), self.reference_id)
            self.metric_status.setText(f"Saved {Path(path).name}")
        except (OSError, ValueError) as exc:
            self.metric_status.setText(f"Export failed: {exc}")

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Type.MouseMove, QEvent.Type.MouseButtonPress):
            self._last_pointer_widget = watched if isinstance(watched, QWidget) else None
            try:
                self._last_pointer_position = event.globalPosition().toPoint()
            except AttributeError:
                pass
        active_button = self._hover_menu_target
        if (event.type() == QEvent.Type.Hide and isinstance(active_button, QComboBox)
                and self._is_active_hover_menu(watched)):
            self._hover_menu_latched = False
        active_combo = isinstance(active_button, QComboBox)
        active_menu = active_button.menu() if isinstance(active_button, QToolButton) else None
        active_popup_visible = (active_menu.isVisible() if active_menu is not None else
                                active_button.view().isVisible() if active_combo else False)
        if (event.type() == QEvent.Type.MouseButtonPress and self._hover_menu_latched
                and (active_menu is not None or active_combo)):
            inside_menu = self._is_active_hover_menu(watched)
            inside_button = isinstance(watched, QWidget) and (
                watched is active_button or active_button.isAncestorOf(watched))
            if not inside_menu and not inside_button:
                self._hover_menu_latched = False
                if isinstance(active_button, QToolButton) and active_button.menu() is not None:
                    active_button.menu().hide()
        if (event.type() == QEvent.Type.MouseMove and not self._hover_menu_latched
                and active_popup_visible):
            self.hover_close_timer.start()
        is_child_widget = isinstance(watched, QWidget) and (watched is self or self.isAncestorOf(watched))
        if is_child_widget:
            if ((isinstance(watched, QToolButton) and watched.menu() is not None)
                    or isinstance(watched, QComboBox)):
                if event.type() == QEvent.Type.Enter:
                    if not self._hover_menu_latched or watched is self._hover_menu_target:
                        self.hover_close_timer.stop()
                        self._hover_menu_target = watched
                        self._hover_button_entered = True
                        self.hover_menu_timer.start()
                elif event.type() == QEvent.Type.Leave:
                    self._hover_button_entered = False
                    self.hover_close_timer.start()
            elif self._is_active_hover_menu(watched):
                if event.type() == QEvent.Type.Enter:
                    self.hover_close_timer.stop()
                elif event.type() == QEvent.Type.Leave:
                    self.hover_close_timer.start()
                elif event.type() == QEvent.Type.MouseMove:
                    self.hover_close_timer.start()
            if event.type() in (QEvent.Type.MouseMove, QEvent.Type.MouseButtonPress, QEvent.Type.Wheel, QEvent.Type.KeyPress):
                if not isinstance(watched, (QMenu, QComboBox, QSlider, QDialog)):
                    self._show_controls()
            if event.type() == QEvent.Type.MouseButtonPress and (
                    (isinstance(watched, QToolButton) and watched.menu() is not None)
                    or isinstance(watched, QComboBox)):
                popup_visible = (watched.menu().isVisible() if isinstance(watched, QToolButton)
                                 else watched.view().isVisible())
                if popup_visible:
                    if self._hover_menu_latched:
                        self._hover_menu_latched = False
                        if isinstance(watched, QComboBox):
                            QTimer.singleShot(0, lambda target=watched: self._dismiss_combo_popup(target))
                            return False
                        self._hide_hover_popup(watched)
                    else:
                        self._hover_menu_latched = True
                    event.accept()
                    return True
                self._hover_menu_target = watched
                self._hover_menu_latched = True
                if isinstance(watched, QToolButton):
                    self._show_hover_popup(watched)
                else:
                    return False
                event.accept()
                return True
            if event.type() == QEvent.Type.MouseMove and self.annotation_mode == "hovered":
                for layer in (self.overlay_annotations, *self.split_annotations):
                    if layer.view is watched:
                        layer.hover_position = event.position().toPoint()
                        layer.update()
        elif event.type() in (QEvent.Type.MouseMove, QEvent.Type.MouseButtonPress, QEvent.Type.Wheel, QEvent.Type.KeyPress):
            if not isinstance(watched, (QMenu, QComboBox, QSlider, QDialog)):
                self._show_controls()
        if event.type() in (QEvent.Type.MouseMove, QEvent.Type.Wheel, QEvent.Type.Resize):
            if watched is self.overlay_view or watched in self.split_views:
                self._refresh_annotations()
        if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_F11:
            window = self.window()
            window.showNormal() if window.isFullScreen() else window.showFullScreen()
            event.accept(); return True
        if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape and self.window().isFullScreen():
            self.window().showNormal(); event.accept(); return True
        if event.type() == QEvent.Type.MouseButtonPress:
            if isinstance(watched, QWidget):
                self._arrow_mode = "orbit" if any(watched is view or view.isAncestorOf(watched) for view in self.split_views + [self.overlay_view]) else "frame"
        elif event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Escape and self._hover_menu_latched and self._hover_menu_target is not None:
                self._hover_menu_latched = False
                if isinstance(self._hover_menu_target, QToolButton):
                    self._hide_hover_popup(self._hover_menu_target)
                    event.accept(); return True
            if event.key() == Qt.Key.Key_Space:
                self._play(); event.accept(); return True
            if self._arrow_mode == "frame" and event.key() == Qt.Key.Key_Left: self._previous(); event.accept(); return True
            if self._arrow_mode == "frame" and event.key() == Qt.Key.Key_Right: self._next_frame(); event.accept(); return True
            if self._arrow_mode == "orbit" and event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
                azim, elev = {Qt.Key.Key_Left: (-3, 0), Qt.Key.Key_Right: (3, 0), Qt.Key.Key_Up: (0, -3), Qt.Key.Key_Down: (0, 3)}[event.key()]
                target = self.split_views[0] if self.mode == "split" else self.overlay_view
                target.orbit(azim, elev)
                if self.mode == "split": self._sync_camera(target)
                self._refresh_annotations()
                event.accept(); return True
        return super().eventFilter(watched, event)

    def _open_hover_menu(self):
        target = self._hover_menu_target
        if self._hover_button_entered and target is not None:
            if isinstance(target, QComboBox):
                if not target.view().isVisible():
                    target.showPopup()
            elif isinstance(target, QToolButton) and target.menu() is not None and not target.menu().isVisible():
                self._opening_menu_by_hover = True
                target.menu().popup(target.mapToGlobal(QPoint(0, target.height())))
                self._opening_menu_by_hover = False

    def _show_hover_popup(self, target):
        self._hover_menu_target = target
        if isinstance(target, QComboBox):
            target.showPopup()
        elif isinstance(target, QToolButton) and target.menu() is not None:
            target.menu().popup(target.mapToGlobal(QPoint(0, target.height())))

    def _hide_hover_popup(self, target):
        if isinstance(target, QComboBox):
            application = QApplication.instance()
            application.removeEventFilter(self)
            try:
                target.hidePopup()
            finally:
                application.installEventFilter(self)
        elif isinstance(target, QToolButton) and target.menu() is not None:
            target.menu().hide()

    def _dismiss_combo_popup(self, combo):
        if combo.view().isVisible():
            self._hide_hover_popup(combo)

    def _is_active_hover_menu(self, target):
        button = self._hover_menu_target
        if isinstance(button, QComboBox):
            view = button.view()
            return target is view or view.isAncestorOf(target) or target is view.window()
        if not isinstance(button, QToolButton) or button.menu() is None:
            return False
        root = button.menu()
        return target is root or root.isAncestorOf(target)

    def _close_hover_menu(self):
        button = self._hover_menu_target
        if self._hover_menu_latched:
            return
        menu = button.menu() if isinstance(button, QToolButton) else None
        combo_view = button.view() if isinstance(button, QComboBox) else None
        position = self._last_pointer_position
        pointer_widget = self._last_pointer_widget
        if pointer_widget is button and position is not None and button.rect().contains(button.mapFromGlobal(position)):
            return
        if position is None:
            from PySide6.QtGui import QCursor
            position = QCursor.pos()
        if button is not None and button.rect().contains(button.mapFromGlobal(position)):
            return
        if menu is not None:
            for child_menu in (menu, *menu.findChildren(QMenu)):
                if child_menu.isVisible() and child_menu.rect().contains(child_menu.mapFromGlobal(position)):
                    return
        if combo_view is not None and combo_view.isVisible() and combo_view.window().rect().contains(
                combo_view.window().mapFromGlobal(position)):
            return
        if menu is not None:
            menu.hide()
        elif combo_view is not None:
            self._hide_hover_popup(button)

    def closeEvent(self, event):
        application = QApplication.instance()
        application.removeEventFilter(self)
        self.hover_menu_timer.stop(); self.hover_close_timer.stop()
        if isinstance(self._hover_menu_target, QComboBox):
            self._hover_menu_target.hidePopup()
        super().closeEvent(event)
