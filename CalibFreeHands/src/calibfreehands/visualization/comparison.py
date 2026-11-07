from __future__ import annotations

import colorsys
import csv
import json
import re
from pathlib import Path
from time import monotonic

import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtCore import QEvent, QObject, QPoint, Qt, QTimer, QSignalBlocker, QRectF, QSize, QVariantAnimation
from PySide6.QtGui import QAction, QActionGroup, QImage, QWindow, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPalette, QColor
from PySide6.QtWidgets import (
    QApplication, QAbstractButton, QCheckBox, QColorDialog, QComboBox, QDialog, QFileDialog, QGridLayout,
    QHBoxLayout, QLabel, QMainWindow, QMenu, QPushButton, QScrollArea, QSizePolicy, QStyle, QStyleOptionToolButton,
    QSlider, QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem, QToolButton,
    QVBoxLayout, QWidget, QWidgetAction,
)

from calibfreehands.media.provider import MediaFrameProvider, release_shared_media_provider, shared_media_provider
from .topology import HAND_CONNECTIONS, RAW_TO_VIEW
from calibfreehands.pipeline.result_bundle import ResultBundle
from calibfreehands.pipeline.metrics import compare_results, metric_registry
from calibfreehands.joint_schema import JOINT_NAMES
from .viewer import (
    _GLResultView, _OverlayCanvas, load_bundle_result, load_npz_result,
    whole_video_hand_framing,
)
from .comparison_presentation import _JointAnnotationLayer, _ViewportPanel
from .playback import PlaybackClock

METRIC_FIELDS = ("source_identity", "reference_result_id", "candidate_result_id", "candidate_name", "metric_id",
                 "metric_name", "category", "protocol_version", "frame_index", "timestamp_s", "hand",
                 "joint_index", "joint_name", "value", "unit", "valid_status")


class _ModeSwitch(QAbstractButton):
    """Compact accessible Overlay/Split toggle with a neutral painted track."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setFixedSize(42, 22)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("View mode")
        self.setAccessibleDescription("Toggle between Overlay and Split views")
        self.setToolTip("View: Overlay")
        self._thumb_position = 0.0
        self._animation = QVariantAnimation(self)
        self._animation.setDuration(120)
        self._animation.valueChanged.connect(self._set_thumb_position)
        self.toggled.connect(self._animate_thumb)

    def sizeHint(self):
        return QSize(42, 22)

    def _animate_thumb(self, checked):
        self._animation.stop()
        self._animation.setStartValue(self._thumb_position)
        self._animation.setEndValue(1.0 if checked else 0.0)
        self._animation.start()

    def _set_thumb_position(self, value):
        self._thumb_position = float(value)
        self.update()

    def _dark_palette(self):
        window = self.palette().color(QPalette.ColorRole.Window)
        luminance = 0.2126 * window.redF() + 0.7152 * window.greenF() + 0.0722 * window.blueF()
        app_style = QApplication.instance().styleSheet().lower()
        return luminance < 0.45 or "#202124" in app_style

    def paintEvent(self, _event):
        dark = self._dark_palette()
        track = QColor("#555b63" if dark else "#b9bec5")
        border = QColor("#858b93" if dark else "#9299a1")
        thumb = QColor("#f2f3f4")
        thumb_border = QColor("#c7cbd0" if dark else "#59616a")
        if self.isDown():
            track = track.darker(108)
        elif self.underMouse():
            track = track.lighter(112)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        track_rect = QRectF(2, 2, self.width() - 4, self.height() - 4)
        painter.setPen(QPen(border, 1))
        painter.setBrush(track)
        painter.drawRoundedRect(track_rect, track_rect.height() / 2, track_rect.height() / 2)

        diameter = 16
        margin = 3
        travel = max(0, self.width() - 2 * margin - diameter)
        x = margin + travel * self._thumb_position
        thumb_rect = QRectF(x, (self.height() - diameter) / 2, diameter, diameter)
        painter.setPen(QPen(thumb_border, 1))
        painter.setBrush(thumb)
        painter.drawEllipse(thumb_rect)

        if self.hasFocus():
            focus_color = self.palette().color(QPalette.ColorRole.Midlight)
            painter.setPen(QPen(focus_color, 1, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(QRectF(0.5, 0.5, self.width() - 1, self.height() - 1), 10, 10)
        painter.end()


class _PinButton(QAbstractButton):
    """Compact, palette-aware icon toggle for keeping viewer controls visible."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setFixedSize(26, 26)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Pin controls")
        self.setToolTip("Keep controls visible")
        self.setAccessibleDescription("Not pinned: controls may hide during playback")

    def sizeHint(self):
        return QSize(26, 26)

    def paintEvent(self, _event):
        palette = self.palette()
        window = palette.color(QPalette.ColorRole.Window)
        luminance = 0.2126 * window.redF() + 0.7152 * window.greenF() + 0.0722 * window.blueF()
        dark = luminance < 0.45
        if dark:
            surface = QColor("#33FF33" if self.isChecked() else "#34383d")
            border = QColor("#249924" if self.isChecked() else "#5a6068")
            ink = QColor("#111111" if self.isChecked() else "#aab0b7")
        else:
            surface = QColor("#33FF33" if self.isChecked() else "#f0f1f2")
            border = QColor("#249924" if self.isChecked() else "#c4c8ce")
            ink = QColor("#111111" if self.isChecked() else "#666c74")
        if self.underMouse():
            surface = surface.darker(108) if self.isChecked() else surface.lighter(108)
        if self.isDown():
            surface = surface.darker(112)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(border, 1))
        painter.setBrush(surface)
        painter.drawRoundedRect(QRectF(1, 1, 24, 24), 5, 5)

        painter.save()
        painter.translate(self.width() / 2, self.height() / 2)
        painter.scale(0.82, 0.82)
        painter.translate(-14, -14)
        icon = QPainterPath()
        icon.moveTo(10, 7)
        icon.lineTo(18, 7)
        icon.lineTo(17, 10)
        icon.lineTo(15, 12)
        icon.lineTo(15, 17)
        icon.lineTo(18, 20)
        icon.lineTo(10, 20)
        icon.lineTo(13, 17)
        icon.lineTo(13, 12)
        icon.lineTo(11, 10)
        icon.closeSubpath()
        painter.setPen(QPen(ink, 1.5, Qt.PenStyle.SolidLine,
                            Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.setBrush(ink)
        painter.drawPath(icon)
        painter.drawLine(14, 20, 14, 24)
        painter.restore()
        if self.hasFocus():
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(palette.color(QPalette.ColorRole.Highlight), 1,
                                Qt.PenStyle.DashLine))
            painter.drawRoundedRect(QRectF(0.5, 0.5, 25, 25), 5, 5)
        painter.end()


def _finite(value):
    return float(value) if np.isfinite(value) else None


def _is_widget_or_popup_window(parent: QWidget, target: QObject) -> bool:
    if isinstance(target, QWidget):
        return target is parent or parent.isAncestorOf(target)
    if isinstance(target, QWindow):
        return target is parent.windowHandle() or target is parent.window().windowHandle()
    return False


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
    source_ids = {str(bundle.metadata.get("source_media_sha256") or "").casefold() for bundle in bundles}
    if len(bundles) > 1 and ("" in source_ids or len(source_ids) != 1):
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
        "source_media_sha256": bundles[0].metadata.get("source_media_sha256"),
        "reference_configuration_id": bundles[0].metadata.get("reference_configuration_id"),
    }
    return session, results


def _comparison_results(source):
    return _bundle_results(source) if isinstance(source, (list, tuple)) else _session_results(source)


def _color_for_index(index: int):
    palette = ((0x33 / 255, 0x33 / 255, 1.0), (1.0, 0x33 / 255, 1.0),
               (1.0, 1.0, 0x33 / 255), (0x33 / 255, 1.0, 1.0),
               (1.0, 0x33 / 255, 0x33 / 255), (0.95, 0.55, 0.12),
               (0.55, 0.78, 1.0), (0.95, 0.72, 0.86))
    if index < len(palette):
        return palette[index]
    hue = (index * 0.61803398875) % 1
    if 0.20 <= hue <= 0.45:
        hue = (hue + 0.35) % 1
    return colorsys.hsv_to_rgb(hue, 0.72, 0.9)


def _bone_color(color):
    return tuple(channel * 0.8 for channel in color)


def _manual_focal_display_name(model_id: str, saved_name: str, focal_px: float | None) -> str:
    """Normalize the legacy calibration label for display without mutating saved metadata."""
    if model_id != "checkerboard_reference":
        return saved_name
    name = saved_name.strip()
    known_label = name.startswith("Checkerboard Reference") or name.startswith("Manual Focal")
    if known_label and "px" in name.casefold():
        name = name.replace("Checkerboard Reference", "Manual Focal", 1)
        return re.sub(r"(?<=\d)\s+px\b", "\u00a0px", name, flags=re.IGNORECASE)
    if focal_px is None:
        return "Manual Focal"
    return f"Manual Focal {focal_px:g}\u00a0px"


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
        self.confidence_opacity_enabled = False
        super().__init__(loaded)
        self.hand_colors = (color, color)
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
            joint_color = (*base, alpha)
            bone_color = (*_bone_color(base), alpha)
            if self.joints[hand_index].visible():
                self.joints[hand_index].setData(color=joint_color)
            if self.bones[hand_index].visible():
                self.bones[hand_index].setData(color=bone_color)
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
        self.confidence_opacity_enabled = False
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
                    if self.show_joints or self.show_bones:
                        points.append(result.keypoints_3d[position, hand] @ RAW_TO_VIEW.T)
                    if self.show_mesh and loaded.has_mesh_topology and result.mano_vertices is not None:
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
            if self.joints[hand].visible(): self.joints[hand].setData(color=(*base, alpha))
            if self.bones[hand].visible(): self.bones[hand].setData(color=(*_bone_color(base), alpha))
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
                hand_color = color
                points = np.asarray(result.keypoints_3d[position, hand]) @ RAW_TO_VIEW.T
                if not np.isfinite(points).all():
                    continue
                rgba = (*hand_color, alpha)
                if self.show_joints:
                    joints.setData(pos=np.ascontiguousarray(points, dtype=np.float32), size=8, color=rgba)
                    joints.setVisible(True)
                if self.show_bones:
                    bone_points = np.ascontiguousarray(np.concatenate([points[[a, b]] for a, b in HAND_CONNECTIONS]), dtype=np.float32)
                    bones.setData(pos=bone_points, color=(*_bone_color(hand_color), alpha), mode="lines")
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
            _set_mesh_color(self.meshes[hand], self.configuration_colors[0], alpha * self.mesh_opacity)
        for config_index, (loaded, hands) in enumerate(zip(self.loaded_results[1:], self.extra_items), 1):
            for hand, (_, _, mesh) in enumerate(hands):
                alpha = _confidence_alpha(loaded.result, min(self.frame, loaded.frame_count - 1), hand) if self.confidence_opacity_enabled else 1.0
                _set_mesh_color(mesh, self.configuration_colors[config_index], alpha * self.mesh_opacity)


class ComparisonViewer(QWidget):
    def __init__(self, session_path: Path | list[ResultBundle]):
        super().__init__()
        self.bundle_results = list(session_path) if isinstance(session_path, (list, tuple)) else None
        self.session_path = None if self.bundle_results is not None else Path(session_path).resolve()
        self.session, entries = _comparison_results(session_path)
        self.single_mode = len(entries) == 1
        self.configurations = [config for config, _ in entries]
        self.loaded_results = [loaded for _, loaded in entries]
        self._owned_media_providers = tuple({
            id(loaded.media_frames): loaded.media_frames for loaded in self.loaded_results
            if isinstance(loaded.media_frames, MediaFrameProvider)
        }.values())
        self._media_providers_released = False
        for config, loaded in zip(self.configurations, self.loaded_results):
            focal = loaded.result.video_focal_px
            if focal is None:
                finite = loaded.result.focal_px[np.isfinite(loaded.result.focal_px)]
                focal = float(np.median(finite)) if finite.size else None
            config["name"] = _manual_focal_display_name(
                config.get("calibration_model_id", ""), config["name"], focal
            )
        names = [config["name"] for config in self.configurations]
        for index, config in enumerate(self.configurations):
            if names.count(config["name"]) > 1:
                ordinal = sum(name == config["name"] for name in names[:index + 1]) + 1
                self.configurations[index] = dict(config, name=f"{config['name']} ({ordinal})")
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
        self.reference_id = None if self.single_mode else self._default_reference_id()
        self.hand_view_cache = {}
        reference_index = (self._configuration_index(self.reference_id)
                           if self.reference_id is not None else None)
        candidate_index = 0
        self.candidate_colors = []
        for index in range(len(entries)):
            if index == reference_index:
                self.candidate_colors.append(_color_for_index(len(entries) - 1))
            else:
                self.candidate_colors.append(_color_for_index(candidate_index))
                candidate_index += 1
        self.configuration_colors = list(self.candidate_colors)
        if self.reference_id is not None:
            self.configuration_colors[reference_index] = (0x33 / 255, 1.0, 0x33 / 255)
        self.metric_definitions = metric_registry()
        self.selected_metric_ids = {"wrist_error", "mpjpe", "wrist_depth_error"}
        self._metric_cache = {}
        self.setWindowTitle("CalibFreeHands - Result" if self.single_mode else "CalibFreeHands - Comparison")
        self.setMinimumSize(1280, 720)

        self.source_overlay = _OverlayCanvas(self.loaded_results[0])
        self.source_overlay.setMinimumSize(360, 300)
        self.overlay_view = _OverlayComparisonView(self.loaded_results, self.configuration_colors)
        self.overlay_view.setMinimumSize(340, 260)
        self.overlay_panel = _ViewportPanel(self.overlay_view, list(range(len(entries))))
        self.overlay_view.hud = self.overlay_panel
        self.hand_view_cache[self.configurations[0]["id"]] = (
            tuple(float(value) for value in self.overlay_view.hand_view_target),
            float(self.overlay_view.hand_view_distance),
        )
        self.split_views = [_LinkedResultView(loaded, color, self._sync_camera)
                            for loaded, color in zip(self.loaded_results, self.configuration_colors)]
        for index, view in enumerate(self.split_views):
            view.setMinimumSize(340, 260); view.setAcceptDrops(not self.single_mode)
            view.viewer = self; view.configuration_id = self.configurations[index]["id"]
        self.split_panels = [_ViewportPanel(view, [index]) for index, view in enumerate(self.split_views)]
        for view, panel in zip(self.split_views, self.split_panels): view.hud = panel
        self.split_result_order = list(range(len(entries)))
        self.annotation_mode = "off"
        self.annotation_content = "name" if self.single_mode else "error"
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
        self.metric_status.setToolTip("Camera-space differences from the selected GT.")

        self.mode_selector = QWidget()
        mode_layout = QHBoxLayout(self.mode_selector)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(6)
        self.mode_label = QLabel("View")
        self.mode_switch = _ModeSwitch()
        self.pin_controls_button = _PinButton()
        mode_layout.addWidget(self.mode_label)
        mode_layout.addWidget(self.mode_switch)
        mode_layout.addWidget(self.pin_controls_button)
        self.mode_switch.toggled.connect(self._set_mode_from_switch)

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
            if config["id"] == self.reference_id:
                swatch.setEnabled(False)
                swatch.setToolTip("GT color is fixed to #33FF33")
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
        if self.single_mode:
            self.results_menu_button.hide()

        self.reference_label = QLabel("GT" + (" (fallback)" if self._reference_fallback else ""))
        self.reference_selector = QComboBox()
        for config in self.configurations:
            self.reference_selector.addItem(config["name"], config["id"])
        self.reference_selector.setCurrentIndex(next((i for i, config in enumerate(self.configurations)
                                                      if config["id"] == self.reference_id), 0))
        self.reference_selector.setMaximumWidth(220)
        self.reference_selector.setMinimumWidth(120)
        self.reference_selector.setToolTip(self.reference_selector.currentText())
        self.reference_selector.currentIndexChanged.connect(self._reference_changed)
        if self.single_mode:
            for widget in (self.reference_label, self.reference_selector):
                widget.hide()

        self.metric_menu = QMenu(self)
        self.metric_checks = {}
        self._rebuild_metrics_menu()
        self.metric_menu_button = QToolButton()
        self.metric_menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.metric_menu_button.setMenu(self.metric_menu)
        self._update_metrics_button()
        if self.single_mode:
            self.metric_menu_button.hide()

        self.confidence_menu = QMenu(self)
        self.confidence_opacity_checkbox = self._menu_checkbox(self.confidence_menu, "Use as opacity")
        self.confidence_numeric_checkbox = self._menu_checkbox(self.confidence_menu, "Numeric value")
        self.confidence_bar_checkbox = self._menu_checkbox(self.confidence_menu, "Bar")
        self.confidence_opacity_checkbox.toggled.connect(self._confidence_opacity_changed)
        self.confidence_numeric_checkbox.toggled.connect(self._refresh_metric_panels)
        self.confidence_bar_checkbox.toggled.connect(self._refresh_metric_panels)
        self.confidence_menu_button = QToolButton()
        self.confidence_menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.confidence_menu_button.setMenu(self.confidence_menu)
        self.confidence_menu_button.setText("Confidence Display")
        confidence_description = "YOLO hand detector bounding-box confidence; not joint or reconstruction confidence."
        self.confidence_menu_button.setToolTip(confidence_description)
        self.confidence_menu_button.setAccessibleName("Hand Detection Confidence")
        self.confidence_menu_button.setAccessibleDescription(confidence_description)
        self._update_confidence_button()

        self.error_vectors_checkbox = QCheckBox("Error vectors")
        self.error_candidate_selector = QComboBox()
        for config in self.configurations:
            if config["id"] != self.reference_id:
                self.error_candidate_selector.addItem(config["name"], config["id"])
        self.error_candidate_selector.setMaximumWidth(190)
        self.error_candidate_selector.setEnabled(self.error_candidate_selector.count() > 0)
        self.error_vectors_checkbox.setEnabled(self.error_candidate_selector.count() > 0)
        self.selected_error_candidate_id = self.error_candidate_selector.currentData()
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
        if self.single_mode:
            self.error_vectors_button.hide()

        self.annotation_menu = QMenu(self)
        self.annotation_group = QActionGroup(self); self.annotation_group.setExclusive(True)
        self.annotation_actions = {}
        for text, mode in (("All joints", "all"), ("Hovered joint", "hovered"),
                           ("Key joints", "key_joints"), ("Worst 5", "worst5"), ("Off", "off")):
            action = QAction(text, self); action.setCheckable(True); action.setChecked(mode == "off")
            action.triggered.connect(lambda checked, value=mode: self.set_annotation_mode(value) if checked else None)
            self.annotation_group.addAction(action); self.annotation_menu.addAction(action)
            self.annotation_actions[mode] = action
        self.annotation_button = QToolButton(); self.annotation_button.setText("Annotations")
        self.annotation_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.annotation_button.setMenu(self.annotation_menu)
        self.annotation_menu.addSeparator()
        self.annotation_content_menu = self.annotation_menu.addMenu("Label")
        self.annotation_content_group = QActionGroup(self); self.annotation_content_group.setExclusive(True)
        self.annotation_content_actions = {}
        for label, content in (("Error (mm)", "error"), ("Joint name", "name")):
            action = QAction(label, self); action.setCheckable(True); action.setChecked(content == self.annotation_content)
            action.triggered.connect(lambda checked, value=content: self._set_annotation_content(value) if checked else None)
            self.annotation_content_group.addAction(action); self.annotation_content_menu.addAction(action)
            self.annotation_content_actions[content] = action
        if self.single_mode:
            self.annotation_content_actions["error"].setVisible(False)
            self.annotation_actions["worst5"].setVisible(False)
        self.hud_menu = QMenu(self)
        size_menu = self.hud_menu.addMenu("Size")
        for size in ("small", "medium", "large"):
            action = QAction(size.title(), self); action.triggered.connect(lambda _=False, value=size: self._set_hud(scale=value))
            size_menu.addAction(action)
        self.hud_background_menu = self.hud_menu.addMenu("Background")
        self.hud_background_group = QActionGroup(self); self.hud_background_group.setExclusive(True)
        self.hud_background_actions = {}
        for label, style in (("None", "none"), ("Subtle", "subtle")):
            action = QAction(label, self); action.setCheckable(True); action.setChecked(style == "none")
            action.triggered.connect(lambda checked, value=style: self._set_hud_background(value) if checked else None)
            self.hud_background_group.addAction(action); self.hud_background_menu.addAction(action)
            self.hud_background_actions[style] = action
        self.hud_opacity_widget = QWidget()
        opacity_layout = QHBoxLayout(self.hud_opacity_widget)
        opacity_layout.setContentsMargins(8, 0, 8, 0)
        self.hud_opacity_label = QLabel("HUD opacity")
        self.hud_opacity_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        opacity_layout.addWidget(self.hud_opacity_label)
        self.hud_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.hud_opacity_slider.setRange(20, 100)
        self.hud_opacity_slider.setValue(100)
        self.hud_opacity_slider.setFixedWidth(100)
        self.hud_opacity_slider.valueChanged.connect(lambda value: self._set_hud(opacity=value / 100))
        opacity_layout.addWidget(self.hud_opacity_slider, 0, Qt.AlignmentFlag.AlignVCenter)
        self.hud_opacity_action = QWidgetAction(self.hud_menu)
        self.hud_opacity_action.setDefaultWidget(self.hud_opacity_widget)
        self.hud_menu.insertAction(size_menu.menuAction(), self.hud_opacity_action)
        self._fit_hud_opacity_row()
        self.hud_button = QToolButton(); self.hud_button.setText("HUD")
        self.hud_button.setCheckable(True); self.hud_button.setChecked(True)
        self.hud_visible = True
        self.hud_button.setAccessibleName("HUD visibility")
        self.hud_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup); self.hud_button.setMenu(self.hud_menu)
        self.hud_button.toggled.connect(self._set_hud_visible)
        self._set_hud_visibility_label(True)

        self.detach_button = QPushButton("Detach")
        self.detach_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.detach_button.clicked.connect(self.detach_all)
        self.more_menu = QMenu(self)
        self.more_menu.addAction("Detach panels", self.detach_all)
        self.performance_stats_action = self.more_menu.addAction("Playback Stats")
        self.performance_stats_action.setCheckable(True)
        self.performance_stats_action.toggled.connect(self._set_performance_stats)
        self.more_button = QToolButton(); self.more_button.setText("More")
        self.more_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup); self.more_button.setMenu(self.more_menu)
        self.export_csv_button = QPushButton("CSV")
        self.export_json_button = QPushButton("JSON")
        for button, callback in ((self.export_csv_button, self._export_csv), (self.export_json_button, self._export_json)):
            button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            button.clicked.connect(callback)
        if self.single_mode:
            self.export_csv_button.hide()
            self.export_json_button.hide()
            self.metric_status.hide()

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(14, 8, 14, 6)
        toolbar.setSpacing(7)
        toolbar.addWidget(self.mode_selector)
        toolbar.addSpacing(5)
        toolbar.addWidget(self.results_menu_button)
        toolbar.addWidget(self.reference_label); toolbar.addWidget(self.reference_selector)
        toolbar.addWidget(self.metric_menu_button)
        toolbar.addWidget(self.confidence_menu_button)
        self.error_vectors_button.setToolTip("Show selected GT versus candidate error vectors")
        toolbar.addWidget(self.annotation_button); toolbar.addWidget(self.hud_button)
        toolbar.addWidget(self.more_button)
        toolbar.addStretch(1)
        self.toolbar = toolbar

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
                             ("Hand View", "hand_view"), ("Metric Overview", "metric_overview"),
                             ("Fit All", "fit_all")):
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
                             ("Hand View", "hand_view"), ("Metric Overview", "metric_overview"),
                             ("Fit All", "fit_all")):
            camera_menu.addAction(text, lambda _=False, name=method: self._apply_camera_preset(name))
        self.more_menu.addSeparator()
        self.export_csv_action = self.more_menu.addAction("Export CSV…", self._export_csv)
        self.export_json_action = self.more_menu.addAction("Export JSON…", self._export_json)
        if self.single_mode:
            self.export_csv_action.setVisible(False)
            self.export_json_action.setVisible(False)

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
        self.playback_group = QWidget()
        self.playback_group.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        playback_group_layout = QHBoxLayout(self.playback_group)
        playback_group_layout.setContentsMargins(0, 0, 0, 0)
        playback_group_layout.setSpacing(3)
        playback_group_layout.addWidget(QLabel("Playback"))
        self.playback_icon_kinds = ["previous_frame", "play", "next_frame"]
        accessible_names = ("Previous frame", "Play", "Next frame")
        foreground = self.palette().color(QPalette.ColorRole.ButtonText)
        disabled = self.palette().color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText)
        self.playback_icons = [self._make_playback_icon(kind, foreground, disabled)
                               for kind in self.playback_icon_kinds]
        for callback, icon, accessible_name in zip((self._previous, self._play, self._next_frame), self.playback_icons, accessible_names):
            button = QToolButton()
            button.setIcon(icon)
            button.setAccessibleName(accessible_name)
            button.setToolTip(accessible_name)
            button.setFixedSize(36, 32)
            button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            button.clicked.connect(callback)
            playback_group_layout.addWidget(button)
            self.playback_buttons.append(button)
        self.speed_group = QWidget()
        self.speed_group.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        speed_group_layout = QHBoxLayout(self.speed_group)
        speed_group_layout.setContentsMargins(0, 0, 0, 0)
        speed_group_layout.setSpacing(3)
        self.speed_label = QLabel("Speed")
        speed_group_layout.addWidget(self.speed_label)
        self.speed_down_button = QToolButton(); self.speed_down_button.setText("▼")
        self.speed_down_button.setFixedSize(28, 32)
        self.speed_down_button.setAccessibleName("Decrease playback speed")
        self.speed_down_button.setToolTip("Decrease playback speed")
        self.speed_down_button.clicked.connect(lambda: self._step_playback_speed(-1))
        self.speed_value_label = QLabel("1.0×")
        self.speed_value_label.setFixedWidth(self.speed_value_label.fontMetrics().horizontalAdvance("1.25×") + 4)
        self.speed_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.speed_up_button = QToolButton(); self.speed_up_button.setText("▲")
        self.speed_up_button.setFixedSize(28, 32)
        self.speed_up_button.setAccessibleName("Increase playback speed")
        self.speed_up_button.setToolTip("Increase playback speed")
        self.speed_up_button.clicked.connect(lambda: self._step_playback_speed(1))
        for control in (self.speed_down_button, self.speed_value_label, self.speed_up_button):
            speed_group_layout.addWidget(control)
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
        self.chrome = QWidget(); chrome_layout = QVBoxLayout(self.chrome); chrome_layout.setContentsMargins(14, 8, 14, 12); chrome_layout.setSpacing(6)
        chrome_layout.addLayout(toolbar); chrome_layout.addWidget(self.slider)
        playback = QHBoxLayout()
        self.playback_controls_layout = playback
        playback.addWidget(self.playback_group)
        playback.addSpacing(12)
        playback.addWidget(self.speed_group)
        playback.addStretch(1)
        playback.addWidget(self.status)
        playback.addWidget(self.metric_status, 1)
        chrome_layout.addLayout(playback); chrome_layout.addWidget(self.performance_stats_label)
        layout.addWidget(self.chrome, 0, 0, Qt.AlignmentFlag.AlignBottom)
        self.auto_hide_timer = QTimer(self); self.auto_hide_timer.setSingleShot(True); self.auto_hide_timer.setInterval(2500)
        self.auto_hide_timer.timeout.connect(self._auto_hide_controls)
        self.pin_controls_button.toggled.connect(self._set_pin_state)
        self.pin_controls_button.toggled.connect(self._set_pin_label)
        self._sync_speed_controls()
        self._sync_play_icon()
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
        self.hand_view()

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

    def _fit_hud_opacity_row(self):
        self.hud_menu.ensurePolished()
        self.hud_menu.adjustSize()
        reference_height = self.hud_menu.actionGeometry(
            self.hud_background_menu.menuAction()
        ).height()
        if reference_height > 0:
            self.hud_opacity_widget.setFixedHeight(reference_height)
        self.hud_menu.adjustSize()

    def _set_hud_visible(self, visible):
        self.hud_visible = bool(visible)
        for panel in (self.overlay_panel, *self.split_panels):
            panel.setVisible(self.hud_visible)
        self._set_hud_visibility_label(self.hud_visible)

    def _set_hud_visibility_label(self, visible):
        label = "Hide HUD" if visible else "Show HUD"
        self.hud_button.setToolTip(label)
        self.hud_button.setAccessibleDescription(label)

    def _select_error_candidate(self, candidate_id):
        if candidate_id == self.reference_id or candidate_id not in self.error_vector_actions:
            return
        self.selected_error_candidate_id = candidate_id
        index = self.error_candidate_selector.findData(candidate_id)
        if index >= 0:
            blocker = QSignalBlocker(self.error_candidate_selector)
            self.error_candidate_selector.setCurrentIndex(index)
            del blocker
        self.error_candidate_selector.hide()
        self.error_vectors_checkbox.setChecked(True)
        self._set_error_vectors(True)
        name = next(config["name"] for config in self.configurations if config["id"] == candidate_id)
        self.error_vectors_button.setText(f"Vectors: {name}")
        self.error_vectors_button.setToolTip(f"Show selected GT versus {name} error vectors")

    def _show_controls(self):
        self.chrome.show()
        if self.timer.isActive() and not self.pin_controls_button.isChecked():
            self.auto_hide_timer.start()

    def _set_pin_state(self, pinned):
        if pinned:
            self.auto_hide_timer.stop()
            self.chrome.show()
        else:
            self._show_controls()

    def _set_pin_label(self, pinned):
        self.pin_controls_button.setToolTip("Controls pinned" if pinned else "Keep controls visible")
        self.pin_controls_button.setAccessibleDescription(
            "Pinned: controls remain visible" if pinned else "Not pinned: controls may hide during playback")

    def _menu_about_to_show(self, menu):
        if not self._opening_menu_by_hover and menu is getattr(self._hover_menu_target, "menu", lambda: None)():
            self._hover_menu_latched = True

    def _menu_about_to_hide(self, menu):
        button = self._hover_menu_target
        if isinstance(button, QToolButton) and button.menu() is menu:
            self._hover_menu_latched = False

    def _auto_hide_controls(self):
        if (not self.timer.isActive() or self.pin_controls_button.isChecked()
                or self._panel_reorder_active): return
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
            is_gt = config["id"] == self.reference_id
            swatch.setEnabled(not is_gt)
            swatch.setToolTip("GT color is fixed to #33FF33" if is_gt else "Choose candidate result color")
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
        if self.single_mode and content == "error":
            raise ValueError("single-result annotations only support joint names")
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
        self.results_menu_button.setText(f"Visible ({visible}/{len(self.visibility_checks)})")
        self.results_menu_button.setToolTip("Show or hide comparison results")

    def _update_metrics_button(self):
        self.metric_menu_button.setText("Error Types")
        self.metric_menu_button.setToolTip("Select error types shown in the HUD")

    def _update_confidence_button(self):
        self.confidence_menu_button.setText("Confidence Display")

    def _configuration_index(self, configuration_id):
        return next(index for index, config in enumerate(self.configurations)
                    if config["id"] == configuration_id)

    def _hand_view_source_id(self):
        return self.reference_id or self.configurations[0]["id"]

    def _hand_view_for(self, configuration_id):
        framing = self.hand_view_cache.get(configuration_id)
        if framing is None:
            index = self._configuration_index(configuration_id)
            result = self.loaded_results[index].result
            framing = whole_video_hand_framing(result.keypoints_3d, result.detected)
            self.hand_view_cache[configuration_id] = (
                tuple(float(value) for value in framing[0]), float(framing[1])
            )
        return self.hand_view_cache[configuration_id]

    def _rebuild_metrics_menu(self):
        self.metric_menu.clear()
        self.metric_checks.clear()
        categories = {}
        available_ids = {definition.metric_id for definition in self.metric_definitions
                         if definition.status.value == "READY" and self._metric_has_data(definition.metric_id)}
        self.selected_metric_ids.intersection_update(available_ids)
        for definition in self.metric_definitions:
            if definition.metric_id not in available_ids:
                continue
            category = categories.get(definition.category)
            if category is None:
                category_label = "GT Differences" if definition.category == "Reference Accuracy" else definition.category
                category = self.metric_menu.addMenu(category_label)
                categories[definition.category] = category
            check = QCheckBox(definition.name)
            check.setChecked(definition.metric_id in self.selected_metric_ids)
            check.toggled.connect(lambda value, key=definition.metric_id: self._set_metric_selected(key, value))
            action = QWidgetAction(category)
            action.setDefaultWidget(check)
            category.addAction(action)
            self.metric_checks[definition.metric_id] = check

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
        if self.single_mode:
            return False
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
        button.setToolTip("Choose candidate result color")

    def _choose_color(self, index):
        if self.configurations[index]["id"] == self.reference_id:
            return
        from PySide6.QtGui import QColor
        current = QColor(*(round(channel * 255) for channel in self.candidate_colors[index]))
        chosen = QColorDialog.getColor(current, self, "Result color")
        if not chosen.isValid():
            return
        self._apply_configuration_color(index, (chosen.redF(), chosen.greenF(), chosen.blueF()))

    def _apply_configuration_color(self, index, color):
        if self.configurations[index]["id"] == self.reference_id:
            return
        self.candidate_colors[index] = tuple(color)
        self._refresh_effective_colors()

    def _refresh_effective_colors(self):
        self.configuration_colors[:] = self.candidate_colors
        if self.reference_id is not None:
            self.configuration_colors[self._configuration_index(self.reference_id)] = (0x33 / 255, 1.0, 0x33 / 255)
        self.overlay_view.configuration_colors[:] = self.configuration_colors
        for index, (view, color) in enumerate(zip(self.split_views, self.configuration_colors)):
            view.configuration_color = color
            view.hand_colors = (color, color)
            view.set_frame(self.slider.value())
            swatch = self.color_buttons[index]
            self._style_swatch(swatch, color)
            is_gt = self.configurations[index]["id"] == self.reference_id
            swatch.setEnabled(not is_gt)
            swatch.setToolTip("GT color is fixed to #33FF33" if is_gt else "Choose candidate result color")
        self.overlay_view.set_frame(self.slider.value())
        for _window, _source, views in self._detached_windows:
            for view, color in zip(views, self.configuration_colors):
                view.configuration_color = color
                view.hand_colors = (color, color)
                view.set_frame(self.slider.value())
        self._render_active_frame()
        self._refresh_metric_panels()

    def _reference_changed(self, _index):
        if self.single_mode:
            return
        self.reference_id = self.reference_selector.currentData()
        self._hand_view_for(self.reference_id)
        self.reference_selector.setToolTip(self.reference_selector.currentText())
        self.reference_label.setText("GT")
        self._rebuild_metrics_menu()
        self._refresh_effective_colors()
        self.error_candidate_selector.clear()
        for config in self.configurations:
            if config["id"] != self.reference_id:
                self.error_candidate_selector.addItem(config["name"], config["id"])
        self.selected_error_candidate_id = self.error_candidate_selector.currentData()
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
        if self.single_mode:
            self._metric_cache = {}
            return
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

    def _confidence_opacity_changed(self, enabled):
        self._update_confidence_button()
        for view in [self.overlay_view, *self.split_views]:
            view.confidence_opacity_enabled = enabled
        for hand, mesh in enumerate(self.overlay_view.meshes):
            alpha = _confidence_alpha(self.overlay_view.loaded.result, self.overlay_view.frame, hand) if enabled else 1.0
            _set_mesh_color(mesh, self.configuration_colors[0], alpha * self.overlay_view.mesh_opacity)
        for config_index, hands in enumerate(self.overlay_view.extra_items, 1):
            result = self.loaded_results[config_index].result
            frame = min(self.overlay_view.frame, self.loaded_results[config_index].frame_count - 1)
            for hand, (_, _, mesh) in enumerate(hands):
                alpha = _confidence_alpha(result, frame, hand) if enabled else 1.0
                _set_mesh_color(mesh, self.configuration_colors[config_index], alpha * self.overlay_view.mesh_opacity)
        for view in self.split_views:
            for hand, mesh in enumerate(view.meshes):
                alpha = _confidence_alpha(view.loaded.result, view.frame, hand) if enabled else 1.0
                _set_mesh_color(mesh, view.hand_colors[hand], alpha * view.mesh_opacity)
        self._render_active_frame()

    def _set_error_vectors(self, enabled):
        if self.single_mode:
            return
        candidate_ids = {config["id"] for config in self.configurations if config["id"] != self.reference_id}
        if self.selected_error_candidate_id not in candidate_ids:
            self.selected_error_candidate_id = next(iter(candidate_ids), None)
        enabled = bool(enabled and self.selected_error_candidate_id is not None)
        if self.error_vectors_checkbox.isChecked() != enabled:
            blocker = QSignalBlocker(self.error_vectors_checkbox)
            self.error_vectors_checkbox.setChecked(enabled)
            del blocker
        self.error_candidate_selector.hide()
        self.error_vectors_checkbox.setEnabled(bool(candidate_ids))
        self.overlay_view.error_vectors_enabled = enabled
        self.overlay_view.reference_index = next((i for i, config in enumerate(self.configurations)
                                                  if config["id"] == self.reference_id), 0)
        self.overlay_view.error_candidate = next((i for i, c in enumerate(self.configurations)
                                                   if c["id"] == self.selected_error_candidate_id), -1)
        self.error_vectors_off.setChecked(not enabled)
        for candidate_id, action in self.error_vector_actions.items():
            action.setChecked(enabled and candidate_id == self.selected_error_candidate_id)
        if enabled and self.mode == "overlay":
            self.overlay_view.set_frame(self.slider.value())
        if not enabled:
            for item in self.overlay_view.error_items:
                item.setVisible(False)
            self.error_vectors_button.setText("Vectors: Off")
            self.error_vectors_button.setToolTip("Show selected GT versus candidate error vectors")
        else:
            name = next(config["name"] for config in self.configurations
                        if config["id"] == self.selected_error_candidate_id)
            self.error_vectors_button.setText(f"Vectors: {name}")
            self.error_vectors_button.setToolTip(f"Show selected GT versus {name} error vectors")

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
        camera_source = self.overlay_view if self.mode == "overlay" else self.split_views[0]
        for view in detached_views:
            view.setCameraParams(**camera_source.cameraParams())
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
        if mode != self.mode:
            camera_source = self.overlay_view if self.mode == "overlay" else self.split_views[0]
            self._sync_camera(camera_source)
        self.mode = mode
        self.views.setCurrentIndex(0 if mode == "overlay" else 1)
        self.mode_switch.setChecked(mode == "split")
        self.mode_switch.setToolTip("View: Split" if mode == "split" else "View: Overlay")
        self._render_active_frame()
        self._refresh_metric_panels()
        self._refresh_annotations()

    def _set_mode_from_switch(self, value):
        target = "split" if value else "overlay"
        if target != self.mode:
            self.set_mode(target)

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
            view._update_reference_visibility()
        self._render_active_frame()

    def _set_mesh(self, value):
        self.mesh_opacity_slider.setEnabled(bool(value) and self.mesh_checkbox.isEnabled())
        if value:
            for check in (self.joint_checkbox, self.bone_checkbox):
                check.setChecked(False)
        for view in [self.overlay_view, *self.split_views]:
            view.show_mesh = value
        self._render_active_frame()

    def _set_joint(self, value):
        if value and self.mesh_checkbox.isChecked():
            self.mesh_checkbox.setChecked(False)
        self._set_geometry("show_joints", value)

    def _set_bone(self, value):
        if value and self.mesh_checkbox.isChecked():
            self.mesh_checkbox.setChecked(False)
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

    def _sync_speed_controls(self):
        index = PlaybackClock.SPEEDS.index(self.playback_clock.speed)
        speed = PlaybackClock.SPEEDS[index]
        self.speed_value_label.setText(f"{speed:.1f}×" if speed == 1.0 else f"{speed:g}×")
        self.speed_down_button.setEnabled(index > 0)
        self.speed_up_button.setEnabled(index < len(PlaybackClock.SPEEDS) - 1)

    def _step_playback_speed(self, direction):
        index = PlaybackClock.SPEEDS.index(self.playback_clock.speed)
        target = min(max(index + int(direction), 0), len(PlaybackClock.SPEEDS) - 1)
        self.playback_clock.set_speed(PlaybackClock.SPEEDS[target])
        self._sync_speed_controls()

    def _sync_play_icon(self):
        playing = self.timer.isActive()
        button = self.playback_buttons[1]
        self.playback_icon_kinds[1] = "pause" if playing else "play"
        button.setIcon(self._make_playback_icon(
            self.playback_icon_kinds[1], self.palette().color(QPalette.ColorRole.ButtonText),
            self.palette().color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText)))
        button.setAccessibleName("Pause" if playing else "Play")
        button.setToolTip("Pause" if playing else "Play")

    @staticmethod
    def _make_playback_icon(kind, foreground, disabled_foreground=None):
        icon = QIcon()
        colors = ((QIcon.Mode.Normal, foreground),
                  (QIcon.Mode.Disabled, disabled_foreground or foreground))
        for mode, color in colors:
            pixmap = QPixmap(32, 32)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QPen(color, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                                Qt.PenJoinStyle.RoundJoin))
            painter.setBrush(color)
            if kind == "previous_frame":
                painter.drawLine(20, 8, 11, 16)
                painter.drawLine(11, 16, 20, 24)
            elif kind == "next_frame":
                painter.drawLine(12, 8, 21, 16)
                painter.drawLine(21, 16, 12, 24)
            elif kind == "play":
                path = QPainterPath()
                path.moveTo(11, 7); path.lineTo(24, 16); path.lineTo(11, 25); path.closeSubpath()
                painter.drawPath(path)
            elif kind == "pause":
                painter.drawRoundedRect(QRectF(9, 7, 5, 18), 1.5, 1.5)
                painter.drawRoundedRect(QRectF(18, 7, 5, 18), 1.5, 1.5)
            painter.end()
            icon.addPixmap(pixmap, mode, QIcon.State.Off)
        return icon

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange,
                            QEvent.Type.StyleChange) and hasattr(self, "playback_buttons"):
            foreground = self.palette().color(QPalette.ColorRole.ButtonText)
            disabled = self.palette().color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText)
            for index, button in enumerate(self.playback_buttons):
                button.setIcon(self._make_playback_icon(self.playback_icon_kinds[index], foreground, disabled))

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
            for view in [self.overlay_view, *self.split_views]:
                if view is not source: view.setCameraParams(**params)
        finally:
            self._syncing_camera = False
        self._refresh_annotations()

    def fit_all(self):
        points = self.overlay_view._geometry(self.slider.value())
        if not len(points):
            return False
        if self.mode == "overlay":
            fitted = self.overlay_view.fit_camera_to_points(points)
            if fitted:
                self._refresh_annotations()
            return fitted

        visible_indices = [index for index, check in enumerate(self.visibility_checks) if check.isChecked()]
        if not visible_indices:
            return False
        visible_views = [self.split_views[index] for index in visible_indices]
        aspect = max((view.width() / view.height() for view in visible_views if view.height() > 0),
                     default=4 / 3)
        source = visible_views[0]
        if not source.fit_camera_to_points(points, aspect=aspect):
            return False
        self._sync_camera(source)
        return True

    def hand_view(self):
        target, distance = self._hand_view_for(self._hand_view_source_id())
        source = self.overlay_view if self.mode == "overlay" else self.split_views[0]
        source.apply_hand_view(target, distance)
        self._sync_camera(source)
        self._refresh_annotations()

    def metric_overview(self):
        source = self.overlay_view if self.mode == "overlay" else self.split_views[0]
        source.metric_overview()
        self._sync_camera(source)
        self._refresh_annotations()

    def reset_camera(self):
        self.hand_view()

    def _apply_camera_preset(self, method):
        if method == "hand_view":
            self.hand_view()
            return
        if method == "metric_overview":
            self.metric_overview()
            return
        if method == "fit_all":
            self.fit_all()
            return
        if method == "reset_camera":
            self.reset_camera()
            return
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
        self._sync_play_icon()

    def _export_csv(self): self._export("csv")
    def _export_json(self): self._export("json")
    def _export(self, kind):
        if self.single_mode:
            return
        if self.session_path is not None:
            initial = self.session_path.with_suffix(f".{kind}")
        else:
            source_name = self.bundle_results[0].metadata.get("source_media_name", "comparison")
            initial = Path(self.bundle_results[0].metadata["source_media_path"]).with_name(f"{Path(source_name).stem}-metrics.{kind}")
        path, _ = QFileDialog.getSaveFileName(self, "Export GT differences", str(initial), f"{kind.upper()} (*.{kind})")
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
                        if watched is self.hud_button:
                            from PySide6.QtGui import QCursor
                            menu_area = self._is_hud_menu_arrow(watched.mapFromGlobal(QCursor.pos()))
                        else:
                            menu_area = True
                        if menu_area:
                            self.hover_close_timer.stop()
                            self._hover_menu_target = watched
                            self._hover_button_entered = True
                            self.hover_menu_timer.start()
                elif event.type() == QEvent.Type.Leave:
                    self._hover_button_entered = False
                    self.hover_close_timer.start()
                elif event.type() == QEvent.Type.MouseMove and watched is self.hud_button:
                    menu_area = self._is_hud_menu_arrow(event.position().toPoint())
                    self._hover_button_entered = menu_area
                    if menu_area:
                        self._hover_menu_target = watched
                        self.hover_close_timer.stop()
                        self.hover_menu_timer.start()
                    else:
                        self.hover_menu_timer.stop()
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
                if watched is self.hud_button and not self._is_hud_menu_arrow(event.position().toPoint()):
                    if watched.menu().isVisible():
                        self._hide_hover_popup(watched)
                    return False
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
                if watched in (self.mode_switch, self.pin_controls_button):
                    return super().eventFilter(watched, event)
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
            return _is_widget_or_popup_window(view, target)
        if not isinstance(button, QToolButton) or button.menu() is None:
            return False
        root = button.menu()
        return _is_widget_or_popup_window(root, target)

    def _is_hud_menu_arrow(self, position):
        option = QStyleOptionToolButton()
        self.hud_button.initStyleOption(option)
        arrow_rect = self.hud_button.style().subControlRect(
            QStyle.ComplexControl.CC_ToolButton, option,
            QStyle.SubControl.SC_ToolButtonMenu, self.hud_button
        )
        return arrow_rect.contains(position)

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
        if not self._media_providers_released:
            self._media_providers_released = True
            for provider in self._owned_media_providers:
                release_shared_media_provider(provider)
        super().closeEvent(event)
