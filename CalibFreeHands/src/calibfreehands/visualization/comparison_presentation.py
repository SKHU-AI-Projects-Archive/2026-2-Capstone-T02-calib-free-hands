from __future__ import annotations

import numpy as np
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QVector4D
from PySide6.QtWidgets import QCheckBox, QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QProgressBar, QPushButton, QSizePolicy, QToolButton, QVBoxLayout, QWidget

from calibfreehands.joint_schema import JOINT_NAMES
from .topology import RAW_TO_VIEW


class _ViewportPanel(QWidget):
    def __init__(self, view, configuration_indices):
        super().__init__(view)
        self.view = view
        self.configuration_indices = configuration_indices
        self.card = QFrame(self)
        self.card.setObjectName("metricOverlay")
        self.card.setFrameShape(QFrame.Shape.NoFrame)
        self.opacity_effect = QGraphicsOpacityEffect(self.card)
        self.card.setGraphicsEffect(self.opacity_effect)
        self.background = "none"
        self._drop_active = False
        self._fit_signature = None
        self.card.setStyleSheet("QFrame#metricOverlay { background: transparent; color: white; border: none; }")
        self.card.setMaximumWidth(330)
        self.card.setMaximumHeight(16777215)
        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(8, 6, 8, 6)
        card_layout.setSpacing(2)
        self.body = QWidget()
        self.body.setStyleSheet("background: transparent;")
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.setSpacing(2)
        self.body.installEventFilter(self)
        card_layout.addWidget(self.body)
        self.anchor = "top-left"
        self.scale = "medium"
        self.opacity = 1.0
        self.opacity_effect.setOpacity(self.opacity)
        self._apply_style()
        self.installEventFilter(self)
        view.installEventFilter(self)
        self._position()
        self.metric_rows = {}
        self.details_buttons = {}
        self.reference_badges = {}
        self.confidence_rows = {}
        self.confidence_bars = {}

    def eventFilter(self, watched, event):
        if watched is self.view and event.type() == QEvent.Type.Resize:
            self._position()
        return super().eventFilter(watched, event)

    def dragEnterEvent(self, event):
        if hasattr(self.view, "configuration_id") and event.mimeData().hasText():
            self._drop_active = True
            self._apply_style()
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._drop_active = False
        self._apply_style()
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        self._drop_active = False
        self._apply_style()
        if hasattr(self.view, "configuration_id") and event.mimeData().hasText():
            self.view.viewer.reorder_result(event.mimeData().text(), self.view.configuration_id)
            event.acceptProposedAction()
        else:
            event.ignore()

    def _apply_style(self):
        background = "rgba(25, 30, 38, 175)" if self.background == "subtle" else "transparent"
        border = "2px solid palette(highlight)" if self._drop_active else "none"
        self.card.setStyleSheet(
            f"QFrame#metricOverlay {{ background-color: {background}; color: white; border: {border}; }}")
        self._fit_signature = None
        self._fit_content()

    def set_background(self, style):
        if style not in ("none", "subtle"):
            raise ValueError(style)
        if style != self.background:
            self.background = style
            self._apply_style()

    def _position(self):
        self.setGeometry(self.view.rect())
        content_width = self.body_layout.sizeHint().width() + 20
        content_height = self.body_layout.sizeHint().height() + 12
        width = min(330, max(200, self.view.width() - 24), max(240, content_width + 60))
        height = min(max(64, content_height), max(64, self.view.height() - 24))
        self.card.setFixedSize(width, height)
        x = 12 if "left" in self.anchor else max(0, self.width() - width - 12)
        y = 12 if "top" in self.anchor else max(0, self.height() - height - 12)
        self.card.move(x, y)
        self.opacity_effect.setOpacity(self.opacity)
        self.raise_()
        self._fit_content()

    def _fit_content(self):
        signature = (self.view.height(), self.body_layout.count(), self.scale)
        if signature == self._fit_signature:
            return
        self._fit_signature = signature
        base = {"small": 8.0, "medium": 9.0, "large": 10.0}[self.scale]
        available = max(24, min(self.view.height() - 36, 220) - 12)
        labels = self.body.findChildren(QLabel)
        size = base
        while True:
            font = QFont()
            font.setPointSizeF(size)
            for label in labels:
                label.setFont(font)
            self.body_layout.activate()
            if self.body_layout.sizeHint().height() <= available or size <= 7.0:
                break
            size = max(7.0, size - 0.5)

    def set_presentation(self, anchor=None, scale=None, opacity=None):
        if anchor is not None: self.anchor = anchor
        if scale is not None and scale != self.scale:
            self.scale = scale
            self._apply_style()
        if opacity is not None: self.opacity = opacity
        self._position()

    def refresh(self, viewer):
        self._clear_layout(self.body_layout)
        self.metric_rows.clear()
        self.details_buttons.clear()
        self.reference_badges.clear()
        self.confidence_rows.clear()
        self.confidence_bars.clear()
        for config_index in self.configuration_indices:
            config = viewer.configurations[config_index]
            if viewer.mode == "overlay" and not viewer.overlay_view.configuration_visible[config_index]:
                continue
            self._add_result_group(viewer, config_index, config)
        if not self.body_layout.count():
            self.body_layout.addWidget(QLabel("No visible results"))
        self.body_layout.activate()
        self._fit_signature = None
        self._position()

    def update_values(self, viewer):
        frame = viewer.slider.value()
        frame_index = int(viewer.loaded_results[0].result.frame_indices[frame])
        for (config_id, metric_id, hand), label in self.metric_rows.items():
            if hand != "left":
                continue
            if metric_id == "per_joint_error":
                continue
            values = []
            for hand in ("left", "right"):
                pair = viewer._metric_cache.get(config_id, {}).get((frame_index, hand))
                value = pair.metrics.get(metric_id) if pair is not None else None
                values.append(f"{value:.1f}" if value is not None and np.isscalar(value) and np.isfinite(value) else "N/A")
            definition = next(item for item in viewer.metric_definitions if item.metric_id == metric_id)
            name = "Wrist Error" if metric_id == "wrist_error" else definition.name
            label.setText(f"{name} | L {values[0]} / R {values[1]} {definition.unit}")
        for key, label in self.confidence_rows.items():
            config_id, hand = key
            config = next(i for i, item in enumerate(viewer.configurations) if item["id"] == config_id)
            hand_index = 0 if hand == "left" else 1
            result = viewer.loaded_results[config].result
            value = result.confidence[frame, hand_index] if result.confidence is not None else np.nan
            display = float(value) if np.isfinite(value) else None
            if label.text() not in ("L detection", "R detection"):
                label.setText(f"{display:.2f}" if display is not None else "N/A")
            bar = self.confidence_bars.get(key)
            if bar is not None:
                bar.setValue(round(display * 100) if display is not None else 0)
                bar.setToolTip("YOLO hand detection score" if display is not None else "Detection score unavailable")

    @classmethod
    def _clear_layout(cls, layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                cls._clear_layout(item.layout())
                item.layout().deleteLater()

    def _add_result_group(self, viewer, index, config):
        header = QHBoxLayout()
        title = QLabel(config["name"])
        title.setStyleSheet("font-weight: 600; color: white;")
        title.setWordWrap(True)
        if viewer.single_mode:
            header.addWidget(title, 1)
        else:
            swatch = QLabel()
            swatch.setFixedSize(10, 10)
            color = tuple(round(value * 255) for value in viewer.configuration_colors[index])
            swatch.setStyleSheet(f"background-color: rgb{color}; border: 1px solid white;")
            header.addWidget(swatch)
            header.addWidget(title, 1)
            handle = _ResultDragHandle(viewer, config["id"], title)
            header.insertWidget(0, handle)
        if not viewer.single_mode and config["id"] == viewer.reference_id:
            badge = QLabel("REFERENCE")
            badge.setStyleSheet("font-weight: 700; color: white; border: 1px solid white; padding: 1px 4px;")
            header.addWidget(badge)
            self.reference_badges[config["id"]] = badge
        self.body_layout.addLayout(header)
        if not viewer.single_mode and config["id"] != viewer.reference_id:
            for definition in viewer.metric_definitions:
                if definition.metric_id not in viewer.selected_metric_ids:
                    continue
                if definition.metric_id == "per_joint_error":
                    row = QHBoxLayout()
                    label = QLabel("Per-Joint Error (21 joints)")
                    details = QPushButton("Details")
                    details.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
                    details.clicked.connect(lambda _=False, cid=config["id"]: viewer._show_joint_table(cid))
                    row.addWidget(label, 1)
                    row.addWidget(details)
                    self.body_layout.addLayout(row)
                    self.metric_rows[(config["id"], definition.metric_id, "left")] = label
                    self.details_buttons[(config["id"], definition.metric_id, "left")] = details
                    continue
                frame_index = int(viewer.loaded_results[0].result.frame_indices[viewer.slider.value()])
                values = []
                for hand_index, hand in enumerate(("left", "right")):
                    pair = viewer._metric_cache.get(config["id"], {}).get((frame_index, hand))
                    value = None if pair is None else pair.metrics.get(definition.metric_id)
                    text_value = f"{value:.1f}" if value is not None and np.isscalar(value) and np.isfinite(value) else "N/A"
                    values.append(text_value)
                display_name = "Wrist Error" if definition.metric_id == "wrist_error" else definition.name
                label = QLabel(f"{display_name} | L {values[0]} / R {values[1]} {definition.unit}")
                label.setStyleSheet("color: white;")
                self.body_layout.addWidget(label)
                for hand in ("left", "right"):
                    self.metric_rows[(config["id"], definition.metric_id, hand)] = label
        if viewer.confidence_numeric_checkbox.isChecked() or viewer.confidence_bar_checkbox.isChecked():
            result = viewer.loaded_results[index].result
            frame = viewer.slider.value()
            for hand_index, hand in enumerate(("left", "right")):
                confidence = result.confidence[frame, hand_index] if result.confidence is not None else np.nan
                display = float(confidence) if np.isfinite(confidence) else None
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(5)
                text = QLabel(("L" if hand_index == 0 else "R") + " detection")
                text.setStyleSheet("color: white;")
                row_layout.addWidget(text)
                value_label = text
                if viewer.confidence_numeric_checkbox.isChecked():
                    number = QLabel(f"{display:.2f}" if display is not None else "N/A")
                    number.setStyleSheet("color: white;")
                    row_layout.addWidget(number)
                    value_label = number
                if viewer.confidence_bar_checkbox.isChecked():
                    bar = QProgressBar()
                    bar.setRange(0, 100)
                    bar.setFixedWidth(85)
                    bar.setTextVisible(False)
                    bar.setValue(round(display * 100) if display is not None else 0)
                    bar.setToolTip("YOLO hand detection score" if display is not None else "Detection score unavailable")
                    row_layout.addWidget(bar)
                    self.confidence_bars[(config["id"], hand)] = bar
                row_layout.addStretch(1)
                self.body_layout.addWidget(row)
                self.confidence_rows[(config["id"], hand)] = value_label
        self._position()


class _ResultDragHandle(QToolButton):
    def __init__(self, viewer, configuration_id, parent=None):
        super().__init__(parent)
        self.viewer, self.configuration_id = viewer, configuration_id
        self.setText("⠿"); self.setToolTip("Drag to reorder result panels")
        self.setFixedWidth(20)
        self._press = None
        self._dragging = False
        self._drop_target = None
        if isinstance(parent, QWidget):
            parent.installEventFilter(self)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press = event.globalPosition().toPoint()
            self._dragging = False
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._press is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self._move_drag(event.globalPosition().toPoint())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._press is not None:
            self._finish_drag(event.globalPosition().toPoint())
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def eventFilter(self, watched, event):
        if watched is self.parentWidget():
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self._press = event.globalPosition().toPoint()
                self._dragging = False
            elif event.type() == QEvent.Type.MouseMove and self._press is not None:
                self._move_drag(event.globalPosition().toPoint())
            elif event.type() == QEvent.Type.MouseButtonRelease:
                self._finish_drag(event.globalPosition().toPoint())
        return super().eventFilter(watched, event)

    def _move_drag(self, global_position):
        if self._press is None:
            return
        if not self._dragging and (global_position - self._press).manhattanLength() < 8:
            return
        self._dragging = True
        self.viewer._panel_reorder_active = True
        target = next((view for view in self.viewer.split_views
                       if view.rect().contains(view.mapFromGlobal(global_position))), None)
        if target is self.viewer.split_views[next(i for i, c in enumerate(self.viewer.configurations)
                                                  if c["id"] == self.configuration_id)]:
            target = None
        if target is not self._drop_target:
            for panel in self.viewer.split_panels:
                panel._drop_active = panel.view is target
                panel._apply_style()
            for view in self.viewer.split_views:
                view.setStyleSheet("border: 2px solid palette(highlight);" if view is target else "")
            self._drop_target = target

    def _finish_drag(self, global_position):
        if self._press is None:
            return
        if (global_position - self._press).manhattanLength() >= 8:
            self._move_drag(global_position)
        if self._dragging:
            target = self._drop_target
            if target is not None:
                self.viewer.reorder_result(self.configuration_id, target.configuration_id)
        for panel in self.viewer.split_panels:
            panel._drop_active = False
            panel._apply_style()
        for view in self.viewer.split_views:
            view.setStyleSheet("")
        self.viewer._panel_reorder_active = False
        self._drop_target = None
        self._press = None
        self._dragging = False


class _JointAnnotationLayer(QWidget):
    def __init__(self, viewer, view, configuration_indices):
        super().__init__(view)
        self.viewer, self.view = viewer, view
        self.configuration_indices = configuration_indices
        self.hover_position = None
        self.setAcceptDrops(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.view.installEventFilter(self)
        self.hide()

    def eventFilter(self, watched, event):
        if watched is self.view and event.type() == QEvent.Type.Resize:
            self.setGeometry(self.view.rect())
        return super().eventFilter(watched, event)

    def refresh(self):
        self.setGeometry(self.view.rect())
        visible = self.viewer.annotation_mode != "off"
        self.setVisible(visible)
        self.raise_()
        self.update()

    def projected_annotations(self):
        viewer, view = self.viewer, self.view
        if viewer.annotation_mode == "off": return []
        frame = int(viewer.loaded_results[0].result.frame_indices[viewer.slider.value()])
        matrix = view.projectionMatrix((0, 0, max(1, view.width()), max(1, view.height())), view.getViewport()) * view.viewMatrix()
        labels = []
        for config_index in self.configuration_indices:
            config = viewer.configurations[config_index]
            if config["id"] == viewer.reference_id: continue
            if hasattr(view, "configuration_visible") and not view.configuration_visible[config_index]: continue
            result = viewer.loaded_results[config_index].result
            positions = np.flatnonzero(result.frame_indices == frame)
            if not positions.size: continue
            position = int(positions[0])
            for hand, hand_name in enumerate(("left", "right")):
                if (not result.detected[position, hand] or (hand == 0 and not view.show_left)
                        or (hand == 1 and not view.show_right)):
                    continue
                pair = viewer._metric_cache.get(config["id"], {}).get((frame, hand_name))
                if pair is None and not (viewer.single_mode and viewer.annotation_content == "name"):
                    continue
                occupied = []
                candidates = []
                errors = pair.joint_errors_mm if pair is not None else np.full(len(JOINT_NAMES), np.nan)
                for joint, (point, error) in enumerate(zip(result.keypoints_3d[position, hand], errors)):
                    if not np.isfinite(point).all(): continue
                    if viewer.annotation_content == "error" and not np.isfinite(error): continue
                    clip = matrix.map(QVector4D(float((point @ RAW_TO_VIEW.T)[0]),
                                                float((point @ RAW_TO_VIEW.T)[1]),
                                                float((point @ RAW_TO_VIEW.T)[2]), 1.0))
                    if clip.w() <= 0: continue
                    x, y = clip.x() / clip.w(), clip.y() / clip.w()
                    if abs(x) > 1 or abs(y) > 1: continue
                    px, base_y = (x + 1) * self.width() / 2 + 5, (1 - y) * self.height() / 2 - 4
                    candidates.append((joint, float(error) if np.isfinite(error) else np.nan, px, base_y))
                if viewer.annotation_mode == "key_joints":
                    candidates = [item for item in candidates if item[0] in (0, 4, 8, 12, 16, 20)]
                elif viewer.annotation_mode == "worst5":
                    candidates = sorted(candidates, key=lambda item: item[1], reverse=True)[:5]
                elif viewer.annotation_mode == "hovered":
                    if self.hover_position is None or not candidates:
                        candidates = []
                    else:
                        mx, my = self.hover_position.x(), self.hover_position.y()
                        candidates = [min(candidates, key=lambda item: (item[2] - mx) ** 2 + (item[3] - my) ** 2)]
                for joint, error, px, base_y in candidates:
                    label = f"{error:.1f}" if viewer.annotation_content == "error" else JOINT_NAMES[joint]
                    width = max(18, len(label) * 8)
                    offsets = [0] + [offset for step in range(1, 22) for offset in (-15 * step, 15 * step)]
                    py = next((base_y + offset for offset in offsets
                               if 14 <= base_y + offset <= self.height() - 4
                               and not any(px < old[0] + old[2] and px + width > old[0]
                                           and base_y + offset - 12 < old[1] + old[3]
                                           and base_y + offset + 2 > old[1] for old in occupied)), base_y)
                    occupied.append((px, py - 12, width, 14))
                    labels.append((config["id"], hand_name, joint, label, px, py))
        return labels

    def paintEvent(self, _event):
        viewer = self.viewer
        if viewer.annotation_mode == "off": return
        painter = QPainter(self)
        font = self.font(); font.setPointSize(max(9, font.pointSize())); font.setBold(True)
        painter.setFont(font)
        try:
            for config_id, hand_name, joint, label, x, y in self.projected_annotations():
                index = next(i for i, config in enumerate(viewer.configurations) if config["id"] == config_id)
                color = QColor(*(round(channel * 255) for channel in viewer.configuration_colors[index]))
                painter.setPen(QPen(QColor(0, 0, 0, 220), 3)); painter.drawText(int(x), int(y), label)
                painter.setPen(color); painter.drawText(int(x), int(y), label)
        finally:
            painter.end()
