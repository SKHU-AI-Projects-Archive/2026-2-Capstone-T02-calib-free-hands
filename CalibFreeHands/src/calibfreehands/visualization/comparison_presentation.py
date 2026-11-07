from __future__ import annotations

import numpy as np
from PySide6.QtCore import QEvent, Qt, QPropertyAnimation, QEasingCurve, QPoint, QTimer, QRectF, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPalette, QPen, QVector4D
from PySide6.QtWidgets import (QApplication, QCheckBox, QFrame, QGraphicsOpacityEffect, QHBoxLayout,
                               QLabel, QProgressBar, QPushButton, QSizePolicy, QSplitter, QSplitterHandle,
                               QToolButton, QVBoxLayout, QWidget)

from calibfreehands.joint_schema import JOINT_NAMES
from .topology import RAW_TO_VIEW


class _PanelSurface(QWidget):
    """Stable logical panel surface around an existing viewer widget."""

    def __init__(self, panel_id, content):
        super().__init__()
        self.panel_id = panel_id
        self.content = content
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(content, 1)


class _SwapButton(QToolButton):
    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("overlay-panel-swap")
        self.setText("")
        self.setToolTip("Swap Source Media and Overlay 3D")
        self.setAccessibleName("Swap Source Media and Overlay 3D")
        self.setAutoRaise(True)
        self.setFixedSize(28, 28)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.underMouse() or self.isDown():
            fill = self.palette().color(QPalette.ColorRole.Button)
            fill.setAlpha(235)
            painter.setPen(QPen(self.palette().color(QPalette.ColorRole.Mid), 1))
            painter.setBrush(fill)
            painter.drawEllipse(self.rect().adjusted(2, 2, -2, -2))
        color = self.palette().color(QPalette.ColorRole.ButtonText)
        pen = QPen(color, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(7, 9, 21, 9)
        painter.drawLine(21, 9, 17, 5)
        painter.drawLine(21, 9, 17, 13)
        painter.drawLine(21, 19, 7, 19)
        painter.drawLine(7, 19, 11, 15)
        painter.drawLine(7, 19, 11, 23)


class _SwapSplitter(QSplitter):
    swapRequested = Signal()

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.setHandleWidth(4)
        self.swap_button = _SwapButton(parent if parent is not None else self)
        self.swap_button.clicked.connect(self.request_swap)
        self.splitterMoved.connect(lambda *_: self._position_swap_button())

    def createHandle(self):
        return QSplitterHandle(self.orientation(), self)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_swap_button()

    def showEvent(self, event):
        super().showEvent(event)
        self._position_swap_button()

    def _position_swap_button(self):
        handle = self.handle(1)
        if handle is None or self.count() < 2:
            return
        center = self.mapTo(self.swap_button.parentWidget(), handle.geometry().center())
        self.swap_button.move(center.x() - self.swap_button.width() // 2,
                              center.y() - self.swap_button.height() // 2)
        self.swap_button.raise_()

    def request_swap(self):
        self.swapRequested.emit()


class _SplitPanelReorderController(QWidget):
    """Intercept only Shift-left drags that begin on Split panel content."""

    def __init__(self, viewer):
        super().__init__(viewer)
        self.viewer = viewer
        self._press_position = None
        self._source_panel_id = None
        self.target_panel_id = None
        self._active = False
        self._application_filter = False
        self._watched_widgets = {
            viewer.split_source_overlay: "source-media",
            **{view: f"result:{config['id']}"
               for view, config in zip(viewer.split_views, viewer.configurations)},
        }
        for widget in self._watched_widgets:
            widget.installEventFilter(self)

    def eventFilter(self, watched, event):
        if (not self._active and watched in self._watched_widgets
                and event.type() == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton
                and event.modifiers() & Qt.KeyboardModifier.ShiftModifier
                and self.viewer.mode == "split"):
            self._begin(watched, event.globalPosition().toPoint())
            event.accept()
            return True
        if self._active and self._application_filter:
            if event.type() == QEvent.Type.MouseMove:
                self._update_target(event.globalPosition().toPoint())
                event.accept()
                return True
            if event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                self._update_target(event.globalPosition().toPoint())
                source, target = self._source_panel_id, self.target_panel_id
                self.cancel()
                if target is not None:
                    self.viewer.move_split_panel_to_slot(source, target)
                event.accept()
                return True
            if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
                self.cancel()
                event.accept()
                return True
            if watched is self.viewer and event.type() in (QEvent.Type.Hide, QEvent.Type.Close):
                self.cancel()
        return super().eventFilter(watched, event)

    def _begin(self, widget, global_position):
        self._source_panel_id = self._watched_widgets[widget]
        self._press_position = global_position
        self._active = True
        self.viewer._panel_reorder_active = True
        QApplication.instance().installEventFilter(self)
        self._application_filter = True
        self.viewer._show_split_drag_feedback(self._source_panel_id, None)
        widget.grabMouse()

    def _update_target(self, global_position):
        if self._press_position is None:
            return
        if (global_position - self._press_position).manhattanLength() < 8:
            return
        self.target_panel_id = self.viewer._split_panel_at(global_position)
        self.viewer._show_split_drag_feedback(self._source_panel_id, self.target_panel_id)

    def cancel(self):
        if self._application_filter:
            QApplication.instance().removeEventFilter(self)
            self._application_filter = False
        grabber = QWidget.mouseGrabber()
        if grabber in self._watched_widgets:
            grabber.releaseMouse()
        self._active = False
        self._press_position = None
        self._source_panel_id = None
        self.target_panel_id = None
        self.viewer._panel_reorder_active = False
        self.viewer._show_split_drag_feedback(None, None)


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
        self._fit_generation = 0
        self._fit_timer = QTimer(self)
        self._fit_timer.setSingleShot(True)
        self._fit_timer.timeout.connect(lambda: self._apply_stable_update(self._fit_generation))
        self.effective_point_size = 0.0
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
        self.insertion_marker = QFrame(self.body)
        self.insertion_marker.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.insertion_marker.setStyleSheet("QFrame { background: palette(highlight); border: none; }")
        self.insertion_marker.hide()
        card_layout.addWidget(self.body)
        self.anchor = "top-left"
        self.scale = "medium"
        self.opacity = 1.0
        self._drag_offset = None
        self.snap_animation = QPropertyAnimation(self.card, b"pos", self)
        self.snap_animation.setDuration(150)
        self.snap_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.card.setCursor(Qt.CursorShape.OpenHandCursor)
        self.card.installEventFilter(self)
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
        self.group_headers = {}

    def eventFilter(self, watched, event):
        if watched is self.view and event.type() == QEvent.Type.Resize:
            self.snap_animation.stop()
            self._position()
        elif watched is self.card or (isinstance(watched, QWidget) and self.card.isAncestorOf(watched)):
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                if isinstance(watched, (QPushButton, QToolButton)):
                    return super().eventFilter(watched, event)
                self.snap_animation.stop()
                self._drag_offset = self.mapFromGlobal(event.globalPosition().toPoint()) - self.card.pos()
                self.card.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return True
            if event.type() == QEvent.Type.MouseMove and self._drag_offset is not None:
                next_position = self.mapFromGlobal(event.globalPosition().toPoint()) - self._drag_offset
                bounds = self.rect().adjusted(0, 0, -self.card.width(), -self.card.height())
                self.card.move(max(bounds.left(), min(bounds.right(), next_position.x())),
                               max(bounds.top(), min(bounds.bottom(), next_position.y())))
                event.accept()
                return True
            if event.type() == QEvent.Type.MouseButtonRelease and self._drag_offset is not None:
                self._drag_offset = None
                self.card.setCursor(Qt.CursorShape.OpenHandCursor)
                self._finish_drag(event.globalPosition().toPoint())
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def _finish_drag(self, _global_position=None):
        center = self.card.geometry().center()
        margin = 12
        anchors = {
            "top-left": QPoint(margin, margin),
            "top-right": QPoint(max(margin, self.width() - self.card.width() - margin), margin),
            "bottom-left": QPoint(margin, max(margin, self.height() - self.card.height() - margin)),
            "bottom-right": QPoint(max(margin, self.width() - self.card.width() - margin),
                                    max(margin, self.height() - self.card.height() - margin)),
        }
        self.anchor = min(anchors, key=lambda name: (anchors[name].x() + self.card.width() / 2 - center.x()) ** 2
                          + (anchors[name].y() + self.card.height() / 2 - center.y()) ** 2)
        self.snap_animation.stop()
        self.snap_animation.setStartValue(self.card.pos())
        self.snap_animation.setEndValue(anchors[self.anchor])
        self.snap_animation.start()

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
        self._schedule_stable_update()

    def set_background(self, style):
        if style not in ("none", "subtle"):
            raise ValueError(style)
        if style != self.background:
            self.background = style
            self._apply_style()

    def _position(self, *, schedule_fit=True):
        self.setGeometry(self.view.rect())
        for child in self.card.findChildren(QWidget):
            child.installEventFilter(self)
        self.body_layout.activate()
        self.card.layout().activate()
        content_width = self.body_layout.sizeHint().width() + 20
        content_height = self.body_layout.sizeHint().height() + 12
        width = min(330, max(200, self.view.width() - 24), max(240, content_width + 60))
        height = min(max(64, content_height), max(64, self.view.height() - 24))
        self.card.setFixedSize(width, height)
        x = 12 if "left" in self.anchor else max(0, self.width() - width - 12)
        y = 12 if "top" in self.anchor else max(0, self.height() - height - 12)
        if self._drag_offset is None and not self.snap_animation.state():
            self.card.move(x, y)
        self.opacity_effect.setOpacity(self.opacity)
        self.raise_()
        if schedule_fit:
            self._schedule_stable_update()

    def _schedule_stable_update(self):
        self._fit_generation += 1
        self.card.setUpdatesEnabled(False)
        for widget in (self.card, self.body, *self.body.findChildren(QWidget)):
            widget_layout = widget.layout()
            if widget_layout is not None:
                widget_layout.invalidate()
                widget.updateGeometry()
        self._fit_timer.start(0)

    def _apply_stable_update(self, generation):
        if generation != self._fit_generation:
            return
        self.body_layout.invalidate()
        self.card.layout().invalidate()
        self.body_layout.activate()
        self.card.layout().activate()
        self._fit_content()
        self.body_layout.invalidate()
        self.card.layout().invalidate()
        self.body_layout.activate()
        self.card.layout().activate()
        self._position(schedule_fit=False)
        self.card.setUpdatesEnabled(True)
        self.card.update()

    def _fit_content(self):
        viewport = self.view.contentsRect()
        labels = self.body.findChildren(QLabel)
        content = tuple((label.text(), label.isVisible(), label.wordWrap()) for label in labels)
        signature = (viewport.size(), self.body_layout.count(), content, self.scale)
        if signature == self._fit_signature:
            return
        self._fit_signature = signature
        base = {"small": 8.0, "medium": 9.0, "large": 10.0}[self.scale]
        available = max(24, min(viewport.height() - 36, 220) - 12)
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
        self.effective_point_size = size

    @property
    def requested_scale(self):
        return self.scale

    def set_presentation(self, anchor=None, scale=None, opacity=None):
        if anchor is not None: self.anchor = anchor
        if scale is not None and scale != self.scale:
            self.scale = scale
            self._apply_style()
        if opacity is not None: self.opacity = opacity
        self._position()

    def set_insertion_marker(self, visible_ids, insertion):
        visible_ids = [config_id for config_id in visible_ids if config_id in self.group_headers]
        if not visible_ids:
            self.clear_insertion_marker()
            return
        starts = [self.group_headers[config_id].geometry().top() for config_id in visible_ids]
        insertion = max(0, min(int(insertion), len(visible_ids)))
        if insertion == 0:
            y = starts[0]
        elif insertion == len(visible_ids):
            y = max(0, self.body.height() - 3)
        else:
            y = round((starts[insertion - 1] + starts[insertion]) / 2)
        self.insertion_marker.setGeometry(2, y, max(0, self.body.width() - 4), 3)
        self.insertion_marker.show()
        self.insertion_marker.raise_()

    def clear_insertion_marker(self):
        self.insertion_marker.hide()

    def refresh(self, viewer):
        self.clear_insertion_marker()
        self.card.setUpdatesEnabled(False)
        self._clear_layout(self.body_layout)
        self.metric_rows.clear()
        self.details_buttons.clear()
        self.reference_badges.clear()
        self.confidence_rows.clear()
        self.confidence_bars.clear()
        self.group_headers.clear()
        ordered_indices = [viewer._configuration_index(config_id)
                           for config_id in viewer.result_order_ids
                           if config_id in {viewer.configurations[index]["id"]
                                            for index in self.configuration_indices}]
        for config_index in ordered_indices:
            config = viewer.configurations[config_index]
            if viewer.mode == "overlay" and not viewer.overlay_view.configuration_visible[config_index]:
                continue
            self._add_result_group(viewer, config_index, config)
        if not self.body_layout.count():
            self.body_layout.addWidget(QLabel("No visible results"))
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
            if "confidence" not in label.text():
                label.setText(f"{display:.2f}" if display is not None else "N/A")
            bar = self.confidence_bars.get(key)
            if bar is not None:
                bar.setValue(round(display * 100) if display is not None else 0)
                bar.setToolTip("YOLO hand detector bounding-box confidence" if display is not None else "Hand detector confidence unavailable")

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
        self.group_headers[config["id"]] = header
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
            badge = QLabel("GT")
            badge.setStyleSheet("font-weight: 700; color: #111111; background-color: #33FF33; border: 1px solid #249924; padding: 1px 4px;")
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
                text = QLabel(("L" if hand_index == 0 else "R") + " hand detection confidence")
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
                    bar.setToolTip("YOLO hand detector bounding-box confidence" if display is not None else "Hand detector confidence unavailable")
                    row_layout.addWidget(bar)
                    self.confidence_bars[(config["id"], hand)] = bar
                row_layout.addStretch(1)
                self.body_layout.addWidget(row)
                self.confidence_rows[(config["id"], hand)] = value_label


class _ResultDragHandle(QToolButton):
    def __init__(self, viewer, configuration_id, parent=None):
        super().__init__(parent)
        self.viewer, self.configuration_id = viewer, configuration_id
        self.setText("\u2195"); self.setToolTip("Drag to reorder result panels")
        self.setFixedWidth(20)
        self._press = None
        self._dragging = False
        self._drop_target = None
        self._filter_installed = False

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

    def _move_drag(self, global_position):
        if self._press is None:
            return
        if not self._dragging and (global_position - self._press).manhattanLength() < 8:
            return
        if not self._dragging:
            QApplication.instance().installEventFilter(self)
            self._filter_installed = True
        self._dragging = True
        self.viewer._panel_reorder_active = True
        panel = (self.viewer.overlay_panel if self.viewer.mode == "overlay" else
                 self.viewer.split_panels[self.viewer._configuration_index(self.configuration_id)])
        body_y = panel.body.mapFromGlobal(global_position).y()
        if not panel.body.rect().contains(panel.body.mapFromGlobal(global_position)):
            self._drop_target = None
            panel.clear_insertion_marker()
            return
        visible_ids = [config_id for config_id in self.viewer.result_order_ids
                       if config_id in panel.group_headers and panel.group_headers[config_id].geometry().isValid()]
        insertion = len(visible_ids)
        starts = [panel.group_headers[config_id].geometry().top() for config_id in visible_ids]
        for position in range(len(visible_ids) - 1):
            boundary = (starts[position] + starts[position + 1]) / 2
            if body_y < boundary:
                insertion = position
                break
        self._drop_target = (visible_ids, insertion) if visible_ids else None
        panel.set_insertion_marker(visible_ids, insertion)

    def _finish_drag(self, global_position):
        if self._press is None:
            return
        if (global_position - self._press).manhattanLength() >= 8:
            self._move_drag(global_position)
        if self._dragging:
            target = self._drop_target
            if target is not None:
                visible_ids, insertion = target
                self.viewer.reorder_result_at(self.configuration_id, visible_ids, insertion)
        self.viewer._panel_reorder_active = False
        self._drop_target = None
        self._press = None
        self._dragging = False
        self._remove_drag_filter()
        self.viewer.overlay_panel.clear_insertion_marker()
        for panel in self.viewer.split_panels:
            panel.clear_insertion_marker()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self._press is not None:
            self._drop_target = None
            self._panel_drag_cancelled()
            event.accept()
            return
        super().keyPressEvent(event)

    def _panel_drag_cancelled(self):
        self.viewer._panel_reorder_active = False
        self._drop_target = None
        self._press = None
        self._dragging = False
        self._remove_drag_filter()
        self.viewer.overlay_panel.clear_insertion_marker()
        for panel in self.viewer.split_panels:
            panel.clear_insertion_marker()

    def eventFilter(self, watched, event):
        if (self._filter_installed and event.type() == QEvent.Type.KeyPress
                and event.key() == Qt.Key.Key_Escape):
            self._panel_drag_cancelled()
            return True
        return super().eventFilter(watched, event)

    def _remove_drag_filter(self):
        if self._filter_installed:
            QApplication.instance().removeEventFilter(self)
            self._filter_installed = False


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
        occupied = []
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
                    width = max(18, QFontMetrics(self.font()).horizontalAdvance(label))
                    horizontal_direction = 1 if px < self.width() / 2 else -1
                    proposed_x = px + horizontal_direction * config_index * 10 + (4 if hand else 0)
                    label_x = min(max(4, proposed_x), max(4, self.width() - width - 4))
                    offsets = [0] + [offset for step in range(1, 22) for offset in (-15 * step, 15 * step)]
                    baseline = min(max(14, base_y), max(14, self.height() - 4))
                    config_offset = config_index * 4
                    if baseline + config_offset > self.height() - 4:
                        config_offset = -config_offset
                    baseline = min(max(14, baseline + config_offset), max(14, self.height() - 4))
                    if viewer.annotation_mode in {"hovered", "key_joints", "worst5"}:
                        label_y = next((baseline + offset for offset in offsets
                                        if 14 <= baseline + offset <= self.height() - 4
                                        and not any(label_x < old[0] + old[2] and label_x + width > old[0]
                                                    and baseline + offset - 12 < old[1] + old[3]
                                                    and baseline + offset + 2 > old[1] for old in occupied)),
                                       baseline)
                        occupied.append((label_x, label_y - 12, width, 14))
                    else:
                        label_y = baseline
                    labels.append((config["id"], hand_name, joint, label, label_x, label_y))
        return labels

    def paintEvent(self, _event):
        viewer = self.viewer
        if viewer.annotation_mode == "off": return
        painter = QPainter(self)
        font = self.font(); font.setPointSize(max(9, font.pointSize())); font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        try:
            for config_id, hand_name, joint, label, x, y in self.projected_annotations():
                index = next(i for i, config in enumerate(viewer.configurations) if config["id"] == config_id)
                color = QColor(*(round(channel * 255) for channel in viewer.configuration_colors[index]))
                label_rect = QRectF(x - 3, y - metrics.ascent() - 2,
                                    metrics.horizontalAdvance(label) + 6, metrics.height() + 4)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(16, 20, 28, 205))
                painter.drawRoundedRect(label_rect, 3, 3)
                painter.setPen(QPen(QColor(0, 0, 0, 220), 3)); painter.drawText(int(x), int(y), label)
                painter.setPen(color); painter.drawText(int(x), int(y), label)
        finally:
            painter.end()
