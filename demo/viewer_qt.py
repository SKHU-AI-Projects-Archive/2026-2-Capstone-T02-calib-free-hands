"""results.npz 뷰어 — 마우스로 돌려 보는 3D 손 스켈레톤 (PyQt5 + pyqtgraph.opengl).

demo_hand_mano.py 가 만든 results.npz 와 원본 영상만 읽는다. torch·WiLoR 를 import 하지
않으므로 GPU 없는 PC 에서도 뜬다. 설치: pip install PyQt5 pyqtgraph  (PyOpenGL 필요)

  python demo/viewer_qt.py <out>/results.npz [--video <영상경로>]

화면:
  좌상  원본 프레임 + 2D 스켈레톤
  우상  3D 스켈레톤 (좌드래그 회전 · 휠 줌 · 우드래그/중드래그 이동, 바닥 격자 0.1 m)
  하단  손목 깊이(z) 시계열 + 현재 프레임 커서 (클릭으로 점프) / 슬라이더 · 재생 · 손 표시 토글

3D 표시 공간은 hand_topology.RAW_TO_VIEW (앱 스켈레톤 뷰·Isaac 과 같은 회전, Z-up).
초기 카메라는 촬영 카메라 쪽에서 손을 바라보는 방향(화면 오른쪽 = 이미지 오른쪽).
"""
import argparse
import sys
from pathlib import Path

# 콘솔 인코딩(cp949 등)에서 출력 문자 때문에 죽지 않게 — 인코딩 불가 문자는 '?' 로
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hand_topology import (  # noqa: E402
    HAND_BASE, HAND_BASE_COLORS, HAND_CONNECTIONS, HAND_LABELS, JOINTS_PER_HAND, LEFT, RIGHT,
    hex_to_bgr, hex_to_rgb, joint_color, raw_to_view,
)

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
except ImportError as exc:  # pragma: no cover
    sys.exit(f"GUI 의존성 없음 ({exc}).  설치:  pip install PyQt5 pyqtgraph")

WRIST = 0


# ═══════════════════════════════════════════════════════════════════════════
# 데이터
# ═══════════════════════════════════════════════════════════════════════════

class Result:
    """results.npz 한 벌. kp3d_abs (F,42,3) raw 카메라 공간, kp2d (F,42,2), detected (F,2)."""

    def __init__(self, npz_path: Path):
        d = np.load(npz_path, allow_pickle=False)
        self.path = npz_path
        self.kp3d = d["kp3d_abs"].astype(np.float64)
        self.kp2d = d["kp2d"].astype(np.float64)
        self.detected = d["detected"].astype(bool)
        self.frame_indices = d["frame_indices"].astype(int)
        self.fps = float(d["fps"]) if float(d["fps"]) > 0 else 30.0
        self.source_path = str(d["source_path"])
        self.scores = d["scores"] if "scores" in d else np.zeros_like(self.detected, dtype=float)
        self.bbox = d["bbox"] if "bbox" in d else None
        self.view = raw_to_view(self.kp3d)                     # (F,42,3) 표시 공간
        self.n = len(self.frame_indices)

    def wrist_z(self, side):
        z = self.kp3d[:, HAND_BASE[side] + WRIST, 2].copy()
        z[~self.detected[:, side]] = np.nan
        return z

    def hand_view(self, k, side):
        """k번째 레코드의 손 표시 좌표 (21,3) 또는 None(미검출)."""
        if not self.detected[k, side]:
            return None
        b = HAND_BASE[side]
        return self.view[k, b:b + JOINTS_PER_HAND]

    def cloud_center_radius(self):
        """검출 관절 구름의 중심·반경 (표시 공간). 깊이 튐이 프레이밍을 잡아먹지 않게 3~97 백분위."""
        pts = [self.hand_view(k, s) for k in range(self.n) for s in (LEFT, RIGHT) if self.detected[k, s]]
        if not pts:
            return np.zeros(3), 0.3
        pts = np.concatenate(pts)
        lo, hi = np.percentile(pts, 3, axis=0), np.percentile(pts, 97, axis=0)
        c = (lo + hi) / 2
        return c, max(float(np.linalg.norm(hi - lo) / 2), 0.1)


class FrameSource:
    """원본 영상에서 프레임 인덱스로 BGR 프레임을 꺼낸다 (순차 읽기는 seek 없이)."""

    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(str(video_path)) if video_path else None
        self.ok = bool(self.cap and self.cap.isOpened())
        self.next_idx = 0

    def get(self, frame_idx):
        if not self.ok:
            return None
        if frame_idx != self.next_idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        self.next_idx = frame_idx + 1
        return frame if ok else None


# ═══════════════════════════════════════════════════════════════════════════
# 그리기 헬퍼
# ═══════════════════════════════════════════════════════════════════════════

def draw_2d(img, res: Result, k):
    """프레임 위에 21점 스켈레톤(손가락 색) + 손 라벨."""
    for side in (LEFT, RIGHT):
        if not res.detected[k, side]:
            continue
        b = HAND_BASE[side]
        kp = res.kp2d[k, b:b + JOINTS_PER_HAND]
        for s, e in HAND_CONNECTIONS:
            cv2.line(img, tuple(kp[s].astype(int)), tuple(kp[e].astype(int)), hex_to_bgr(joint_color(e)), 2, cv2.LINE_AA)
        for i, (u, v) in enumerate(kp):
            cv2.circle(img, (int(u), int(v)), 5 if i == WRIST else 3, hex_to_bgr(joint_color(i)), -1, cv2.LINE_AA)
        u, v = kp[WRIST].astype(int)
        cv2.putText(img, HAND_LABELS[side], (u + 8, v + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    hex_to_bgr(HAND_BASE_COLORS[side]), 2, cv2.LINE_AA)
    return img


def rgba(hex_color, alpha=1.0):
    r, g, b = hex_to_rgb(hex_color)
    return (r / 255.0, g / 255.0, b / 255.0, alpha)


class HandGL:
    """한 손의 3D 아이템 묶음 (뼈 선 + 관절 점)."""

    def __init__(self, view: gl.GLViewWidget):
        bone_cols = np.array([rgba(joint_color(e)) for s, e in HAND_CONNECTIONS for _ in (s, e)], dtype=np.float32)
        joint_cols = np.array([rgba(joint_color(i)) for i in range(JOINTS_PER_HAND)], dtype=np.float32)
        joint_size = np.array([9 if i == WRIST else 6 for i in range(JOINTS_PER_HAND)], dtype=np.float32)
        self.lines = gl.GLLinePlotItem(pos=np.zeros((2, 3), np.float32), color=bone_cols, width=3,
                                       mode="lines", antialias=True, glOptions="translucent")
        self.pts = gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), color=joint_cols,
                                        size=joint_size, pxMode=True, glOptions="translucent")
        for it in (self.lines, self.pts):
            view.addItem(it)
        self.idx = np.array([i for pair in HAND_CONNECTIONS for i in pair])

    def set(self, pts21):
        if pts21 is None:
            self.lines.setVisible(False); self.pts.setVisible(False)
            return
        p = pts21.astype(np.float32)
        self.lines.setData(pos=p[self.idx]); self.pts.setData(pos=p)
        self.lines.setVisible(True); self.pts.setVisible(True)

    def hide(self):
        self.set(None)


# ═══════════════════════════════════════════════════════════════════════════
# 메인 창
# ═══════════════════════════════════════════════════════════════════════════

class Viewer(QtWidgets.QMainWindow):
    def __init__(self, res: Result, video_path):
        super().__init__()
        self.res = res
        self.frames = FrameSource(video_path)
        self.k = 0
        self.show_side = {LEFT: True, RIGHT: True}
        self.setWindowTitle(f"hand demo viewer — {res.path.parent.name}")
        self.resize(1500, 950)
        self._build_ui()
        self._init_3d()
        self._init_plot()
        self.set_frame(0)

    # ── UI ──
    def _build_ui(self):
        pg.setConfigOptions(antialias=True)
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        top = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.img_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.img_label.setStyleSheet("background:#000;color:#aaa")
        self.img_label.setMinimumSize(320, 240)
        self.img_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.gl = gl.GLViewWidget()
        self.gl.setBackgroundColor("#101418")
        top.addWidget(self.img_label); top.addWidget(self.gl)
        top.setSizes([700, 800])

        self.plot = pg.PlotWidget(title="wrist depth z [m]  (raw)")
        self.plot.setMaximumHeight(220)
        self.plot.setLabel("bottom", "frame")
        self.plot.showGrid(x=True, y=True, alpha=0.25)

        ctrl = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QPushButton("▶ 재생"); self.btn_play.setCheckable(True)
        self.btn_play.toggled.connect(self._toggle_play)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, self.res.n - 1)
        self.slider.valueChanged.connect(self.set_frame)
        self.chk = {}
        for side in (LEFT, RIGHT):
            cb = QtWidgets.QCheckBox(HAND_LABELS[side]); cb.setChecked(True)
            cb.setStyleSheet(f"color:{HAND_BASE_COLORS[side]}; font-weight:bold")
            cb.toggled.connect(lambda on, s=side: self._toggle_side(s, on))
            self.chk[side] = cb
        self.btn_reset = QtWidgets.QPushButton("카메라 리셋"); self.btn_reset.clicked.connect(self._reset_camera)
        self.info = QtWidgets.QLabel(); self.info.setMinimumWidth(360)
        ctrl.addWidget(self.btn_play); ctrl.addWidget(self.slider, 1)
        ctrl.addWidget(self.chk[LEFT]); ctrl.addWidget(self.chk[RIGHT]); ctrl.addWidget(self.btn_reset)
        ctrl.addWidget(self.info)

        root.addWidget(top, 1); root.addWidget(self.plot); root.addLayout(ctrl)

        step = int(np.median(np.diff(self.res.frame_indices))) if self.res.n > 1 else 1
        self.timer = QtCore.QTimer(self); self.timer.setInterval(int(1000.0 / self.res.fps * max(step, 1)))
        self.timer.timeout.connect(self._tick)

    def _init_3d(self):
        center, radius = self.res.cloud_center_radius()
        self.center, self.radius = center, radius
        floor_z = float(min(self.res.view[self.res.detected.any(1)][..., 2].min(), center[2] - radius)) - 0.02 \
            if self.res.detected.any() else -0.3
        grid = gl.GLGridItem(); grid.setSize(2, 2); grid.setSpacing(0.1, 0.1)
        grid.translate(center[0], center[1], floor_z); grid.setColor((255, 255, 255, 40))
        self.gl.addItem(grid)
        axis = gl.GLAxisItem(); axis.setSize(0.1, 0.1, 0.1)
        axis.translate(center[0] - radius, center[1] - radius, floor_z)   # X 빨강 · Y(깊이 -) 노랑 · Z(위) 파랑
        self.gl.addItem(axis)
        cam = gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), color=(1, 1, 1, 0.9), size=10, pxMode=True)
        self.gl.addItem(cam)   # 촬영 카메라 = 원점
        self.hands = {s: HandGL(self.gl) for s in (LEFT, RIGHT)}
        self._reset_camera()

    def _reset_camera(self):
        c = self.center
        self.gl.opts["center"] = QtGui.QVector3D(float(c[0]), float(c[1]), float(c[2]))
        # azimuth 90 = +y 쪽(촬영 카메라 쪽)에서 -y 방향의 손을 바라봄 → 화면 오른쪽 = 이미지 오른쪽
        self.gl.setCameraPosition(distance=self.radius * 2.6, elevation=18, azimuth=90)

    def _init_plot(self):
        self.plot.addLegend(offset=(10, 5))
        x = self.res.frame_indices
        for side in (LEFT, RIGHT):
            self.plot.plot(x, self.res.wrist_z(side), pen=pg.mkPen(HAND_BASE_COLORS[side], width=1.6),
                           connect="finite", name=HAND_LABELS[side])
        self.cursor = pg.InfiniteLine(pos=x[0], angle=90, movable=True, pen=pg.mkPen("#FFFFFF", width=1.5))
        self.cursor.sigDragged.connect(lambda line: self._jump_to_frame_index(line.value()))
        self.plot.addItem(self.cursor)
        self.plot.scene().sigMouseClicked.connect(self._plot_clicked)

    # ── 상호작용 ──
    def _plot_clicked(self, ev):
        p = self.plot.plotItem.vb.mapSceneToView(ev.scenePos())
        self._jump_to_frame_index(p.x())

    def _jump_to_frame_index(self, frame_index):
        k = int(np.argmin(np.abs(self.res.frame_indices - frame_index)))
        self.slider.setValue(k)

    def _toggle_play(self, on):
        self.btn_play.setText("❚❚ 일시정지" if on else "▶ 재생")
        (self.timer.start if on else self.timer.stop)()

    def _tick(self):
        self.slider.setValue((self.k + 1) % self.res.n)

    def _toggle_side(self, side, on):
        self.show_side[side] = on
        self.set_frame(self.k)

    def keyPressEvent(self, ev):
        if ev.key() == QtCore.Qt.Key_Space:
            self.btn_play.toggle()
        elif ev.key() == QtCore.Qt.Key_Left:
            self.slider.setValue(max(self.k - 1, 0))
        elif ev.key() == QtCore.Qt.Key_Right:
            self.slider.setValue(min(self.k + 1, self.res.n - 1))
        else:
            super().keyPressEvent(ev)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._update_2d()

    # ── 프레임 갱신 ──
    def set_frame(self, k):
        self.k = int(k)
        if self.slider.value() != self.k:
            self.slider.blockSignals(True); self.slider.setValue(self.k); self.slider.blockSignals(False)
        fi = int(self.res.frame_indices[self.k])
        self._frame_bgr = self.frames.get(fi)
        self._update_2d()
        for side in (LEFT, RIGHT):
            self.hands[side].set(self.res.hand_view(self.k, side) if self.show_side[side] else None)
        self.cursor.blockSignals(True); self.cursor.setValue(fi); self.cursor.blockSignals(False)
        z = [self.res.kp3d[self.k, HAND_BASE[s] + WRIST, 2] if self.res.detected[self.k, s] else np.nan for s in (LEFT, RIGHT)]
        self.info.setText(f"frame {fi}  ({self.k + 1}/{self.res.n})   wrist z  L {z[0]:.3f} m   R {z[1]:.3f} m")

    def _update_2d(self):
        if not hasattr(self, "_frame_bgr"):
            return
        if self._frame_bgr is None:
            self.img_label.setPixmap(QtGui.QPixmap())
            self.img_label.setText("원본 영상을 열 수 없습니다\n(--video 로 경로 지정)")
            return
        img = draw_2d(self._frame_bgr.copy(), self.res, self.k)
        h, w = img.shape[:2]
        rgb = np.ascontiguousarray(img[:, :, ::-1])
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.img_label.size(), QtCore.Qt.KeepAspectRatio,
                                                    QtCore.Qt.SmoothTransformation)
        self.img_label.setPixmap(pix)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", help="demo_hand_mano.py 출력 results.npz")
    ap.add_argument("--video", help="원본 영상 경로 (npz 에 저장된 경로가 안 맞을 때)")
    ap.add_argument("--frame", type=int, default=0, help="시작 레코드 번호")
    ap.add_argument("--screenshot", help="창을 이 PNG 로 저장하고 바로 종료 (동작 확인용)")
    args = ap.parse_args()

    res = Result(Path(args.results).resolve())
    video = args.video or res.source_path
    if not Path(video).exists():
        print(f"[warn] 원본 영상 없음: {video} - 2D 패널은 비워 둡니다 (--video 로 지정 가능)")
        video = None
    if "torch" in sys.modules:
        print("[warn] torch 가 import 되어 있습니다 - 뷰어는 torch 없이 동작해야 합니다")

    app = QtWidgets.QApplication(sys.argv)
    w = Viewer(res, video)
    w.show()
    if args.frame:
        w.set_frame(min(max(args.frame, 0), res.n - 1))
    if args.screenshot:
        def _shot():
            w.grab().save(args.screenshot)
            print(f"[screenshot] {args.screenshot}")
            app.quit()
        QtCore.QTimer.singleShot(1500, _shot)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
