"""손 검출 → MANO 피팅 → 3D 시각화 — 단계가 드러나는 최소 데모 (AnyHand-WiLoR).

파이프라인 4단계를 함수 하나씩으로 분리했다. 한 단계만 바꿔 끼우고(예: 검출기 교체,
깊이 재척도) 결과가 어떻게 달라지는지 눈으로 비교하는 용도.

  stage1_detect          이미지 → 손 bbox (YOLO)                         [rgb_predictor.detect_hands]
  stage2_fit_mano        bbox → MANO 파라미터·메시·21관절·cam_t (WiLoR)   [rgb_predictor.reconstruct]
  stage3_to_camera_space 손목 상대 관절 + cam_t → 절대 카메라 공간(미터)
  stage4_visualize       원본+2D 오버레이 / 3D 스켈레톤 PNG (matplotlib)

실행:
  python demo/demo_hand_mano.py --input <영상.mp4 | 이미지.jpg> --out <출력폴더>
      [--max_frames N] [--stride k] [--checkpoint anyhand|wilor]
      [--det_conf 0.3] [--rescale_factor 2.0]
      [--save_mesh] [--make_video] [--no_png] [--gui]

출력(--out 아래):
  results.npz          프레임별 42점 절대 3D(raw, 스무딩·정규화 없음)·2D·cam_t·MANO 파라미터
  frame_000000.png …   좌: 원본+2D 스켈레톤 / 우: 3D 스켈레톤 (--no_png 면 생략)
  wrist_depth.png      손목 깊이(z) 시계열 + 지터 수치 (영상 입력일 때)
  meshes/*.obj         MANO 메시 (--save_mesh)
  demo.mp4             PNG 이어붙인 영상 (--make_video)

좌표계: X=이미지 오른쪽, Y=이미지 아래, Z=카메라 전방(깊이), 원점=카메라. 미터 단위.
3D 그림·Qt 뷰어는 hand_topology.RAW_TO_VIEW 로 Z-up 표시 공간으로 돌려 그린다(앱과 동일 방향).

GUI 로 돌려 보려면 viewer_qt.py (PyQt5+pyqtgraph, torch 불필요) 에 results.npz 를 넘긴다.
이 파일은 Qt 를 import 하지 않는다 — --gui 는 완료 후 뷰어를 별도 프로세스로 띄울 뿐이다.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# 콘솔 인코딩(cp949 등)에서 출력 문자 때문에 죽지 않게 — 인코딩 불가 문자는 '?' 로
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

# ── 실행 환경 가드 (extract_hand.py 와 동일) ─────────────────────────────────
_DEMO_DIR = Path(__file__).resolve().parent
_SERVICE_ROOT = _DEMO_DIR.parent
# WiLoR load_wilor 가 MANO 경로를 CWD 상대('./mano_data/')로 하드코딩 → 서비스 루트로 고정
os.chdir(_SERVICE_ROOT)
# WiLoR renderer 모듈이 PYOPENGL_PLATFORM=egl 강제(리눅스 헤드리스) → Windows 는 미리 선점
if os.name == "nt":
    os.environ.setdefault("PYOPENGL_PLATFORM", "nt")
sys.path.insert(0, str(_SERVICE_ROOT))
sys.path.insert(0, str(_DEMO_DIR))

import cv2
import numpy as np

from hand_topology import (
    HAND_BASE, HAND_BASE_COLORS, HAND_CONNECTIONS, HAND_LABELS, JOINTS_PER_HAND, LEFT, RIGHT,
    hex_to_bgr, joint_color, raw_to_view,
)

# 체크포인트 ↔ config 짝. ⚠️ 반드시 짝으로 — 원본 WiLoR 는 FOCAL_LENGTH=5000, AnyHand 재학습본은
# 1000 이라 짝이 어긋나면 cam_t.z(깊이)가 5배 틀어진다.
CHECKPOINTS = {
    "anyhand": ("anyhand_wilor.ckpt", "model_config_wilor.yaml"),
    "wilor": ("wilor_final.ckpt", "model_config_wilor_original.yaml"),
}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}
WRIST = 0


# ═══════════════════════════════════════════════════════════════════════════
# 단계 1~4
# ═══════════════════════════════════════════════════════════════════════════

def load_predictor(checkpoint: str, det_conf: float, det_iou: float, rescale_factor: float):
    """모델 로드(검출기 + WiLoR). 4GB VRAM 내외, 최초 1회 수십 초."""
    from rgb_predictor import AnyHandPredictor
    ckpt_file, cfg_file = CHECKPOINTS[checkpoint]
    return AnyHandPredictor(
        backend="wilor",
        wilor_ckpt=str(_SERVICE_ROOT / "models" / ckpt_file),
        wilor_cfg=str(_SERVICE_ROOT / "models" / cfg_file),
        det_conf=det_conf,
        det_iou=det_iou,
        rescale_factor=rescale_factor,
    )


def stage1_detect(predictor, img_bgr):
    """1단계 — 손 검출. 반환: boxes (N,4) [x1,y1,x2,y2] 픽셀, is_right (N,) bool, scores (N,).

    검출기를 바꾸려면 이 함수 본문만 갈아끼우면 된다 (같은 형식만 지키면 2단계는 그대로).
    """
    return predictor.detect_hands(img_bgr)


def stage2_fit_mano(predictor, img_bgr, boxes, is_right, scores):
    """2단계 — bbox 마다 크롭 → WiLoR → MANO. 반환: HandPrediction 리스트 (boxes 순서).

    HandPrediction 필드: mano_pose (48,) axis-angle, mano_shape (10,), vertices (778,3),
    keypoints_3d (21,3) 손목 상대·미터, keypoints_2d (21,2) 원본 픽셀, cam_t (3,), is_right, score.
    """
    return predictor.reconstruct(img_bgr, boxes, is_right, scores)


def stage3_to_camera_space(hand):
    """3단계 — 손목 상대 좌표 + cam_t = 절대 카메라 공간 (미터). 반환: (kp3d_abs (21,3), verts_abs (778,3)).

    단안 깊이 지터·절대 스케일은 거의 전부 cam_t (특히 cam_t[2]) 에서 나온다.
    RGB-D 등 다른 깊이 근거로 재척도하려면 여기서 cam_t 를 바꿔 끼우면 된다.
    """
    cam_t = hand.cam_t[np.newaxis, :]
    return hand.keypoints_3d + cam_t, hand.vertices + cam_t


def stage4_visualize(frame_bgr, rec, view_lims, out_png):
    """4단계 — 좌: 원본 + bbox + 2D 스켈레톤 / 우: 3D 스켈레톤(표시 공간, 고정 범위) PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 7), facecolor="black")
    ax2d = fig.add_subplot(121, facecolor="black")
    ax3d = fig.add_subplot(122, projection="3d", facecolor="black")

    ax2d.imshow(draw_2d_overlay(frame_bgr, rec)[:, :, ::-1])
    ax2d.axis("off")
    ax2d.set_title(f"frame {rec['frame_idx']} - 2D keypoints", color="white")

    for side in (LEFT, RIGHT):
        if not rec["detected"][side]:
            continue
        b = HAND_BASE[side]
        pts = raw_to_view(rec["kp3d_abs"][b:b + JOINTS_PER_HAND])
        for s, e in HAND_CONNECTIONS:
            ax3d.plot(*zip(pts[s], pts[e]), color=joint_color(e), linewidth=2)
        ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                     c=[joint_color(i) for i in range(JOINTS_PER_HAND)], s=18, depthshade=False)
        ax3d.text(*pts[WRIST], f" {HAND_LABELS[side]}", color=HAND_BASE_COLORS[side], fontsize=9)

    ax3d.set_xlim(*view_lims[0]); ax3d.set_ylim(*view_lims[1]); ax3d.set_zlim(*view_lims[2])
    ax3d.set_xlabel("X [m]", color="white"); ax3d.set_ylabel("depth [m]", color="white")
    ax3d.set_zlabel("up [m]", color="white")
    ax3d.tick_params(colors="white", labelsize=7)
    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("gray")
    ax3d.set_title("3D (camera space, Z-up view)", color="white")
    fig.tight_layout()
    fig.savefig(out_png, facecolor="black", dpi=90)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 보조: 2D 그리기 · 깊이 그래프 · 메시 · 영상
# ═══════════════════════════════════════════════════════════════════════════

def draw_2d_overlay(frame_bgr, rec):
    """원본 위에 bbox + 21점 스켈레톤(손가락 색) 을 cv2 로 그린 복사본 반환."""
    img = frame_bgr.copy()
    for side in (LEFT, RIGHT):
        if not rec["detected"][side]:
            continue
        b = HAND_BASE[side]
        kp = rec["kp2d"][b:b + JOINTS_PER_HAND]
        base_bgr = hex_to_bgr(HAND_BASE_COLORS[side])
        x1, y1, x2, y2 = rec["bbox"][side].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), base_bgr, 2)
        cv2.putText(img, f"{HAND_LABELS[side]} {rec['scores'][side]:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, base_bgr, 2)
        for s, e in HAND_CONNECTIONS:
            cv2.line(img, tuple(kp[s].astype(int)), tuple(kp[e].astype(int)),
                     hex_to_bgr(joint_color(e)), 2, cv2.LINE_AA)
        for i, (u, v) in enumerate(kp):
            cv2.circle(img, (int(u), int(v)), 5 if i == WRIST else 3, hex_to_bgr(joint_color(i)), -1, cv2.LINE_AA)
    return img


def wrist_depth_series(records):
    """손목 z(깊이) 시계열 — 반환 (frames (F,), z (F,2) 미검출은 NaN)."""
    frames = np.array([r["frame_idx"] for r in records])
    z = np.full((len(records), 2), np.nan, dtype=np.float64)
    for k, r in enumerate(records):
        for side in (LEFT, RIGHT):
            if r["detected"][side]:
                z[k, side] = r["kp3d_abs"][HAND_BASE[side] + WRIST, 2]
    return frames, z


def jitter_mm(z):
    """연속 프레임 차분으로 본 깊이 지터 (mm) — (표준편차, 절대값 중앙값). 유효 구간 없으면 NaN.
    표준편차는 튐(outlier)에 민감하고 중앙값은 둔감하다 — 둘을 같이 보면 '자잘한 떨림'과 '튐'이 구분된다."""
    d = np.diff(z)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return float("nan"), float("nan")
    return float(np.std(d) * 1000.0), float(np.median(np.abs(d)) * 1000.0)


def jitter_text(z):
    sd, med = jitter_mm(z)
    return f"std {sd:.1f} / med {med:.1f} mm"


def smooth_runs(z, window=9, polyorder=2):
    """현행 MimicForge 보정(extract_hand.smooth_depth 와 같은 파라미터) — 비교 참고선용.
    검출 연속 구간 단위 Savitzky-Golay, 미검출(NaN)은 그대로."""
    from scipy.signal import savgol_filter
    out = z.copy()
    valid = np.isfinite(z)
    edges = np.flatnonzero(np.diff(np.concatenate(([False], valid, [False]))))
    for s, e in zip(edges[::2], edges[1::2]):
        n = e - s
        if n < 5:
            continue
        w = min(window, n)
        w -= (w + 1) % 2   # 홀수로
        out[s:e] = savgol_filter(z[s:e], w, polyorder)
    return out


def plot_wrist_depth(records, fps, out_png):
    """손목 z 시계열: raw 실선 + 현행 보정 점선, 제목에 raw 지터(mm)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames, z = wrist_depth_series(records)
    t = frames / fps
    fig, ax = plt.subplots(figsize=(12, 4))
    title = []
    for side in (LEFT, RIGHT):
        c = HAND_BASE_COLORS[side]
        ax.plot(t, z[:, side], color=c, linewidth=1.2, label=f"{HAND_LABELS[side]} raw")
        ax.plot(t, smooth_runs(z[:, side]), color=c, linestyle="--", linewidth=1.0, alpha=0.7,
                label=f"{HAND_LABELS[side]} savgol(9,2) ref")
        title.append(f"{HAND_LABELS[side]} jitter {jitter_text(z[:, side])}")
    ax.set_xlabel("time [s]"); ax.set_ylabel("wrist depth z [m]")
    ax.set_title("wrist depth (raw, no smoothing)   " + "  |  ".join(title))
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


def save_meshes(records, faces, out_dir):
    """프레임·손별 MANO 메시 .obj (절대 카메라 공간). 왼손은 면 방향 반전(faces[:, [0,2,1]])."""
    import trimesh
    out_dir.mkdir(parents=True, exist_ok=True)
    faces_left = faces[:, [0, 2, 1]]
    n = 0
    for r in records:
        for side, tag in ((LEFT, "L"), (RIGHT, "R")):
            v = r["verts_abs"][side]
            if v is None:
                continue
            trimesh.Trimesh(v, faces if side == RIGHT else faces_left, process=False).export(
                out_dir / f"frame_{r['frame_idx']:06d}_{tag}.obj")
            n += 1
    return n


def make_video(png_paths, fps, out_mp4):
    first = cv2.imread(str(png_paths[0]))
    h, w = first.shape[:2]
    vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for p in png_paths:
        vw.write(cv2.imread(str(p)))
    vw.release()


# ═══════════════════════════════════════════════════════════════════════════
# 프레임 → 레코드 (손 슬롯 2개: 0 왼손, 1 오른손)
# ═══════════════════════════════════════════════════════════════════════════

def empty_record(frame_idx):
    return {
        "frame_idx": frame_idx,
        "kp3d_abs": np.zeros((42, 3), np.float32),
        "kp2d": np.zeros((42, 2), np.float32),
        "cam_t": np.zeros((2, 3), np.float32),
        "mano_pose": np.zeros((2, 48), np.float32),
        "mano_shape": np.zeros((2, 10), np.float32),
        "bbox": np.zeros((2, 4), np.float32),
        "scores": np.zeros(2, np.float32),
        "detected": np.zeros(2, bool),
        "verts_abs": [None, None],
    }


def process_frame(predictor, frame_bgr, frame_idx, keep_verts):
    """한 프레임에 1→2→3 단계를 적용해 레코드로 정리. 같은 손이 둘 이상이면 score 최고만."""
    rec = empty_record(frame_idx)
    boxes, is_right, scores = stage1_detect(predictor, frame_bgr)           # 1
    hands = stage2_fit_mano(predictor, frame_bgr, boxes, is_right, scores)  # 2
    for hand in hands:
        side = RIGHT if hand.is_right else LEFT
        if rec["detected"][side] and rec["scores"][side] >= hand.score:
            continue
        kp3d_abs, verts_abs = stage3_to_camera_space(hand)                  # 3
        b = HAND_BASE[side]
        rec["kp3d_abs"][b:b + JOINTS_PER_HAND] = kp3d_abs
        rec["kp2d"][b:b + JOINTS_PER_HAND] = hand.keypoints_2d
        rec["cam_t"][side] = hand.cam_t
        rec["mano_pose"][side] = hand.mano_pose
        rec["mano_shape"][side] = hand.mano_shape
        rec["bbox"][side] = hand.bbox
        rec["scores"][side] = hand.score
        rec["detected"][side] = True
        rec["verts_abs"][side] = verts_abs.astype(np.float32) if keep_verts else None
    return rec


def iter_frames(input_path, stride, max_frames):
    """영상이면 (frame_idx, frame_bgr) 를 stride 간격으로, 이미지면 한 장. 마지막에 (fps, W, H) 는 속성으로."""
    if input_path.suffix.lower() in VIDEO_EXTENSIONS:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise SystemExit(f"영상을 열 수 없습니다: {input_path}")
        iter_frames.meta = (cap.get(cv2.CAP_PROP_FPS) or 30.0,
                            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        idx = yielded = 0
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames and yielded >= max_frames):
                break
            if idx % stride == 0:
                yield idx, frame
                yielded += 1
            idx += 1
        cap.release()
    else:
        img = cv2.imread(str(input_path))
        if img is None:
            raise SystemExit(f"이미지를 열 수 없습니다: {input_path}")
        iter_frames.meta = (0.0, img.shape[1], img.shape[0], 1)
        yield 0, img


def view_limits(records, pad=0.08):
    """3D 패널 고정 범위 — 시퀀스 전체 검출 관절의 표시 공간 경계(정육면체).
    깊이 튐(outlier)이 범위를 잡아먹지 않도록 축별 3~97 백분위로 잡는다. 범위 밖 프레임은 잘려 보인다."""
    pts = [raw_to_view(r["kp3d_abs"][HAND_BASE[s]:HAND_BASE[s] + JOINTS_PER_HAND])
           for r in records for s in (LEFT, RIGHT) if r["detected"][s]]
    if not pts:
        return [(-0.3, 0.3), (-0.8, -0.2), (-0.3, 0.3)]
    pts = np.concatenate(pts)
    lo, hi = np.percentile(pts, 3, axis=0) - pad, np.percentile(pts, 97, axis=0) + pad
    mid, half = (lo + hi) / 2, (hi - lo).max() / 2
    return [(mid[i] - half, mid[i] + half) for i in range(3)]


def save_results(records, out_dir, input_path, meta):
    fps, w, h, _ = meta
    np.savez(
        out_dir / "results.npz",
        kp3d_abs=np.stack([r["kp3d_abs"] for r in records]),
        kp2d=np.stack([r["kp2d"] for r in records]),
        cam_t=np.stack([r["cam_t"] for r in records]),
        mano_pose=np.stack([r["mano_pose"] for r in records]),
        mano_shape=np.stack([r["mano_shape"] for r in records]),
        bbox=np.stack([r["bbox"] for r in records]),
        scores=np.stack([r["scores"] for r in records]),
        detected=np.stack([r["detected"] for r in records]),
        frame_indices=np.array([r["frame_idx"] for r in records]),
        fps=fps, width=w, height=h,
        source_path=str(input_path.resolve()),
        coordinate_system="absolute_camera_space (X right, Y down, Z depth, metres) - raw, no smoothing",
        layout="42 = 0-20 left, 21-41 right; OpenPose 21 order",
    )


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="영상(.mp4 등) 또는 이미지 파일")
    ap.add_argument("--out", required=True, help="출력 폴더")
    ap.add_argument("--max_frames", type=int, default=0, help="처리할 최대 프레임 수 (0=전체)")
    ap.add_argument("--stride", type=int, default=1, help="프레임 건너뛰기 간격")
    ap.add_argument("--checkpoint", choices=list(CHECKPOINTS), default="anyhand")
    ap.add_argument("--det_conf", type=float, default=0.3, help="YOLO 검출 확신도 임계")
    ap.add_argument("--det_iou", type=float, default=0.3, help="YOLO NMS IoU")
    ap.add_argument("--rescale_factor", type=float, default=2.0, help="크롭 확대 배율 (클로즈업은 1.2~1.5)")
    ap.add_argument("--save_mesh", action="store_true", help="MANO 메시 .obj 저장 (meshes/)")
    ap.add_argument("--make_video", action="store_true", help="PNG 를 demo.mp4 로 이어붙임")
    ap.add_argument("--no_png", action="store_true", help="프레임 PNG 생략 (뷰어로만 볼 때)")
    ap.add_argument("--gui", action="store_true", help="완료 후 viewer_qt.py 를 별도 프로세스로 실행")
    args = ap.parse_args()

    input_path = Path(args.input).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[load] checkpoint={args.checkpoint} ...", flush=True)
    predictor = load_predictor(args.checkpoint, args.det_conf, args.det_iou, args.rescale_factor)
    print(f"[load] done in {time.time() - t0:.1f}s", flush=True)

    # ── 1차 패스: 추론 (1→2→3 단계) ──
    records, frames_cache = [], {}
    for idx, frame in iter_frames(input_path, args.stride, args.max_frames):
        rec = process_frame(predictor, frame, idx, keep_verts=args.save_mesh)
        records.append(rec)
        if not args.no_png:
            frames_cache[idx] = frame
        if len(records) % 30 == 0:
            print(f"[infer] {len(records)} frames ...", flush=True)
    meta = iter_frames.meta
    fps = meta[0]
    n_det = int(sum(r["detected"].any() for r in records))
    print(f"[infer] {len(records)} frames, hands in {n_det} - {time.time() - t0:.1f}s", flush=True)
    if not records:
        raise SystemExit("처리할 프레임이 없습니다.")

    save_results(records, out_dir, input_path, meta)
    print(f"[save] results.npz")

    # ── 2차 패스: 시각화 (4 단계) — 3D 축 범위는 전체 시퀀스로 고정 ──
    png_paths = []
    if not args.no_png:
        lims = view_limits(records)
        for r in records:
            p = out_dir / f"frame_{r['frame_idx']:06d}.png"
            stage4_visualize(frames_cache[r["frame_idx"]], r, lims, p)
            png_paths.append(p)
        print(f"[save] {len(png_paths)} PNG")

    is_video = len(records) > 1
    if is_video:
        plot_wrist_depth(records, fps / args.stride, out_dir / "wrist_depth.png")
        _, z = wrist_depth_series(records)
        print(f"[save] wrist_depth.png  (raw wrist-z jitter  L {jitter_text(z[:, LEFT])}  |  "
              f"R {jitter_text(z[:, RIGHT])})")
    if args.save_mesh:
        n = save_meshes(records, predictor.mano_faces, out_dir / "meshes")
        print(f"[save] {n} .obj in meshes/")
    if args.make_video and png_paths:
        make_video(png_paths, fps / args.stride, out_dir / "demo.mp4")
        print("[save] demo.mp4")
    print(f"[done] {out_dir}  ({time.time() - t0:.1f}s)")

    if args.gui:
        viewer = _DEMO_DIR / "viewer_qt.py"
        print(f"[gui] {sys.executable} {viewer} {out_dir / 'results.npz'}")
        try:
            subprocess.Popen([sys.executable, str(viewer), str(out_dir / "results.npz")], cwd=str(_DEMO_DIR))
        except OSError as exc:
            print(f"[gui] 실행 실패: {exc}\n      GUI 는 선택 설치: pip install PyQt5 pyqtgraph")


if __name__ == "__main__":
    main()
