"""영상에서 양손 3D 키포인트(42=21×2)를 추출하는 AnyHand-WiLoR 기반 포터블 도구.

pose-engine 추론 job 의 실모델 러너로 쓰인다 — 워커가 서브프로세스로 호출
(`services/pose-engine/extractor_runner.py`). 원본은 deployment/test6/AnyHand 의
`video_hand_analysis_absolute.py` — 앱에 불필요한 산출물(CSV/관절각/웹뷰어 JSON)을
덜어내고 2-pass(글로벌 bounds 스캔)를 1-pass 로 줄인 이식본.

출력 (--task_dir 바로 밑, 기존 hand-data 규약과 동일):
  frames/frame_000000.jpg ...   원본 프레임 (JPEG 95)
  hand_3d_keypoints.npy         (frames, 42, 3) float32, 미터 — 배치 정규화 적용(아래)
  hand_3d_keypoints.raw.npy     정규화 전 절대 카메라 공간 원본 (--no-normalize 면 없음)
  hand_2d_keypoints.npy         (frames, 42, 2) float32, 이미지 픽셀 좌표
  hand_metadata.json            fps/해상도/프레임별 손 검출 여부 + normalization 오프셋
  KEYPOINT_2D_STRUCTURE.md      레이아웃 문서 (인덱스 0-20 왼손, 21-41 오른손)

--progress 지정 시 stdout 에 `PROGRESS <0-100>` 라인을 출력한다(pose-engine 워커가 파싱).

좌표계 (원본 스크립트 도크스트링 요약):
  X → 이미지 오른쪽, Y → 이미지 아래, Z → 카메라 전방(깊이). 원점 = 카메라 광학 중심.
  절대 좌표 = 모델의 root-relative keypoints_3d + cam_t (손목 센터링 없음).
  기본값으로 배치 정규화(normalize_placement — 손목 중점 원점 + 최저점 접지)를 적용해
  뷰어 격자 원점 근처에 놓이게 한다. 축 방향은 그대로(강체 이동만).
"""
import argparse
import json
import os
import sys
from pathlib import Path

# MANO 경로가 CWD 상대('./mano_data/', WiLoR load_wilor 하드코딩)라 서비스 루트로 고정
_SERVICE_ROOT = Path(__file__).resolve().parent
os.chdir(_SERVICE_ROOT)

# WiLoR renderer 모듈이 PYOPENGL_PLATFORM=egl 을 강제(리눅스 헤드리스 전제) — Windows 엔
# EGL 이 없어 import 가 죽는다. 미리 네이티브 플랫폼을 지정해 선점(우리는 렌더링 안 씀).
if os.name == "nt":
    os.environ.setdefault("PYOPENGL_PLATFORM", "nt")

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}

# 선택 가능한 체크포인트 — (ckpt, 짝 config). config 는 반드시 체크포인트와 짝으로:
# 원본 WiLoR 는 FOCAL_LENGTH=5000, AnyHand 재학습본은 1000 으로 깊이 스케일 가정이 다르다.
CHECKPOINTS = {
    "anyhand": ("anyhand_wilor.ckpt", "model_config_wilor.yaml"),
    "wilor": ("wilor_final.ckpt", "model_config_wilor_original.yaml"),
}

_KEYPOINT_2D_DOC = """# 2D Hand Keypoints (`hand_2d_keypoints.npy`)

## Format
- **Shape**: `(frames, 42, 2)` float32 — indices 0-20: left hand, 21-41: right hand
- **Coordinate system**: image pixel space (`u` rightward, `v` downward)
- **Undetected hands**: filled with zeros

## Keypoint Indices (per hand)
| Index | Joint |
|-------|-------|
| 0 | Wrist |
| 1-4 | Thumb (CMC, MCP, IP, TIP) |
| 5-8 | Index (MCP, PIP, DIP, TIP) |
| 9-12 | Middle (MCP, PIP, DIP, TIP) |
| 13-16 | Ring (MCP, PIP, DIP, TIP) |
| 17-20 | Pinky (MCP, PIP, DIP, TIP) |

The matching 3D keypoints are in `hand_3d_keypoints.npy` (`(frames, 42, 3)`, meters,
placement-normalized: first-frame wrist midpoint at origin X/Z, lowest point at Y=0;
pre-normalization absolute camera space in `hand_3d_keypoints.raw.npy`); indices align
one-to-one. Extracted by AnyHand fine-tuned WiLoR (services/hand-extractor).
"""


def smooth_depth(keypoints: np.ndarray, hands_per_frame: list, window: int = 9) -> np.ndarray:
    """z(깊이)만 Savitzky-Golay 스무딩 — 단안 추정의 축별 오차 비대칭 대응 (2026-09-02).

    x/y 는 이미지 픽셀에 잠겨 있어 안정적이지만 z 는 "크롭에서 손이 얼마나 크게
    보이나"의 역산이라 프레임마다 출렁인다. z 만 부드럽게 하면 2D 오버레이 정합은
    그대로 두고 깊이 지터만 잡힌다. 손별(왼 0-20 / 오른 21-41)로 **검출 연속 구간
    단위** 처리 — 미검출(0) 슬롯은 건드리지 않고, 구간 경계를 넘어 번지지 않는다.

    window: 홀수 프레임 창(30fps 에서 9 ≈ 0.3초). 0 이면 스무딩 없음.
    """
    if window < 5:
        return keypoints
    from scipy.signal import savgol_filter

    out = keypoints.copy()
    for base, side in ((0, "left"), (21, "right")):
        detected = np.array([d[side] for d in hands_per_frame], dtype=bool)
        # 검출 연속 구간(run) 추출
        edges = np.flatnonzero(np.diff(np.concatenate(([False], detected, [False]))))
        for start, end in zip(edges[::2], edges[1::2]):
            length = end - start
            if length < 5:
                continue
            w = min(window, length)
            if w % 2 == 0:
                w -= 1
            out[start:end, base:base + 21, 2] = savgol_filter(
                keypoints[start:end, base:base + 21, 2], w, polyorder=2, axis=0
            )
    return out


def normalize_placement(keypoints: np.ndarray, hands_per_frame: list) -> tuple:
    """배치 정규화 — 절대 카메라 좌표를 씬 원점 근처로 옮기는 강체 이동 (2026-09-02).

    절대 카메라 공간은 손이 원점에서 0.5~1m+ 떨어져 있어 뷰어(격자 원점 기준)에서
    멀리 뜬다. 시퀀스 전체에 **단일 오프셋**만 빼서 모션·양손 상대관계는 그대로 두고:
      - X/Z(가로/깊이): 양손이 모두 검출된 첫 프레임의 두 손목(0, 21) 중점 → 원점
        (한 손만 나오는 영상은 그 손목, 기준 프레임은 검출된 첫 프레임)
      - Y(높이, 카메라 공간은 Y-down): 전체 프레임에서 물리적으로 가장 낮은 지점
        (= max y) → 0. 뷰어/브릿지 모두 바닥이 raw_y=0 평면이라 최저점이 격자에 닿는다.
    미검출 슬롯(0,0,0)은 결측 마커이므로 이동시키지 않는다.

    반환: (정규화 배열, 정규화 정보 dict | None — 손이 하나도 없으면 None)
    """
    detected = np.abs(keypoints).sum(axis=2) > 0   # (F, 42) 관절 검출 마스크
    if not detected.any():
        return keypoints, None

    ref_frame = next(
        (i for i, d in enumerate(hands_per_frame) if d["left"] and d["right"]), None
    )
    if ref_frame is None:
        ref_frame = next(i for i, d in enumerate(hands_per_frame) if d["left"] or d["right"])

    wrists = []
    if hands_per_frame[ref_frame]["left"]:
        wrists.append(keypoints[ref_frame, 0])
    if hands_per_frame[ref_frame]["right"]:
        wrists.append(keypoints[ref_frame, 21])
    mid = np.mean(wrists, axis=0)

    offset = np.array(
        [mid[0], float(keypoints[detected][:, 1].max()), mid[2]], dtype=np.float32
    )

    normalized = keypoints.copy()
    normalized[detected] -= offset
    info = {
        "offset_xyz": [float(v) for v in offset],
        "reference_frame": int(ref_frame),
        "method": "first_frame_wrist_midpoint_xz + global_lowest_to_ground_y",
    }
    return normalized, info


class _Progress:
    """PROGRESS <0-100> 라인 출력 (같은 값 중복 억제) — --progress 미지정 시 no-op."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.last = -1

    def emit(self, pct: int) -> None:
        pct = max(0, min(100, int(pct)))
        if self.enabled and pct != self.last:
            self.last = pct
            print(f"PROGRESS {pct}", flush=True)


def extract(
    video_path: Path,
    task_dir: Path,
    progress: _Progress,
    normalize: bool = True,
    checkpoint: str = "anyhand",
    det_conf: float = 0.3,
    det_iou: float = 0.3,
    rescale_factor: float = 2.0,
    z_smooth_window: int = 9,
) -> None:
    progress.emit(0)

    ckpt_file, cfg_file = CHECKPOINTS[checkpoint]
    inference_params = {
        "checkpoint": checkpoint,
        "det_conf": det_conf,
        "det_iou": det_iou,
        "rescale_factor": rescale_factor,
        "z_smooth_window": z_smooth_window,
    }
    print(f"Inference params: {inference_params}")

    # 모델 로드가 오래 걸리므로 import 자체를 진행률 구간에 포함
    from rgb_predictor import AnyHandPredictor
    predictor = AnyHandPredictor(
        backend="wilor",
        wilor_ckpt=str(_SERVICE_ROOT / "models" / ckpt_file),
        wilor_cfg=str(_SERVICE_ROOT / "models" / cfg_file),
        det_conf=det_conf,
        det_iou=det_iou,
        rescale_factor=rescale_factor,
    )
    progress.emit(5)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")

    frames_dir = task_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    # 재추론 시 이전 프레임 잔재 제거 (skeleton-extractor --task_dir 모드와 동일 관례)
    for f in frames_dir.iterdir():
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            f.unlink()

    all_joints_3d = []       # (42, 3) per frame — 절대 카메라 공간
    all_joints_2d = []       # (42, 2) per frame — 픽셀 좌표
    hands_per_frame = []     # {"left": bool, "right": bool} per frame

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imwrite(
            str(frames_dir / f"frame_{frame_idx:06d}.jpg"),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

        frame_joints = np.zeros((42, 3), dtype=np.float32)
        frame_joints_2d = np.zeros((42, 2), dtype=np.float32)
        detected = {"left": False, "right": False}

        hands = predictor.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for hand in hands:
            keypoints_3d_absolute = hand.keypoints_3d + hand.cam_t[np.newaxis, :]
            # 2D 는 모델의 크롭 투영 산출(keypoints_2d)을 그대로 쓴다 — 원본 스크립트의
            # project_3d_to_2d(스케일드 focal 재투영)는 프레임과 어긋나는 결함이 있었음
            # (기존 hand-data 의 hand_2d_keypoints.npy 에도 같은 왜곡이 존재).
            keypoints_2d = hand.keypoints_2d
            base = 21 if hand.is_right else 0
            n = min(keypoints_3d_absolute.shape[0], 21)
            frame_joints[base:base + n] = keypoints_3d_absolute[:n]
            frame_joints_2d[base:base + n] = keypoints_2d[:n]
            detected["right" if hand.is_right else "left"] = True

        all_joints_3d.append(frame_joints)
        all_joints_2d.append(frame_joints_2d)
        hands_per_frame.append(detected)

        frame_idx += 1
        if total_frames > 0:
            progress.emit(5 + int(90 * frame_idx / total_frames))
        if frame_idx % 30 == 0 or frame_idx == total_frames:
            print(f"Processed {frame_idx}/{total_frames} frames", flush=True)

    cap.release()

    if frame_idx == 0:
        raise ValueError("영상에서 프레임을 읽지 못했습니다")

    keypoints_3d = np.array(all_joints_3d, dtype=np.float32)
    keypoints_2d = np.array(all_joints_2d, dtype=np.float32)
    raw_keypoints = keypoints_3d.copy()

    # 깊이(z) 스무딩 → 배치 정규화 순서 — 원본(스무딩·정규화 전)은 .raw.npy 로 보존
    smoothed = z_smooth_window >= 5
    if smoothed:
        keypoints_3d = smooth_depth(keypoints_3d, hands_per_frame, z_smooth_window)
        print(f"Depth smoothed: savgol window={z_smooth_window} (z-axis only)")

    if normalize:
        keypoints_3d, norm_info = normalize_placement(keypoints_3d, hands_per_frame)
    else:
        norm_info = None
    coordinate_system = "wrist_centered_grounded" if norm_info else "absolute_camera_space"
    if norm_info or smoothed:
        np.save(task_dir / "hand_3d_keypoints.raw.npy", raw_keypoints)
        print("Raw (pre-smoothing/normalization) saved to hand_3d_keypoints.raw.npy")
    if norm_info:
        print(f"Placement normalized: offset={np.round(norm_info['offset_xyz'], 3).tolist()}"
              f" (ref frame {norm_info['reference_frame']})")

    np.save(task_dir / "hand_3d_keypoints.npy", keypoints_3d)
    np.save(task_dir / "hand_2d_keypoints.npy", keypoints_2d)
    print(f"Saved hand_3d_keypoints.npy {keypoints_3d.shape} / hand_2d_keypoints.npy {keypoints_2d.shape}")

    metadata = {
        "total_frames": frame_idx,
        "fps": fps,
        "width": width,
        "height": height,
        "hands_per_frame": hands_per_frame,
        "coordinate_system": coordinate_system,
        "model": "anyhand-wilor",
        "inference_params": inference_params,
    }
    if norm_info:
        metadata["normalization"] = {
            **norm_info,
            "raw_file": "hand_3d_keypoints.raw.npy",
            "raw_coordinate_system": "absolute_camera_space",
        }
    with open(task_dir / "hand_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    (task_dir / "KEYPOINT_2D_STRUCTURE.md").write_text(_KEYPOINT_2D_DOC, encoding="utf-8")

    n_detected = sum(1 for d in hands_per_frame if d["left"] or d["right"])
    print(f"Done: {frame_idx} frames, hands detected in {n_detected} frames")
    progress.emit(100)


def main() -> int:
    parser = argparse.ArgumentParser(description="AnyHand-WiLoR 양손 3D 키포인트 추출기")
    parser.add_argument("--input_path", required=True, help="입력 영상 파일")
    parser.add_argument("--task_dir", required=True, help="출력 작업 폴더 (frames/ + npy flat 기록)")
    parser.add_argument(
        "--progress", action="store_true",
        help='stdout 에 "PROGRESS <0-100>" 라인 출력 (pose-engine 워커가 파싱)',
    )
    parser.add_argument(
        "--no-normalize", action="store_true",
        help="배치 정규화(손목 중점 원점 + 최저점 접지) 생략 — 절대 카메라 공간 그대로 저장",
    )
    # 추론 파라미터 (모델 레지스트리 params 와 동일 어휘 — pose-engine 워커가 그대로 전달)
    parser.add_argument(
        "--checkpoint", choices=sorted(CHECKPOINTS), default="anyhand",
        help="사용할 체크포인트: anyhand(fine-tuned, 기본) | wilor(원본)",
    )
    parser.add_argument(
        "--det_conf", type=float, default=0.3,
        help="YOLO 검출 확신도 임계 (낮추면 미검출↓·오검출↑)",
    )
    parser.add_argument(
        "--det_iou", type=float, default=0.3,
        help="YOLO NMS IoU 임계 (겹친 손 처리)",
    )
    parser.add_argument(
        "--rescale_factor", type=float, default=2.0,
        help="검출 박스 크롭 확대 배율 (클로즈업은 낮게, 원거리는 높게)",
    )
    parser.add_argument(
        "--z_smooth_window", type=int, default=9,
        help="깊이(z) Savitzky-Golay 스무딩 창(홀수 프레임, 30fps 에서 9≈0.3초). 5 미만 = 끔",
    )
    args = parser.parse_args()

    video_path = Path(args.input_path).resolve()
    if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        print(f"Error: 영상 파일이 아닙니다: {video_path}")
        return 1

    task_dir = Path(args.task_dir).resolve()
    task_dir.mkdir(parents=True, exist_ok=True)

    extract(
        video_path, task_dir, _Progress(args.progress),
        normalize=not args.no_normalize,
        checkpoint=args.checkpoint,
        det_conf=args.det_conf,
        det_iou=args.det_iou,
        rescale_factor=args.rescale_factor,
        z_smooth_window=args.z_smooth_window,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
