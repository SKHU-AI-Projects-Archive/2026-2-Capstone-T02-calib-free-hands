"""손 21점 위상·색·표시 좌표 변환 — 상수 전용 모듈.

demo_hand_mano.py(추론, torch 필요)와 viewer_qt.py(GUI, Qt 필요)가 공통으로 import 한다.
torch/Qt 어느 쪽도 import 하지 않으므로 어느 환경에서든 가져올 수 있다.

관절 인덱스 (한 손 21점, OpenPose 순서 — WiLoR mano_to_openpose):
  0 손목
  1-4  엄지 (CMC, MCP, IP, TIP)
  5-8  검지 (MCP, PIP, DIP, TIP)
  9-12 중지
  13-16 약지
  17-20 소지
양손 배열은 42점: 0-20 왼손, 21-41 오른손 (MimicForge hand-data 규약).
"""
import numpy as np

JOINTS_PER_HAND = 21
LEFT, RIGHT = 0, 1                      # 손 슬롯 인덱스
HAND_BASE = {LEFT: 0, RIGHT: 21}        # 42점 배열에서 각 손의 시작 인덱스

# 뼈 연결 20쌍 — 앱 apps/renderer/.../constants/handKeypointMapping.ts 의 SINGLE_HAND_CONNECTIONS 와 동일 순서
HAND_CONNECTIONS = [
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),   # 손목 → 각 손가락 기저
    (1, 2), (2, 3), (3, 4),                     # 엄지
    (5, 6), (6, 7), (7, 8),                     # 검지
    (9, 10), (10, 11), (11, 12),                # 중지
    (13, 14), (14, 15), (15, 16),               # 약지
    (17, 18), (18, 19), (19, 20),               # 소지
]

# 손가락별 색 (앱 HAND_FINGER_COLORS 그대로)
FINGER_COLORS = {
    "wrist":  "#FFFFFF",
    "thumb":  "#FF6B6B",
    "index":  "#FFA500",
    "middle": "#FFD93D",
    "ring":   "#6BCF7F",
    "pinky":  "#4D96FF",
}
_FINGER_OF_JOINT = (
    ["wrist"] + ["thumb"] * 4 + ["index"] * 4 + ["middle"] * 4 + ["ring"] * 4 + ["pinky"] * 4
)

# 손 구분 색 (앱 HAND_PART_COLORS) — 2D 오버레이 뼈대·범례에 사용
HAND_BASE_COLORS = {LEFT: "#FF00FF", RIGHT: "#00FF00"}
HAND_LABELS = {LEFT: "Left", RIGHT: "Right"}


def joint_color(local_idx: int) -> str:
    """한 손 안의 관절 인덱스(0-20) → 손가락 색 hex."""
    return FINGER_COLORS[_FINGER_OF_JOINT[local_idx]]


def hex_to_rgb(hex_color: str):
    """'#RRGGBB' → (r, g, b) 0-255 정수 튜플."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def hex_to_bgr(hex_color: str):
    """'#RRGGBB' → (b, g, r) — cv2 그리기용."""
    r, g, b = hex_to_rgb(hex_color)
    return (b, g, r)


# WiLoR raw 카메라 공간 (X 오른쪽, Y 아래, Z 깊이) → 표시 공간 (Z-up)
#   (x, y, z) → (-x, -z, -y)
# MimicForge 앱 protocol 의 RAW_TO_ISAAC_HAND 와 동일 (packages/protocol/src/transforms.ts:19,
# python/mimicforge_protocol/transforms.py). det = +1 (proper rotation) 이라 거울상이 생기지 않고,
# 앱의 스켈레톤 뷰·Isaac 화면과 같은 방향으로 보인다. 데모 PNG 3D 패널과 Qt 뷰어가 같이 쓴다.
RAW_TO_VIEW = np.array([
    [-1, 0, 0],
    [0, 0, -1],
    [0, -1, 0],
], dtype=np.float64)


def raw_to_view(points: np.ndarray) -> np.ndarray:
    """(..., 3) raw 카메라 공간 → 표시 공간. 0 벡터(미검출)는 0 그대로."""
    return points @ RAW_TO_VIEW.T
