"""CoTracker3(오프라인)로 기준 프레임의 관절을 영상 전체에서 직접 추적 (실험 98).

광류 사슬(klt.py)은 손 피부의 무늬점 움직임을 평면 변환 하나로 요약해 관절을 옮긴다. 그래서 그 평면 밖의 관절은 손이 돌수록 어긋난다.
CoTracker3 는 처음 찍은 점 자체를 영상 끝까지 따라가도록 학습된 모델이라, 평면 가정 없이 관절마다 따로 움직인다.

  - 모델: facebookresearch/co-tracker 의 cotracker3_offline (torch hub 캐시에서 불러옴, 내려받지 않음). 영상 전체를 보고 앞뒤로 추적한다.
  - 질의점: 기준 프레임의 MediaPipe 관절 21개 + 같은 프레임 손 영역 안의 보조점(support_points). 보조점은 함께 추적해 관절 추적을 돕고, 결과에는 쓰지 않는다.
  - 입력 영상은 scale 배로 줄여 GPU 에 올린다 (모델은 안에서 다시 model_resolution 으로 줄임). 결과 좌표는 원래 크기로 되돌린다.
"""
from pathlib import Path

import cv2
import numpy as np

from .klt import kt_hand_mask


def load_model(hub_dir: str, device: str):
    """torch hub 캐시의 저장소에서 cotracker3_offline 을 불러온다 (source=local, 가중치도 캐시에서)."""
    import torch
    m = torch.hub.load(str(Path(hub_dir).expanduser()), "cotracker3_offline", source="local")
    return m.to(device).eval()


def support_points(kp_ref: np.ndarray, box_ref: np.ndarray, shape, dilate_frac: float, step_frac: float) -> np.ndarray:
    """기준 프레임 손 영역(관절 볼록 껍질을 넓힌 것, kt_hand_mask) 안의 격자점. 간격 = 손 상자 긴 변 x step_frac."""
    m = kt_hand_mask(kp_ref, box_ref, shape, dilate_frac)
    s = max(2, int(round(step_frac * max(box_ref[2] - box_ref[0], box_ref[3] - box_ref[1]))))
    ys, xs = np.mgrid[s // 2:shape[0]:s, s // 2:shape[1]:s]
    k = m[ys, xs] > 0
    return np.c_[xs[k], ys[k]].astype(float)


def track_points(model, pngs: list, ref: int, P: np.ndarray, scale: float, device: str) -> tuple:
    """pngs 의 모든 프레임에서 기준 프레임 ref 의 점 P (N, 2, 원래 픽셀)를 추적. 반환: (점 (프레임, N, 2) 원래 픽셀, 보임 (프레임, N))."""
    import torch
    fr = []
    for p in pngs:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)[:, :, ::-1]
        fr.append(cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
    video = torch.from_numpy(np.stack(fr)).permute(0, 3, 1, 2)[None].float().to(device)
    q = torch.from_numpy(np.c_[np.full(len(P), float(ref)), P * scale]).float()[None].to(device)
    with torch.no_grad():
        tracks, vis = model(video, queries=q, backward_tracking=True)
    out = tracks[0].cpu().numpy() / scale
    v = vis[0].cpu().numpy() > 0.5 if vis.dtype != torch.bool else vis[0].cpu().numpy()
    del video
    torch.cuda.empty_cache()
    return out, v
