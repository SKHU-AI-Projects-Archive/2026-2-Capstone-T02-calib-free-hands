"""손 광류(KLT) 추적 사슬 - 실험 56 이후 주 파이프라인이 쓰는 부분.

원본: handfocal_report.py 의 kt_hand_mask, kt_chain, kt_carry, kt_map 과 run.py 의 pp_points, sg_video_job(사슬 부분).
원본은 그대로 두었고(추가만 한다는 규칙), 이 파일은 같은 계산을 한 곳에 모은 사본이다. 함수 이름은 원본과 같게 두어 대응을 쉽게 했다.
numpy, OpenCV 외에는 다른 프로젝트 파일을 import 하지 않는다.

흐름 (한 영상):
  1. MediaPipe 로 검출하고 추적한 손 관절(tr_kp, 프레임마다 21개)과 손 상자를 받는다. 검출이 없는 프레임은 직전 검출값을 이어 쓴다(kt_carry).
  2. 기준 프레임 = 검출된 프레임 범위의 한가운데에 가장 가까운 검출 프레임(choose_reference).
  3. 기준 프레임에서 앞뒤로 이웃 프레임 사이 평면 변환(호모그래피)을 구해 차례로 곱한다(kt_chain).
     - 이전 프레임의 손 영역(관절 볼록 껍질을 손 상자 긴 변의 dilate_frac 만큼 넓힘, kt_hand_mask) 안에서 Shi-Tomasi 특징점을 새로 뽑는다.
     - 피라미드 Lucas-Kanade 로 다음 프레임 위치를 찾고, 거꾸로 다시 추적해 fb_px 넘게 어긋나는 점은 버린다.
     - 남은 점으로 USAC_MAGSAC 호모그래피를 구한다. inlier 가 min_inliers 미만이면 그 지점부터 사슬이 끊긴다.
  4. 기준 프레임의 검출 관절 21개를 각 프레임까지의 변환으로 옮긴 것이 추적점이다(kt_map, chain_points).
  5. 옮긴 점의 상자가 화면 가장자리(2 px)에 닿는 프레임은 쓰지 않는다(edge_flags).
"""
import cv2
import numpy as np


# ----------------------------------------------------------------------------------------------------
# 작은 도우미 (원본: handfocal_estimate.box_long, handfocal_report.edge_flags)
# ----------------------------------------------------------------------------------------------------
def box_long(box):
    """손 상자 (x0, y0, x1, y1) 의 긴 변 길이."""
    return np.maximum(box[..., 2] - box[..., 0], box[..., 3] - box[..., 1])


def edge_flags(box, present, W, H, margin=2.0):
    """상자가 화면 가장자리 margin px 안에 닿는 프레임이면 True (present 인 프레임만 검사)."""
    e = np.zeros(len(box), bool)
    for i in np.flatnonzero(present):
        b = box[i]
        e[i] = bool(b[0] <= margin or b[1] <= margin or b[2] >= W - margin or b[3] >= H - margin)
    return e


# ----------------------------------------------------------------------------------------------------
# 사슬의 구성 요소 (원본: handfocal_report.kt_hand_mask / kt_chain / kt_carry / kt_map)
# ----------------------------------------------------------------------------------------------------
def kt_hand_mask(kp: np.ndarray, box: np.ndarray, shape, dilate_frac: float) -> np.ndarray:
    """손 영역 마스크: 관절들의 볼록 껍질을 채우고, 손 상자 긴 변 x dilate_frac 만큼(최소 1 px) 넓힌다. 관절이 3개 미만이면 빈 마스크."""
    m = np.zeros(shape, np.uint8)
    p = kp[np.isfinite(kp).all(1)].astype(np.float32)
    if len(p) >= 3:
        cv2.fillConvexPoly(m, cv2.convexHull(p).astype(np.int32), 255)
        r = max(1, int(round(float(dilate_frac) * float(box_long(np.asarray(box, float)[None])[0]))))
        m = cv2.dilate(m, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
    return m


def kt_chain(pngs: list, kps: np.ndarray, boxes: np.ndarray, order: list, kcfg: dict) -> dict:
    """order[0] 에서 order 의 뒤 프레임들까지의 호모그래피 사슬.

    단계마다(prev -> cur):
      - prev 프레임의 손 마스크 안에서 goodFeaturesToTrack (최대 max_corners 개, 최고 점수의 quality 배 이상, 서로 min_distance px 이상, blockSize=block)
      - 피라미드 LK (win x win 창, levels 단계) 로 cur 에서의 위치를 찾고, cur -> prev 로 되돌려 fb_px 미만으로 돌아온 점만 남김
      - 남은 점으로 USAC_MAGSAC 호모그래피 (threshold_px). inlier 가 min_inliers 미만이면 사슬이 끊겨 그 뒤 프레임은 모두 None
      - 기준 -> cur 변환 = (prev -> cur 변환) @ (기준 -> prev 변환)
    kps / boxes: 프레임마다의 관절과 상자(검출이 없는 프레임은 직전 값을 이어 쓴 것).
    반환: {H: {프레임: 3x3 또는 None}, breaks: 끊긴 횟수, inl: 단계별 inlier 수 목록}.
    """
    lk = dict(winSize=(int(kcfg["win"]), int(kcfg["win"])), maxLevel=int(kcfg["levels"]), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    Hc = {order[0]: np.eye(3)}
    inls, breaks, broken = [], 0, False
    gray = lambda i: cv2.cvtColor(cv2.imread(pngs[i], cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)      # noqa: E731
    g0 = gray(order[0])
    for prev, cur in zip(order[:-1], order[1:]):
        g1 = gray(cur)
        if broken:
            Hc[cur] = None
            g0 = g1
            continue
        m = kt_hand_mask(kps[prev], boxes[prev], g0.shape, kcfg["dilate_frac"])
        c = cv2.goodFeaturesToTrack(g0, int(kcfg["max_corners"]), float(kcfg["quality"]), float(kcfg["min_distance"]), mask=m, blockSize=int(kcfg["block"]))
        Hs = None
        if c is not None and len(c) >= 8:
            p0 = c.reshape(-1, 2).astype(np.float32)
            p1, s1, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0.reshape(-1, 1, 2), None, **lk)
            pb, s2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, **lk)
            ok = (s1.ravel() == 1) & (s2.ravel() == 1) & (np.linalg.norm(pb.reshape(-1, 2) - p0, axis=1) < float(kcfg["fb_px"]))
            if ok.sum() >= int(kcfg["min_inliers"]):
                Hs, inl = cv2.findHomography(p0[ok].astype(np.float64), p1.reshape(-1, 2)[ok].astype(np.float64), cv2.USAC_MAGSAC, float(kcfg["threshold_px"]))
                if Hs is None or inl is None or int(inl.sum()) < int(kcfg["min_inliers"]):
                    Hs = None
                else:
                    inls.append(int(inl.sum()))
        if Hs is None:
            breaks += 1
            broken = True
            Hc[cur] = None
        else:
            Hc[cur] = Hs @ Hc[prev]
        g0 = g1
    return {"H": Hc, "breaks": breaks, "inl": inls}


def kt_carry(present: np.ndarray, arr: np.ndarray) -> np.ndarray:
    """프레임마다의 값을, 검출이 없는 프레임은 직전 검출값으로 채운다(첫 검출 전 프레임은 첫 검출값)."""
    out = np.array(arr, float, copy=True)
    idx = np.flatnonzero(present)
    if not len(idx):
        return out
    last = idx[0]
    for i in range(len(out)):
        if present[i]:
            last = i
        out[i] = arr[last]
    return out


def kt_map(P: np.ndarray, H) -> np.ndarray:
    """점들 P (n, 2) 를 호모그래피 H 로 옮긴다. H 가 None(사슬이 끊김)이면 NaN."""
    if H is None:
        return np.full_like(np.asarray(P, float), np.nan)
    return cv2.perspectiveTransform(np.asarray(P, np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)


# ----------------------------------------------------------------------------------------------------
# 사슬로 추적점 만들기 (원본: run.py 의 pp_points 와 sg_video_job 의 사슬 부분)
# ----------------------------------------------------------------------------------------------------
def chain_points(P: np.ndarray, Hall: dict, n: int, W: int, H: int):
    """기준 관절 P 를 프레임별 변환 Hall 로 옮긴 추적점과, 그 점들의 상자, 유효 여부, 화면 가장자리 접촉 여부. (원본: run.pp_points)"""
    kp = np.full((n, len(P), 2), np.nan)
    for i, Hm in Hall.items():
        kp[i] = kt_map(P, Hm)
    valid = np.isfinite(kp).all((1, 2))
    tb = np.full((n, 4), np.nan)
    tb[valid] = np.c_[kp[valid].min(1), kp[valid].max(1)]
    edge = np.zeros(n, bool)
    edge[valid] = edge_flags(tb[valid], np.ones(valid.sum(), bool), W, H)
    return kp, tb, valid, edge


def choose_reference(present: np.ndarray) -> int:
    """실험 56 의 기준 프레임: 첫 검출 ~ 마지막 검출 범위의 한가운데에 가장 가까운 검출 프레임."""
    pres = np.flatnonzero(present)
    a, b = int(pres[0]), int(pres[-1])
    return int(pres[np.argmin(np.abs(pres - (a + b) / 2))])


def track_hand(pngs: list, tr_kp: np.ndarray, tr_present: np.ndarray, tr_box: np.ndarray, size_wh, kcfg: dict, ref: int = None) -> dict:
    """한 영상의 추적점 (sg_video_job 의 사슬 부분과 같은 계산).

    pngs: 프레임 이미지 경로 목록(30 fps 로 추출한 것). tr_kp (n, 21, 2), tr_present (n,), tr_box (n, 4): MediaPipe 검출을 추적(track)한 결과.
    kcfg: 실험 56 의 tracking 설정 (fps, dilate_frac, max_corners, quality, min_distance, block, win, levels, fb_px, threshold_px, min_inliers).
    ref: 기준 프레임(없으면 choose_reference). 실험 88 처럼 다른 기준을 쓸 때 넘긴다.
    반환: kp (추적점), box, use (사용 프레임 = 유효하고 가장자리에 닿지 않음), ref, breaks, inl (단계별 inlier 수).
    """
    W, H = size_wh
    n = len(pngs)
    pres = np.flatnonzero(tr_present)
    a, b = int(pres[0]), int(pres[-1])
    ref = choose_reference(tr_present) if ref is None else int(ref)
    kps = np.stack([kt_carry(tr_present, tr_kp[:, j, k]) for j in range(21) for k in range(2)], axis=1).reshape(n, 21, 2)
    boxes = np.stack([kt_carry(tr_present, tr_box[:, k]) for k in range(4)], axis=1)
    fwd = kt_chain(pngs, kps, boxes, list(range(ref, b + 1)), kcfg)
    bwd = kt_chain(pngs, kps, boxes, list(range(ref, a - 1, -1)), kcfg)
    kp, tb, valid, edge = chain_points(tr_kp[ref], {**bwd["H"], **fwd["H"]}, n, W, H)
    return {"kp": kp, "box": tb, "use": valid & ~edge, "ref": ref, "breaks": fwd["breaks"] + bwd["breaks"], "inl": fwd["inl"] + bwd["inl"]}
