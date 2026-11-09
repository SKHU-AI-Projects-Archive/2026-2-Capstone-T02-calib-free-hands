"""광류 추적 사슬의 실험용 변형 (실험 77, 80, 83, 86, 87).

주 파이프라인(klt.py)은 바꾸지 않고, 원인을 확인하려고 만든 변형만 모았다. 원본은 handfocal_report.py 에 그대로 있다(함수 이름 같음).
  - kt_chain_log (77): 사슬과 같은 계산 + 단계마다 inlier 특징점의 손 좌표와 잔차 기록
  - rt_windows (80): 영상을 일정 길이 구간으로 나누고 구간마다 기준(앵커) 프레임을 정함 - 추적을 구간마다 다시 시작
  - bc_chain (83): 특징점을 손 몸통과 손가락에서 같은 수로 뽑는 사슬
  - wt_track (83): 손목 주변 특징점만으로 손목을 따로 옮김
  - kv_step (86): 사슬의 한 단계를 중간 결과까지 모두 돌려줌 (그림용)
  - eg_paper / eg_window_frac / eg_chain (87): 흰 종이와 맞닿은 윤곽(경계) 특징점을 기록하거나 빼고 추적
"""
import cv2
import numpy as np

from .klt import box_long, kt_hand_mask, kt_map


def _lk(kcfg: dict) -> dict:
    """사슬과 같은 피라미드 LK 설정."""
    return dict(winSize=(int(kcfg["win"]), int(kcfg["win"])), maxLevel=int(kcfg["levels"]), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


# ----------------------------------------------------------------------------------------------------
# 77: 특징점이 손의 어디에 있는지 기록
# ----------------------------------------------------------------------------------------------------
def kt_chain_log(pngs: list, kps: np.ndarray, boxes: np.ndarray, order: list, kcfg: dict, mask_kps: np.ndarray = None) -> dict:
    """kt_chain 과 같은 단계 + 단계마다 inlier 특징점의 손 좌표와 호모그래피 잔차(px).

    손 좌표: 이전 프레임 관절 기준. s = 손목(0) -> 중지 MCP(9) 방향 (0 = 손목, 1 = 중지 MCP), t = 그에 수직, 같은 단위.
    mask_kps: 특징점 마스크를 만들 관절 (기본 kps. 실험 58 의 손등 몸통 사슬은 손가락 관절을 NaN 으로 넘김).
    반환: kt_chain 의 dict + s, t, res (모든 단계를 이어 붙인 float32 배열), n_ok (단계별 전후방 검사 통과 수).
    """
    lk = _lk(kcfg)
    mk = kps if mask_kps is None else mask_kps
    Hc = {order[0]: np.eye(3)}
    inls, n_ok, S, T, R = [], [], [], [], []
    breaks, broken = 0, False
    gray = lambda i: cv2.cvtColor(cv2.imread(pngs[i], cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)      # noqa: E731
    g0 = gray(order[0])
    for prev, cur in zip(order[:-1], order[1:]):
        g1 = gray(cur)
        if broken:
            Hc[cur] = None
            g0 = g1
            continue
        m = kt_hand_mask(mk[prev], boxes[prev], g0.shape, kcfg["dilate_frac"])
        c = cv2.goodFeaturesToTrack(g0, int(kcfg["max_corners"]), float(kcfg["quality"]), float(kcfg["min_distance"]), mask=m, blockSize=int(kcfg["block"]))
        Hs = None
        if c is not None and len(c) >= 8:
            p0 = c.reshape(-1, 2).astype(np.float32)
            p1, s1, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0.reshape(-1, 1, 2), None, **lk)
            pb, s2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, **lk)
            ok = (s1.ravel() == 1) & (s2.ravel() == 1) & (np.linalg.norm(pb.reshape(-1, 2) - p0, axis=1) < float(kcfg["fb_px"]))
            n_ok.append(int(ok.sum()))
            if ok.sum() >= int(kcfg["min_inliers"]):
                Hs, inl = cv2.findHomography(p0[ok].astype(np.float64), p1.reshape(-1, 2)[ok].astype(np.float64), cv2.USAC_MAGSAC, float(kcfg["threshold_px"]))
                if Hs is None or inl is None or int(inl.sum()) < int(kcfg["min_inliers"]):
                    Hs = None
                else:
                    inls.append(int(inl.sum()))
                    q0 = p0[ok][inl.ravel() > 0].astype(np.float64)
                    q1 = p1.reshape(-1, 2)[ok][inl.ravel() > 0].astype(np.float64)
                    R.append(np.linalg.norm(kt_map(q0, Hs) - q1, axis=1))
                    w0, m9 = np.asarray(kps[prev][0], float), np.asarray(kps[prev][9], float)
                    ax = m9 - w0
                    L2 = float(ax @ ax)
                    d = q0 - w0
                    S.append(d @ ax / L2 if L2 > 0 else np.full(len(q0), np.nan))
                    T.append((ax[0] * d[:, 1] - ax[1] * d[:, 0]) / L2 if L2 > 0 else np.full(len(q0), np.nan))
        if Hs is None:
            breaks += 1
            broken = True
            Hc[cur] = None
        else:
            Hc[cur] = Hs @ Hc[prev]
        g0 = g1
    cat = lambda X: np.concatenate(X).astype(np.float32) if X else np.zeros(0, np.float32)      # noqa: E731
    return {"H": Hc, "breaks": breaks, "inl": inls, "n_ok": n_ok, "s": cat(S), "t": cat(T), "res": cat(R)}


# ----------------------------------------------------------------------------------------------------
# 80: 구간마다 추적 다시 시작
# ----------------------------------------------------------------------------------------------------
def rt_windows(a: int, b: int, present: np.ndarray, win: int) -> list:
    """a..b 를 win 프레임 구간 [lo, hi] 로 나눈다(마지막 구간은 b 에서 끝남). 구간마다 앵커 = 구간 가운데에 가장 가까운 검출 프레임 (검출 프레임이 없는 구간은 뺌).
    반환: [(앵커, lo, hi), ...]. 앵커에서 앞뒤로 kt_chain 을 돌려 그 구간만 추적한다."""
    out = []
    for lo in range(int(a), int(b) + 1, int(win)):
        hi = min(lo + int(win) - 1, int(b))
        det = np.flatnonzero(present[lo:hi + 1]) + lo
        if len(det):
            out.append((int(det[np.argmin(np.abs(det - (lo + hi) / 2))]), lo, hi))
    return out


# ----------------------------------------------------------------------------------------------------
# 83: 균형 사슬, 손목 따로 추적
# ----------------------------------------------------------------------------------------------------
def bc_chain(pngs: list, kps: np.ndarray, boxes: np.ndarray, order: list, kcfg: dict, body_joints) -> dict:
    """특징점을 손 몸통(body_joints 의 볼록 껍질, 손 마스크와 같이 넓힘, 손 마스크 안)과 나머지 손 마스크(손가락)에서 max_corners // 2 개씩 따로 뽑는 사슬.
    영역마다 따로 뽑으므로 quality 문턱도 그 영역의 최고 점수 기준이다. 그 뒤(LK, 전후방 검사, MAGSAC)는 kt_chain 과 같다.
    반환: kt_chain 의 dict + 단계별 몸통/손가락 특징점 수 (n_body, n_finger)."""
    lk = _lk(kcfg)
    bj = [int(j) for j in body_joints]
    half = max(1, int(kcfg["max_corners"]) // 2)
    Hc = {order[0]: np.eye(3)}
    inls, n_body, n_finger = [], [], []
    breaks, broken = 0, False
    gray = lambda i: cv2.cvtColor(cv2.imread(pngs[i], cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)      # noqa: E731
    g0 = gray(order[0])
    for prev, cur in zip(order[:-1], order[1:]):
        g1 = gray(cur)
        if broken:
            Hc[cur] = None
            g0 = g1
            continue
        m = kt_hand_mask(kps[prev], boxes[prev], g0.shape, kcfg["dilate_frac"])
        mb = cv2.bitwise_and(kt_hand_mask(kps[prev][bj], boxes[prev], g0.shape, kcfg["dilate_frac"]), m)
        mf = cv2.bitwise_and(m, cv2.bitwise_not(mb))
        cs = [cv2.goodFeaturesToTrack(g0, half, float(kcfg["quality"]), float(kcfg["min_distance"]), mask=mm, blockSize=int(kcfg["block"])) if mm.any() else None for mm in (mb, mf)]
        n_body.append(0 if cs[0] is None else len(cs[0]))
        n_finger.append(0 if cs[1] is None else len(cs[1]))
        cs = [c.reshape(-1, 2) for c in cs if c is not None]
        Hs = None
        if cs and sum(len(c) for c in cs) >= 8:
            p0 = np.concatenate(cs).astype(np.float32)
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
    return {"H": Hc, "breaks": breaks, "inl": inls, "n_body": n_body, "n_finger": n_finger}


def wt_track(pngs: list, boxes: np.ndarray, order: list, kcfg: dict, p0, radius_frac: float, min_inliers: int) -> dict:
    """손목만 따로 옮긴다. 단계마다 현재 손목 주변 원(반지름 = 손 상자 긴 변 x radius_frac, 손 마스크 없음 - 팔뚝도 포함) 안의 특징점을
    kt_chain 과 같은 LK·전후방 검사로 추적하고, 닮음 변환(estimateAffinePartial2D, RANSAC, threshold_px)을 구해 손목에 적용한다.
    inlier 가 min_inliers 미만이면 끊긴다(그 뒤 프레임은 None). p0: order[0] 의 손목 위치. 반환: {p: {프레임: (2,) 또는 None}, breaks, inl}."""
    lk = _lk(kcfg)
    P = {order[0]: np.asarray(p0, float)}
    inls, breaks, broken = [], 0, False
    gray = lambda i: cv2.cvtColor(cv2.imread(pngs[i], cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)      # noqa: E731
    g0 = gray(order[0])
    for prev, cur in zip(order[:-1], order[1:]):
        g1 = gray(cur)
        if broken:
            P[cur] = None
            g0 = g1
            continue
        w = P[prev]
        r = max(2, int(round(float(radius_frac) * float(box_long(np.asarray(boxes[prev], float)[None])[0]))))
        m = np.zeros(g0.shape, np.uint8)
        cv2.circle(m, (int(round(w[0])), int(round(w[1]))), r, 255, -1)
        c = cv2.goodFeaturesToTrack(g0, int(kcfg["max_corners"]), float(kcfg["quality"]), float(kcfg["min_distance"]), mask=m, blockSize=int(kcfg["block"])) if m.any() else None
        A = None
        if c is not None and len(c) >= 3:
            q0 = c.reshape(-1, 2).astype(np.float32)
            q1, s1, _ = cv2.calcOpticalFlowPyrLK(g0, g1, q0.reshape(-1, 1, 2), None, **lk)
            qb, s2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, q1, None, **lk)
            ok = (s1.ravel() == 1) & (s2.ravel() == 1) & (np.linalg.norm(qb.reshape(-1, 2) - q0, axis=1) < float(kcfg["fb_px"]))
            if ok.sum() >= int(min_inliers):
                A, inl = cv2.estimateAffinePartial2D(q0[ok], q1.reshape(-1, 2)[ok], method=cv2.RANSAC, ransacReprojThreshold=float(kcfg["threshold_px"]))
                if A is None or inl is None or int(inl.sum()) < int(min_inliers):
                    A = None
                else:
                    inls.append(int(inl.sum()))
        if A is None:
            breaks += 1
            broken = True
            P[cur] = None
        else:
            P[cur] = A[:, :2] @ w + A[:, 2]
        g0 = g1
    return {"p": P, "breaks": breaks, "inl": inls}


# ----------------------------------------------------------------------------------------------------
# 86: 한 단계의 중간 결과 (그림용)
# ----------------------------------------------------------------------------------------------------
def kv_step(g0: np.ndarray, g1: np.ndarray, kp_prev: np.ndarray, box_prev: np.ndarray, kcfg: dict) -> dict:
    """kt_chain 의 한 단계(g0 -> g1, 회색 영상)를 같은 호출·설정으로 하고 중간 결과를 모두 돌려준다:
    손 마스크, 마스크 안 Shi-Tomasi 점수(cornerMinEigenVal), 특징점, 정방향·역방향 LK, 전후방 오차와 통과 여부, MAGSAC 호모그래피와 inlier, 잔차."""
    lk = _lk(kcfg)
    m = kt_hand_mask(kp_prev, box_prev, g0.shape, kcfg["dilate_frac"])
    score = cv2.cornerMinEigenVal(g0, int(kcfg["block"]), 3)
    score[m == 0] = 0.0
    c = cv2.goodFeaturesToTrack(g0, int(kcfg["max_corners"]), float(kcfg["quality"]), float(kcfg["min_distance"]), mask=m, blockSize=int(kcfg["block"]))
    p0 = c.reshape(-1, 2).astype(np.float32)
    p1, s1, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0.reshape(-1, 1, 2), None, **lk)
    pb, s2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, **lk)
    p1, pb = p1.reshape(-1, 2), pb.reshape(-1, 2)
    fb = np.linalg.norm(pb - p0, axis=1)
    ok = (s1.ravel() == 1) & (s2.ravel() == 1) & (fb < float(kcfg["fb_px"]))
    Hs, inl = cv2.findHomography(p0[ok].astype(np.float64), p1[ok].astype(np.float64), cv2.USAC_MAGSAC, float(kcfg["threshold_px"]))
    res = np.linalg.norm(kt_map(p0[ok], Hs) - p1[ok], axis=1)
    return {"mask": m, "score": score, "score_max": float(score.max()), "p0": p0, "p1": p1, "fb": fb, "ok": ok, "H": Hs, "inl": inl.ravel() > 0, "res": res}


# ----------------------------------------------------------------------------------------------------
# 87: 종이와 맞닿은 윤곽(경계) 특징점
# ----------------------------------------------------------------------------------------------------
def eg_paper(img_bgr: np.ndarray, s_max: int, v_min: int) -> np.ndarray:
    """흰 종이로 보이는 픽셀(HSV 채도 <= s_max, 명도 >= v_min, OpenCV 0-255 눈금)을 0/1 uint8 로."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return ((hsv[:, :, 1] <= int(s_max)) & (hsv[:, :, 2] >= int(v_min))).astype(np.uint8)


def eg_window_frac(binary: np.ndarray, pts: np.ndarray, win: int) -> np.ndarray:
    """각 점 주변 win x win 창(화면 밖은 잘라냄) 안에서 binary 가 1 인 픽셀의 비율."""
    ii = cv2.integral(binary)
    h, w = binary.shape
    r = int(win) // 2
    x = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    x0, x1, y0, y1 = np.clip(x - r, 0, w), np.clip(x + r + 1, 0, w), np.clip(y - r, 0, h), np.clip(y + r + 1, 0, h)
    s = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
    return s / np.maximum(1, (x1 - x0) * (y1 - y0))


def eg_chain(pngs: list, kps: np.ndarray, boxes: np.ndarray, order: list, kcfg: dict, ecfg: dict, exclude: bool) -> dict:
    """kt_chain 과 같은 단계 + 특징점마다 LK 창 안 종이 비율(이전 프레임 기준). 비율 >= ecfg.edge_frac 이면 경계 점.
    exclude=True 면 경계 점을 LK 전에 뺀다(나머지는 그대로). 반환: kt_chain 의 dict + 단계별 특징점 수, 경계 점 수, inlier 중 경계 점 수."""
    lk = _lk(kcfg)
    Hc = {order[0]: np.eye(3)}
    inls, n_c, n_e, n_ei = [], [], [], []
    breaks, broken = 0, False
    im0 = cv2.imread(pngs[order[0]], cv2.IMREAD_COLOR)
    g0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
    for prev, cur in zip(order[:-1], order[1:]):
        im1 = cv2.imread(pngs[cur], cv2.IMREAD_COLOR)
        g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        if broken:
            Hc[cur] = None
            im0, g0 = im1, g1
            continue
        m = kt_hand_mask(kps[prev], boxes[prev], g0.shape, kcfg["dilate_frac"])
        c = cv2.goodFeaturesToTrack(g0, int(kcfg["max_corners"]), float(kcfg["quality"]), float(kcfg["min_distance"]), mask=m, blockSize=int(kcfg["block"]))
        Hs = None
        if c is not None:
            p0 = c.reshape(-1, 2).astype(np.float32)
            edge = eg_window_frac(eg_paper(im0, ecfg["paper_s_max"], ecfg["paper_v_min"]), p0, int(kcfg["win"])) >= float(ecfg["edge_frac"])
            n_c.append(int(len(p0)))
            n_e.append(int(edge.sum()))
            if exclude:
                p0, edge = p0[~edge], edge[~edge]
            if len(p0) >= 8:
                p1, s1, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0.reshape(-1, 1, 2), None, **lk)
                pb, s2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, **lk)
                ok = (s1.ravel() == 1) & (s2.ravel() == 1) & (np.linalg.norm(pb.reshape(-1, 2) - p0, axis=1) < float(kcfg["fb_px"]))
                if ok.sum() >= int(kcfg["min_inliers"]):
                    Hs, inl = cv2.findHomography(p0[ok].astype(np.float64), p1.reshape(-1, 2)[ok].astype(np.float64), cv2.USAC_MAGSAC, float(kcfg["threshold_px"]))
                    if Hs is None or inl is None or int(inl.sum()) < int(kcfg["min_inliers"]):
                        Hs = None
                    else:
                        inls.append(int(inl.sum()))
                        n_ei.append(int(edge[ok][inl.ravel() > 0].sum()))
        if Hs is None:
            breaks += 1
            broken = True
            Hc[cur] = None
        else:
            Hc[cur] = Hs @ Hc[prev]
        im0, g0 = im1, g1
    return {"H": Hc, "breaks": breaks, "inl": inls, "n_corners": n_c, "n_edge": n_e, "n_edge_inl": n_ei}
