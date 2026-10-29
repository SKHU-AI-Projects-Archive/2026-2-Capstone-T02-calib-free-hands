"""초점 거리(f) 추정 알고리즘만 모은 파일: 손 추적 -> 정면 템플릿 -> 기울기 -> f 추정 문제와 비용 -> f 선택 (fixed_template, joint_delta, 그리고 교수님의 free_template, free_shape3d).

함수 본문은 이전 handfocal.py와 같다(주석만 더했고, 비교해서 확인했다). 영상 변환, 검출 결과 읽기, 불확실도, 판정, 각 실험 분석, 결과 출력과 보고서 작성은 handfocal_report.py에 있고,
이 파일은 그쪽 코드를 가져오지 않는다(이 파일만으로 import 된다). analyze()(handfocal_report.py)가 아래 함수들을 다음 순서로 부른다:

    track -> front_window -> template_build -> _theta(affine_tilt) -> auto_transition / tilt_indices
          -> Problem -> f_grid -> estimate_run(fixed_template 또는 joint_delta) -> cost(K_of, frame_fit, _huber) -> _parabola

f 값은 fixed_template(기본) 또는 joint_delta가 결정하고, 그 뒤의 진단, 부트스트랩, 판정은 f 값을 바꾸지 않는다. free_template과 free_shape3d는 실험 40~43에서 같은 점 추적 위에 따로 적용했다.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import minimize


# ====================================================================================================
# 1. 입력 준비: 손 추적, 정면 템플릿, 기울기
# ====================================================================================================

# 박스의 긴 변 길이(px). 잔차와 Huber 폭을 손 크기로 정규화할 때 쓴다.
def box_long(box):
    return np.maximum(box[..., 2] - box[..., 0], box[..., 3] - box[..., 1])


# 목표 손 하나를 영상 전체에서 따라간다.
# 1) 정면 구간 시작 후 init_window_s 안에서 (박스 넓이 x 신뢰도)가 가장 큰 후보를 목표 손으로 정한다.
# 2) 그 프레임에서 앞으로, 이어서 뒤로 IoU가 가장 큰 후보를 이어 붙인다(iou_min 미만이면 그 프레임은 건너뜀).
# 3) 신뢰도와 이미지 안쪽 조건을 만족하는 관절만 mask=True로 표시한다.
def track(det: dict, times: np.ndarray, size_wh, tcfg: dict, front_start_s: float = 0.0) -> dict:
    """Target = the candidate with the largest box area x confidence in the first init_window_s of the front phase; then IoU tracking."""
    n = len(times)
    W, H = size_wh
    init = [i for i in range(n) if front_start_s <= times[i] < front_start_s + tcfg["init_window_s"]]
    best, best_i = None, None
    for i in init:
        for k in candidates(det, i):
            b = det["boxes"][k]
            s = (b[2] - b[0]) * (b[3] - b[1]) * det["box_conf"][k]
            if best is None or s > best[0]:
                best, best_i = (s, k), i
    out = {"present": np.zeros(n, bool), "box": np.full((n, 4), np.nan), "cls": np.full(n, -1), "box_conf": np.zeros(n),
           "kp": np.full((n, 21, 2), np.nan), "kconf": np.zeros((n, 21)), "mask": np.zeros((n, 21), bool), "n_candidates": np.zeros(n, int),
           "init_frame": best_i}
    for i in range(n):
        out["n_candidates"][i] = len(candidates(det, i))
    if best is None:
        return out
    ref = det["boxes"][best[1]].copy()
    order = list(range(best_i, n)) + list(range(best_i - 1, -1, -1))     # forward from the init frame, then backward
    last_fwd, last_bwd = ref.copy(), ref.copy()
    for i in order:
        refb = last_fwd if i >= best_i else last_bwd
        cs = candidates(det, i)
        if not cs:
            continue
        ious = [iou(refb, det["boxes"][k]) for k in cs]
        j = int(np.argmax(ious))
        if i == best_i:
            k = best[1]
        elif ious[j] >= tcfg["iou_min"]:
            k = cs[j]
        else:
            continue
        out["present"][i] = True
        out["box"][i], out["cls"][i], out["box_conf"][i] = det["boxes"][k], det["cls"][k], det["box_conf"][k]
        out["kp"][i], out["kconf"][i] = det["kxy"][k], det["kconf"][k]
        if i >= best_i:
            last_fwd = det["boxes"][k].copy()
        else:
            last_bwd = det["boxes"][k].copy()
    kp = out["kp"]
    inside = (kp[..., 0] >= 0) & (kp[..., 0] < W) & (kp[..., 1] >= 0) & (kp[..., 1] < H)
    out["mask"] = out["present"][:, None] & (out["kconf"] >= tcfg["conf_min"]) & inside
    if tcfg.get("smoothing", {}).get("enabled"):
        out["kp"] = smooth(out["kp"], out["mask"], int(tcfg["smoothing"]["window"]))
    return out


# src를 dst에 맞추는 최소제곱 닮음변환(스케일, 회전, 이동)을 구한다: dst ~ s R src + t (반사 없음).
def similarity_align(src: np.ndarray, dst: np.ndarray):
    """Least-squares similarity src -> dst (2D). Returns (s, R, t) with dst ~ s R src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a)
    D = np.diag([1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = float((S * np.diag(D)).sum() / max((a ** 2).sum(), 1e-12))
    return s, R, mu_d - s * R @ mu_s


# 정면 프레임들의 관절 좌표를 서로 닮음변환으로 겹쳐(Procrustes 평균) 평평한 손의 2D 템플릿 하나를 만든다.
# 마지막에 템플릿 크기를 창 안 실제 손 크기(px)에 맞추고, 프레임별 잔차(지터)를 함께 돌려준다.
def template_build(front_pts: np.ndarray, iters: int = 4) -> dict:
    """front_pts (N, J, 2) of frames with every joint visible -> template (J, 2), centred, in the window's pixel scale."""
    X = front_pts.astype(float)
    T = X[0] - X[0].mean(0)
    for _ in range(iters):
        aligned = []
        for P in X:
            s, R, t = similarity_align(P, T)
            aligned.append((s * (R @ P.T)).T + t)
        A = np.array(aligned)
        T = A.mean(0)
        T = T - T.mean(0)
    # scale = mean pixel size of the window
    sizes = [np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean()) for P in X]
    T = T * (np.mean(sizes) / np.sqrt((T ** 2).sum(1).mean()))
    res = []
    for P in X:                                     # residuals in each frame's own pixels
        s, R, t = similarity_align(T, P)
        res.append(np.linalg.norm((s * (R @ T.T)).T + t - P, axis=1))
    res = np.array(res)
    return {"template": T, "jitter_rms_px": float(np.sqrt((res ** 2).mean())), "n_frames": int(len(X))}


# 템플릿 T에서 관측 P로 가는 아핀 변환의 선형부를 SVD로 나눠 기울기를 구한다.
# theta2d = arccos(작은 특이값 / 큰 특이값): 약투시 근사에서 손바닥이 기울어진 각도. axis는 눌려 보이는 방향(0~180도).
def affine_tilt(T: np.ndarray, P: np.ndarray):
    """Least-squares affine T -> P; theta2d = arccos(s2 / s1) of its linear part; axis = direction of the compressed image axis (deg, 0-180)."""
    A = np.c_[T, np.ones(len(T))]
    M, *_ = np.linalg.lstsq(A, P, rcond=None)
    L = M[:2].T                                       # 2x2 linear part, P ~ L T + t
    U, S, Vt = np.linalg.svd(L)
    ratio = float(np.clip(S[1] / S[0], 0.0, 1.0)) if S[0] > 0 else 0.0
    theta = float(np.degrees(np.arccos(ratio)))
    axis = float(np.degrees(np.arctan2(U[1, 1], U[0, 1])) % 180.0)
    return theta, axis


# 정면 구간에서 손이 가장 안 움직이는 연속 창(front_window_s 길이)을 찾는다.
# 검출률이 기준 이상인 창을 우선하고, 그 안에서 프레임 간 관절 이동(손 박스 대비)이 가장 작은 창을 고른다.
def front_window(times, present, kp, mask, box, pcfg) -> dict:
    """Lowest-motion contiguous window of front_window_s inside [front_skip_s, front_phase_s) with detection rate >= the minimum."""
    t = np.asarray(times)
    cand = np.flatnonzero((t >= pcfg["front_skip_s"]) & (t < pcfg["front_phase_s"]))
    bl = box_long(box)
    speed = np.full(len(t), np.nan)
    for i in range(1, len(t)):
        if present[i] and present[i - 1]:
            m = mask[i] & mask[i - 1]
            if m.sum() >= 4:
                speed[i] = float(np.median(np.linalg.norm(kp[i, m] - kp[i - 1, m], axis=-1)) / max(np.nanmean([bl[i], bl[i - 1]]), 1e-9))
    best, warn = None, []
    for s in cand:
        idx = cand[(t[cand] >= t[s]) & (t[cand] < t[s] + pcfg["front_window_s"] - 1e-9)]
        if t[s] + pcfg["front_window_s"] > pcfg["front_phase_s"] + 1e-9:
            break
        rate = float(present[idx].mean()) if len(idx) else 0.0
        sp = float(np.nanmean(speed[idx])) if np.isfinite(speed[idx]).any() else np.inf
        key = (rate < pcfg["front_min_detect_rate"], sp)
        if best is None or key < best[0]:
            best = (key, idx, rate, sp)
    if best is None:
        return {"idx": np.zeros(0, int), "detect_rate": 0.0, "speed_rel": float("nan"), "warnings": ["정면 구간에 창을 만들 프레임이 없습니다"]}
    if best[2] < pcfg["front_min_detect_rate"]:
        warn.append(f"정면 창의 검출률이 낮습니다({best[2]:.2f})")
    return {"idx": best[1], "detect_rate": best[2], "speed_rel": best[3], "t0": float(t[best[1][0]]), "t1": float(t[best[1][-1]]), "warnings": warn}


# theta2d가 auto_theta_deg 이상인 상태가 auto_sustain_s 동안 이어지는 첫 시각(정면에서 기울임으로 바뀌는 때).
def auto_transition(times, theta, present, pcfg):
    """First time at which theta2d stays >= auto_theta_deg for auto_sustain_s."""
    t = np.asarray(times)
    ok = present & np.isfinite(theta) & (theta >= pcfg["auto_theta_deg"])
    for i in range(len(t)):
        if not ok[i]:
            continue
        j = np.flatnonzero((t >= t[i]) & (t < t[i] + pcfg["auto_sustain_s"]))
        if len(j) and ok[j].all():
            return float(t[i])
    return None


# 기울임 구간(t_start 이후)에서 손이 검출된 프레임의 인덱스.
def tilt_indices(times, present, t_start) -> np.ndarray:
    t = np.asarray(times)
    return np.flatnonzero((t >= t_start) & present)


def _theta(T, kp, mask, present, ids):
    n = len(kp)
    th, ax = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        if not present[i]:
            continue
        v = mask[i, ids]
        if v.sum() >= 6:
            th[i], ax[i] = affine_tilt(T[v], kp[i, ids][v])
    return th, ax


def principal_point(ecfg, W, H):
    pp = ecfg.get("principal_point", "center")
    return (W / 2.0, H / 2.0) if pp == "center" else (float(pp[0]), float(pp[1]))


def undistort(kp, ecfg, W, H):
    """Optional distortion [k1, k2, p1, p2] with the convention K_d = [[L, 0, cx], [0, L, cy]] (L = long side), applied once."""
    d = ecfg.get("distortion")
    if not d:
        return kp
    cx, cy = principal_point(ecfg, W, H)
    L = float(max(W, H))
    Kd = np.array([[L, 0, cx], [0, L, cy], [0, 0, 1.0]])
    flat = kp.reshape(-1, 1, 2).astype(np.float64)
    ok = np.isfinite(flat).all(-1).ravel()
    out = flat.copy()
    out[ok] = cv2.undistortPoints(flat[ok], Kd, np.asarray(d, float), P=Kd)
    return out.reshape(kp.shape)


# ====================================================================================================
# 2. f 추정 문제와 비용 (후보 f마다 평면 템플릿의 재투영 오차)
# ====================================================================================================

# f 추정 문제 하나: 템플릿, 프레임별 관측 점(보이는 점 표시 포함), 주점, 프레임별 Huber 폭을 묶어 둔다.
class Problem:
    def __init__(self, template: np.ndarray, frames: list, cx: float, cy: float, huber_px: np.ndarray, min_points: int = 4, weights: np.ndarray | None = None):
        """template (J, 2); frames: list of (obs (J, 2), visible (J,)) ; huber_px: per frame Huber width; weights (J,): per-joint weight of the
        reprojection cost (21's wrist_mode downweight); None = every joint counts the same (17's behaviour)."""
        self.w = None if weights is None else np.asarray(weights, float)
        self.T = np.asarray(template, float)
        self.frames = [(np.asarray(o, float), np.asarray(v, bool)) for o, v in frames]
        self.keep = [i for i, (o, v) in enumerate(self.frames) if v.sum() >= max(4, min_points)]
        self.cx, self.cy = float(cx), float(cy)
        self.huber = np.asarray(huber_px, float)
        self.n_points = int(sum(self.frames[i][1].sum() for i in self.keep))

    def subset(self, idx):
        p = Problem.__new__(Problem)
        p.T, p.frames, p.cx, p.cy, p.huber, p.w = self.T, self.frames, self.cx, self.cy, self.huber, self.w
        p.keep = [i for i in idx if i in set(self.keep)]
        p.n_points = int(sum(self.frames[i][1].sum() for i in p.keep))
        return p


# (joint_delta용) 정면 손이 방향 phi로 delta만큼 기울어 있었다고 보고 템플릿을 1/cos(delta)만큼 늘린다(1차 약투시 보정).
def stretch(T: np.ndarray, delta_deg: float, phi_deg: float) -> np.ndarray:
    u = np.array([np.cos(np.radians(phi_deg)), np.sin(np.radians(phi_deg))])
    k = 1.0 / np.cos(np.radians(min(abs(delta_deg), 80.0))) - 1.0
    return T + k * (T @ u)[:, None] * u[None, :]


# Huber 손실: 잔차 r이 d 이하면 r^2, 크면 선형으로 늘어나 이상치의 영향을 줄인다.
def _huber(r, d):
    return np.where(r <= d, r * r, 2 * d * r - d * d)


# 한 프레임에서 평면 템플릿(z=0)의 포즈를 보이는 점들로 구한다(OpenCV IPPE PnP).
# IPPE의 두 해 중 재투영 오차가 작은 쪽을 쓰고, 점별 재투영 잔차(px)와 R, t를 돌려준다.
def frame_fit(T2: np.ndarray, obs: np.ndarray, vis: np.ndarray, K: np.ndarray):
    """IPPE pose of the planar template on the visible points; the lower-error of the two solutions. Returns (residuals, R, t)."""
    obj = np.c_[T2[vis], np.zeros(vis.sum())].astype(np.float64)
    img = obs[vis].astype(np.float64)
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
    k = int(np.argmin(np.asarray(errs).ravel())) if n > 1 else 0
    R, _ = cv2.Rodrigues(rvecs[k])
    t = tvecs[k].ravel()
    X = obj @ R.T + t
    proj = K[:2, :2] @ (X[:, :2] / X[:, 2:3]).T + K[:2, 2:3]
    return np.linalg.norm(proj.T - img, axis=1), R, t


# 핀홀 카메라 행렬: 초점거리 f, 주점 (cx, cy), 정사각 픽셀, 왜곡 없음.
def K_of(f, cx, cy):
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])


# 후보 f에서의 비용: 프레임마다 IPPE 포즈를 구하고 점별 Huber 재투영 손실의 평균을 낸 뒤,
# 프레임 평균을 다시 평균한다. f가 맞을수록 기울인 프레임의 템플릿도 점들과 잘 맞아 비용이 작아진다.
def cost(pb: Problem, f: float, T2: np.ndarray | None = None, details: bool = False):
    T2 = pb.T if T2 is None else T2
    K = K_of(f, pb.cx, pb.cy)
    per, out = [], []
    for i in pb.keep:
        o, v = pb.frames[i]
        r, R, t = frame_fit(T2, o, v, K)
        per.append(float(_huber(r, pb.huber[i]).mean()) if pb.w is None else float((pb.w[v] * _huber(r, pb.huber[i])).sum() / pb.w[v].sum()))
        if details:
            out.append({"frame": i, "res": r, "vis": v, "R": R, "t": t})
    c = float(np.mean(per)) if per else float("inf")
    return (c, out) if details else c


# ====================================================================================================
# 3. f 선택: 후보 격자, 포물선 보간, 추정 방식
# ====================================================================================================

# 후보 f 격자: 긴 변의 lo~hi배 사이를 로그 간격으로 f_grid_n개.
def f_grid(cfg_e: dict, long_side: float) -> np.ndarray:
    lo, hi = cfg_e["f_ratio_range"]
    return np.geomspace(lo * long_side, hi * long_side, int(cfg_e["f_grid_n"]))


# 격자 최솟값(k) 주변 세 점에 포물선을 맞춰 로그 f에서 최솟값 위치를 보간한다(격자 가장자리면 그 점 그대로).
def _parabola(lf: np.ndarray, c: np.ndarray, k: int) -> float:
    if k == 0 or k == len(lf) - 1:
        return float(lf[k])
    x, y = lf[k - 1:k + 2], c[k - 1:k + 2]
    A = np.vstack([x ** 2, x, np.ones(3)]).T
    a, b, _ = np.linalg.solve(A, y)
    return float(-b / (2 * a)) if a > 0 else float(lf[k])


# 기본 추정 방식: 템플릿은 정면 손의 진짜 평면 모양이라고 고정하고, 격자의 모든 f에서 비용을 계산해
# 최솟값을 고른 뒤 포물선 보간으로 다듬는다. 비용 곡선(profile)과 격자 가장자리 여부도 함께 돌려준다.
def fixed_template(pb: Problem, grid: np.ndarray) -> dict:
    c = np.array([cost(pb, f) for f in grid])
    k = int(np.argmin(c))
    lf = _parabola(np.log(grid), c, k)
    f = float(np.exp(np.clip(lf, np.log(grid[0]), np.log(grid[-1]))))
    return {"f": f, "delta_deg": 0.0, "phi_deg": 0.0, "profile_f": grid.tolist(), "profile_cost": c.tolist(), "cost": cost(pb, f),
            "at_grid_edge": bool(k in (0, len(grid) - 1))}


# 변형 방식: 정면 손도 사실 조금 기울었다고 보고 (f, 기울기 delta, 방향 phi)를 함께 추정한다.
# 먼저 (delta, phi) 격자 x f 격자에서 최선의 시작점을 찾고, Nelder-Mead로 다듬는다.
# 목적함수 = N x 평균비용 / sigma^2 + (delta / delta_prior)^2 (delta를 작게 두는 약한 사전 확률).
def joint_delta(pb: Problem, grid: np.ndarray, jd: dict, sigma2: float, start=None) -> dict:
    """(f, delta, phi): grid then Nelder-Mead on the total = N * mean cost / sigma^2 + (delta / delta_prior)^2."""
    N = max(pb.n_points, 1)
    dp = float(jd["delta_prior_deg"])

    def total(lf, d, ph):
        return N * cost(pb, float(np.exp(lf)), stretch(pb.T, d, ph)) / sigma2 + (d / dp) ** 2

    phis = np.linspace(0, 180, int(jd["phi_n"]), endpoint=False)
    prof = np.full(len(grid), np.inf)
    best = (np.inf, None)
    if start is None:
        for d in jd["delta_grid_deg"]:
            for ph in (phis if d > 0 else [0.0]):
                T2 = stretch(pb.T, d, ph)
                for k, f in enumerate(grid):
                    v = N * cost(pb, f, T2) / sigma2 + (d / dp) ** 2
                    prof[k] = min(prof[k], v)
                    if v < best[0]:
                        best = (v, (np.log(f), float(d), float(ph)))
        x0 = best[1]
    else:
        x0 = start
    lo, hi = np.log(grid[0]), np.log(grid[-1])
    fun = lambda x: total(np.clip(x[0], lo, hi), abs(x[1]), x[2] % 180.0)   # noqa: E731
    r = minimize(fun, np.array(x0, float), method="Nelder-Mead",
                 options={"maxiter": int(jd["nm_max_iter"]), "xatol": 1e-4, "fatol": 1e-6, "initial_simplex": None})
    lf, d, ph = float(np.clip(r.x[0], lo, hi)), abs(float(r.x[1])), float(r.x[2] % 180.0)
    f = float(np.exp(lf))
    return {"f": f, "delta_deg": d, "phi_deg": ph, "profile_f": grid.tolist(), "profile_cost": (prof * sigma2 / N).tolist() if start is None else None,
            "cost": cost(pb, f, stretch(pb.T, d, ph)), "objective": float(r.fun), "nm_iters": int(r.nit), "at_grid_edge": bool(lf <= lo + 1e-6 or lf >= hi - 1e-6)}


# 추정 방식(fixed_template 또는 joint_delta)에 따라 알맞은 함수를 부른다.
def estimate_run(pb: Problem, method: str, grid: np.ndarray, ecfg: dict, sigma2: float, start=None) -> dict:
    if method == "fixed_template":
        return fixed_template(pb, grid)
    if method == "joint_delta":
        return joint_delta(pb, grid, ecfg["joint_delta"], sigma2, start)
    raise ValueError(method)


# ====================================================================================================
# 4. 손 추적에 쓰는 보조 함수 (track이 부른다)
# ====================================================================================================


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ar = lambda r: max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])   # noqa: E731
    return inter / max(ar(a) + ar(b) - inter, 1e-9)


def smooth(kp, mask, w):
    """Optional centred moving average over visible points (off by default)."""
    out = kp.copy()
    h = w // 2
    for i in range(len(kp)):
        lo, hi = max(0, i - h), min(len(kp), i + h + 1)
        for j in range(21):
            m = mask[lo:hi, j]
            if mask[i, j] and m.sum() > 1:
                out[i, j] = kp[lo:hi, j][m].mean(0)
    return out


def candidates(det: dict, i: int):
    a, b = int(det["offsets"][i]), int(det["offsets"][i + 1])
    return list(range(a, b))


# ====================================================================================================
# 5. 교수님이 보내주신 free_template / free_shape3d: 템플릿도 f와 함께 추정하는 방식 (평면 템플릿 -> 점마다 작은 깊이 변위 허용)
#    교수님 코드를 그대로 쓰되 실행되지 않던 곳만 고쳤다(np.c -> np.c_, _backproject -> backproject, rodrigues -> _rodrigues, np.r -> np.r_,
#    lm_schur -> _lm_schur, Problem.new -> Problem.__new__). ft_estimate_run과 s3_estimate_run은 교수님이 보내주신 estimate_run을 기존 estimate_run을 건드리지 않고 따로 둔 것이다.
# ====================================================================================================


def procrustes_init(pb, iters=6):
    J = pb.T.shape[0]
    frames = [pb.frames[i] for i in pb.keep]
    ref = max(range(len(frames)), key=lambda k: frames[k][1].sum())
    o, v = frames[ref]
    T = np.where(v[:, None], o, np.nan)
    T = T - np.nanmean(T, 0)
    for _ in range(iters):
        acc = np.zeros((J, 2))
        cnt = np.zeros(J)
        for o, v in frames:
            m = v & np.isfinite(T).all(1)
            if m.sum() < 3:
                continue
            s, R, t = similarity_align(o[m], T[m])
            acc[v] += (s * (R @ o[v].T)).T + t
            cnt[v] += 1
        new = np.where(cnt[:, None] > 0, acc / np.maximum(cnt, 1)[:, None], np.nan)
        new = new - np.nanmean(new, 0)
        T = new
    sizes = [np.sqrt(((o[v] - o[v].mean(0)) ** 2).sum(1).mean()) for o, v in frames]
    T = T * (np.mean(sizes) / np.sqrt(np.nansum(T ** 2, 1)[np.isfinite(T).all(1)].mean()))
    return T


def backproject(obs, R, t, K):
    H = K @ np.c_[R[:, 0], R[:, 1], t]
    p = np.c_[obs, np.ones(len(obs))] @ np.linalg.inv(H).T
    return p[:, :2] / p[:, 2:3]


def _gauge(T, T0):
    T = T - T.mean(0)
    T = T * (np.sqrt((T0 ** 2).sum(1).mean()) / np.sqrt((T ** 2).sum(1).mean()))
    _, R, _ = similarity_align(T, T0)
    return (R @ T.T).T


def alt_template(pb, f, T0, iters=5):
    K = K_of(f, pb.cx, pb.cy)
    T = T0.copy()
    for _ in range(iters):
        acc = [[] for _ in range(len(T))]
        for i in pb.keep:
            o, v = pb.frames[i]
            _, R, t = frame_fit(T, o, v, K)
            q = backproject(o[v], R, t, K)
            for j, p in zip(np.flatnonzero(v), q):
                acc[j].append(p)
        new = np.array([np.median(np.array(a), axis=0) if a else T[j] for j, a in enumerate(acc)])
        T = _gauge(new, T0)
    return T


def _rodrigues(rv):
    th = np.linalg.norm(rv, axis=1)
    k = rv / np.maximum(th, 1e-12)[:, None]
    c, s = np.cos(th), np.sin(th)
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    Kx = np.zeros((len(rv), 3, 3))
    Kx[:, 0, 1], Kx[:, 0, 2] = -kz, ky
    Kx[:, 1, 0], Kx[:, 1, 2] = kz, -kx
    Kx[:, 2, 0], Kx[:, 2, 1] = -ky, kx
    kk = k[:, :, None] * k[:, None, :]
    return c[:, None, None] * np.eye(3) + (1 - c)[:, None, None] * kk + s[:, None, None] * Kx


def _anchors(T, frames):
    vis = np.sum([v for o, v in frames], axis=0)
    good = np.flatnonzero(vis >= 0.9 * vis.max())
    d = np.linalg.norm(T[good][:, None] - T[good][None], axis=-1)
    a, b = np.unravel_index(np.argmax(d), d.shape)
    return int(good[a]), int(good[b])


def _lm_schur(resid, x0, G, N, fi, gl_groups, lf_bounds, hp, max_iter=60, tol=1e-8):
    M = len(fi)
    fr2 = np.r_[fi, fi]
    order = np.argsort(fr2, kind="stable")
    fs = fr2[order]
    starts = np.r_[0, np.flatnonzero(np.diff(fs)) + 1]
    present = fs[starts]
    pose_cols = [G + 6 * fr2 + d for d in range(6)]
    i6 = np.arange(6)

    def robust(r):
        e = np.hypot(r[:M], r[M:2 * M])
        w = np.where(e <= hp, 1.0, hp / np.maximum(e, 1e-12))
        rho = np.where(e <= hp, e * e, 2 * hp * e - hp * hp)
        return w, float(rho.sum() + (r[2 * M:] ** 2).sum())

    def seg(A):
        out = np.zeros((N,) + A.shape[1:])
        out[present] = np.add.reduceat(A[order], starts, axis=0)
        return out

    def jacs(x, nr):
        Jg = np.zeros((nr, G))
        Jp = np.zeros((2 * M, 6))
        h = 1e-6 * np.maximum(1.0, np.abs(x))
        for cols, m in gl_groups:
            xp, xm = x.copy(), x.copy()
            xp[cols] += h[cols]
            xm[cols] -= h[cols]
            dr = resid(xp) - resid(xm)
            ok = np.flatnonzero(m >= 0)
            Jg[ok, m[ok]] = dr[ok] / (2 * h[m[ok]])
        for d in range(6):
            cols = G + 6 * np.arange(N) + d
            xp, xm = x.copy(), x.copy()
            xp[cols] += h[cols]
            xm[cols] -= h[cols]
            Jp[:, d] = (resid(xp) - resid(xm))[:2 * M] / (2 * h[pose_cols[d]])
        return Jg, Jp

    x = x0.copy()
    r = resid(x)
    w, c = robust(r)
    lam, it, success = 1e-3, 0, False
    for it in range(1, max_iter + 1):
        nr = len(r)
        Jg, Jp = jacs(x, nr)
        rw = np.r_[w, w, np.ones(nr - 2 * M)]
        Jgw = Jg * rw[:, None]
        U = Jg.T @ Jgw
        bg = -Jgw.T @ r
        Jpw = Jp * rw[:2 * M, None]
        V = seg(Jpw[:, :, None] * Jp[:, None, :])
        Wm = seg(Jgw[:2 * M, :, None] * Jp[:, None, :])
        bp = seg(-Jpw * r[:2 * M, None])
        improved = False
        for _ in range(12):
            Ud = U + lam * np.diag(np.diag(U) + 1e-9)
            Vd = V.copy()
            Vd[:, i6, i6] += lam * (V[:, i6, i6] + 1e-9)
            Vi = np.linalg.inv(Vd)
            WVi = Wm @ Vi
            S = Ud - np.einsum("ngk,nhk->gh", WVi, Wm)
            dg = np.linalg.solve(S, bg - np.einsum("ngk,nk->g", WVi, bp))
            dp = np.einsum("nij,nj->ni", Vi, bp - np.einsum("ngk,g->nk", Wm, dg))
            xn = x + np.r_[dg, dp.ravel()]
            xn[0] = np.clip(xn[0], lf_bounds[0], lf_bounds[1])
            rn = resid(xn)
            wn, cn = robust(rn)
            if np.isfinite(cn) and cn < c:
                improved = True
                break
            lam *= 4
        if not improved:
            success = True
            break
        rel = (c - cn) / max(c, 1e-12)
        x, r, w, c = xn, rn, wn, cn
        lam = max(lam / 3, 1e-12)
        if rel < tol:
            success = True
            break
    return x, r, it, success


def ba_refine(pb, f0, T0, f_bounds, huber_px=None, max_iter=60):
    J = T0.shape[0]
    frames = [pb.frames[i] for i in pb.keep]
    N = len(frames)
    K0 = K_of(f0, pb.cx, pb.cy)
    poses = np.zeros((N, 6))
    for n, (o, v) in enumerate(frames):
        _, R, t = frame_fit(T0, o, v, K0)
        poses[n, :3] = cv2.Rodrigues(R)[0].ravel()
        poses[n, 3:] = t
    fi = np.concatenate([np.full(v.sum(), n) for n, (o, v) in enumerate(frames)])
    ji = np.concatenate([np.flatnonzero(v) for o, v in frames])
    uv = np.concatenate([o[v] for o, v in frames])
    sw = np.ones(len(ji)) if pb.w is None else np.sqrt(pb.w[ji])
    a, b = _anchors(T0, frames)
    free = np.array([j for j in range(J) if j not in (a, b)])
    col = np.full(J, -1)
    col[free] = np.arange(len(free))
    nT = 2 * len(free)
    G = 1 + nT
    M = len(ji)

    def unpack(x):
        T = T0.copy()
        T[free] = x[1:1 + nT].reshape(-1, 2)
        return np.exp(x[0]), T, x[1 + nT:].reshape(N, 6)

    def resid(x):
        f, T, P = unpack(x)
        R = _rodrigues(P[:, :3])
        X = np.einsum("nab,nb->na", R[fi][:, :, :2], T[ji]) + P[fi, 3:]
        u = f * X[:, 0] / X[:, 2] + pb.cx - uv[:, 0]
        w = f * X[:, 1] / X[:, 2] + pb.cy - uv[:, 1]
        return np.r_[sw * u, sw * w]

    groups = [(np.array([0]), np.zeros(2 * M, int))]
    for d in range(2):
        cols = 1 + 2 * np.arange(len(free)) + d
        cj = np.where(col[ji] >= 0, cols[np.maximum(col[ji], 0)], -1)
        groups.append((cols, np.r_[cj, cj]))
    x0 = np.r_[np.log(f0), T0[free].ravel(), poses.ravel()]
    lb = (np.log(f_bounds[0]), np.log(f_bounds[1]))
    x0[0] = np.clip(x0[0], *lb)
    hp = float(np.median(pb.huber[pb.keep])) if huber_px is None else float(huber_px)
    x, r, it, ok = _lm_schur(resid, x0, G, N, fi, groups, lb, hp, max_iter)
    f, T, _ = unpack(x)
    return {"f": float(f), "template": T, "success": bool(ok), "nfev": int(it),
            "rms_px": float(np.sqrt(np.mean(r[:2 * M] ** 2)))}


def _restrict(pb, min_frames=3, max_frames=120):
    keep = list(pb.keep)
    if max_frames and len(keep) > max_frames:
        keep = [keep[k] for k in np.unique(np.linspace(0, len(keep) - 1, max_frames).round().astype(int))]
    cnt = np.sum([pb.frames[i][1] for i in keep], axis=0)
    ids = np.flatnonzero(cnt >= min_frames)
    q = Problem.__new__(Problem)
    q.T = pb.T[ids]
    q.frames = [(o[ids], v[ids]) for o, v in pb.frames]
    q.keep = [i for i in keep if q.frames[i][1].sum() >= 4]
    q.cx, q.cy, q.huber = pb.cx, pb.cy, pb.huber
    q.w = None if pb.w is None else pb.w[ids]
    q.n_points = int(sum(q.frames[i][1].sum() for i in q.keep))
    return q, ids


def free_template(pb, grid, alt_iters=3, refine=True, max_frames=120, coarse_n=8):
    if len(grid) > coarse_n:
        grid = grid[np.unique(np.linspace(0, len(grid) - 1, coarse_n).round().astype(int))]
    if not pb.keep:
        raise ValueError("no usable frames")
    pb, ids = _restrict(pb, max_frames=max_frames)
    if len(ids) < 4 or not pb.keep:
        raise ValueError("fewer than 4 usable joints")
    T0 = procrustes_init(pb)
    c, Ts = [], []
    for f in grid:
        T = alt_template(pb, f, T0, alt_iters)
        c.append(cost(pb, f, T))
        Ts.append(T)
    c = np.array(c)
    k = int(np.argmin(c))
    lf = _parabola(np.log(grid), c, k)
    f_grid_best = float(np.exp(np.clip(lf, np.log(grid[0]), np.log(grid[-1]))))
    out = {"f_grid": f_grid_best, "profile_f": grid.tolist(), "profile_cost": c.tolist(), "delta_deg": 0.0, "phi_deg": 0.0,
           "at_grid_edge": bool(k in (0, len(grid) - 1)), "template": Ts[k], "joint_ids": ids.tolist()}
    if refine:
        ba = ba_refine(pb, f_grid_best, Ts[k], (grid[0], grid[-1]))
        out.update({"f": ba["f"], "template": ba["template"], "ba": {kk: vv for kk, vv in ba.items() if kk != "template"}})
    else:
        out["f"] = f_grid_best
    out["cost"] = cost(pb, out["f"], out["template"])
    return out


def free_problem(trk, size_wh, ecfg, huber_rel, ids=None, weights=None):
    W, H = size_wh
    cx, cy = principal_point(ecfg, W, H)
    kp = undistort(trk["kp"], ecfg, W, H)
    J = kp.shape[1]
    sel = np.ones(J, bool) if ids is None else np.isin(np.arange(J), ids)
    idx = np.flatnonzero(trk["present"])
    frames = [(kp[i], trk["mask"][i] & sel & np.isfinite(kp[i]).all(1)) for i in idx]
    hub = float(huber_rel) * box_long(trk["box"][idx])
    return Problem(np.zeros((J, 2)), frames, cx, cy, hub, weights=weights), idx


def ft_estimate_run(pb, method, grid, ecfg, sigma2, start=None):
    """The supervisor's estimate_run: estimate_run (unchanged, above) plus the free_template branch."""
    if method == "free_template":
        return free_template(pb, grid)
    return estimate_run(pb, method, grid, ecfg, sigma2, start)


def _anchors3(T, frames):
    a, b = _anchors(T, frames)
    vis = np.sum([v for o, v in frames], axis=0)
    good = [j for j in np.flatnonzero(vis >= 0.9 * vis.max()) if j not in (a, b)]
    if not good:
        good = [j for j in range(len(T)) if j not in (a, b)]
    u = (T[b] - T[a]) / np.linalg.norm(T[b] - T[a])
    d = [abs(u[0] * (T[j] - T[a])[1] - u[1] * (T[j] - T[a])[0]) for j in good]
    return a, b, int(good[int(np.argmax(d))])


def _relief(S, rms0):
    c = S - S.mean(0)
    n = np.linalg.svd(c)[2][-1]
    d = c @ n
    return d, float(np.sqrt(np.mean(d ** 2)) / rms0)


def ba3_refine(pb, f0, T2, f_bounds, z_rel=0.2, huber_px=None, max_iter=60):
    J = T2.shape[0]
    frames = [pb.frames[i] for i in pb.keep]
    N = len(frames)
    K0 = K_of(f0, pb.cx, pb.cy)
    poses = np.zeros((N, 6))
    for n, (o, v) in enumerate(frames):
        _, R, t = frame_fit(T2, o, v, K0)
        poses[n, :3] = cv2.Rodrigues(R)[0].ravel()
        poses[n, 3:] = t
    fi = np.concatenate([np.full(v.sum(), n) for n, (o, v) in enumerate(frames)])
    ji = np.concatenate([np.flatnonzero(v) for o, v in frames])
    uv = np.concatenate([o[v] for o, v in frames])
    sw = np.ones(len(ji)) if pb.w is None else np.sqrt(pb.w[ji])
    a, b, c = _anchors3(T2, frames)
    S0 = np.c_[T2, np.zeros(J)]
    fixed = np.zeros((J, 3), bool)
    fixed[a], fixed[b], fixed[c, 2] = True, True, True
    fidx = np.flatnonzero(~fixed.ravel())
    nS = len(fidx)
    col_of = np.full(3 * J, -1)
    col_of[fidx] = 1 + np.arange(nS)
    rms0 = float(np.sqrt((T2 ** 2).sum(1).mean()))
    hp = float(np.median(pb.huber[pb.keep])) if huber_px is None else float(huber_px)
    zs = None if z_rel is None else float(z_rel) * rms0
    M = len(ji)
    nr = 2 * M + (J if zs else 0)

    def unpack(x):
        S = S0.ravel().copy()
        S[fidx] = x[1:1 + nS]
        return np.exp(x[0]), S.reshape(J, 3), x[1 + nS:].reshape(N, 6)

    def proj(x):
        f, S, Pz = unpack(x)
        R = _rodrigues(Pz[:, :3])
        X = np.einsum("nab,nb->na", R[fi], S[ji]) + Pz[fi, 3:]
        u = f * X[:, 0] / X[:, 2] + pb.cx - uv[:, 0]
        w = f * X[:, 1] / X[:, 2] + pb.cy - uv[:, 1]
        return u, w, S

    def resid(x):
        u, w, S = proj(x)
        r = [sw * u, sw * w]
        if zs:
            r.append(hp * S[:, 2] / zs)
        return np.concatenate(r)

    G = 1 + nS
    groups = [(np.array([0]), np.zeros(nr, int))]
    for d in range(3):
        cols = col_of[3 * np.arange(J) + d]
        m = np.full(nr, -1)
        m[:M] = cols[ji]
        m[M:2 * M] = cols[ji]
        if zs and d == 2:
            m[2 * M:] = cols
        groups.append((cols[cols >= 0], m))
    x0 = np.r_[np.log(f0), S0.ravel()[fidx], poses.ravel()]
    lb = (np.log(f_bounds[0]), np.log(f_bounds[1]))
    x0[0] = np.clip(x0[0], *lb)
    xs, rr, it, okk = _lm_schur(resid, x0, G, N, fi, groups, lb, hp, max_iter)
    f, S, Pz = unpack(xs)
    u, w, _ = proj(xs)
    e = np.hypot(u, w)
    keep = np.array(pb.keep)
    hb = pb.huber[keep][fi]
    lossv = np.where(e <= hb, e * e, 2 * hb * e - hb * hb)
    if pb.w is not None:
        num = np.bincount(fi, pb.w[ji] * lossv, N)
        den = np.bincount(fi, pb.w[ji], N)
    else:
        num = np.bincount(fi, lossv, N)
        den = np.bincount(fi, None, N)
    rel_d, rel = _relief(S, rms0)
    return {"f": float(f), "shape": S, "poses": Pz, "success": bool(okk), "nfev": int(it),
            "rms_px": float(np.sqrt(np.mean(e ** 2))), "cost": float(np.mean(num / np.maximum(den, 1e-12))),
            "relief_rel": rel, "relief_per_joint": rel_d / rms0, "anchors": (a, b, c),
            "at_bound": bool(xs[0] <= lb[0] + 1e-6 or xs[0] >= lb[1] - 1e-6)}


def free_shape3d(pb, grid, z_rel=0.2, alt_iters=3, max_frames=120, planar=None):
    A = free_template(pb, grid, alt_iters=alt_iters, max_frames=max_frames) if planar is None else planar
    pr, ids = _restrict(pb, max_frames=max_frames)
    if list(ids) != list(A["joint_ids"]):
        raise ValueError("planar result was computed on different joints")
    B = ba3_refine(pr, A["f"], A["template"], (grid[0], grid[-1]), z_rel=z_rel)
    return {"f": B["f"], "f_planar": A["f"], "shape": B["shape"], "joint_ids": A["joint_ids"], "relief_rel": B["relief_rel"],
            "relief_per_joint": B["relief_per_joint"].tolist(), "cost": B["cost"], "cost_planar": A["cost"],
            "profile_f": A["profile_f"], "profile_cost": A["profile_cost"], "at_grid_edge": A["at_grid_edge"], "at_bound": B["at_bound"],
            "delta_deg": 0.0, "phi_deg": 0.0, "ba": {k: B[k] for k in ("success", "nfev", "rms_px", "anchors")}}


def s3_estimate_run(pb, method, grid, ecfg, sigma2, start=None):
    """The supervisor's estimate_run with the free_template and free_shape3d branches (as received); the existing estimate_run is not changed."""
    if method == "fixed_template":
        return fixed_template(pb, grid)
    if method == "joint_delta":
        return joint_delta(pb, grid, ecfg["joint_delta"], sigma2, start)
    if method == "free_template":
        return free_template(pb, grid)
    if method == "free_shape3d":
        fc = ecfg.get("free_template", {})
        return free_shape3d(pb, grid, z_rel=fc.get("z_rel", 0.2), alt_iters=int(fc.get("alt_iters", 3)), max_frames=int(fc.get("max_frames", 120)))
    raise ValueError(method)
