"""두 평면 추정기: 손가락 평면과 손등 평면을 함께 써서 손 경사 없이 f 를 추정한다.

배경: 손은 책상 위에서 돌고(책상 법선 둘레 회전) 미끄러지기(책상 면 안의 이동)만 한다. 이 운동에서 평면 하나만 보면 f 와 그 평면의 기울기가 서로 바뀌어도
     화면이 거의 같아 보여 둘을 가를 수 없다. 기울기가 다른 평면 두 개가 같은 손에 붙어 함께 움직이면 손이 입체가 되어 f 가 정해질 수 있다.

관측: 기준 프레임의 2D 점 p_i (검출 관절) 를 두 추적 사슬로 옮긴 위치.
      A = 손 전체 마스크 사슬(실험 90: 손가락 평면을 따라감), B = 손등 몸통 마스크 사슬(실험 91).
모델 (기준 프레임의 카메라 좌표):
  - 카메라 K: 초점 f, 주점 = 영상 가운데, 왜곡 없음
  - 평면 A: a·X = 1 (a = (ax, ay, 1): 광축과 깊이 1 에서 만남 = 크기 기준), 평면 B: b·X = 1 (b 자유)
  - 기준 프레임의 3D 점: X_i = r_i / (a·r_i)  (r_i = K^-1 [p_i, 1], 평면 B 도 같은 식)
  - 프레임 k: X' = R(n, θ_k) (X - q) + q + u_k,  n = 책상 법선(카메라 쪽, 자유), q = 기준 프레임 평면 A 점들의 중심(회전축이 손을 지나게),
    u_k 는 책상 면 안(n 에 수직)의 미끄러짐
  - 손 경사(평면과 책상 사이 각)는 따로 두지 않는다. 평면 A, B 의 방향과 n 이 모두 자유이므로 경사는 결과로 나온다.
풀이: f 를 격자 값에 고정하고 나머지를 푼 비용 곡선(프로파일) -> 가장 낮은 몇 곳에서 f 도 풀어 비용 최소를 고른다.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def rodrigues(axis: np.ndarray, ang: np.ndarray) -> np.ndarray:
    """단위 축 axis (3,) 둘레로 각 ang (K,) 만큼 도는 회전 행렬 (K, 3, 3)."""
    k = axis / np.linalg.norm(axis)
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    s, c = np.sin(ang)[:, None, None], np.cos(ang)[:, None, None]
    return np.eye(3)[None] + s * Kx[None] + (1 - c) * (Kx @ Kx)[None]


def desk_basis(n: np.ndarray):
    """책상 법선 n 에 수직인 단위 벡터 두 개."""
    n = n / np.linalg.norm(n)
    t = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, t)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def unpack(x: np.ndarray, K: int, lf_fixed=None):
    """매개변수 벡터 -> (f, n, a, b, θ (K,), u (K, 2)). lf_fixed 가 있으면 f 는 매개변수에 없음."""
    i = 0
    if lf_fixed is None:
        lf = x[0]
        i = 1
    else:
        lf = lf_fixed
    n = np.array([x[i], x[i + 1], -1.0])
    a = np.array([x[i + 2], x[i + 3], 1.0])
    b = x[i + 4:i + 7]
    th = x[i + 7:i + 7 + K]
    u = x[i + 7 + K:i + 7 + 3 * K].reshape(K, 2)
    return float(np.exp(lf)), n / np.linalg.norm(n), a, b, th, u


def predict(f: float, c: np.ndarray, n, a, b, th, u, p_ref: np.ndarray):
    """모델이 그리는 두 점 묶음 (K, J, 2) 씩."""
    r = np.c_[(p_ref - c) / f, np.ones(len(p_ref))]
    XA = r / (r @ a)[:, None]
    XB = r / (r @ b)[:, None]
    R = rodrigues(n, th)
    e1, e2 = desk_basis(n)
    U = u[:, :1] * e1[None] + u[:, 1:] * e2[None]
    q = XA.mean(0)
    out = []
    for X in (XA, XB):
        Y = np.einsum("kij,nj->kni", R, X - q) + q + U[:, None, :]
        Y[..., 2] = np.maximum(Y[..., 2], 1e-3)                      # 카메라 뒤로 넘어가는 시도는 잔차를 크게 만들되 발산하지 않게
        out.append(f * Y[..., :2] / Y[..., 2:3] + c)
    return out


def _residual(x, K, p_ref, obsA, obsB, c, lf_fixed, scale):
    f, n, a, b, th, u = unpack(x, K, lf_fixed)
    pa, pb = predict(f, c, n, a, b, th, u, p_ref)
    ra, rb = (pa - obsA) / scale, (pb - obsB) / scale
    ra[~np.isfinite(ra)] = 0.0
    rb[~np.isfinite(rb)] = 0.0
    return np.concatenate([ra.ravel(), rb.ravel()])


def _sparsity(K: int, J: int, with_f: bool):
    g = 7 + (1 if with_f else 0)
    m = lil_matrix((2 * 2 * K * J, g + 3 * K), dtype=int)
    for s in range(2):
        for k in range(K):
            rows = slice(s * 2 * K * J + k * 2 * J, s * 2 * K * J + (k + 1) * 2 * J)
            m[rows, :g] = 1
            m[rows, g + k] = 1
            m[rows, g + K + 2 * k:g + K + 2 * k + 2] = 1
    return m


def init_theta(p_ref: np.ndarray, obsA: np.ndarray, i0: int = 0, i1: int = 9) -> np.ndarray:
    """화면에서 손 방향(i0 -> i1)의 회전 각(기준 대비, 라디안). 회전 각의 초기값."""
    v0 = p_ref[i1] - p_ref[i0]
    v = obsA[:, i1] - obsA[:, i0]
    return np.unwrap(np.arctan2(v[:, 1], v[:, 0]) - np.arctan2(v0[1], v0[0]))


def _sparsity_a(K: int, J: int):
    """평면 A 만 맞출 때(1단계)의 야코비안 모양: 전역 4개(n 2, a 2) + 프레임마다 θ, u."""
    m = lil_matrix((2 * K * J, 4 + 3 * K), dtype=int)
    for k in range(K):
        rows = slice(k * 2 * J, (k + 1) * 2 * J)
        m[rows, :4] = 1
        m[rows, 4 + k] = 1
        m[rows, 4 + K + 2 * k:4 + K + 2 * k + 2] = 1
    return m


def solve_fixed_f(f, p_ref, obsA, obsB, size_wh, tilt_deg, sign, scale, b_offsets=(0.03, -0.03)):
    """f 고정 풀이 (두 단계).
    1단계: 평면 A 만으로 n, a, θ, u 를 맞춘다 (평면 하나 = 지금까지의 방식과 같은 정보).
    2단계: b 를 a 근처(b_offsets 만큼 기울여)에서 시작해 전부 함께 푼다. 오프셋마다 풀어 비용이 낮은 쪽.
    tilt_deg: 책상 법선 초기값(카메라가 책상을 내려다보는 각), sign: 회전 방향 초기값(+1/-1)."""
    W, H = size_wh
    c = np.array([W / 2.0, H / 2.0])
    K, J = obsA.shape[:2]
    ny = -np.tan(np.radians(tilt_deg))
    xa = np.r_[0.0, ny, 0.0, -ny, sign * init_theta(p_ref, obsA), np.zeros(2 * K)]

    def res_a(x):
        return _residual(np.r_[x[:4], x[2], x[3], 1.0, x[4:]], K, p_ref, obsA, obsA, c, np.log(f), scale)[:2 * K * J]
    ra = least_squares(res_a, xa, jac_sparsity=_sparsity_a(K, J), method="trf", max_nfev=400, x_scale="jac")
    best = None
    for d in b_offsets:
        b0 = np.array([ra.x[2], ra.x[3] + d, 1.0])
        x0 = np.r_[ra.x[:4], b0, ra.x[4:]]
        r = least_squares(_residual, x0, jac_sparsity=_sparsity(K, J, False), method="trf", args=(K, p_ref, obsA, obsB, c, np.log(f), scale), max_nfev=400, x_scale="jac")
        if best is None or r.cost < best.cost:
            best = r
    return best


def _full_x(x: np.ndarray, mode: str, with_f: bool) -> np.ndarray:
    """평면 하나(mode A 또는 B)의 매개변수 [lf?, n 2, a 2, θ, u] -> 두 평면 형식 (b = a)."""
    g = 1 if with_f else 0
    return np.r_[x[:g + 4], x[g + 2], x[g + 3], 1.0, x[g + 4:]]


def _sparsity_one(K: int, J: int, with_f: bool):
    g = 4 + (1 if with_f else 0)
    m = lil_matrix((2 * K * J, g + 3 * K), dtype=int)
    for k in range(K):
        rows = slice(k * 2 * J, (k + 1) * 2 * J)
        m[rows, :g] = 1
        m[rows, g + k] = 1
        m[rows, g + K + 2 * k:g + K + 2 * k + 2] = 1
    return m


def solve(p_ref, obsA, obsB, size_wh, f_grid, mode="AB", tilts=(30.0, 45.0, 60.0), n_final=3, scale=1.0):
    """mode: "AB" = 두 평면 함께, "A" / "B" = 그 평면 하나만 (비교용).
    f 프로파일(격자의 각 f 에서 시작점 tilts x 회전 방향 2개 중 비용 최소) -> 비용이 낮은 n_final 곳에서 f 까지 풀어 비용 최소를 고른다."""
    W, H = size_wh
    c = np.array([W / 2.0, H / 2.0])
    K, J = obsA.shape[:2]
    one = None if mode == "AB" else (obsA if mode == "A" else obsB)
    prof = []
    for f in f_grid:
        best = None
        for t in tilts:
            for sg in (1.0, -1.0):
                if one is None:
                    r = solve_fixed_f(f, p_ref, obsA, obsB, size_wh, t, sg, scale)
                    x = r.x
                else:
                    ny = -np.tan(np.radians(t))
                    xa = np.r_[0.0, ny, 0.0, -ny, sg * init_theta(p_ref, one), np.zeros(2 * K)]
                    r = least_squares(lambda x: _residual(_full_x(x, mode, False), K, p_ref, one, one, c, np.log(f), scale)[:2 * K * J], xa,
                                      jac_sparsity=_sparsity_one(K, J, False), method="trf", max_nfev=400, x_scale="jac")
                    x = r.x
                if best is None or r.cost < best[0]:
                    best = (float(r.cost), x)
        prof.append(best)
    cost = np.array([b[0] for b in prof])
    finals = []
    for k in np.argsort(cost)[:n_final]:
        x0 = np.r_[np.log(f_grid[k]), prof[k][1]]
        if one is None:
            r = least_squares(_residual, x0, jac_sparsity=_sparsity(K, J, True), method="trf", args=(K, p_ref, obsA, obsB, c, None, scale), max_nfev=500, x_scale="jac")
            finals.append((r.cost, r.x))
        else:
            r = least_squares(lambda x: _residual(_full_x(x, mode, True), K, p_ref, one, one, c, None, scale)[:2 * K * J], x0,
                              jac_sparsity=_sparsity_one(K, J, True), method="trf", max_nfev=500, x_scale="jac")
            finals.append((r.cost, _full_x(r.x, mode, True)))
    cst, xb = min(finals, key=lambda t: t[0])
    f, n, a, b, th, u = unpack(xb, K)
    pa, pb = predict(f, c, n, a, b, th, u, p_ref)
    if one is None:
        d2 = np.r_[np.sum((pa - obsA) ** 2, -1).ravel(), np.sum((pb - obsB) ** 2, -1).ravel()]
    else:
        d2 = np.sum((pa - one) ** 2, -1).ravel()
    ang = lambda v: float(np.degrees(np.arccos(abs(v @ n) / np.linalg.norm(v))))      # noqa: E731  (평면과 책상 사이 각)
    return {"f": f, "cost": float(cst), "rms_px": float(np.sqrt(np.nanmean(d2))), "plane_a_deg": ang(a), "plane_b_deg": ang(b),
            "dihedral_deg": float(np.degrees(np.arccos(min(1.0, abs(a @ b) / np.linalg.norm(a) / np.linalg.norm(b))))),
            "desk_tilt_deg": float(np.degrees(np.arccos(abs(n[2])))), "turn_range_deg": float(np.degrees(th.max() - th.min())),
            "profile_f": [float(x) for x in f_grid], "profile_cost": [float(x) for x in cost], "at_bound": bool(f <= f_grid[0] * 1.001 or f >= f_grid[-1] * 0.999)}


# ----------------------------------------------------------------------------------------------------
# 합성 장면 (실험 92)
# ----------------------------------------------------------------------------------------------------
def synthetic(f=1672.0, tilt_deg=45.0, dist_cm=60.0, ramp_a=11.0, ramp_b=15.0, turn_deg=110.0, n=90, noise_px=0.5, seed=0, size_wh=(1920, 1080), hand_cm=None):
    """책상 위 손: 평면 A(손가락 쪽, 손끝 선이 책상에 닿고 경사 ramp_a)와 평면 B(MCP 선에서 A 와 만나고 경사 ramp_b, 손목 쪽). 손은 책상 위에서 사인 곡선으로 turn_deg 범위만큼 돌고
    조금 미끄러진다. 기준 프레임(가운데)의 2D 점 = 손 배치(hand_cm)를 평면 A 에 올려 찍은 것. 관측 = 그 광선과 평면 A·B 의 교점을 프레임마다 찍고 잡음(noise_px, 프레임·점마다 독립)을 더한 것."""
    W, H = size_wh
    c = np.array([W / 2.0, H / 2.0])
    rng = np.random.default_rng(seed)
    P2 = np.asarray(hand_cm, float)
    P2 = P2 - P2.mean(0)
    tips = P2[[8, 12, 16, 20]].mean(0)
    mcps = P2[[5, 9, 13, 17]].mean(0)
    ax = (mcps - tips) / np.linalg.norm(mcps - tips)                  # 손끝 -> MCP 방향 (손목 쪽)
    s = (P2 - tips) @ ax                                              # 손끝 선에서 손목 쪽으로의 거리
    sm = float((mcps - tips) @ ax)
    hA = np.tan(np.radians(ramp_a)) * s
    hB = np.tan(np.radians(ramp_a)) * sm + np.tan(np.radians(ramp_b)) * (s - sm)
    a_ = np.radians(tilt_deg)
    C = np.array([0.0, -dist_cm * np.sin(a_), dist_cm * np.cos(a_)])  # 책상 좌표의 카메라 위치 (책상 z 위쪽)
    z = -C / np.linalg.norm(C)
    x = np.array([1.0, 0.0, 0.0])
    Rc = np.array([x, np.cross(z, x), z])
    t = np.linspace(0, 1, n)
    th = np.radians(turn_deg / 2.0) * np.sin(2 * np.pi * t)
    mv = np.c_[3.0 * np.sin(2 * np.pi * 1.5 * t + 0.4), 4.0 * np.cos(2 * np.pi * t)]
    ref = n // 2

    def cam(Pw):
        return (Pw - C) @ Rc.T

    def place(h, k):
        cc, sn = np.cos(th[k]), np.sin(th[k])
        return np.c_[P2 @ np.array([[cc, sn], [-sn, cc]]) + mv[k], h]

    XA_ref = cam(place(hA, ref))
    p_ref = f * XA_ref[:, :2] / XA_ref[:, 2:3] + c
    # 평면 B 위 점: 기준 프레임에서 p_ref 광선과 평면 B 의 교점 -> 손 좌표로 되돌려 움직임
    XBtrue_ref = cam(place(hB, ref))
    nB = np.cross(XBtrue_ref[9] - XBtrue_ref[0], XBtrue_ref[17] - XBtrue_ref[0])
    dB = nB @ XBtrue_ref[0]
    r = np.c_[(p_ref - c) / f, np.ones(len(p_ref))]
    XB_ref = r * (dB / (r @ nB))[:, None]
    XBw_ref = XB_ref @ Rc + C                                         # 카메라 -> 책상 좌표 (X_c = Rc (X_w - C) 의 역)
    cr, sr = np.cos(th[ref]), np.sin(th[ref])
    loc = (XBw_ref[:, :2] - mv[ref]) @ np.array([[cr, sr], [-sr, cr]]).T
    hBv = XBw_ref[:, 2]
    obsA, obsB = [], []
    for k in range(n):
        for h, P, lst in ((hA, None, obsA), (hBv, loc, obsB)):
            if P is None:
                Pw = place(h, k)
            else:
                cc, sn = np.cos(th[k]), np.sin(th[k])
                Pw = np.c_[P @ np.array([[cc, sn], [-sn, cc]]) + mv[k], h]
            X = cam(Pw)
            lst.append(f * X[:, :2] / X[:, 2:3] + c + rng.normal(scale=noise_px, size=(len(P2), 2)))
    return {"p_ref": p_ref, "obsA": np.array(obsA), "obsB": np.array(obsB), "ref": ref, "size": (W, H)}
