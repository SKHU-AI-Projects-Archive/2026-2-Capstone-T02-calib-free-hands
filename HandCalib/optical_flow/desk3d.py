"""책상 운동 + 입체 손 모양 추정기 (실험 114).

지금까지의 책상 운동 추정기(handfocal_report.dk_solve)는 손을 '경사진 평면 하나'로 본다(평면 템플릿 T + 경사 pitch·옆 기울기 roll).
여기서는 손을 관절마다 높이를 가진 단단한 3D 모양 S 로 두고, 움직임은 그대로 둔다:

    x_cam = Rd (Rz(theta_i) S + [x_i, y_i, 0]) + td

  - Rd, td: 영상 전체가 공유하는 책상 자세. 프레임마다 책상 법선 둘레 회전 theta_i 와 책상 면 안 미끄러짐 (x_i, y_i) (기준 프레임 = 0).
  - S (관절 수 x 3): 책상 좌표의 손 모양. 경사 가정 없음. 게이지(책상 면 안 위치 2, 회전 1, 크기 1, 높이 원점 1)는 고정 관절 두 개로 막는다:
    a = (x, y, z) 고정, b = (x, y) 고정.
  - z_rel 이 있으면 높이에 약한 제약: 잔차 hp * (z_j - 평균 z) / (z_rel * 템플릿 크기) 를 더한다(손 높이가 손 크기의 z_rel 배 정도를 넘지 않게). None 이면 제약 없음.
  - 평면 운동에서 단단한 입체 물체는 모양을 몰라도 f 가 정해진다(각 점이 책상과 평행한 평면 위를 돎). 손이 평면이면 이 정보가 없어진다.

시작값: f 마다 기존 dk_solve(경사 ramp_init 고정)의 해에서 S0 = [T, 0] 을 경사만큼 기울인 것과 그 책상 자세·프레임 움직임.
"""
import numpy as np


def dk3_solve(pr, f: float, init: dict, ids, free_f: bool, f_bounds, z_rel=None, max_nfev: int = 200) -> dict:
    """입체 손 모양 책상 모델의 최소제곱 (dk_solve 와 같은 풀이: scipy trf, 희소 수치 미분, Huber f_scale = 프레임 Huber 폭의 중앙값).
    init: {"f", "S" (J, 3), "Rd", "td", "theta" (N,), "xy" (N, 2), "ref"}."""
    import cv2
    from scipy.optimize import least_squares
    from scipy.optimize._numdiff import approx_derivative, group_columns
    from scipy.sparse import lil_matrix
    import handfocal as hf
    frames = [pr.frames[i] for i in pr.keep]
    N = len(frames)
    S0 = np.asarray(init["S"], float)
    J = len(S0)
    r = int(init["ref"])
    a, b = hf._anchors(S0[:, :2], frames)
    fixed = np.zeros((J, 3), bool)
    fixed[a] = True
    fixed[b, :2] = True
    sfree = np.flatnonzero(~fixed.ravel())
    fr_free = np.array([k for k in range(N) if k != r], int)
    fi = np.concatenate([np.full(v.sum(), k) for k, (o, v) in enumerate(frames)])
    ji = np.concatenate([np.flatnonzero(v) for o, v in frames])
    uv = np.concatenate([o[v] for o, v in frames])
    M = len(ji)
    ng = int(free_f) + 6
    ns = len(sfree)
    hp = float(np.median(pr.huber[pr.keep]))
    size = float(np.sqrt(((S0[:, :2] - S0[:, :2].mean(0)) ** 2).sum(1)).mean()) if J else 1.0
    zs = None if z_rel is None else float(z_rel) * size

    def unpack(x):
        k = 0
        ff = float(np.exp(x[0])) if free_f else float(init["f"])
        k += int(free_f)
        Rd = hf._rodrigues(x[k:k + 3][None])[0]
        td = x[k + 3:k + 6]
        k += 6
        S = S0.ravel().copy()
        S[sfree] = x[k:k + ns]
        k += ns
        P = np.zeros((N, 3))
        P[fr_free] = x[k:].reshape(-1, 3)
        return ff, Rd, td, S.reshape(J, 3), P

    def proj(x):
        ff, Rd, td, S, P = unpack(x)
        c, s_ = np.cos(P[fi, 0]), np.sin(P[fi, 0])
        h = S[ji]
        Xd = np.c_[c * h[:, 0] - s_ * h[:, 1] + P[fi, 1], s_ * h[:, 0] + c * h[:, 1] + P[fi, 2], h[:, 2]]
        X = Xd @ Rd.T + td
        return np.c_[ff * X[:, 0] / X[:, 2] + pr.cx, ff * X[:, 1] / X[:, 2] + pr.cy], S

    def resid(x):
        p, S = proj(x)
        res = (p - uv).ravel()
        if zs:
            res = np.r_[res, hp * (S[:, 2] - S[:, 2].mean()) / zs]
        return res

    x0 = []
    if free_f:
        x0.append([np.log(init["f"])])
    x0 += [cv2.Rodrigues(np.asarray(init["Rd"], float))[0].ravel(), np.asarray(init["td"], float), S0.ravel()[sfree]]
    x0.append(np.c_[init["theta"], init["xy"]][fr_free].ravel())
    x0 = np.concatenate(x0).astype(float)
    lo, hi = np.full(len(x0), -np.inf), np.full(len(x0), np.inf)
    if free_f:
        lo[0], hi[0] = np.log(f_bounds[0]), np.log(f_bounds[1])
        x0[0] = np.clip(x0[0], lo[0] + 1e-9, hi[0] - 1e-9)
    nr = 2 * M + (J if zs else 0)
    Sp = lil_matrix((nr, len(x0)), dtype=int)
    Sp[:2 * M, :ng] = 1
    col_s = np.full(3 * J, -1)
    col_s[sfree] = ng + np.arange(ns)
    pos = {int(k): ng + ns + 3 * m for m, k in enumerate(fr_free)}
    for m in range(M):
        for d in range(3):
            c = col_s[3 * int(ji[m]) + d]
            if c >= 0:
                Sp[2 * m:2 * m + 2, c] = 1
        if int(fi[m]) in pos:
            Sp[2 * m:2 * m + 2, pos[int(fi[m])]:pos[int(fi[m])] + 3] = 1
    if zs:
        zc = col_s[3 * np.arange(J) + 2]
        for j in range(J):
            Sp[2 * M + j, zc[zc >= 0]] = 1
    Sp = Sp.tocsc()
    grp = group_columns(Sp)

    def jac(x):
        return approx_derivative(resid, x, sparsity=(Sp, grp), bounds=(lo, hi)).toarray()
    sol = least_squares(resid, x0, jac=jac, bounds=(lo, hi), loss="huber", f_scale=hp, x_scale="jac", max_nfev=max_nfev, method="trf", tr_solver="exact")
    ff, Rd, td, S, P = unpack(sol.x)
    e = np.hypot(*(proj(sol.x)[0] - uv).T)
    lossv = np.where(e <= hp, e * e, 2 * hp * e - hp * hp)
    zr = S[:, 2] - S[:, 2].min()
    return {"f": ff, "desk_tilt_deg": float(np.degrees(np.arccos(min(abs(Rd[2, 2]), 1.0)))), "rms_px": float(np.sqrt(np.mean(e ** 2))), "cost": float(lossv.mean()),
            "height_rel": float(np.ptp(S[:, 2]) / max(size, 1e-12)), "success": bool(sol.success), "nfev": int(sol.nfev),
            "at_bound": bool(free_f and (sol.x[0] <= lo[0] + 1e-6 or sol.x[0] >= hi[0] - 1e-6)),
            "height_per_joint": (zr / max(size, 1e-12)).tolist(), "joint_ids": [int(j) for j in ids],
            "state": {"f": ff, "S": S, "Rd": Rd, "td": td, "theta": P[:, 0], "xy": P[:, 1:], "ref": r}}


def dk3_init(pr, f: float, T2: np.ndarray, ids, ramp_deg: float, f_bounds) -> dict:
    """f 에서의 시작값: 기존 dk_solve(경사 ramp_deg 고정, f 고정)의 해를 입체 모양으로 옮김 (S = [T, 0] 을 그 경사로 기울인 책상 좌표)."""
    import handfocal as hf
    s = hf.dk_solve(pr, float(f), T2, ids, np.radians(float(ramp_deg)), 0.0, False, False, f_bounds)["state"]
    u, w = hf.dk_axes(T2, ids)
    S = np.c_[s["T"], np.zeros(len(s["T"]))] @ hf.dk_hand_rot(u, w, s["up"] * s["pitch"], s["roll"]).T
    return {"f": float(f), "S": S, "Rd": s["Rd"], "td": s["td"], "theta": s["theta"], "xy": s["xy"], "ref": s["ref"]}


def dk3_estimate(pb, grid: np.ndarray, A: dict, ramp_init: float, z_rel=None, coarse_n: int = 8, max_frames: int = 120, n_final: int = 3) -> dict:
    """dk_estimate_ms 와 같은 짜임: f 격자 coarse_n 곳에서 f 고정으로 풀고(시작 = dk3_init), 이웃 해에서 다시 풀어 더 낮으면 바꾸고, 비용 낮은 n_final 곳에서 f 까지 푼다."""
    import handfocal as hf
    pr, ids = hf._restrict(pb, max_frames=max_frames)
    if list(ids) != list(A["joint_ids"]):
        raise ValueError("free_template result was computed on different joints")
    jid = [int(j) for j in A["joint_ids"]]
    T2 = np.asarray(A["template"], float)
    g = grid[np.unique(np.linspace(0, len(grid) - 1, coarse_n).round().astype(int))] if len(grid) > coarse_n else grid
    fb = (grid[0], grid[-1])
    prof = [dk3_solve(pr, float(f), dk3_init(pr, float(f), T2, jid, ramp_init, fb), jid, False, fb, z_rel) for f in g]
    for d, idx in ((-1, range(1, len(g))), (1, range(len(g) - 2, -1, -1))):
        for i in idx:
            x = dk3_solve(pr, float(g[i]), dict(prof[i + d]["state"], f=float(g[i])), jid, False, fb, z_rel)
            if x["cost"] < prof[i]["cost"]:
                prof[i] = x
    c = np.array([x["cost"] for x in prof])
    finals = []
    for k in np.argsort(c)[:n_final]:
        k = int(k)
        loc = 0 < k < len(g) - 1 and c[k] <= c[k - 1] and c[k] <= c[k + 1]
        fs = float(np.exp(np.clip(hf._parabola(np.log(g), c, k), np.log(fb[0]), np.log(fb[1])))) if loc else float(g[k])
        B = dk3_solve(pr, fs, dict(prof[k]["state"], f=fs), jid, True, fb, z_rel)
        finals.append((k, fs, B))
    k, fs, B = min(finals, key=lambda t: t[2]["cost"])
    B = dict(B)
    B.pop("state")
    B.update({"f_start": fs, "profile_f": g.tolist(), "profile_cost": c.tolist(), "starts": [{"k": kk, "f_start": ff, "f": bb["f"], "cost": bb["cost"]} for kk, ff, bb in finals]})
    return B


def dk3_job(args) -> dict:
    """한 추정: args = (kp, mask, frames, box, size_wh, ecfg, ids, A, ramp_init, z_rel). od_bridge -> dk3_estimate. 실패도 결과로."""
    import os
    import time
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import cv2
    cv2.setNumThreads(1)
    import handfocal as hf
    kp, mask, frames, box, size_wh, ecfg, ids, A, ramp_init, z_rel = args
    grid = hf.f_grid(ecfg, max(*size_wh))
    t0 = time.time()
    try:
        pb = hf.od_bridge(kp, mask, frames, box, size_wh, ecfg, ids)
        B = dk3_estimate(pb, grid, A, ramp_init, z_rel)
        B["boundary"] = bool(B["at_bound"]) or hf.od_boundary(float(B["f"]), grid)
    except Exception as e:
        B = {"failed": f"{type(e).__name__}: {e}"}
    B["elapsed_s"] = time.time() - t0
    return B



# ---- 115: 닿는 관절 높이 고정 ----
def dk3c_solve(pr, f: float, init: dict, ids, free_f: bool, f_bounds, contact, z_rel=None, max_nfev: int = 200) -> dict:
    """dk3_solve 에 닿는 관절 조건을 더한 것 (115): contact (관절 번호) 의 높이를 모두 0 (같은 높이) 으로 고정. 책상에 붙인 손가락의 끝 쪽 관절은 높이가 거의 같다는 물리 조건이다.
    높이 원점은 이 조건이 정하므로 고정 관절 a 는 (x, y) 만 고정. 시작 모양은 닿는 관절 높이의 평균이 0 이 되게 옮기고(책상 위치 td 로 보정) 그 관절들을 0 으로 둔다."""
    import cv2
    from scipy.optimize import least_squares
    from scipy.optimize._numdiff import approx_derivative, group_columns
    from scipy.sparse import lil_matrix
    import handfocal as hf
    frames = [pr.frames[i] for i in pr.keep]
    N = len(frames)
    S0 = np.asarray(init["S"], float).copy()
    J = len(S0)
    ci = [list(ids).index(int(j)) for j in contact if int(j) in list(ids)]
    zc = float(S0[ci, 2].mean())
    S0[:, 2] -= zc
    S0[ci, 2] = 0.0
    td0 = np.asarray(init["td"], float) + np.asarray(init["Rd"], float)[:, 2] * zc
    r = int(init["ref"])
    a, b = hf._anchors(S0[:, :2], frames)
    fixed = np.zeros((J, 3), bool)
    fixed[a, :2] = True
    fixed[b, :2] = True
    fixed[ci, 2] = True
    sfree = np.flatnonzero(~fixed.ravel())
    fr_free = np.array([k for k in range(N) if k != r], int)
    fi = np.concatenate([np.full(v.sum(), k) for k, (o, v) in enumerate(frames)])
    ji = np.concatenate([np.flatnonzero(v) for o, v in frames])
    uv = np.concatenate([o[v] for o, v in frames])
    M = len(ji)
    ng = int(free_f) + 6
    ns = len(sfree)
    hp = float(np.median(pr.huber[pr.keep]))
    size = float(np.sqrt(((S0[:, :2] - S0[:, :2].mean(0)) ** 2).sum(1)).mean()) if J else 1.0
    zs = None if z_rel is None else float(z_rel) * size

    def unpack(x):
        k = 0
        ff = float(np.exp(x[0])) if free_f else float(init["f"])
        k += int(free_f)
        Rd = hf._rodrigues(x[k:k + 3][None])[0]
        td = x[k + 3:k + 6]
        k += 6
        S = S0.ravel().copy()
        S[sfree] = x[k:k + ns]
        k += ns
        P = np.zeros((N, 3))
        P[fr_free] = x[k:].reshape(-1, 3)
        return ff, Rd, td, S.reshape(J, 3), P

    def proj(x):
        ff, Rd, td, S, P = unpack(x)
        c, s_ = np.cos(P[fi, 0]), np.sin(P[fi, 0])
        h = S[ji]
        Xd = np.c_[c * h[:, 0] - s_ * h[:, 1] + P[fi, 1], s_ * h[:, 0] + c * h[:, 1] + P[fi, 2], h[:, 2]]
        X = Xd @ Rd.T + td
        return np.c_[ff * X[:, 0] / X[:, 2] + pr.cx, ff * X[:, 1] / X[:, 2] + pr.cy], S

    def resid(x):
        p, S = proj(x)
        res = (p - uv).ravel()
        if zs:
            res = np.r_[res, hp * (S[:, 2] - S[:, 2].mean()) / zs]
        return res

    x0 = []
    if free_f:
        x0.append([np.log(init["f"])])
    x0 += [cv2.Rodrigues(np.asarray(init["Rd"], float))[0].ravel(), td0, S0.ravel()[sfree]]
    x0.append(np.c_[init["theta"], init["xy"]][fr_free].ravel())
    x0 = np.concatenate(x0).astype(float)
    lo, hi = np.full(len(x0), -np.inf), np.full(len(x0), np.inf)
    if free_f:
        lo[0], hi[0] = np.log(f_bounds[0]), np.log(f_bounds[1])
        x0[0] = np.clip(x0[0], lo[0] + 1e-9, hi[0] - 1e-9)
    nr = 2 * M + (J if zs else 0)
    Sp = lil_matrix((nr, len(x0)), dtype=int)
    Sp[:2 * M, :ng] = 1
    col_s = np.full(3 * J, -1)
    col_s[sfree] = ng + np.arange(ns)
    pos = {int(k): ng + ns + 3 * m for m, k in enumerate(fr_free)}
    for m in range(M):
        for d in range(3):
            c = col_s[3 * int(ji[m]) + d]
            if c >= 0:
                Sp[2 * m:2 * m + 2, c] = 1
        if int(fi[m]) in pos:
            Sp[2 * m:2 * m + 2, pos[int(fi[m])]:pos[int(fi[m])] + 3] = 1
    if zs:
        zc = col_s[3 * np.arange(J) + 2]
        for j in range(J):
            Sp[2 * M + j, zc[zc >= 0]] = 1
    Sp = Sp.tocsc()
    grp = group_columns(Sp)

    def jac(x):
        return approx_derivative(resid, x, sparsity=(Sp, grp), bounds=(lo, hi)).toarray()
    sol = least_squares(resid, x0, jac=jac, bounds=(lo, hi), loss="huber", f_scale=hp, x_scale="jac", max_nfev=max_nfev, method="trf", tr_solver="exact")
    ff, Rd, td, S, P = unpack(sol.x)
    e = np.hypot(*(proj(sol.x)[0] - uv).T)
    lossv = np.where(e <= hp, e * e, 2 * hp * e - hp * hp)
    zr = S[:, 2] - S[:, 2].min()
    return {"f": ff, "desk_tilt_deg": float(np.degrees(np.arccos(min(abs(Rd[2, 2]), 1.0)))), "rms_px": float(np.sqrt(np.mean(e ** 2))), "cost": float(lossv.mean()),
            "height_rel": float(np.ptp(S[:, 2]) / max(size, 1e-12)), "success": bool(sol.success), "nfev": int(sol.nfev),
            "at_bound": bool(free_f and (sol.x[0] <= lo[0] + 1e-6 or sol.x[0] >= hi[0] - 1e-6)),
            "height_per_joint": (zr / max(size, 1e-12)).tolist(), "joint_ids": [int(j) for j in ids],
            "state": {"f": ff, "S": S, "Rd": Rd, "td": td, "theta": P[:, 0], "xy": P[:, 1:], "ref": r}}


def dk3c_estimate(pb, grid: np.ndarray, A: dict, ramp_init: float, contact, z_rel=None, coarse_n: int = 8, max_frames: int = 120, n_final: int = 3) -> dict:
    """dk3_estimate 와 같고 풀이만 dk3c_solve (닿는 관절 contact 의 높이 고정). dk_estimate_ms 와 같은 짜임: f 격자 coarse_n 곳에서 f 고정으로 풀고(시작 = dk3_init), 이웃 해에서 다시 풀어 더 낮으면 바꾸고, 비용 낮은 n_final 곳에서 f 까지 푼다."""
    import handfocal as hf
    pr, ids = hf._restrict(pb, max_frames=max_frames)
    if list(ids) != list(A["joint_ids"]):
        raise ValueError("free_template result was computed on different joints")
    jid = [int(j) for j in A["joint_ids"]]
    T2 = np.asarray(A["template"], float)
    g = grid[np.unique(np.linspace(0, len(grid) - 1, coarse_n).round().astype(int))] if len(grid) > coarse_n else grid
    fb = (grid[0], grid[-1])
    prof = [dk3c_solve(pr, float(f), dk3_init(pr, float(f), T2, jid, ramp_init, fb), jid, False, fb, contact, z_rel) for f in g]
    for d, idx in ((-1, range(1, len(g))), (1, range(len(g) - 2, -1, -1))):
        for i in idx:
            x = dk3c_solve(pr, float(g[i]), dict(prof[i + d]["state"], f=float(g[i])), jid, False, fb, contact, z_rel)
            if x["cost"] < prof[i]["cost"]:
                prof[i] = x
    c = np.array([x["cost"] for x in prof])
    finals = []
    for k in np.argsort(c)[:n_final]:
        k = int(k)
        loc = 0 < k < len(g) - 1 and c[k] <= c[k - 1] and c[k] <= c[k + 1]
        fs = float(np.exp(np.clip(hf._parabola(np.log(g), c, k), np.log(fb[0]), np.log(fb[1])))) if loc else float(g[k])
        B = dk3c_solve(pr, fs, dict(prof[k]["state"], f=fs), jid, True, fb, contact, z_rel)
        finals.append((k, fs, B))
    k, fs, B = min(finals, key=lambda t: t[2]["cost"])
    B = dict(B)
    B.pop("state")
    B.update({"f_start": fs, "profile_f": g.tolist(), "profile_cost": c.tolist(), "starts": [{"k": kk, "f_start": ff, "f": bb["f"], "cost": bb["cost"]} for kk, ff, bb in finals]})
    return B


def dk3c_job(args) -> dict:
    """한 추정: args = (kp, mask, frames, box, size_wh, ecfg, ids, A, ramp_init, z_rel, contact). od_bridge -> dk3c_estimate. 실패도 결과로."""
    import os
    import time
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import cv2
    cv2.setNumThreads(1)
    import handfocal as hf
    kp, mask, frames, box, size_wh, ecfg, ids, A, ramp_init, z_rel, contact = args
    grid = hf.f_grid(ecfg, max(*size_wh))
    t0 = time.time()
    try:
        pb = hf.od_bridge(kp, mask, frames, box, size_wh, ecfg, ids)
        B = dk3c_estimate(pb, grid, A, ramp_init, contact, z_rel)
        B["boundary"] = bool(B["at_bound"]) or hf.od_boundary(float(B["f"]), grid)
    except Exception as e:
        B = {"failed": f"{type(e).__name__}: {e}"}
    B["elapsed_s"] = time.time() - t0
    return B

