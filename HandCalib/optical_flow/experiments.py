"""광류 방식의 원인 확인 실험 (run.py 를 거치지 않음).

  python optical_flow/experiments.py tracking-plane  --config configs/90_tracking_plane.yaml
  python optical_flow/experiments.py dorsum-corners  --config configs/91_dorsum_clahe.yaml
  python optical_flow/experiments.py <명령> --config <설정> results-yaml --run <실행 폴더>

tracking-plane (90): 추적 사슬이 손의 어느 평면을 따라가는지 본다. 기준 프레임 -> 각 프레임의 사슬 변환을, 같은 두 프레임의 검출 관절로 맞춘
                     세 변환(손등 몸통 관절 / 손가락 관절 / j16)과 비교한다. 손 회전 각도 구간별로도 나눈다. f 추정 없음.
dorsum-corners (91): 특징점을 손등 몸통에서만 고르기, 특징점을 고를 때만 대비 보정(CLAHE), 둘 다. 추적점과 검출 관절의 거리, 책상 운동 추정기의 f.
설정은 run_optical_flow.load_settings 로 74b 의 설정 사슬을 읽는다. 기준 f 와 오차는 summary.md 에만 적는다.
"""
import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import handfocal as hf                             # noqa: E402
import optical_flow as of                          # noqa: E402
from optical_flow import run_optical_flow as rof   # noqa: E402  (설정 읽기, 로그, 추정 입력)

DORSUM = [0, 1, 5, 9, 13, 17]                      # 손등 몸통: 손목, 엄지 뿌리, MCP 4개
FINGER = [6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]      # 손가락: PIP, DIP, 끝 (검지~소지)


# ----------------------------------------------------------------------------------------------------
# 공통: 한 영상의 프레임·검출·추적
# ----------------------------------------------------------------------------------------------------
def prepare(e: dict, cfg: dict, run_root: str):
    """프레임 추출(30 fps), MediaPipe 검출, 추적. 프레임 폴더는 호출한 쪽이 지운다. 반환: (pngs, times, tr, (W, H), tmp 폴더)."""
    pc = copy.deepcopy(cfg["pipeline"])
    pc["frames"]["detect_fps"] = int(cfg["tracking"]["fps"])
    v = e["file"]
    tmp = Path(run_root) / f"_frames_{Path(v).stem}"
    tmp.mkdir(exist_ok=True)
    prep = hf.fresp_prepare((v, cfg["input_dir"], pc, {}, str(tmp)))
    frames, (W, H) = prep["frames"], prep["size"]
    times = np.array([r["time_s"] for r in frames])
    npz = hf.fresp_detect_many([((v, "orig"), frames)], cfg["detector_settings"]["mediapipe_hands"], Path(run_root) / "detection_cache", 1, lambda s: None)[(v, "orig")]
    d = hf.load_detections(Path(npz))
    hf.check_alignment(frames, d)
    p2 = copy.deepcopy(pc)
    p2["tracking"]["conf_min"] = float(cfg["detector_settings"]["mediapipe_hands"]["conf_min"])
    tr = hf.track(d, times, (W, H), p2["tracking"], front_start_s=float(p2["phases"]["front_skip_s"]))
    return [r["png"] for r in frames], times, tr, (W, H), tmp


def hand_angle(tr: dict) -> np.ndarray:
    """프레임마다 손 방향(검출 손목 -> 중지 MCP, 도, 이어 붙임). 검출이 없으면 NaN."""
    ok = tr["present"] & np.isfinite(tr["kp"][:, [0, 9]]).all((1, 2))
    a = np.full(len(tr["kp"]), np.nan)
    a[ok] = np.degrees(np.unwrap(np.arctan2(tr["kp"][ok, 9, 1] - tr["kp"][ok, 0, 1], tr["kp"][ok, 9, 0] - tr["kp"][ok, 0, 0])))
    return a


def region_drift(kp: np.ndarray, kd: np.ndarray, box: np.ndarray, use: np.ndarray, ids) -> np.ndarray:
    """프레임마다 관절 ids 에서 추적점과 검출 관절의 거리 중앙값 (검출 상자 긴 변 대비 %)."""
    ids = [int(j) for j in ids]
    out = np.full(len(kp), np.nan)
    k = use & np.isfinite(kp[:, ids]).all((1, 2)) & np.isfinite(kd[:, ids]).all((1, 2)) & np.isfinite(box).all(1)
    if k.any():
        out[k] = np.median(np.linalg.norm(kp[k][:, ids] - kd[k][:, ids], axis=2), axis=1) / of.box_long(np.asarray(box, float)[k]) * 100.0
    return out


def med(x):
    x = np.asarray(x, float)
    return float(np.nanmedian(x)) if np.isfinite(x).any() else None


# ----------------------------------------------------------------------------------------------------
# 90: 추적 사슬이 따라가는 평면
# ----------------------------------------------------------------------------------------------------
def tp_job(args):
    e, cfg, run_root = args
    import cv2
    cv2.setNumThreads(1)
    pngs, times, tr, (W, H), tmp = prepare(e, cfg, run_root)
    try:
        t = of.track_hand(pngs, tr["kp"], tr["present"], tr["box"], (W, H), cfg["tracking"])
        # track_hand 는 변환을 돌려주지 않으므로, 같은 사슬을 한 번 더 만든다(같은 입력이면 같은 결과)
        pres = np.flatnonzero(tr["present"])
        n = len(pngs)
        kps = np.stack([of.kt_carry(tr["present"], tr["kp"][:, j, k]) for j in range(21) for k in range(2)], axis=1).reshape(n, 21, 2)
        boxes = np.stack([of.kt_carry(tr["present"], tr["box"][:, k]) for k in range(4)], axis=1)
        ref = t["ref"]
        Hall = {**of.kt_chain(pngs, kps, boxes, list(range(ref, int(pres[0]) - 1, -1)), cfg["tracking"])["H"],
                **of.kt_chain(pngs, kps, boxes, list(range(ref, int(pres[-1]) + 1)), cfg["tracking"])["H"]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ang = hand_angle(tr)
    P = tr["kp"][ref]
    j16 = cfg["joint_ids"]
    sets = {"dorsum": DORSUM, "finger": FINGER, "j16": j16}
    rows = []
    for i in np.flatnonzero(tr["present"] & t["use"]):
        Hc = Hall.get(int(i))
        if i == ref or Hc is None or not np.isfinite(tr["kp"][i]).all():
            continue
        bl = float(of.box_long(tr["box"][i][None])[0])
        pc = of.kt_map(P, Hc)
        r = {"frame": int(i), "t": float(times[i]), "turn_deg": float(ang[i] - ang[ref]) if np.isfinite(ang[i]) and np.isfinite(ang[ref]) else None}
        for nm, ids in sets.items():
            Hx, _ = cv2.findHomography(P[ids].astype(np.float64), tr["kp"][i][ids].astype(np.float64), 0)
            r[f"d_{nm}"] = None if Hx is None else float(np.median(np.linalg.norm(pc - of.kt_map(P, Hx), axis=1)) / bl * 100)      # 사슬 변환과 그 평면 변환의 차이 (21개 관절에서)
            r[f"fit_{nm}"] = None if Hx is None else float(np.median(np.linalg.norm(of.kt_map(P[ids], Hx) - tr["kp"][i][ids], axis=1)) / bl * 100)      # 그 관절들이 평면 하나로 맞는 정도
            r[f"drift_{nm}"] = float(np.median(np.linalg.norm(pc[ids] - tr["kp"][i][ids], axis=1)) / bl * 100)      # 사슬로 옮긴 그 관절과 검출의 거리
        rows.append(r)
    return {"ref": ref, "ref_t": float(times[ref]), "n_frames": len(pngs), "rows": rows}, [f"  {e['file']}: 기준 {times[ref]:.1f} s, 비교 프레임 {len(rows)}"]


def tp_summary(res: dict, cfg: dict) -> str:
    bins = [(0, 30), (30, 60), (60, 90), (90, 180)]
    L = ["# 90 추적 평면 summary", "",
         "**광류 추적 사슬이 손의 어느 평면을 따라가는지 본다. 기준 프레임 -> 각 프레임의 사슬 변환을, 같은 두 프레임의 검출 관절로 맞춘 변환 세 가지(손등 몸통 관절 0·1·5·9·13·17 / 손가락 관절 "
         "PIP·DIP·끝 12개 / j16)와 비교한다. 판정 없음.**", "",
         "- d(X) = 사슬 변환과 X 변환으로 기준 프레임의 관절 21개를 옮겼을 때 위치 차이의 중앙값 (검출 상자 긴 변 대비 %). 작을수록 사슬이 X 평면을 따라간다.",
         "- 거리(X) = 사슬로 옮긴 X 관절과 그 프레임의 검출 관절의 거리. 맞춤(X) = X 관절이 평면 변환 하나로 맞는 정도(검출 잡음 + 평면이 아닌 정도).",
         "- 회전 = 기준 프레임 대비 손 방향 변화(검출 손목 -> 중지 MCP).", "",
         "## 1. 영상별 (프레임 중앙값)", "", "| 영상 | 묶음 | 기준 | 프레임 | d(손등) | d(손가락) | d(j16) | 거리(손등) | 거리(손가락) | 맞춤(손등) | 맞춤(손가락) | 사슬이 손가락 쪽에 더 가까운 프레임 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f1 = lambda x: "-" if x is None else f"{x:.1f}"      # noqa: E731
    for e in cfg["videos"]:
        x = res[e["file"]]
        R = x["rows"]
        g = lambda k: med([r[k] for r in R if r[k] is not None])      # noqa: E731
        closer = [r["d_finger"] < r["d_dorsum"] for r in R if r["d_finger"] is not None and r["d_dorsum"] is not None]
        L.append(f"| {e['file']} | {e['person']} | {x['ref_t']:.1f} s | {len(R)} | {f1(g('d_dorsum'))} | {f1(g('d_finger'))} | {f1(g('d_j16'))} | {f1(g('drift_dorsum'))} | {f1(g('drift_finger'))} | "
                 f"{f1(g('fit_dorsum'))} | {f1(g('fit_finger'))} | {100 * np.mean(closer):.0f}% |" if closer else " - |")
    L += ["", "## 2. 회전 구간별 (프레임 중앙값)", "", "| 영상 | 묶음 | 회전 | 프레임 | d(손등) | d(손가락) | 거리(손등) | 거리(손가락) |", "|---|---|---|---|---|---|---|---|"]
    for e in cfg["videos"]:
        R = [r for r in res[e["file"]]["rows"] if r["turn_deg"] is not None]
        for lo, hi in bins:
            S = [r for r in R if lo <= abs(r["turn_deg"]) < hi]
            if not S:
                continue
            g = lambda k: med([r[k] for r in S if r[k] is not None])      # noqa: E731
            L.append(f"| {e['file']} | {e['person']} | {lo}~{hi}° | {len(S)} | {f1(g('d_dorsum'))} | {f1(g('d_finger'))} | {f1(g('drift_dorsum'))} | {f1(g('drift_finger'))} |")
    L += ["", "## 한계", "", "- 검출 관절로 맞춘 변환에는 검출기 잡음과 관절 중심(피부가 아님)의 치우침이 들어 있다. 손등 6점, 손가락 12점으로 맞춘 변환이다.", ""]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------------------------------
# 91: 손등 특징점, 대비 보정
# ----------------------------------------------------------------------------------------------------
def dc_job(args):
    e, cfg, run_root = args
    import cv2
    cv2.setNumThreads(1)
    pngs, times, tr, (W, H), tmp = prepare(e, cfg, run_root)
    out = {}
    try:
        pres = np.flatnonzero(tr["present"])
        n = len(pngs)
        a, b = int(pres[0]), int(pres[-1])
        ref = of.choose_reference(tr["present"])
        kps = np.stack([of.kt_carry(tr["present"], tr["kp"][:, j, k]) for j in range(21) for k in range(2)], axis=1).reshape(n, 21, 2)
        boxes = np.stack([of.kt_carry(tr["present"], tr["box"][:, k]) for k in range(4)], axis=1)
        for nm, v in cfg["variants"].items():
            mj = DORSUM if v["mask"] == "dorsum" else None
            cl = cfg["clahe"] if v["clahe"] else None
            fwd = of.dc_chain(pngs, kps, boxes, list(range(ref, b + 1)), cfg["tracking"], mj, cl)
            bwd = of.dc_chain(pngs, kps, boxes, list(range(ref, a - 1, -1)), cfg["tracking"], mj, cl)
            kp, tb, valid, edge = of.chain_points(tr["kp"][ref], {**bwd["H"], **fwd["H"]}, n, W, H)
            use = valid & ~edge
            nc = fwd["n_corners"] + bwd["n_corners"]
            out[nm] = {"kp": kp, "box": tb, "use": use, "size": (W, H), "n_use": int(use.sum()), "breaks": fwd["breaks"] + bwd["breaks"],
                       "corners_per_step": med(nc) if nc else None, "inliers_per_step": med(fwd["inl"] + bwd["inl"]) if fwd["inl"] + bwd["inl"] else None,
                       "drift_j16": med(region_drift(kp, tr["kp"], tr["box"], use, cfg["joint_ids"])), "drift_dorsum": med(region_drift(kp, tr["kp"], tr["box"], use, DORSUM)),
                       "drift_finger": med(region_drift(kp, tr["kp"], tr["box"], use, FINGER)), "drift_wrist": med(region_drift(kp, tr["kp"], tr["box"], use, [0]))}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    lines = [f"  {e['file']} {nm}: 끊김 {o['breaks']}, 사용 {o['n_use']} / {n}, 특징점/단계 {o['corners_per_step']}, 거리 j16 {o['drift_j16']:.1f}% 손등 {o['drift_dorsum']:.1f}% "
             f"손가락 {o['drift_finger']:.1f}% 손목 {o['drift_wrist']:.1f}%" for nm, o in out.items()]
    return {"ref": ref, "ref_t": float(times[ref]), "variants": out}, lines


def dc_estimate(cfg: dict, vid: dict, log) -> dict:
    """변형마다 free_template 한 번, 그 뒤 경사 고정(ramps_deg)과 경사 함께 추정(free_starts_deg 중 비용 최소)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    keys = [(f, nm) for f, x in vid.items() for nm in x["variants"]]
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        fts = dict(zip(keys, ex.map(hf.tl_template_job, [rof.estimate_args(cfg, vid[f]["variants"][nm]) for f, nm in keys], chunksize=1)))
    todo = [(f"ramp_{r:g}", "ramp_fixed", r) for r in cfg["ramps_deg"]] + [(f"free_{r:g}", "ramp_free", r) for r in cfg["free_starts_deg"]]
    res, jobs, jk = {}, [], []
    for f, nm in keys:
        x = vid[f]["variants"][nm]
        A = fts[(f, nm)].get("A")
        grid = hf.f_grid(cfg["pipeline"]["estimate"], max(*x["size"]))
        res.setdefault(f, {})[nm] = {k: x[k] for k in ("n_use", "breaks", "corners_per_step", "inliers_per_step", "drift_j16", "drift_dorsum", "drift_finger", "drift_wrist")}
        res[f][nm]["free_template"] = {"failed": fts[(f, nm)].get("failed")} if A is None else \
            {"f": float(A["f"]), "boundary": hf.od_boundary(float(A["f"]), grid), "rms_px": float((A.get("ba") or {}).get("rms_px", np.nan)), "cost": float("inf")}
        if A is None:
            continue
        for mk, mode, r in todo:
            jobs.append(rof.estimate_args(cfg, x, A) + (mode, r))
            jk.append((f, nm, mk))
    log(f"  추정 {len(jobs)}건")
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        for (f, nm, mk), o in zip(jk, ex.map(hf.ms2_job, jobs, chunksize=1)):
            res[f][nm][mk] = o
    return res


def dc_summary(res: dict, cfg: dict, own: dict) -> str:
    prev = json.loads((hf.resolve(own["compare_run"]) / "results.json").read_text())["real"]
    js = cfg["joint_set"]
    ref_f = lambda e: float(e.get("ref_f_px") or cfg["ref_f"]["value"])      # noqa: E731
    best = lambda d: min([d[k] for k in d if k.startswith("free_") and isinstance(d[k], dict) and d[k].get("cost") not in (None, float("inf"))], key=lambda x: x["cost"], default=None)      # noqa: E731
    fe = lambda x, e: "-" if not x or "f" not in x else f"{100 * (x['f'] / ref_f(e) - 1):+.1f}%" + ("*" if x.get("boundary") else "")      # noqa: E731
    f1 = lambda x: "-" if x is None else f"{x:.1f}"      # noqa: E731
    vs = ", ".join(f"{k} = 마스크 {v['mask']}, 대비 보정 {'있음' if v['clahe'] else '없음'}" for k, v in cfg["variants"].items())
    L = ["# 91 손등 특징점 · 대비 보정 summary", "",
         f"**특징점을 고르는 영역과 영상을 바꾼 추적 사슬: {vs}. 대비 보정(CLAHE clip {cfg['clahe']['clip']}, tile {cfg['clahe']['tile']})은 특징점을 고를 때만 쓰고 추적(LK)은 원래 영상으로 한다. "
         f"기존 사슬(hand, 보정 없음)의 f 는 {own['compare_run']} 에서 읽는다. 판정 없음.**", "",
         "- 기준 f 는 이 파일에만. 오차 = 기준 f 대비. * = f 탐색 범위 끝. 거리 = 추적점과 검출 관절의 거리(검출 상자 긴 변 대비 %, 프레임 중앙값).", "",
         "## 1. 추적", "", "| 영상 | 묶음 | 사슬 | 끊김 | 사용 프레임 | 특징점/단계 | inlier/단계 | 거리 j16 | 거리 손등 | 거리 손가락 | 거리 손목 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in cfg["videos"]:
        for nm, o in res[e["file"]].items():
            L.append(f"| {e['file']} | {e['person']} | {nm} | {o['breaks']} | {o['n_use']} | {f1(o['corners_per_step'])} | {f1(o['inliers_per_step'])} | {f1(o['drift_j16'])} | {f1(o['drift_dorsum'])} | "
                     f"{f1(o['drift_finger'])} | {f1(o['drift_wrist'])} |")
    nms = list(cfg["variants"])
    L += ["", f"## 2. f ({js})", "", "| 영상 | 묶음 | 추정 | 기존 사슬 | " + " | ".join(nms) + " |", "|---|---|---|---|" + "---|" * len(nms)]
    for e in cfg["videos"]:
        p = prev.get(f"{e['file']}|{js}") or {}
        for lab, get in [("free_template", lambda d: d.get("free_template"))] + [(f"경사 {r:g}°", (lambda r: lambda d: d.get(f"ramp_{r:g}"))(r)) for r in cfg["ramps_deg"]] + [("함께 추정", best)]:
            cell = lambda d: fe(get(d), e) + (f" ({get(d)['pitch_deg']:.1f}°)" if lab == "함께 추정" and get(d) and "f" in get(d) else "")      # noqa: E731
            L.append(f"| {e['file']} | {e['person']} | {lab} | {cell(p)} | " + " | ".join(cell(res[e['file']][nm]) for nm in nms) + " |")
    L += ["", "## 한계", "", "- 영상 4개. 대비 보정 값은 하나만 썼다. 손등 마스크는 관절 0·1·5·9·13·17 의 볼록 껍질(손 마스크와 같은 양만큼 넓힘).", ""]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------------------------------
# 92: 두 평면 추정기 (optical_flow.two_plane)
# ----------------------------------------------------------------------------------------------------
def tw_points(args):
    """한 영상: 평면 A 관측(손 마스크 사슬 = kt_chain) 과 평면 B 관측(손등 몸통 마스크 사슬 = dc_chain) -> 두 사슬이 모두 유효하고 가장자리에 닿지 않는
    프레임을 max_frames 개 이하로 고르게 골라 둔다."""
    e, cfg, run_root = args
    import cv2
    cv2.setNumThreads(1)
    pngs, times, tr, (W, H), tmp = prepare(e, cfg, run_root)
    try:
        pres = np.flatnonzero(tr["present"])
        n = len(pngs)
        a, b = int(pres[0]), int(pres[-1])
        ref = of.choose_reference(tr["present"])
        kps = np.stack([of.kt_carry(tr["present"], tr["kp"][:, j, k]) for j in range(21) for k in range(2)], axis=1).reshape(n, 21, 2)
        boxes = np.stack([of.kt_carry(tr["present"], tr["box"][:, k]) for k in range(4)], axis=1)
        HA = {**of.kt_chain(pngs, kps, boxes, list(range(ref, a - 1, -1)), cfg["tracking"])["H"], **of.kt_chain(pngs, kps, boxes, list(range(ref, b + 1)), cfg["tracking"])["H"]}
        HB = {**of.dc_chain(pngs, kps, boxes, list(range(ref, a - 1, -1)), cfg["tracking"], DORSUM)["H"], **of.dc_chain(pngs, kps, boxes, list(range(ref, b + 1)), cfg["tracking"], DORSUM)["H"]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    P = tr["kp"][ref]
    kA, _, vA, eA = of.chain_points(P, HA, n, W, H)
    kB, _, vB, eB = of.chain_points(P, HB, n, W, H)
    ok = np.flatnonzero(vA & ~eA & vB & ~eB)
    m = int(cfg["two_plane"]["max_frames"])
    sel = ok if len(ok) <= m else ok[np.unique(np.linspace(0, len(ok) - 1, m).round().astype(int))]
    return {"ref": ref, "ref_t": float(times[ref]), "size": (W, H), "p_ref": P, "obsA": kA[sel], "obsB": kB[sel], "frames": sel, "n_ok": int(len(ok))}, \
        [f"  {e['file']}: 기준 {times[ref]:.1f} s, 두 사슬 모두 쓸 수 있는 프레임 {len(ok)}, 고른 프레임 {len(sel)}"]


def tw_job(args):
    """(영상 또는 합성 장면, 방식) 하나: two_plane.solve."""
    key, mode, x, grid, tc = args
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from optical_flow import two_plane as tp
    t0 = time.time()
    try:
        r = tp.solve(x["p_ref"], x["obsA"], x["obsB"], x["size"], np.asarray(grid), mode=mode, tilts=tuple(tc["start_tilts_deg"]), n_final=int(tc["n_final"]))
    except Exception as ex:                                            # 실패도 결과로 남김
        r = {"failed": f"{type(ex).__name__}: {ex}"}
    r["elapsed_s"] = time.time() - t0
    return key, mode, r


def tw_run(cfg: dict, vid: dict, log) -> dict:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from optical_flow import two_plane as tp
    tc = cfg["two_plane"]
    jobs = []
    for f, x in vid.items():
        grid = hf.f_grid({"f_ratio_range": tc["f_ratio_range"], "f_grid_n": tc["f_grid_n"]}, max(*x["size"])).tolist()
        for mode in tc["modes"]:
            jobs.append((f, mode, x, grid, tc))
    for sc in tc.get("synthetic", []):                                 # 합성 확인 (풀이기 점검용)
        s = tp.synthetic(f=float(sc["f"]), ramp_a=float(sc["ramp_a"]), ramp_b=float(sc["ramp_b"]), noise_px=float(sc["noise_px"]), n=int(sc["frames"]), hand_cm=hf.OD_HAND_CM, seed=int(sc.get("seed", 0)))
        grid = hf.f_grid({"f_ratio_range": tc["f_ratio_range"], "f_grid_n": tc["f_grid_n"]}, 1920).tolist()
        for mode in tc["modes"]:
            jobs.append((f"합성 f {sc['f']:g}, 경사 A {sc['ramp_a']:g}° B {sc['ramp_b']:g}°, 잡음 {sc['noise_px']:g} px", mode, s, grid, tc))
    log(f"  풀이 {len(jobs)}건")
    res = {}
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        for key, mode, r in ex.map(tw_job, jobs, chunksize=1):
            res.setdefault(key, {})[mode] = r
            log(f"  {key} {mode}: " + (r["failed"] if "failed" in r else f"f {r['f']:.0f}, RMS {r['rms_px']:.2f} px, 평면 A {r['plane_a_deg']:.1f}° B {r['plane_b_deg']:.1f}°, 책상 {r['desk_tilt_deg']:.1f}°")
                + f" ({r['elapsed_s']:.0f} s)")
    for f, x in vid.items():
        res[f]["info"] = {"ref_t": x["ref_t"], "n_ok": x["n_ok"], "n_used": int(len(x["frames"]))}
    return res


def tw_summary(res: dict, cfg: dict, own: dict) -> str:
    prev = json.loads((hf.resolve(own["compare_run"]) / "results.json").read_text())["real"]
    js = cfg["joint_set"]
    tc = cfg["two_plane"]
    best = lambda d: min([d[k] for k in d if k.startswith("free_") and isinstance(d[k], dict) and d[k].get("cost") not in (None, float("inf"))], key=lambda x: x["cost"], default=None)      # noqa: E731
    lab = {"A": "평면 A만 (손 마스크 사슬)", "B": "평면 B만 (손등 몸통 사슬)", "AB": "두 평면 함께"}
    L = ["# 92 두 평면 추정기 summary", "",
         "**손가락 평면(손 마스크 사슬, 90 에서 손가락 평면을 따라감)과 손등 평면(손등 몸통 마스크 사슬, 91)을 함께 맞춘다. 평면 A·B 의 방향, 책상 법선, f, 프레임마다 책상 법선 둘레 회전과 "
         "책상 면 안 미끄러짐을 함께 풀고, 손 경사는 가정하지 않는다(결과로 나옴). 비교로 같은 풀이를 평면 하나씩에도 돌린다. 74b(경사 11° 고정, 함께 추정)의 값은 옆에. 판정 없음.**", "",
         f"- f 격자: 긴 변 x {tc['f_ratio_range']} 에서 {tc['f_grid_n']}개, 시작 책상 기울기 {tc['start_tilts_deg']}°, 회전 방향 두 가지, 낮은 {tc['n_final']}곳에서 f 까지 풀어 비용 최소. "
         f"프레임 최대 {tc['max_frames']}개, 관측 점 = 기준 프레임 검출 관절 21개를 각 사슬로 옮긴 것.",
         "- 기준 f 는 이 파일에만. 오차 = 기준 f 대비. * = f 격자 끝. 평면 각 = 그 평면과 책상 사이 각(손 경사에 해당).", "",
         "| 영상 / 장면 | 기준 f | 방식 | f (오차) | RMS (px) | 평면 A 각 | 평면 B 각 | 두 평면 사이 각 | 책상 기울기 | 회전 범위 |", "|---|---|---|---|---|---|---|---|---|---|"]
    vids = {e["file"]: e for e in cfg["videos"]}
    for key, d in res.items():
        e = vids.get(key)
        rf = float(e.get("ref_f_px") or cfg["ref_f"]["value"]) if e else float(key.split("f ")[1].split(",")[0])
        if e:
            p = prev.get(f"{key}|{js}") or {}
            for nm, x in (("74b 경사 11° 고정", p.get("ramp_11")), ("74b 경사 함께 추정", best(p))):
                L.append(f"| {key} | {rf:g} | {nm} | " + ("-" if not x or "f" not in x else f"{x['f']:.0f} ({100 * (x['f'] / rf - 1):+.1f}%)") + " | - | " +
                         ("-" if not x or "pitch_deg" not in x else f"{x['pitch_deg']:.1f}°") + " | - | - | - | - |")
        for mode in tc["modes"]:
            x = d.get(mode) or {}
            if "f" not in x:
                L.append(f"| {key} | {rf:g} | {lab[mode]} | 실패: {x.get('failed')} | | | | | | |")
                continue
            L.append(f"| {key} | {rf:g} | {lab[mode]} | {x['f']:.0f} ({100 * (x['f'] / rf - 1):+.1f}%){'*' if x['at_bound'] else ''} | {x['rms_px']:.2f} | {x['plane_a_deg']:.1f}° | "
                     + ("-" if mode != "AB" else f"{x['plane_b_deg']:.1f}°") + " | " + ("-" if mode != "AB" else f"{x['dihedral_deg']:.1f}°") + f" | {x['desk_tilt_deg']:.1f}° | {x['turn_range_deg']:.0f}° |")
    L += ["", "## 비용 곡선 (f 고정 풀이의 비용, 최솟값 = 1)", "", "| 영상 / 장면 | 방식 | " + " | ".join("f 격자 %d" % (i + 1) for i in range(int(tc["f_grid_n"]))) + " |",
          "|---|---|" + "---|" * int(tc["f_grid_n"])]
    for key, d in res.items():
        for mode in tc["modes"]:
            x = d.get(mode) or {}
            if "profile_cost" in x:
                c = np.array(x["profile_cost"])
                L.append(f"| {key} | {mode} | " + " | ".join(f"{v:.2f}" for v in c / max(c.min(), 1e-12)) + " |")
    L += ["", "## 한계", "", "- 두 사슬 모두 평면 하나로 옮긴 점이라, 실제 손이 두 평면이 아니면(엄지, 손목) 그만큼 맞지 않는다. 영상 4개.",
          "- 합성 장면은 무작위 잡음만 있어 평면 하나로도 f 가 정해진다. 풀이기가 참값을 되찾는지만 본다.", ""]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------------------------------
# 실행
# ----------------------------------------------------------------------------------------------------
def results_yaml(cmd: str, run_dir: Path, own: dict) -> Path:
    """값만 (기준 f, 오차 없음). 90 은 프레임별 행 대신 영상별 중앙값."""
    r = json.loads((run_dir / "results.json").read_text())["real"]
    m = json.loads((run_dir / "metadata.json").read_text())
    rv = lambda x: x if not isinstance(x, float) else (round(x, 3) if np.isfinite(x) else None)      # noqa: E731
    out = {"experiment": own["experiment"]["name"], "run": str(run_dir.resolve().relative_to(hf.PROJECT_ROOT)), "code_commit": m["git"]["commit"][:7],
           "git_dirty_at_run": m["git"]["git_dirty"], "videos": {}}
    for f, x in r.items():
        if cmd == "tracking-plane":
            R = x["rows"]
            out["videos"][f] = {"ref_t": rv(x["ref_t"]), "n_rows": len(R), **{k: rv(med([q[k] for q in R if q[k] is not None])) for k in R[0] if k.startswith(("d_", "drift_", "fit_"))}} if R else {}
        else:
            keys = ("f", "rms_px", "pitch_deg", "roll_deg", "desk_tilt_deg", "boundary", "start")
            out["videos"][f] = {nm: {k: (rv(v) if not isinstance(v, dict) else ({kk: rv(v[kk]) for kk in keys if kk in v} if "f" in v else v)) for k, v in o.items()} for nm, o in x.items()}
    p = hf.resolve(own["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


def tw_results_yaml(run_dir: Path, own: dict) -> Path:
    """92 의 값만: 실제 영상만 (합성 장면은 참 f 가 기준 f 와 같은 값이라 넣지 않음), 비용 곡선 제외."""
    r = json.loads((run_dir / "results.json").read_text())["real"]
    m = json.loads((run_dir / "metadata.json").read_text())
    rv = lambda x: x if not isinstance(x, float) else (round(x, 3) if np.isfinite(x) else None)      # noqa: E731
    out = {"experiment": own["experiment"]["name"], "run": str(run_dir.resolve().relative_to(hf.PROJECT_ROOT)), "code_commit": m["git"]["commit"][:7],
           "git_dirty_at_run": m["git"]["git_dirty"], "videos": {}}
    for f in own["videos"]:
        d = r.get(f) or {}
        out["videos"][f] = {k: ({kk: rv(vv) for kk, vv in v.items() if not kk.startswith("profile")} if isinstance(v, dict) else v) for k, v in d.items()}
    p = hf.resolve(own["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description="광류 방식 원인 확인 실험")
    ap.add_argument("cmd", choices=["tracking-plane", "dorsum-corners", "two-plane"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--videos", type=int, default=None, help="시험 실행: 앞의 N 개 영상만, 추정 1건씩")
    sub = ap.add_subparsers(dest="sub")
    ry = sub.add_parser("results-yaml")
    ry.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    text = Path(a.config).read_text()
    own = yaml.safe_load(text)
    if a.sub == "results-yaml":
        print((tw_results_yaml(hf.resolve(a.run), own)) if a.cmd == "two-plane" else results_yaml(a.cmd, hf.resolve(a.run), own))
        return None
    cfg = rof.load_settings(own)
    cfg.update({k: own[k] for k in ("variants", "clahe", "two_plane") if k in own})
    if a.videos:
        cfg["videos"] = cfg["videos"][:a.videos]
        cfg["free_starts_deg"] = []
        if "two_plane" in cfg:
            cfg["two_plane"] = {**cfg["two_plane"], "start_tilts_deg": [45], "n_final": 1, "synthetic": cfg["two_plane"].get("synthetic", [])[:1]}
    import datetime as dt
    import os
    run_root = Path(a.out).expanduser() if a.out else hf.resolve(cfg["output"]["root"]) / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root.mkdir(parents=True)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    log = rof.log_to(run_root / "log.txt")
    log(f"{a.cmd}: 영상 {len(cfg['videos'])}개 → {run_root}")
    t0 = time.time()
    if own.get("detection_cache_from"):
        shutil.copytree(hf.resolve(own["detection_cache_from"]), run_root / "detection_cache")
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    job = {"tracking-plane": tp_job, "dorsum-corners": dc_job, "two-plane": tw_points}[a.cmd]
    vid = {}
    with ProcessPoolExecutor(max_workers=int(cfg["video_workers"]), mp_context=mp.get_context("spawn")) as ex:
        for e, (x, lines) in zip(cfg["videos"], ex.map(job, [(e, cfg, str(run_root)) for e in cfg["videos"]])):
            vid[e["file"]] = x
            for s in lines:
                log(s)
    log(f"  추적: {time.time() - t0:.0f} s")
    if a.cmd == "tracking-plane":
        res = vid
        text_sum = tp_summary(res, cfg)
    elif a.cmd == "two-plane":
        res = tw_run(cfg, vid, log)
        text_sum = tw_summary(res, cfg, own)
    else:
        res = dc_estimate(cfg, vid, log)
        text_sum = dc_summary(res, cfg, own)
    (run_root / "results.json").write_text(json.dumps(hf.jsonable({"real": res}), ensure_ascii=False, indent=1))
    (run_root / "summary.md").write_text(text_sum)
    meta = hf.provenance_build(text, a.config, {"code": own.get("code"), "elapsed_s": time.time() - t0})
    (run_root / "metadata.json").write_text(json.dumps(hf.jsonable(meta), ensure_ascii=False, indent=1))
    log(f"완료 {time.time() - t0:.0f} s → {run_root}")
    log.close()
    return run_root


if __name__ == "__main__":
    main()
