"""광류 방식의 원인 확인 실험 (run.py 를 거치지 않음).

  python optical_flow/experiments.py tracking-plane  --config configs/90_tracking_plane.yaml
  python optical_flow/experiments.py dorsum-corners  --config configs/91_dorsum_clahe.yaml
  python optical_flow/experiments.py geometry-synth  --config configs/96_geometry_synth.yaml
  python optical_flow/experiments.py pause-plane     --config configs/97_pause_plane.yaml
  python optical_flow/experiments.py cotracker       --config configs/98_cotracker.yaml
  python optical_flow/experiments.py rotation-side   --config configs/99_rotation_side.yaml
  python optical_flow/experiments.py <명령> --config <설정> results-yaml --run <실행 폴더>

tracking-plane (90): 추적 사슬이 손의 어느 평면을 따라가는지 본다. 기준 프레임 -> 각 프레임의 사슬 변환을, 같은 두 프레임의 검출 관절로 맞춘
                     세 변환(손등 몸통 관절 / 손가락 관절 / j16)과 비교한다. 손 회전 각도 구간별로도 나눈다. f 추정 없음.
dorsum-corners (91): 특징점을 손등 몸통에서만 고르기, 특징점을 고를 때만 대비 보정(CLAHE), 둘 다. 추적점과 검출 관절의 거리, 책상 운동 추정기의 f.
geometry-synth (96): 영상의 카메라 기울기·손 모양·프레임별 움직임을 그대로 둔 합성 장면을 참 f 로 만들어 같은 추정을 돌린다 (촬영 구도·동작 탓인지, 실제 점 탓인지).
pause-plane (97): 멈춤 구간마다 손 평면 방향을 알려진 f 로 호모그래피 분해해 손 경사가 자세마다 바뀌는지 본다 (실제 점, 합성 없음).
rotation-side (99): 97 이 저장한 추적점을 손 방향으로 나눠(한쪽만 / 대칭으로 자름) 경사 고정 추정만 다시 돌린다.
cotracker (98): 광류 사슬 대신 CoTracker3 로 기준 프레임의 관절을 직접 추적해 같은 추정을 돌린다.
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
# 93: 관절별 따로 추적 + 점 영상
# ----------------------------------------------------------------------------------------------------
JT_GROUPS = {"wrist": [0], "thumb": [1, 2, 3, 4], "mcp": [5, 9, 13, 17], "pip": [6, 10, 14, 18], "dip_tip": [7, 8, 11, 12, 15, 16, 19, 20]}


def jt_reference(tr: dict, size_wh, cfg: dict) -> int:
    """기준 프레임. reference = "middle": 검출 범위의 한가운데(실험 56). "first_pause": 손을 돌리기 전 첫 정지 구간의 가운데 검출 프레임
    (정지 = 기본 설정의 still 규칙, 정지 구간 = 간격 pause_gap_frames 이하로 이어진 min_pause_frames 개 이상; 정지 구간이 없으면 middle)."""
    rule = cfg.get("reference", {"mode": "middle"})
    if rule["mode"] == "middle":
        return of.choose_reference(tr["present"])
    motion = hf.od_select(tr["kp"], tr["mask"], tr["present"], tr["box"], tuple(size_wh), cfg["selection"])["motion"]
    still = np.isfinite(motion) & (motion < float(cfg["selection"]["still_frac"])) & tr["present"]
    pauses = [np.asarray(g) for g in hf.hp_groups(np.flatnonzero(still), int(rule["pause_gap_frames"])) if len(g) >= int(rule["min_pause_frames"])]
    if not pauses:
        return of.choose_reference(tr["present"])
    g = pauses[0]
    return int(g[np.argmin(np.abs(g - (g[0] + g[-1]) / 2))])


def jt_render(path: Path, pngs: list, times: np.ndarray, frames: np.ndarray, kd: np.ndarray, kc: np.ndarray, kj: np.ndarray, how: np.ndarray, rc: dict, title: str):
    """점 영상: 왼쪽 전체 화면, 오른쪽 손 확대. 주황 = MediaPipe, 파랑 = 기존 사슬(평면 변환 하나), 초록 = 관절별 추적(속 빈 점 = 그 단계는 손 전체 변환으로 대신 옮김).
    프레임 이미지는 임시 폴더에 그린 뒤 ffmpeg 로 묶고(메타데이터 제거) 지운다."""
    import subprocess
    import tempfile
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    from optical_flow import JT_GLOBAL
    font = lambda s: ImageFont.truetype(rc["font"], s, index=int(rc.get("font_index", 0)))      # noqa: E731
    LW, RW, HD = int(rc["left_width"]), int(rc["zoom_size"]), 70
    ORANGE, BLUE, GREEN = (235, 104, 52), (42, 120, 214), (27, 175, 122)
    with tempfile.TemporaryDirectory(dir=str(path.parent)) as td:
        for n_out, i in enumerate(frames):
            img = cv2.imread(pngs[i])[:, :, ::-1]
            H, W = img.shape[:2]
            sc = LW / W
            left = Image.fromarray(img).resize((LW, int(round(H * sc))))
            pts = np.concatenate([x[np.isfinite(x).all(1)] for x in (kd[i], kc[i], kj[i])] or [np.zeros((0, 2))])
            if len(pts):
                cx, cy = (pts.min(0) + pts.max(0)) / 2
                half = max(60.0, float((pts.max(0) - pts.min(0)).max()) * float(rc["crop_margin"]) / 2)
            else:
                cx, cy, half = W / 2, H / 2, H / 4
            box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
            zoom = Image.fromarray(img).crop(box).resize((RW, RW))
            zs = RW / (2 * half)
            canvas = Image.new("RGB", (LW + RW, max(left.height, RW) + HD), (255, 255, 255))
            canvas.paste(left, (0, HD))
            canvas.paste(zoom, (LW, HD))
            d = ImageDraw.Draw(canvas)
            for kp, col, rad, hollow in ((kd[i], ORANGE, 4, None), (kc[i], BLUE, 4, None), (kj[i], GREEN, 5, how[i] == JT_GLOBAL)):
                for j in range(len(kp)):
                    if not np.isfinite(kp[j]).all():
                        continue
                    for (ox, oy, s_, r_) in ((0, HD, sc, rad - 1), (LW, HD, zs, rad + 2)):
                        x = ox + (kp[j][0] - (0 if ox == 0 else box[0])) * s_
                        y = oy + (kp[j][1] - (0 if ox == 0 else box[1])) * s_
                        if hollow is not None and bool(hollow[j]):
                            d.ellipse((x - r_, y - r_, x + r_, y + r_), outline=col, width=2)
                        else:
                            d.ellipse((x - r_, y - r_, x + r_, y + r_), fill=col, outline=(255, 255, 255))
            n_glob = int(np.sum(how[i] == JT_GLOBAL)) if np.isfinite(kj[i]).any() else 0
            d.text((10, 6), f"{title}   {times[i]:.2f} s   프레임 {i}   관절별 추적: 손 전체 변환으로 대신 옮긴 관절 {n_glob}/21", fill=(0, 0, 0), font=font(22))
            x0 = 10
            for col, lab in ((ORANGE, "MediaPipe 검출 관절"), (BLUE, "기존 추적(평면 변환 하나)"), (GREEN, "관절별 추적(속 빈 점 = 손 전체 변환으로 대신)")):
                d.ellipse((x0, 44, x0 + 14, 58), fill=col)
                d.text((x0 + 20, 38), lab, fill=(0, 0, 0), font=font(18))
                x0 += 30 + int(d.textlength(lab, font=font(18)))
            canvas.save(Path(td) / f"{n_out:06d}.png")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", f"{float(rc['fps']):.4f}", "-i", str(Path(td) / "%06d.png"), "-c:v", "libx264", "-crf", "20",
                        "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-map_metadata", "-1", str(path)], check=True)


def jt_job(args):
    """한 영상: 기존 사슬과 관절별 추적을 기준 프레임에서 앞뒤로, 검출 관절과의 거리, 점 영상."""
    e, cfg, run_root = args
    import cv2
    cv2.setNumThreads(1)
    pngs, times, tr, (W, H), tmp = prepare(e, cfg, run_root)
    try:
        pres = np.flatnonzero(tr["present"])
        n = len(pngs)
        a, b = int(pres[0]), int(pres[-1])
        ref = jt_reference(tr, (W, H), cfg)
        kps = np.stack([of.kt_carry(tr["present"], tr["kp"][:, j, k]) for j in range(21) for k in range(2)], axis=1).reshape(n, 21, 2)
        boxes = np.stack([of.kt_carry(tr["present"], tr["box"][:, k]) for k in range(4)], axis=1)
        tk = cfg["tracking"]
        Hc = {**of.kt_chain(pngs, kps, boxes, list(range(ref, a - 1, -1)), tk)["H"], **of.kt_chain(pngs, kps, boxes, list(range(ref, b + 1)), tk)["H"]}
        kc, _, vc, ec = of.chain_points(tr["kp"][ref], Hc, n, W, H)
        jb = of.jt_track(pngs, kps, boxes, list(range(ref, a - 1, -1)), tk, cfg["joint_track"], tr["kp"][ref])
        jf = of.jt_track(pngs, kps, boxes, list(range(ref, b + 1)), tk, cfg["joint_track"], tr["kp"][ref])
        kj = np.full((n, 21, 2), np.nan)
        how = np.full((n, 21), of.JT_BROKEN)
        for src in (jb, jf):
            for i, P in src["P"].items():
                kj[i], how[i] = P, src["how"][i]
        span = np.arange(a, b + 1)
        out = {"ref": ref, "ref_t": float(times[ref]), "n_frames": n, "how_share": {}, "drift": {}}
        hs = how[span]
        for nm, code in (("local", of.JT_LOCAL), ("wide", of.JT_WIDE), ("global", of.JT_GLOBAL), ("broken", of.JT_BROKEN)):
            out["how_share"][nm] = float(np.mean(hs == code))
        out["how_share_by_group"] = {g: {nm: float(np.mean(hs[:, ids] == code)) for nm, code in (("local", of.JT_LOCAL), ("wide", of.JT_WIDE), ("global", of.JT_GLOBAL))} for g, ids in JT_GROUPS.items()}
        use = vc & ~ec & tr["present"]
        for lab, kp in (("chain", kc), ("joint", kj)):
            out["drift"][lab] = {g: med(region_drift(kp, tr["kp"], tr["box"], use, ids)) for g, ids in {**JT_GROUPS, "j16": cfg["joint_ids"]}.items()}
        rc = {**cfg["render"], "fps": float(cfg["tracking"]["fps"])}
        vpath = Path(run_root) / f"{Path(e['file']).stem.replace(' ', '')}_joint_track.mp4"
        jt_render(vpath, pngs, times, span, tr["kp"], kc, kj, how, rc, f"{Path(e['file']).stem} ({'본인' if e['person'] == 'self' else '팀원'})")
        out["video"] = vpath.name
        out["kp_joint"], out["how"], out["kp_chain"] = kj, how, kc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    dj, dc = out["drift"]["joint"], out["drift"]["chain"]
    f1 = lambda x: "-" if x is None else f"{x:.1f}"      # noqa: E731
    return out, [f"  {e['file']}: 관절별 직접 {100 * out['how_share']['local']:.0f}% / 넓혀서 {100 * out['how_share']['wide']:.0f}% / 대신 {100 * out['how_share']['global']:.0f}%, "
                 f"거리(기존 -> 관절별) 손목 {f1(dc['wrist'])} -> {f1(dj['wrist'])}, 엄지 {f1(dc['thumb'])} -> {f1(dj['thumb'])}, j16 {f1(dc['j16'])} -> {f1(dj['j16'])}, 영상 {out['video']}"]


def jt_summary(res: dict, cfg: dict) -> str:
    f1 = lambda x: "-" if x is None else f"{x:.1f}"      # noqa: E731
    jc = cfg["joint_track"]
    L = ["# 93 관절별 따로 추적 summary", "",
         f"**관절마다 현재 위치 주변(반지름 = 손 상자 긴 변 x {jc['radius_frac']}) 특징점이 {jc['min_points']}개 이상이면 그 점들의 닮음 변환으로 그 관절만 옮기고, 모자라면 반지름 {jc['widen']}배, "
         "그래도 모자라면 그 단계만 손 전체 호모그래피(기존 방식)로 옮긴다. 특징점·LK·전후방 검사는 기존 사슬과 같다. 기준 프레임에서 앞뒤로. f 추정 없음, 판정 없음.**", "",
         "- 거리 = 추적점과 MediaPipe 검출 관절의 거리 (검출 상자 긴 변 대비 %, 프레임 중앙값). 검출 관절도 관절 중심 추정값이라 0 이 정답은 아니다.",
         f"- 기준 프레임: {cfg.get('reference', {'mode': 'middle'})['mode']} (기존 사슬과 관절별 추적 모두 같은 기준). 영상별 기준 시각은 아래 표.", "",
         "## 1. 검출 관절과의 거리: 기존 사슬 -> 관절별", "", "| 영상 | 손 주인 | 기준 시각 | 손목 | 엄지 | MCP | PIP | DIP·끝 | j16 | 영상 파일 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for e in cfg["videos"]:
        x = res[e["file"]]
        dc, dj = x["drift"]["chain"], x["drift"]["joint"]
        L.append(f"| {e['file']} | {'본인' if e['person'] == 'self' else '팀원'} | {x['ref_t']:.1f} s | " + " | ".join(f"{f1(dc[g])} -> {f1(dj[g])}" for g in ("wrist", "thumb", "mcp", "pip", "dip_tip", "j16")) + f" | {x['video']} |")
    L += ["", "## 2. 관절을 옮긴 방식의 비율 (관절 x 프레임)", "", "| 영상 | 묶음 | 직접 | 넓혀서 | 손 전체 변환으로 대신 |", "|---|---|---|---|---|"]
    for e in cfg["videos"]:
        x = res[e["file"]]
        L.append(f"| {e['file']} | 전체 | {100 * x['how_share']['local']:.0f}% | {100 * x['how_share']['wide']:.0f}% | {100 * x['how_share']['global']:.0f}% |")
        for g, v in x["how_share_by_group"].items():
            L.append(f"| | {g} | {100 * v['local']:.0f}% | {100 * v['wide']:.0f}% | {100 * v['global']:.0f}% |")
    L += ["", "## 한계", "", "- 관절마다 이웃 프레임끼리 이어 붙이므로 점마다 오차가 쌓일 수 있다. 영상 3개.", ""]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------------------------------
# 96: 영상의 구도·동작 그대로의 합성 장면 (원인 A 촬영 구도·동작 / B 실제 점 가르기)
# ----------------------------------------------------------------------------------------------------
def gs_fit(x: dict, cfg: dict, A: dict, f_fit: float) -> dict:
    """한 영상의 추적점에 책상 운동 모델(경사 ramp_deg 고정)을 f_fit 에서 맞춘 상태 (dk_solve, f 고정, 프로파일 한 점과 같은 시작).
    반환: 제한된 문제 pr, 모델 상태, 쓴 프레임(원래 프레임 번호), 관절 번호, RMS."""
    gc = cfg["geometry_synth"]
    pb = hf.od_bridge(x["kp"], np.isfinite(x["kp"]).all(2), np.flatnonzero(x["use"]), x["box"], tuple(x["size"]), cfg["pipeline"]["estimate"], cfg["joint_ids"])
    pr, _ = hf._restrict(pb, max_frames=int(gc["max_frames"]))
    jid = [int(j) for j in A["joint_ids"]]
    T2 = np.asarray(A["template"], float)
    grid = hf.f_grid(cfg["pipeline"]["estimate"], max(*x["size"]))
    s = hf.dk_solve(pr, float(f_fit), T2, jid, np.radians(float(gc["ramp_deg"])), 0.0, False, False, (grid[0], grid[-1]))
    return {"pr": pr, "state": s["state"], "rms_px": s["rms_px"], "T2": T2, "jid": jid, "frames": np.flatnonzero(x["use"])[pr.keep]}


def gs_render(fit: dict, f: float, depth_scale: float, axes_from: str = "fit") -> np.ndarray:
    """모델 상태를 f 로 투영 (dk_solve.proj 와 같은 식). 카메라와 책상 원점 사이 거리(td)만 depth_scale 배 -> 화면 속 손 크기는 거의 그대로, 원근만 f 에 맞게 바뀜.
    axes_from = "fit": 손 경사 축을 맞추기 전 템플릿(fit['T2'])에서 = dk_solve 가 쓴 모델 그대로 (실제 잔차 계산용).
    axes_from = "solved": 맞춘 뒤의 손 모양에서 = 합성 장면의 free_template 이 되찾는 모양과 같은 축 (합성 장면용).
    반환 (프레임, 관절, 2) 픽셀, 관절 순서 = fit['jid']."""
    st, pr = fit["state"], fit["pr"]
    u, w = hf.dk_axes(fit["T2"] if axes_from == "fit" else st["T"], fit["jid"])
    Hd = np.c_[st["T"], np.zeros(len(st["T"]))] @ hf.dk_hand_rot(u, w, st["up"] * st["pitch"], st["roll"]).T
    out = np.zeros((len(st["theta"]), len(st["T"]), 2))
    for k, (th, xy) in enumerate(zip(st["theta"], st["xy"])):
        c, s_ = np.cos(th), np.sin(th)
        Xd = np.c_[c * Hd[:, 0] - s_ * Hd[:, 1] + xy[0], s_ * Hd[:, 0] + c * Hd[:, 1] + xy[1], Hd[:, 2]]
        X = Xd @ st["Rd"].T + depth_scale * np.asarray(st["td"])
        out[k] = np.c_[f * X[:, 0] / X[:, 2] + pr.cx, f * X[:, 1] / X[:, 2] + pr.cy]
    return out


def gs_scenes(x: dict, cfg: dict, A: dict, f_fit: float, f_true: float) -> tuple:
    """한 영상의 합성 장면들 (각각 estimate_args 가 받는 x 모양: kp (프레임, 21, 2), use, box, size).
    - real_check: 모델(f_fit) + 실제 잔차 = 실제 점 그대로 (확인용, 실제 결과와 같아야 함)
    - clean: 같은 구도·동작·손 모양을 참 f 로 (td 만 f_true / f_fit 배), 잡음 없음
    - noise_s<시드>: clean + 손 상자 긴 변 x noise_rel 의 정규 잡음
    - residual: clean + 실제 잔차 (실제 점이 모델에서 벗어난 모양을 그대로 얹음)"""
    gc = cfg["geometry_synth"]
    fit = gs_fit(x, cfg, A, f_fit)
    pr, jid = fit["pr"], fit["jid"]
    vis = np.array([pr.frames[i][1] for i in pr.keep])
    obs = np.array([pr.frames[i][0] for i in pr.keep])
    m_fit = gs_render(fit, f_fit, 1.0)
    m_true = gs_render(fit, f_true, f_true / f_fit, axes_from="solved")
    res = np.where(vis[..., None], obs - m_fit, 0.0)
    box = x["box"][fit["frames"]]
    bl = np.maximum(box[:, 2] - box[:, 0], box[:, 3] - box[:, 1])

    def pack(P):
        kp = np.full((len(P), 21, 2), np.nan)
        kp[:, jid] = np.where(vis[..., None], P, np.nan)
        return {"kp": kp, "use": np.ones(len(P), bool), "box": box, "size": x["size"]}
    sc = {"real_check": pack(m_fit + res), "clean": pack(m_true)}
    for sd in gc["noise_seeds"]:
        g = np.random.default_rng([int(sd), 96])
        sc[f"noise_s{sd}"] = pack(m_true + g.normal(size=m_true.shape) * (float(gc["noise_rel"]) * bl)[:, None, None])
    sc["residual"] = pack(m_true + res)
    info = {"fit_rms_px": fit["rms_px"], "n_frames": int(len(pr.keep)), "residual_rms_px": float(np.sqrt(np.mean(np.sum(res[vis] ** 2, 1)))),
            "desk_tilt_deg": float(np.degrees(np.arccos(min(abs(fit["state"]["Rd"][2, 2]), 1.0)))), "turn_range_deg": float(np.degrees(np.ptp(fit["state"]["theta"])))}
    return sc, info


def gs_scene_job(args):
    """한 영상: 실제 추적점의 free_template, 94 의 f(경사 고정)에서 모델 맞추기, 합성 장면 만들기 (작업 프로세스)."""
    f, x, cfg, f_fit, f_true = args
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import cv2
    cv2.setNumThreads(1)
    A = hf.tl_template_job(rof.estimate_args(cfg, x))["A"]
    sc, info = gs_scenes(x, cfg, A, f_fit, f_true)
    return f, sc, info


def gs_run(cfg: dict, own: dict, vid: dict, log) -> dict:
    """96: 영상마다 합성 장면을 만들고, 각 장면에 94 와 같은 추정(free_template -> 경사 고정, 함께 추정)을 돌린다."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    gc = cfg["geometry_synth"]
    prev = json.loads((hf.resolve(gc["fit_run"]) / "results.json").read_text())["real"]
    rk = f"ramp_{float(gc['ramp_deg']):g}"
    ref_old = float(cfg["ref_f"]["value"])
    args = [(e["file"], vid[e["file"]], cfg, float(prev[e["file"]][rk]["f"]), float(e.get("ref_f_px") or ref_old)) for e in cfg["videos"]]
    res = {}
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        scenes = {f: (sc, info) for f, sc, info in ex.map(gs_scene_job, args, chunksize=1)}
    for (f, _, _, f_fit, _), e in zip(args, cfg["videos"]):
        res[f] = {"info": {**scenes[f][1], "f_fit": f_fit, "ref": vid[f]["ref"]}}
        log(f"  {f}: 모델 RMS {scenes[f][1]['fit_rms_px']:.2f} px (94 {prev[f][rk]['rms_px']:.2f}), 잔차 RMS {scenes[f][1]['residual_rms_px']:.2f} px, 장면 {len(scenes[f][0])}개")
    keys = [(f, s) for f in scenes for s in scenes[f][0]]
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        fts = dict(zip(keys, ex.map(hf.tl_template_job, [rof.estimate_args(cfg, scenes[f][0][s]) for f, s in keys], chunksize=1)))
    todo = [(rk, "ramp_fixed", float(gc["ramp_deg"]))] + [(f"free_{r:g}", "ramp_free", r) for r in cfg["free_starts_deg"]]
    jobs, jk = [], []
    for f, s in keys:
        x = scenes[f][0][s]
        A = fts[(f, s)].get("A")
        res[f][s] = {"free_template": {"failed": fts[(f, s)].get("failed")} if A is None else {"f": float(A["f"])}}
        if A is None:
            continue
        for mk, mode, r in todo:
            jobs.append(rof.estimate_args(cfg, x, A) + (mode, r))
            jk.append((f, s, mk))
    log(f"  추정 {len(jobs)}건")
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        for (f, s, mk), o in zip(jk, ex.map(hf.ms2_job, jobs, chunksize=1)):
            res[f][s][mk] = o
    return res


def gs_summary(res: dict, cfg: dict, own: dict) -> str:
    gc = cfg["geometry_synth"]
    rk = f"ramp_{float(gc['ramp_deg']):g}"
    best = lambda d: min([d[k] for k in d if k.startswith("free_") and isinstance(d[k], dict) and d[k].get("cost") not in (None, float("inf"))], key=lambda x: x["cost"], default=None)      # noqa: E731
    vids = {e["file"]: e for e in cfg["videos"]}
    lab = {"real_check": "실제 점 (모델 + 실제 잔차, 확인용)", "clean": "합성: 같은 구도·동작·손 모양, 참 f, 잡음 없음", "residual": "합성 + 실제 잔차"}
    L = ["# 96 영상 구도 그대로의 합성 장면 summary", "",
         f"**94 의 추적점(첫 정지 구간 기준, {cfg['joint_set']})에 책상 운동 모델을 94 의 f(경사 {gc['ramp_deg']:g}° 고정)로 맞춘 뒤, 그 카메라 기울기·손 모양(템플릿)·프레임별 회전과 미끄러짐을 그대로 두고 "
         "참 f 로 다시 투영한 합성 장면에 같은 추정을 돌린다. 카메라와 책상 사이 거리만 참 f / 94 f 배(화면 속 손 크기 유지). 합성에서 참 f 가 나오면 구도·동작 탓(A)이 아니다. "
         "실제 잔차를 얹은 장면에서 다시 틀리면 실제 점이 모델에서 벗어난 모양(B)이 f 를 민다. 판정 없음.**", "",
         f"- 정규 잡음 = 손 상자 긴 변 x {gc['noise_rel']}, 시드 {gc['noise_seeds']}. 프레임 최대 {gc['max_frames']}개(94 와 같음). 기준 f 는 이 파일에만. * = f 탐색 범위 끝.", "",
         "## 장면 정보", "", "| 영상 | 94 f (경사 고정) | 모델 RMS (px) | 실제 잔차 RMS (px) | 카메라 기울기 | 회전 범위 | 프레임 |", "|---|---|---|---|---|---|---|"]
    for f, d in res.items():
        i = d["info"]
        L.append(f"| {f} | {i['f_fit']:.0f} | {i['fit_rms_px']:.2f} | {i['residual_rms_px']:.2f} | {i['desk_tilt_deg']:.1f}° | {i['turn_range_deg']:.0f}° | {i['n_frames']} |")
    L += ["", "## f 오차", "", f"| 영상 | 묶음 | 기준 f | 장면 | 경사 {gc['ramp_deg']:g}° 고정 | 경사 함께 추정 (경사) | free_template |", "|---|---|---|---|---|---|---|"]
    for f, d in res.items():
        e = vids[f]
        rf = float(e.get("ref_f_px") or cfg["ref_f"]["value"])
        fmt = lambda x: "-" if not x or "f" not in x else f"{x['f']:.0f} ({100 * (x['f'] / rf - 1):+.1f}%)" + ("*" if x.get("boundary") else "")      # noqa: E731
        for s, o in d.items():
            if s == "info":
                continue
            b = best(o)
            L.append(f"| {f} | {e['person']} | {rf:g} | {lab.get(s, '합성 + 정규 잡음 ' + s[7:])} | {fmt(o.get(rk))} | {fmt(b)}" + ("" if not b else f" ({b['pitch_deg']:.1f}°)") + f" | {fmt(o.get('free_template'))} |")
    L += ["", "## 한계", "", "- 합성 장면의 손은 94 의 해에서 온 평면(경사 고정) 손이다. 실제 손의 두께나 손가락 굽힘은 '실제 잔차' 장면에만 들어 있다.",
          "- 실제 잔차는 94 의 f 에서 맞춘 모델 기준이라, 참 f 의 모델 기준 잔차와는 조금 다르다. 영상 4개.", ""]
    return "\n".join(L) + "\n"


def gs_results_yaml(run_dir: Path, own: dict) -> Path:
    """96 의 값만: 영상·장면별 f, 경사, RMS (기준 f, 오차 없음)."""
    r = json.loads((run_dir / "results.json").read_text())["real"]
    m = json.loads((run_dir / "metadata.json").read_text())
    rv = lambda x: x if not isinstance(x, float) else (round(x, 3) if np.isfinite(x) else None)      # noqa: E731
    keys = ("f", "rms_px", "pitch_deg", "roll_deg", "desk_tilt_deg", "boundary", "start")
    out = {"experiment": own["experiment"]["name"], "run": str(run_dir.resolve().relative_to(hf.PROJECT_ROOT)), "code_commit": m["git"]["commit"][:7],
           "git_dirty_at_run": m["git"]["git_dirty"], "videos": {}}
    for f, d in r.items():
        out["videos"][f] = {s: ({k: rv(v) for k, v in o.items()} if s == "info" else
                                {mk: ({kk: rv(v[kk]) for kk in keys if kk in v} if isinstance(v, dict) else v) for mk, v in o.items()}) for s, o in d.items()}
    p = hf.resolve(own["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


def gs_real_results_yaml(run_dir: Path, own: dict) -> Path:
    """96 의 값만, 실제 점 장면(real_check)과 장면 정보만. 합성 장면은 참 f 를 기준 f 로 두어 그 f 값이 기준 f 를 드러내므로 넣지 않는다 (92 와 같은 규칙, 값은 summary.md)."""
    r = json.loads((run_dir / "results.json").read_text())["real"]
    m = json.loads((run_dir / "metadata.json").read_text())
    rv = lambda x: x if not isinstance(x, float) else (round(x, 3) if np.isfinite(x) else None)      # noqa: E731
    keys = ("f", "rms_px", "pitch_deg", "roll_deg", "desk_tilt_deg", "boundary", "start")
    out = {"experiment": own["experiment"]["name"], "run": str(run_dir.resolve().relative_to(hf.PROJECT_ROOT)), "code_commit": m["git"]["commit"][:7],
           "git_dirty_at_run": m["git"]["git_dirty"], "videos": {}}
    for f, d in r.items():
        out["videos"][f] = {"info": {k: rv(v) for k, v in d["info"].items()},
                            "real_check": {mk: ({kk: rv(v[kk]) for kk in keys if kk in v} if isinstance(v, dict) else v) for mk, v in d.get("real_check", {}).items()}}
    p = hf.resolve(own["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p

# ----------------------------------------------------------------------------------------------------
# 추적점 저장 (97 부터: 다음 실험이 프레임 추출·검출·추적을 다시 하지 않고 읽어 씀)
# ----------------------------------------------------------------------------------------------------
def save_points(run_root, file: str, x: dict, size_wh, ang: np.ndarray, times: np.ndarray):
    """run_root/points/<영상 이름>.npz: 추적점 kp, 상자 box, 사용 프레임 use, 기준 프레임 ref, 크기, 프레임별 손 방향(검출, 도), 시각."""
    d = Path(run_root) / "points"
    d.mkdir(exist_ok=True)
    np.savez_compressed(d / f"{Path(file).stem}.npz", kp=x["kp"], box=x["box"], use=x["use"], ref=int(x["ref"]), size=np.asarray(size_wh), hand_angle=ang, times=times)


def load_points(points_run: str, file: str) -> dict:
    """save_points 로 저장한 한 영상의 추적점 (estimate_args 가 받는 모양 + hand_angle, times)."""
    z = np.load(hf.resolve(points_run) / "points" / f"{Path(file).stem}.npz")
    return {"kp": z["kp"], "box": z["box"], "use": z["use"].astype(bool), "ref": int(z["ref"]), "size": tuple(int(v) for v in z["size"]), "hand_angle": z["hand_angle"], "times": z["times"]}

# ----------------------------------------------------------------------------------------------------
# 97: 멈춤 구간마다 손 평면의 기울기 (실제 점, 알려진 f)
# ----------------------------------------------------------------------------------------------------
def pp_pauses(tr: dict, size_wh, cfg: dict) -> list:
    """멈춤 구간 목록 (choose_reference_pose 와 같은 규칙: still 규칙, 간격 pause_gap_frames 이하, min_pause_frames 개 이상)."""
    rule = cfg["reference"]
    motion = hf.od_select(tr["kp"], tr["mask"], tr["present"], tr["box"], tuple(size_wh), cfg["selection"])["motion"]
    still = np.isfinite(motion) & (motion < float(cfg["selection"]["still_frac"])) & tr["present"]
    return [np.asarray(g) for g in hf.hp_groups(np.flatnonzero(still), int(rule["pause_gap_frames"])) if len(g) >= int(rule["min_pause_frames"])]


def pp_normals(P: list, K: np.ndarray) -> dict:
    """멈춤마다의 점 P[k] (같은 관절, 픽셀) 로 첫 멈춤 -> 각 멈춤의 호모그래피를 구해 알려진 K 로 분해 (cv2.decomposeHomographyMat).
    해 후보 중 두 시점 모두 점이 카메라 앞에 있는 것만 남기고, 첫 멈춤의 평면 법선이 쌍들 사이에서 가장 많이 일치하는 후보를 고른다.
    반환: 첫 멈춤 기준 회전 R[k], 멈춤별 평면 법선 n[k] (카메라 좌표, 카메라 쪽을 향하게), 호모그래피 맞춤 잔차(px)."""
    import cv2
    Ki = np.linalg.inv(K)
    cands, fitres = [], []
    for k in range(1, len(P)):
        Hm, _ = cv2.findHomography(P[0].astype(np.float64), P[k].astype(np.float64), 0)
        y = np.c_[P[0], np.ones(len(P[0]))] @ Hm.T
        fitres.append(float(np.sqrt(np.mean(np.sum((y[:, :2] / y[:, 2:3] - P[k]) ** 2, 1)))))
        _, Rs, ts, ns = cv2.decomposeHomographyMat(Hm, K)
        m0 = (np.c_[P[0], np.ones(len(P[0]))] @ Ki.T)
        ok = []
        for R, t, n in zip(Rs, ts, ns):
            n, t = n.ravel(), t.ravel()
            d = m0 @ n                                                  # 평면 n.X = 1 위의 점 X = m / (n.m): 앞에 있으려면 n.m > 0
            if (d <= 0).any():
                continue
            X1 = (m0 / d[:, None]) @ R.T + t
            if (X1[:, 2] <= 0).any():
                continue
            ok.append((R, n / np.linalg.norm(n)))
        cands.append(ok)
    allc = [c[1] for cs in cands for c in cs]
    if not allc or any(not cs for cs in cands):
        return {"failed": "호모그래피 분해에서 앞에 있는 해가 없음"}
    score = [sum(min(np.degrees(np.arccos(np.clip(abs(a @ b[1]), -1, 1))) for b in cs) for cs in cands) for a in allc]
    n0 = allc[int(np.argmin(score))]
    Rk = [np.eye(3)] + [min(cs, key=lambda c: np.arccos(np.clip(abs(n0 @ c[1]), -1, 1)))[0] for cs in cands]
    ns = [R @ n0 for R in Rk]
    ns = [n if n[2] < 0 else -n for n in ns]                            # 카메라 쪽(-z)을 향하게
    return {"R": Rk, "n": ns, "fit_rms_px": fitres}


def pp_angles(out: dict) -> dict:
    """책상 법선 = 첫 멈춤 대비 회전들의 회전축(회전각 가중 평균, dk_init 과 같은 방식). 멈춤마다: 손 평면과 책상 사이 각(손 경사), 손 평면과 카메라 광축 사이 각.
    책상과 광축 사이 각(카메라 기울기)."""
    import cv2
    acc = np.zeros(3)
    for R in out["R"][1:]:
        rv = cv2.Rodrigues(R)[0].ravel()
        acc += rv if acc @ rv >= 0 else -rv
    a = acc / np.linalg.norm(acc)
    if a[2] > 0:
        a = -a
    ang = lambda u, v: float(np.degrees(np.arccos(np.clip(abs(u @ v), -1.0, 1.0))))      # noqa: E731
    z = np.array([0.0, 0.0, 1.0])
    return {"desk_tilt_deg": ang(a, z), "hand_desk_deg": [ang(n, a) for n in out["n"]], "hand_cam_deg": [ang(n, z) for n in out["n"]],
            "turn_deg": [float(np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0]))) for R in out["R"]], "fit_rms_px": [0.0] + out["fit_rms_px"]}


def pp_job(args):
    """한 영상: 프레임·검출·광류 사슬(첫 정지 구간 기준) -> 멈춤 구간마다 j 관절의 중앙값(추적점 / 검출 관절) -> f 마다 손 평면 기울기.
    f = 기준 f(reference) 와 fit_run 의 fit_key 추정 f(fit_run)."""
    e, cfg, run_root = args
    import cv2 as _cv2
    _cv2.setNumThreads(1)
    pc = cfg["pause_plane"]
    fit = json.loads((hf.resolve(pc["fit_run"]) / "results.json").read_text())["real"][e["file"]][pc["fit_key"]]["f"]
    fs = {"reference": float(e.get("ref_f_px") or cfg["ref_f"]["value"]), "fit_run": float(fit)}
    fs = {k: fs[k] for k in pc["f_sources"]}
    pngs, times, tr, (W, H), tmp = prepare(e, cfg, run_root)
    try:
        ref = rof.choose_reference_pose(tr, (W, H), cfg)
        x = of.track_hand(pngs, tr["kp"], tr["present"], tr["box"], (W, H), cfg["tracking"], ref=ref)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ids = cfg["joint_ids"]
    ang = hand_angle(tr)
    save_points(run_root, e["file"], x, (W, H), ang, times)
    pauses = pp_pauses(tr, (W, H), cfg)
    out = {"ref_t": float(times[ref]), "pauses": [], "by_points": {}}
    P = {"tracked": [], "detected": []}
    for g in pauses:
        gt = g[x["use"][g] & np.isfinite(x["kp"][g][:, ids]).all((1, 2))]
        gd = g[tr["present"][g] & np.isfinite(tr["kp"][g][:, ids]).all((1, 2))]
        if len(gt) == 0 or len(gd) == 0:
            continue
        out["pauses"].append({"t0": float(times[g[0]]), "t1": float(times[g[-1]]), "hand_angle_deg": med(ang[g] - ang[np.isfinite(ang)][0])})
        P["tracked"].append(np.median(x["kp"][gt][:, ids], 0))
        P["detected"].append(np.median(tr["kp"][gd][:, ids], 0))
    for src, Q in P.items():
        out["by_points"][src] = {}
        for name, f in fs.items():
            K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])
            r = pp_normals(Q, K) if len(Q) >= 3 else {"failed": f"멈춤 {len(Q)}개"}
            out["by_points"][src][name] = r if "failed" in r else {"f": float(f), **pp_angles(r)}
    return out, [f"  {e['file']}: 기준 {times[ref]:.1f} s, 멈춤 {len(out['pauses'])}개"]


def pp_summary(res: dict, cfg: dict, own: dict) -> str:
    pc = cfg["pause_plane"]
    L = ["# 97 멈춤 구간마다 손 평면 기울기 summary", "",
         f"**94 와 같은 광류 사슬(첫 정지 구간 기준)의 추적점과 MediaPipe 검출 관절({cfg['joint_set']})을 멈춤 구간마다 중앙값으로 모은다. 첫 멈춤 -> 각 멈춤의 호모그래피를 알려진 f 로 분해해 "
         "멈춤마다 손 평면의 방향을 구하고, 첫 멈춤 대비 회전들의 회전축을 책상 법선으로 둔다. 손 경사 = 손 평면과 책상 사이 각. 기준 f(촬영 기기 체커보드)와 94 의 추정 f(경사 11° 고정) 두 가지로 계산. "
         "합성 장면 없음, 판정 없음.**", "",
         "- 손 경사가 멈춤마다 거의 같으면 '손을 돌리는 동안 기울기가 변한다'는 가설은 맞지 않는다. 카메라 기울기 = 책상 법선과 광축 사이 각(촬영 때 수평계 45°).",
         "- 호모그래피 잔차 = 첫 멈춤 -> 그 멈춤의 평면 맞춤 잔차(px, 검출 관절은 평면이 아닌 만큼 커짐). 기준 f 는 이 파일에만.", ""]
    for src, lab in (("tracked", "광류 추적점"), ("detected", "MediaPipe 검출 관절")):
        L += [f"## {lab}", "", "| 영상 | 묶음 | f | 카메라 기울기 | 멈춤별 손 각도 | 멈춤별 손 경사 | 경사 범위 | 호모그래피 잔차 (px) |", "|---|---|---|---|---|---|---|---|"]
        for e in cfg["videos"]:
            d = res.get(e["file"]) or {}
            for name in pc["f_sources"]:
                x = (d.get("by_points", {}).get(src) or {}).get(name) or {}
                lab_f = "기준 f" if name == "reference" else "94 추정 f"
                if "hand_desk_deg" not in x:
                    L.append(f"| {e['file']} | {e['person']} | {lab_f} | 실패: {x.get('failed')} | | | | |")
                    continue
                hd = x["hand_desk_deg"]
                fv = f"{x['f']:.0f} ({lab_f})"
                L.append(f"| {e['file']} | {e['person']} | {fv} | {x['desk_tilt_deg']:.1f}° | " + ", ".join(f"{p['hand_angle_deg']:+.0f}°" for p in d["pauses"]) + " | "
                         + ", ".join(f"{v:.1f}°" for v in hd) + f" | {max(hd) - min(hd):.1f}° | " + ", ".join(f"{v:.1f}" for v in x["fit_rms_px"][1:]) + " |")
        L.append("")
    L += ["## 한계", "", "- 멈춤 구간 3~5개뿐이고, 호모그래피 분해는 점 16개의 작은 손에서 잡음에 민감하다. 손 평면 하나로 본 값이라 손가락 굽힘은 평균된다.",
          "- 책상 법선은 손 회전축에서 구하므로, 손 경사가 멈춤마다 다르면 회전축도 조금 기운다.", ""]
    return "\n".join(L) + "\n"


def pp_results_yaml(run_dir: Path, own: dict) -> Path:
    """97 의 값만: 영상·점 종류별, 94 추정 f 로 계산한 각도만 (기준 f 로 계산한 행은 f 값이 기준 f 라 넣지 않음, summary.md 에)."""
    r = json.loads((run_dir / "results.json").read_text())["real"]
    m = json.loads((run_dir / "metadata.json").read_text())
    rv = lambda x: x if not isinstance(x, float) else (round(x, 3) if np.isfinite(x) else None)      # noqa: E731
    deep = lambda d: {k: (deep(v) if isinstance(v, dict) else [rv(q) for q in v] if isinstance(v, list) else rv(v)) for k, v in d.items()}      # noqa: E731
    out = {"experiment": own["experiment"]["name"], "run": str(run_dir.resolve().relative_to(hf.PROJECT_ROOT)), "code_commit": m["git"]["commit"][:7],
           "git_dirty_at_run": m["git"]["git_dirty"], "videos": {}}
    for f, d in r.items():
        out["videos"][f] = {"ref_t": rv(d["ref_t"]), "pauses": [deep(p) for p in d["pauses"]],
                            "by_points": {src: {k: deep(v) for k, v in x.items() if k != "reference"} for src, x in d["by_points"].items()}}
    p = hf.resolve(own["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


# ----------------------------------------------------------------------------------------------------
# 98: CoTracker3 로 관절 직접 추적 -> 같은 추정
# ----------------------------------------------------------------------------------------------------
def ct_job(args):
    """한 영상: 프레임·검출 -> 기준 프레임(첫 정지 구간) -> CoTracker3 로 관절 21개 + 보조점 추적 (stride 프레임마다) -> 사용 프레임.
    GPU = 작업 프로세스마다 하나 고정 (프로세스 번호 % GPU 수, 한 GPU 에 두 영상이 겹치지 않게 gpu_workers <= GPU 수). 사용 프레임 = 첫~마지막 검출 범위, 추적한 프레임, joint_ids 중 보이는 관절이 min_visible 개 이상, 보이는 점의 상자가 가장자리에 닿지 않음."""
    i, e, cfg, run_root = args
    import torch
    from optical_flow import cotracker as ct
    cc = cfg["cotracker"]
    import multiprocessing as mp
    dev = f"cuda:{(mp.current_process()._identity[0] - 1) % torch.cuda.device_count()}"
    pngs, times, tr, (W, H), tmp = prepare(e, cfg, run_root)
    try:
        ref = rof.choose_reference_pose(tr, (W, H), cfg)
        st = int(cc["stride"])
        sel = np.arange(ref % st, len(pngs), st)
        sp = ct.support_points(tr["kp"][ref], tr["box"][ref], (H, W), float(cfg["tracking"]["dilate_frac"]), float(cc["support_step_frac"]))
        t0 = time.time()
        model = ct.load_model(cc["hub_dir"], dev)
        k, v = ct.track_points(model, [pngs[j] for j in sel], int(np.flatnonzero(sel == ref)[0]), np.r_[tr["kp"][ref], sp], float(cc["scale"]), dev)
        el = time.time() - t0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n = len(pngs)
    kp = np.full((n, 21, 2), np.nan)
    vis = np.zeros((n, 21), bool)
    kp[sel], vis[sel] = k[:, :21], v[:, :21]
    kp[~vis] = np.nan
    ids = cfg["joint_ids"]
    pres = np.flatnonzero(tr["present"])
    inside = np.zeros(n, bool)
    inside[pres[0]:pres[-1] + 1] = True
    enough = vis[:, ids].sum(1) >= int(cc["min_visible"])
    box = np.full((n, 4), np.nan)
    hv = vis.any(1)
    box[hv] = np.c_[np.nanmin(kp[hv], 1), np.nanmax(kp[hv], 1)]
    edge = np.zeros(n, bool)
    edge[hv] = of.edge_flags(box[hv], np.ones(hv.sum(), bool), W, H)
    use = inside & enough & ~edge & hv
    dr = {g: med(region_drift(kp, tr["kp"], tr["box"], use & tr["present"], j)) for g, j in (("wrist", [0]), ("thumb", [1, 2, 3, 4]), ("j16", list(range(5, 21))))}
    x = {"kp": kp, "box": box, "use": use, "size": (W, H), "ref": ref, "breaks": 0}
    save_points(run_root, e["file"], x, (W, H), hand_angle(tr), times)
    return x, {"ref_t": float(times[ref]), "n_tracked": int(len(sel)), "n_support": int(len(sp)), "n_use": int(use.sum()), "visible_frac": vis[sel][:, ids].mean().item(),
               "drift_pct": dr, "track_s": el, "device": dev}


def ct_run(cfg: dict, own: dict, log, run_root: Path) -> dict:
    """98: 영상마다 ct_job (GPU 수만큼 동시에), 그다음 94 와 같은 추정 (free_template -> 경사 고정, 함께 추정)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    pts, info = {}, {}
    with ProcessPoolExecutor(max_workers=int(cfg["cotracker"]["gpu_workers"]), mp_context=mp.get_context("spawn")) as ex:
        for e, (x, inf) in zip(cfg["videos"], ex.map(ct_job, [(i, e, cfg, str(run_root)) for i, e in enumerate(cfg["videos"])])):
            pts[e["file"]], info[e["file"]] = x, inf
            log(f"  {e['file']}: 기준 {inf['ref_t']:.1f} s, 추적 프레임 {inf['n_tracked']}, 보조점 {inf['n_support']}, 사용 {inf['n_use']}, 보임 {inf['visible_frac']:.2f}, "
                f"검출 관절과 거리 손목 {inf['drift_pct']['wrist']}, j16 {inf['drift_pct']['j16']} ({inf['track_s']:.0f} s, {inf['device']})")
    files = [e["file"] for e in cfg["videos"]]
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        fts = dict(zip(files, ex.map(hf.tl_template_job, [rof.estimate_args(cfg, pts[f]) for f in files], chunksize=1)))
    todo = [(f"ramp_{r:g}", "ramp_fixed", r) for r in cfg["ramps_deg"]] + [(f"free_{r:g}", "ramp_free", r) for r in cfg["free_starts_deg"]]
    res, jobs, keys = {}, [], []
    for f in files:
        x = pts[f]
        grid = hf.f_grid(cfg["pipeline"]["estimate"], max(*x["size"]))
        A = fts[f].get("A")
        res[f] = {"info": info[f], "ref": x["ref"], "n_use": int(x["use"].sum()),
                  "free_template": {"failed": fts[f].get("failed")} if A is None else {"f": float(A["f"]), "boundary": hf.od_boundary(float(A["f"]), grid), "rms_px": float((A.get("ba") or {}).get("rms_px", np.nan))}}
        if A is None:
            continue
        for mk, mode, r in todo:
            jobs.append(rof.estimate_args(cfg, x, A) + (mode, r))
            keys.append((f, mk))
    log(f"  추정 {len(jobs)}건")
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        for (f, mk), o in zip(keys, ex.map(hf.ms2_job, jobs, chunksize=1)):
            res[f][mk] = o
    return res


def ct_summary(res: dict, cfg: dict, own: dict) -> str:
    cc = cfg["cotracker"]
    prev = json.loads((hf.resolve(own["compare_run"]) / "results.json").read_text())["real"]
    best = lambda d: min([d[k] for k in d if k.startswith("free_") and isinstance(d[k], dict) and d[k].get("cost") not in (None, float("inf"))], key=lambda x: x["cost"], default=None)      # noqa: E731
    head = ["", f"- 점: {cfg['joint_set']}, 경사 고정 {cfg['ramps_deg']}, 함께 추정 시작 {cfg['free_starts_deg']} 중 비용 최소. 기준 f 는 이 파일에만. * = f 탐색 범위 끝.",
            f"- 비교: {own['compare_run']} (94, 광류 사슬, 같은 추정)", "",
            "| 영상 | 묶음 | 기준 f | 추정 | CoTracker3 f (오차) | 94 광류 사슬 f (오차) |", "|---|---|---|---|---|---|"]
    for e in cfg["videos"]:
        r, p = res.get(e["file"]) or {}, prev.get(e["file"]) or {}
        rf = float(e.get("ref_f_px") or cfg["ref_f"]["value"])
        fmt = lambda x: "-" if not x or "f" not in x else f"{x['f']:.1f} ({100 * (x['f'] / rf - 1):+.1f}%)" + ("*" if x.get("boundary") else "") + (f", 경사 {x['pitch_deg']:.1f}°" if "pitch_deg" in x and x is not None and x.get("start") and False else "")      # noqa: E731
        for k in ["free_template"] + [f"ramp_{r_:g}" for r_ in cfg["ramps_deg"]] + ["free_best"]:
            x, y = (best(r), best(p)) if k == "free_best" else (r.get(k), p.get(k))
            ex_ = lambda z: "" if k != "free_best" or not z or "pitch_deg" not in z else f", 경사 {z['pitch_deg']:.1f}°"      # noqa: E731
            head.append(f"| {e['file']} | {e['person']} | {rf:g} | {'함께 추정' if k == 'free_best' else k} | {fmt(x)}{ex_(x)} | {fmt(y)}{ex_(y)} |")
    L = ["# 98 CoTracker3 관절 추적 summary", "",
         f"**94 와 같은 기준 프레임(첫 정지 구간)의 MediaPipe 관절 21개를 광류 사슬 대신 CoTracker3(오프라인, {cc['stride']} 프레임마다, 보조점 손 영역 격자)로 직접 추적하고, 94 와 같은 추정을 돌린다. "
         "비교 실행 = 94(광류 사슬). 판정 없음.**", "",
         "## 추적 정보", "", "| 영상 | 기준 시각 | 추적 프레임 | 보조점 | 사용 프레임 | 관절 보임 비율 | 검출 관절과 거리 손목 (%) | 엄지 (%) | j16 (%) |", "|---|---|---|---|---|---|---|---|---|"]
    for f, d in res.items():
        i = d["info"]
        fm = lambda v: "-" if v is None else f"{v:.1f}"      # noqa: E731
        L.append(f"| {f} | {i['ref_t']:.1f} s | {i['n_tracked']} | {i['n_support']} | {i['n_use']} | {i['visible_frac']:.2f} | {fm(i['drift_pct']['wrist'])} | {fm(i['drift_pct']['thumb'])} | {fm(i['drift_pct']['j16'])} |")
    L += ["", "- 거리 = 추적점과 그 프레임 검출 관절의 거리 중앙값, 검출 손 상자 긴 변 대비 %. 검출기 오차가 섞인다.", "", "## f"] + head
    return "\n".join(L) + "\n"


def ct_results_yaml(run_dir: Path, own: dict) -> Path:
    """98 의 값만 (기준 f, 오차 없음): run_optical_flow.results_yaml 과 같은 항목 + 추적 정보."""
    r = json.loads((run_dir / "results.json").read_text())["real"]
    m = json.loads((run_dir / "metadata.json").read_text())
    rv = lambda x: x if not isinstance(x, float) else (round(x, 3) if np.isfinite(x) else None)      # noqa: E731
    deep = lambda d: {k: (deep(v) if isinstance(v, dict) else rv(v)) for k, v in d.items()}      # noqa: E731
    keys = ("f", "rms_px", "pitch_deg", "roll_deg", "desk_tilt_deg", "boundary", "start")
    out = {"experiment": own["experiment"]["name"], "run": str(run_dir.resolve().relative_to(hf.PROJECT_ROOT)), "code_commit": m["git"]["commit"][:7],
           "git_dirty_at_run": m["git"]["git_dirty"], "videos": {}}
    for f, d in r.items():
        out["videos"][f] = {k: (deep(v) if k == "info" else {kk: rv(v[kk]) for kk in keys if kk in v} if isinstance(v, dict) and "f" in v else v if not isinstance(v, float) else rv(v))
                            for k, v in d.items()}
    p = hf.resolve(own["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


# ----------------------------------------------------------------------------------------------------
# 99: 회전 방향 비대칭 (저장된 추적점, 한쪽만 / 대칭으로 자르기)
# ----------------------------------------------------------------------------------------------------
def rs_subsets(x: dict, margin: float) -> dict:
    """손 방향(첫 검출 대비, 도)으로 사용 프레임을 나눈다. all = 그대로, plus = 각도 >= -margin (정면 + '+' 쪽), minus = 각도 <= margin (정면 + '-' 쪽),
    symmetric = |각도| <= L, L = 사용 프레임의 '+' 쪽 최대와 '-' 쪽 최대(절댓값) 중 작은 것 (많이 돈 쪽을 잘라 양쪽 범위를 같게)."""
    a = x["hand_angle"] - x["hand_angle"][np.isfinite(x["hand_angle"])][0]
    u = x["use"] & np.isfinite(a)
    L = float(min(np.max(a[u]), -np.min(a[u])))
    return {"all": x["use"], "plus": u & (a >= -margin), "minus": u & (a <= margin), "symmetric": u & (np.abs(a) <= L)}, \
        {"angle_min": float(np.min(a[u])), "angle_max": float(np.max(a[u])), "sym_limit": L}


def rs_load(args):
    """한 영상: points_run 이 있으면 저장된 추적점을 읽고(프레임 추출·검출·추적 없음), 없으면 94 와 같은 광류 사슬(첫 정지 구간 기준)을 만들어 이 실행에 저장한다."""
    e, cfg, run_root = args
    pr = cfg["rotation_side"].get("points_run")
    if pr:
        x = load_points(pr, e["file"])
        return x, [f"  {e['file']}: 저장된 추적점, 기준 {x['times'][x['ref']]:.1f} s, 사용 {int(x['use'].sum())}"]
    import cv2
    cv2.setNumThreads(1)
    pngs, times, tr, (W, H), tmp = prepare(e, cfg, run_root)
    try:
        ref = rof.choose_reference_pose(tr, (W, H), cfg)
        x = of.track_hand(pngs, tr["kp"], tr["present"], tr["box"], (W, H), cfg["tracking"], ref=ref)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    save_points(run_root, e["file"], x, (W, H), hand_angle(tr), times)
    x = load_points(str(run_root), e["file"])
    return x, [f"  {e['file']}: 추적 (저장), 기준 {times[ref]:.1f} s, 사용 {int(x['use'].sum())}"]


def rs_run(cfg: dict, vid: dict, log) -> dict:
    """99: 영상 x 부분집합마다 free_template -> 경사 고정 추정만 (함께 추정 없음, 시간 단축)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    rc = cfg["rotation_side"]
    res, items = {}, []
    for f, x in vid.items():
        subs, info = rs_subsets(x, float(rc["margin_deg"]))
        res[f] = {"info": {**info, **{f"n_{k}": int(v.sum()) for k, v in subs.items()}}}
        for k in rc["subsets"]:
            items.append((f, k, {**x, "use": subs[k]}))
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        fts = list(ex.map(hf.tl_template_job, [rof.estimate_args(cfg, y) for _, _, y in items], chunksize=1))
    jobs, keys = [], []
    for (f, k, y), ft in zip(items, fts):
        A = ft.get("A")
        res[f][k] = {"free_template": {"failed": ft.get("failed")} if A is None else {"f": float(A["f"])}}
        if A is None:
            continue
        for r in cfg["ramps_deg"]:
            jobs.append(rof.estimate_args(cfg, y, A) + ("ramp_fixed", r))
            keys.append((f, k, f"ramp_{r:g}"))
    log(f"  추정 {len(jobs)}건")
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        for (f, k, mk), o in zip(keys, ex.map(hf.ms2_job, jobs, chunksize=1)):
            res[f][k][mk] = o
    return res


def rs_summary(res: dict, cfg: dict, own: dict) -> str:
    rc = cfg["rotation_side"]
    rk = [f"ramp_{r:g}" for r in cfg["ramps_deg"]]
    vids = {e["file"]: e for e in cfg["videos"]}
    lab = {"all": "전체 (94 와 같아야 함)", "plus": "정면 + '+' 쪽", "minus": "정면 + '-' 쪽", "symmetric": "양쪽 같은 각도까지"}
    L = ["# 99 회전 방향 비대칭 summary", "",
         f"**{rc['points_run']} 에 저장된 추적점(94 와 같은 광류 사슬, 첫 정지 구간 기준)을 손 방향(검출 손목 -> 중지 MCP, 첫 검출 대비)으로 나눠, 같은 추정(free_template -> 경사 "
         f"{cfg['ramps_deg']}° 고정)을 따로 돌린다. 한쪽 = 그쪽 + 정면(반대쪽 {rc['margin_deg']}° 까지). 대칭 = 덜 돈 쪽의 최대 각도까지 양쪽을 자름. 함께 추정은 돌리지 않음(시간 단축). 판정 없음.**", "",
         "- 기준 f 는 이 파일에만. * = f 탐색 범위 끝. 각도의 +/- 는 화면 속 손 방향 기준(실제 좌우와의 대응은 영상으로 확인).", "",
         "| 영상 | 묶음 | 기준 f | 손 방향 범위 | 부분집합 | 사용 프레임 | " + " | ".join(f"경사 {k[5:]}° 고정" for k in rk) + " | free_template |", "|---|---|---|---|---|---|" + "---|" * len(rk) + "---|"]
    for f, d in res.items():
        e, i = vids[f], d["info"]
        rf = float(e.get("ref_f_px") or cfg["ref_f"]["value"])
        fmt = lambda x: "-" if not x or "f" not in x else f"{x['f']:.0f} ({100 * (x['f'] / rf - 1):+.1f}%)" + ("*" if x.get("boundary") else "")      # noqa: E731
        for k in rc["subsets"]:
            o = d.get(k) or {}
            rng = f"{i['angle_min']:+.0f}° ~ {i['angle_max']:+.0f}°" + (f" (대칭 ±{i['sym_limit']:.0f}°)" if k == "symmetric" else "")
            L.append(f"| {f} | {e['person']} | {rf:g} | {rng} | {lab[k]} | {i['n_' + k]} | " + " | ".join(fmt(o.get(m)) for m in rk) + f" | {fmt(o.get('free_template'))} |")
    L += ["", "## 한계", "", "- 한쪽만 쓰면 회전 범위가 반으로 줄어 f 정보도 줄어든다(흩어짐이 커질 수 있음). 영상 4개.", ""]
    return "\n".join(L) + "\n"


def rs_results_yaml(run_dir: Path, own: dict) -> Path:
    """99 의 값만: 영상·부분집합별 f, RMS, 사용 프레임 (기준 f, 오차 없음)."""
    r = json.loads((run_dir / "results.json").read_text())["real"]
    m = json.loads((run_dir / "metadata.json").read_text())
    rv = lambda x: x if not isinstance(x, float) else (round(x, 3) if np.isfinite(x) else None)      # noqa: E731
    keys = ("f", "rms_px", "pitch_deg", "roll_deg", "desk_tilt_deg", "boundary", "start")
    out = {"experiment": own["experiment"]["name"], "run": str(run_dir.resolve().relative_to(hf.PROJECT_ROOT)), "code_commit": m["git"]["commit"][:7],
           "git_dirty_at_run": m["git"]["git_dirty"], "videos": {}}
    for f, d in r.items():
        out["videos"][f] = {k: ({kk: rv(vv) for kk, vv in v.items()} if k == "info" else {mk: {kk: rv(o[kk]) for kk in keys if kk in o} for mk, o in v.items()}) for k, v in d.items()}
    p = hf.resolve(own["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


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


def jt_results_yaml(run_dir: Path, own: dict) -> Path:
    """93 의 값만: 영상별 기준 시각, 검출 관절과의 거리(기존 사슬 / 관절별), 옮긴 방식 비율. 프레임별 점은 넣지 않음."""
    r = json.loads((run_dir / "results.json").read_text())["real"]
    m = json.loads((run_dir / "metadata.json").read_text())
    rv = lambda x: x if not isinstance(x, float) else (round(x, 3) if np.isfinite(x) else None)      # noqa: E731
    deep = lambda d: {k: (deep(v) if isinstance(v, dict) else rv(v)) for k, v in d.items()}      # noqa: E731
    out = {"experiment": own["experiment"]["name"], "run": str(run_dir.resolve().relative_to(hf.PROJECT_ROOT)), "code_commit": m["git"]["commit"][:7],
           "git_dirty_at_run": m["git"]["git_dirty"], "videos": {}}
    for f in own["videos"]:
        x = r.get(f) or {}
        out["videos"][f] = {k: (deep(x[k]) if isinstance(x[k], dict) else rv(x[k])) for k in ("ref", "ref_t", "n_frames", "video", "how_share", "how_share_by_group", "drift") if k in x}
    p = hf.resolve(own["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


class _nullpool:
    """작업 풀이 필요 없는 명령용 (with 문 자리 채우기)."""
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description="광류 방식 원인 확인 실험")
    ap.add_argument("cmd", choices=["tracking-plane", "dorsum-corners", "two-plane", "joint-track-video", "geometry-synth", "pause-plane", "cotracker", "rotation-side"])
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
        print(tw_results_yaml(hf.resolve(a.run), own) if a.cmd == "two-plane" else gs_real_results_yaml(hf.resolve(a.run), own) if a.cmd == "geometry-synth" else pp_results_yaml(hf.resolve(a.run), own) if a.cmd == "pause-plane"
              else ct_results_yaml(hf.resolve(a.run), own) if a.cmd == "cotracker" else rs_results_yaml(hf.resolve(a.run), own) if a.cmd == "rotation-side" else jt_results_yaml(hf.resolve(a.run), own) if a.cmd == "joint-track-video"
              else results_yaml(a.cmd, hf.resolve(a.run), own))
        return None
    cfg = rof.load_settings(own)
    cfg["videos"] = cfg["videos"] + [dict(v) for v in own.get("extra_videos", [])]      # 기본 설정 목록에 없는 새 영상 (file, person, ref_f_px)
    cfg.update({k: own[k] for k in ("variants", "clahe", "two_plane", "joint_track", "render", "reference", "geometry_synth", "pause_plane", "cotracker", "rotation_side") if k in own})
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
    job = {"tracking-plane": tp_job, "dorsum-corners": dc_job, "two-plane": tw_points, "joint-track-video": jt_job, "geometry-synth": rof.video_points, "pause-plane": pp_job, "rotation-side": rs_load}.get(a.cmd)
    vid = {}
    if a.cmd == "cotracker":                                           # GPU 작업 + 추정을 ct_run 이 한꺼번에
        res = ct_run(cfg, own, log, run_root)
        job = None
    with ProcessPoolExecutor(max_workers=int(cfg["video_workers"]), mp_context=mp.get_context("spawn")) if job else _nullpool() as ex:
        for e, (x, lines) in (zip(cfg["videos"], ex.map(job, [(e, cfg, str(run_root)) for e in cfg["videos"]])) if job else []):
            vid[e["file"]] = x
            for s in lines:
                log(s)
    log(f"  추적: {time.time() - t0:.0f} s")
    if a.cmd == "cotracker":
        text_sum = ct_summary(res, cfg, own)
    elif a.cmd == "rotation-side":
        res = rs_run(cfg, vid, log)
        text_sum = rs_summary(res, cfg, own)
    elif a.cmd == "pause-plane":
        res = vid
        text_sum = pp_summary(res, cfg, own)
    elif a.cmd == "tracking-plane":
        res = vid
        text_sum = tp_summary(res, cfg)
    elif a.cmd == "joint-track-video":
        res = vid
        text_sum = jt_summary(res, cfg)
    elif a.cmd == "geometry-synth":
        res = gs_run(cfg, own, vid, log)
        text_sum = gs_summary(res, cfg, own)
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
