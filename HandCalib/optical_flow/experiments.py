"""광류 방식의 원인 확인 실험 (run.py 를 거치지 않음).

  python optical_flow/experiments.py tracking-plane  --config configs/90_tracking_plane.yaml
  python optical_flow/experiments.py dorsum-corners  --config configs/91_dorsum_clahe.yaml
  python optical_flow/experiments.py geometry-synth  --config configs/96_geometry_synth.yaml
  python optical_flow/experiments.py pause-plane     --config configs/97_pause_plane.yaml
  python optical_flow/experiments.py cotracker       --config configs/98_cotracker.yaml
  python optical_flow/experiments.py rotation-side   --config configs/99_rotation_side.yaml
  python optical_flow/experiments.py shape3d         --config configs/101_cotracker_shape3d.yaml
  python optical_flow/experiments.py points-video    --config configs/102_points_video_new_pair.yaml
  python optical_flow/experiments.py side-consistency --config configs/103_side_consistency.yaml
  python optical_flow/experiments.py shape-change    --config configs/106_shape_change.yaml
  python optical_flow/experiments.py pip-axis        --config configs/111_pip_axis_j16.yaml
  python optical_flow/experiments.py pause-plane2    --config configs/113_pause_plane2.yaml
  python optical_flow/experiments.py desk3d          --config configs/114_desk3d.yaml
  python optical_flow/experiments.py desk3d-contact  --config configs/115_desk3d_contact.yaml
  python optical_flow/experiments.py <명령> --config <설정> results-yaml --run <실행 폴더>

tracking-plane (90): 추적 사슬이 손의 어느 평면을 따라가는지 본다. 기준 프레임 -> 각 프레임의 사슬 변환을, 같은 두 프레임의 검출 관절로 맞춘
                     세 변환(손등 몸통 관절 / 손가락 관절 / j16)과 비교한다. 손 회전 각도 구간별로도 나눈다. f 추정 없음.
dorsum-corners (91): 특징점을 손등 몸통에서만 고르기, 특징점을 고를 때만 대비 보정(CLAHE), 둘 다. 추적점과 검출 관절의 거리, 책상 운동 추정기의 f.
geometry-synth (96): 영상의 카메라 기울기·손 모양·프레임별 움직임을 그대로 둔 합성 장면을 참 f 로 만들어 같은 추정을 돌린다 (촬영 구도·동작 탓인지, 실제 점 탓인지).
pause-plane (97): 멈춤 구간마다 손 평면 방향을 알려진 f 로 호모그래피 분해해 손 경사가 자세마다 바뀌는지 본다 (실제 점, 합성 없음).
desk3d-contact (115): 114 + 책상에 닿는 손가락 끝 쪽 관절의 높이를 같게 고정.
desk3d (114): 책상 운동 + 입체 손 모양 추정기(optical_flow/desk3d.py)를 저장된 관절별 추적점에 돌린다 (경사 가정 없음).
pip-axis (111, 112): 손 축 뿌리를 PIP 줄로 바꿔(MCP 없이도) 책상 운동 추정기를 돌린다. pause-plane2 (113): 97 을 고쳐 멈춤 자세마다 손 평면 방향을 잰다.
shape-change (106): 저장된 관절별 추적점이 평평한 단단한 손(호모그래피 하나)에서 얼마나 벗어나는지 (f 없음).
side-consistency (103): 저장된 추적점으로 회전 방향별 f 가 같아지는 경사를 고른다 (기준 f 없이 경사 정하기).
shape3d (101): 저장된 점에 free_template / free_shape3d(손 3D 모양 + f)를 돌린다. points-video: 저장된 추적점과 검출 관절을 그린 점 영상.
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
# 점 영상 (저장된 추적점 + MediaPipe 검출 관절을 원본 프레임에 그림)
# ----------------------------------------------------------------------------------------------------
def pv_render(path: Path, pngs: list, times: np.ndarray, frames: np.ndarray, kd: np.ndarray, kc: np.ndarray, use: np.ndarray, rc: dict, title: str):
    """점 영상: 왼쪽 전체 화면, 오른쪽 손 확대. 주황 = MediaPipe 검출 관절, 파랑 = 추적점(f 추정에 쓰는 점, 기준 프레임 관절을 광류 사슬로 옮긴 것; 속 빈 점 = 그 프레임은 추정에 안 씀).
    jt_render 와 같은 화면 구성. 프레임 이미지는 임시 폴더에 그린 뒤 ffmpeg 로 묶고(메타데이터 제거) 지운다."""
    import subprocess
    import tempfile
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    font = lambda s: ImageFont.truetype(rc["font"], s, index=int(rc.get("font_index", 0)))      # noqa: E731
    LW, RW, HD = int(rc["left_width"]), int(rc["zoom_size"]), 70
    ORANGE, BLUE = (235, 104, 52), (42, 120, 214)
    with tempfile.TemporaryDirectory(dir=str(path.parent)) as td:
        for n_out, i in enumerate(frames):
            img = cv2.imread(pngs[i])[:, :, ::-1]
            H, W = img.shape[:2]
            sc = LW / W
            left = Image.fromarray(img).resize((LW, int(round(H * sc))))
            pts = np.concatenate([x[np.isfinite(x).all(1)] for x in (kd[i], kc[i])] or [np.zeros((0, 2))])
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
            for kp, col, rad, hollow in ((kd[i], ORANGE, 4, False), (kc[i], BLUE, 5, not bool(use[i]))):
                for j in range(len(kp)):
                    if not np.isfinite(kp[j]).all():
                        continue
                    for (ox, oy, s_, r_) in ((0, HD, sc, rad - 1), (LW, HD, zs, rad + 2)):
                        x = ox + (kp[j][0] - (0 if ox == 0 else box[0])) * s_
                        y = oy + (kp[j][1] - (0 if ox == 0 else box[1])) * s_
                        if hollow:
                            d.ellipse((x - r_, y - r_, x + r_, y + r_), outline=col, width=2)
                        else:
                            d.ellipse((x - r_, y - r_, x + r_, y + r_), fill=col, outline=(255, 255, 255))
            d.text((10, 6), f"{title}   {times[i]:.2f} s   프레임 {i}", fill=(0, 0, 0), font=font(22))
            x0 = 10
            for col, lab in ((ORANGE, "MediaPipe 검출 관절"), (BLUE, "추적점 (f 추정에 쓰는 점, 속 빈 점 = 이 프레임은 안 씀)")):
                d.ellipse((x0, 44, x0 + 14, 58), fill=col)
                d.text((x0 + 20, 38), lab, fill=(0, 0, 0), font=font(18))
                x0 += 30 + int(d.textlength(lab, font=font(18)))
            canvas.save(Path(td) / f"{n_out:06d}.png")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", f"{float(rc['fps']):.4f}", "-i", str(Path(td) / "%06d.png"), "-c:v", "libx264", "-crf", "20",
                        "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-map_metadata", "-1", str(path)], check=True)


def pv_job(args):
    """한 영상: 프레임·검출 (검출 캐시 재사용) + points_run 에 저장된 추적점 -> 점 영상 (render.every 프레임마다)."""
    e, cfg, run_root = args
    import cv2
    cv2.setNumThreads(1)
    rc = cfg["render"]
    x = load_points(cfg["points_video"]["points_run"], e["file"])
    pngs, times, tr, (W, H), tmp = prepare(e, cfg, run_root)
    try:
        if len(pngs) != len(x["kp"]):
            raise ValueError(f"프레임 수가 저장된 추적점과 다름: {len(pngs)} / {len(x['kp'])}")
        fr = np.arange(0, len(pngs), int(rc["every"]))
        out = Path(run_root) / f"{Path(e['file']).stem}_points.mp4"
        pv_render(out, pngs, times, fr, np.where(tr["present"][:, None, None], tr["kp"], np.nan), x["kp"], x["use"], {**rc, "fps": 30.0 / int(rc["every"])}, f"{e['file']} ({e['person']})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"video": out.name, "n_frames": int(len(fr))}, [f"  {e['file']}: 점 영상 {out.name} ({len(fr)} 프레임)"]


def pv_summary(res: dict, cfg: dict) -> str:
    L = ["# 점 영상", "", f"**{cfg['points_video']['points_run']} 에 저장된 추적점(파랑)과 MediaPipe 검출 관절(주황)을 원본 프레임에 그림. 왼쪽 전체 화면, 오른쪽 손 확대. 영상은 커밋하지 않음.**", "",
         "| 영상 | 점 영상 | 프레임 |", "|---|---|---|"]
    for f, x in res.items():
        L.append(f"| {f} | {x['video']} | {x['n_frames']} |")
    return "\n".join(L) + "\n"



# ----------------------------------------------------------------------------------------------------
# 101: 저장된 점(CoTracker3 등) + 손 3D 모양(free_shape3d) -> f
# ----------------------------------------------------------------------------------------------------
def s3_job(args):
    """(영상, 점 출처, 관절 묶음) 하나: 저장된 추적점 -> od_bridge -> free_template 과 free_shape3d (od_estimate, 책상 제약 없음, 손 3D 모양을 f 와 함께 품)."""
    key, x, cfg, ids = args
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import cv2
    cv2.setNumThreads(1)
    sc = cfg["shape3d"]
    grid = hf.f_grid(cfg["pipeline"]["estimate"], max(*x["size"]))
    t0 = time.time()
    try:
        pb = hf.od_bridge(x["kp"], np.isfinite(x["kp"]).all(2), np.flatnonzero(x["use"]), x["box"], tuple(x["size"]), cfg["pipeline"]["estimate"], ids)
        r = hf.od_estimate(pb, grid, sc["free_shape3d"])
        A, B = r["free_template"], r["free_shape3d"]
        out = {"free_template": {"f": float(A["f"]), "boundary": hf.od_boundary(float(A["f"]), grid)},
               "free_shape3d": {"f": float(B["f"]), "boundary": bool(B["at_bound"]) or hf.od_boundary(float(B["f"]), grid), "relief_rel": float(B["relief_rel"]),
                                "rms_px": float(B["ba"]["rms_px"]), "success": bool(B["ba"]["success"])}}
    except Exception as ex:                                            # 실패도 결과로 남김
        out = {"failed": f"{type(ex).__name__}: {ex}"}
    out["elapsed_s"] = time.time() - t0
    return key, out


def s3_run(cfg: dict, own: dict, log) -> dict:
    """101: 점 출처(points_runs)마다 저장된 추적점을 읽어 관절 묶음마다 s3_job."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    sc = cfg["shape3d"]
    jobs = []
    for src, run in sc["points_runs"].items():
        for e in cfg["videos"]:
            x = load_points(run, e["file"])
            for js, ids in sc["joint_sets"].items():
                jobs.append(((e["file"], src, js), x, cfg, [int(j) for j in ids]))
    log(f"  추정 {len(jobs)}건")
    res = {}
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        for (f, src, js), o in ex.map(s3_job, jobs, chunksize=1):
            res.setdefault(f, {}).setdefault(src, {})[js] = o
    return res


def s3_summary(res: dict, cfg: dict, own: dict) -> str:
    sc = cfg["shape3d"]
    vids = {e["file"]: e for e in cfg["videos"]}
    L = ["# 101 정확한 점 + 손 3D 모양 summary", "",
         f"**저장된 추적점({', '.join(f'{k} = {v}' for k, v in sc['points_runs'].items())})에 교수님 코드의 free_template(평면 손)과 free_shape3d(손 3D 모양을 f 와 함께 품, "
         f"z_rel {sc['free_shape3d']['z_rel']})를 그대로 돌린다. 책상 제약·경사 가정 없음. 판정 없음.**", "",
         "- 기준 f 는 이 파일에만. * = f 탐색 범위 끝. 높이 = free_shape3d 가 찾은 손 모양의 높이(relief_rel, 손 크기 대비).", "",
         "| 영상 | 묶음 | 기준 f | 점 | 관절 | free_template f (오차) | free_shape3d f (오차) | 높이 | RMS (px) |", "|---|---|---|---|---|---|---|---|---|"]
    for f, d in res.items():
        e = vids[f]
        rf = float(e.get("ref_f_px") or cfg["ref_f"]["value"])
        fmt = lambda x: "-" if not x or "f" not in x else f"{x['f']:.0f} ({100 * (x['f'] / rf - 1):+.1f}%)" + ("*" if x.get("boundary") else "")      # noqa: E731
        for src, dd in d.items():
            for js, o in dd.items():
                if "failed" in o:
                    L.append(f"| {f} | {e['person']} | {rf:g} | {src} | {js} | 실패: {o['failed']} | | | |")
                    continue
                b = o["free_shape3d"]
                L.append(f"| {f} | {e['person']} | {rf:g} | {src} | {js} | {fmt(o['free_template'])} | {fmt(b)} | {b['relief_rel']:.3f} | {b['rms_px']:.2f} |")
    L += ["", "## 한계", "", "- free_shape3d 는 손 높이를 z_rel 로 약하게 눌러 둔다(완전히 자유롭지 않음). 책상 위 회전만으로는 손 높이와 f 가 서로 맞바꿔질 수 있다.", ""]
    return "\n".join(L) + "\n"


def s3_results_yaml(run_dir: Path, own: dict) -> Path:
    """101 의 값만: 영상·점 출처·관절 묶음별 f, 높이, RMS (기준 f, 오차 없음)."""
    r = json.loads((run_dir / "results.json").read_text())["real"]
    m = json.loads((run_dir / "metadata.json").read_text())
    rv = lambda x: x if not isinstance(x, float) else (round(x, 3) if np.isfinite(x) else None)      # noqa: E731
    deep = lambda d: {k: (deep(v) if isinstance(v, dict) else rv(v)) for k, v in d.items()}      # noqa: E731
    out = {"experiment": own["experiment"]["name"], "run": str(run_dir.resolve().relative_to(hf.PROJECT_ROOT)), "code_commit": m["git"]["commit"][:7],
           "git_dirty_at_run": m["git"]["git_dirty"], "videos": {f: deep(d) for f, d in r.items()}}
    p = hf.resolve(own["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


# ----------------------------------------------------------------------------------------------------
# 103: 회전 방향별 f 가 같아지는 경사 (저장된 추적점)
# ----------------------------------------------------------------------------------------------------
def sc_load(args):
    """한 영상: side_consistency.points 에 적힌 실행에서 저장된 추적점 읽기."""
    e, cfg, run_root = args
    x = load_points(cfg["side_consistency"]["points"][e["file"]], e["file"])
    return x, [f"  {e['file']}: 저장된 추적점, 사용 {int(x['use'].sum())}"]


def sc_run(cfg: dict, vid: dict, log) -> dict:
    """103: 영상 x 부분집합(all, plus, minus; rs_subsets) x 경사(ramps_deg) 마다 free_template -> 경사 고정 추정."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    sc = cfg["side_consistency"]
    res, items = {}, []
    for f, x in vid.items():
        subs, info = rs_subsets(x, float(sc["margin_deg"]))
        res[f] = {"info": {**info, **{f"n_{k}": int(v.sum()) for k, v in subs.items()}}}
        for k in sc["subsets"]:
            items.append((f, k, {**x, "use": subs[k]}))
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        fts = list(ex.map(hf.tl_template_job, [rof.estimate_args(cfg, y) for _, _, y in items], chunksize=1))
    jobs, keys = [], []
    for (f, k, y), ft in zip(items, fts):
        A = ft.get("A")
        res[f][k] = {}
        if A is None:
            res[f][k]["failed"] = ft.get("failed")
            continue
        for r in cfg["ramps_deg"]:
            jobs.append(rof.estimate_args(cfg, y, A) + ("ramp_fixed", r))
            keys.append((f, k, f"ramp_{r:g}"))
    log(f"  추정 {len(jobs)}건")
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        for (f, k, mk), o in zip(keys, ex.map(hf.ms2_job, jobs, chunksize=1)):
            res[f][k][mk] = o
    for f, d in res.items():
        d["crossing"] = sc_crossing(d, cfg["ramps_deg"])
    return res


def sc_crossing(d: dict, ramps) -> dict:
    """plus 와 minus 의 log f 차이가 부호를 바꾸는 경사(선형 보간)와, 그 경사에서 all 의 f (log f 선형 보간). 부호가 안 바뀌면 차이가 가장 작은 경사."""
    rs = np.array([float(r) for r in ramps])
    g = lambda k: np.array([np.log(d[k][f"ramp_{r:g}"]["f"]) if "f" in d.get(k, {}).get(f"ramp_{r:g}", {}) else np.nan for r in ramps])      # noqa: E731
    dp, fa = g("plus") - g("minus"), g("all")
    ok = np.isfinite(dp) & np.isfinite(fa)
    rs, dp, fa = rs[ok], dp[ok], fa[ok]
    if len(rs) < 2:
        return {"ramp_deg": None, "f": None, "crossed": False}
    for i in range(len(rs) - 1):
        if dp[i] == 0 or dp[i] * dp[i + 1] < 0:
            t = 0.0 if dp[i] == 0 else dp[i] / (dp[i] - dp[i + 1])
            return {"ramp_deg": float(rs[i] + t * (rs[i + 1] - rs[i])), "f": float(np.exp(fa[i] + t * (fa[i + 1] - fa[i]))), "crossed": True}
    i = int(np.argmin(np.abs(dp)))
    return {"ramp_deg": float(rs[i]), "f": float(np.exp(fa[i])), "crossed": False}


def sc_summary(res: dict, cfg: dict, own: dict) -> str:
    sc = cfg["side_consistency"]
    vids = {e["file"]: e for e in cfg["videos"]}
    L = ["# 103 회전 방향별 f 가 같아지는 경사 summary", "",
         f"**저장된 광류 사슬 추적점(94 와 같은 계산, 첫 정지 구간 기준, {cfg['joint_set']})을 손 방향으로 나눠(정면 + '+' 쪽 / 정면 + '-' 쪽, 반대쪽 {sc['margin_deg']}° 까지; 99 와 같음), 경사 "
         f"{cfg['ramps_deg']}° 고정으로 각각 추정한다. 두 쪽의 f 가 같아지는 경사(선형 보간)를 고르고, 그 경사에서 전체 프레임의 f 를 읽는다. 기준 f 를 쓰지 않고 경사를 정하는 방법. 판정 없음.**", "",
         "- 기준 f 는 이 파일에만. * = f 탐색 범위 끝. 교차 없음 = 시험한 경사 안에서 두 쪽 차이의 부호가 안 바뀜(차이가 가장 작은 경사).", "",
         "## 고른 경사와 f", "", "| 영상 | 묶음 | 기준 f | 고른 경사 | 교차 | 그 경사의 전체 f (오차) | 경사 11° 전체 f (오차) |", "|---|---|---|---|---|---|---|"]
    for f, d in res.items():
        e = vids[f]
        rf = float(e.get("ref_f_px") or cfg["ref_f"]["value"])
        c = d["crossing"]
        a11 = (d.get("all") or {}).get("ramp_11") or {}
        L.append(f"| {f} | {e['person']} | {rf:g} | " + ("-" if c["ramp_deg"] is None else f"{c['ramp_deg']:.1f}°") + f" | {'예' if c['crossed'] else '아니오'} | "
                 + ("-" if c["f"] is None else f"{c['f']:.0f} ({100 * (c['f'] / rf - 1):+.1f}%)") + " | " + ("-" if "f" not in a11 else f"{a11['f']:.0f} ({100 * (a11['f'] / rf - 1):+.1f}%)") + " |")
    L += ["", "## 경사별 f 오차 (전체 / '+' 쪽 / '-' 쪽)", "", "| 영상 | 묶음 | " + " | ".join(f"{r:g}°" for r in cfg["ramps_deg"]) + " |", "|---|---|" + "---|" * len(cfg["ramps_deg"])]
    for f, d in res.items():
        e = vids[f]
        rf = float(e.get("ref_f_px") or cfg["ref_f"]["value"])
        cell = lambda k, r: "-" if "f" not in (d.get(k) or {}).get(f"ramp_{r:g}", {}) else f"{100 * (d[k][f'ramp_{r:g}']['f'] / rf - 1):+.0f}" + ("*" if d[k][f"ramp_{r:g}"].get("boundary") else "")      # noqa: E731
        L.append(f"| {f} | {e['person']} | " + " | ".join(f"{cell('all', r)} / {cell('plus', r)} / {cell('minus', r)}" for r in cfg["ramps_deg"]) + " |")
    L += ["", "## 한계", "", "- 한쪽만 쓰면 회전 범위가 반이라 f 가 흩어진다. 옆 기울기(roll)는 0 으로 고정(경사만 바꿈). 영상 8개.", ""]
    return "\n".join(L) + "\n"


def sc_results_yaml(run_dir: Path, own: dict) -> Path:
    """103 의 값만: 영상별 고른 경사, 그 경사의 f, 부분집합 x 경사별 f (기준 f, 오차 없음)."""
    r = json.loads((run_dir / "results.json").read_text())["real"]
    m = json.loads((run_dir / "metadata.json").read_text())
    rv = lambda x: x if not isinstance(x, float) else (round(x, 3) if np.isfinite(x) else None)      # noqa: E731
    out = {"experiment": own["experiment"]["name"], "run": str(run_dir.resolve().relative_to(hf.PROJECT_ROOT)), "code_commit": m["git"]["commit"][:7],
           "git_dirty_at_run": m["git"]["git_dirty"], "videos": {}}
    for f, d in r.items():
        out["videos"][f] = {"info": {k: rv(v) for k, v in d["info"].items()}, "crossing": {k: rv(v) for k, v in d["crossing"].items()},
                            **{k: {mk: ({kk: rv(o[kk]) for kk in ("f", "rms_px", "boundary") if kk in o} if isinstance(o, dict) else o) for mk, o in d[k].items()}
                               for k in d if k not in ("info", "crossing")}}
    p = hf.resolve(own["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


# ----------------------------------------------------------------------------------------------------
# 106: 평평한 단단한 손에서 벗어나는 정도 (저장된 관절별 추적점, 예: CoTracker3)
# ----------------------------------------------------------------------------------------------------
SH_NAMES = {5: "검지 MCP", 6: "검지 PIP", 7: "검지 DIP", 8: "검지 끝", 9: "중지 MCP", 10: "중지 PIP", 11: "중지 DIP", 12: "중지 끝", 13: "약지 MCP", 14: "약지 PIP",
            15: "약지 DIP", 16: "약지 끝", 17: "소지 MCP", 18: "소지 PIP", 19: "소지 DIP", 20: "소지 끝"}


def sh_job(args):
    """한 영상: 저장된 추적점에서 프레임마다 기준 프레임 관절(joint_ids) -> 그 프레임 관절의 최소제곱 호모그래피(평평하고 단단한 판이면 정확히 맞음)를 구하고,
    남는 거리(관절별, RMS)를 기준 프레임 손 상자 긴 변 대비 % 로. 손 방향(첫 검출 대비) 절댓값 구간별 중앙값과, angle_min_deg 넘게 돈 프레임의 관절별 중앙값."""
    e, cfg, run_root = args
    import cv2
    sc = cfg["shape_change"]
    x = load_points(sc["points"][e["file"]], e["file"])
    ids = [int(j) for j in cfg["joint_ids"]]
    kp, u, r = x["kp"], x["use"], x["ref"]
    a = x["hand_angle"] - x["hand_angle"][np.isfinite(x["hand_angle"])][0]
    P0 = kp[r][ids]
    bl = float(np.max(P0.max(0) - P0.min(0)))
    rms, per, ang = [], [], []
    for i in np.flatnonzero(u):
        P = kp[i][ids]
        if not np.isfinite(P).all() or not np.isfinite(a[i]):
            continue
        Hm, _ = cv2.findHomography(P0.astype(np.float64), P.astype(np.float64), 0)
        y = np.c_[P0, np.ones(len(P0))] @ Hm.T
        d = np.linalg.norm(y[:, :2] / y[:, 2:3] - P, axis=1) / bl * 100.0
        rms.append(float(np.sqrt(np.mean(d ** 2))))
        per.append(d)
        ang.append(abs(float(a[i])))
    rms, per, ang = np.array(rms), np.array(per), np.array(ang)
    bins = {}
    for lo, hi in sc["angle_bins_deg"]:
        k = (ang >= lo) & (ang < hi)
        bins[f"{lo}-{hi}"] = {"n": int(k.sum()), "rms_pct": med(rms[k]) if k.sum() >= int(sc["min_frames"]) else None}
    k = ang > float(sc["angle_min_deg"])
    joint = {str(j): (med(per[k][:, n]) if k.sum() >= int(sc["min_frames"]) else None) for n, j in enumerate(ids)}
    out = {"box_px": bl, "n_frames": int(len(rms)), "rms_pct_all": med(rms), "bins": bins, "joint_pct_turned": joint, "n_turned": int(k.sum())}
    return out, [f"  {e['file']}: 프레임 {len(rms)}, 전체 중앙값 {med(rms):.2f}%"]


def sh_summary(res: dict, cfg: dict, own: dict) -> str:
    sc = cfg["shape_change"]
    ids = [int(j) for j in cfg["joint_ids"]]
    vids = {e["file"]: e for e in cfg["videos"]}
    fm = lambda v: "-" if v is None else f"{v:.2f}"      # noqa: E731
    bk = [f"{lo}-{hi}" for lo, hi in sc["angle_bins_deg"]]
    L = ["# 106 평평한 단단한 손에서 벗어나는 정도 summary", "",
         f"**저장된 관절별 추적점({cfg['joint_set']})에서 프레임마다 기준 프레임 관절 -> 그 프레임 관절의 최소제곱 호모그래피를 구한다. 평평하고 단단한 판은 어느 방향에서 봐도 호모그래피 하나로 "
         "정확히 옮겨지므로, 남는 거리는 손이 평면이 아니거나(입체 모양, 예: 솟은 너클) 모양이 바뀐 만큼이다(둘은 구분하지 않음). 기준 프레임 손 상자 긴 변 대비 %. f 를 쓰지 않음. 판정 없음.**", "",
         "## 손 회전 각도별 (프레임 RMS 의 중앙값, %)", "", "| 영상 | 묶음 | 점 | 손 크기 (px) | " + " | ".join(f"{b}°" for b in bk) + " | 전체 |", "|---|---|---|---|" + "---|" * len(bk) + "---|"]
    for f, x in res.items():
        e = vids[f]
        L.append(f"| {f} | {e['person']} | {Path(sc['points'][f]).parts[1]} | {x['box_px']:.0f} | " + " | ".join(fm(x["bins"][b]["rms_pct"]) for b in bk) + f" | {fm(x['rms_pct_all'])} |")
    L += ["", f"## 관절별 ({sc['angle_min_deg']}° 넘게 돈 프레임의 중앙값, %)", "", "| 영상 | 묶음 | " + " | ".join(SH_NAMES.get(j, str(j)) for j in ids) + " |", "|---|---|" + "---|" * len(ids)]
    for f, x in res.items():
        L.append(f"| {f} | {vids[f]['person']} | " + " | ".join(fm(x["joint_pct_turned"][str(j)]) for j in ids) + " |")
    L += ["", "## 한계", "", "- 입체적이지만 단단한 손과 모양이 바뀌는 손을 구분하지 못한다. 추적점 자체의 오차도 섞인다(CoTracker3 는 관절 위치의 피부 점을 따라감).", ""]
    return "\n".join(L) + "\n"


def sh_results_yaml(run_dir: Path, own: dict) -> Path:
    """106 의 값만 (f 를 쓰지 않음)."""
    r = json.loads((run_dir / "results.json").read_text())["real"]
    m = json.loads((run_dir / "metadata.json").read_text())
    rv = lambda x: x if not isinstance(x, float) else (round(x, 3) if np.isfinite(x) else None)      # noqa: E731
    deep = lambda d: {k: (deep(v) if isinstance(v, dict) else rv(v)) for k, v in d.items()}      # noqa: E731
    out = {"experiment": own["experiment"]["name"], "run": str(run_dir.resolve().relative_to(hf.PROJECT_ROOT)), "code_commit": m["git"]["commit"][:7],
           "git_dirty_at_run": m["git"]["git_dirty"], "videos": {f: deep(d) for f, d in r.items()}}
    p = hf.resolve(own["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


# ----------------------------------------------------------------------------------------------------
# 111: 손 축을 PIP 줄 -> 손가락 끝으로 (MCP 를 빼고도 책상 운동 추정기를 쓸 수 있게)
# ----------------------------------------------------------------------------------------------------
def pa_ms2_job(args):
    """hf.ms2_job 과 같고, 손 축의 뿌리 관절(handfocal_report.DK_MCPS, 원래 MCP 4개)만 base 로 바꿔 부른다 (작업 프로세스 안에서만, 파일은 그대로).
    args = (base 관절 튜플,) + ms2_job 의 args."""
    import handfocal_report as _rep
    _rep.DK_MCPS = tuple(int(j) for j in args[0])
    return hf.ms2_job(args[1:])


def pa_run(cfg: dict, vid: dict, log) -> dict:
    """111: sc_run 과 같은 구성(영상 x 부분집합 x 경사) 이고, 추정만 pa_ms2_job (손 축 뿌리 = axis_base_joints). 결과 모양은 sc_run 과 같아 sc_summary 로 적는다."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    sc = cfg["side_consistency"]
    base = tuple(int(j) for j in cfg["axis_base_joints"])
    res, items = {}, []
    for f, x in vid.items():
        subs, info = rs_subsets(x, float(sc["margin_deg"]))
        res[f] = {"info": {**info, **{f"n_{k}": int(v.sum()) for k, v in subs.items()}}}
        for k in sc["subsets"]:
            items.append((f, k, {**x, "use": subs[k]}))
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        fts = list(ex.map(hf.tl_template_job, [rof.estimate_args(cfg, y) for _, _, y in items], chunksize=1))
    jobs, keys = [], []
    for (f, k, y), ft in zip(items, fts):
        A = ft.get("A")
        res[f][k] = {}
        if A is None:
            res[f][k]["failed"] = ft.get("failed")
            continue
        for r in cfg["ramps_deg"]:
            jobs.append((base,) + rof.estimate_args(cfg, y, A) + ("ramp_fixed", r))
            keys.append((f, k, f"ramp_{r:g}"))
    log(f"  추정 {len(jobs)}건 (손 축 뿌리 관절 {list(base)})")
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        for (f, k, mk), o in zip(keys, ex.map(pa_ms2_job, jobs, chunksize=1)):
            res[f][k][mk] = o
    for f, d in res.items():
        d["crossing"] = sc_crossing(d, cfg["ramps_deg"])
    return res


# ----------------------------------------------------------------------------------------------------
# 113: 멈춤 자세마다 손 평면 방향 (97 고침: 저장된 점, 해 고르기를 회전축·법선 일치로)
# ----------------------------------------------------------------------------------------------------
def pq_pauses(x: dict, still_frac: float, gap: int, min_len: int) -> list:
    """저장된 추적점에서 멈춤 구간: j 관절 중심의 프레임 사이 이동 < still_frac x 손 상자 긴 변, 간격 gap 이하로 이어진 min_len 개 이상."""
    kp, u = x["kp"], x["use"]
    c = np.nanmean(kp, axis=1)
    bl = np.nanmax(np.nanmax(kp, 1) - np.nanmin(kp, 1), 1)
    mv = np.full(len(kp), np.inf)
    mv[1:] = np.linalg.norm(np.diff(c, axis=0), axis=1) / bl[1:]
    still = np.flatnonzero(u & np.isfinite(mv) & (mv < still_frac))
    return [np.asarray(g) for g in hf.hp_groups(still, int(gap)) if len(g) >= int(min_len)]


def pq_decompose(P: list, K: np.ndarray) -> dict:
    """멈춤마다의 점 P[k] 로 첫 멈춤 -> 각 멈춤의 호모그래피를 알려진 K 로 분해. 쌍마다 두 시점 모두 점이 카메라 앞에 있는 해만 남기고,
    모든 쌍의 해 조합 중 (첫 멈춤 평면 법선의 흩어짐 + 회전축 방향의 흩어짐) 이 가장 작은 조합을 고른다 (97 은 법선만 봤음).
    반환: 회전 R[k] (첫 멈춤 = 단위), 멈춤별 평면 법선 n[k] (카메라 쪽), 회전축 평균 = 책상 법선, 조합 흩어짐(도), 호모그래피 잔차(px)."""
    import itertools
    import cv2
    Ki = np.linalg.inv(K)
    cands, fitres = [], []
    m0 = np.c_[P[0], np.ones(len(P[0]))] @ Ki.T
    for k in range(1, len(P)):
        Hm, _ = cv2.findHomography(P[0].astype(np.float64), P[k].astype(np.float64), 0)
        y = np.c_[P[0], np.ones(len(P[0]))] @ Hm.T
        fitres.append(float(np.sqrt(np.mean(np.sum((y[:, :2] / y[:, 2:3] - P[k]) ** 2, 1)))))
        _, Rs, ts, ns = cv2.decomposeHomographyMat(Hm, K)
        ok = []
        for R, t, n in zip(Rs, ts, ns):
            n, t = n.ravel() / np.linalg.norm(n), t.ravel()
            d = m0 @ n
            if (d <= 0).any() or (((m0 / d[:, None]) @ R.T + t)[:, 2] <= 0).any():
                continue
            rv = cv2.Rodrigues(R)[0].ravel()
            ok.append((R, n, rv / max(np.linalg.norm(rv), 1e-12)))
        cands.append(ok)
    if any(not c for c in cands):
        return {"failed": "호모그래피 분해에서 앞에 있는 해가 없음"}
    ang = lambda u, v: float(np.degrees(np.arccos(np.clip(abs(u @ v), -1.0, 1.0))))      # noqa: E731
    best = None
    for combo in itertools.product(*cands):
        n0 = [c[1] for c in combo]
        ax = [c[2] for c in combo]
        sp = (np.mean([ang(a, b) for a, b in itertools.combinations(n0, 2)]) if len(n0) > 1 else 0.0) + \
             (np.mean([ang(a, b) for a, b in itertools.combinations(ax, 2)]) if len(ax) > 1 else 0.0)
        if best is None or sp < best[0]:
            best = (sp, combo)
    combo = best[1]
    n0 = np.mean([c[1] if c[1] @ combo[0][1] >= 0 else -c[1] for c in combo], axis=0)
    n0 /= np.linalg.norm(n0)
    Rk = [np.eye(3)] + [c[0] for c in combo]
    ns = [R @ n0 for R in Rk]
    ns = [n if n[2] < 0 else -n for n in ns]
    acc = np.zeros(3)
    for c in combo:
        rv = cv2.Rodrigues(c[0])[0].ravel()
        acc += rv if acc @ rv >= 0 else -rv
    a = acc / np.linalg.norm(acc)
    a = a if a[2] < 0 else -a
    z = np.array([0.0, 0.0, 1.0])
    return {"desk_tilt_deg": ang(a, z), "hand_desk_deg": [ang(n, a) for n in ns], "turn_deg": [float(np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0]))) for R in Rk],
            "spread_deg": float(best[0]), "fit_rms_px": [0.0] + fitres}


def pq_job(args):
    """한 영상 x 점 출처: 저장된 추적점 -> 멈춤 구간(pq_pauses) -> 멈춤마다 관절 중앙값 -> 기준 f 와 fit 의 f 에서 pq_decompose."""
    e, cfg, run_root = args
    pc = cfg["pause_plane2"]
    ids = [int(j) for j in cfg["joint_ids"]]
    out = {}
    for src, runs in pc["points"].items():
        x = load_points(runs[e["file"]], e["file"])
        a = x["hand_angle"] - x["hand_angle"][np.isfinite(x["hand_angle"])][0]
        W, H = x["size"]
        P, info = [], []
        for g in pq_pauses(x, float(pc["still_frac"]), int(pc["pause_gap_frames"]), int(pc["min_pause_frames"])):
            g = g[np.isfinite(x["kp"][g][:, ids]).all((1, 2))]
            if len(g) == 0:
                continue
            P.append(np.median(x["kp"][g][:, ids], 0))
            info.append({"t0": float(x["times"][g[0]]), "t1": float(x["times"][g[-1]]), "hand_angle_deg": med(a[g])})
        fs = {"reference": float(e.get("ref_f_px") or cfg["ref_f"]["value"])}
        fr = pc.get("fit_f", {}).get(src, {}).get(e["file"])
        if fr:
            fs["fit"] = float(fr)
        out[src] = {"pauses": info}
        for name, f in fs.items():
            K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])
            r = pq_decompose(P, K) if len(P) >= 3 else {"failed": f"멈춤 {len(P)}개"}
            out[src][name] = r if "failed" in r else {"f": f, **r}
    return out, [f"  {e['file']}: " + ", ".join(f"{s} 멈춤 {len(o['pauses'])}개" for s, o in out.items())]


def pq_summary(res: dict, cfg: dict, own: dict) -> str:
    pc = cfg["pause_plane2"]
    vids = {e["file"]: e for e in cfg["videos"]}
    L = ["# 113 멈춤 자세마다 손 평면 방향 summary", "",
         f"**저장된 추적점({', '.join(pc['points'])}, {cfg['joint_set']})을 멈춤 구간마다 중앙값으로 모으고, 첫 멈춤 -> 각 멈춤의 호모그래피를 알려진 f 로 분해한다(cv2.decomposeHomographyMat). "
         "해는 두 시점 모두 점이 앞에 있는 것 중, 모든 멈춤에서 첫 멈춤 평면 법선과 회전축이 가장 잘 일치하는 조합(97 은 법선만 봐서 값이 맞지 않았음). 회전축 평균 = 책상 법선, "
         "손 경사 = 손 평면과 책상 사이 각. 기준 f 와 그 점의 경사 11° 추정 f 두 가지. 판정 없음.**", "",
         "- 확인용: 카메라 기울기는 촬영 때 수평계 45°. 흩어짐 = 고른 조합에서 법선·회전축이 멈춤마다 어긋난 정도(작을수록 평면 하나가 책상 법선 둘레로만 돈다는 모델과 맞음). 기준 f 는 이 파일에만.", ""]
    for src in pc["points"]:
        L += [f"## {src}", "", "| 영상 | 묶음 | f | 카메라 기울기 | 멈춤별 손 각도 | 멈춤별 손 경사 | 경사 범위 | 흩어짐 | 호모그래피 잔차 (px) |", "|---|---|---|---|---|---|---|---|---|"]
        for e in cfg["videos"]:
            d = (res.get(e["file"]) or {}).get(src) or {}
            for name in ("reference", "fit"):
                x = d.get(name)
                if x is None:
                    continue
                lab = f"{x['f']:.0f} (기준)" if name == "reference" and "f" in x else (f"{x['f']:.0f} (경사 11° 추정)" if "f" in x else name)
                if "hand_desk_deg" not in x:
                    L.append(f"| {e['file']} | {e['person']} | {lab} | 실패: {x.get('failed')} | | | | | |")
                    continue
                hd = x["hand_desk_deg"]
                L.append(f"| {e['file']} | {e['person']} | {lab} | {x['desk_tilt_deg']:.1f}° | " + ", ".join("-" if p["hand_angle_deg"] is None else f"{p['hand_angle_deg']:+.0f}°" for p in d["pauses"])
                         + " | " + ", ".join(f"{v:.1f}°" for v in hd) + f" | {max(hd) - min(hd):.1f}° | {x['spread_deg']:.1f}° | " + ", ".join(f"{v:.1f}" for v in x["fit_rms_px"][1:]) + " |")
        L.append("")
    L += ["## 한계", "", "- 멈춤 3~5개, 점 16개라 분해가 잡음에 민감하다. 광류 사슬 점은 평면 변환으로 옮긴 점이라 잔차가 0 에 가깝고, CoTracker3 점은 실제 관절 피부 점이다.", ""]
    return "\n".join(L) + "\n"


def pq_results_yaml(run_dir: Path, own: dict) -> Path:
    """113 의 값만: 경사 11° 추정 f 로 계산한 행만 (기준 f 로 계산한 행은 f 값이 기준 f 라 넣지 않음, summary.md 에)."""
    r = json.loads((run_dir / "results.json").read_text())["real"]
    m = json.loads((run_dir / "metadata.json").read_text())
    rv = lambda x: x if not isinstance(x, float) else (round(x, 3) if np.isfinite(x) else None)      # noqa: E731
    deep = lambda d: {k: (deep(v) if isinstance(v, dict) else [rv(q) if not isinstance(q, dict) else deep(q) for q in v] if isinstance(v, list) else rv(v)) for k, v in d.items()}      # noqa: E731
    out = {"experiment": own["experiment"]["name"], "run": str(run_dir.resolve().relative_to(hf.PROJECT_ROOT)), "code_commit": m["git"]["commit"][:7],
           "git_dirty_at_run": m["git"]["git_dirty"], "videos": {f: {s: deep({k: v for k, v in o.items() if k != "reference"}) for s, o in d.items()} for f, d in r.items()}}
    p = hf.resolve(own["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


# ----------------------------------------------------------------------------------------------------
# 114: 책상 운동 + 입체 손 모양 (optical_flow.desk3d), 저장된 관절별 추적점
# ----------------------------------------------------------------------------------------------------
def d3_run(cfg: dict, own: dict, log) -> dict:
    """114: 영상마다 저장된 추적점(desk3d.points) -> 방식(desk3d.variants: 관절, z_rel)마다 free_template -> dk3_job."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from optical_flow import desk3d
    dc = cfg["desk3d"]
    items = []
    for e in cfg["videos"]:
        x = load_points(dc["points"][e["file"]], e["file"])
        for v in dc["variants"]:
            items.append((e["file"], v["name"], x, [int(j) for j in v["joint_ids"]], v.get("z_rel")))
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        fts = list(ex.map(hf.tl_template_job, [(x["kp"], np.isfinite(x["kp"]).all(2), np.flatnonzero(x["use"]), x["box"], tuple(x["size"]), cfg["pipeline"]["estimate"], ids)
                                                for _, _, x, ids, _ in items], chunksize=1))
    jobs, keys, res = [], [], {}
    for (f, nm, x, ids, zr), ft in zip(items, fts):
        A = ft.get("A")
        res.setdefault(f, {})[nm] = {"free_template": {"failed": ft.get("failed")} if A is None else {"f": float(A["f"])}}
        if A is None:
            continue
        jobs.append((x["kp"], np.isfinite(x["kp"]).all(2), np.flatnonzero(x["use"]), x["box"], tuple(x["size"]), cfg["pipeline"]["estimate"], ids, A, float(dc["ramp_init"]), zr))
        keys.append((f, nm))
    log(f"  추정 {len(jobs)}건")
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        for (f, nm), o in zip(keys, ex.map(desk3d.dk3_job, jobs, chunksize=1)):
            res[f][nm]["desk3d"] = o
    return res


def d3_summary(res: dict, cfg: dict, own: dict) -> str:
    dc = cfg["desk3d"]
    vids = {e["file"]: e for e in cfg["videos"]}
    cmp_ = {k: json.loads((hf.resolve(v) / "results.json").read_text())["real"] for k, v in dc.get("compare_runs", {}).items()}
    L = ["# 114 책상 운동 + 입체 손 모양 summary", "",
         "**저장된 CoTracker3 관절별 추적점에, 손을 평면 + 경사가 아니라 관절마다 높이를 가진 단단한 3D 모양으로 둔 책상 운동 추정기(optical_flow.desk3d)를 돌린다. 경사 가정 없음"
         f"(시작값만 경사 {dc['ramp_init']:g}° 평면). 비교 = 같은 점의 평면 손·경사 11° 고정(98/105). 판정 없음.**", "",
         "- 기준 f 는 이 파일에만. * = f 탐색 범위 끝. 높이 = 찾은 손 모양의 높이 범위 / 손 크기. z_rel 없음 = 높이 제약 없음.", "",
         "| 영상 | 묶음 | 기준 f | 평면 손 경사 11° (98/105) | " + " | ".join(f"3D {v['name']}" for v in dc["variants"]) + " |", "|---|---|---|---|" + "---|" * len(dc["variants"])]
    for f, d in res.items():
        e = vids[f]
        rf = float(e.get("ref_f_px") or cfg["ref_f"]["value"])
        fmt = lambda x: "-" if not x or "f" not in x else f"{x['f']:.0f} ({100 * (x['f'] / rf - 1):+.1f}%)" + ("*" if x.get("boundary") else "")      # noqa: E731
        base = next((c[f].get("ramp_11") for c in cmp_.values() if f in c), None)
        cells = []
        for v in dc["variants"]:
            o = (d.get(v["name"]) or {}).get("desk3d") or {}
            cells.append(f"실패: {o['failed'][:40]}" if "failed" in o else f"{fmt(o)}, 기울기 {o['desk_tilt_deg']:.0f}°, 높이 {o['height_rel']:.2f}, RMS {o['rms_px']:.1f}")
        L.append(f"| {f} | {e['person']} | {rf:g} | {fmt(base)} | " + " | ".join(cells) + " |")
    L += ["", "## 한계", "", "- 손이 실제로 변형되면 단단한 3D 모양으로도 다 맞지 않는다. CoTracker3 점은 2 프레임마다(15 fps).", ""]
    return "\n".join(L) + "\n"


def d3_results_yaml(run_dir: Path, own: dict) -> Path:
    """114 의 값만 (기준 f, 오차 없음): 영상 x 방식별 f, 카메라 기울기, 높이, RMS, 비용."""
    r = json.loads((run_dir / "results.json").read_text())["real"]
    m = json.loads((run_dir / "metadata.json").read_text())
    rv = lambda x: x if not isinstance(x, float) else (round(x, 3) if np.isfinite(x) else None)      # noqa: E731
    keys = ("f", "desk_tilt_deg", "height_rel", "rms_px", "cost", "boundary", "failed")
    out = {"experiment": own["experiment"]["name"], "run": str(run_dir.resolve().relative_to(hf.PROJECT_ROOT)), "code_commit": m["git"]["commit"][:7],
           "git_dirty_at_run": m["git"]["git_dirty"],
           "videos": {f: {nm: {k: {kk: rv(o[kk]) for kk in keys if kk in o} for k, o in dd.items()} for nm, dd in d.items()} for f, d in r.items()}}
    p = hf.resolve(own["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


# ----------------------------------------------------------------------------------------------------
# 115: 114 + 닿는 관절 높이 고정 (optical_flow.desk3d.dk3c_job)
# ----------------------------------------------------------------------------------------------------
def d3c_run(cfg: dict, own: dict, log) -> dict:
    """115: d3_run 과 같고, 방식마다 contact(높이를 같게 고정할 관절)를 넘겨 dk3c_job 으로 푼다. 결과 모양이 d3_run 과 같아 d3_summary / d3_results_yaml 로 적는다."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from optical_flow import desk3d
    dc = cfg["desk3d"]
    items = []
    for e in cfg["videos"]:
        x = load_points(dc["points"][e["file"]], e["file"])
        for v in dc["variants"]:
            items.append((e["file"], v["name"], x, [int(j) for j in v["joint_ids"]], v.get("z_rel"), [int(j) for j in v["contact"]]))
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        fts = list(ex.map(hf.tl_template_job, [(x["kp"], np.isfinite(x["kp"]).all(2), np.flatnonzero(x["use"]), x["box"], tuple(x["size"]), cfg["pipeline"]["estimate"], ids)
                                                for _, _, x, ids, _, _ in items], chunksize=1))
    jobs, keys, res = [], [], {}
    for (f, nm, x, ids, zr, ct), ft in zip(items, fts):
        A = ft.get("A")
        res.setdefault(f, {})[nm] = {"free_template": {"failed": ft.get("failed")} if A is None else {"f": float(A["f"])}}
        if A is None:
            continue
        jobs.append((x["kp"], np.isfinite(x["kp"]).all(2), np.flatnonzero(x["use"]), x["box"], tuple(x["size"]), cfg["pipeline"]["estimate"], ids, A, float(dc["ramp_init"]), zr, ct))
        keys.append((f, nm))
    log(f"  추정 {len(jobs)}건")
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        for (f, nm), o in zip(keys, ex.map(desk3d.dk3c_job, jobs, chunksize=1)):
            res[f][nm]["desk3d"] = o
    return res


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


# ----------------------------------------------------------------------------------------------------
# 116: 함수(단계)별 중간값 비교 (교수님 제안: "함수별로 값이 어떻게 얼마나 다른지") - 팀원 영상 vs 내 영상
# ----------------------------------------------------------------------------------------------------
SV_FINGERS = {"index": (5, 8), "middle": (9, 12), "ring": (13, 16), "pinky": (17, 20)}


def sv_pct(x, q):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if len(x) else None


def sv_job(args):
    """한 영상: 앞 단계(프레임 추출 -> MediaPipe 검출 -> track -> 기준 프레임 -> 광류 사슬)를 run.py 와 같은 함수로 다시 돌리며 단계마다 값을 남긴다. 추적점은 저장된 점(stage_values.points_run)과 비교."""
    e, cfg, run_root = args
    import cv2
    cv2.setNumThreads(1)
    sv = cfg["stage_values"]
    pngs, times, tr, (W, H), tmp = prepare(e, cfg, run_root)
    try:
        ref = rof.choose_reference_pose(tr, (W, H), cfg)
        x = of.track_hand(pngs, tr["kp"], tr["present"], tr["box"], (W, H), cfg["tracking"], ref=ref)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    x["size"] = (W, H)
    pres = tr["present"]
    bl = of.box_long(np.asarray(tr["box"], float))
    sel = hf.od_select(tr["kp"], tr["mask"], pres, tr["box"], (W, H), cfg["selection"])
    still = np.isfinite(sel["motion"]) & (sel["motion"] < float(cfg["selection"]["still_frac"])) & pres
    ang = hand_angle(tr)
    ids = cfg["joint_ids"]
    drift = region_drift(x["kp"], np.where(pres[:, None, None], tr["kp"], np.nan), tr["box"], x["use"], ids)
    saved = load_points(sv["points_run"], e["file"])
    same = saved["kp"].shape == x["kp"].shape
    dsave = float(np.nanmax(np.abs(saved["kp"] - x["kp"]))) if same and np.isfinite(x["kp"]).any() else None
    inl = np.asarray(x["inl"], float)
    out = {
        "fresp_prepare": {"n_frames": int(len(times)), "width": int(W), "height": int(H), "duration_s": float(times[-1] - times[0])},
        "mediapipe_track": {"n_present": int(pres.sum()), "detect_rate": float(pres.mean()), "box_long_px_median": sv_pct(bl[pres], 50),
                            "box_long_over_height": sv_pct(bl[pres] / H, 50), "kconf_median": sv_pct(tr["kconf"][pres][:, ids], 50),
                            "hand_angle_min_deg": sv_pct(ang - ang[np.isfinite(ang)][0], 0), "hand_angle_max_deg": sv_pct(ang - ang[np.isfinite(ang)][0], 100)},
        "od_select": {"n_still": int(still.sum()), "still_motion_median_pct": sv_pct(100 * sel["motion"][still], 50), "motion_median_pct": sv_pct(100 * sel["motion"][pres], 50),
                      "shape_z_median": sv_pct(sel["shape_z"][pres], 50)},
        "choose_reference_pose": {"ref": int(ref), "ref_time_s": float(times[ref]), "ref_box_long_px": float(bl[ref])},
        "track_hand": {"n_use": int(x["use"].sum()), "breaks": int(x["breaks"]), "inliers_median": sv_pct(inl, 50), "inliers_p10": sv_pct(inl, 10), "inliers_min": sv_pct(inl, 0),
                       "drift_median_pct": sv_pct(drift, 50), "drift_p90_pct": sv_pct(drift, 90), "max_diff_to_saved_px": dsave},
    }
    pts = {"kp": x["kp"], "box": x["box"], "use": x["use"], "ref": x["ref"], "size": (W, H)}
    return {"stages": out, "points": pts}, [f"  {e['file']}: 프레임 {len(times)}, 기준 {ref}, 사용 {int(x['use'].sum())}, 저장점과 최대 차이 {dsave}"]


def sv_problem_values(x: dict, cfg: dict, A: dict) -> dict:
    """od_bridge -> free_problem / _restrict 의 값과, free_template 결과(평면 손 모양, 프레임별 자세 od_poses)의 값."""
    ecfg = cfg["pipeline"]["estimate"]
    ids = cfg["joint_ids"]
    pb = hf.od_bridge(x["kp"], np.isfinite(x["kp"]).all(2), np.flatnonzero(x["use"]), x["box"], tuple(x["size"]), ecfg, ids)
    pr, rid = hf._restrict(pb, max_frames=120)
    out = {"od_bridge": {"n_frames": int(len(pb.keep)), "n_points": int(pb.n_points), "cx": float(pb.cx), "cy": float(pb.cy), "huber_px_median": sv_pct(pb.huber, 50)},
           "_restrict": {"n_frames": int(len(pr.keep)), "n_joints": int(len(rid)), "n_points": int(pr.n_points)}}
    if A is None:
        return out
    T = np.asarray(A["template"], float)
    jid = [int(j) for j in A["joint_ids"]]
    P = {j: T[jid.index(j)] for j in jid}
    L = lambda a, b: float(np.linalg.norm(P[a] - P[b])) if a in P and b in P else None      # noqa: E731
    width = L(5, 17)
    fin = {k: L(a, b) for k, (a, b) in SV_FINGERS.items()}
    c = np.asarray(A["profile_cost"], float)
    out["free_template"] = {"f": float(A["f"]), "f_grid": float(A["f_grid"]), "at_grid_edge": bool(A["at_grid_edge"]), "rms_px": float((A.get("ba") or {}).get("rms_px", np.nan)),
                            "profile_f": list(map(float, A["profile_f"])), "profile_cost": c.tolist(), "profile_max_over_min": float(c.max() / c.min()),
                            "finger_over_palm_width": {k: (v / width if v and width else None) for k, v in fin.items()},
                            "finger_over_middle": {k: (v / fin["middle"] if v and fin["middle"] else None) for k, v in fin.items()}}
    po = hf.od_poses(pr, float(A["f"]), T, list(range(len(jid))), of.box_long(np.asarray(x["box"], float)[np.flatnonzero(x["use"])]))
    k = np.asarray(pr.keep, int)
    tl, tn, rs = po["tilt_deg"][k], po["turn_deg"][k], po["res_pct"][k]
    out["od_poses"] = {"tilt_median_deg": sv_pct(tl, 50), "tilt_p10_deg": sv_pct(tl, 10), "tilt_p90_deg": sv_pct(tl, 90),
                       "turn_range_deg": (sv_pct(tn, 100) - sv_pct(tn, 0)) if np.isfinite(tn).any() else None, "res_median_pct": sv_pct(rs, 50)}
    return out


def sv_run(cfg: dict, vid: dict, log) -> dict:
    """116: 영상마다 free_template(tl_template_job) -> 경사 고정·함께 추정(ms2_job, 실험 94 와 같은 목록)과, 그 사이 함수들의 값."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    files = list(vid)
    pts = {f: vid[f]["points"] for f in files}
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        fts = dict(zip(files, ex.map(hf.tl_template_job, [rof.estimate_args(cfg, pts[f]) for f in files], chunksize=1)))
    todo = [(f"ramp_{r:g}", "ramp_fixed", r) for r in cfg["ramps_deg"]] + [(f"free_{r:g}", "ramp_free", r) for r in cfg["free_starts_deg"]]
    res, jobs, keys = {}, [], []
    for f in files:
        A = fts[f].get("A")
        res[f] = dict(vid[f]["stages"])
        res[f].update(sv_problem_values(pts[f], cfg, A))
        if A is None:
            res[f]["free_template"] = {"failed": fts[f].get("failed")}
            continue
        for mk, mode, r in todo:
            jobs.append(rof.estimate_args(cfg, pts[f], A) + (mode, r))
            keys.append((f, mk))
    log(f"  추정 {len(jobs)}건")
    ms = {}
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        for (f, mk), o in zip(keys, ex.map(hf.ms2_job, jobs, chunksize=1)):
            ms.setdefault(f, {})[mk] = o
    keep = ("f", "pitch_deg", "roll_deg", "desk_tilt_deg", "turn_range_deg", "rms_px", "cost", "boundary", "start", "profile_f", "profile_cost")
    for f, d in ms.items():
        fixed = {k: v for k, v in d.items() if k.startswith("ramp_")}
        free = {k: v for k, v in d.items() if k.startswith("free_") and "failed" not in v}
        res[f]["dk_estimate_ms"] = {**{k: ({kk: v[kk] for kk in keep if kk in v} if "failed" not in v else v) for k, v in fixed.items()}}
        if free:
            bk = min(free, key=lambda k: free[k]["cost"])
            res[f]["dk_estimate_ms"]["ramp_free"] = {**{kk: free[bk][kk] for kk in keep if kk in free[bk]}, "best_start": bk,
                                                     "all_starts": {k: {kk: v[kk] for kk in ("f", "pitch_deg", "cost")} for k, v in free.items()}}
    return res


def sv_summary(res: dict, cfg: dict, own: dict) -> str:
    vids = {e["file"]: e for e in cfg["videos"]}
    order = sorted(res, key=lambda f: (vids[f]["person"] != "teammate", f))
    who = {"teammate": "팀원", "self": "나"}
    head = "| 함수 | 값 | " + " | ".join(f"{Path(f).stem} {who.get(vids[f]['person'], vids[f]['person'])}" for f in order) + " |"
    sep = "|---|---|" + "---|" * len(order)

    def g(d, path):
        for p in path:
            if not isinstance(d, dict) or p not in d:
                return None
            d = d[p]
        return d

    def fmt(v, nd):
        if v is None:
            return "-"
        if isinstance(v, bool):
            return "예" if v else "아니오"
        return f"{v:.{nd}f}" if isinstance(v, float) else str(v)

    def row(fn, lab, path, nd=1):
        return f"| {fn} | {lab} | " + " | ".join(fmt(g(res[f], path), nd) for f in order) + " |"

    def frow(fn, lab, path):
        cells = []
        for f in order:
            v = g(res[f], path)
            rf = float(vids[f].get("ref_f_px") or cfg["ref_f"]["value"])
            cells.append("-" if v is None else f"{v:.0f} ({100 * (v / rf - 1):+.1f}%)")
        return f"| {fn} | {lab} | " + " | ".join(cells) + " |"
    R = [("fresp_prepare", "프레임 수", ["fresp_prepare", "n_frames"], 0), ("fresp_prepare", "길이 (s)", ["fresp_prepare", "duration_s"], 1),
         ("MediaPipe + track", "검출 프레임", ["mediapipe_track", "n_present"], 0), ("MediaPipe + track", "검출률", ["mediapipe_track", "detect_rate"], 3),
         ("MediaPipe + track", "손 상자 긴 변 중앙값 (px)", ["mediapipe_track", "box_long_px_median"], 0), ("MediaPipe + track", "손 상자 / 화면 높이", ["mediapipe_track", "box_long_over_height"], 3),
         ("MediaPipe + track", "관절 신뢰도 중앙값", ["mediapipe_track", "kconf_median"], 3),
         ("MediaPipe + track", "손 방향 최소 (도, 첫 검출 대비)", ["mediapipe_track", "hand_angle_min_deg"], 0), ("MediaPipe + track", "손 방향 최대 (도)", ["mediapipe_track", "hand_angle_max_deg"], 0),
         ("od_select", "정지 프레임 수", ["od_select", "n_still"], 0), ("od_select", "정지 중 검출 흔들림 (상자 대비 %)", ["od_select", "still_motion_median_pct"], 3),
         ("od_select", "움직임 중앙값 (상자 대비 %)", ["od_select", "motion_median_pct"], 2), ("od_select", "손 모양 z 중앙값", ["od_select", "shape_z_median"], 2),
         ("choose_reference_pose", "기준 프레임", ["choose_reference_pose", "ref"], 0), ("choose_reference_pose", "기준 시각 (s)", ["choose_reference_pose", "ref_time_s"], 2),
         ("choose_reference_pose", "기준 프레임 손 상자 (px)", ["choose_reference_pose", "ref_box_long_px"], 0),
         ("track_hand", "사용 프레임", ["track_hand", "n_use"], 0), ("track_hand", "끊김", ["track_hand", "breaks"], 0),
         ("track_hand", "inlier 중앙값", ["track_hand", "inliers_median"], 0), ("track_hand", "inlier 하위 10%", ["track_hand", "inliers_p10"], 0),
         ("track_hand", "추적점-검출 관절 거리 중앙값 (상자 %)", ["track_hand", "drift_median_pct"], 2), ("track_hand", "같은 거리 상위 10% (상자 %)", ["track_hand", "drift_p90_pct"], 2),
         ("track_hand", "저장된 점(100)과 최대 차이 (px)", ["track_hand", "max_diff_to_saved_px"], 3),
         ("od_bridge", "프레임", ["od_bridge", "n_frames"], 0), ("od_bridge", "Huber 폭 중앙값 (px)", ["od_bridge", "huber_px_median"], 2),
         ("_restrict", "추정에 쓰는 프레임", ["_restrict", "n_frames"], 0), ("_restrict", "관절", ["_restrict", "n_joints"], 0)]
    L = ["# 116 함수별 값 비교 summary", "",
         "**같은 카메라·같은 장소에서 찍은 새 영상 4개(팀원 7384·7387, 나 7385·7388)에 광류 파이프라인(실험 94 와 같은 설정: 첫 정지 구간 기준, j16, 경사 11° 고정 + 함께 추정)을 "
         "다시 돌리며 함수마다 나온 값을 나란히 적는다. 판정 없음.**", "",
         "- 기준 f(1672 px, 새 폰 체커보드)는 이 파일에만.", "", "## 앞 단계 (영상 -> 추적점)", "", head, sep]
    L += [row(*r) for r in R]
    L += ["", "## 뒤 단계 (추적점 -> f)", "", head, sep,
          frow("free_template", "f (오차)", ["free_template", "f"]), row("free_template", "탐색 범위 끝 (거친 단계)", ["free_template", "at_grid_edge"]),
          row("free_template", "RMS (px)", ["free_template", "rms_px"], 2), row("free_template", "f 곡선 최대/최소 비용", ["free_template", "profile_max_over_min"], 2)]
    for k in SV_FINGERS:
        L.append(row("free_template 손 모양", f"{k} 길이 / 손바닥 폭(5-17)", ["free_template", "finger_over_palm_width", k], 3))
    L += [row("od_poses (free_template f)", "손 평면 기울기 중앙값 (도)", ["od_poses", "tilt_median_deg"], 1),
          row("od_poses", "기울기 하위 10% (도)", ["od_poses", "tilt_p10_deg"], 1),
          row("od_poses", "기울기 상위 10% (도)", ["od_poses", "tilt_p90_deg"], 1), row("od_poses", "회전 범위 (도)", ["od_poses", "turn_range_deg"], 0),
          row("od_poses", "재투영 오차 중앙값 (상자 %)", ["od_poses", "res_median_pct"], 2)]
    for r in cfg["ramps_deg"]:
        k = f"ramp_{r:g}"
        L += [frow(f"dk_estimate_ms 경사 {r:g}° 고정", "f (오차)", ["dk_estimate_ms", k, "f"]), row("", "책상 기울기 (도)", ["dk_estimate_ms", k, "desk_tilt_deg"], 1),
              row("", "회전 범위 (도)", ["dk_estimate_ms", k, "turn_range_deg"], 0), row("", "RMS (px)", ["dk_estimate_ms", k, "rms_px"], 2)]
    L += [frow("dk_estimate_ms 경사 함께 추정", "f (오차)", ["dk_estimate_ms", "ramp_free", "f"]), row("", "경사 pitch (도)", ["dk_estimate_ms", "ramp_free", "pitch_deg"], 1),
          row("", "roll (도)", ["dk_estimate_ms", "ramp_free", "roll_deg"], 1), row("", "책상 기울기 (도)", ["dk_estimate_ms", "ramp_free", "desk_tilt_deg"], 1),
          row("", "RMS (px)", ["dk_estimate_ms", "ramp_free", "rms_px"], 2), row("", "고른 시작", ["dk_estimate_ms", "ramp_free", "best_start"])]
    L += ["", "## f 곡선 (경사 11° 고정, f 마다 비용 / 최소 비용)", ""]
    for f in order:
        d = g(res[f], ["dk_estimate_ms", f"ramp_{cfg['ramps_deg'][0]:g}"]) or {}
        if d.get("profile_f"):
            c = np.asarray(d["profile_cost"], float)
            L.append(f"- {f}: " + ", ".join(f"{pf:.0f}: {cc / c.min():.2f}" for pf, cc in zip(d["profile_f"], c)))
    L += ["", "## 한계", "", "- 영상 4개(사람당 2개). 차이가 사람 때문인지 그 영상 때문인지는 이 표만으로 가를 수 없다.",
          "- 손 모양 비율은 free_template 이 f 와 함께 찾은 평면 모양이라 f 가 틀리면 같이 틀어진다.", ""]
    return "\n".join(L) + "\n"


def sv_results_yaml(run_dir: Path, own: dict) -> Path:
    """116 의 값만 (기준 f, 오차 없음)."""
    r = json.loads((run_dir / "results.json").read_text())["real"]
    m = json.loads((run_dir / "metadata.json").read_text())
    rv = lambda x: x if not isinstance(x, float) else (round(x, 4) if np.isfinite(x) else None)      # noqa: E731
    deep = lambda d: {k: (deep(v) if isinstance(v, dict) else [rv(t) for t in v] if isinstance(v, list) else rv(v)) for k, v in d.items()}      # noqa: E731
    out = {"experiment": own["experiment"]["name"], "run": str(run_dir.resolve().relative_to(hf.PROJECT_ROOT)), "code_commit": m["git"]["commit"][:7],
           "git_dirty_at_run": m["git"]["git_dirty"], "videos": {f: deep(d) for f, d in r.items()}}
    p = hf.resolve(own["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


# ----------------------------------------------------------------------------------------------------
# 117: 함수마다 실제로 들어가고 나온 값 (116 의 보충: 입력의 구체적인 값, 특징점 수와 손 위 분포, 손 모양 좌표, 책상 자세)
# ----------------------------------------------------------------------------------------------------
def si_job(args):
    """한 영상: 116 의 앞 단계를 다시 돌리되 광류 사슬은 kt_chain_log(같은 단계 + 특징점 기록)로. 프레임 간격, 후보 수, 손 위치, 정지 구간, 기준 관절 좌표, 특징점 수와 손 위 분포를 남긴다."""
    e, cfg, run_root = args
    import cv2
    cv2.setNumThreads(1)
    si = cfg["stage_inputs"]
    kc = cfg["tracking"]
    pngs, times, tr, (W, H), tmp = prepare(e, cfg, run_root)
    try:
        n = len(pngs)
        ref = rof.choose_reference_pose(tr, (W, H), cfg)
        pauses = pp_pauses(tr, (W, H), cfg)
        pres = np.flatnonzero(tr["present"])
        a, b = int(pres[0]), int(pres[-1])
        kps = np.stack([of.kt_carry(tr["present"], tr["kp"][:, j, k]) for j in range(21) for k in range(2)], axis=1).reshape(n, 21, 2)
        boxes = np.stack([of.kt_carry(tr["present"], tr["box"][:, k]) for k in range(4)], axis=1)
        fwd = of.kt_chain_log(pngs, kps, boxes, list(range(ref, b + 1)), kc)
        bwd = of.kt_chain_log(pngs, kps, boxes, list(range(ref, a - 1, -1)), kc)
        kp, tb, valid, edge = of.chain_points(tr["kp"][ref], {**bwd["H"], **fwd["H"]}, n, W, H)
        raw = []
        for i in range(a, b + 1, int(si["corner_sample_every"])):                      # 전후방 검사 전의 특징점 수 (표본 프레임)
            g = cv2.cvtColor(cv2.imread(pngs[i], cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
            m = of.kt_hand_mask(kps[i], boxes[i], g.shape, kc["dilate_frac"])
            c = cv2.goodFeaturesToTrack(g, int(kc["max_corners"]), float(kc["quality"]), float(kc["min_distance"]), mask=m, blockSize=int(kc["block"]))
            raw.append((0 if c is None else len(c), int((m > 0).sum())))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    use = valid & ~edge
    saved = load_points(si["points_run"], e["file"])
    dsave = float(np.nanmax(np.abs(saved["kp"] - kp))) if saved["kp"].shape == kp.shape else None
    P = tr["present"]
    box = np.asarray(tr["box"], float)
    cen = np.c_[(box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2][P]
    s = np.concatenate([fwd["s"], bwd["s"]])
    t = np.concatenate([fwd["t"], bwd["t"]])
    res = np.concatenate([fwd["res"], bwd["res"]])
    n_ok = np.asarray(fwd["n_ok"] + bwd["n_ok"], float)
    inl = np.asarray(fwd["inl"] + bwd["inl"], float)
    raw = np.asarray(raw, float)
    nc = tr["n_candidates"]
    dt = np.diff(times)
    out = {
        "extract_frames": {"n_frames": int(n), "frame_step_s_median": float(np.median(dt)), "first_time_s": float(times[0]), "last_time_s": float(times[-1])},
        "track": {"init_frame": int(tr["init_frame"]), "frames_with_0_candidates": int((nc == 0).sum()), "frames_with_1_candidate": int((nc == 1).sum()),
                  "frames_with_2_candidates": int((nc >= 2).sum()), "cls_left_frames": int((tr["cls"][P] == 0).sum()), "cls_right_frames": int((tr["cls"][P] == 1).sum()),
                  "box_conf_median": sv_pct(tr["box_conf"][P], 50), "box_center_x_mean": float(cen[:, 0].mean()), "box_center_y_mean": float(cen[:, 1].mean()),
                  "box_center_x_min": float(cen[:, 0].min()), "box_center_x_max": float(cen[:, 0].max()), "box_center_y_min": float(cen[:, 1].min()), "box_center_y_max": float(cen[:, 1].max()),
                  "box_width_median": sv_pct((box[:, 2] - box[:, 0])[P], 50), "box_height_median": sv_pct((box[:, 3] - box[:, 1])[P], 50)},
        "choose_reference_pose": {"n_pauses": int(len(pauses)), "first_pause_start": int(pauses[0][0]) if pauses else None, "first_pause_end": int(pauses[0][-1]) if pauses else None,
                                  "first_pause_n": int(len(pauses[0])) if pauses else None, "ref": int(ref), "ref_time_s": float(times[ref]),
                                  "ref_box": [float(v) for v in box[ref]], "ref_joints_xy": [[float(v) for v in p] for p in tr["kp"][ref]]},
        "kt_chain": {"steps": int(len(n_ok)), "mask_area_px_median": sv_pct(raw[:, 1], 50), "corners_found_median": sv_pct(raw[:, 0], 50), "corners_found_min": sv_pct(raw[:, 0], 0),
                     "corners_found_max": sv_pct(raw[:, 0], 100), "frames_at_max_corners": float((raw[:, 0] >= int(kc["max_corners"])).mean()),
                     "corners_per_1000px_median": sv_pct(1000 * raw[:, 0] / np.maximum(raw[:, 1], 1), 50),
                     "fb_pass_median": sv_pct(n_ok, 50), "inliers_median": sv_pct(inl, 50), "inlier_over_fb_pass_median": sv_pct(inl / np.maximum(n_ok[:len(inl)], 1), 50) if len(inl) == len(n_ok) else None,
                     "homography_residual_px_median": sv_pct(res, 50), "homography_residual_px_p90": sv_pct(res, 90),
                     "inlier_frac_forearm_s_lt_0": float((s < 0).mean()), "inlier_frac_palm_s_0_1": float(((s >= 0) & (s < 1)).mean()), "inlier_frac_fingers_s_ge_1": float((s >= 1).mean()),
                     "inlier_s_median": sv_pct(s, 50), "inlier_s_p10": sv_pct(s, 10), "inlier_s_p90": sv_pct(s, 90), "inlier_abs_t_median": sv_pct(np.abs(t), 50),
                     "breaks": int(fwd["breaks"] + bwd["breaks"])},
        "chain_points": {"n_valid": int(valid.sum()), "n_edge": int(edge.sum()), "n_use": int(use.sum()), "max_diff_to_saved_px": dsave,
                         "tracked_box_long_median": sv_pct(of.box_long(tb[valid]), 50)},
    }
    pts = {"kp": kp, "box": tb, "use": use, "ref": ref, "size": (W, H)}
    return {"stages": out, "points": pts}, [f"  {e['file']}: 프레임 {n}, 기준 {ref}, 사용 {int(use.sum())}, 저장점과 최대 차이 {dsave}"]


def si_dk_estimate_ms(pb, grid: np.ndarray, A: dict, mode: str, ramp_deg: float, coarse_n: int = 8, max_frames: int = 120, n_final: int = 3) -> dict:
    """handfocal.dk_estimate_ms 와 같은 절차(거친 f 곡선 -> 따뜻한 재풀이 -> 시작점 n_final + free_template f 에서 모두 풀기 -> 비용 최소)를 그대로 따르되, 고른 해의 state 를 버리지 않고 돌려준다
    (dk_estimate_ms 는 state 를 지운다). 같은 dk_solve 호출을 같은 순서로 하므로 f·비용은 dk_estimate_ms 와 같아야 한다(실험 116 의 값과 비교)."""
    pr, ids = hf._restrict(pb, max_frames=max_frames)
    if list(ids) != list(A["joint_ids"]):
        raise ValueError("free_template result was computed on different joints")
    jid = [int(j) for j in A["joint_ids"]]
    T2 = np.asarray(A["template"], float)
    g = grid[np.unique(np.linspace(0, len(grid) - 1, coarse_n).round().astype(int))] if len(grid) > coarse_n else grid
    p0 = np.radians(0.0 if mode == "flat" else float(ramp_deg))
    fr = mode == "ramp_free"
    fb = (grid[0], grid[-1])
    prof = [hf.dk_solve(pr, float(f), T2, jid, p0, 0.0, False, fr, fb) for f in g]
    c_first = [x["cost"] for x in prof]
    n_warm = 0
    for d, idx in ((-1, range(1, len(g))), (1, range(len(g) - 2, -1, -1))):
        for i in idx:
            s = prof[i + d]["state"]
            x = hf.dk_solve(pr, float(g[i]), T2, jid, s["pitch"], s["roll"], False, fr, fb, init=dict(s, f=float(g[i])))
            if x["cost"] < prof[i]["cost"]:
                prof[i] = x
                n_warm += 1
    c = np.array([x["cost"] for x in prof])
    starts = []
    for k in np.argsort(c)[:n_final]:
        k = int(k)
        loc = 0 < k < len(g) - 1 and c[k] <= c[k - 1] and c[k] <= c[k + 1]
        fs = float(np.exp(np.clip(hf._parabola(np.log(g), c, k), np.log(fb[0]), np.log(fb[1])))) if loc else float(g[k])
        starts.append((f"profile_{k}", fs, prof[k]["state"]))
    fa = float(np.clip(A["f"], fb[0], fb[1]))
    k = int(np.argmin(np.abs(np.log(g) - np.log(fa))))
    s = prof[k]["state"]
    xa = hf.dk_solve(pr, fa, T2, jid, s["pitch"], s["roll"], False, fr, fb, init=dict(s, f=fa))
    starts.append(("free_template", fa, xa["state"]))
    finals = []
    for name, fs, s in starts:
        B = hf.dk_solve(pr, fs, T2, jid, s["pitch"], s["roll"], True, fr, fb, init=dict(s, f=fs))
        finals.append((name, fs, B))
    best = min(range(len(finals)), key=lambda i: finals[i][2]["cost"])
    B = dict(finals[best][2])
    B.update({"start": finals[best][0], "f_start": finals[best][1], "joint_ids": jid, "profile_f": g.tolist(), "profile_cost_first": c_first, "profile_cost": c.tolist(), "n_warm_improved": n_warm,
              "starts": [{"start": nm, "f_start": fs, "f": b["f"], "cost": b["cost"], "rms_px": b["rms_px"], "pitch_deg": b["pitch_deg"]} for nm, fs, b in finals],
              "anchors": [int(jid[a]) for a in hf._anchors(T2, [pr.frames[i] for i in pr.keep])], "n_frames": int(len(pr.keep))})
    return B


def si_ms_job(args) -> dict:
    """handfocal.ms2_job 과 같은 입력·같은 출력 + 고른 해의 상태(책상 법선·위치, 프레임별 회전·이동, 최종 손 모양). args = (kp, mask, frames, box, size_wh, ecfg, ids, A, mode, ramp_deg)."""
    kp, mask, frames, box, size_wh, ecfg, ids, A, mode, ramp_deg = args
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import cv2
    cv2.setNumThreads(1)
    grid = hf.f_grid(ecfg, max(*size_wh))
    t0 = time.time()
    try:
        pb = hf.od_bridge(kp, mask, frames, box, size_wh, ecfg, ids)
        B = si_dk_estimate_ms(pb, grid, A, mode, ramp_deg)
    except Exception as e:
        return {"failed": f"{type(e).__name__}: {e}"}
    out = {kk: B[kk] for kk in ("f", "pitch_deg", "roll_deg", "desk_tilt_deg", "turn_range_deg", "rms_px", "cost", "success", "start", "f_start", "n_warm_improved", "starts",
                                 "profile_f", "profile_cost_first", "profile_cost")}
    out["boundary"] = bool(B["at_bound"]) or hf.od_boundary(float(B["f"]), grid)
    st = B["state"]
    jid = B["joint_ids"]
    T = np.asarray(st["T"], float)
    Rd, td = np.asarray(st["Rd"], float), np.asarray(st["td"], float)
    L = lambda p, q: float(np.linalg.norm(T[jid.index(p)] - T[jid.index(q)]))      # noqa: E731
    th = np.degrees(np.asarray(st["theta"], float))
    xy = np.asarray(st["xy"], float)
    pw = L(5, 17)
    out["state"] = {"up": float(st["up"]), "desk_normal_cam": [float(v) for v in Rd[:, 2]], "td": [float(v) for v in td], "depth_z": float(td[2]), "plane_distance": float(abs(Rd[:, 2] @ td)),
                    "theta_min_deg": float(th.min()), "theta_max_deg": float(th.max()), "slide_x_min": float(xy[:, 0].min()), "slide_x_max": float(xy[:, 0].max()),
                    "slide_y_min": float(xy[:, 1].min()), "slide_y_max": float(xy[:, 1].max()), "anchor_joints": B["anchors"], "middle_frame_index": int(st["ref"]),
                    "n_frames": B["n_frames"], "palm_width_5_17": pw, "middle_len_9_12": L(9, 12), "depth_over_palm_width": float(td[2] / pw),
                    "slide_range_over_palm_width": float(np.hypot(xy[:, 0].max() - xy[:, 0].min(), xy[:, 1].max() - xy[:, 1].min()) / pw),
                    "template_xy": {str(j): [float(v) for v in T[k]] for k, j in enumerate(jid)}}
    out["elapsed_s"] = float(time.time() - t0)
    return out


def si_run(cfg: dict, vid: dict, log) -> dict:
    """117: free_template -> si_ms_job (ms2_job 과 같은 계산 + 고른 해의 상태; 116 과 같은 목록)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    files = list(vid)
    pts = {f: vid[f]["points"] for f in files}
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        fts = dict(zip(files, ex.map(hf.tl_template_job, [rof.estimate_args(cfg, pts[f]) for f in files], chunksize=1)))
    todo = [(f"ramp_{r:g}", "ramp_fixed", r) for r in cfg["ramps_deg"]] + [(f"free_{r:g}", "ramp_free", r) for r in cfg["free_starts_deg"]]
    res, jobs, keys = {}, [], []
    for f in files:
        A = fts[f].get("A")
        x = pts[f]
        res[f] = dict(vid[f]["stages"])
        res[f]["estimate_args"] = {"kp_shape": list(x["kp"].shape), "n_frames_used": int(x["use"].sum()), "joint_ids": [int(j) for j in cfg["joint_ids"]], "size_wh": list(x["size"])}
        if A is None:
            res[f]["free_template"] = {"failed": fts[f].get("failed")}
            continue
        jid = [int(j) for j in A["joint_ids"]]
        T = np.asarray(A["template"], float)
        ba = A.get("ba") or {}
        res[f]["free_template"] = {"f": float(A["f"]), "f_grid": float(A["f_grid"]), "at_grid_edge": bool(A["at_grid_edge"]), "profile_f": [float(v) for v in A["profile_f"]],
                                   "profile_cost": [float(v) for v in A["profile_cost"]], "cost": float(A["cost"]), "ba_rms_px": float(ba.get("rms_px", np.nan)),
                                   "ba_success": bool(ba.get("success", False)), "ba_iterations": int(ba.get("nfev", -1)),
                                   "template_xy": {str(j): [float(v) for v in T[k]] for k, j in enumerate(jid)}}
        for mk, mode, r in todo:
            jobs.append(rof.estimate_args(cfg, x, A) + (mode, r))
            keys.append((f, mk))
    log(f"  추정 {len(jobs)}건")
    ms = {}
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        for (f, mk), o in zip(keys, ex.map(si_ms_job, jobs, chunksize=1)):
            ms.setdefault(f, {})[mk] = o
    for f, d in ms.items():
        free = {k: v for k, v in d.items() if k.startswith("free_") and "failed" not in v}
        res[f]["dk_estimate_ms"] = {**d, "ramp_free_best": min(free, key=lambda k: free[k]["cost"]) if free else None}
    return res


def si_summary(res: dict, cfg: dict, own: dict) -> str:
    vids = {e["file"]: e for e in cfg["videos"]}
    order = sorted(res, key=lambda f: (vids[f]["person"] != "teammate", f))
    who = {"teammate": "팀원", "self": "나"}
    head = "| 함수 | 값 | " + " | ".join(f"{Path(f).stem} {who.get(vids[f]['person'], vids[f]['person'])}" for f in order) + " |"
    sep = "|---|---|" + "---|" * len(order)

    def g(d, path):
        for p in path:
            if not isinstance(d, dict) or p not in d:
                return None
            d = d[p]
        return d

    def fmt(v, nd):
        if v is None:
            return "-"
        if isinstance(v, bool):
            return "예" if v else "아니오"
        if isinstance(v, (list, tuple)):
            return "(" + ", ".join(f"{t:.{nd}f}" if isinstance(t, float) else str(t) for t in v) + ")"
        return f"{v:.{nd}f}" if isinstance(v, float) else str(v)

    def rows(fn, items, base):
        return [f"| {fn} | {lab} | " + " | ".join(fmt(g(res[f], base(f) + [key]), nd) for f in order) + " |" for lab, key, nd in items]
    L = ["# 117 함수별 입력·출력의 구체적인 값 summary", "",
         "**116 과 같은 영상 4개·같은 설정. 함수마다 실제로 들어가고 나온 값을 영상별로 적는다(116 의 보충). 판정 없음.**", "",
         "- 기준 f(1672 px)는 이 파일에만. 좌표는 화면 px (왼쪽 위가 0, 0). 손 좌표 s: 0 = 손목, 1 = 중지 MCP (s < 0 팔 쪽, s >= 1 손가락 쪽).", "", "## 앞 단계", "", head, sep]
    L += rows("extract_frames", [("프레임 수", "n_frames", 0), ("프레임 간격 (s)", "frame_step_s_median", 4), ("마지막 시각 (s)", "last_time_s", 2)], lambda f: ["extract_frames"])
    L += rows("track", [("목표 손을 정한 프레임", "init_frame", 0), ("후보 0개 프레임", "frames_with_0_candidates", 0), ("후보 1개 프레임", "frames_with_1_candidate", 0),
                        ("후보 2개 이상 프레임", "frames_with_2_candidates", 0), ("왼손으로 분류된 프레임", "cls_left_frames", 0), ("오른손으로 분류된 프레임", "cls_right_frames", 0),
                        ("box_conf 중앙값", "box_conf_median", 3), ("상자 중심 x 평균", "box_center_x_mean", 0), ("상자 중심 y 평균", "box_center_y_mean", 0),
                        ("상자 중심 x 최소", "box_center_x_min", 0), ("상자 중심 x 최대", "box_center_x_max", 0), ("상자 중심 y 최소", "box_center_y_min", 0), ("상자 중심 y 최대", "box_center_y_max", 0),
                        ("상자 폭 중앙값", "box_width_median", 0), ("상자 높이 중앙값", "box_height_median", 0)], lambda f: ["track"])
    L += rows("choose_reference_pose", [("정지 구간 수", "n_pauses", 0), ("첫 정지 구간 시작 프레임", "first_pause_start", 0), ("첫 정지 구간 끝 프레임", "first_pause_end", 0),
                                        ("첫 정지 구간의 정지 프레임 수", "first_pause_n", 0), ("기준 프레임", "ref", 0), ("기준 상자 (x0, y0, x1, y1)", "ref_box", 0)], lambda f: ["choose_reference_pose"])
    L += rows("kt_chain", [("단계 수", "steps", 0), ("마스크 넓이 중앙값 (px²)", "mask_area_px_median", 0), ("찾은 특징점 중앙값", "corners_found_median", 0), ("찾은 특징점 최소", "corners_found_min", 0),
                           ("찾은 특징점 최대", "corners_found_max", 0), ("상한(400)에 닿은 프레임 비율", "frames_at_max_corners", 2), ("특징점 / 마스크 1000 px²", "corners_per_1000px_median", 2),
                           ("전후방 검사 통과 중앙값", "fb_pass_median", 0), ("inlier 중앙값", "inliers_median", 0), ("inlier / 통과 중앙값", "inlier_over_fb_pass_median", 3),
                           ("호모그래피 잔차 중앙값 (px)", "homography_residual_px_median", 3), ("호모그래피 잔차 상위 10% (px)", "homography_residual_px_p90", 3),
                           ("inlier 비율: 팔 쪽 (s < 0)", "inlier_frac_forearm_s_lt_0", 3), ("inlier 비율: 손등 (0 <= s < 1)", "inlier_frac_palm_s_0_1", 3),
                           ("inlier 비율: 손가락 (s >= 1)", "inlier_frac_fingers_s_ge_1", 3), ("inlier s 중앙값", "inlier_s_median", 2), ("inlier s 하위 10%", "inlier_s_p10", 2),
                           ("inlier s 상위 10%", "inlier_s_p90", 2), ("inlier 의 |t| 중앙값", "inlier_abs_t_median", 2)], lambda f: ["kt_chain"])
    L += rows("chain_points", [("유효 프레임", "n_valid", 0), ("가장자리 접촉 프레임", "n_edge", 0), ("사용 프레임", "n_use", 0), ("추적점 상자 긴 변 중앙값 (px)", "tracked_box_long_median", 0),
                               ("저장된 점(100)과 최대 차이 (px)", "max_diff_to_saved_px", 3)], lambda f: ["chain_points"])
    L += ["", "## 기준 프레임 관절 좌표 (track_hand 의 입력 P, px)", "", "| 관절 | " + " | ".join(f"{Path(f).stem} {who.get(vids[f]['person'], '')}" for f in order) + " |", "|---|" + "---|" * len(order)]
    for j in range(21):
        L.append(f"| {j} | " + " | ".join(fmt(g(res[f], ["choose_reference_pose", "ref_joints_xy"])[j], 0) for f in order) + " |")
    L += ["", "## free_template", "", head, sep]
    L += rows("free_template", [("f", "f", 0), ("거친 단계 f", "f_grid", 0), ("거친 단계 최솟값이 범위 끝", "at_grid_edge", 0), ("비용", "cost", 3), ("RMS (px)", "ba_rms_px", 2),
                                ("ba_refine 반복 수", "ba_iterations", 0), ("ba_refine 수렴", "ba_success", 0)], lambda f: ["free_template"])
    pf = g(res[order[0]], ["free_template", "profile_f"]) or []
    for k, fv in enumerate(pf):
        L.append(f"| free_template | f = {fv:.0f} 비용 | " + " | ".join(fmt((g(res[f], ["free_template", "profile_cost"]) or [None] * len(pf))[k], 3) for f in order) + " |")
    L += ["", "## free_template 손 모양 T (dk_solve 의 입력, 모양 좌표)", "", "| 관절 | " + " | ".join(f"{Path(f).stem}" for f in order) + " |", "|---|" + "---|" * len(order)]
    for j in cfg["joint_ids"]:
        L.append(f"| {j} | " + " | ".join(fmt(g(res[f], ["free_template", "template_xy", str(j)]), 1) for f in order) + " |")
    for title, pick in (("경사 고정", lambda f: f"ramp_{cfg['ramps_deg'][0]:g}"), ("경사 함께 추정 (비용 최소 시작)", lambda f: g(res[f], ["dk_estimate_ms", "ramp_free_best"]))):
        L += ["", f"## dk_estimate_ms / dk_solve: {title}", "", head, sep]
        L.append("| ms2_job | 실행 | " + " | ".join(str(pick(f)) for f in order) + " |")
        L += rows("dk_estimate_ms", [("f", "f", 0), ("경사 pitch (도)", "pitch_deg", 2), ("roll (도)", "roll_deg", 2), ("책상 기울기 (도)", "desk_tilt_deg", 2), ("회전 범위 (도)", "turn_range_deg", 1), ("비용", "cost", 3), ("RMS (px)", "rms_px", 2), ("고른 시작점", "start", 0),
                                     ("그 시작 f", "f_start", 0), ("따뜻한 재풀이로 바뀐 곡선점 수", "n_warm_improved", 0)], lambda f: ["dk_estimate_ms", pick(f)])
        pf = g(res[order[0]], ["dk_estimate_ms", pick(order[0]), "profile_f"]) or []
        for k, fv in enumerate(pf):
            L.append(f"| dk_estimate_ms | f = {fv:.0f} 고정 비용 | " + " | ".join(fmt((g(res[f], ["dk_estimate_ms", pick(f), "profile_cost"]) or [None] * len(pf))[k], 3) for f in order) + " |")
        tail = []
        for f in order:
            st = g(res[f], ["dk_estimate_ms", pick(f), "starts"]) or []
            tail.append(f"- 시작점별 결과 {Path(f).stem}: " + " ; ".join(f"{s['start']}: {s['f_start']:.0f} → {s['f']:.0f} (비용 {s['cost']:.3f})" for s in st))
        L += rows("dk_solve (고른 해)", [("책상 법선 (카메라 좌표)", "desk_normal_cam", 3),
                                        ("책상 위치 td", "td", 0), ("깊이 td_z", "depth_z", 0), ("손바닥 폭 5-17 (모양 단위)", "palm_width_5_17", 1), ("중지 길이 9-12 (모양 단위)", "middle_len_9_12", 1),
                                        ("깊이 / 손바닥 폭", "depth_over_palm_width", 2), ("θ 최소 (도)", "theta_min_deg", 1), ("θ 최대 (도)", "theta_max_deg", 1),
                                        ("이동 x 최소", "slide_x_min", 0), ("이동 x 최대", "slide_x_max", 0), ("이동 y 최소", "slide_y_min", 0), ("이동 y 최대", "slide_y_max", 0),
                                        ("이동 범위 / 손바닥 폭", "slide_range_over_palm_width", 2), ("기준 관절 (고정)", "anchor_joints", 0), ("up", "up", 0)], lambda f: ["dk_estimate_ms", pick(f), "state"])
        L += [""] + tail
    L += ["", "## 한계", "", "- dk_solve 의 해(책상 자세 등)는 ms2_job 이 내보내지 않아, 같은 절차를 따라 하며 해를 남기는 si_dk_estimate_ms 로 얻었다(f·비용이 116 과 같은지로 확인).",
          "- 모양 단위는 free_template 이 정한 임의 단위(추적점의 화면 크기 px 에 맞춘 것)라 cm 가 아니다.", ""]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------------------------------
# 118: f 를 고정했을 때의 비용을 회전 각도·관절별로 쪼갬 (팀원 영상과 내 영상의 비용 곡선이 어디서 달라지는지)
# ----------------------------------------------------------------------------------------------------
CDX_ROWS = {"MCP": (5, 9, 13, 17), "PIP": (6, 10, 14, 18), "DIP": (7, 11, 15, 19), "TIP": (8, 12, 16, 20)}
CDX_FINGERS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}


def cdx_predict(pr, T2: np.ndarray, jid: list, st: dict) -> tuple:
    """dk_solve 의 해 st 로 줄인 문제 pr 의 사용 프레임마다 모델이 예측한 점 (N, J, 2) 과 관측점 (N, J, 2; 안 보이면 NaN). dk_solve 의 proj 와 같은 식."""
    frames = [pr.frames[i] for i in pr.keep]
    u, w = hf.dk_axes(T2, jid)
    T = np.asarray(st["T"], float)
    Hd = np.c_[T, np.zeros(len(T))] @ hf.dk_hand_rot(u, w, float(st["up"]) * float(st["pitch"]), float(st["roll"])).T
    Rd, td = np.asarray(st["Rd"], float), np.asarray(st["td"], float)
    pred = np.full((len(frames), len(T), 2), np.nan)
    obs = np.full((len(frames), len(T), 2), np.nan)
    for n, (o, v) in enumerate(frames):
        c, s = np.cos(st["theta"][n]), np.sin(st["theta"][n])
        Xd = np.c_[c * Hd[:, 0] - s * Hd[:, 1] + st["xy"][n][0], s * Hd[:, 0] + c * Hd[:, 1] + st["xy"][n][1], Hd[:, 2]]
        X = Xd @ Rd.T + td
        pred[n] = np.c_[float(st["f"]) * X[:, 0] / X[:, 2] + pr.cx, float(st["f"]) * X[:, 1] / X[:, 2] + pr.cy]
        obs[n][v] = o[v]
    return pred, obs


def cdx_job(args) -> dict:
    """한 영상: 저장된 추적점 -> od_bridge -> free_template -> 경사 고정 f 곡선(dk_estimate_ms 의 1, 2 단계: f 8개 고정 풀이 + 따뜻한 재풀이) -> f 마다 점별 Huber 손실, 손 길이·폭의 예측/관측 비."""
    file, x, cfg = args
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import cv2
    cv2.setNumThreads(1)
    ecfg, ids = cfg["pipeline"]["estimate"], cfg["joint_ids"]
    cd = cfg["cost_decompose"]
    grid = hf.f_grid(ecfg, max(*x["size"]))
    used = np.flatnonzero(x["use"])
    pb = hf.od_bridge(x["kp"], np.isfinite(x["kp"]).all(2), used, x["box"], tuple(x["size"]), ecfg, ids)
    A = hf.free_template(pb, grid)
    pr, rid = hf._restrict(pb, max_frames=120)
    jid = [int(j) for j in A["joint_ids"]]
    T2 = np.asarray(A["template"], float)
    g = grid[np.unique(np.linspace(0, len(grid) - 1, 8).round().astype(int))]
    p0 = np.radians(float(cd["ramp_deg"]))
    fb = (grid[0], grid[-1])
    prof = [hf.dk_solve(pr, float(f), T2, jid, p0, 0.0, False, False, fb) for f in g]
    for d, idx in ((-1, range(1, len(g))), (1, range(len(g) - 2, -1, -1))):
        for i in idx:
            s = prof[i + d]["state"]
            y = hf.dk_solve(pr, float(g[i]), T2, jid, s["pitch"], s["roll"], False, False, fb, init=dict(s, f=float(g[i])))
            if y["cost"] < prof[i]["cost"]:
                prof[i] = y
    hp = float(np.median(pr.huber[pr.keep]))
    fr = used[np.asarray(pr.keep, int)]
    ang = x["hand_angle"] - x["hand_angle"][x["ref"]]
    tips, mcps = [jid.index(j) for j in CDX_ROWS["TIP"]], [jid.index(j) for j in CDX_ROWS["MCP"]]
    a5, a17 = jid.index(5), jid.index(17)
    out = {"frames": [int(i) for i in fr], "angle_deg": [float(a) for a in ang[fr]], "joint_ids": jid, "huber_px": hp, "f": [float(f) for f in g], "per_f": []}
    for k in range(len(g)):
        pred, obs = cdx_predict(pr, T2, jid, prof[k]["state"])
        e = np.linalg.norm(pred - obs, axis=2)
        loss = np.where(e <= hp, e * e, 2 * hp * e - hp * hp)
        ln = lambda P: np.linalg.norm(np.nanmean(P[:, tips], axis=1) - np.nanmean(P[:, mcps], axis=1), axis=1)      # noqa: E731
        wd = lambda P: np.linalg.norm(P[:, a5] - P[:, a17], axis=1)      # noqa: E731
        out["per_f"].append({"f": float(g[k]), "cost": float(prof[k]["cost"]), "cost_recomputed": float(np.nanmean(loss)), "desk_tilt_deg": float(prof[k]["desk_tilt_deg"]),
                             "loss": [[None if not np.isfinite(v) else round(float(v), 3) for v in row] for row in loss],
                             "length_pred_over_obs": [float(v) for v in ln(pred) / ln(obs)], "width_pred_over_obs": [float(v) for v in wd(pred) / wd(obs)]})
    return out


def cdx_run(cfg: dict, own: dict, log) -> dict:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    cd = cfg["cost_decompose"]
    jobs = [(e["file"], load_points(cd["points_run"], e["file"]), cfg) for e in cfg["videos"]]
    log(f"  f 곡선 {len(jobs)}건")
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        return {j[0]: o for j, o in zip(jobs, ex.map(cdx_job, jobs, chunksize=1))}


def cdx_tables(d: dict, cd: dict) -> dict:
    """한 영상의 결과 -> 비교 쌍(낮은 f, 높은 f)마다: 전체 비용 차이, 회전 각도 구간·관절 줄·손가락별 기여(합 = 전체 차이)와 구간 안 평균 차이, 길이·폭 예측/관측 비의 구간 평균."""
    fs = np.asarray(d["f"], float)
    ang = np.asarray(d["angle_deg"], float)
    jid = d["joint_ids"]
    L = {k: np.asarray([[np.nan if v is None else v for v in row] for row in p["loss"]], float) for k, p in enumerate(d["per_f"])}
    bins = [float(b) for b in cd["angle_bins_deg"]]
    bl = [f"{bins[i]:g}~{bins[i + 1]:g}" for i in range(len(bins) - 1)]
    bi = np.clip(np.digitize(ang, bins) - 1, 0, len(bl) - 1)
    out = {"n_frames_by_angle": {bl[i]: int((bi == i).sum()) for i in range(len(bl))}, "angle_min": float(ang.min()), "angle_max": float(ang.max()), "pairs": {}}
    for lo, hi in cd["pairs"]:
        a, b = int(np.argmin(np.abs(fs - lo))), int(np.argmin(np.abs(fs - hi)))
        D = L[a] - L[b]
        n = np.isfinite(D).sum()
        t = {"f_low": float(fs[a]), "f_high": float(fs[b]), "cost_low": float(np.nanmean(L[a])), "cost_high": float(np.nanmean(L[b])), "delta": float(np.nansum(D) / n),
             "by_angle": {}, "by_row": {}, "by_finger": {}, "length_ratio_low": {}, "length_ratio_high": {}, "width_ratio_low": {}, "width_ratio_high": {}}
        for i, lab in enumerate(bl):
            m = bi == i
            t["by_angle"][lab] = {"contribution": float(np.nansum(D[m]) / n), "mean_delta": float(np.nanmean(D[m])) if m.any() else None,
                                  "mean_loss_low": float(np.nanmean(L[a][m])) if m.any() else None, "mean_loss_high": float(np.nanmean(L[b][m])) if m.any() else None}
            for key, src, k in (("length_ratio_low", "length_pred_over_obs", a), ("length_ratio_high", "length_pred_over_obs", b), ("width_ratio_low", "width_pred_over_obs", a), ("width_ratio_high", "width_pred_over_obs", b)):
                v = np.asarray(d["per_f"][k][src], float)[m]
                t[key][lab] = float(np.nanmean(v)) if m.any() else None
        for name, grp, key in [(k, v, "by_row") for k, v in CDX_ROWS.items()] + [(k, v, "by_finger") for k, v in CDX_FINGERS.items()]:
            c = [jid.index(j) for j in grp if j in jid]
            t[key][name] = {"contribution": float(np.nansum(D[:, c]) / n), "mean_delta": float(np.nanmean(D[:, c]))}
        out["pairs"][f"{lo:g}_vs_{hi:g}"] = t
    return out


def cdx_summary(res: dict, cfg: dict, own: dict) -> str:
    cd = cfg["cost_decompose"]
    vids = {e["file"]: e for e in cfg["videos"]}
    order = sorted(res, key=lambda f: (vids[f]["person"] != "teammate", f))
    who = {"teammate": "팀원", "self": "나"}
    names = " | ".join(f"{Path(f).stem} {who.get(vids[f]['person'], '')}" for f in order)
    sep = "---|" * len(order)
    T = {f: cdx_tables(res[f], cd) for f in order}
    fm = lambda v, nd=2: "-" if v is None else f"{v:+.{nd}f}"      # noqa: E731
    L = ["# 118 비용 곡선 쪼개기 summary", "",
         f"**저장된 추적점(100)으로 경사 {cd['ramp_deg']}° 고정 f 곡선(f 8개 고정 풀이 + 따뜻한 재풀이)을 다시 만들고, f 마다 점별 Huber 손실(px²)을 회전 각도·관절별로 나눈다. 판정 없음.**", "",
         "- 각도 = 검출 손 방향(손목 -> 중지 MCP), 기준 프레임 대비. 기여 = 그 구간 점들의 (낮은 f 손실 - 높은 f 손실) 합 / 전체 점 수 (구간 기여의 합 = 전체 비용 차이).",
         "- 길이 = 손끝 4개 평균 - MCP 4개 평균 사이 거리, 폭 = 관절 5 - 17 거리 (화면 px). 예측/관측 - 1 을 % 로.", "",
         "## f 마다 비용 (다시 계산한 값, px²)", "", f"| f | {names} |", "|---|" + sep]
    for k, fv in enumerate(res[order[0]]["f"]):
        L.append(f"| {fv:.0f} | " + " | ".join(f"{res[f]['per_f'][k]['cost_recomputed']:.3f}" for f in order) + " |")
    L += ["", "## 각도 구간별 프레임 수 (추정에 쓴 120 프레임)", "", f"| 각도 (도) | {names} |", "|---|" + sep]
    for lab in T[order[0]]["n_frames_by_angle"]:
        L.append(f"| {lab} | " + " | ".join(str(T[f]["n_frames_by_angle"][lab]) for f in order) + " |")
    L.append("| 최소 ~ 최대 | " + " | ".join(f"{T[f]['angle_min']:.0f} ~ {T[f]['angle_max']:.0f}" for f in order) + " |")
    for pk in T[order[0]]["pairs"]:
        t0 = T[order[0]]["pairs"][pk]
        L += ["", f"## f {t0['f_low']:.0f} 대 {t0['f_high']:.0f}", "", f"| 값 | {names} |", "|---|" + sep,
              f"| 비용 (f {t0['f_low']:.0f}) | " + " | ".join(f"{T[f]['pairs'][pk]['cost_low']:.3f}" for f in order) + " |",
              f"| 비용 (f {t0['f_high']:.0f}) | " + " | ".join(f"{T[f]['pairs'][pk]['cost_high']:.3f}" for f in order) + " |",
              "| 차이 (낮은 f - 높은 f) | " + " | ".join(fm(T[f]["pairs"][pk]["delta"], 3) for f in order) + " |"]
        for key, title in (("by_angle", "각도"), ("by_row", "관절 줄"), ("by_finger", "손가락")):
            L += ["", f"### {title}별 기여 (합 = 차이) / 구간 안 평균 차이", "", f"| {title} | {names} |", "|---|" + sep]
            for lab in t0[key]:
                L.append(f"| {lab} | " + " | ".join(f"{fm(T[f]['pairs'][pk][key][lab]['contribution'], 3)} / {fm(T[f]['pairs'][pk][key][lab]['mean_delta'])}" for f in order) + " |")
        L += ["", "### 각도별 평균 손실 (낮은 f / 높은 f, px²)", "", f"| 각도 | {names} |", "|---|" + sep]
        for lab in t0["by_angle"]:
            c = lambda f: T[f]["pairs"][pk]["by_angle"][lab]      # noqa: E731
            L.append(f"| {lab} | " + " | ".join("-" if c(f)["mean_loss_low"] is None else f"{c(f)['mean_loss_low']:.2f} / {c(f)['mean_loss_high']:.2f}" for f in order) + " |")
        for key, title in (("length_ratio", "손 길이"), ("width_ratio", "손 폭")):
            L += ["", f"### {title} 예측/관측 - 1 (%, 낮은 f / 높은 f)", "", f"| 각도 | {names} |", "|---|" + sep]
            for lab in t0["by_angle"]:
                c = lambda f, s: T[f]["pairs"][pk][f"{key}_{s}"][lab]      # noqa: E731
                L.append(f"| {lab} | " + " | ".join("-" if c(f, "low") is None else f"{100 * (c(f, 'low') - 1):+.2f} / {100 * (c(f, 'high') - 1):+.2f}" for f in order) + " |")
    L += ["", "## 한계", "", "- 영상 4개. 각도 구간마다 프레임 수가 달라 구간 평균의 흩어짐이 다르다.", "- f 고정 풀이가 비용이 큰 해에서 멈춘 f(7384 의 1983, 3072)는 비교 쌍에서 뺐다.", ""]
    return "\n".join(L) + "\n"


def cdx_results_yaml(run_dir: Path, own: dict) -> Path:
    """118 의 값만: 영상별 f 곡선 비용과 cdx_tables (점별 손실 행렬은 results.json 에만)."""
    r = json.loads((run_dir / "results.json").read_text())["real"]
    m = json.loads((run_dir / "metadata.json").read_text())
    rv = lambda x: x if not isinstance(x, float) else (round(x, 4) if np.isfinite(x) else None)      # noqa: E731
    deep = lambda d: {k: (deep(v) if isinstance(v, dict) else rv(v)) for k, v in d.items()}      # noqa: E731
    vid = {f: {"f": [rv(v) for v in d["f"]], "cost": [rv(p["cost"]) for p in d["per_f"]], "cost_recomputed": [rv(p["cost_recomputed"]) for p in d["per_f"]],
               "huber_px": rv(d["huber_px"]), **deep(cdx_tables(d, own["cost_decompose"]))} for f, d in r.items()}
    out = {"experiment": own["experiment"]["name"], "run": str(run_dir.resolve().relative_to(hf.PROJECT_ROOT)), "code_commit": m["git"]["commit"][:7],
           "git_dirty_at_run": m["git"]["git_dirty"], "videos": vid}
    p = hf.resolve(own["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


# ----------------------------------------------------------------------------------------------------
# 119: 렌즈 모델 (주점, 왜곡)을 넣었을 때의 f - 저장된 추적점, 같은 추정
# ----------------------------------------------------------------------------------------------------
def lm_ecfg(base: dict, var: dict, long_side: float) -> dict:
    """추정 설정 base 에 렌즈 변형 var 를 넣은 것. var = {principal_point: "center" | [cx, cy], distortion: null | {k1, k2, p1, p2, calib_f_px}}.
    handfocal.undistort 는 왜곡 계수를 K_d = [[L, 0, cx], [0, L, cy]] (L = 긴 변) 기준으로 받는다. 체커보드 보정의 계수는 보정 f 기준(정규화 좌표 r = r_px / f)이므로
    k1' = k1 (L / f)^2, k2' = k2 (L / f)^4, p' = p (L / f) 로 바꿔 넣는다 (같은 px 위치에서 같은 왜곡량)."""
    e = copy.deepcopy(base)
    e["principal_point"] = var.get("principal_point", "center")
    d = var.get("distortion")
    if d:
        s = float(long_side) / float(d["calib_f_px"])
        e["distortion"] = [float(d["k1"]) * s ** 2, float(d["k2"]) * s ** 4, float(d.get("p1", 0.0)) * s, float(d.get("p2", 0.0)) * s]
    else:
        e["distortion"] = None
    return e


def lm_run(cfg: dict, own: dict, log) -> dict:
    """119: 영상 x 렌즈 변형마다 free_template(tl_template_job) -> 경사 고정·함께 추정(ms2_job). 추적점은 저장된 점 그대로(왜곡 보정은 free_problem 안의 undistort 가 한다)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    lm = cfg["lens_model"]
    pts = {e["file"]: load_points(lm["points"][e["file"]], e["file"]) for e in cfg["videos"]}
    res, items = {}, []
    for f, x in pts.items():
        W, H = x["size"]
        res[f] = {"n_use": int(x["use"].sum()), "variants": {}}
        for vn, var in lm["variants"].items():
            ec = lm_ecfg(cfg["pipeline"]["estimate"], var, max(W, H))
            kp = x["kp"][x["use"]][:, cfg["joint_ids"]]
            und = hf.undistort(kp, ec, W, H)
            sh = np.linalg.norm(und - kp, axis=2)
            res[f]["variants"][vn] = {"principal_point": list(hf.principal_point(ec, W, H)), "distortion_L": ec["distortion"],
                                      "undistort_shift_px_median": float(np.nanmedian(sh)), "undistort_shift_px_p90": float(np.nanpercentile(sh, 90)), "undistort_shift_px_max": float(np.nanmax(sh))}
            items.append((f, vn, (x["kp"], np.isfinite(x["kp"]).all(2), np.flatnonzero(x["use"]), x["box"], tuple(x["size"]), ec, cfg["joint_ids"])))
    log(f"  free_template {len(items)}건")
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        fts = list(ex.map(hf.tl_template_job, [a for _, _, a in items], chunksize=1))
    todo = [(f"ramp_{r:g}", "ramp_fixed", r) for r in cfg["ramps_deg"]] + [(f"free_{r:g}", "ramp_free", r) for r in cfg["free_starts_deg"]]
    jobs, keys = [], []
    keep = ("f", "pitch_deg", "roll_deg", "desk_tilt_deg", "turn_range_deg", "rms_px", "cost", "boundary", "start")
    for (f, vn, a), ft in zip(items, fts):
        A = ft.get("A")
        grid = hf.f_grid(a[5], max(*a[4]))
        res[f]["variants"][vn]["free_template"] = {"failed": ft.get("failed")} if A is None else {"f": float(A["f"]), "boundary": hf.od_boundary(float(A["f"]), grid), "rms_px": float((A.get("ba") or {}).get("rms_px", np.nan))}
        if A is None:
            continue
        for mk, mode, r in todo:
            jobs.append(a + (A, mode, r))
            keys.append((f, vn, mk))
    log(f"  추정 {len(jobs)}건")
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        for (f, vn, mk), o in zip(keys, ex.map(hf.ms2_job, jobs, chunksize=1)):
            res[f]["variants"][vn][mk] = {kk: o[kk] for kk in keep if kk in o} if "failed" not in o else o
    for f in res:
        for vn, d in res[f]["variants"].items():
            free = {k: v for k, v in d.items() if k.startswith("free_") and k != "free_template" and isinstance(v, dict) and "cost" in v}
            if free:
                bk = min(free, key=lambda k: free[k]["cost"])
                d["ramp_free"] = {**free[bk], "best_start": bk}
    return res


def lm_summary(res: dict, cfg: dict, own: dict) -> str:
    lm = cfg["lens_model"]
    vids = {e["file"]: e for e in cfg["videos"]}
    order = sorted(res, key=lambda f: (vids[f]["person"] != "teammate", f))
    vn = list(lm["variants"])
    who = {"teammate": "팀원", "self": "나"}
    rf = lambda f: float(vids[f].get("ref_f_px") or cfg["ref_f"]["value"])      # noqa: E731

    def cell(f, v, key):
        d = res[f]["variants"][v].get(key) or {}
        if "f" not in d:
            return "-"
        s = f"{d['f']:.0f} ({100 * (d['f'] / rf(f) - 1):+.1f}%)" + ("*" if d.get("boundary") else "")
        return s + (f", {d['pitch_deg']:.1f}°" if key == "ramp_free" else "")
    L = ["# 119 렌즈 모델 summary", "",
         "**같은 폰(새 폰)으로 찍은 영상의 저장된 추적점에, 체커보드 보정의 주점과 왜곡 계수를 넣어 같은 추정(free_template -> 경사 고정 / 함께 추정)을 다시 돌린다. 판정 없음.**", "",
         "- 기준 f 는 이 파일에만. * = f 탐색 범위 끝. 함께 추정 칸의 각도 = 추정된 경사.",
         "- 변형: " + " ; ".join(f"{k} = 주점 {v.get('principal_point', 'center')}, 왜곡 {'없음' if not v.get('distortion') else {kk: v['distortion'][kk] for kk in ('k1', 'k2')}}" for k, v in lm["variants"].items()), "",
         "## 왜곡 보정으로 추적점이 옮겨진 거리 (px, 사용 프레임의 j16 관절)", "", "| 영상 | 묶음 | " + " | ".join(f"{v} 중앙값 / 상위 10% / 최대" for v in vn) + " |", "|---|---|" + "---|" * len(vn)]
    for f in order:
        L.append(f"| {f} | {who.get(vids[f]['person'], vids[f]['person'])} | " + " | ".join(
            f"{res[f]['variants'][v]['undistort_shift_px_median']:.2f} / {res[f]['variants'][v]['undistort_shift_px_p90']:.2f} / {res[f]['variants'][v]['undistort_shift_px_max']:.2f}" for v in vn) + " |")
    for key, title in [(f"ramp_{r:g}", f"경사 {r:g}° 고정") for r in cfg["ramps_deg"]] + [("ramp_free", "경사 함께 추정"), ("free_template", "free_template")]:
        L += ["", f"## {title}: f (오차)", "", "| 영상 | 묶음 | " + " | ".join(vn) + " |", "|---|---|" + "---|" * len(vn)]
        for f in order:
            L.append(f"| {f} | {who.get(vids[f]['person'], vids[f]['person'])} | " + " | ".join(cell(f, v, key) for v in vn) + " |")
        if key != "free_template":
            for grp in ("teammate", "self"):
                fs = [f for f in order if vids[f]["person"] == grp]
                if not fs:
                    continue
                row = []
                for v in vn:
                    e = [100 * ((res[f]["variants"][v].get(key) or {}).get("f", np.nan) / rf(f) - 1) for f in fs]
                    row.append(f"중앙값 {np.nanmedian(e):+.1f}%, 절댓값 중앙값 {np.nanmedian(np.abs(e)):.1f}%, 범위 {np.nanmin(e):+.1f} ~ {np.nanmax(e):+.1f}%")
                L.append(f"| {who[grp]} {len(fs)}개 | | " + " | ".join(row) + " |")
    key = f"ramp_{cfg['ramps_deg'][0]:g}"
    L += ["", f"## 경사 {cfg['ramps_deg'][0]:g}° 고정: 책상 기울기 (도) / RMS (px)", "", "| 영상 | 묶음 | " + " | ".join(vn) + " |", "|---|---|" + "---|" * len(vn)]
    for f in order:
        g = lambda v: res[f]["variants"][v].get(key) or {}      # noqa: E731
        L.append(f"| {f} | {who.get(vids[f]['person'], '')} | " + " | ".join("-" if "f" not in g(v) else f"{g(v)['desk_tilt_deg']:.1f} / {g(v)['rms_px']:.2f}" for v in vn) + " |")
    L += ["", "## 한계", "", "- 왜곡 계수를 긴 변 기준으로 바꿀 때 체커보드 보정의 f(기준 f 와 같은 값)를 썼다. 진단용이며, 실제 사용에서는 왜곡 계수가 f 없이 주어지지 않는다.",
          "- 추적(광류 사슬)은 왜곡이 있는 원래 영상에서 한 그대로다. 보정은 추적점 좌표에만 적용했다.", "- 영상 8개(사람당 4개), 새 폰만.", ""]
    return "\n".join(L) + "\n"


def lm_results_yaml(run_dir: Path, own: dict) -> Path:
    """119 의 값만 (기준 f, 오차 없음)."""
    r = json.loads((run_dir / "results.json").read_text())["real"]
    m = json.loads((run_dir / "metadata.json").read_text())
    rv = lambda x: x if not isinstance(x, float) else (round(x, 4) if np.isfinite(x) else None)      # noqa: E731
    deep = lambda d: {k: (deep(v) if isinstance(v, dict) else [rv(t) for t in v] if isinstance(v, list) else rv(v)) for k, v in d.items()}      # noqa: E731
    out = {"experiment": own["experiment"]["name"], "run": str(run_dir.resolve().relative_to(hf.PROJECT_ROOT)), "code_commit": m["git"]["commit"][:7],
           "git_dirty_at_run": m["git"]["git_dirty"], "videos": {f: deep(d) for f, d in r.items()}}
    p = hf.resolve(own["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


# ----------------------------------------------------------------------------------------------------
# 120: 같은 자세로 돌아왔을 때 어긋남이 같은지 (반복성) - 책상 모델의 잔차를 같은 자세의 두 시점끼리 비교
# ----------------------------------------------------------------------------------------------------
def rp_frame_pred(c: dict, th: float, xy) -> np.ndarray:
    """책상 모델의 한 프레임 예측점 (J, 2). c = rp_consts 의 값(영상 하나에 고정), th·xy = 그 프레임의 회전·이동. dk_solve 의 proj 와 같은 식."""
    co, si = np.cos(th), np.sin(th)
    Hd = c["Hd"]
    Xd = np.c_[co * Hd[:, 0] - si * Hd[:, 1] + xy[0], si * Hd[:, 0] + co * Hd[:, 1] + xy[1], Hd[:, 2]]
    X = Xd @ c["Rd"].T + c["td"]
    return np.c_[c["f"] * X[:, 0] / X[:, 2] + c["cx"], c["f"] * X[:, 1] / X[:, 2] + c["cy"]]


def rp_consts(pr, T2: np.ndarray, jid: list, st: dict) -> dict:
    u, w = hf.dk_axes(T2, jid)
    T = np.asarray(st["T"], float)
    return {"Hd": np.c_[T, np.zeros(len(T))] @ hf.dk_hand_rot(u, w, float(st["up"]) * float(st["pitch"]), float(st["roll"])).T,
            "Rd": np.asarray(st["Rd"], float), "td": np.asarray(st["td"], float), "f": float(st["f"]), "cx": float(pr.cx), "cy": float(pr.cy), "T": T}


def rp_all_frames(pb, pr, c: dict, st: dict, hp: float) -> tuple:
    """추정에 쓴 프레임(pr.keep)의 해에서 출발해, 모든 사용 프레임(pb.keep)의 회전·이동만 프레임마다 다시 푼다(책상 자세·손 모양·f 는 고정). 반환: th (n,), xy (n, 2), 잔차 pred - obs (n, J, 2; 안 보이면 NaN)."""
    from scipy.optimize import least_squares
    keep_all, keep_fit = np.asarray(pb.keep, int), np.asarray(pr.keep, int)
    th0 = np.interp(keep_all, keep_fit, np.asarray(st["theta"], float))
    xy0 = np.c_[np.interp(keep_all, keep_fit, np.asarray(st["xy"], float)[:, 0]), np.interp(keep_all, keep_fit, np.asarray(st["xy"], float)[:, 1])]
    J = len(c["T"])
    th, xy, res = np.zeros(len(keep_all)), np.zeros((len(keep_all), 2)), np.full((len(keep_all), J, 2), np.nan)
    for n, i in enumerate(keep_all):
        o, v = pr.frames[i]
        fun = lambda p: (rp_frame_pred(c, p[0], p[1:])[v] - o[v]).ravel()      # noqa: E731
        s = least_squares(fun, np.r_[th0[n], xy0[n]], loss="huber", f_scale=hp, method="trf")
        th[n], xy[n] = s.x[0], s.x[1:]
        res[n][v] = rp_frame_pred(c, s.x[0], s.x[1:])[v] - o[v]
    return th, xy, res


def rp_pairs(t: np.ndarray, th_deg: np.ndarray, xy: np.ndarray, res: np.ndarray, ang: np.ndarray, pw: float, lr: np.ndarray, wr: np.ndarray, rc: dict) -> dict:
    """같은 자세(회전 차이 <= pose_tol_deg, 이동 차이 <= pose_tol_slide x 손바닥 폭)의 프레임 쌍마다 잔차 차이의 RMS (px). 가까운 쌍(시간 차 <= near_s) = 잡음 바닥, 먼 쌍(>= far_s) = 다시 돌아온 자세."""
    n = len(t)
    i, j = np.triu_indices(n, 1)
    ok = (np.abs(th_deg[i] - th_deg[j]) <= float(rc["pose_tol_deg"])) & (np.linalg.norm(xy[i] - xy[j], axis=1) <= float(rc["pose_tol_slide"]) * pw)
    i, j = i[ok], j[ok]
    dt = t[j] - t[i]
    D = np.sqrt(np.nanmean(((res[i] - res[j]) ** 2).sum(2), axis=1))
    dl, dw = 100 * (lr[j] / lr[i] - 1), 100 * (wr[j] / wr[i] - 1)                # 관측/예측 비의 변화 (%, 나중 - 먼저)
    near, far = dt <= float(rc["near_s"]), dt >= float(rc["far_s"])
    exc = np.zeros(len(i))                                             # 두 프레임 사이에 손이 돈 최대 각도 (먼저 프레임 대비, 도)
    for a in np.unique(i[far]):
        cm = np.maximum.accumulate(np.abs(th_deg[a:] - th_deg[a]))
        m = far & (i == a)
        exc[m] = cm[j[m] - a]
    ret, still = far & (exc >= float(rc["excursion_deg"])), far & (exc < float(rc["still_deg"]))
    st = lambda m: {"n_pairs": int(m.sum()), "D_median_px": sv_pct(D[m], 50), "D_p90_px": sv_pct(D[m], 90)}      # noqa: E731
    out = {"n_frames": int(n), "near": st(near), "far": st(far), "far_by_gap_s": {}, "far_by_angle_deg": {},
           "far_length_change_pct": {"median": sv_pct(dl[far], 50), "p25": sv_pct(dl[far], 25), "p75": sv_pct(dl[far], 75), "abs_median": sv_pct(np.abs(dl[far]), 50)},
           "far_width_change_pct": {"median": sv_pct(dw[far], 50), "p25": sv_pct(dw[far], 25), "p75": sv_pct(dw[far], 75), "abs_median": sv_pct(np.abs(dw[far]), 50)},
           "near_length_change_abs_median_pct": sv_pct(np.abs(dl[near]), 50), "near_width_change_abs_median_pct": sv_pct(np.abs(dw[near]), 50),
           "far_frames_covered": int(len(np.unique(np.r_[i[far], j[far]]))),
           "return": {**st(ret), "frames_covered": int(len(np.unique(np.r_[i[ret], j[ret]]))), "gap_s_median": sv_pct(dt[ret], 50),
                      "length_change_pct_median": sv_pct(dl[ret], 50), "length_change_pct_abs_median": sv_pct(np.abs(dl[ret]), 50),
                      "width_change_pct_median": sv_pct(dw[ret], 50), "width_change_pct_abs_median": sv_pct(np.abs(dw[ret]), 50), "by_angle_deg": {}},
           "still": {**st(still), "gap_s_median": sv_pct(dt[still], 50), "length_change_pct_abs_median": sv_pct(np.abs(dl[still]), 50), "width_change_pct_abs_median": sv_pct(np.abs(dw[still]), 50)}}
    g = [float(v) for v in rc["gap_bins_s"]]
    for a, b in zip(g[:-1], g[1:]):
        out["far_by_gap_s"][f"{a:g}~{b:g}"] = st(far & (dt >= a) & (dt < b))
    bins = [float(v) for v in rc["angle_bins_deg"]]
    am = (ang[i] + ang[j]) / 2
    for a, b in zip(bins[:-1], bins[1:]):
        out["far_by_angle_deg"][f"{a:g}~{b:g}"] = st(far & (am >= a) & (am < b))
        out["return"]["by_angle_deg"][f"{a:g}~{b:g}"] = st(ret & (am >= a) & (am < b))
    return out


def rp_job(args) -> dict:
    """한 영상: 저장된 추적점 -> od_bridge -> free_template -> 경사 고정 f 곡선(f 8개 + 따뜻한 재풀이) -> f 를 평가 값(기준 f, 격자의 낮은 f)에 고정한 해 -> 모든 사용 프레임의 자세와 잔차 -> 같은 자세 쌍의 잔차 차이."""
    file, x, cfg, f_ref, run_root = args
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import cv2
    cv2.setNumThreads(1)
    ecfg, ids = cfg["pipeline"]["estimate"], cfg["joint_ids"]
    rc = cfg["repeatability"]
    grid = hf.f_grid(ecfg, max(*x["size"]))
    used = np.flatnonzero(x["use"])
    pb = hf.od_bridge(x["kp"], np.isfinite(x["kp"]).all(2), used, x["box"], tuple(x["size"]), ecfg, ids)
    A = hf.free_template(pb, grid)
    pr, rid = hf._restrict(pb, max_frames=120)
    jid = [int(j) for j in A["joint_ids"]]
    T2 = np.asarray(A["template"], float)
    g = grid[np.unique(np.linspace(0, len(grid) - 1, 8).round().astype(int))]
    p0 = np.radians(float(rc["ramp_deg"]))
    fb = (grid[0], grid[-1])
    prof = [hf.dk_solve(pr, float(f), T2, jid, p0, 0.0, False, False, fb) for f in g]
    for d, idx in ((-1, range(1, len(g))), (1, range(len(g) - 2, -1, -1))):
        for k in idx:
            s = prof[k + d]["state"]
            y = hf.dk_solve(pr, float(g[k]), T2, jid, s["pitch"], s["roll"], False, False, fb, init=dict(s, f=float(g[k])))
            if y["cost"] < prof[k]["cost"]:
                prof[k] = y
    hp = float(np.median(pr.huber[pr.keep]))
    keep_all = np.asarray(pb.keep, int)
    fr = used[keep_all]
    t = x["times"][fr]
    ang = (x["hand_angle"] - x["hand_angle"][x["ref"]])[fr]
    tips, mcps = [jid.index(j) for j in CDX_ROWS["TIP"]], [jid.index(j) for j in CDX_ROWS["MCP"]]
    a5, a17 = jid.index(5), jid.index(17)
    obs = np.stack([np.where(pr.frames[i][1][:, None], pr.frames[i][0], np.nan) for i in keep_all])
    ln = lambda P: np.linalg.norm(np.nanmean(P[:, tips], axis=1) - np.nanmean(P[:, mcps], axis=1), axis=1)      # noqa: E731
    wd = lambda P: np.linalg.norm(P[:, a5] - P[:, a17], axis=1)      # noqa: E731
    out = {"n_use": int(len(fr)), "duration_s": float(t[-1] - t[0]), "huber_px": hp, "eval": {}}
    save = {"frames": fr, "times": t, "angle_deg": ang}
    for name, fe in (("ref_f", float(f_ref)), ("low_f", float(g[int(np.argmin(np.abs(g - float(rc["low_f"]))))]))):
        k = int(np.argmin(np.abs(np.log(g) - np.log(fe))))
        s = prof[k]["state"]
        B = hf.dk_solve(pr, fe, T2, jid, s["pitch"], s["roll"], False, False, fb, init=dict(s, f=fe))
        st = B["state"]
        c = rp_consts(pr, T2, jid, st)
        th, xy, res = rp_all_frames(pb, pr, c, st, hp)
        pred = obs + res
        pw = float(np.linalg.norm(c["T"][a5] - c["T"][a17]))
        e = np.linalg.norm(res, axis=2)
        d = rp_pairs(t, np.degrees(th), xy, res, ang, pw, ln(obs) / ln(pred), wd(obs) / wd(pred), rc)
        d.update({"cost_fit_frames": float(B["cost"]), "rms_all_frames_px": float(np.sqrt(np.nanmean(e ** 2))), "theta_range_deg": float(np.degrees(th.max() - th.min()))})
        out["eval"][name] = d
        save.update({f"{name}_theta_deg": np.degrees(th), f"{name}_xy": xy, f"{name}_res": res.astype(np.float32)})
    dd = Path(run_root) / "frames"
    dd.mkdir(exist_ok=True)
    np.savez_compressed(dd / f"{Path(file).stem}.npz", **save)
    return out


def rp_run(cfg: dict, own: dict, log, run_root) -> dict:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    rc = cfg["repeatability"]
    jobs = [(e["file"], load_points(rc["points"][e["file"]], e["file"]), cfg, float(e.get("ref_f_px") or cfg["ref_f"]["value"]), str(run_root)) for e in cfg["videos"]]
    log(f"  영상 {len(jobs)}건")
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        return {j[0]: o for j, o in zip(jobs, ex.map(rp_job, jobs, chunksize=1))}


def rp_summary(res: dict, cfg: dict, own: dict) -> str:
    rc = cfg["repeatability"]
    vids = {e["file"]: e for e in cfg["videos"]}
    order = sorted(res, key=lambda f: (vids[f]["person"] != "teammate", f))
    who = {"teammate": "팀원", "self": "나"}
    names = " | ".join(f"{Path(f).stem.replace(' - 복사본', '')} {who.get(vids[f]['person'], '')}" for f in order)
    sep = "---|" * len(order)
    fm = lambda v, nd=2: "-" if v is None else f"{v:.{nd}f}"      # noqa: E731
    fs = lambda v, nd=2: "-" if v is None else f"{v:+.{nd}f}"      # noqa: E731
    L = ["# 120 같은 자세의 반복성 summary", "",
         f"**저장된 추적점에 책상 모델(경사 {rc['ramp_deg']}° 고정)을 f 를 고정해 맞추고, 모든 사용 프레임의 잔차(예측 - 추적점)를 구한 뒤, 같은 자세(회전 차이 <= {rc['pose_tol_deg']}°, 이동 차이 <= 손바닥 폭의 {rc['pose_tol_slide']}배)인 "
         f"두 프레임의 잔차를 뺀다. 모델이 틀린 부분은 같은 자세에서 같으므로 빠지고, 시간이 지나 달라진 부분(추적의 밀림, 손 모양 변화)이 남는다. 판정 없음.**", "",
         f"- D = 두 프레임의 잔차 차이의 관절 RMS (px). 가까운 쌍 = 시간 차 {rc['near_s']} s 이하 (잡음 바닥), 먼 쌍 = {rc['far_s']} s 이상.",
         f"- 먼 쌍을 둘로 나눔: 돌았다 돌아온 쌍 = 두 프레임 사이에 손이 {rc['excursion_deg']}° 이상 돌았다가 같은 자세로 온 것, 멈춰 있던 쌍 = 사이에 {rc['still_deg']}° 미만으로만 움직인 것(같은 정지 구간).",
         "- 길이·폭 변화 = (관측/예측)의 나중 프레임 / 먼저 프레임 - 1 (%). 길이 = MCP 평균 -> 손끝 평균, 폭 = 관절 5 - 17.",
         "- 기준 f 는 이 파일에만 (ref_f = 1672 px 에 고정한 해, low_f = 격자의 1280 px 에 고정한 해)."]
    for name, title in (("ref_f", "f 를 기준 f 에 고정"), ("low_f", "f 를 1280 에 고정 (모델이 달라도 결과가 같은지)")):
        g = lambda f, *p: (lambda d: d)(__import__("functools").reduce(lambda d, k: d.get(k) if isinstance(d, dict) else None, p, res[f]["eval"][name]))      # noqa: E731
        L += ["", f"## {title}", "", f"| 값 | {names} |", "|---|" + sep,
              "| 사용 프레임 | " + " | ".join(str(g(f, "n_frames")) for f in order) + " |",
              "| 전체 프레임 잔차 RMS (px) | " + " | ".join(fm(g(f, "rms_all_frames_px")) for f in order) + " |",
              "| 가까운 쌍 수 | " + " | ".join(str(g(f, "near", "n_pairs")) for f in order) + " |",
              "| 가까운 쌍 D 중앙값 (px) | " + " | ".join(fm(g(f, "near", "D_median_px")) for f in order) + " |",
              "| 먼 쌍 수 | " + " | ".join(str(g(f, "far", "n_pairs")) for f in order) + " |",
              "| 먼 쌍에 들어간 프레임 수 | " + " | ".join(str(g(f, "far_frames_covered")) for f in order) + " |",
              "| **돌았다 돌아온 쌍** 수 (쌍에 들어간 프레임 수) | " + " | ".join(f"{g(f, 'return', 'n_pairs')} ({g(f, 'return', 'frames_covered')})" for f in order) + " |",
              "| 돌았다 돌아온 쌍의 시간 차 중앙값 (s) | " + " | ".join(fm(g(f, "return", "gap_s_median"), 1) for f in order) + " |",
              "| **돌았다 돌아온 쌍 D 중앙값 (px)** | " + " | ".join(fm(g(f, "return", "D_median_px")) for f in order) + " |",
              "| 돌았다 돌아온 쌍 D 상위 10% (px) | " + " | ".join(fm(g(f, "return", "D_p90_px")) for f in order) + " |",
              "| 돌았다 돌아온 쌍 길이 변화 중앙값 / 절댓값 중앙값 (%) | " + " | ".join(f"{fs(g(f, 'return', 'length_change_pct_median'))} / {fm(g(f, 'return', 'length_change_pct_abs_median'))}" for f in order) + " |",
              "| 돌았다 돌아온 쌍 폭 변화 중앙값 / 절댓값 중앙값 (%) | " + " | ".join(f"{fs(g(f, 'return', 'width_change_pct_median'))} / {fm(g(f, 'return', 'width_change_pct_abs_median'))}" for f in order) + " |",
              "| **멈춰 있던 쌍** 수 | " + " | ".join(str(g(f, "still", "n_pairs")) for f in order) + " |",
              "| 멈춰 있던 쌍의 시간 차 중앙값 (s) | " + " | ".join(fm(g(f, "still", "gap_s_median"), 1) for f in order) + " |",
              "| **멈춰 있던 쌍 D 중앙값 (px)** | " + " | ".join(fm(g(f, "still", "D_median_px")) for f in order) + " |",
              "| 멈춰 있던 쌍 길이 / 폭 변화 절댓값 중앙값 (%) | " + " | ".join(f"{fm(g(f, 'still', 'length_change_pct_abs_median'))} / {fm(g(f, 'still', 'width_change_pct_abs_median'))}" for f in order) + " |",
              "| 먼 쌍 전체 D 중앙값 (px) | " + " | ".join(fm(g(f, "far", "D_median_px")) for f in order) + " |",
              "| 먼 쌍 D 상위 10% (px) | " + " | ".join(fm(g(f, "far", "D_p90_px")) for f in order) + " |"]
        for lab in res[order[0]]["eval"][name]["far_by_gap_s"]:
            L.append(f"| 먼 쌍 D 중앙값, 시간 차 {lab} s (쌍 수) | " + " | ".join(f"{fm(g(f, 'far_by_gap_s', lab, 'D_median_px'))} ({g(f, 'far_by_gap_s', lab, 'n_pairs')})" for f in order) + " |")
        for lab in res[order[0]]["eval"][name]["far_by_angle_deg"]:
            L.append(f"| 돌았다 돌아온 쌍 D 중앙값, 손 방향 {lab}° (쌍 수) | " + " | ".join(f"{fm(g(f, 'return', 'by_angle_deg', lab, 'D_median_px'))} ({g(f, 'return', 'by_angle_deg', lab, 'n_pairs')})" for f in order) + " |")
        for lab in res[order[0]]["eval"][name]["far_by_angle_deg"]:
            L.append(f"| 먼 쌍 D 중앙값, 손 방향 {lab}° (쌍 수) | " + " | ".join(f"{fm(g(f, 'far_by_angle_deg', lab, 'D_median_px'))} ({g(f, 'far_by_angle_deg', lab, 'n_pairs')})" for f in order) + " |")
        L += ["| 먼 쌍 길이 변화 중앙값 (%) [25% ~ 75%] | " + " | ".join(f"{fs(g(f, 'far_length_change_pct', 'median'))} [{fs(g(f, 'far_length_change_pct', 'p25'))} ~ {fs(g(f, 'far_length_change_pct', 'p75'))}]" for f in order) + " |",
              "| 먼 쌍 길이 변화 절댓값 중앙값 (%) | " + " | ".join(fm(g(f, "far_length_change_pct", "abs_median")) for f in order) + " |",
              "| 가까운 쌍 길이 변화 절댓값 중앙값 (%) | " + " | ".join(fm(g(f, "near_length_change_abs_median_pct")) for f in order) + " |",
              "| 먼 쌍 폭 변화 중앙값 (%) [25% ~ 75%] | " + " | ".join(f"{fs(g(f, 'far_width_change_pct', 'median'))} [{fs(g(f, 'far_width_change_pct', 'p25'))} ~ {fs(g(f, 'far_width_change_pct', 'p75'))}]" for f in order) + " |",
              "| 먼 쌍 폭 변화 절댓값 중앙값 (%) | " + " | ".join(fm(g(f, "far_width_change_pct", "abs_median")) for f in order) + " |",
              "| 가까운 쌍 폭 변화 절댓값 중앙값 (%) | " + " | ".join(fm(g(f, "near_width_change_abs_median_pct")) for f in order) + " |"]
    L += ["", "## 한계", "", "- 같은 자세의 기준은 책상 모델이 준 회전·이동이다. 모델이 자세를 조금 틀리게 주면 쌍이 덜 정확히 맞는다(허용 범위 안의 자세 차이도 D 에 조금 들어간다).",
          "- D 는 추적의 밀림과 손 모양 변화를 구분하지 못한다. 둘의 합이다.", "- 먼 쌍은 손이 실제로 같은 자세로 돌아온 구간에서만 생긴다(주로 정지 구간). 영상마다 쌍이 생긴 자세가 다르다.", ""]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------------------------------
# 121: 손에 찍은 빨간 마커를 따라가 f 추정 (마커 영상) + 마커 없는 영상은 지금까지와 같은 광류 사슬
# ----------------------------------------------------------------------------------------------------
def mt_detect(bgr: np.ndarray, region: tuple, mc: dict, box_long_px: float) -> np.ndarray:
    """한 프레임의 마커 중심 (k, 2): handfocal.mk_blobs (HSV 색, 넓이, 둥근 정도; 채도 가중 중심) 뒤, 서로 merge_frac x 손 상자 긴 변 안에 있는 조각은 넓이 가중 평균으로 합친다(번진 점이 둘로 나뉜 것)."""
    b = hf.mk_blobs(bgr, region, mc)
    xy, ar = b["xy"], b["area"]
    lim = float(mc["merge_frac"]) * float(box_long_px)
    lab = np.arange(len(xy))
    for i in range(len(xy)):
        for j in range(i + 1, len(xy)):
            if np.linalg.norm(xy[i] - xy[j]) <= lim:
                lab[lab == lab[j]] = lab[i]
    out = [(xy[lab == g] * ar[lab == g][:, None]).sum(0) / ar[lab == g].sum() for g in np.unique(lab)]
    return np.array(out, float).reshape(-1, 2)


def mt_track(dets_xy: list, ref_index: int, ref_xy: np.ndarray, box_long_px: np.ndarray, mc: dict, used: list) -> dict:
    """기준 프레임의 마커를 앞뒤로 따라간다. handfocal.mk_track_reacquire 는 기준 -> 현재 프레임의 호모그래피 하나로 검사해서(평평한 그림용) 입체인 실제 손을 크게 돌리면 프레임을 버린다.
    여기서는 검사를 직전에 받아들인 프레임 -> 현재 프레임으로 한다(이웃 프레임 사이는 입체 손도 거의 호모그래피로 움직임).
    프레임마다: 예측 위치 = 직전 위치를 직전 단계의 호모그래피로 한 번 더 옮긴 것(등속 가정). 예측 위치와 검출을 헝가리안으로 매칭(거리 <= move_frac x 손 상자 x min(건너뛴 프레임 수, gap_cap)).
    매칭된 쌍으로 직전 -> 현재 호모그래피(RANSAC, step_ransac_frac x 손 상자). inlier 가 매칭의 min_inlier_ratio 이상이고 used 마커의 inlier 가 min_markers_ratio 이상이면 그 프레임을 받아들인다(outlier 마커는 그 프레임에서 비움).
    보이지 않은 마커의 '직전 위치'는 그 호모그래피로 옮겨 둔다(다시 보이면 이어 붙음). 반환: {xy (n, m, 2), ok (n,), reason (n,)}."""
    import cv2
    from scipy.optimize import linear_sum_assignment
    n, m = len(dets_xy), len(ref_xy)
    xy = np.full((n, m, 2), np.nan)
    ok = np.zeros(n, bool)
    reason = np.array([""] * n, dtype=object)
    xy[ref_index], ok[ref_index] = ref_xy, True
    used = [int(j) for j in used]
    need = int(np.ceil(float(mc["min_markers_ratio"]) * len(used) - 1e-9))
    mv = lambda P, Hm: cv2.perspectiveTransform(P.reshape(-1, 1, 2).astype(np.float64), Hm).reshape(-1, 2)      # noqa: E731
    for order in (range(ref_index + 1, n), range(ref_index - 1, -1, -1)):
        last, last_i, H_step = ref_xy.copy(), ref_index, np.eye(3)
        for i in order:
            d = dets_xy[i]
            if d is None or len(d) < need:
                reason[i] = "too_few_markers"
                continue
            gap = abs(i - last_i)
            pred = last.copy()
            for _ in range(min(gap, int(mc["gap_cap"]))):
                pred = mv(pred, H_step)
            lim = float(mc["move_frac"]) * float(box_long_px[i]) * min(gap, int(mc["gap_cap"]))
            cost = np.linalg.norm(pred[:, None, :] - d[None, :, :], axis=-1)
            r, c = linear_sum_assignment(np.where(cost <= lim, cost, 1e9))
            good = cost[r, c] <= lim
            r, c = r[good], c[good]
            if np.isin(r, used).sum() < need:
                reason[i] = "match_failed"
                continue
            Hm, inl = hf.mk_homography(last[r], d[c], float(mc["step_ransac_frac"]) * float(box_long_px[i]))
            if Hm is None or inl is None or inl.sum() < float(mc["min_inlier_ratio"]) * len(r) or np.isin(r[inl], used).sum() < need:
                reason[i] = "homography_failed"
                continue
            cur = np.full((m, 2), np.nan)
            cur[r[inl]] = d[c[inl]]
            xy[i], ok[i] = cur, True
            Hs = Hm if gap == 1 else np.eye(3)                         # 건너뛴 뒤에는 등속 예측을 쓰지 않음
            last, last_i, H_step = np.where(np.isnan(cur), mv(last, Hm), cur), i, Hs
    return {"xy": xy, "ok": ok, "reason": reason}


def mt_markers(pngs: list, tr: dict, size_wh, cfg: dict) -> dict:
    """마커 추적점. MediaPipe 는 (1) 찾을 영역(손 상자를 region_margin 만큼 넓힌 것)과 (2) 기준 프레임에서 마커에 관절 번호를 붙이는 데만 쓴다(가장 가까운 관절, naming_radius_frac x 상자 안). 좌표는 모두 마커 중심이다.
    기준 프레임 = 첫 정지 구간에서 관절 묶음(joint_ids)의 마커가 가장 많이 이름 붙은 프레임(같으면 구간 가운데에 가까운 것). 추적 = mt_track (직전 프레임 기준 매칭과 호모그래피 검사).
    반환: track_hand 와 같은 모양 {kp (n, 21, 2), box, use, ref, breaks, inl} + info."""
    import cv2
    mc = cfg["marker_track"]["markers"]
    W, H = size_wh
    n = len(pngs)
    ids = [int(j) for j in cfg["joint_ids"]]
    boxes = np.stack([of.kt_carry(tr["present"], tr["box"][:, k]) for k in range(4)], axis=1)
    bl = of.box_long(boxes)
    det = [mt_detect(cv2.imread(pngs[i], cv2.IMREAD_COLOR), hf.mk_region(boxes[i], float(mc["region_margin"]), W, H), mc, bl[i]) for i in range(n)]
    pauses = pp_pauses(tr, (W, H), cfg)
    cand = [int(i) for i in (pauses[0] if pauses else np.flatnonzero(tr["present"]))]
    best = None
    for i in cand:
        if len(det[i]) < 4:
            continue
        nm = hf.mk_assign_names(det[i], tr["kp"][i])
        d = np.linalg.norm(det[i] - tr["kp"][i][np.maximum(nm, 0)], axis=1)
        nm = np.where((nm >= 0) & (d <= float(mc["naming_radius_frac"]) * bl[i]), nm, -1)
        key = (int(np.isin(nm, ids).sum()), -abs(i - (cand[0] + cand[-1]) / 2))
        if best is None or key > best[0]:
            best = (key, i, nm)
    if best is None:
        raise ValueError("기준 프레임에서 마커를 찾지 못함")
    _, ref, names = best
    used = [j for j, x in enumerate(names) if int(x) in ids]
    trk = mt_track(det, ref, det[ref], bl, mc, used)
    resid = []                                                         # 기준 -> 각 프레임을 호모그래피 하나로 맞췄을 때의 잔차 (마커가 한 평면처럼 움직이는 정도; 추적에는 쓰지 않음)
    for i in np.flatnonzero(trk["ok"]):
        v = np.isfinite(trk["xy"][i][:, 0])
        Hm, _ = hf.mk_homography(det[ref][v], trk["xy"][i][v]) if v.sum() >= 8 else (None, None)
        if Hm is not None:
            q = cv2.perspectiveTransform(det[ref][v].reshape(-1, 1, 2).astype(np.float64), Hm).reshape(-1, 2)
            resid.append(float(np.sqrt(np.mean(((q - trk["xy"][i][v]) ** 2).sum(1))) / bl[i] * 100.0))
    kp = np.full((n, 21, 2), np.nan)
    for j, x in enumerate(names):
        if x >= 0:
            kp[:, int(x)] = trk["xy"][:, j]
    cnt = np.isfinite(kp[:, ids, 0]).sum(1)
    valid = trk["ok"] & (cnt >= int(mc["min_joints"]))
    tb = np.full((n, 4), np.nan)
    tb[valid] = np.c_[np.nanmin(kp[valid][:, ids], axis=1), np.nanmax(kp[valid][:, ids], axis=1)]
    edge = np.zeros(n, bool)
    edge[valid] = of.edge_flags(tb[valid], np.ones(valid.sum(), bool), W, H)
    reasons = {}
    for r in trk["reason"]:
        if r:
            reasons[str(r)] = reasons.get(str(r), 0) + 1
    both = valid[:, None] & np.isfinite(kp[:, ids, 0]) & tr["present"][:, None]
    dist = np.where(both, np.linalg.norm(kp[:, ids] - tr["kp"][:, ids], axis=2) / bl[:, None] * 100.0, np.nan)
    info = {"detections_median": float(np.median([len(d) for d in det])), "ref_detections": int(len(det[ref])), "ref_named_in_joint_set": int(len(used)), "named_joints": sorted(int(x) for x in names if x >= 0),
            "missing_joints": [j for j in ids if j not in set(int(x) for x in names)], "valid_ratio": float(valid.mean()), "reasons": reasons, "joints_per_valid_frame_median": float(np.median(cnt[valid])) if valid.any() else None,
            "joint_visible_ratio_min": float(np.isfinite(kp[valid][:, [j for j in ids if j in set(int(x) for x in names)], 0]).mean(0).min()) if valid.any() else None,
            "marker_to_mediapipe_median_pct": sv_pct(dist, 50), "ref_homography_rms_pct_median": sv_pct(resid, 50), "ref_homography_rms_pct_max": sv_pct(resid, 100)}
    return {"kp": kp, "box": tb, "use": valid & ~edge, "ref": int(ref), "breaks": 0, "inl": [], "info": info}


def mt_job(args):
    """한 영상: 프레임·MediaPipe 검출·track -> 광류 사슬 추적점(지금까지와 같음: 첫 정지 구간 기준). markers: true 인 영상은 마커 추적점도 만든다. 추적점은 run 폴더의 points/ 에 저장."""
    e, cfg, run_root = args
    import cv2
    cv2.setNumThreads(1)
    pngs, times, tr, (W, H), tmp = prepare(e, cfg, run_root)
    out, lines = {}, []
    try:
        ang = hand_angle(tr)
        x = of.track_hand(pngs, tr["kp"], tr["present"], tr["box"], (W, H), cfg["tracking"], ref=rof.choose_reference_pose(tr, (W, H), cfg))
        x["size"] = (W, H)
        save_points(run_root, e["file"], x, (W, H), ang, times)
        out["chain"] = {"kp": x["kp"], "box": x["box"], "use": x["use"], "ref": x["ref"], "size": (W, H), "info": {"breaks": int(x["breaks"]), "inliers_median": sv_pct(x["inl"], 50)}}
        lines.append(f"  {e['file']} 광류 사슬: 프레임 {len(times)}, 기준 {x['ref']}, 끊김 {x['breaks']}, 사용 {int(x['use'].sum())}")
        if e.get("markers"):
            m = mt_markers(pngs, tr, (W, H), cfg)
            m["size"] = (W, H)
            save_points(run_root, Path(e["file"]).stem + "_marker.MOV", m, (W, H), ang, times)
            out["marker"] = {"kp": m["kp"], "box": m["box"], "use": m["use"], "ref": m["ref"], "size": (W, H), "info": m["info"]}
            lines.append(f"  {e['file']} 마커: 기준 {m['ref']}, 기준 프레임 마커 {m['info']['ref_detections']}개 중 관절 묶음 {m['info']['ref_named_in_joint_set']}개, 사용 {int(m['use'].sum())} ({m['info']['valid_ratio']:.2f}), {m['info']['reasons']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out, lines


def mt_run(cfg: dict, vid: dict, log) -> dict:
    """121: (영상, 점 출처)마다 free_template(tl_template_job) -> 경사 고정·함께 추정(ms2_job)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    items = [(f, v, x) for f, d in vid.items() for v, x in d.items()]
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        fts = list(ex.map(hf.tl_template_job, [rof.estimate_args(cfg, x) for _, _, x in items], chunksize=1))
    todo = [(f"ramp_{r:g}", "ramp_fixed", r) for r in cfg["ramps_deg"]] + [(f"free_{r:g}", "ramp_free", r) for r in cfg["free_starts_deg"]]
    keep = ("f", "pitch_deg", "roll_deg", "desk_tilt_deg", "turn_range_deg", "rms_px", "cost", "boundary", "start")
    res, jobs, keys = {}, [], []
    for (f, v, x), ft in zip(items, fts):
        A = ft.get("A")
        grid = hf.f_grid(cfg["pipeline"]["estimate"], max(*x["size"]))
        d = res.setdefault(f, {}).setdefault(v, {})
        d.update({"ref": int(x["ref"]), "n_use": int(x["use"].sum()), "info": x["info"],
                  "free_template": {"failed": ft.get("failed")} if A is None else {"f": float(A["f"]), "boundary": hf.od_boundary(float(A["f"]), grid), "rms_px": float((A.get("ba") or {}).get("rms_px", np.nan))}})
        if A is None:
            continue
        for mk, mode, r in todo:
            jobs.append(rof.estimate_args(cfg, x, A) + (mode, r))
            keys.append((f, v, mk))
    log(f"  추정 {len(jobs)}건")
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        for (f, v, mk), o in zip(keys, ex.map(hf.ms2_job, jobs, chunksize=1)):
            res[f][v][mk] = {kk: o[kk] for kk in keep if kk in o} if "failed" not in o else o
    for f in res:
        for v, d in res[f].items():
            free = {k: y for k, y in d.items() if k.startswith("free_") and k != "free_template" and isinstance(y, dict) and "cost" in y}
            if free:
                bk = min(free, key=lambda k: free[k]["cost"])
                d["ramp_free"] = {**free[bk], "best_start": bk}
    return res


def mt_summary(res: dict, cfg: dict, own: dict) -> str:
    vids = {e["file"]: e for e in cfg["videos"]}
    who = {"teammate": "팀원", "self": "나"}
    src = {"marker": "마커", "chain": "광류 사슬"}
    rf = lambda f: float(vids[f].get("ref_f_px") or cfg["ref_f"]["value"])      # noqa: E731

    def cell(f, v, key):
        d = res[f][v].get(key) or {}
        if "f" not in d:
            return "실패" if d.get("failed") else "-"
        return f"{d['f']:.0f} ({100 * (d['f'] / rf(f) - 1):+.1f}%)" + ("*" if d.get("boundary") else "") + (f", {d['pitch_deg']:.1f}°" if key == "ramp_free" else "")
    k11 = f"ramp_{cfg['ramps_deg'][0]:g}"
    L = ["# 121 마커 추적 summary", "",
         "**새로 찍은 영상 6개. 마커 영상(팀원 손에 빨간 마커)은 마커 중심을 프레임마다 검출·매칭한 점으로, 나머지는 지금까지와 같은 광류 사슬(첫 정지 구간 기준)로 같은 추정(j16, free_template -> 경사 고정 / 함께 추정)을 돌린다. "
         "마커 영상에는 광류 사슬도 함께 돌려 같은 영상에서 두 점 출처를 비교한다. 판정 없음.**", "",
         "- 기준 f 는 이 파일에만 (새 폰 체커보드 값; 같은 폰으로 찍었다고 보고 적용). * = f 탐색 범위 끝. 함께 추정 칸의 각도 = 추정된 경사.",
         "- 마커 추적에서 MediaPipe 는 찾을 영역(손 상자)과 기준 프레임에서 마커에 관절 번호를 붙이는 데만 쓴다. 좌표는 모두 마커 중심.", "",
         "## f", "", f"| 영상 | 손 | 마커 | 점 출처 | 사용 프레임 | 경사 {cfg['ramps_deg'][0]:g}° 고정 | 함께 추정 | free_template | 책상 기울기 (도) | RMS (px) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for f in res:
        for v in res[f]:
            d = res[f][v]
            g = d.get(k11) or {}
            L.append(f"| {f} | {who.get(vids[f]['person'], vids[f]['person'])} | {'있음' if vids[f].get('markers') else '없음'} | {src[v]} | {d['n_use']} | {cell(f, v, k11)} | {cell(f, v, 'ramp_free')} | {cell(f, v, 'free_template')} | "
                     + ("-" if "f" not in g else f"{g['desk_tilt_deg']:.1f}") + " | " + ("-" if "f" not in g else f"{g['rms_px']:.2f}") + " |")
    L += ["", "## 마커 추적 정보", "", "| 영상 | 기준 프레임 | 기준 프레임 마커 수 | 그중 j16 이름 붙은 수 | 이름 못 붙은 관절 | 사용 프레임 비율 | 프레임당 j16 마커 중앙값 | 마커-검출 관절 거리 중앙값 (상자 %) | 기준 프레임에서 호모그래피 하나로 맞춘 잔차 중앙값 / 최대 (상자 %) | 버린 프레임 이유 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for f in res:
        if "marker" in res[f]:
            i = res[f]["marker"]["info"]
            L.append(f"| {f} | {res[f]['marker']['ref']} | {i['ref_detections']} | {i['ref_named_in_joint_set']} | {i['missing_joints'] or '없음'} | {i['valid_ratio']:.2f} | {i['joints_per_valid_frame_median']} | {i['marker_to_mediapipe_median_pct']:.2f} | {i['ref_homography_rms_pct_median']:.2f} / {i['ref_homography_rms_pct_max']:.2f} | {i['reasons'] or '없음'} |")
    L += ["", "## 광류 사슬 정보", "", "| 영상 | 기준 프레임 | 끊김 | inlier 중앙값 |", "|---|---|---|---|"]
    for f in res:
        i = res[f]["chain"]["info"]
        L.append(f"| {f} | {res[f]['chain']['ref']} | {i['breaks']} | {i['inliers_median']:.0f} |")
    L += ["", "## 한계", "", "- 영상 6개(마커 2, 나 2, 팀원 2). 마커는 피부에 찍은 점이라 피부와 함께 움직인다(뼈 관절의 위치가 아님).",
          "- 마커 중심의 높이는 피부 표면이다. 책상 모델(평면 손 모양, 경사 고정)은 그대로다.", "- 마커 영상의 광류 사슬은 마커가 만든 무늬도 특징점으로 쓴다(마커 없는 영상과 조건이 다름).", ""]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------------------------------
# 122: 마커 검출을 "주변보다 붉은 정도"로 (121 의 HSV 색 규칙은 팀원 손 영상에 맞춘 값이라 내 손의 흐린 마커를 21개 중 14개만 잡음)
# ----------------------------------------------------------------------------------------------------
def mt2_detect(bgr: np.ndarray, region: tuple, mc: dict, box_long_px: float) -> np.ndarray:
    """한 프레임의 마커 중심 (k, 2): handfocal.mo_blobs (39 의 함수: Lab a* 에서 주변 중앙값을 뺀 값이 contrast_min 이상인 조각, 넓이·둥근 정도, 붉은 정도 가중 중심)를 영역 안에서 돌리고,
    mt_detect 와 같이 merge_frac x 손 상자 긴 변 안의 조각을 넓이 가중 평균으로 합친다. 영역은 median_ksize 만큼 넓혀 잘라 가장자리의 주변 중앙값이 달라지지 않게 한다."""
    x0, y0, x1, y1 = region
    pad = int(mc["median_ksize"])
    X0, Y0 = max(0, x0 - pad), max(0, y0 - pad)
    b = hf.mo_blobs(np.ascontiguousarray(bgr[Y0:y1 + pad, X0:x1 + pad]), mc)
    xy, ar = b["xy"] + np.array([X0, Y0], float), b["area"]
    inb = (xy[:, 0] >= x0) & (xy[:, 0] < x1) & (xy[:, 1] >= y0) & (xy[:, 1] < y1) if len(xy) else np.zeros(0, bool)
    xy, ar = xy[inb], ar[inb]
    lim = float(mc["merge_frac"]) * float(box_long_px)
    lab = np.arange(len(xy))
    for i in range(len(xy)):
        for j in range(i + 1, len(xy)):
            if np.linalg.norm(xy[i] - xy[j]) <= lim:
                lab[lab == lab[j]] = lab[i]
    out = [(xy[lab == g] * ar[lab == g][:, None]).sum(0) / ar[lab == g].sum() for g in np.unique(lab)]
    return np.array(out, float).reshape(-1, 2)


def mt2_markers(pngs: list, tr: dict, size_wh, cfg: dict) -> dict:
    """mt_markers 와 같되 마커 검출만 mt2_detect(주변보다 붉은 정도)로 바꾼 것. 마커 추적점. MediaPipe 는 (1) 찾을 영역(손 상자를 region_margin 만큼 넓힌 것)과 (2) 기준 프레임에서 마커에 관절 번호를 붙이는 데만 쓴다(가장 가까운 관절, naming_radius_frac x 상자 안). 좌표는 모두 마커 중심이다.
    기준 프레임 = 첫 정지 구간에서 관절 묶음(joint_ids)의 마커가 가장 많이 이름 붙은 프레임(같으면 구간 가운데에 가까운 것). 추적 = mt_track (직전 프레임 기준 매칭과 호모그래피 검사).
    반환: track_hand 와 같은 모양 {kp (n, 21, 2), box, use, ref, breaks, inl} + info."""
    import cv2
    mc = cfg["marker_track"]["markers"]
    W, H = size_wh
    n = len(pngs)
    ids = [int(j) for j in cfg["joint_ids"]]
    boxes = np.stack([of.kt_carry(tr["present"], tr["box"][:, k]) for k in range(4)], axis=1)
    bl = of.box_long(boxes)
    det = [mt2_detect(cv2.imread(pngs[i], cv2.IMREAD_COLOR), hf.mk_region(boxes[i], float(mc["region_margin"]), W, H), mc, bl[i]) for i in range(n)]
    pauses = pp_pauses(tr, (W, H), cfg)
    cand = [int(i) for i in (pauses[0] if pauses else np.flatnonzero(tr["present"]))]
    best = None
    for i in cand:
        if len(det[i]) < 4:
            continue
        nm = hf.mk_assign_names(det[i], tr["kp"][i])
        d = np.linalg.norm(det[i] - tr["kp"][i][np.maximum(nm, 0)], axis=1)
        nm = np.where((nm >= 0) & (d <= float(mc["naming_radius_frac"]) * bl[i]), nm, -1)
        key = (int(np.isin(nm, ids).sum()), -abs(i - (cand[0] + cand[-1]) / 2))
        if best is None or key > best[0]:
            best = (key, i, nm)
    if best is None:
        raise ValueError("기준 프레임에서 마커를 찾지 못함")
    _, ref, names = best
    used = [j for j, x in enumerate(names) if int(x) in ids]
    trk = mt_track(det, ref, det[ref], bl, mc, used)
    resid = []                                                         # 기준 -> 각 프레임을 호모그래피 하나로 맞췄을 때의 잔차 (마커가 한 평면처럼 움직이는 정도; 추적에는 쓰지 않음)
    for i in np.flatnonzero(trk["ok"]):
        v = np.isfinite(trk["xy"][i][:, 0])
        Hm, _ = hf.mk_homography(det[ref][v], trk["xy"][i][v]) if v.sum() >= 8 else (None, None)
        if Hm is not None:
            q = cv2.perspectiveTransform(det[ref][v].reshape(-1, 1, 2).astype(np.float64), Hm).reshape(-1, 2)
            resid.append(float(np.sqrt(np.mean(((q - trk["xy"][i][v]) ** 2).sum(1))) / bl[i] * 100.0))
    kp = np.full((n, 21, 2), np.nan)
    for j, x in enumerate(names):
        if x >= 0:
            kp[:, int(x)] = trk["xy"][:, j]
    cnt = np.isfinite(kp[:, ids, 0]).sum(1)
    valid = trk["ok"] & (cnt >= int(mc["min_joints"]))
    tb = np.full((n, 4), np.nan)
    tb[valid] = np.c_[np.nanmin(kp[valid][:, ids], axis=1), np.nanmax(kp[valid][:, ids], axis=1)]
    edge = np.zeros(n, bool)
    edge[valid] = of.edge_flags(tb[valid], np.ones(valid.sum(), bool), W, H)
    reasons = {}
    for r in trk["reason"]:
        if r:
            reasons[str(r)] = reasons.get(str(r), 0) + 1
    both = valid[:, None] & np.isfinite(kp[:, ids, 0]) & tr["present"][:, None]
    dist = np.where(both, np.linalg.norm(kp[:, ids] - tr["kp"][:, ids], axis=2) / bl[:, None] * 100.0, np.nan)
    info = {"detections_median": float(np.median([len(d) for d in det])), "ref_detections": int(len(det[ref])), "ref_named_in_joint_set": int(len(used)), "named_joints": sorted(int(x) for x in names if x >= 0),
            "missing_joints": [j for j in ids if j not in set(int(x) for x in names)], "valid_ratio": float(valid.mean()), "reasons": reasons, "joints_per_valid_frame_median": float(np.median(cnt[valid])) if valid.any() else None,
            "joint_visible_ratio_min": float(np.isfinite(kp[valid][:, [j for j in ids if j in set(int(x) for x in names)], 0]).mean(0).min()) if valid.any() else None,
            "marker_to_mediapipe_median_pct": sv_pct(dist, 50), "ref_homography_rms_pct_median": sv_pct(resid, 50), "ref_homography_rms_pct_max": sv_pct(resid, 100)}
    return {"kp": kp, "box": tb, "use": valid & ~edge, "ref": int(ref), "breaks": 0, "inl": [], "info": info}


def mt2_job(args):
    """한 영상: 프레임·MediaPipe 검출·track -> 광류 사슬 추적점(지금까지와 같음: 첫 정지 구간 기준). markers: true 인 영상은 마커 추적점도 만든다. 추적점은 run 폴더의 points/ 에 저장."""
    e, cfg, run_root = args
    import cv2
    cv2.setNumThreads(1)
    pngs, times, tr, (W, H), tmp = prepare(e, cfg, run_root)
    out, lines = {}, []
    try:
        ang = hand_angle(tr)
        x = of.track_hand(pngs, tr["kp"], tr["present"], tr["box"], (W, H), cfg["tracking"], ref=rof.choose_reference_pose(tr, (W, H), cfg))
        x["size"] = (W, H)
        save_points(run_root, e["file"], x, (W, H), ang, times)
        out["chain"] = {"kp": x["kp"], "box": x["box"], "use": x["use"], "ref": x["ref"], "size": (W, H), "info": {"breaks": int(x["breaks"]), "inliers_median": sv_pct(x["inl"], 50)}}
        lines.append(f"  {e['file']} 광류 사슬: 프레임 {len(times)}, 기준 {x['ref']}, 끊김 {x['breaks']}, 사용 {int(x['use'].sum())}")
        if e.get("markers"):
            m = mt2_markers(pngs, tr, (W, H), cfg)
            m["size"] = (W, H)
            save_points(run_root, Path(e["file"]).stem + "_marker.MOV", m, (W, H), ang, times)
            out["marker"] = {"kp": m["kp"], "box": m["box"], "use": m["use"], "ref": m["ref"], "size": (W, H), "info": m["info"]}
            lines.append(f"  {e['file']} 마커: 기준 {m['ref']}, 기준 프레임 마커 {m['info']['ref_detections']}개 중 관절 묶음 {m['info']['ref_named_in_joint_set']}개, 사용 {int(m['use'].sum())} ({m['info']['valid_ratio']:.2f}), {m['info']['reasons']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out, lines


def mt2_summary(res: dict, cfg: dict, own: dict) -> str:
    """mt_summary 의 표 + 122 의 머리말."""
    s = mt_summary(res, cfg, own)
    i = s.index("- 기준 f 는")
    head = ("# 122 마커 추적 (두 사람) summary\n\n**마커를 찍은 영상 4개(팀원 7399·7400, 나 7405·7406). 마커 중심을 프레임마다 '주변보다 붉은 정도'로 검출하고(121 의 HSV 색 규칙 대신; 두 사람 영상에 같은 규칙) "
            "mt_track 으로 매칭한 점으로 같은 추정(j16, free_template -> 경사 고정 / 함께 추정)을 돌린다. 같은 영상에 광류 사슬도 돌려 두 점 출처를 비교한다. 판정 없음.**\n\n")
    return head + s[i:].replace("- 영상 6개(마커 2, 나 2, 팀원 2). ", "- 영상 4개(사람당 2개). ")


# ----------------------------------------------------------------------------------------------------
# 123: 광류 사슬이 마커를 얼마나 정확히 옮기는지 (마커 영상, 저장된 점)
# ----------------------------------------------------------------------------------------------------
def cm_load(args):
    """한 영상: points_run 에 저장된 광류 사슬 점과 마커 점. 사슬의 기준 -> 각 프레임 호모그래피는 사슬 점 21개(정확히 호모그래피 하나로 옮긴 점)에서 되찾는다.
    세 가지 점을 만든다(j 관절 번호 자리에): marker = 마커 그대로, chain_on_marker = 기준 프레임의 마커를 사슬의 호모그래피로 옮긴 것, planar_marker = 기준 프레임의 마커를 그 프레임 마커에 가장 잘 맞는 호모그래피(최소제곱)로 옮긴 것."""
    e, cfg, run_root = args
    import cv2
    cm = cfg["chain_vs_marker"]
    c = load_points(cm["points_run"], e["file"])
    m = load_points(cm["points_run"], Path(e["file"]).stem + "_marker.MOV")
    ids = [int(j) for j in cfg["joint_ids"]]
    n = len(c["kp"])
    r = int(c["ref"])
    if not m["use"][r]:
        raise ValueError("사슬의 기준 프레임에 마커가 없음")
    M0 = m["kp"][r]
    have = np.isfinite(M0[:, 0])
    jj = [j for j in ids if have[j]]
    mv = lambda P, Hm: cv2.perspectiveTransform(P.reshape(-1, 1, 2).astype(np.float64), Hm).reshape(-1, 2)      # noqa: E731
    con, pla = np.full((n, 21, 2), np.nan), np.full((n, 21, 2), np.nan)
    for i in range(n):
        if not (c["use"][i] and m["use"][i]):
            continue
        Hc, _ = cv2.findHomography(c["kp"][r].astype(np.float64), c["kp"][i].astype(np.float64), 0)
        con[i][jj] = mv(M0[jj], Hc)
        v = [j for j in jj if np.isfinite(m["kp"][i][j, 0])]
        if len(v) >= 8:
            Hb, _ = cv2.findHomography(M0[v].astype(np.float64), m["kp"][i][v].astype(np.float64), 0)
            pla[i][jj] = mv(M0[jj], Hb)
    use = c["use"] & m["use"] & np.isfinite(con[:, jj[0], 0]) & np.isfinite(pla[:, jj[0], 0])
    box = lambda K: np.c_[np.nanmin(K[:, jj], axis=1), np.nanmax(K[:, jj], axis=1)]      # noqa: E731
    out = {"joints": jj, "angle": c["hand_angle"] - c["hand_angle"][r], "times": c["times"], "ref": r, "size": c["size"], "use": use, "marker_box": m["box"],
           "pts": {"marker": {"kp": m["kp"], "box": m["box"], "use": use, "ref": r, "size": c["size"]},
                   "chain_on_marker": {"kp": con, "box": np.where(use[:, None], box(np.where(use[:, None, None], con, 0.0)), np.nan), "use": use, "ref": r, "size": c["size"]},
                   "planar_marker": {"kp": pla, "box": np.where(use[:, None], box(np.where(use[:, None, None], pla, 0.0)), np.nan), "use": use, "ref": r, "size": c["size"]}}}
    return out, [f"  {e['file']}: 프레임 {n}, 함께 쓰는 프레임 {int(use.sum())}, 관절 {len(jj)}"]


def cm_stats(x: dict, cm: dict) -> dict:
    """프레임·관절별 거리(px, 마커 상자 긴 변 대비 %)의 요약: total = 사슬로 옮긴 마커 - 실제 마커, nonplanar = 가장 잘 맞는 호모그래피로 옮긴 마커 - 실제 마커, chain_h = 사슬 - 가장 잘 맞는 호모그래피.
    전체, 손 방향 구간별, 관절 줄별. 그리고 손 길이(MCP 평균 -> 손끝 평균)·폭(가장 바깥 두 MCP)의 예측/실제 - 1 (%)을 구간별로."""
    jj, u = x["joints"], x["use"]
    M, C, P = x["pts"]["marker"]["kp"][:, jj], x["pts"]["chain_on_marker"]["kp"][:, jj], x["pts"]["planar_marker"]["kp"][:, jj]
    bl = of.box_long(np.asarray(x["marker_box"], float))
    d = {"total": np.linalg.norm(C - M, axis=2), "nonplanar": np.linalg.norm(P - M, axis=2), "chain_h": np.linalg.norm(C - P, axis=2)}
    ang = x["angle"]
    bins = [float(b) for b in cm["angle_bins_deg"]]
    labs = [f"{a:g}~{b:g}" for a, b in zip(bins[:-1], bins[1:])]
    bi = np.clip(np.digitize(ang, bins) - 1, 0, len(labs) - 1)
    rms = lambda A, m: float(np.sqrt(np.nanmean(A[m] ** 2))) if m.any() else None      # noqa: E731
    out = {"n_frames": int(u.sum()), "box_long_px_median": sv_pct(bl[u], 50), "angle_min": float(np.nanmin(ang[u])), "angle_max": float(np.nanmax(ang[u])),
           "n_frames_by_angle": {labs[k]: int((u & (bi == k)).sum()) for k in range(len(labs))}, "rms_px": {}, "rms_pct_box": {}, "rms_px_by_angle": {}, "rms_px_by_row": {}, "length_pct_by_angle": {}, "width_pct_by_angle": {}}
    for k, A in d.items():
        out["rms_px"][k] = rms(A, u)
        out["rms_pct_box"][k] = float(np.sqrt(np.nanmean((A[u] / bl[u][:, None] * 100.0) ** 2)))
        out["rms_px_by_angle"][k] = {labs[b]: rms(A, u & (bi == b)) for b in range(len(labs))}
        out["rms_px_by_row"][k] = {name: (float(np.sqrt(np.nanmean(A[u][:, [jj.index(j) for j in grp if j in jj]] ** 2))) if any(j in jj for j in grp) else None) for name, grp in CDX_ROWS.items()}
    tips, mcps = [jj.index(j) for j in CDX_ROWS["TIP"] if j in jj], [jj.index(j) for j in CDX_ROWS["MCP"] if j in jj]
    w0, w1 = mcps[0], mcps[-1]
    ln = lambda K: np.linalg.norm(np.nanmean(K[:, tips], axis=1) - np.nanmean(K[:, mcps], axis=1), axis=1)      # noqa: E731
    wd = lambda K: np.linalg.norm(K[:, w0] - K[:, w1], axis=1)      # noqa: E731
    out["width_joints"] = [int(jj[w0]), int(jj[w1])]
    with np.errstate(invalid="ignore", divide="ignore"):
        for k, K in (("chain", C), ("planar", P)):
            lr, wr = 100 * (ln(K) / ln(M) - 1), 100 * (wd(K) / wd(M) - 1)
            out["length_pct_by_angle"][k] = {labs[b]: sv_pct(lr[u & (bi == b)], 50) for b in range(len(labs))}
            out["width_pct_by_angle"][k] = {labs[b]: sv_pct(wr[u & (bi == b)], 50) for b in range(len(labs))}
    return out


def cm_run(cfg: dict, vid: dict, log) -> dict:
    """123: 영상마다 cm_stats, 그리고 세 가지 점(marker, chain_on_marker, planar_marker)으로 free_template -> 경사 고정·함께 추정."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    cm = cfg["chain_vs_marker"]
    res = {f: {"stats": cm_stats(x, cm), "f": {}} for f, x in vid.items()}
    items = [(f, v, {**y, "info": {}}) for f, x in vid.items() for v, y in x["pts"].items()]
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        fts = list(ex.map(hf.tl_template_job, [rof.estimate_args(cfg, y) for _, _, y in items], chunksize=1))
    todo = [(f"ramp_{r:g}", "ramp_fixed", r) for r in cfg["ramps_deg"]] + [(f"free_{r:g}", "ramp_free", r) for r in cfg["free_starts_deg"]]
    keep = ("f", "pitch_deg", "roll_deg", "desk_tilt_deg", "turn_range_deg", "rms_px", "cost", "boundary", "start")
    jobs, keys = [], []
    for (f, v, y), ft in zip(items, fts):
        A = ft.get("A")
        res[f]["f"][v] = {"n_use": int(y["use"].sum())}
        if A is None:
            res[f]["f"][v]["failed"] = ft.get("failed")
            continue
        for mk, mode, r in todo:
            jobs.append(rof.estimate_args(cfg, y, A) + (mode, r))
            keys.append((f, v, mk))
    log(f"  추정 {len(jobs)}건")
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        for (f, v, mk), o in zip(keys, ex.map(hf.ms2_job, jobs, chunksize=1)):
            res[f]["f"][v][mk] = {kk: o[kk] for kk in keep if kk in o} if "failed" not in o else o
    for f in res:
        for v, d in res[f]["f"].items():
            free = {k: y for k, y in d.items() if k.startswith("free_") and isinstance(y, dict) and "cost" in y}
            if free:
                bk = min(free, key=lambda k: free[k]["cost"])
                d["ramp_free"] = {**free[bk], "best_start": bk}
    return res


def cm_summary(res: dict, cfg: dict, own: dict) -> str:
    vids = {e["file"]: e for e in cfg["videos"]}
    order = sorted(res, key=lambda f: (vids[f]["person"] != "teammate", f))
    who = {"teammate": "팀원", "self": "나"}
    names = " | ".join(f"{Path(f).stem} {who.get(vids[f]['person'], '')}" for f in order)
    sep = "---|" * len(order)
    rf = lambda f: float(vids[f].get("ref_f_px") or cfg["ref_f"]["value"])      # noqa: E731
    fm = lambda v, nd=2: "-" if v is None else f"{v:.{nd}f}"      # noqa: E731
    fs = lambda v, nd=2: "-" if v is None else f"{v:+.{nd}f}"      # noqa: E731
    S = lambda f: res[f]["stats"]      # noqa: E731
    lab = {"total": "사슬로 옮긴 마커 - 실제 마커 (전체)", "nonplanar": "가장 잘 맞는 평면 변환 - 실제 마커 (손이 평면이 아닌 부분)", "chain_h": "사슬 - 가장 잘 맞는 평면 변환 (사슬의 평면 변환이 틀린 부분)"}
    L = ["# 123 광류 사슬 대 마커 summary", "",
         "**마커 영상 4개(122 의 저장된 점). 기준 프레임의 마커를 광류 사슬의 호모그래피로 옮긴 위치를 그 프레임의 실제 마커와 비교한다. 오차를 손이 평면이 아닌 부분과 사슬의 평면 변환이 틀린 부분으로 나누고, "
         "세 가지 점(마커 그대로 / 마커를 가장 잘 맞는 평면 변환으로 옮긴 것 / 마커를 사슬로 옮긴 것)으로 같은 추정을 돌린다. 판정 없음.**", "",
         "- 기준 f 는 이 파일에만. 거리는 관절 묶음(j16 중 마커가 있는 것)의 RMS, px. 손 방향 = 검출 손 방향, 사슬의 기준 프레임 대비.", "",
         "## 거리 RMS (px)", "", f"| 값 | {names} |", "|---|" + sep,
         "| 함께 쓰는 프레임 | " + " | ".join(str(S(f)["n_frames"]) for f in order) + " |", "| 마커 상자 긴 변 중앙값 (px) | " + " | ".join(fm(S(f)["box_long_px_median"], 0) for f in order) + " |"]
    for k in ("total", "nonplanar", "chain_h"):
        L.append(f"| {lab[k]} | " + " | ".join(f"{fm(S(f)['rms_px'][k])} ({fm(S(f)['rms_pct_box'][k])}%)" for f in order) + " |")
    for k in ("total", "nonplanar", "chain_h"):
        L += ["", f"### 손 방향 구간별: {lab[k]}", "", f"| 손 방향 (도) (프레임 수) | {names} |", "|---|" + sep]
        for b in S(order[0])["n_frames_by_angle"]:
            L.append(f"| {b} | " + " | ".join(f"{fm(S(f)['rms_px_by_angle'][k][b])} ({S(f)['n_frames_by_angle'][b]})" for f in order) + " |")
    L += ["", "### 관절 줄별: 사슬의 평면 변환이 틀린 부분 / 손이 평면이 아닌 부분", "", f"| 관절 줄 | {names} |", "|---|" + sep]
    for r in CDX_ROWS:
        L.append(f"| {r} | " + " | ".join(f"{fm(S(f)['rms_px_by_row']['chain_h'][r])} / {fm(S(f)['rms_px_by_row']['nonplanar'][r])}" for f in order) + " |")
    for key, title in (("length_pct_by_angle", "손 길이 (MCP 평균 -> 손끝 평균)"), ("width_pct_by_angle", "손 폭 (가장 바깥 두 MCP)")):
        L += ["", f"### {title}: 예측 / 실제 마커 - 1 (%, 중앙값), 사슬 / 가장 잘 맞는 평면 변환", "", f"| 손 방향 (도) | {names} |", "|---|" + sep]
        for b in S(order[0])["n_frames_by_angle"]:
            L.append(f"| {b} | " + " | ".join(f"{fs(S(f)[key]['chain'][b])} / {fs(S(f)[key]['planar'][b])}" for f in order) + " |")
    k11 = f"ramp_{cfg['ramps_deg'][0]:g}"
    vl = {"marker": "마커 그대로", "planar_marker": "마커를 가장 잘 맞는 평면 변환으로 옮김", "chain_on_marker": "마커를 사슬로 옮김"}

    def cell(f, v, key):
        d = res[f]["f"][v].get(key) or {}
        if "f" not in d:
            return "-"
        return f"{d['f']:.0f} ({100 * (d['f'] / rf(f) - 1):+.1f}%)" + ("*" if d.get("boundary") else "") + (f", {d['pitch_deg']:.1f}°" if key == "ramp_free" else "")
    for key, title in ((k11, f"경사 {cfg['ramps_deg'][0]:g}° 고정"), ("ramp_free", "경사 함께 추정")):
        L += ["", f"## f: {title}", "", f"| 점 | {names} |", "|---|" + sep]
        for v in ("marker", "planar_marker", "chain_on_marker"):
            L.append(f"| {vl[v]} | " + " | ".join(cell(f, v, key) for f in order) + " |")
    L += ["", "## 한계", "", "- 영상 4개(사람당 2개). 마커 중심의 검출 규칙에 따라 f 가 몇 %p 달라진다(121 대 122).", "- 사슬의 호모그래피는 저장된 사슬 점 21개에서 되찾았다(사슬 점은 호모그래피 하나로 옮긴 점이라 정확히 되찾아진다).",
          "- 내 손은 새끼 MCP(17) 마커가 없어 폭을 5 - 13 으로 쟀다(팀원은 5 - 17).", ""]
    return "\n".join(L) + "\n"



def main(argv=None):
    ap = argparse.ArgumentParser(description="광류 방식 원인 확인 실험")
    ap.add_argument("cmd", choices=["tracking-plane", "dorsum-corners", "two-plane", "joint-track-video", "geometry-synth", "pause-plane", "cotracker", "rotation-side", "points-video", "shape3d", "side-consistency", "shape-change", "pip-axis", "pause-plane2", "desk3d", "desk3d-contact", "stage-values", "stage-inputs", "cost-decompose", "lens-model", "repeatability", "marker-track", "marker-track2", "chain-vs-marker"])
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
              else ct_results_yaml(hf.resolve(a.run), own) if a.cmd == "cotracker" else rs_results_yaml(hf.resolve(a.run), own) if a.cmd == "rotation-side"
              else s3_results_yaml(hf.resolve(a.run), own) if a.cmd == "shape3d" else sc_results_yaml(hf.resolve(a.run), own) if a.cmd == "side-consistency"
              else sh_results_yaml(hf.resolve(a.run), own) if a.cmd == "shape-change"
              else sc_results_yaml(hf.resolve(a.run), own) if a.cmd == "pip-axis" else pq_results_yaml(hf.resolve(a.run), own) if a.cmd == "pause-plane2"
              else d3_results_yaml(hf.resolve(a.run), own) if a.cmd in ("desk3d", "desk3d-contact") else sv_results_yaml(hf.resolve(a.run), own) if a.cmd in ("stage-values", "stage-inputs") else cdx_results_yaml(hf.resolve(a.run), own) if a.cmd == "cost-decompose" else lm_results_yaml(hf.resolve(a.run), own) if a.cmd in ("lens-model", "repeatability", "marker-track", "marker-track2", "chain-vs-marker") else jt_results_yaml(hf.resolve(a.run), own) if a.cmd == "joint-track-video"
              else results_yaml(a.cmd, hf.resolve(a.run), own))
        return None
    cfg = rof.load_settings({**own, "joint_set": "j16"} if own.get("joint_ids") else own)      # joint_ids 를 직접 적으면 joint_set 은 이름일 뿐이라 기본 묶음으로 읽고 아래에서 바꿈
    cfg["joint_set"] = own["joint_set"]
    cfg["videos"] = cfg["videos"] + [dict(v) for v in own.get("extra_videos", [])]      # 기본 설정 목록에 없는 새 영상 (file, person, ref_f_px)
    if own.get("joint_ids"):                                           # 기본 설정의 관절 묶음(j16, j21) 대신 직접 적은 관절 번호 (joint_set 은 이름으로만 씀)
        cfg["joint_ids"] = [int(j) for j in own["joint_ids"]]
    cfg.update({k: own[k] for k in ("variants", "clahe", "two_plane", "joint_track", "render", "reference", "geometry_synth", "pause_plane", "cotracker", "rotation_side", "points_video", "shape3d", "side_consistency", "shape_change", "axis_base_joints", "pause_plane2", "desk3d", "stage_values", "stage_inputs", "cost_decompose", "lens_model", "repeatability", "marker_track", "chain_vs_marker") if k in own})
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
    job = {"tracking-plane": tp_job, "dorsum-corners": dc_job, "two-plane": tw_points, "joint-track-video": jt_job, "geometry-synth": rof.video_points, "pause-plane": pp_job, "rotation-side": rs_load, "points-video": pv_job, "side-consistency": sc_load, "shape-change": sh_job, "pip-axis": sc_load, "pause-plane2": pq_job, "stage-values": sv_job, "stage-inputs": si_job, "marker-track": mt_job, "marker-track2": mt2_job, "chain-vs-marker": cm_load}.get(a.cmd)
    vid = {}
    if a.cmd == "cotracker":                                           # GPU 작업 + 추정을 ct_run 이 한꺼번에
        res = ct_run(cfg, own, log, run_root)
        job = None
    elif a.cmd == "shape3d":                                           # 저장된 점만 읽음 (영상 처리 없음)
        res = s3_run(cfg, own, log)
        job = None
    elif a.cmd == "desk3d":
        res = d3_run(cfg, own, log)
        job = None
    elif a.cmd == "desk3d-contact":
        res = d3c_run(cfg, own, log)
        job = None
    elif a.cmd == "cost-decompose":
        res = cdx_run(cfg, own, log)
        job = None
    elif a.cmd == "lens-model":
        res = lm_run(cfg, own, log)
        job = None
    elif a.cmd == "repeatability":
        res = rp_run(cfg, own, log, run_root)
        job = None
    with ProcessPoolExecutor(max_workers=int(cfg["video_workers"]), mp_context=mp.get_context("spawn")) if job else _nullpool() as ex:
        for e, (x, lines) in (zip(cfg["videos"], ex.map(job, [(e, cfg, str(run_root)) for e in cfg["videos"]])) if job else []):
            vid[e["file"]] = x
            for s in lines:
                log(s)
    log(f"  추적: {time.time() - t0:.0f} s")
    if a.cmd == "cotracker":
        text_sum = ct_summary(res, cfg, own)
    elif a.cmd == "shape3d":
        text_sum = s3_summary(res, cfg, own)
    elif a.cmd in ("desk3d", "desk3d-contact"):
        text_sum = d3_summary(res, cfg, own)
    elif a.cmd == "cost-decompose":
        text_sum = cdx_summary(res, cfg, own)
    elif a.cmd == "lens-model":
        text_sum = lm_summary(res, cfg, own)
    elif a.cmd == "repeatability":
        text_sum = rp_summary(res, cfg, own)
    elif a.cmd == "points-video":
        res = vid
        text_sum = pv_summary(res, cfg)
    elif a.cmd == "stage-values":
        res = sv_run(cfg, vid, log)
        text_sum = sv_summary(res, cfg, own)
    elif a.cmd == "stage-inputs":
        res = si_run(cfg, vid, log)
        text_sum = si_summary(res, cfg, own)
    elif a.cmd == "marker-track":
        res = mt_run(cfg, vid, log)
        text_sum = mt_summary(res, cfg, own)
    elif a.cmd == "marker-track2":
        res = mt_run(cfg, vid, log)
        text_sum = mt2_summary(res, cfg, own)
    elif a.cmd == "chain-vs-marker":
        res = cm_run(cfg, vid, log)
        text_sum = cm_summary(res, cfg, own)
    elif a.cmd == "pip-axis":
        res = pa_run(cfg, vid, log)
        text_sum = sc_summary(res, cfg, own)
    elif a.cmd == "pause-plane2":
        res = vid
        text_sum = pq_summary(res, cfg, own)
    elif a.cmd == "shape-change":
        res = vid
        text_sum = sh_summary(res, cfg, own)
    elif a.cmd == "side-consistency":
        res = sc_run(cfg, vid, log)
        text_sum = sc_summary(res, cfg, own)
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
