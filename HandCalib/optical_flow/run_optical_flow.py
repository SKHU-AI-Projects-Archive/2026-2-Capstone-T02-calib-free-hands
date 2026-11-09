"""광류(KLT) 방식으로 f 를 추정하는 실행 파일. run.py 를 거치지 않는다.

  광류 방식 : python optical_flow/run_optical_flow.py --config configs/89_optical_flow_entry.yaml
  검출기 방식: python run.py <명령> ...   (예: detected-joints = 실험 81, free-shape3d-detectors = 실험 51)

한 영상의 흐름:
  1. 프레임 추출 (30 fps)                       - handfocal.fresp_prepare           (두 방식 공통)
  2. MediaPipe 손 검출과 추적                    - handfocal.fresp_detect_many, track  (두 방식 공통)
  3. 광류 추적 사슬로 추적점 만들기              - optical_flow.track_hand            (광류 방식만)
  4. free_template 한 번 (여러 시작점 풀이의 시작점 중 하나) - handfocal.tl_template_job
  5. 책상 운동 추정기 (실험 74b): 경사 고정, 경사 함께 추정(시작 경사 여러 개 중 비용 최소) - handfocal.ms2_job
설정은 실험 74b 의 설정 사슬(74b -> 74 -> 56 / 60 / 67 / 73 -> 53 -> 49 / 47)을 그대로 읽는다(값을 복사하지 않음).
결과는 runs/<이름>/<UTC>/ 에 results.json, summary.md(기준 f 와 오차는 여기에만), metadata.json, log.txt 로 남긴다.
"""
import argparse
import copy
import datetime as dt
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import handfocal as hf                             # noqa: E402  (프레임·검출·추정기: 두 방식 공통)
import optical_flow as of                          # noqa: E402  (광류 추적: 이 방식만)


def log_to(path: Path):
    """화면과 파일에 함께 쓰는 로그 함수."""
    fh = path.open("a")

    def log(msg: str = ""):
        print(msg, flush=True)
        fh.write(msg + "\n")
        fh.flush()
    log.close = fh.close
    return log


def load_settings(own: dict) -> dict:
    """74b 의 설정 사슬을 읽어 한 dict 로 만든다 (run.ck_cfg 와 같은 결과).
    - settings_config(74b) -> settings_config(74): 경사 목록, 함께 추정 시작 경사
    - 74 의 rules_config(56): 영상 목록(기존 폰), tracking(광류 설정), base_config(53)
    - 53 의 base_config(49) / mm_config(47): pipeline(프레임·추정 설정), MediaPipe 설정
    - 74 의 estimator_config(67): 관절 묶음(j16 등),  new_phone_config(73): 새 폰 영상과 그 기준 f
    """
    rd = lambda p: yaml.safe_load(hf.resolve(p).read_text())      # noqa: E731
    o74b = rd(own["settings_config"])
    o74 = rd(o74b["settings_config"])
    r56, r67, r73 = rd(o74["rules_config"]), rd(o74["estimator_config"]), rd(o74["new_phone_config"])
    b53 = rd(r56["base_config"])
    b49, m47 = rd(b53["base_config"]), rd(b53["mm_config"])
    ds = {"mediapipe_hands": b49["detector_settings"]["mediapipe_hands"], **{k: v for k, v in m47["detector_settings"].items() if k != "mediapipe_hands"}}
    videos = [{**e, "person": "old"} for e in r56["videos"]] + [{**e, "ref_f_px": float(r73["ref_f"]["value"])} for e in r73["videos"]]
    if own.get("videos"):
        videos = [e for e in videos if e["file"] in own["videos"]]
    return {**b53, "pipeline": b49["pipeline"], "detector_settings": {d: ds[d] for d in b53["detectors"]}, "videos": videos, "tracking": r56["tracking"],
            "joint_ids": [int(j) for j in r67["joint_sets"][own["joint_set"]]], "joint_set": own["joint_set"],
            "ramps_deg": [float(r) for r in own["ramps_deg"]], "free_starts_deg": [float(r) for r in o74["free_starts_deg"]],
            "reference": own.get("reference", REFERENCE_DEFAULT),
            **{k: own[k] for k in ("experiment", "workers", "video_workers", "output")}}


# 기준 프레임 규칙. 처음(실험 56)에는 "검출 범위의 한가운데 시간"이었는데, 손 자세를 보지 않아서 처음 자세로 돌아오지 않는 영상(팀원 영상)에서는 손을 크게 돌려 둔
# 순간이 기준이 됐다. 모든 추적점이 그 순간의 검출 관절에서 출발하므로, 손을 돌리기 전 첫 정지 구간(손이 가만히 있고 정면에 가까워 검출이 가장 안정적인 때)으로 바꾼다.
REFERENCE_DEFAULT = {"mode": "first_pause", "pause_gap_frames": 15, "min_pause_frames": 5}


def choose_reference_pose(tr: dict, size_wh, cfg: dict) -> int:
    """기준 프레임.
    mode = "first_pause": 손을 돌리기 전 첫 정지 구간의 가운데 검출 프레임. 정지 = 기본 설정의 selection 규칙(손 상자 중심의 움직임 < still_frac),
                          정지 구간 = 간격 pause_gap_frames 이하로 이어진 min_pause_frames 개 이상의 정지 프레임. 정지 구간이 없으면 "middle".
    mode = "middle": 검출 범위의 한가운데 (실험 56 의 규칙, optical_flow.choose_reference)."""
    rule = cfg.get("reference") or REFERENCE_DEFAULT
    if rule["mode"] == "middle":
        return of.choose_reference(tr["present"])
    motion = hf.od_select(tr["kp"], tr["mask"], tr["present"], tr["box"], tuple(size_wh), cfg["selection"])["motion"]
    still = np.isfinite(motion) & (motion < float(cfg["selection"]["still_frac"])) & tr["present"]
    pauses = [np.asarray(g) for g in hf.hp_groups(np.flatnonzero(still), int(rule["pause_gap_frames"])) if len(g) >= int(rule["min_pause_frames"])]
    if not pauses:
        return of.choose_reference(tr["present"])
    g = pauses[0]
    return int(g[np.argmin(np.abs(g - (g[0] + g[-1]) / 2))])


def video_points(args):
    """한 영상: 프레임 추출 -> MediaPipe 검출·추적 -> 광류 추적점 (작업 프로세스에서 실행). 반환: (결과, 로그 줄)."""
    e, cfg, run_root = args
    import cv2
    cv2.setNumThreads(1)
    lines = []
    pc = copy.deepcopy(cfg["pipeline"])
    pc["frames"]["detect_fps"] = int(cfg["tracking"]["fps"])
    v = e["file"]
    tmp = Path(run_root) / f"_frames_{Path(v).stem}"
    tmp.mkdir(exist_ok=True)
    try:
        prep = hf.fresp_prepare((v, cfg["input_dir"], pc, {}, str(tmp)))
        frames, (W, H) = prep["frames"], prep["size"]
        times = np.array([r["time_s"] for r in frames])
        npz = hf.fresp_detect_many([((v, "orig"), frames)], cfg["detector_settings"]["mediapipe_hands"], Path(run_root) / "detection_cache", 1, lines.append)[(v, "orig")]
        d = hf.load_detections(Path(npz))
        hf.check_alignment(frames, d)
        p2 = copy.deepcopy(pc)
        p2["tracking"]["conf_min"] = float(cfg["detector_settings"]["mediapipe_hands"]["conf_min"])
        tr = hf.track(d, times, (W, H), p2["tracking"], front_start_s=float(p2["phases"]["front_skip_s"]))
        out = of.track_hand([r["png"] for r in frames], tr["kp"], tr["present"], tr["box"], (W, H), cfg["tracking"], ref=choose_reference_pose(tr, (W, H), cfg))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out["size"] = (W, H)
    lines.append(f"  {v}: 프레임 {len(times)}, 기준 프레임 {out['ref']} ({times[out['ref']]:.1f} s), 끊김 {out['breaks']}, 사용 {int(out['use'].sum())}")
    return out, lines


def estimate_args(cfg: dict, x: dict, A: dict = None) -> tuple:
    """handfocal.tl_template_job / ms2_job 의 입력 (A = free_template 결과가 있으면 붙임)."""
    a = (x["kp"], np.isfinite(x["kp"]).all(2), np.flatnonzero(x["use"]), x["box"], tuple(x["size"]), cfg["pipeline"]["estimate"], cfg["joint_ids"])
    return a if A is None else a + (A,)


def results_yaml(run_dir: Path, own: dict, path: Path) -> Path:
    """값만 (기준 f, 오차 없음): 영상별 기준 프레임, 사용 프레임, 추정마다 f·경사·RMS 등."""
    r = json.loads((run_dir / "results.json").read_text())["real"]
    m = json.loads((run_dir / "metadata.json").read_text())
    rv = lambda x: x if not isinstance(x, float) else (round(x, 3) if np.isfinite(x) else None)      # noqa: E731
    keys = ("f", "rms_px", "pitch_deg", "roll_deg", "desk_tilt_deg", "boundary", "start")
    out = {"experiment": own["experiment"]["name"], "run": str(run_dir.resolve().relative_to(hf.PROJECT_ROOT)), "code_commit": m["git"]["commit"][:7],
           "git_dirty_at_run": m["git"]["git_dirty"], "videos": {}}
    for f, d in r.items():
        out["videos"][f] = {k: (rv(v) if not isinstance(v, dict) else {kk: rv(v[kk]) for kk in keys if kk in v} if "f" in v else v) for k, v in d.items()}
    path.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="광류(KLT) 추적점 + 책상 운동 추정기(74b)로 f 추정")
    ap.add_argument("--config", default=str(HERE.parent / "configs/89_optical_flow_entry.yaml"))
    ap.add_argument("--out", default=None, help="결과 폴더 (기본: output.root/<UTC>)")
    ap.add_argument("--videos", type=int, default=None, help="시험 실행: 앞의 N 개 영상만")
    ap.add_argument("--results-yaml", default=None, help="이미 끝난 실행 폴더의 결과 yaml 을 output.results_yaml 에 쓴다")
    a = ap.parse_args(argv)
    text = Path(a.config).read_text()
    own = yaml.safe_load(text)
    if a.results_yaml:
        print(results_yaml(hf.resolve(a.results_yaml), own, hf.resolve(own["output"]["results_yaml"])))
        return None
    cfg = load_settings(own)
    if a.videos:
        cfg["videos"] = cfg["videos"][:a.videos]
    run_root = Path(a.out).expanduser() if a.out else hf.resolve(cfg["output"]["root"]) / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root.mkdir(parents=True)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    log = log_to(run_root / "log.txt")
    log(f"광류 방식: 영상 {len(cfg['videos'])}개, 점 {cfg['joint_set']}, 기준 프레임 {cfg['reference']['mode']}, 경사 {cfg['ramps_deg']} 고정 + 함께 추정(시작 {cfg['free_starts_deg']}) → {run_root}")
    t0 = time.time()
    if own.get("detection_cache_from"):                                # 검출 결과 재사용 (같은 검출기 설정일 때만 맞음)
        shutil.copytree(hf.resolve(own["detection_cache_from"]), run_root / "detection_cache")
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    pts = {}
    with ProcessPoolExecutor(max_workers=int(cfg["video_workers"]), mp_context=mp.get_context("spawn")) as ex:
        for e, (x, lines) in zip(cfg["videos"], ex.map(video_points, [(e, cfg, str(run_root)) for e in cfg["videos"]])):
            pts[e["file"]] = x
            for s in lines:
                log(s)
    log(f"  추적: {time.time() - t0:.0f} s")
    files = [e["file"] for e in cfg["videos"]]
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        fts = dict(zip(files, ex.map(hf.tl_template_job, [estimate_args(cfg, pts[f]) for f in files], chunksize=1)))
    todo = [(f"ramp_{r:g}", "ramp_fixed", r) for r in cfg["ramps_deg"]] + [(f"free_{r:g}", "ramp_free", r) for r in cfg["free_starts_deg"]]
    res, jobs, keys = {}, [], []
    for f in files:
        x = pts[f]
        grid = hf.f_grid(cfg["pipeline"]["estimate"], max(*x["size"]))
        A = fts[f].get("A")
        res[f] = {"ref": x["ref"], "breaks": x["breaks"], "n_use": int(x["use"].sum()),
                  "free_template": {"failed": fts[f].get("failed")} if A is None else
                  {"f": float(A["f"]), "boundary": hf.od_boundary(float(A["f"]), grid), "rms_px": float((A.get("ba") or {}).get("rms_px", np.nan)), "cost": float("inf")}}
        if A is None:
            continue
        for mk, mode, r in todo:
            jobs.append(estimate_args(cfg, x, A) + (mode, r))
            keys.append((f, mk))
    log(f"  추정 {len(jobs)}건")
    with ProcessPoolExecutor(max_workers=int(cfg["workers"]), mp_context=mp.get_context("spawn")) as ex:
        for (f, mk), o in zip(keys, ex.map(hf.ms2_job, jobs, chunksize=1)):
            res[f][mk] = o
    log(f"  추정 끝: {time.time() - t0:.0f} s")
    (run_root / "results.json").write_text(json.dumps(hf.jsonable({"real": res}), ensure_ascii=False, indent=1))
    (run_root / "summary.md").write_text(summary(res, cfg, own))
    meta = hf.provenance_build(text, a.config, {"code_89": own.get("code_89"), "elapsed_s": time.time() - t0})
    (run_root / "metadata.json").write_text(json.dumps(hf.jsonable(meta), ensure_ascii=False, indent=1))
    log(f"완료 {time.time() - t0:.0f} s → {run_root}")
    log.close()
    return run_root, res


def summary(res: dict, cfg: dict, own: dict) -> str:
    """영상별 f 와 기준 f 대비 오차 (기준 f 는 이 파일에만). 비교용 이전 실행(compare_run)이 있으면 같은 추정의 f 를 옆에 적는다."""
    ref_old = float(cfg["ref_f"]["value"])                            # 기존 폰 기준 f (실험 53 설정). 새 폰 영상은 영상마다 ref_f_px
    prev = json.loads((hf.resolve(own["compare_run"]) / "results.json").read_text())["real"] if own.get("compare_run") else {}
    js = cfg["joint_set"]
    best = lambda d: min([d[k] for k in d if k.startswith("free_") and isinstance(d[k], dict) and d[k].get("cost") not in (None, float("inf"))], key=lambda x: x["cost"], default=None)      # noqa: E731
    keys = ["free_template"] + [f"ramp_{r:g}" for r in cfg["ramps_deg"]]
    L = ["# 광류 방식 summary", "", f"- 점: {js}, 경사 고정 {cfg['ramps_deg']}, 함께 추정 시작 {cfg['free_starts_deg']} 중 비용 최소. 기준 f 는 이 파일에만 적는다. * = f 탐색 범위 끝.",
         f"- 비교: {own.get('compare_run') or '-'} (같은 추정의 f, 차이 px)", "",
         "| 영상 | 묶음 | 기준 f | 추정 | f (오차) | 비교 실행 f | 차이 (px) |", "|---|---|---|---|---|---|---|"]
    for e in cfg["videos"]:
        r = res.get(e["file"]) or {}
        rf = float(e.get("ref_f_px") or ref_old)
        p = prev.get(f"{e['file']}|{js}") or {}
        for k in keys + ["free_best"]:
            x, y = (best(r), best(p)) if k == "free_best" else (r.get(k), p.get(k))
            fx = "-" if not x or "f" not in x else f"{x['f']:.1f} ({100 * (x['f'] / rf - 1):+.1f}%)" + ("*" if x.get("boundary") else "") + (f", 경사 {x['pitch_deg']:.1f}°" if k == "free_best" else "")
            fy = "-" if not y or "f" not in y else f"{y['f']:.1f}"
            df = "-" if "-" in (fx[:1], fy[:1]) else f"{x['f'] - y['f']:+.3f}"
            L.append(f"| {e['file']} | {e['person']} | {rf:g} | {'함께 추정' if k == 'free_best' else k} | {fx} | {fy} | {df} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
