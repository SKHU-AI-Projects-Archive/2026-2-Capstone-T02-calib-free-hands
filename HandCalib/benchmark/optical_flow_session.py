"""Run the optical-flow f pipeline on the hand videos of benchmark sessions.

Runs in the main project venv (mediapipe), not in the benchmark venv:

    python benchmark/optical_flow_session.py --config configs/131_benchmark_optical_flow.yaml
    python benchmark/optical_flow_session.py --config ... --videos 1          # trial: first video only
    python benchmark/optical_flow_session.py --config ... --results-yaml runs/131_.../<run>

The per-video work is optical_flow/experiments.py unchanged:
    rv2_job  frames (30 fps), MediaPipe, track, first-pause reference frame, then for each
             chain variant (region_variants.variants) the forward/backward homography chain
    rv_run   free_template -> desk-motion estimator with the hand tilt fixed (ramps_deg)
Variants used here: chain (the current chain, all 21 joints) and no_thumb (feature mask
without the thumb, the best variant of experiments 127-129).

Results are kept apart from session.json (videos[].f holds the calibrator.py value):
    runs/<output.root>/<UTC>/results.json    f etc. per video and variant (no reference f)
    runs/<output.root>/<UTC>/summary.md      errors against reference_f, next to calibrator.py
    results/<...>.yaml                       values only (no reference f), via --results-yaml

Session videos are linked (symlinks) into the run folder under unique names
(<session>__<file>) so that frame folders of different sessions cannot collide.
"""
import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import handfocal as hf                                    # noqa: E402
from optical_flow import experiments as ex                # noqa: E402
from optical_flow import run_optical_flow as rof          # noqa: E402
from session_rules import HAND_STEMS, check_session, find_sessions, find_video   # noqa: E402


def session_videos(own: dict, link_dir: Path) -> list:
    out = []
    for session in find_sessions(own["data_root"]):
        if own.get("sessions") and session.name not in own["sessions"]:
            continue
        errors, _, meta = check_session(session, probe_videos=False)
        if errors:
            print(f"skip {session.name}: {len(errors)} guide errors")
            continue
        calib = {v["file"]: v.get("f") for v in meta.get("videos") or []}
        for stem in HAND_STEMS:
            path = find_video(session, stem)
            if path is None:
                continue
            name = f"{session.name}__{path.name}"
            (link_dir / name).symlink_to(path.resolve())
            out.append({"file": name, "session": session.name, "video": path.name, "motion": stem.split("_")[1],
                        "ref_f_px": float(meta["reference_f"]), "calibrator_f": calib.get(path.name)})
    return out


def err(f, ref):
    return 100.0 * (f / ref - 1.0) if f is not None and np.isfinite(f) else np.nan


def stats(e: list) -> str:
    e = np.asarray([x for x in e if np.isfinite(x)])
    if e.size == 0:
        return "| - | - | - | - | - |"
    a = np.abs(e)
    return (f"| {e.size} | {np.median(a):.1f}% | {np.median(e):+.1f}% | {int((a <= 10).sum())}/{e.size} | "
            f"{int((a <= 20).sum())}/{e.size} |")


def summary(res: dict, cfg: dict, own: dict) -> str:
    var = list(own["region_variants"]["variants"])
    k = f"ramp_{cfg['ramps_deg'][0]:g}"
    vids = cfg["videos"]
    f_of = lambda e, nm: (res.get(e["file"], {}).get(nm, {}).get(k) or {}).get("f")      # noqa: E731
    methods = [("calibrator.py", lambda e: e["calibrator_f"])] + [(f"optical flow {nm}", (lambda nm: lambda e: f_of(e, nm))(nm)) for nm in var]
    L = [f"# {own['experiment']['name']} summary", "",
         f"**벤치마크 세션의 손 영상 {len(vids)}개에 광류 파이프라인(첫 정지 구간 기준, {cfg['joint_set']}, 경사 {cfg['ramps_deg'][0]:g}° 고정)을 돌리고 calibrator.py(검출기, 같은 세션의 videos[].f)와 나란히 적는다. 판정 없음.**", "",
         "- 광류 변형: " + " ; ".join(f"{nm} = {'관절 21개 전부(지금 사슬)' if v['mask'] is None else '관절 ' + str(v['mask'])}" for nm, v in own["region_variants"]["variants"].items()),
         "- 기준 f 는 세션의 reference_f(체커보드 before/after 평균)이고 이 파일에만 적는다. * = f 탐색 범위 끝.",
         "- 광류 파이프라인의 추정 모델은 손을 책상에 대고 돌리는 영상을 전제로 만들었다. 가이드 영상은 손을 공중에 들고 찍으므로 모델 전제와 다르다.", "",
         "## 방법별 요약", "", "| 방법 | 영상 수 | 절댓값 오차 중앙값 | 오차 중앙값 | ±10% 안 | ±20% 안 |", "|---|---|---|---|---|---|"]
    for name, fn in methods:
        L.append(f"| {name} " + stats([err(fn(e), e["ref_f_px"]) for e in vids]))
    L += ["", "## 동작별 절댓값 오차 중앙값", "", "| 동작 | " + " | ".join(n for n, _ in methods) + " |", "|---|" + "---|" * len(methods)]
    for m in ("twinkle", "nod", "circle", "mix"):
        es = [e for e in vids if e["motion"] == m]
        if not es:
            continue
        cells = []
        for _, fn in methods:
            a = [abs(err(fn(e), e["ref_f_px"])) for e in es]
            a = [x for x in a if np.isfinite(x)]
            cells.append(f"{np.median(a):.1f}%" if a else "-")
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L += ["", "## 영상별 f (오차)", "",
          "| 세션 | 영상 | 기준 f | " + " | ".join(n for n, _ in methods) + " | 끊김 (변형별) | 사용 프레임 (변형별) |",
          "|---|---|---|" + "---|" * (len(methods) + 2)]
    for e in vids:
        cells = []
        for name, fn in methods:
            f = fn(e)
            d = (res.get(e["file"], {}).get(name.replace("optical flow ", ""), {}).get(k) or {}) if name != "calibrator.py" else {}
            cells.append("-" if f is None else f"{f:.0f} ({err(f, e['ref_f_px']):+.1f}%)" + ("*" if d.get("boundary") else ""))
        info = [res.get(e["file"], {}).get(nm, {}).get("info", {}) for nm in var]
        L.append(f"| {e['session']} | {e['video']} | {e['ref_f_px']:g} | " + " | ".join(cells) + " | "
                 + " / ".join(str(i.get("breaks", "-")) for i in info) + " | "
                 + " / ".join(f"{i.get('n_use', '-')}/{i.get('n_frames', '-')}" for i in info) + " |")
    L += ["", "## 한계", "", "- 세션 하나(한 사람, 한 기기, 한 설정)의 영상 8개뿐이다.", "- 광류는 경사 고정만 돌렸다(함께 추정 없음).", ""]
    return "\n".join(L) + "\n"


def results_yaml(run_dir: Path, own: dict) -> Path:
    r = json.loads((run_dir / "results.json").read_text())["real"]
    m = json.loads((run_dir / "metadata.json").read_text())
    rv = lambda x: x if not isinstance(x, float) else (round(x, 4) if np.isfinite(x) else None)      # noqa: E731
    deep = lambda d: {k: (deep(v) if isinstance(v, dict) else [rv(t) for t in v] if isinstance(v, list) else rv(v)) for k, v in d.items()}      # noqa: E731
    out = {"experiment": own["experiment"]["name"], "run": str(run_dir.resolve().relative_to(hf.PROJECT_ROOT)),
           "code_commit": m["git"]["commit"][:7], "git_dirty_at_run": m["git"]["git_dirty"],
           "videos": {f: deep(d) for f, d in r.items()}}
    p = hf.resolve(own["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--videos", type=int, default=None, help="trial: first N videos only")
    ap.add_argument("--out", default=None)
    ap.add_argument("--results-yaml", default=None, metavar="RUN")
    a = ap.parse_args()
    text = Path(a.config).read_text()
    own = yaml.safe_load(text)
    if a.results_yaml:
        print(results_yaml(hf.resolve(a.results_yaml), own))
        return 0

    cfg = rof.load_settings(own)
    cfg["region_variants"] = own["region_variants"]
    run_root = Path(a.out).expanduser() if a.out else hf.resolve(own["output"]["root"]) / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root.mkdir(parents=True)
    link_dir = run_root / "videos"
    link_dir.mkdir()
    cfg["input_dir"] = str(link_dir)
    cfg["videos"] = session_videos(own, link_dir)
    if a.videos:
        cfg["videos"] = cfg["videos"][:a.videos]
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    log = rof.log_to(run_root / "log.txt")
    log(f"benchmark optical flow: 영상 {len(cfg['videos'])}개 → {run_root}")
    t0 = time.time()
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    vid = {}
    with ProcessPoolExecutor(max_workers=int(cfg["video_workers"]), mp_context=mp.get_context("spawn")) as pool:
        for e, (x, lines) in zip(cfg["videos"], pool.map(ex.rv2_job, [(e, cfg, str(run_root)) for e in cfg["videos"]])):
            vid[e["file"]] = x
            for s in lines:
                log(s)
    log(f"  추적: {time.time() - t0:.0f} s")
    res = ex.rv_run(cfg, vid, log)
    (run_root / "results.json").write_text(json.dumps(hf.jsonable({"real": res}), ensure_ascii=False, indent=1))
    (run_root / "summary.md").write_text(summary(res, cfg, own))
    meta = hf.provenance_build(text, a.config, {"code": own.get("code"), "elapsed_s": time.time() - t0,
                                                "videos": [{k: e[k] for k in ("file", "session", "video", "motion")} for e in cfg["videos"]]})
    (run_root / "metadata.json").write_text(json.dumps(hf.jsonable(meta), ensure_ascii=False, indent=1))
    log(f"완료 {time.time() - t0:.0f} s → {run_root}")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
