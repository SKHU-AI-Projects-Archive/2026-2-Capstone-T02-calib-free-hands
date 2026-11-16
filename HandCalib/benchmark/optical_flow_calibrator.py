"""Optical-flow tracked hand points, scored with calibrator.py's own estimator.

Runs in the main project venv (mediapipe):

    python benchmark/optical_flow_calibrator.py --config configs/132_benchmark_flow_calibrator.yaml
    python benchmark/optical_flow_calibrator.py --config ... --videos 1
    python benchmark/optical_flow_calibrator.py --config ... --results-yaml runs/132_.../<run>

Only the source of the 21 hand points changes; the estimator is Calibrator.estimate from
calibrator.py, unchanged (Procrustes-mean planar template, frames above the median
homography perspective score, cv2.calibrateCamera with fixed principal point, fx = fy,
no distortion). The detector points (RTMW, Calibrator("hand")) are the benchmark values
already in session.json videos[].f.

Point sources per variant (optical_flow/experiments.py rv2_job, unchanged): MediaPipe
joints of the first-pause reference frame carried to every frame by the forward/backward
homography chain; frames where the chain is valid and the box is off the image edge.
calibrator.py imports rtmlib at module level; the estimator does not use it, so a stub
module is put in sys.modules when rtmlib is not installed (this venv).
"""
import argparse
import datetime as dt
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

if "rtmlib" not in sys.modules:
    try:
        import rtmlib  # noqa: F401
    except ImportError:
        sys.modules["rtmlib"] = types.SimpleNamespace(Wholebody=None)

import handfocal as hf                                    # noqa: E402
from calibrator import Calibrator                         # noqa: E402
from optical_flow import experiments as ex                # noqa: E402
from optical_flow import run_optical_flow as rof          # noqa: E402
from optical_flow_session import err, session_videos, stats   # noqa: E402


def calibrator_f(kp: np.ndarray, use: np.ndarray, size) -> dict:
    """Calibrator.estimate on the frames in use (all 21 points finite)."""
    ok = np.asarray(use, bool) & np.isfinite(kp).all(axis=(1, 2))
    cal = Calibrator("checkerboard")          # no model is loaded for this target
    cal.target = "hand"
    cal.size = tuple(int(s) for s in size)
    cal.observations = [np.float32(p) for p in kp[ok]]
    out = {"n_frames": int(len(kp)), "n_points_frames": int(ok.sum())}
    try:
        out["f"] = float(cal.estimate())
    except Exception as e:                    # calibrator.py ends with an error when too few frames
        out["failed"] = f"{type(e).__name__}: {e}"[:200]
    return out


def summary(res: dict, cfg: dict, own: dict) -> str:
    var = list(own["region_variants"]["variants"])
    vids = cfg["videos"]
    methods = [("calibrator.py (RTMW 검출)", lambda e: e["calibrator_f"])] + \
              [(f"광류 {nm}", (lambda nm: lambda e: res.get(e["file"], {}).get(nm, {}).get("f"))(nm)) for nm in var]
    L = [f"# {own['experiment']['name']} summary", "",
         f"**벤치마크 세션의 손 영상 {len(vids)}개. 손 점 21개를 어디서 얻는지만 다르고, f 는 모두 calibrator.py 의 Calibrator.estimate(수정 없음)로 구한다. 판정 없음.**", "",
         "- calibrator.py (RTMW 검출): 프레임마다 rtmlib Wholebody 오른손 21점. 값은 같은 세션 session.json 의 videos[].f (벤치마크 실행).",
         "- 광류 변형: 첫 정지 구간 기준 프레임의 MediaPipe 관절을 앞뒤 호모그래피 사슬로 모든 프레임에 옮긴 21점. "
         + " ; ".join(f"{nm} = 특징점 영역 {'관절 21개 전부(지금 사슬)' if v['mask'] is None else '관절 ' + str(v['mask'])}" for nm, v in own["region_variants"]["variants"].items()),
         "- 기준 f 는 세션의 reference_f(체커보드 before/after 평균)이고 이 파일에만 적는다.", "",
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
          "| 세션 | 영상 | 기준 f | " + " | ".join(n for n, _ in methods) + " | 광류 사용 프레임 (변형별) | 사슬 끊김 (변형별) |",
          "|---|---|---|" + "---|" * (len(methods) + 2)]
    for e in vids:
        cells = []
        for _, fn in methods:
            f = fn(e)
            cells.append("-" if f is None else f"{f:.0f} ({err(f, e['ref_f_px']):+.1f}%)")
        r = res.get(e["file"], {})
        L.append(f"| {e['session']} | {e['video']} | {e['ref_f_px']:g} | " + " | ".join(cells) + " | "
                 + " / ".join(f"{r.get(nm, {}).get('n_points_frames', '-')}/{r.get(nm, {}).get('n_frames', '-')}" for nm in var) + " | "
                 + " / ".join(str(r.get(nm, {}).get("breaks", "-")) for nm in var) + " |")
    L += ["", "## 한계", "", "- 세션 하나(한 사람, 한 기기, 한 설정)의 영상 8개뿐이다.",
          "- 검출 점은 RTMW(calibrator.py), 광류의 기준 관절은 MediaPipe 라서 두 방법은 검출기도 다르다.", ""]
    return "\n".join(L) + "\n"


def results_yaml(run_dir: Path, own: dict) -> Path:
    r = json.loads((run_dir / "results.json").read_text())["real"]
    m = json.loads((run_dir / "metadata.json").read_text())
    rv = lambda x: x if not isinstance(x, float) else (round(x, 4) if np.isfinite(x) else None)      # noqa: E731
    out = {"experiment": own["experiment"]["name"], "run": str(run_dir.resolve().relative_to(hf.PROJECT_ROOT)),
           "code_commit": m["git"]["commit"][:7], "git_dirty_at_run": m["git"]["git_dirty"],
           "videos": {f: {nm: {k: rv(v) for k, v in d.items()} for nm, d in x.items()} for f, x in r.items()}}
    p = hf.resolve(own["output"]["results_yaml"])
    p.write_text(yaml.safe_dump(hf.jsonable(out), allow_unicode=True, sort_keys=False))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--videos", type=int, default=None)
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
    log(f"benchmark optical flow + calibrator estimator: 영상 {len(cfg['videos'])}개 → {run_root}")
    t0 = time.time()
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    res = {}
    with ProcessPoolExecutor(max_workers=int(cfg["video_workers"]), mp_context=mp.get_context("spawn")) as pool:
        for e, (x, lines) in zip(cfg["videos"], pool.map(ex.rv2_job, [(e, cfg, str(run_root)) for e in cfg["videos"]])):
            for s in lines:
                log(s)
            res[e["file"]] = {}
            for nm, d in x.items():
                o = calibrator_f(d["kp"], d["use"], d["size"])
                o.update({"breaks": d["info"]["breaks"], "ref_frame": d["ref"]})
                res[e["file"]][nm] = o
                log(f"  {e['file']} {nm}: f = {o.get('f', o.get('failed'))} ({o['n_points_frames']}/{o['n_frames']} 프레임)")
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
