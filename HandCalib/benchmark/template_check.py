"""135: is the calibrator.py hand template squashed by the motion?

    python template_check.py --config ../configs/135_template_check.json
    python template_check.py --config ... --videos 1

Calibrator.template (hand) is the Procrustes mean of the 2D hand shape over ALL detected
frames. In nod (fingertips tilted toward/away from the camera) most frames are foreshortened
along the fingers, so the mean shape is shorter along the fingers than the real frontal hand;
the planar target is then not the real one (like a checkerboard with non-square cells).

Per hand video of the valid benchmark sessions:
    base     Calibrator.estimate unchanged (must reproduce the benchmark f)
    frontal  the same estimate, but the template is the Procrustes mean of the frames of the
             first frontal_seconds of the video only (the guide: start frontal, hold 2 s);
             Calibrator.template itself is reused on that subset, nothing else changes.
Shape check: aspect = |wrist(0) - middle tip(12)| / |index MCP(5) - pinky MCP(17)| of each template.

The RTMW right-hand points (21 x 2) of every frame are saved in the run folder
(keypoints/<session>__<video>.npz: frame, points, size) so later analyses need no detection.
Outputs: run folder summary.md (with reference f), results.json (no reference f).
"""
import argparse
import os
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from calibrator import Calibrator
from run_benchmark import environment, git_state, thermal_guard
from session_rules import HAND_STEMS, check_session, find_sessions, find_video


class TemplateFrom(Calibrator):
    """Calibrator whose hand template is built from a given subset of shapes (same Procrustes code)."""

    def __init__(self, subset):
        super().__init__("checkerboard")      # no model needed; switched to hand below
        self.target = "hand"
        self.subset = np.float32(subset)

    def template(self, points):
        return super().template(self.subset)


def aspect(T):
    return float(np.linalg.norm(T[0] - T[12]) / np.linalg.norm(T[5] - T[17]))


def detect_all(hand, path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx, pts, i, size = [], [], 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        size = frame.shape[1::-1]
        p = hand.detect(frame)
        if p is not None:
            idx.append(i)
            pts.append(p)
        i += 1
    cap.release()
    return np.array(idx), np.float32(pts), tuple(size), float(fps), i


def estimate(cal, pts, size):
    cal.size = size
    cal.observations = list(pts)
    try:
        return float(cal.estimate()), cal.template(np.float32(pts))
    except Exception as e:
        return None, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--videos", type=int, default=None)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    repo = Path(__file__).resolve().parent.parent
    out = repo / cfg["output_dir"] / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (out / "keypoints").mkdir(parents=True)
    info = {"config": cfg, "git": git_state(repo), "environment": environment(), "started": datetime.now(timezone.utc).isoformat()}

    todo = []
    for s in find_sessions(cfg["data_root"]):
        if cfg.get("sessions") and s.name not in cfg["sessions"]:
            continue
        errors, _, meta = check_session(s, probe_videos=False)
        if errors:
            print(f"skip {s.name}: {len(errors)} errors")
            continue
        bench = {v["file"]: v.get("f") for v in meta["videos"]}
        for stem in HAND_STEMS:
            p = find_video(s, stem)
            if p is not None:
                todo.append((s.name, p, float(meta["reference_f"]), bench.get(p.name)))
    if args.videos:
        todo = todo[:args.videos]

    hand = Calibrator("hand")
    res, rows = {}, []
    for k, (sname, path, ref, fb) in enumerate(todo, 1):
        thermal_guard(cfg)
        t0 = time.time()
        idx, pts, size, fps, n = detect_all(hand, path)
        np.savez_compressed(out / "keypoints" / f"{sname}__{path.stem}.npz", frame=idx, points=pts, size=np.array(size), fps=fps, n_frames=n)
        base = Calibrator("checkerboard")
        base.target = "hand"
        f_base, T_base = estimate(base, pts, size)
        front = idx < int(round(cfg["frontal_seconds"] * fps))
        r = {"n_frames": int(n), "n_detected": int(len(idx)), "n_frontal": int(front.sum()),
             "base": {"f": f_base, "aspect": aspect(T_base) if f_base else None}}
        if front.sum() >= 5:
            fr = TemplateFrom(pts[front])
            f_fr, T_fr = estimate(fr, pts, size)
            r["frontal"] = {"f": f_fr, "aspect": aspect(T_fr) if f_fr else None}
        else:
            r["frontal"] = {"f": None, "failed": "fewer than 5 detected frames in the frontal window"}
        r["seconds"] = round(time.time() - t0, 1)
        res[f"{sname}/{path.name}"] = r
        rows.append((sname, path, ref, fb, r))
        print(f"[{k}/{len(todo)}] {sname}/{path.name}: base {f_base} (bench {fb}), frontal {r['frontal'].get('f')}, "
              f"aspect {r['base']['aspect']} -> {r['frontal'].get('aspect')}", flush=True)

    (out / "results.json").write_text(json.dumps({"experiment": cfg["experiment"], "videos": res}, ensure_ascii=False, indent=1) + "\n")
    info["finished"] = datetime.now(timezone.utc).isoformat()
    (out / "run.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")

    err = lambda f, ref: 100 * (f / ref - 1) if f else np.nan      # noqa: E731
    L = [f"# {cfg['experiment']} summary", "",
         "**calibrator.py 의 손 템플릿(전체 프레임 손 모양의 평균)을 영상 앞 "
         f"{cfg['frontal_seconds']:g}초(가이드: 정면에서 2초 멈춤)의 손 모양 평균으로만 바꿨을 때. 나머지는 Calibrator.estimate 그대로. 판정 없음.**", "",
         "- 길쭉함 = |손목 - 중지 끝| / |검지 뿌리 - 새끼 뿌리| (템플릿).", "- 기준 f 는 이 파일에만.", "",
         "| 세션 | 영상 | 기준 f | base f (오차) | 벤치마크 f | frontal f (오차) | 길쭉함 base | 길쭉함 frontal | 정면 프레임 |", "|---|---|---|---|---|---|---|---|---|"]
    for sname, path, ref, fb, r in rows:
        b, fr = r["base"], r["frontal"]
        L.append(f"| {sname} | {path.stem} | {ref:g} | {b['f']:.1f} ({err(b['f'], ref):+.1f}%) | {fb} | "
                 + (f"{fr['f']:.1f} ({err(fr['f'], ref):+.1f}%)" if fr.get("f") else "-") + f" | {b['aspect']:.3f} | "
                 + (f"{fr['aspect']:.3f}" if fr.get("aspect") else "-") + f" | {r['n_frontal']} |")
    L += ["", "## 동작별", "", "| 동작 | base 절대오차 평균값 | base 10% 이내 | frontal 절대오차 평균값 | frontal 10% 이내 |", "|---|---|---|---|---|"]
    for m in ("twinkle", "nod", "circle", "mix", None):
        sel = [x for x in rows if m is None or x[1].stem.split("_")[1] == m]
        eb = np.abs([err(x[4]["base"]["f"], x[2]) for x in sel])
        ef = np.abs([err(x[4]["frontal"].get("f"), x[2]) for x in sel])
        L.append(f"| {m or '전체'} | {np.nanmean(eb):.1f}% | {int((eb <= 10).sum())} / {len(sel)} | {np.nanmean(ef):.1f}% | {int((ef <= 10).sum())} / {len(sel)} |")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    p = repo / cfg["results_json"]
    p.write_text(json.dumps({"experiment": cfg["experiment"], "run": os.path.relpath(out, repo), "code_commit": info["git"]["commit"][:7],
                             "git_dirty_at_run": info["git"]["dirty"], "videos": res}, ensure_ascii=False, indent=1) + "\n")
    print(out / "summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
