"""Diagonal-information check (136): split each hand_mix video at the start of its circle motion and run
Calibrator.estimate() (calibrator.py, unchanged) on each part.

    pre  : frame 0 .. cut-1   (twinkle + nod)
    post : frame cut .. end   (circle)
    full : every frame        (must reproduce the f in session.json)

The cut times (seconds, chosen by eye from the video) are in the config. Detection rule unchanged: a frame
counts if all 21 right-hand scores > 3.

    python HandCalib/benchmark/mix_cut.py --config HandCalib/configs/136_mix_cut.json --cache   # GPU, once
    python HandCalib/benchmark/mix_cut.py --config HandCalib/configs/136_mix_cut.json           # estimates

GPU thermal guard while caching: pause at >= 70 C, resume below 65 C.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibrator import RIGHT_HAND, Calibrator  # noqa: E402

HANDCALIB = HERE.parent


def gpu_temp():
    out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    return max(int(x) for x in out.split())


def guard(pause_c, resume_c, poll_s):
    if gpu_temp() >= pause_c:
        print(f"  GPU {gpu_temp()}C >= {pause_c}C, waiting for < {resume_c}C", flush=True)
        while gpu_temp() >= resume_c:
            time.sleep(poll_s)


def cache(cfg, cache_dir):
    import cv2
    model = Calibrator("hand").model
    th = cfg["thermal"]
    for key in sorted(cfg["cuts"]):
        sess, stem = key.split("/")
        dst = cache_dir / sess / f"{stem}.npz"
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        video = next((Path(cfg["data_root"]) / sess).glob(f"{stem}.*"))
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        pts, sc, size = [], [], None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if len(pts) % 100 == 0:
                guard(th["pause_c"], th["resume_c"], th["poll_s"])
            size = frame.shape[1::-1]
            k, s = model(frame)
            pts.append(k[0, RIGHT_HAND])
            sc.append(s[0, RIGHT_HAND])
        cap.release()
        np.savez(dst, points=np.float32(pts), scores=np.float32(sc), size=np.int32(size), fps=fps)
        print(f"{key}: {len(pts)} frames", flush=True)


def estimate(points, scores, size, lo, hi):
    cal = Calibrator.__new__(Calibrator)
    cal.target = "hand"
    cal.reset()
    cal.size = tuple(int(x) for x in size)
    for i in range(lo, hi):
        if scores[i].min() > 3:
            cal.observations.append(points[i])
    return float(cal.estimate()), len(cal.observations), hi - lo


def evaluate(cfg, cache_dir):
    root = Path(cfg["data_root"])
    videos, circle = {}, {}
    for key in sorted(cfg["cuts"]):
        sess, stem = key.split("/")
        meta = json.load(open(root / sess / "session.json"))
        ref = meta["reference_f"]
        d = np.load(cache_dir / sess / f"{stem}.npz")
        n = len(d["points"])
        cut = int(round(cfg["cuts"][key] * float(d["fps"])))
        row = {"reference_f": ref, "cut_s": cfg["cuts"][key], "cut_frame": cut, "n_frames": n}
        for part, lo, hi in (("full", 0, n), ("pre", 0, cut), ("post", cut, n)):
            f, det, frames = estimate(d["points"], d["scores"], d["size"], lo, hi)
            row[part] = {"f": f, "err_pct": 100 * (f / ref - 1), "n_detected": det, "n_frames": frames}
        row["f_session_json"] = next(v["f"] for v in meta["videos"] if v["file"].startswith(stem))
        videos[key] = row
        if sess not in circle:
            circle[sess] = {v["file"]: {"f": v["f"], "err_pct": 100 * (v["f"] / ref - 1)}
                            for v in meta["videos"] if v["motion"] == "circle"}
        print(f"{key}: full {row['full']['err_pct']:+.1f}%  pre {row['pre']['err_pct']:+.1f}%  "
              f"post {row['post']['err_pct']:+.1f}%", flush=True)

    def mae(vals):
        return float(np.mean(np.abs(vals)))

    sessions = {}
    for sess in sorted(circle):
        mixes = [v for k, v in videos.items() if k.startswith(sess + "/")]
        sessions[sess] = {
            "circle": mae([v["err_pct"] for v in circle[sess].values()]),
            "pre": mae([v["pre"]["err_pct"] for v in mixes]),
            "post": mae([v["post"]["err_pct"] for v in mixes]),
            "full": mae([v["full"]["err_pct"] for v in mixes]),
        }
    overall = {
        "circle": mae([v["err_pct"] for c in circle.values() for v in c.values()]),
        "pre": mae([v["pre"]["err_pct"] for v in videos.values()]),
        "post": mae([v["post"]["err_pct"] for v in videos.values()]),
        "full": mae([v["full"]["err_pct"] for v in videos.values()]),
        "n_sessions": len(sessions),
        "n_videos": len(videos),
        "sessions_circle_lower": sum(s["circle"] < s["pre"] for s in sessions.values()),
        "sessions_pre_lower": sum(s["pre"] < s["circle"] for s in sessions.values()),
    }
    return {"videos": videos, "circle_videos": circle, "sessions_mean_abs_err_pct": sessions, "overall": overall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", action="store_true", help="run the hand model and store per-frame points")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    cache_dir = HANDCALIB / cfg["cache_dir"]
    if a.cache:
        cache(cfg, cache_dir)
        return
    res = {"experiment": cfg["experiment"], "cache_dir": cfg["cache_dir"]}
    res.update(evaluate(cfg, cache_dir))
    out = HANDCALIB / cfg["results_json"]
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(res["overall"], indent=1))


if __name__ == "__main__":
    main()
