"""Overlay video of each detector's 2D hand keypoints (the tracked target hand the pipeline used) on the normalised frames.

    python HandCalib/run_keypoint_overlay.py --video-dir runs/17_protocol_rerun/<UTC>/IMG_XXXX [--input /path/IMG_XXXX.MOV]

Reads <video-dir>/<detector>/{results.json, detections.npz}, re-creates the same normalised PNG frames (hashes checked), re-tracks
with 14's function and the detector's conf_min, and writes <video-dir>/overlay/<detector>.mp4 and side_by_side.mp4.
Kept points: filled circles and skeleton; masked points (low confidence / outside): hollow grey circles. Temporary frames are deleted.
The overlay videos stay in the (git-ignored) run folder.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hand_focal import frames as FR                       # noqa: E402
from hand_focal import normalize as NZ                    # noqa: E402
from hand_focal import track as TR                        # noqa: E402
from hand_focal.detectors import adapters as DA           # noqa: E402
from hand_focal.paths import PROJECT_ROOT, resolve        # noqa: E402

BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
         (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]
FINGER_COLOR = {0: (255, 255, 255), 1: (0, 140, 255), 2: (0, 220, 0), 3: (255, 160, 0), 4: (255, 0, 200), 5: (0, 0, 255)}   # BGR: wrist, thumb..pinky
ROLE_KO = {"front": "front window", "tilt": "tilt", "none": "excluded"}


def finger_of(j):
    return 0 if j == 0 else (j - 1) // 4 + 1


def draw(img, kp, mask, box, label, lines):
    out = img.copy()
    s = max(1.0, max(img.shape[:2]) / 1920 * 1.6)
    if box is not None and np.isfinite(box).all():
        cv2.rectangle(out, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (200, 200, 200), max(1, int(s)))
    if kp is not None:
        for a, b in BONES:
            if mask[a] and mask[b]:
                c = FINGER_COLOR[finger_of(b)]
                cv2.line(out, tuple(np.round(kp[a]).astype(int)), tuple(np.round(kp[b]).astype(int)), c, max(1, int(2 * s)), cv2.LINE_AA)
        for j in range(21):
            if not np.isfinite(kp[j]).all():
                continue
            p = tuple(np.round(kp[j]).astype(int))
            if mask[j]:
                cv2.circle(out, p, int(5 * s), FINGER_COLOR[finger_of(j)], -1, cv2.LINE_AA)
                cv2.circle(out, p, int(5 * s), (0, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.circle(out, p, int(5 * s), (150, 150, 150), max(1, int(s)), cv2.LINE_AA)
    y = int(40 * s)
    for i, t in enumerate([label] + lines):
        org = (int(15 * s), y + i * int(45 * s))
        cv2.putText(out, t, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, (0, 0, 0), max(3, int(4 * s)), cv2.LINE_AA)
        cv2.putText(out, t, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, (255, 255, 255), max(2, int(1.5 * s)), cv2.LINE_AA)
    return out


def encode(frames_dir: Path, pattern: str, fps: float, out: Path):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", f"{fps:.4f}", "-i", str(frames_dir / pattern), "-c:v", "libx264",
                    "-crf", "18", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-map_metadata", "-1", str(out)], check=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True, help="run folder of one video (contains one sub-folder per detector)")
    ap.add_argument("--input", default=None, help="original video (default: input_dir/<name>.MOV from the pipeline config)")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/17_protocol_rerun.yaml"))
    a = ap.parse_args(argv)
    vdir = resolve(a.video_dir)
    cfg = yaml.safe_load(Path(a.config).read_text())
    dets = [d for d in cfg["detectors"] if (vdir / d / "results.json").exists()]
    r0 = json.loads((vdir / dets[0] / "results.json").read_text())
    src = Path(a.input).expanduser() if a.input else Path(cfg["input_dir"]).expanduser() / r0["video"]
    tmp = Path(tempfile.mkdtemp(prefix="hfov_"))
    try:
        norm = NZ.normalize(src, tmp / "n.mkv", cfg["pipeline"])
        W, H = norm["output"]["display_w"], norm["output"]["display_h"]
        frames, k = FR.extract(tmp / "n.mkv", tmp / "f", norm["output"]["fps"], None, cfg["pipeline"]["frames"]["detect_fps"])
        if [(x["index"], x["sha256"]) for x in frames] != [(x["index"], x["sha256"]) for x in r0["frames"]["list"]]:
            raise SystemExit("re-created frames differ from the run's frame list")
        times = np.array([x["time_s"] for x in frames])
        fps_out = norm["output"]["fps"] / k
        out_dir = vdir / "overlay"
        out_dir.mkdir(exist_ok=True)
        tracks = {}
        for d in dets:
            r = json.loads((vdir / d / "results.json").read_text())
            det = DA._load(vdir / d / "detections.npz")
            DA.check_alignment(frames, det)
            tc = dict(cfg["pipeline"]["tracking"], conf_min=float(cfg["detector_settings"][d]["conf_min"]))
            tr = TR.track(det, times, (W, H), tc, front_start_s=float(cfg["pipeline"]["phases"]["front_skip_s"]))
            a_ = r.get("analysis", {})
            th = a_.get("theta2d_deg") or [None] * len(frames)
            role = a_.get("role") or ["none"] * len(frames)
            tracks[d] = (tr, th, role)
        for d in dets + ["side_by_side"]:
            (tmp / d).mkdir()
        for i, fr in enumerate(frames):
            img = cv2.imread(fr["png"], cv2.IMREAD_COLOR)
            panels = []
            for d in dets:
                tr, th, role = tracks[d]
                kp = tr["kp"][i] if tr["present"][i] else None
                lines = [f"t {fr['time_s']:.2f} s   {ROLE_KO.get(role[i], role[i])}",
                         "no hand" if kp is None else f"theta2d {th[i]:.0f} deg   kept {int(tr['mask'][i].sum())}/21" if th[i] is not None else f"kept {int(tr['mask'][i].sum())}/21"]
                o = draw(img, kp, tr["mask"][i], tr["box"][i] if tr["present"][i] else None, d, lines)
                cv2.imwrite(str(tmp / d / f"o_{i:05d}.png"), o)
                panels.append(o)
            sbs = np.concatenate(panels, axis=1)
            scale = min(1.0, 2880 / sbs.shape[1])
            if scale < 1:
                sbs = cv2.resize(sbs, (int(sbs.shape[1] * scale) // 2 * 2, int(sbs.shape[0] * scale) // 2 * 2), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(tmp / "side_by_side" / f"o_{i:05d}.png"), sbs)
        made = []
        for d in dets + ["side_by_side"]:
            encode(tmp / d, "o_%05d.png", fps_out, out_dir / f"{d}.mp4")
            made.append(str(out_dir / f"{d}.mp4"))
        print("\n".join(made))
        return made
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
