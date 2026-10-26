"""HandCalib hand-template focal length (one module): everything the detector-swap (16), protocol (17, incl. the frame-filter diagnostic)
and same-recording (18) experiments use, consolidated from the former packages hand_focal/ (14), detector_swap/ (16), protocol_rerun/ (17)
and same_session/ (18). Each former module is a section below ("was HandCalib/..."); colliding names carry a prefix.

Orchestration and the command line are in run.py; the detector workers (separate virtual environments) are in detector_worker.py.
"""
from __future__ import annotations

import contextlib
import copy
import csv
import cv2
import hashlib
import json
import math
import numpy as np
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import yaml
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path
from scipy.optimize import minimize


# ====================================================================================================
# paths  (was HandCalib/hand_focal/paths.py)
# ====================================================================================================

PROJECT_ROOT = Path(__file__).resolve().parent         # HandCalib/
WORKER = PROJECT_ROOT / "detector_worker.py"            # standalone detector workers (run by each detector's own venv)
REPO_ROOT = PROJECT_ROOT.parent


def resolve(p) -> Path:
    q = Path(p).expanduser()
    return q if q.is_absolute() else PROJECT_ROOT / q


# ====================================================================================================
# normalize  (was HandCalib/hand_focal/normalize.py)
# Video probing and normalisation (ffmpeg / ffprobe). Only technical stream fields are kept: no location, no device tags.
# ====================================================================================================

SAFE_STREAM_KEYS = ("codec_name", "profile", "width", "height", "pix_fmt", "color_transfer", "color_primaries", "color_space",
                    "r_frame_rate", "avg_frame_rate", "nb_frames", "duration", "bits_per_raw_sample")
HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}


def tool_info() -> dict:
    out = {}
    for t in ("ffmpeg", "ffprobe"):
        p = shutil.which(t)
        out[t] = {"path": p, "version": subprocess.check_output([p, "-hide_banner", "-version"], text=True).splitlines()[0] if p else None}
    filt = subprocess.check_output(["ffmpeg", "-hide_banner", "-filters"], text=True) if out["ffmpeg"]["path"] else ""
    out["filters"] = {f: (f" {f} " in filt) for f in ("zscale", "tonemap", "colorspace")}
    return out


def probe(path: Path) -> dict:
    """Technical fields of the first video stream (rotation from the display matrix); tags are never read."""
    j = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                            "stream=" + ",".join(SAFE_STREAM_KEYS), "-show_entries", "stream_side_data_list",
                                            "-show_entries", "format=duration,size,format_name", "-of", "json", str(path)], text=True))
    s = j["streams"][0]
    rot = 0
    for sd in s.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    info = {k: s.get(k) for k in SAFE_STREAM_KEYS}
    info["rotation_deg"] = rot
    info["file_name"] = path.name
    info["file_size"] = int(j["format"].get("size", 0))
    info["container"] = j["format"].get("format_name")
    info["duration_s"] = float(j["format"].get("duration") or s.get("duration") or 0)
    info["fps"] = float(Fraction(s.get("avg_frame_rate") or s.get("r_frame_rate")))
    pix = s.get("pix_fmt") or ""
    info["bit_depth"] = int(s.get("bits_per_raw_sample") or (10 if "10" in pix else 8))
    info["hdr"] = (s.get("color_transfer") in HDR_TRANSFERS) or info["bit_depth"] > 8
    rotated = abs(rot) % 180 == 90
    info["display_w"], info["display_h"] = (s["height"], s["width"]) if rotated else (s["width"], s["height"])
    return info


def target_dims(w: int, h: int, long_side: int) -> tuple[int, int]:
    """Aspect kept, never upscaled, even sides."""
    s = min(1.0, long_side / max(w, h))
    tw, th = int(round(w * s / 2)) * 2, int(round(h * s / 2)) * 2
    return max(tw, 2), max(th, 2)


def build_command(src: Path, dst: Path, info: dict, cfg: dict, tools: dict) -> tuple[list, dict]:
    n = cfg["normalize"]
    tw, th = target_dims(info["display_w"], info["display_h"], int(n["long_side"]))
    vf, path = [], {"autorotate": True, "rotation_applied_deg": info["rotation_deg"]}
    if info["hdr"]:
        if tools["filters"]["zscale"] and tools["filters"]["tonemap"] and "zscale_tonemap" in n["hdr_path"]:
            vf += ["zscale=t=linear:npl=100", "format=gbrpf32le", "zscale=p=bt709", "tonemap=hable:desat=0", "zscale=t=bt709:m=bt709:r=tv"]
            path["tonemap"] = "zscale+tonemap(hable)"
        elif tools["filters"]["colorspace"] and "colorspace" in n["hdr_path"]:
            vf += ["colorspace=all=bt709:iall=bt2020:fast=1"]
            path["tonemap"] = "colorspace (no tone curve)"
        else:
            raise RuntimeError("HDR input but neither zscale+tonemap nor colorspace is available")
    else:
        path["tonemap"] = "none (8-bit SDR input)"
    vf.append(f"scale={tw}:{th}:flags=bicubic")
    fps_in = info["fps"]
    if fps_in > float(n["max_fps"]) + 0.01:
        vf.append(f"fps={n['max_fps']}")
        path["fps"] = f"{fps_in:.3f} -> {n['max_fps']}"
    else:
        path["fps"] = f"{fps_in:.3f} (kept)"
    vf.append("format=yuv444p" if n["codec"] == "ffv1" else "format=yuv420p")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-map", "0:v:0", "-an", "-sn", "-dn",
           "-map_metadata", "-1", "-map_chapters", "-1", "-vf", ",".join(vf), "-c:v", n["codec"], "-colorspace", "bt709",
           "-color_primaries", "bt709", "-color_trc", "bt709", str(dst)]
    path.update({"filters": vf, "out_size": [tw, th], "codec": n["codec"]})
    return cmd, path


def normalize(src: Path, dst: Path, cfg: dict) -> dict:
    tools = tool_info()
    info = probe(src)
    cmd, path = build_command(src, dst, info, cfg, tools)
    subprocess.run(cmd, check=True)
    out = probe(dst)
    if [out["display_w"], out["display_h"]] != path["out_size"]:
        raise RuntimeError(f"normalised size {out['display_w']}x{out['display_h']} != {path['out_size']}")
    return {"input": info, "output": {k: out[k] for k in ("display_w", "display_h", "fps", "duration_s", "nb_frames", "pix_fmt", "color_transfer", "rotation_deg")},
            "path": path, "command": cmd[:-1] + ["<out>"], "tools": tools}


# ====================================================================================================
# frames  (was HandCalib/hand_focal/frames.py)
# Frame selection (detect_fps) from the normalised video into lossless PNG files with a frame / time / hash list.
# ====================================================================================================

def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def extract_frames(video: Path, out_dir: Path, fps: float, n_frames: int | None, detect_fps: float) -> list:
    """Every k-th frame (k = round(fps / detect_fps), at least 1). Returns [{index, frame, time_s, png, sha256}]."""
    out_dir.mkdir(parents=True, exist_ok=True)
    k = max(1, int(round(fps / detect_fps)))
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-vf", f"select=not(mod(n\\,{k}))",
                    "-vsync", "0", "-start_number", "0", str(out_dir / "f_%06d.png")], check=True)
    files = sorted(out_dir.glob("f_*.png"))
    rows = []
    for i, p in enumerate(files):
        fr = i * k
        rows.append({"index": i, "frame": fr, "time_s": fr / fps, "png": str(p), "sha256": sha256(p)})
    return rows, k


# ====================================================================================================
# adapters  (was HandCalib/hand_focal/detectors/adapters.py)
# Detector adapters. Both return the same dict: frame_index, offsets, boxes, box_conf, cls, kxy, kconf, info.
# ====================================================================================================

class DetectorError(RuntimeError):
    pass


def load_detections(npz: Path) -> dict:
    z = np.load(npz, allow_pickle=False)
    d = {k: z[k] for k in z.files if k != "info"}
    d["info"] = json.loads(str(z["info"])) if "info" in z.files else {}
    return d


def cache_key(frames: list, dcfg: dict) -> str:
    h = hashlib.sha256()
    for r in frames:
        h.update(f"{r['index']}:{r['sha256']};".encode())
    h.update(json.dumps({k: dcfg.get(k) for k in ("weights", "conf", "iou", "imgsz")}, sort_keys=True).encode())
    return h.hexdigest()[:16]


def detector_wilor_yolo(frames: list, dcfg: dict, cache_dir: Path, log=print) -> dict:
    """Run the worker in the configured (read-only) venv; cache by frame hashes + settings."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key(frames, dcfg)
    out = cache_dir / f"detections_{key}.npz"
    if dcfg.get("cache", True) and out.exists():
        log(f"  detection cache hit: {out.name}")
        d = load_detections(out)
    else:
        lst = cache_dir / f"frames_{key}.tsv"
        lst.write_text("".join(f"{r['index']}\t{r['png']}\t{r['sha256']}\n" for r in frames))
        cmd = [dcfg["python"], str(WORKER), "wilor_yolo", "--weights", dcfg["weights"], "--list", str(lst), "--out", str(out),
               "--conf", str(dcfg["conf"]), "--iou", str(dcfg["iou"]), "--device", str(dcfg["device"])]
        if dcfg.get("imgsz"):
            cmd += ["--imgsz", str(dcfg["imgsz"])]
        env = dict(os.environ, PYTHONNOUSERSITE="1")
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if p.returncode != 0 or not out.exists():
            raise DetectorError(f"YOLO worker failed (exit {p.returncode}): {p.stderr.strip()[-800:]}")
        lst.unlink()
        d = load_detections(out)
    check_alignment(frames, d)
    d["cache_file"] = str(out)
    return d


def detector_mock(frames: list, dcfg: dict, cache_dir: Path, log=print) -> dict:
    d = load_detections(Path(dcfg["mock_npz"]))
    if "sha256" not in d:
        d["sha256"] = np.array([r["sha256"] for r in frames])
    check_alignment(frames, d)
    d["cache_file"] = str(dcfg["mock_npz"])
    return d


def check_alignment(frames: list, d: dict) -> None:
    """The worker's frame numbers and PNG hashes must equal the frame list of the main code."""
    idx = [int(x) for x in d["frame_index"]]
    if idx != [r["index"] for r in frames]:
        raise DetectorError("detector frame indices differ from the frame list")
    if [str(x) for x in d["sha256"]] != [r["sha256"] for r in frames]:
        raise DetectorError("detector PNG hashes differ from the frame list")


def detector_run(frames, dcfg, cache_dir, log=print) -> dict:
    if dcfg["name"] == "wilor_yolo":
        return detector_wilor_yolo(frames, dcfg, cache_dir, log)
    if dcfg["name"] == "mock":
        return detector_mock(frames, dcfg, cache_dir, log)
    raise DetectorError(f"unknown detector {dcfg['name']}")


# ====================================================================================================
# track  (was HandCalib/hand_focal/track.py)
# Target-hand selection, IoU tracking and point masks.
# ====================================================================================================

def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ar = lambda r: max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])   # noqa: E731
    return inter / max(ar(a) + ar(b) - inter, 1e-9)


def candidates(det: dict, i: int):
    a, b = int(det["offsets"][i]), int(det["offsets"][i + 1])
    return list(range(a, b))


def track(det: dict, times: np.ndarray, size_wh, tcfg: dict, front_start_s: float = 0.0) -> dict:
    """Target = the candidate with the largest box area x confidence in the first init_window_s of the front phase; then IoU tracking."""
    n = len(times)
    W, H = size_wh
    init = [i for i in range(n) if front_start_s <= times[i] < front_start_s + tcfg["init_window_s"]]
    best, best_i = None, None
    for i in init:
        for k in candidates(det, i):
            b = det["boxes"][k]
            s = (b[2] - b[0]) * (b[3] - b[1]) * det["box_conf"][k]
            if best is None or s > best[0]:
                best, best_i = (s, k), i
    out = {"present": np.zeros(n, bool), "box": np.full((n, 4), np.nan), "cls": np.full(n, -1), "box_conf": np.zeros(n),
           "kp": np.full((n, 21, 2), np.nan), "kconf": np.zeros((n, 21)), "mask": np.zeros((n, 21), bool), "n_candidates": np.zeros(n, int),
           "init_frame": best_i}
    for i in range(n):
        out["n_candidates"][i] = len(candidates(det, i))
    if best is None:
        return out
    ref = det["boxes"][best[1]].copy()
    order = list(range(best_i, n)) + list(range(best_i - 1, -1, -1))     # forward from the init frame, then backward
    last_fwd, last_bwd = ref.copy(), ref.copy()
    for i in order:
        refb = last_fwd if i >= best_i else last_bwd
        cs = candidates(det, i)
        if not cs:
            continue
        ious = [iou(refb, det["boxes"][k]) for k in cs]
        j = int(np.argmax(ious))
        if i == best_i:
            k = best[1]
        elif ious[j] >= tcfg["iou_min"]:
            k = cs[j]
        else:
            continue
        out["present"][i] = True
        out["box"][i], out["cls"][i], out["box_conf"][i] = det["boxes"][k], det["cls"][k], det["box_conf"][k]
        out["kp"][i], out["kconf"][i] = det["kxy"][k], det["kconf"][k]
        if i >= best_i:
            last_fwd = det["boxes"][k].copy()
        else:
            last_bwd = det["boxes"][k].copy()
    kp = out["kp"]
    inside = (kp[..., 0] >= 0) & (kp[..., 0] < W) & (kp[..., 1] >= 0) & (kp[..., 1] < H)
    out["mask"] = out["present"][:, None] & (out["kconf"] >= tcfg["conf_min"]) & inside
    if tcfg.get("smoothing", {}).get("enabled"):
        out["kp"] = smooth(out["kp"], out["mask"], int(tcfg["smoothing"]["window"]))
    return out


def smooth(kp, mask, w):
    """Optional centred moving average over visible points (off by default)."""
    out = kp.copy()
    h = w // 2
    for i in range(len(kp)):
        lo, hi = max(0, i - h), min(len(kp), i + h + 1)
        for j in range(21):
            m = mask[lo:hi, j]
            if mask[i, j] and m.sum() > 1:
                out[i, j] = kp[lo:hi, j][m].mean(0)
    return out


def box_long(box):
    return np.maximum(box[..., 2] - box[..., 0], box[..., 3] - box[..., 1])


def order_check(kp: np.ndarray, mask: np.ndarray) -> dict:
    """OpenPose order on 2D: wrist->MCP distance (middle longest, pinky shortest) and the thumb base nearer the index MCP."""
    ok = mask[:, [0, 1, 5, 9, 13, 17]].all(1)
    if ok.sum() == 0:
        return {"ok": None, "reason": "no frame with the needed points"}
    P = kp[ok]
    d = {n: float(np.median(np.linalg.norm(P[:, j] - P[:, 0], axis=-1))) for n, j in (("index", 5), ("middle", 9), ("ring", 13), ("pinky", 17))}
    ci = float(np.median(np.linalg.norm(P[:, 1] - P[:, 5], axis=-1)))
    cp = float(np.median(np.linalg.norm(P[:, 1] - P[:, 17], axis=-1)))
    res = {"wrist_to_mcp_px": d, "middle_longest": d["middle"] == max(d.values()), "pinky_shortest": d["pinky"] == min(d.values()),
           "thumb_base_nearer_index": ci < cp, "n_frames": int(ok.sum())}
    res["ok"] = bool(res["middle_longest"] and res["pinky_shortest"] and res["thumb_base_nearer_index"])
    return res


# ====================================================================================================
# template  (was HandCalib/hand_focal/template.py)
# Frontal template (similarity Procrustes mean) and the weak-perspective 2D tilt indicator theta2d.
# ====================================================================================================

def similarity_align(src: np.ndarray, dst: np.ndarray):
    """Least-squares similarity src -> dst (2D). Returns (s, R, t) with dst ~ s R src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a)
    D = np.diag([1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = float((S * np.diag(D)).sum() / max((a ** 2).sum(), 1e-12))
    return s, R, mu_d - s * R @ mu_s


def template_build(front_pts: np.ndarray, iters: int = 4) -> dict:
    """front_pts (N, J, 2) of frames with every joint visible -> template (J, 2), centred, in the window's pixel scale."""
    X = front_pts.astype(float)
    T = X[0] - X[0].mean(0)
    for _ in range(iters):
        aligned = []
        for P in X:
            s, R, t = similarity_align(P, T)
            aligned.append((s * (R @ P.T)).T + t)
        A = np.array(aligned)
        T = A.mean(0)
        T = T - T.mean(0)
    # scale = mean pixel size of the window
    sizes = [np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean()) for P in X]
    T = T * (np.mean(sizes) / np.sqrt((T ** 2).sum(1).mean()))
    res = []
    for P in X:                                     # residuals in each frame's own pixels
        s, R, t = similarity_align(T, P)
        res.append(np.linalg.norm((s * (R @ T.T)).T + t - P, axis=1))
    res = np.array(res)
    return {"template": T, "jitter_rms_px": float(np.sqrt((res ** 2).mean())), "n_frames": int(len(X))}


def affine_tilt(T: np.ndarray, P: np.ndarray):
    """Least-squares affine T -> P; theta2d = arccos(s2 / s1) of its linear part; axis = direction of the compressed image axis (deg, 0-180)."""
    A = np.c_[T, np.ones(len(T))]
    M, *_ = np.linalg.lstsq(A, P, rcond=None)
    L = M[:2].T                                       # 2x2 linear part, P ~ L T + t
    U, S, Vt = np.linalg.svd(L)
    ratio = float(np.clip(S[1] / S[0], 0.0, 1.0)) if S[0] > 0 else 0.0
    theta = float(np.degrees(np.arccos(ratio)))
    axis = float(np.degrees(np.arctan2(U[1, 1], U[0, 1])) % 180.0)
    return theta, axis


def finger_extension(kp: np.ndarray) -> dict:
    """MCP -> tip straight distance / sum of the three bone lengths, per finger (index .. pinky), on 2D points."""
    out = {}
    for name, ids in (("index", [5, 6, 7, 8]), ("middle", [9, 10, 11, 12]), ("ring", [13, 14, 15, 16]), ("pinky", [17, 18, 19, 20])):
        P = kp[:, ids]
        straight = np.linalg.norm(P[:, 3] - P[:, 0], axis=-1)
        path = np.linalg.norm(np.diff(P, axis=1), axis=-1).sum(1)
        out[name] = float(np.nanmedian(straight / np.maximum(path, 1e-9)))
    out["min"] = float(min(out.values()))
    out["mean"] = float(np.mean([out[k] for k in ("index", "middle", "ring", "pinky")]))
    return out


# ====================================================================================================
# phases  (was HandCalib/hand_focal/phases.py)
# Front window and tilt phase.
# ====================================================================================================

def front_window(times, present, kp, mask, box, pcfg) -> dict:
    """Lowest-motion contiguous window of front_window_s inside [front_skip_s, front_phase_s) with detection rate >= the minimum."""
    t = np.asarray(times)
    cand = np.flatnonzero((t >= pcfg["front_skip_s"]) & (t < pcfg["front_phase_s"]))
    bl = box_long(box)
    speed = np.full(len(t), np.nan)
    for i in range(1, len(t)):
        if present[i] and present[i - 1]:
            m = mask[i] & mask[i - 1]
            if m.sum() >= 4:
                speed[i] = float(np.median(np.linalg.norm(kp[i, m] - kp[i - 1, m], axis=-1)) / max(np.nanmean([bl[i], bl[i - 1]]), 1e-9))
    best, warn = None, []
    for s in cand:
        idx = cand[(t[cand] >= t[s]) & (t[cand] < t[s] + pcfg["front_window_s"] - 1e-9)]
        if t[s] + pcfg["front_window_s"] > pcfg["front_phase_s"] + 1e-9:
            break
        rate = float(present[idx].mean()) if len(idx) else 0.0
        sp = float(np.nanmean(speed[idx])) if np.isfinite(speed[idx]).any() else np.inf
        key = (rate < pcfg["front_min_detect_rate"], sp)
        if best is None or key < best[0]:
            best = (key, idx, rate, sp)
    if best is None:
        return {"idx": np.zeros(0, int), "detect_rate": 0.0, "speed_rel": float("nan"), "warnings": ["정면 구간에 창을 만들 프레임이 없습니다"]}
    if best[2] < pcfg["front_min_detect_rate"]:
        warn.append(f"정면 창의 검출률이 낮습니다({best[2]:.2f})")
    return {"idx": best[1], "detect_rate": best[2], "speed_rel": best[3], "t0": float(t[best[1][0]]), "t1": float(t[best[1][-1]]), "warnings": warn}


def auto_transition(times, theta, present, pcfg):
    """First time at which theta2d stays >= auto_theta_deg for auto_sustain_s."""
    t = np.asarray(times)
    ok = present & np.isfinite(theta) & (theta >= pcfg["auto_theta_deg"])
    for i in range(len(t)):
        if not ok[i]:
            continue
        j = np.flatnonzero((t >= t[i]) & (t < t[i] + pcfg["auto_sustain_s"]))
        if len(j) and ok[j].all():
            return float(t[i])
    return None


def tilt_indices(times, present, t_start) -> np.ndarray:
    t = np.asarray(times)
    return np.flatnonzero((t >= t_start) & present)


# ====================================================================================================
# estimate  (was HandCalib/hand_focal/estimate.py)
# f from a planar hand template: per-frame IPPE pose at every candidate f, robust mean reprojection cost.
#
# fixed_template: the frontal template is the true planar shape; f from a log grid refined by a parabola in log f.
# joint_delta:    the frontal hand was really tilted by delta towards phi; the template is stretched by 1 / cos(delta) along phi
#                 (first-order weak-perspective correction) and (f, delta, phi) is found on a grid and refined by Nelder-Mead, with a
#                 soft prior (delta / delta_prior)^2 in units of the per-point noise variance.
# Pinhole, principal point fixed, square pixels, no distortion.
# ====================================================================================================

class Problem:
    def __init__(self, template: np.ndarray, frames: list, cx: float, cy: float, huber_px: np.ndarray, min_points: int = 4, weights: np.ndarray | None = None):
        """template (J, 2); frames: list of (obs (J, 2), visible (J,)) ; huber_px: per frame Huber width; weights (J,): per-joint weight of the
        reprojection cost (21's wrist_mode downweight); None = every joint counts the same (17's behaviour)."""
        self.w = None if weights is None else np.asarray(weights, float)
        self.T = np.asarray(template, float)
        self.frames = [(np.asarray(o, float), np.asarray(v, bool)) for o, v in frames]
        self.keep = [i for i, (o, v) in enumerate(self.frames) if v.sum() >= max(4, min_points)]
        self.cx, self.cy = float(cx), float(cy)
        self.huber = np.asarray(huber_px, float)
        self.n_points = int(sum(self.frames[i][1].sum() for i in self.keep))

    def subset(self, idx):
        p = Problem.__new__(Problem)
        p.T, p.frames, p.cx, p.cy, p.huber, p.w = self.T, self.frames, self.cx, self.cy, self.huber, self.w
        p.keep = [i for i in idx if i in set(self.keep)]
        p.n_points = int(sum(self.frames[i][1].sum() for i in p.keep))
        return p


def stretch(T: np.ndarray, delta_deg: float, phi_deg: float) -> np.ndarray:
    u = np.array([np.cos(np.radians(phi_deg)), np.sin(np.radians(phi_deg))])
    k = 1.0 / np.cos(np.radians(min(abs(delta_deg), 80.0))) - 1.0
    return T + k * (T @ u)[:, None] * u[None, :]


def _huber(r, d):
    return np.where(r <= d, r * r, 2 * d * r - d * d)


def frame_fit(T2: np.ndarray, obs: np.ndarray, vis: np.ndarray, K: np.ndarray):
    """IPPE pose of the planar template on the visible points; the lower-error of the two solutions. Returns (residuals, R, t)."""
    obj = np.c_[T2[vis], np.zeros(vis.sum())].astype(np.float64)
    img = obs[vis].astype(np.float64)
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
    k = int(np.argmin(np.asarray(errs).ravel())) if n > 1 else 0
    R, _ = cv2.Rodrigues(rvecs[k])
    t = tvecs[k].ravel()
    X = obj @ R.T + t
    proj = K[:2, :2] @ (X[:, :2] / X[:, 2:3]).T + K[:2, 2:3]
    return np.linalg.norm(proj.T - img, axis=1), R, t


def K_of(f, cx, cy):
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])


def cost(pb: Problem, f: float, T2: np.ndarray | None = None, details: bool = False):
    T2 = pb.T if T2 is None else T2
    K = K_of(f, pb.cx, pb.cy)
    per, out = [], []
    for i in pb.keep:
        o, v = pb.frames[i]
        r, R, t = frame_fit(T2, o, v, K)
        per.append(float(_huber(r, pb.huber[i]).mean()) if pb.w is None else float((pb.w[v] * _huber(r, pb.huber[i])).sum() / pb.w[v].sum()))
        if details:
            out.append({"frame": i, "res": r, "vis": v, "R": R, "t": t})
    c = float(np.mean(per)) if per else float("inf")
    return (c, out) if details else c


def f_grid(cfg_e: dict, long_side: float) -> np.ndarray:
    lo, hi = cfg_e["f_ratio_range"]
    return np.geomspace(lo * long_side, hi * long_side, int(cfg_e["f_grid_n"]))


def _parabola(lf: np.ndarray, c: np.ndarray, k: int) -> float:
    if k == 0 or k == len(lf) - 1:
        return float(lf[k])
    x, y = lf[k - 1:k + 2], c[k - 1:k + 2]
    A = np.vstack([x ** 2, x, np.ones(3)]).T
    a, b, _ = np.linalg.solve(A, y)
    return float(-b / (2 * a)) if a > 0 else float(lf[k])


def fixed_template(pb: Problem, grid: np.ndarray) -> dict:
    c = np.array([cost(pb, f) for f in grid])
    k = int(np.argmin(c))
    lf = _parabola(np.log(grid), c, k)
    f = float(np.exp(np.clip(lf, np.log(grid[0]), np.log(grid[-1]))))
    return {"f": f, "delta_deg": 0.0, "phi_deg": 0.0, "profile_f": grid.tolist(), "profile_cost": c.tolist(), "cost": cost(pb, f),
            "at_grid_edge": bool(k in (0, len(grid) - 1))}


def joint_delta(pb: Problem, grid: np.ndarray, jd: dict, sigma2: float, start=None) -> dict:
    """(f, delta, phi): grid then Nelder-Mead on the total = N * mean cost / sigma^2 + (delta / delta_prior)^2."""
    N = max(pb.n_points, 1)
    dp = float(jd["delta_prior_deg"])

    def total(lf, d, ph):
        return N * cost(pb, float(np.exp(lf)), stretch(pb.T, d, ph)) / sigma2 + (d / dp) ** 2

    phis = np.linspace(0, 180, int(jd["phi_n"]), endpoint=False)
    prof = np.full(len(grid), np.inf)
    best = (np.inf, None)
    if start is None:
        for d in jd["delta_grid_deg"]:
            for ph in (phis if d > 0 else [0.0]):
                T2 = stretch(pb.T, d, ph)
                for k, f in enumerate(grid):
                    v = N * cost(pb, f, T2) / sigma2 + (d / dp) ** 2
                    prof[k] = min(prof[k], v)
                    if v < best[0]:
                        best = (v, (np.log(f), float(d), float(ph)))
        x0 = best[1]
    else:
        x0 = start
    lo, hi = np.log(grid[0]), np.log(grid[-1])
    fun = lambda x: total(np.clip(x[0], lo, hi), abs(x[1]), x[2] % 180.0)   # noqa: E731
    r = minimize(fun, np.array(x0, float), method="Nelder-Mead",
                 options={"maxiter": int(jd["nm_max_iter"]), "xatol": 1e-4, "fatol": 1e-6, "initial_simplex": None})
    lf, d, ph = float(np.clip(r.x[0], lo, hi)), abs(float(r.x[1])), float(r.x[2] % 180.0)
    f = float(np.exp(lf))
    return {"f": f, "delta_deg": d, "phi_deg": ph, "profile_f": grid.tolist(), "profile_cost": (prof * sigma2 / N).tolist() if start is None else None,
            "cost": cost(pb, f, stretch(pb.T, d, ph)), "objective": float(r.fun), "nm_iters": int(r.nit), "at_grid_edge": bool(lf <= lo + 1e-6 or lf >= hi - 1e-6)}


def estimate_run(pb: Problem, method: str, grid: np.ndarray, ecfg: dict, sigma2: float, start=None) -> dict:
    if method == "fixed_template":
        return fixed_template(pb, grid)
    if method == "joint_delta":
        return joint_delta(pb, grid, ecfg["joint_delta"], sigma2, start)
    raise ValueError(method)


def diagnostics(pb: Problem, sol: dict, box_long_med: float, joint_ids: list, tilt_ids=None) -> dict:
    T2 = stretch(pb.T, sol["delta_deg"], sol["phi_deg"])
    c0, det = cost(pb, sol["f"], T2, details=True)
    res = np.concatenate([d["res"] for d in det]) if det else np.zeros(0)
    per_joint = {}
    J = len(joint_ids)
    acc = [[] for _ in range(J)]
    for d in det:
        for pos, j in enumerate(np.flatnonzero(d["vis"])):
            acc[j].append(d["res"][pos])
    for j in range(J):
        per_joint[int(joint_ids[j])] = float(np.sqrt(np.mean(np.square(acc[j])))) if acc[j] else float("nan")
    sel = [d for d in det if tilt_ids is None or d["frame"] in set(tilt_ids)]
    normals = np.array([d["R"][:, 2] for d in sel]) if sel else np.zeros((0, 3))
    az = np.degrees(np.arctan2(normals[:, 1], normals[:, 0])) % 360 if len(normals) else np.zeros(0)
    tilt = np.degrees(np.arccos(np.clip(np.abs(normals[:, 2]), 0, 1))) if len(normals) else np.zeros(0)
    cp = cost(pb, sol["f"] * 1.15, T2)
    cm = cost(pb, sol["f"] / 1.15, T2)
    rms = float(np.sqrt(np.mean(res ** 2))) if len(res) else float("nan")
    return {"reproj_rms_px": rms, "reproj_rms_rel": rms / box_long_med if box_long_med else float("nan"), "per_joint_rms_px": per_joint,
            "sharpness": float(min(cp, cm) / max(c0, 1e-12)), "cost": c0, "cost_plus15": cp, "cost_minus15": cm,
            "pnp_azimuth_deg": az.tolist(), "pnp_tilt_deg": tilt.tolist(), "azimuth_bins8": int(len(np.unique((az // 45).astype(int)))) if len(az) else 0,
            "frames_used": [int(d["frame"]) for d in det], "frame_rms_px": [float(np.sqrt(np.mean(d["res"] ** 2))) for d in det]}


# ====================================================================================================
# uncertainty  (was HandCalib/hand_focal/uncertainty.py)
# Heuristic uncertainty of the hand f: joint-set range, block bootstrap, consistency splits, estimator difference.
#
# f_rel_uncertainty = max(2 x bootstrap relative std, joint-set range / 2, consistency range / 2, estimator difference / 2).
# This combination is a heuristic, not a calibrated interval.
# ====================================================================================================

def rel_range(values, ref):
    v = [x for x in values if np.isfinite(x)]
    return float((max(v) - min(v)) / ref) if len(v) >= 2 and ref else float("nan")


def block_bootstrap(pb: Problem, front_ids, tilt_ids, method, grid, ecfg, sigma2, sol, n, block, seed, budget_s):
    """Resample blocks of `block` consecutive tilt frames with replacement (front frames always kept)."""
    rng = np.random.default_rng(seed)
    tilt = list(tilt_ids)
    blocks = [tilt[i:i + block] for i in range(0, len(tilt), block)] or [[]]
    fs, t0, reduced = [], time.time(), False
    start = (np.log(sol["f"]), sol["delta_deg"], sol["phi_deg"])
    for b in range(n):
        if b >= 2 and (time.time() - t0) / b * (b + 1) > budget_s:     # the next repetition would exceed the budget
            reduced = True
            break
        pick = [blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]
        ids = list(front_ids) + [i for blk in pick for i in blk]
        sub = pb.subset(ids)
        r = estimate_run(sub, method, grid, ecfg, sigma2, start=start if method == "joint_delta" else None)
        fs.append(r["f"])
    fs = np.array(fs)
    return {"f": fs.tolist(), "n_done": int(len(fs)), "n_requested": int(n), "reduced": reduced,
            "rel_std": float(np.std(fs, ddof=1) / np.mean(fs)) if len(fs) > 1 else float("nan")}


def consistency(pb, front_ids, tilt_ids, axis_deg, method, grid, ecfg, sigma2, sol):
    tilt = list(tilt_ids)
    h = len(tilt) // 2
    start = (np.log(sol["f"]), sol["delta_deg"], sol["phi_deg"])
    def f_of(ids):
        if len(ids) < 4:
            return float("nan")
        return estimate_run(pb.subset(list(front_ids) + list(ids)), method, grid, ecfg, sigma2, start=start if method == "joint_delta" else None)["f"]
    ax = np.asarray([axis_deg[i] for i in tilt])
    horiz = [i for i, a in zip(tilt, ax) if (a < 45 or a >= 135)]
    vert = [i for i, a in zip(tilt, ax) if 45 <= a < 135]
    out = {"first_half": f_of(tilt[:h]), "second_half": f_of(tilt[h:]), "axis_horizontal": f_of(horiz), "axis_vertical": f_of(vert),
           "n_horizontal": len(horiz), "n_vertical": len(vert)}
    ref = sol["f"]
    out["halves_range"] = rel_range([out["first_half"], out["second_half"]], ref)
    out["axis_range"] = rel_range([out["axis_horizontal"], out["axis_vertical"]], ref)
    out["range"] = float(np.nanmax([out["halves_range"], out["axis_range"]])) if np.isfinite([out["halves_range"], out["axis_range"]]).any() else float("nan")
    return out


def combine(boot_rel_std, jointset_range, consistency_range, method_diff) -> float:
    parts = [2 * boot_rel_std, jointset_range / 2, consistency_range / 2, method_diff / 2]
    parts = [p for p in parts if np.isfinite(p)]
    return float(max(parts)) if parts else float("nan")


# ====================================================================================================
# verdict  (was HandCalib/hand_focal/verdict.py)
# GOOD / ROUGH / REJECT for the hand f (thresholds from the yaml, fixed before the first run) and Korean re-shoot advice.
# ====================================================================================================

def _le(x, t):
    return bool(np.isfinite(x) and x <= t)


def _ge(x, t):
    return bool(np.isfinite(x) and x >= t)


def judge(m: dict, vcfg: dict) -> dict:
    g, r = vcfg["good"], vcfg["rough"]
    checks_good = {
        "detect_rate": _ge(m["detect_rate"], g["detect_rate_min"]),
        "front_frames": m["front_frames"] >= g["front_frames_min"],
        "finger_extension": (not g["finger_extension_required"]) or bool(m["finger_extension_ok"]),
        "theta_p90": _ge(m["theta_p90"], g["theta_p90_min_deg"]),
        "axis_bins": m["axis_bins6"] >= g["axis_bins_min"],
        "reproj_rms": _le(m["reproj_rms_rel"], g["reproj_rms_rel_max"]),
        "jointset_range": _le(m["jointset_range"], g["jointset_range_max"]),
        "bootstrap": _le(m["bootstrap_rel_std"], g["bootstrap_rel_std_max"]),
        "consistency": _le(m["consistency_range"], g["consistency_range_max"]),
        "delta": _le(abs(m["delta_deg"]), g["delta_abs_max_deg"]),
        "sharpness": _ge(m["sharpness"], g["sharpness_min"]),
    }
    checks_rough = {
        "detect_rate": _ge(m["detect_rate"], r["detect_rate_min"]),
        "theta_p90": _ge(m["theta_p90"], r["theta_p90_min_deg"]),
        "axis_bins": m["axis_bins6"] >= r["axis_bins_min"],
        "reproj_rms": _le(m["reproj_rms_rel"], r["reproj_rms_rel_max"]),
        "jointset_range": _le(m["jointset_range"], r["jointset_range_max"]),
    }
    if m["n_target_frames"] < vcfg["min_target_frames"]:
        v = "REJECT"
    elif all(checks_good.values()):
        v = "GOOD"
    elif all(checks_rough.values()):
        v = "ROUGH"
    else:
        v = "REJECT"
    return {"verdict": v, "good_checks": checks_good, "rough_checks": checks_rough, "failed_good": [k for k, ok in checks_good.items() if not ok],
            "failed_rough": [k for k, ok in checks_rough.items() if not ok]}


def advice(m: dict, j: dict) -> list:
    a = []
    f = set(j["failed_good"])
    if "theta_p90" in f:
        a.append(f"기울기가 부족합니다(p90 {m['theta_p90']:.0f}°). 손을 30~45°까지 크게 기울이세요")
    if "axis_bins" in f:
        dom = m.get("axis_dominant")
        a.append("상하 기울기만 있고 좌우가 없습니다" if dom == "vertical_only" else "좌우 기울기만 있고 상하가 없습니다" if dom == "horizontal_only"
                 else "기울기 방향이 일부에 몰려 있습니다. 상하좌우로 고르게 기울이세요")
    if m["front_jitter_px"] == m["front_jitter_px"] and ("front_frames" in f or m["front_jitter_rel"] > 0.01):
        a.append(f"정면 구간에서 손이 충분히 정지하지 않았습니다(지터 {m['front_jitter_px']:.1f} px). 손을 폰과 평행하게 1초 이상 멈추세요")
    if "finger_extension" in f:
        a.append("손가락이 펴지지 않았습니다")
    if m.get("edge_frame_rate", 0) > 0.05:
        a.append("손이 화면 가장자리에 걸렸습니다. 영상 중앙에서 시작하세요")
    if np.isfinite(m.get("method_diff", np.nan)) and m["method_diff"] > 0.04:
        a.append(f"정면 가정 민감도가 큽니다(두 방식 차이 {100 * m['method_diff']:.1f}%)")
    return a


SHOOTING_GUIDE = [
    "정면 구간: 손바닥을 렌즈 쪽이 아니라 폰 화면과 평행하게 두고, 손을 영상 중앙에서 1초 이상 멈춥니다(처음 약 3초).",
    "그 뒤 상하좌우로 30~45°까지 천천히 기울입니다(한 방향만이 아니라 네 방향 모두).",
    "거리는 30~50 cm, 흐려지지 않게 천천히 움직이고 손가락은 쫙 편 상태를 유지합니다.",
]


# ====================================================================================================
# synthetic  (was HandCalib/hand_focal/synthetic.py)
# Synthetic study: a planar 17-point hand (wrist, 4 x MCP/PIP/DIP/tip; palm ~80 x 100 mm, phalanges 16-44 mm) seen by a pinhole
# camera, analysed by the same analysis core as the real videos (thumb joints 1-4 are masked; front / tilt frames are given).
#
# Error = f_est / f_true - 1. This is the only place in experiment 14 where an error against a known f is computed.
# ====================================================================================================

# hand layout in mm (x right, y towards the fingertips = -y in the image when upright); index .. pinky
_MCP = {"index": (-30.0, -92.0), "middle": (-9.0, -100.0), "ring": (12.0, -95.0), "pinky": (31.0, -84.0)}
_BONES = {"index": (40.0, 24.0, 18.0), "middle": (44.0, 28.0, 19.0), "ring": (41.0, 27.0, 18.0), "pinky": (32.0, 20.0, 16.0)}
_FAN = {"index": -6.0, "middle": -1.0, "ring": 4.0, "pinky": 10.0}
_IDS = {"index": [5, 6, 7, 8], "middle": [9, 10, 11, 12], "ring": [13, 14, 15, 16], "pinky": [17, 18, 19, 20]}


def synth_hand_points(arch_mm: float = 0.0) -> np.ndarray:
    """(21, 3) hand-frame points (thumb rows NaN). z = transverse arch offset along the palm normal (MCP row raised in the middle)."""
    P = np.full((21, 3), np.nan)
    P[0] = (0.0, 0.0, 0.0)
    for name, (x, y) in _MCP.items():
        a = np.radians(_FAN[name])
        d = np.array([np.sin(a), -np.cos(a)])
        pts = [np.array([x, y])]
        for L in _BONES[name]:
            pts.append(pts[-1] + L * d)
        z = -arch_mm * np.cos(np.pi * x / 80.0)           # towards the camera (negative z) at the middle of the MCP row
        for j, q in zip(_IDS[name], pts):
            P[j] = (q[0], q[1], z)
    P[[5, 9, 13, 17], 2] = [-arch_mm * np.cos(np.pi * _MCP[k][0] / 80.0) for k in ("index", "middle", "ring", "pinky")]
    return P


def _rot(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def synth_pose(theta_deg, azim_deg, inplane_deg):
    """Hand normal tilted by theta towards azimuth (image plane), after an in-plane rotation."""
    a = np.radians(azim_deg)
    Rt = _rot([-np.sin(a), np.cos(a), 0.0], np.radians(theta_deg)) if theta_deg else np.eye(3)
    return Rt @ _rot([0, 0, 1.0], np.radians(inplane_deg))


def synth_project(P, R, t, f, cx, cy):
    X = P @ R.T + t
    return np.c_[f * X[:, 0] / X[:, 2] + cx, f * X[:, 1] / X[:, 2] + cy]


def synth_generate(cond: dict, scfg: dict, rng) -> dict:
    f = float(scfg["f_px"])
    W, H = scfg["image"]
    cx, cy = W / 2.0, H / 2.0
    D = float(scfg["distance_mm"])
    P = synth_hand_points(cond.get("arch_mm", 0.0))
    P[:, 1] += 95.0                                         # centre the hand (wrist .. tips ~ 190 mm) on the optical axis
    vis_ids = [0] + list(range(5, 21))
    nF, nT = int(cond.get("front_frames", scfg["front_frames"])), int(scfg["tilt_frames"])
    phi = rng.uniform(0, 360)
    g0 = rng.uniform(-10, 10)
    c0 = np.array([rng.uniform(-30, 30), rng.uniform(-8, 8), D * rng.uniform(0.97, 1.05)])
    poses = []
    for _ in range(nF):
        v = cond["delta_deg"] * np.array([np.cos(np.radians(phi)), np.sin(np.radians(phi))]) + rng.normal(0, 1.0, 2)
        th, az = float(np.hypot(*v)), float(np.degrees(np.arctan2(v[1], v[0])))
        poses.append((synth_pose(th, az, g0 + rng.uniform(-5, 5)), c0 + rng.normal(0, 2.0, 3), th))
    lo, hi = cond["tilt_range_deg"]
    for _ in range(nT):
        th = rng.uniform(lo, hi)
        poses.append((synth_pose(th, rng.uniform(0, 360), rng.uniform(-30, 30)),
                      np.array([rng.uniform(-40, 40), rng.uniform(-10, 10), D * rng.uniform(0.95, 1.1)]), th))
    n = len(poses)
    kp = np.full((n, 21, 2), np.nan)
    bias_dir = rng.normal(size=(21, 2))
    bias_dir /= np.linalg.norm(bias_dir, axis=1, keepdims=True)
    noise = cond["noise"]
    for i, (R, t, th) in enumerate(poses):
        q = synth_project(P[vis_ids], R, t, f, cx, cy)
        bl = float(np.ptp(q, axis=0).max() * 1.1)
        sig = noise["px"] if "px" in noise else noise["rel"] * bl
        e = rng.normal(0, sig, q.shape)
        if cond.get("heavy_tail"):
            m = rng.random(len(q)) < 0.10
            e[m] *= 3.0
        q = q + e + cond.get("bias_rel", 0.0) * bl * (th / 45.0) * bias_dir[vis_ids]
        kp[i, vis_ids] = q
    mask = np.isfinite(kp).all(-1) & (kp[..., 0] >= 0) & (kp[..., 0] < W) & (kp[..., 1] >= 0) & (kp[..., 1] < H)
    box = np.full((n, 4), np.nan)
    for i in range(n):
        q = kp[i][mask[i]]
        lo2, hi2 = q.min(0), q.max(0)
        pad = 0.05 * (hi2 - lo2).max()
        box[i] = [lo2[0] - pad, lo2[1] - pad, hi2[0] + pad, hi2[1] + pad]
    times = np.r_[np.arange(nF) / 15.0 + 0.5, 3.0 + np.arange(nT) / 15.0]
    return {"kp": kp, "mask": mask, "present": np.ones(n, bool), "box": box, "cls": np.ones(n, int), "times": times,
            "size": (W, H), "front_idx": np.arange(nF), "tilt_start": 3.0, "f_true": f, "theta_true": [p[2] for p in poses]}


def expected_error(study: dict, jitter_rel: float, method: str) -> dict:
    """Noise-only condition (base, noise_rel*, noise_px3) whose median front jitter (relative) is closest to the video's."""
    cands = [k for k in study["summary"] if k == "base" or k.startswith("noise_")]
    best = min(cands, key=lambda k: abs(np.log(max(study["summary"][k]["front_jitter_rel_median"], 1e-9)) - np.log(max(jitter_rel, 1e-9))))
    s = study["summary"][best]
    return {"condition": best, "front_jitter_rel_condition": s["front_jitter_rel_median"], "front_jitter_rel_video": jitter_rel,
            "median_abs": s[method]["median_abs"], "p90_abs": s[method]["p90_abs"], "method": method,
            "note": "합성 연구에서 정면 지터가 가장 가까운 잡음 조건의 오차(δ=0, 휨·편향 없음 가정). 이 영상의 오차가 아니다."}


# ====================================================================================================
# report  (was HandCalib/hand_focal/report.py)
# Writers: figures, per_frame.csv, results.json, result.yaml, report.md (Korean). Points only - no video frames or images of the hand.
# ====================================================================================================

MANDATORY_14 = "이 f는 핀홀·주점 중심·왜곡 없음 가정에서 손 2D 키포인트만으로 얻은 추정값이며 정답 데이터와 비교하지 않았다."
METHOD_KO = {"fixed_template": "fixed_template(정면=평면 가정)", "joint_delta": "joint_delta(정면 기울기 δ 동시 추정)"}


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(jsonable(obj), ensure_ascii=False, indent=1))


def fmt(x, d=1, pct=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "nan"
    return f"{100 * x:.{d}f}%" if pct else f"{x:.{d}f}"


def derived(f: float, W: int, H: int, target_sizes) -> dict:
    L = max(W, H)
    fov = lambda s: math.degrees(2 * math.atan(s / (2 * f)))      # noqa: E731
    out = {"f_over_long_side": f / L, "fov_h_deg": fov(W), "fov_v_deg": fov(H), "fov_d_deg": fov(math.hypot(W, H)), "at_sizes": []}
    for tw, th in target_sizes:
        same = (tw >= th) == (W >= H) and abs(tw / th - W / H) < 0.02
        out["at_sizes"].append({"size": [tw, th], "f_px": f * max(tw, th) / L,
                                "note": None if same else "가로세로 비율/방향이 달라 긴 변 기준으로만 환산(같은 화각 크롭 가정 아님)"})
    return out


# ---------------------------------------------------------------- figures
def report_figures(out_dir: Path, a: dict, times, chosen: str, anyc: dict | None) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _font()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    made = []
    times = np.asarray(times)
    th = np.array([np.nan if v is None else v for v in a["theta2d_deg"]], float)
    ax_ = np.array([np.nan if v is None else v for v in a["axis2d_deg"]], float)
    role = np.array(a["role"])

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(times, th, ".-", lw=1, ms=3, color="#2a6fbb", label="θ2d")
    fw = a["front_window"]
    ax.axvspan(fw["t0"], fw["t1"], color="#999", alpha=0.25, label="정면 창")
    ax.axvline(a["tilt_start_s"], color="#444", ls="--", lw=1, label="기울기 구간 시작")
    if a.get("auto_transition_s") is not None:
        ax.axvline(a["auto_transition_s"], color="#c47a00", ls=":", lw=1, label="자동 전환")
    ax.set_xlabel("시간 (s)")
    ax.set_ylabel("θ2d (°)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    made.append(_save(fig, fig_dir / "theta2d_time.png"))

    fig, ax = plt.subplots(figsize=(4, 3))
    cov = a["axis_coverage"]
    ax.bar([f"{30 * k}-{30 * k + 30}" for k in range(6)], cov["counts"], color="#2a6fbb")
    ax.set_xlabel(f"압축 축 방향 (°), θ2d ≥ {cov['theta_min_deg']:g}°")
    ax.set_ylabel("프레임 수")
    ax.tick_params(axis="x", labelsize=7)
    made.append(_save(fig, fig_dir / "axis_distribution.png"))

    fig, ax = plt.subplots(figsize=(6, 3.4))
    for m, col in (("fixed_template", "#2a6fbb"), ("joint_delta", "#c47a00")):
        s = a["solutions"][m]
        if s.get("profile_cost"):
            c = np.array(s["profile_cost"], float)
            ax.plot(s["profile_f"], c / np.nanmin(c), "-", color=col, lw=1.5, label=m)
            ax.axvline(s["f"], color=col, ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("f (px, 정규화 해상도)")
    ax.set_ylabel("비용 / 최소")
    ax.set_ylim(0.9, 4)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    made.append(_save(fig, fig_dir / "f_profile.png"))

    pm = a["per_method"][chosen]
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    T = np.array(a["template"])
    fr = a["template_frames"][len(a["template_frames"]) // 2]
    if "obs_front_example" in a:
        P = np.array(a["obs_front_example"])
        s, R, t = similarity_align(T, P)
        Q = (s * (R @ T.T)).T + t
        ax.scatter(P[:, 0], P[:, 1], s=14, color="#2a6fbb", label=f"관측(프레임 {fr})")
        ax.scatter(Q[:, 0], Q[:, 1], s=30, facecolors="none", edgecolors="#c47a00", label="템플릿(닮음 정합)")
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.legend(fontsize=7)
    made.append(_save(fig, fig_dir / "template_overlay.png"))

    fig, ax = plt.subplots(figsize=(6, 3))
    pj = pm["diagnostics"]["per_joint_rms_px"]
    ax.bar([str(k) for k in pj], [np.nan if v is None else v for v in pj.values()], color="#2a6fbb")
    ax.set_xlabel("관절 번호 (OpenPose)")
    ax.set_ylabel("재투영 RMS (px)")
    made.append(_save(fig, fig_dir / "per_joint_residual.png"))

    fig, ax = plt.subplots(figsize=(4.5, 3))
    bf = pm["bootstrap"]["f"]
    if bf:
        ax.hist(bf, bins=min(15, max(5, len(bf) // 2)), color="#2a6fbb")
    ax.set_xlabel("부트스트랩 f (px)")
    ax.set_ylabel("횟수")
    made.append(_save(fig, fig_dir / "bootstrap_f.png"))

    if anyc:
        fig, ax = plt.subplots(figsize=(8, 3))
        for m, d in anyc["models"].items():
            ax.plot(anyc["times"], d["f"], ".-", ms=3, lw=1, label=f"AnyCalib {m}")
        ax.axvline(a["tilt_start_s"], color="#444", ls="--", lw=1)
        ax.set_xlabel("시간 (s)")
        ax.set_ylabel("f (px)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        made.append(_save(fig, fig_dir / "anycalib_f_time.png"))
    return made


def _save(fig, p: Path) -> str:
    import matplotlib.pyplot as plt
    fig.tight_layout()
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p.name


def _font():
    import matplotlib
    from matplotlib import font_manager
    for name in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "UnDotum", "Malgun Gothic"):
        if any(name == f.name for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    return None


# ---------------------------------------------------------------- tables
def per_frame_csv(path: Path, frames, tr, a, chosen, anyc) -> None:
    res_by_frame = {}
    sol = a["per_method"][chosen]
    for fi, r in zip(sol["diagnostics"]["frames_used"], sol["diagnostics"].get("frame_rms_px", [])):
        res_by_frame[fi] = r
    af = {}
    if anyc:
        m0 = next(iter(anyc["models"]))
        af = dict(zip(anyc["frames"], anyc["models"][m0]["f"]))
    bl = np.maximum(tr["box"][:, 2] - tr["box"][:, 0], tr["box"][:, 3] - tr["box"][:, 1])
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "video_frame", "time_s", "present", "cls", "box_conf", "box_long_px", "n_visible_points", "theta2d_deg", "axis2d_deg",
                    "role", "removed_reason", "reproj_rms_px", "anycalib_f_px"])
        for i, r in enumerate(frames):
            th, ax_ = a["theta2d_deg"][i], a["axis2d_deg"][i]
            w.writerow([r["index"], r["frame"], f"{r['time_s']:.4f}", int(tr["present"][i]), int(tr["cls"][i]), f"{tr['box_conf'][i]:.4f}",
                        "" if not np.isfinite(bl[i]) else f"{bl[i]:.2f}", int(tr["mask"][i].sum()),
                        "" if th is None or not np.isfinite(th) else f"{th:.3f}", "" if ax_ is None or not np.isfinite(ax_) else f"{ax_:.2f}",
                        a["role"][i], a["reason"][i], "" if i not in res_by_frame else f"{res_by_frame[i]:.3f}",
                        "" if i not in af else f"{af[i]:.2f}"])


def summary_row(name, a, chosen, anyc, duration) -> dict:
    if "per_method" not in a:
        return {"video": name, "duration_s": duration, "verdict": "REJECT", "f_hand": None, "unc": None, "fixed": None, "joint": None, "anycalib": None,
                "reason": a.get("failed")}
    pm = a["per_method"][chosen]
    edge = " (f 격자 경계)" if a["solutions"][chosen]["at_grid_edge"] else ""
    return {"video": name, "duration_s": duration, "verdict": pm["verdict"]["verdict"] + edge, "f_hand": pm["f"], "unc": pm["f_rel_uncertainty"],
            "fixed": a["solutions"]["fixed_template"]["f"], "joint": a["solutions"]["joint_delta"]["f"],
            "anycalib": (anyc["models"][next(iter(anyc["models"]))]["summary"]["median"] if anyc else None)}


# ---------------------------------------------------------------- report.md
def report_md(path: Path, ctx: dict) -> None:
    a, chosen, v = ctx["analysis"], ctx["estimator"], ctx["verdict"]
    pm = a["per_method"][chosen]
    f = pm["f"]
    L = []
    L.append(f"# 손 템플릿 초점거리 추정: {ctx['video']}")
    L.append("")
    label = "신뢰 가능한 f" if v == "GOOD" else ("대략적인 f(ROUGH, 신뢰 가능한 f가 아님)" if v == "ROUGH" else "참고용 값(REJECT, 신뢰 가능한 f가 아님)")
    L.append(f"- 판정(손 방식에만 적용): **{v}**")
    if a["solutions"][chosen]["at_grid_edge"]:
        L.append(f"- f_hand ({chosen}): **{f:.1f} px = f 격자 경계** (탐색 범위 {a['grid'][0]:.0f}–{a['grid'][-1]:.0f} px) — 내부 최솟값이 없어 추정값으로 쓸 수 없음, {label}")
    else:
        L.append(f"- f_hand ({chosen}): **{f:.1f} px** — {label}, 정규화 해상도 {ctx['W']}×{ctx['H']} 기준")
    L.append(f"- 휴리스틱 상대 불확실도 f_rel_uncertainty: {fmt(pm['f_rel_uncertainty'], 1, True)} (보정된 신뢰구간이 아님)")
    L.append(f"- 두 방식 값: fixed_template {a['solutions']['fixed_template']['f']:.1f} px, joint_delta {a['solutions']['joint_delta']['f']:.1f} px "
             f"(δ {a['solutions']['joint_delta']['delta_deg']:.1f}°, φ {a['solutions']['joint_delta']['phi_deg']:.0f}°); 정면 가정 민감도 {fmt(a['method_diff_rel'], 1, True)}")
    if ctx.get("anycalib"):
        for m, d in ctx["anycalib"]["models"].items():
            s = d["summary"]
            L.append(f"- AnyCalib {m} (별도 값, 손 방식과 무관): 중앙값 {s['median']:.1f} px, IQR {s['iqr'][0]:.1f}–{s['iqr'][1]:.1f}, "
                     f"p10–p90 {s['p10_p90'][0]:.1f}–{s['p10_p90'][1]:.1f}, 정면 {fmt(s['front_median'])} / 기울기 {fmt(s['tilt_median'])}, CV {fmt(s['cv'], 1, True)}")
    else:
        L.append("- AnyCalib: 실행하지 않음")
    if ctx.get("expected") and not a["solutions"][chosen]["at_grid_edge"]:
        e = ctx["expected"]
        L.append(f"- 합성 기준 예상 오차({e['condition']}, 정면 지터 {fmt(e['front_jitter_rel_condition'], 2, True)} vs 영상 {fmt(e['front_jitter_rel_video'], 2, True)}): "
                 f"중앙값 {fmt(e['median_abs'], 1, True)}, p90 {fmt(e['p90_abs'], 1, True)} — {e['note']}")
    L.append("")
    L.append(MANDATORY_14)
    L.append("")
    if pm["advice"] and v != "GOOD":
        L.append("## 재촬영 안내")
        L += [f"- {x}" for x in pm["advice"]]
        L.append("")
    L.append("## 판정 근거 (yaml 문턱값, 첫 실행 전에 고정)")
    met = pm["metrics"]
    L.append("| 항목 | 값 | GOOD 통과 | ROUGH 통과 |")
    L.append("|---|---|---|---|")
    keys = [("detect_rate", "검출률", lambda x: fmt(x, 3)), ("front_frames", "정면 프레임 수", str), ("finger_extension", "손가락 펴짐", str),
            ("theta_p90", "θ2d p90 (°)", fmt), ("axis_bins", "축 구간(6) 커버", None), ("reproj_rms", "재투영 RMS / 박스", None),
            ("jointset_range", "관절 집합 범위", lambda x: fmt(x, 1, True)), ("bootstrap", "부트스트랩 상대 표준편차", None),
            ("consistency", "일관성 범위", None), ("delta", "|δ| (°)", None), ("sharpness", "프로파일 선명도", None)]
    vals = {"detect_rate": met["detect_rate"], "front_frames": met["front_frames"], "finger_extension": met["finger_extension_ok"],
            "theta_p90": met["theta_p90"], "axis_bins": met["axis_bins6"], "reproj_rms": fmt(met["reproj_rms_rel"], 2, True),
            "jointset_range": met["jointset_range"], "bootstrap": fmt(met["bootstrap_rel_std"], 2, True), "consistency": fmt(met["consistency_range"], 1, True),
            "delta": fmt(abs(met["delta_deg"])), "sharpness": fmt(met["sharpness"], 2)}
    g, r = pm["verdict"]["good_checks"], pm["verdict"]["rough_checks"]
    for k, name, fn in keys:
        x = vals[k]
        s = fn(x) if fn else str(x)
        L.append(f"| {name} | {s} | {'O' if g.get(k) else 'X'} | {('O' if r[k] else 'X') if k in r else '-'} |")
    L.append(f"| 대상 프레임 수 | {met['n_target_frames']} | 최소 {ctx['min_target_frames']} | |")
    L.append("")
    L.append("## 진단")
    d = pm["diagnostics"]
    L.append(f"- 재투영 RMS {fmt(d['reproj_rms_px'], 2)} px ({fmt(d['reproj_rms_rel'], 2, True)} of 박스 긴 변), 비용 +15% {d['cost_plus15']:.3g}, −15% {d['cost_minus15']:.3g}, 최소 {d['cost']:.3g}")
    L.append(f"- θ2d (기울기 구간) 중앙값 {fmt(a['theta_tilt']['median'])}°, p90 {fmt(a['theta_tilt']['p90'])}°, 최대 {fmt(a['theta_tilt']['max'])}°; 축 6구간 {a['axis_coverage']['counts']}; "
             f"PnP 법선 방위 8구간 커버 {d['azimuth_bins8']}/8")
    L.append(f"- 정면 창 {fmt(a['front_window']['t0'], 2)}–{fmt(a['front_window']['t1'], 2)} s, 템플릿 프레임 {a['front_frames']}장, 지터 {fmt(a['front_jitter_px'], 2)} px "
             f"({fmt(a['front_jitter_rel'], 2, True)}), 손가락 펴짐 최소 {fmt(a['finger_extension']['min'], 3)}")
    L.append(f"- 기울기 구간 시작 {a['tilt_start_s']:.2f} s ({a['phase_mode']}), 자동 전환 {fmt(a.get('auto_transition_s'), 2)} s")
    L.append(f"- 사용 프레임: 정면 {a['n_front_used']}, 기울기 {a['n_tilt_used']}; 제외 사유 {ctx['removed']}")
    L.append(f"- 손 크기(박스 긴 변 중앙값) {fmt(a['hand_size_px']['box_long_median'])} px, 가장자리 프레임 비율 {fmt(a['edge_frame_rate'], 1, True)}, "
             f"좌우 클래스 일관성 {fmt(a['lr']['consistency'], 3) if a.get('lr') else 'nan'} (다수 클래스 {ctx['lr_name']})")
    L.append(f"- 관절 집합별 f: " + ", ".join(f"{k} {fmt(x)}" for k, x in pm["jointsets"].items()))
    b = pm["bootstrap"]
    L.append(f"- 블록 부트스트랩 {b['n_done']}/{b['n_requested']}회{' (시간 예산으로 축소)' if b['reduced'] else ''}, 상대 표준편차 {fmt(b['rel_std'], 2, True)}")
    c = pm["consistency"]
    L.append(f"- 일관성: 앞/뒤 절반 {fmt(c['first_half'])} / {fmt(c['second_half'])}, 수평축/수직축 {fmt(c['axis_horizontal'])} ({c['n_horizontal']}장) / "
             f"{fmt(c['axis_vertical'])} ({c['n_vertical']}장)")
    L.append(f"- 순서 점검(OpenPose): {ctx['order_check']}")
    L.append("")
    L.append("## 환산 (f_hand 기준)")
    dv = ctx["derived"]
    L.append(f"- f / 긴 변 {dv['f_over_long_side']:.4f}, 화각 가로 {dv['fov_h_deg']:.1f}°, 세로 {dv['fov_v_deg']:.1f}°, 대각 {dv['fov_d_deg']:.1f}°")
    for s in dv["at_sizes"]:
        L.append(f"- {s['size'][0]}×{s['size'][1]}: {s['f_px']:.1f} px" + (f" ({s['note']})" if s["note"] else ""))
    L.append("")
    if ctx["warnings"]:
        L.append("## 경고")
        L += [f"- {w}" for w in ctx["warnings"]]
        L.append("")
    L.append("## 촬영 가이드")
    L += [f"- {x}" for x in SHOOTING_GUIDE]
    L.append("")
    L.append("## 한계")
    L.append("- 핀홀, 주점 = 영상 중심, 정사각 화소, 왜곡 없음(기본값)을 가정한다. 손바닥과 펴진 손가락을 평면으로 본다.")
    L.append("- 키포인트는 WiLoR 파이프라인의 YOLO 검출 모듈(detector.pt)의 2D 출력만 사용했다. 이 검출기의 2D 라벨 생성 방식은 확인하지 못했다.")
    L.append("- 불확실도 결합식 max(2·부트스트랩, 집합범위/2, 일관성/2, 방식차/2)은 휴리스틱이며 정답 데이터로 보정되지 않았다.")
    L.append("- 정답 f가 없어 이 영상의 오차는 계산하지 않았다. 합성 기준 예상 오차는 합성 조건의 값이다.")
    path.write_text("\n".join(L) + "\n")


# ====================================================================================================
# provenance  (was HandCalib/hand_focal/provenance.py)
# metadata.json: git state, config copy and hash, library / tool versions, weight hashes.
# ====================================================================================================

def git(*args) -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True).strip()


def git_state() -> dict:
    dirty = git("status", "--porcelain", "--untracked-files=no", "--", "HandCalib")
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "git_dirty": bool(dirty),
            "dirty_files": dirty.splitlines()[:50], "anycalib_submodule": _submodule()}


def _submodule():
    try:
        return git("-C", str(REPO_ROOT / "HandCalib" / "AnyCalib"), "rev-parse", "HEAD")
    except Exception:
        return None


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def versions() -> dict:
    import cv2
    import matplotlib
    import numpy
    import scipy
    import yaml
    out = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": numpy.__version__, "opencv": cv2.__version__,
           "scipy": scipy.__version__, "matplotlib": matplotlib.__version__, "pyyaml": yaml.__version__}
    try:
        import torch
        out["torch"] = torch.__version__
    except Exception:
        out["torch"] = None
    return out


def provenance_build(cfg_text: str, cfg_path: str, extra: dict) -> dict:
    return {"git": git_state(), "config_path": cfg_path, "config_sha256": sha256_text(cfg_text), "config": cfg_text, "versions": versions(),
            "worktrees": {"used": [], "note": "다른 실험 코드는 쓰지 않음. AnyCalib는 같은 체크아웃의 models/anycalib.py(01 어댑터, main에 있음)를 수정 없이 import"},
            **extra}


# ====================================================================================================
# anycalib_ref  (was HandCalib/hand_focal/anycalib_ref.py)
# Separate AnyCalib f on the same PNG frames (01's model_id / cam_id / predict path). Never an input to the hand method.
#
# predict() resizes / centre-crops internally and returns intrinsics already mapped back to the input image (reverse_scale_and_shift),
# so f = (fx + fy) / 2 is in the normalised resolution of the PNGs. Offline: HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are set and
# torch.hub downloads are blocked; the hub directory is listed before and after.
# ====================================================================================================

class DownloadBlocked(RuntimeError):
    pass


def hub_snapshot(hub_dir) -> dict:
    root = resolve(hub_dir)
    out = {}
    if root.exists():
        for p in root.rglob("*"):
            if p.is_file():
                out[str(p.relative_to(root))] = p.stat().st_size
    return out


@contextlib.contextmanager
def offline(env: dict):
    """Offline environment variables and a torch.hub download guard (raises instead of downloading)."""
    import torch.hub
    old = {k: os.environ.get(k) for k in env}
    os.environ.update({k: str(v) for k, v in env.items()})
    orig = torch.hub.download_url_to_file

    def _blocked(url, *a, **k):
        raise DownloadBlocked(f"download blocked: {url}")
    torch.hub.download_url_to_file = _blocked
    try:
        yield
    finally:
        torch.hub.download_url_to_file = orig
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def available_models(acfg: dict) -> list:
    hub = resolve(acfg["torch_hub_dir"]) / "anycalib"
    ids = [acfg["model_id"]] + list(acfg.get("extra_models") or [])
    return [(m, hub / f"{m}.pt") for m in ids if (hub / f"{m}.pt").exists() or m == acfg["model_id"]]


def load_rgb(png):
    import cv2
    im = cv2.imread(str(png), cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"cannot read {png}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def predict_frames(model_id: str, cam_id: str, images, device: str, batch: int = 8) -> np.ndarray:
    """images: list of (H, W, 3) uint8 RGB arrays (same size). Returns (N, 4) fx, fy, cx, cy in the input resolution."""
    import torch
    from models.anycalib import AnyCalibAdapter        # 01's adapter (main, unchanged)
    ad = AnyCalibAdapter(model_id=model_id, cam_id=cam_id, device=device).build()
    out = []
    for i in range(0, len(images), batch):
        x = torch.from_numpy(np.stack(images[i:i + batch])).permute(0, 3, 1, 2).float() / 255.0
        p = ad.predict(x)
        intr = p["intrinsics"]
        intr = torch.stack([torch.as_tensor(v) for v in intr]) if isinstance(intr, (list, tuple)) else torch.as_tensor(intr)
        out.append(intr.reshape(-1, intr.shape[-1])[:, :4].detach().cpu().double().numpy())
    del ad
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return np.concatenate(out, 0)


def anycalib_summarize(f: np.ndarray, times: np.ndarray, t_split: float) -> dict:
    f = np.asarray(f, float)
    ok = np.isfinite(f)
    g = f[ok]
    if not len(g):
        return {"n": 0}
    q10, q25, q50, q75, q90 = np.percentile(g, [10, 25, 50, 75, 90])
    t = np.asarray(times)[ok]
    return {"n": int(len(g)), "median": float(q50), "iqr": [float(q25), float(q75)], "p10_p90": [float(q10), float(q90)],
            "front_median": float(np.median(g[t < t_split])) if (t < t_split).any() else None,
            "tilt_median": float(np.median(g[t >= t_split])) if (t >= t_split).any() else None,
            "cv": float(np.std(g) / np.mean(g))}


def anycalib_run(frames: list, acfg: dict, t_split: float, log=print) -> dict:
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    sel = frames[::max(1, int(acfg.get("stride", 1)))]
    before = hub_snapshot(acfg["torch_hub_dir"])
    images = [load_rgb(r["png"]) for r in sel]
    times = np.array([r["time_s"] for r in sel])
    res = {"frames": [int(r["index"]) for r in sel], "times": times.tolist(), "models": {}}
    with offline(acfg["offline_env"]):
        for m, wpath in available_models(acfg):
            if not wpath.exists():
                raise FileNotFoundError(f"AnyCalib weight missing: {wpath} (name {m}.pt, source github.com/javrtg/AnyCalib release v1.0.0)")
            intr = predict_frames(m, acfg["cam_id"], images, acfg["device"])
            f = (intr[:, 0] + intr[:, 1]) / 2
            res["models"][m] = {"weight": str(wpath), "weight_sha256": sha256_file(wpath), "fx": intr[:, 0].tolist(), "fy": intr[:, 1].tolist(),
                                "cx": intr[:, 2].tolist(), "cy": intr[:, 3].tolist(), "f": f.tolist(), "summary": anycalib_summarize(f, times, t_split)}
            log(f"  AnyCalib {m}: 중앙값 f {res['models'][m]['summary']['median']:.1f} px ({len(f)}장)")
    skipped = [m for m in (acfg.get("extra_models") or []) if m not in res["models"]]
    after = hub_snapshot(acfg["torch_hub_dir"])
    new = sorted(set(after) - set(before))
    changed = sorted(k for k in before if k in after and after[k] != before[k])
    res["hub_check"] = {"dir": str(resolve(acfg["torch_hub_dir"])), "n_before": len(before), "n_after": len(after), "new_files": new, "changed_files": changed}
    res["skipped_extra_models"] = skipped
    if new or changed:
        raise RuntimeError(f"torch hub cache changed during AnyCalib: new {new}, changed {changed}")
    return res


# ====================================================================================================
# analysis  (was HandCalib/hand_focal/analysis.py)
# Analysis core shared by real videos and the synthetic study: tracked 2D keypoints -> template, theta2d, f (both estimators),
# diagnostics, uncertainty and the verdict inputs. Uses 2D keypoints, their mask and the box only.
# ====================================================================================================

METHODS = ("fixed_template", "joint_delta")


def principal_point(ecfg, W, H):
    pp = ecfg.get("principal_point", "center")
    return (W / 2.0, H / 2.0) if pp == "center" else (float(pp[0]), float(pp[1]))


def undistort(kp, ecfg, W, H):
    """Optional distortion [k1, k2, p1, p2] with the convention K_d = [[L, 0, cx], [0, L, cy]] (L = long side), applied once."""
    d = ecfg.get("distortion")
    if not d:
        return kp
    cx, cy = principal_point(ecfg, W, H)
    L = float(max(W, H))
    Kd = np.array([[L, 0, cx], [0, L, cy], [0, 0, 1.0]])
    flat = kp.reshape(-1, 1, 2).astype(np.float64)
    ok = np.isfinite(flat).all(-1).ravel()
    out = flat.copy()
    out[ok] = cv2.undistortPoints(flat[ok], Kd, np.asarray(d, float), P=Kd)
    return out.reshape(kp.shape)


def _theta(T, kp, mask, present, ids):
    n = len(kp)
    th, ax = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        if not present[i]:
            continue
        v = mask[i, ids]
        if v.sum() >= 6:
            th[i], ax[i] = affine_tilt(T[v], kp[i, ids][v])
    return th, ax


def axis_coverage(theta, axis, ids, theta_min=10.0, min_count=2):
    a = np.asarray([axis[i] for i in ids if np.isfinite(theta[i]) and theta[i] >= theta_min])
    counts = np.bincount((a // 30).astype(int) % 6, minlength=6) if len(a) else np.zeros(6, int)
    bins = int((counts >= min_count).sum())
    horiz = counts[[0, 5]].sum()                 # compressed image axis near 0 / 180 deg: left-right tilt
    vert = counts[[2, 3]].sum()                  # near 90 deg: up-down tilt
    dom = None
    if len(a) and vert > 0.85 * len(a):
        dom = "vertical_only"
    elif len(a) and horiz > 0.85 * len(a):
        dom = "horizontal_only"
    return {"counts": counts.tolist(), "bins": bins, "dominant": dom, "theta_min_deg": theta_min, "min_count": min_count}


def analyze(kp, mask, present, box, cls, times, size_wh, cfg, front_idx=None, tilt_start=None, methods_full=("fixed_template",),
            n_boot=None, seed=0, budget_s=None, log=None, wrist_weight=None) -> dict:
    t_start_clock = time.time()
    timings = {}
    W, H = size_wh
    J, ecfg, pcfg = cfg["joints"], cfg["estimate"], cfg["phases"]
    est_ids = list(J["sets"][J["estimate_set"]])
    kp = undistort(np.asarray(kp, float), ecfg, W, H)
    mask, present = np.asarray(mask, bool), np.asarray(present, bool)
    times = np.asarray(times, float)
    n = len(times)
    bl = box_long(np.asarray(box, float))
    warnings = []

    # ---- front window and template
    if front_idx is None:
        fw = front_window(times, present, kp, mask, box, pcfg)
        warnings += fw["warnings"]
    else:
        fw = {"idx": np.asarray(front_idx, int), "detect_rate": float(present[front_idx].mean()), "speed_rel": float("nan"),
              "t0": float(times[front_idx[0]]), "t1": float(times[front_idx[-1]]), "warnings": []}
    fr_full = [int(i) for i in fw["idx"] if present[i] and mask[i, est_ids].all()]
    base = {"warnings": warnings, "front_window": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in fw.items()},
            "front_frames": len(fr_full), "detect_rate": float(present.mean()) if n else 0.0, "n_frames": n}
    if len(fr_full) < 3:
        base["warnings"].append("모든 관절이 보이는 정면 프레임이 3장 미만이라 템플릿을 만들 수 없습니다")
        base["failed"] = "no_template"
        return base
    tpl = template_build(kp[fr_full][:, est_ids], int(J["template_align_iters"]))
    T = tpl["template"]
    jitter_px = tpl["jitter_rms_px"]
    jitter_rel = jitter_px / float(np.nanmedian(bl[fr_full]))
    ext = finger_extension(kp[fr_full])
    if ext["min"] < pcfg["finger_extension_min"]:
        base["warnings"].append(f"정면 구간 손가락 펴짐 지표가 낮습니다(최소 {ext['min']:.3f} < {pcfg['finger_extension_min']})")

    # ---- theta2d, phases
    theta, axis = _theta(T, kp, mask, present, est_ids)
    t_auto = auto_transition(times, theta, present, pcfg)
    if tilt_start is None:
        t_nom = float(pcfg["front_phase_s"])
        if t_auto is not None and abs(t_auto - t_nom) > pcfg["auto_warn_diff_s"]:
            base["warnings"].append(f"자동 전환 시각({t_auto:.2f} s)이 명목 {t_nom:.1f} s와 {pcfg['auto_warn_diff_s']} s 넘게 다릅니다")
        tilt_start = t_auto if (pcfg["phase_mode"] == "auto" and t_auto is not None) else t_nom
    tilt_all = tilt_indices(times, present, tilt_start)

    # ---- estimation problem (front window frames with all joints + tilt frames)
    cx, cy = principal_point(ecfg, W, H)
    huber = ecfg["huber_rel"] * np.where(np.isfinite(bl), bl, 0.0)
    used = sorted(set(fr_full) | set(int(i) for i in tilt_all))

    def problem(set_ids):
        pos = [est_ids.index(j) for j in set_ids]
        frames = [(kp[i][set_ids], mask[i][set_ids]) for i in range(n)]
        w = None if wrist_weight is None or wrist_weight == 1.0 else np.array([float(wrist_weight) if j == 0 else 1.0 for j in set_ids])
        return Problem(T[pos], frames, cx, cy, huber, ecfg["min_points"], weights=w).subset(used)

    pb = problem(est_ids)
    tilt_ids = [i for i in pb.keep if i in set(int(x) for x in tilt_all)]
    front_ids = [i for i in pb.keep if i in set(fr_full)]
    grid = f_grid(ecfg, max(W, H))
    sigma2 = max(jitter_px ** 2, 0.25)
    t0 = time.time()
    sols = {m: estimate_run(pb, m, grid, ecfg, sigma2) for m in METHODS}
    timings["estimate_s"] = time.time() - t0
    for m in METHODS:
        if sols[m]["at_grid_edge"]:
            edge_side = "상한" if sols[m]["f"] >= grid[-1] * (1 - 1e-6) else "하한"
            warnings.append(f"{m}: 비용 최소가 f 격자 {edge_side}({sols[m]['f']:.0f} px)에 걸려 내부 최솟값이 없습니다. 이 값은 추정값이 아니라 격자 경계이며, "
                            "부트스트랩·일관성 값도 경계에 고정되어 변동이 0으로 보일 수 있습니다")

    # ---- per-frame bookkeeping
    role = np.array(["none"] * n, dtype=object)
    reason = np.array([""] * n, dtype=object)
    keep = set(pb.keep)
    for i in range(n):
        if not present[i]:
            reason[i] = "not_detected"
        elif i in set(fr_full):
            role[i] = "front"
        elif i in set(int(x) for x in fw["idx"]):
            reason[i] = "front_window_missing_joints"
        elif times[i] < tilt_start:
            reason[i] = "front_phase_outside_window"
        elif i in keep:
            role[i] = "tilt"
        else:
            reason[i] = "too_few_points"
    edge = np.zeros(n, bool)
    for i in np.flatnonzero(present):
        b = box[i]
        edge[i] = bool(b[0] <= 2 or b[1] <= 2 or b[2] >= W - 2 or b[3] >= H - 2)
    cls_p = np.asarray(cls)[present]
    lr = None
    if len(cls_p):
        vals, cnt = np.unique(cls_p, return_counts=True)
        lr = {"majority_cls": int(vals[np.argmax(cnt)]), "consistency": float(cnt.max() / cnt.sum()), "counts": {int(v): int(c) for v, c in zip(vals, cnt)}}
    th_tilt = theta[tilt_ids] if tilt_ids else np.zeros(0)
    th_tilt = th_tilt[np.isfinite(th_tilt)]
    cov = axis_coverage(theta, axis, tilt_ids)
    out = dict(base)
    out.update({
        "template": T.tolist(), "template_frames": fr_full, "obs_front_example": kp[fr_full[len(fr_full) // 2]][est_ids].tolist(), "front_jitter_px": jitter_px, "front_jitter_rel": jitter_rel, "finger_extension": ext,
        "finger_extension_ok": bool(ext["min"] >= pcfg["finger_extension_min"]),
        "theta2d_deg": theta.tolist(), "axis2d_deg": axis.tolist(), "tilt_start_s": float(tilt_start), "auto_transition_s": t_auto,
        "phase_mode": pcfg["phase_mode"] if front_idx is None else "given",
        "theta_tilt": {"median": float(np.median(th_tilt)) if len(th_tilt) else float("nan"), "p90": float(np.percentile(th_tilt, 90)) if len(th_tilt) else float("nan"),
                       "max": float(th_tilt.max()) if len(th_tilt) else float("nan")},
        "axis_coverage": cov, "n_target_frames": len(pb.keep), "n_front_used": len(front_ids), "n_tilt_used": len(tilt_ids),
        "role": role.tolist(), "reason": reason.tolist(), "edge_frame_rate": float(edge[present].mean()) if present.any() else 0.0,
        "lr": lr, "hand_size_px": {"box_long_median": float(np.nanmedian(bl[present])) if present.any() else float("nan")},
        "sigma2_px2": sigma2, "grid": grid.tolist(), "solutions": sols, "principal_point": [cx, cy],
        "method_diff_rel": abs(sols["fixed_template"]["f"] - sols["joint_delta"]["f"]) / sols["fixed_template"]["f"],
    })
    # ---- full uncertainty, diagnostics and verdict per method
    ucfg = cfg["uncertainty"]
    nb = int(ucfg["n_bootstrap"] if n_boot is None else n_boot)
    budget = float(ucfg["time_budget_s"] if budget_s is None else budget_s)
    box_med = float(np.nanmedian(bl[pb.keep]))
    out["per_method"] = {}
    for m in methods_full:
        t1 = time.time()
        sol = sols[m]
        start = (np.log(sol["f"]), sol["delta_deg"], sol["phi_deg"])
        diag = diagnostics(pb, sol, box_med, est_ids, tilt_ids)
        js = {}
        for name in J["uncertainty_sets"]:
            sub = problem(list(J["sets"][name]))
            js[name] = estimate_run(sub, m, grid, ecfg, sigma2, start=start if m == "joint_delta" else None)["f"] if len(sub.keep) >= 3 else float("nan")
        js_range = rel_range(list(js.values()), sol["f"])
        boot = block_bootstrap(pb, front_ids, tilt_ids, m, grid, ecfg, sigma2, sol, nb, int(ucfg["block_frames"]), seed, budget)
        cons = consistency(pb, front_ids, tilt_ids, axis, m, grid, ecfg, sigma2, sol)
        comb = combine(boot["rel_std"], js_range, cons["range"], out["method_diff_rel"])
        metrics = {"detect_rate": out["detect_rate"], "front_frames": out["front_frames"], "finger_extension_ok": out["finger_extension_ok"],
                   "theta_p90": out["theta_tilt"]["p90"], "axis_bins6": cov["bins"], "axis_dominant": cov["dominant"], "reproj_rms_rel": diag["reproj_rms_rel"],
                   "jointset_range": js_range, "bootstrap_rel_std": boot["rel_std"], "consistency_range": cons["range"], "delta_deg": sol["delta_deg"],
                   "sharpness": diag["sharpness"], "n_target_frames": out["n_target_frames"], "front_jitter_px": jitter_px, "front_jitter_rel": jitter_rel,
                   "edge_frame_rate": out["edge_frame_rate"], "method_diff": out["method_diff_rel"]}
        j = judge(metrics, cfg["verdict"])
        out["per_method"][m] = {"f": sol["f"], "diagnostics": diag, "jointsets": js, "jointset_range": js_range, "bootstrap": boot,
                                "consistency": cons, "f_rel_uncertainty": comb, "metrics": metrics, "verdict": j, "advice": advice(metrics, j),
                                "time_s": time.time() - t1}
    timings["total_s"] = time.time() - t_start_clock
    out["timings"] = timings
    return out


# ====================================================================================================
# diagnostic  (was HandCalib/hand_focal/diagnostic.py)
# Experiment 14 diagnostics (not pre-registered; never reported as f). Reads the official runs' detection cache and frame list.
#
# D1  cost profiles on frame subsets (theta2d <= 30 / 45 / 60 deg, boxes not touching the image edge) x joint sets
# D2  per-frame contribution: slope of each frame's cost in log f, against theta2d, edge contact and frame RMS
# D3  same-pose synthetic replay: per-frame PnP poses at an assumed f (0.7 / 1.0 / 1.3 x long side) -> planar template projected,
#     real masks kept, front-jitter noise -> fixed_template profile
# D4  per-joint residuals by theta2d bin (at the reference f = 1.0 x long side)
#
# Classification rule (fixed before the run):
#   - D3 synthetic replays also end at the upper edge of the official grid for every assumed f  -> pose distribution (shooting)
#   - D3 synthetic minima are interior and within 10 % of the assumed f for every assumed f, while the real data end at the edge
#     -> detector / finger planarity
#   - otherwise -> undetermined
# ====================================================================================================

def load_official(run_dir: Path, cfg: dict) -> dict:
    """Detections (cached npz copy) + frame list from an official run; tracking and the analysis core are re-run unchanged."""
    res = json.loads((run_dir / "results.json").read_text())
    det = load_detections(run_dir / "detections.npz")
    frames = res["frames"]["list"]
    check_alignment(frames, det)
    W, H = res["normalization"]["output"]["display_w"], res["normalization"]["output"]["display_h"]
    times = np.array([r["time_s"] for r in frames])
    tr = track(det, times, (W, H), cfg["tracking"], front_start_s=float(cfg["phases"]["front_skip_s"]))
    a = analyze(tr["kp"], tr["mask"], tr["present"], tr["box"], tr["cls"], times, (W, H), cfg, methods_full=())
    return {"res": res, "tr": tr, "a": a, "W": W, "H": H, "times": times}


def frame_cost_matrix(T, kp, mask, ids_frames, set_pos, set_ids, grid, cx, cy, huber):
    """C[i, k] = Huber mean cost of frame ids_frames[i] at grid[k] (mean over frames = the estimator's cost)."""
    C = np.full((len(ids_frames), len(grid)), np.nan)
    R = np.full((len(ids_frames), len(grid)), np.nan)
    for k, f in enumerate(grid):
        K = K_of(f, cx, cy)
        for r, i in enumerate(ids_frames):
            v = mask[i][set_ids]
            if v.sum() < 4:
                continue
            res, _, _ = frame_fit(T[set_pos], kp[i][set_ids], v, K)
            C[r, k] = float(_huber(res, huber[i]).mean())
            R[r, k] = float(np.sqrt(np.mean(res ** 2)))
    return C, R


# ====================================================================================================
# analysis16  (was HandCalib/detector_swap/analysis16.py)
# Diagnostics on top of 14's per-detector run: cost profiles (official and extended grid) per joint set, and 2D differences
# between detectors on the same frames and hand. Profile minima are diagnostics, never f values.
# ====================================================================================================

def swap_grids(pcfg, cfg16, L):
    off = f_grid(pcfg["estimate"], L)
    lo, hi = cfg16["profiles"]["extended_ratio_range"]
    ext = np.geomspace(lo * L, hi * L, int(cfg16["profiles"]["extended_n"]))
    return off, ext


def swap_profile_stats(grid, c):
    ok = np.isfinite(c)
    if ok.sum() < 3:
        return {"interior": None, "at": "none", "min_at_px": None, "sharpness": None}
    k = int(np.nanargmin(c))
    at = "interior" if 0 < k < len(c) - 1 else ("lower" if k == 0 else "upper")
    lg = np.log(grid)
    side = [np.interp(lg[k] + d, lg[ok], c[ok], left=np.nan, right=np.nan) for d in (np.log(1.15), -np.log(1.15))]
    side = [s for s in side if np.isfinite(s)]
    return {"interior": at == "interior", "at": at, "min_at_px": float(grid[k]), "sharpness": float(min(side) / c[k]) if side and c[k] > 0 else None}


def swap_profiles(run_dir, pcfg, cfg16):
    L0 = load_official(run_dir, pcfg)
    a, tr, W, H = L0["a"], L0["tr"], L0["W"], L0["H"]
    if "template" not in a:
        return {"failed": a.get("failed")}, L0
    L = float(max(W, H))
    off, ext = swap_grids(pcfg, cfg16, L)
    J = pcfg["joints"]
    est_ids = list(J["sets"][J["estimate_set"]])
    T = np.array(a["template"])
    role = np.array(a["role"])
    used = [i for i in range(len(role)) if role[i] in ("front", "tilt")]
    cx, cy = a["principal_point"]
    bl = box_long(tr["box"])
    huber = pcfg["estimate"]["huber_rel"] * np.where(np.isfinite(bl), bl, 0.0)
    out = {"official_grid": off.tolist(), "extended_grid": ext.tolist(), "n_used": len(used), "sets": {}}
    for s in cfg16["profiles"]["joint_sets"]:
        ids = list(J["sets"][s])
        pos = [est_ids.index(j) for j in ids]
        res = {}
        for gname, g in (("official", off), ("extended", ext)):
            C, _ = frame_cost_matrix(T, tr["kp"], tr["mask"], used, pos, ids, g, cx, cy, huber)
            ok = np.isfinite(C).all(1)
            prof = C[ok].mean(0) if ok.sum() else np.full(len(g), np.nan)
            res[gname] = {"profile": prof.tolist(), "n_frames": int(ok.sum()), **swap_profile_stats(g, prof)}
        out["sets"][s] = res
    return out, L0


def swap_diff2d(L0s: dict, pcfg, cfg16):
    """Pairs of detectors: per joint 2D difference (B - A) on frames where both track the same hand (target-box IoU >= min).
    Radial = component along the direction from A's hand centroid to A's joint (positive: B's point lies further out)."""
    names = [n for n in L0s if L0s[n] is not None and "template" in L0s[n]["a"]]
    bins = cfg16["diff2d"]["theta_bins_deg"]
    est_ids = list(pcfg["joints"]["sets"][pcfg["joints"]["estimate_set"]])
    out = {}
    for i, A in enumerate(names):
        for B in names[i + 1:]:
            ta, tb = L0s[A]["tr"], L0s[B]["tr"]
            th = np.array([np.nan if v is None else v for v in L0s[A]["a"]["theta2d_deg"]], float)
            rows = {b: {"d": [], "rel": [], "rad": [], "per_joint": {j: [] for j in range(21)}, "frames": 0} for b in range(len(bins) - 1)}
            n_pair = 0
            pa, pb_, pm = [], [], []
            for f in range(len(th)):
                if not (ta["present"][f] and tb["present"][f]) or not np.isfinite(th[f]):
                    continue
                if iou(ta["box"][f], tb["box"][f]) < cfg16["diff2d"]["same_hand_iou_min"]:
                    continue
                m = ta["mask"][f] & tb["mask"][f]
                if m.sum() < 4:
                    continue
                n_pair += 1
                pa.append(ta["kp"][f])
                pb_.append(tb["kp"][f])
                pm.append(m)
                b = int(np.clip(np.searchsorted(bins, th[f], side="right") - 1, 0, len(bins) - 2))
                rows[b]["frames"] += 1
                bl = float(box_long(ta["box"][f]))
                ctr = ta["kp"][f][[j for j in est_ids if ta["mask"][f][j]]].mean(0)
                for j in np.flatnonzero(m):
                    d = tb["kp"][f][j] - ta["kp"][f][j]
                    u = ta["kp"][f][j] - ctr
                    nu = np.linalg.norm(u)
                    rows[b]["d"].append(float(np.linalg.norm(d)))
                    rows[b]["rel"].append(float(np.linalg.norm(d) / bl))
                    rows[b]["rad"].append(float(d @ u / nu / bl) if nu > 1e-6 else np.nan)
                    rows[b]["per_joint"][j].append(float(np.linalg.norm(d) / bl))
            tab = []
            for b in range(len(bins) - 1):
                r = rows[b]
                tab.append({"theta": [bins[b], bins[b + 1]], "frames": r["frames"], "n_points": len(r["d"]),
                            "median_px": float(np.median(r["d"])) if r["d"] else None, "median_rel": float(np.median(r["rel"])) if r["rel"] else None,
                            "p90_rel": float(np.percentile(r["rel"], 90)) if r["rel"] else None,
                            "radial_median_rel": float(np.nanmedian(r["rad"])) if r["rad"] else None,
                            "per_joint_median_rel": {j: (float(np.median(v)) if v else None) for j, v in r["per_joint"].items()}})
            out[f"{A}|{B}"] = {"a": A, "b": B, "n_frames": n_pair, "by_theta": tab,
                               "order_assignment": order_assignment(np.array(pa), np.array(pb_), np.array(pm)) if pa else None}
    return out


def order_assignment(kpa, kpb, mask):
    """Median distance D[i, j] between point i of detector B and point j of detector A over shared frames; the point order of the two
    detectors agrees when the optimal one-to-one assignment (Hungarian) of D is the identity."""
    from scipy.optimize import linear_sum_assignment
    D = np.full((21, 21), np.nan)
    for i in range(21):
        for j in range(21):
            ok = mask[:, i] & mask[:, j]
            if ok.any():
                D[i, j] = float(np.median(np.linalg.norm(kpb[ok, i] - kpa[ok, j], axis=-1)))
    Dc = np.where(np.isfinite(D), D, 1e9)
    r, c = linear_sum_assignment(Dc)
    perm = c[np.argsort(r)].tolist()
    return {"identity": perm == list(range(21)), "assignment": perm, "row_argmin": [int(np.argmin(Dc[i])) for i in range(21)]}


# ====================================================================================================
# detect  (was HandCalib/detector_swap/detect.py)
# Run every detector on the same PNG list and return 14-format detection npz files (cached by frame hashes + settings).
# ====================================================================================================

class SkipDetector(RuntimeError):
    pass


def frames_key(frames, settings: dict, name: str) -> str:
    h = hashlib.sha256()
    for r in frames:
        h.update(f"{r['index']}:{r['sha256']};".encode())
    h.update(json.dumps({"detector": name, **settings}, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def write_list(frames, path: Path):
    path.write_text("".join(f"{r['index']}\t{r['png']}\t{r['sha256']}\n" for r in frames))


def _run_worker(cmd, log):
    env = dict(os.environ, PYTHONNOUSERSITE="1")
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise SkipDetector(f"worker failed (exit {p.returncode}): {p.stderr.strip()[-600:]}")


def swap_wilor_yolo(frames, s, cache_dir: Path, cfg16, log):
    dcfg = {"python": s["python"], "weights": s["weights"], "conf": s["conf"], "iou": s["iou"], "imgsz": s.get("imgsz"), "device": s["device"], "cache": True}
    key = cache_key(frames, dcfg)
    src14 = resolve(cfg16["reuse_14"]["detection_cache"]) / f"detections_{key}.npz"
    dst = cache_dir / f"detections_{key}.npz"
    reused = False
    if not dst.exists() and src14.exists():
        shutil.copy2(src14, dst)                     # 14's cache is only read; the copy lives in 16's cache
        reused = True
    d = detector_wilor_yolo(frames, dcfg, cache_dir, log)
    return Path(d["cache_file"]), {"reused_14_cache": reused, "cache_key": key}


def swap_mediapipe_hands(frames, s, cache_dir: Path, cfg16, log):
    settings = {k: s[k] for k in ("weights", "num_hands", "min_hand_detection_confidence", "min_hand_presence_confidence", "box_margin", "remap")}
    out = cache_dir / f"mediapipe_hands_{frames_key(frames, settings, 'mediapipe_hands')}.npz"
    if not out.exists():
        lst = cache_dir / (out.stem + ".tsv")
        write_list(frames, lst)
        cmd = [s["python"], str(WORKER), "mediapipe_hands", "--model", s["weights"], "--list", str(lst), "--out", str(out),
               "--num-hands", str(s["num_hands"]), "--min-detection", str(s["min_hand_detection_confidence"]),
               "--min-presence", str(s["min_hand_presence_confidence"]), "--box-margin", str(s["box_margin"])]
        if s.get("remap"):
            cmd += ["--remap", ",".join(map(str, s["remap"]))]
        _run_worker(cmd, log)
        lst.unlink()
    return out, {}


def swap_rtmpose_hand(frames, s, cache_dir: Path, cfg16, log, box_npz: Path):
    settings = {k: s[k] for k in ("weights", "input_size", "box_source", "box_scale", "remap", "device")}
    settings["boxes_sha256"] = hashlib.sha256(box_npz.read_bytes()).hexdigest()
    out = cache_dir / f"rtmpose_hand_{frames_key(frames, settings, 'rtmpose_hand')}.npz"
    if not out.exists():
        z = np.load(box_npz)
        bx = cache_dir / (out.stem + "_boxes.npz")
        np.savez(bx, offsets=z["offsets"], boxes=z["boxes"], box_conf=z["box_conf"], cls=z["cls"])      # boxes only, no keypoints
        lst = cache_dir / (out.stem + ".tsv")
        write_list(frames, lst)
        cmd = [s["python"], str(WORKER), "rtmpose_hand", "--onnx", s["weights"], "--list", str(lst), "--boxes", str(bx),
               "--out", str(out), "--input-size", ",".join(map(str, s["input_size"])), "--box-scale", str(s["box_scale"]), "--device", s["device"]]
        if s.get("remap"):
            cmd += ["--remap", ",".join(map(str, s["remap"]))]
        _run_worker(cmd, log)
        lst.unlink()
        bx.unlink()
    return out, {"box_source": s["box_source"]}


def swap_run_all(frames, cfg16, cache_dir: Path, log) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    order = list(cfg16["detectors"])
    if "rtmpose_hand" in order:                      # the box source must run first
        src = cfg16["detector_settings"]["rtmpose_hand"]["box_source"]
        src = {"mediapipe": "mediapipe_hands"}.get(src, src)
        order = [d for d in order if d == src] + [d for d in order if d != src]
        if src not in order:
            order.insert(0, src)
    for name in order:
        s = cfg16["detector_settings"][name]
        try:
            if name == "wilor_yolo":
                p, extra = swap_wilor_yolo(frames, s, cache_dir, cfg16, log)
            elif name == "mediapipe_hands":
                p, extra = swap_mediapipe_hands(frames, s, cache_dir, cfg16, log)
            elif name == "rtmpose_hand":
                src = {"mediapipe": "mediapipe_hands"}.get(s["box_source"], s["box_source"])
                if src not in out or "npz" not in out[src]:
                    raise SkipDetector(f"box source {src} unavailable")
                p, extra = swap_rtmpose_hand(frames, s, cache_dir, cfg16, log, out[src]["npz"])
            else:
                raise SkipDetector(f"unknown detector {name}")
            d = load_detections(p)
            check_alignment(frames, d)
            out[name] = {"npz": p, **extra, "info": d["info"]}
            log(f"  검출 {name}: 후보 {len(d['boxes'])}개 ({p.name})")
        except (SkipDetector, DetectorError, FileNotFoundError) as e:
            out[name] = {"skipped": str(e)}
            log(f"  검출 {name}: 건너뜀 — {e}")
    return {k: out[k] for k in order}


# ====================================================================================================
# conditions  (was HandCalib/protocol_rerun/conditions.py)
# Shooting-condition metrics (14's indicators) from one detector's run, and the pass / fail check against the yaml thresholds.
# ====================================================================================================

def edge_flags(box, present, W, H, margin=2.0):
    e = np.zeros(len(box), bool)
    for i in np.flatnonzero(present):
        b = box[i]
        e[i] = bool(b[0] <= margin or b[1] <= margin or b[2] >= W - margin or b[3] >= H - margin)
    return e


def metrics_from_arrays(times, present, box, W, H, theta, role, auto_transition_s, front_jitter_rel, axis_bins, finger_ext_min) -> dict:
    """All inputs are per-frame arrays of one tracked hand (box (n, 4); theta in degrees, NaN where not computed; role 'tilt' marks
    the tilt phase) plus 14's scalar indicators. Returns the condition metrics (None where not computable)."""
    present = np.asarray(present, bool)
    box = np.asarray(box, float)
    bl = box_long(box)
    short = float(min(W, H))
    edge = edge_flags(box, present, W, H)
    th = np.asarray(theta, float)
    tilt = (np.asarray(role) == "tilt") & np.isfinite(th)
    return {"box_short_ratio": float(np.nanmedian(bl[present]) / short) if present.any() else None,
            "edge_rate": float(edge[present].mean()) if present.any() else None,
            "transition_s": auto_transition_s, "front_jitter_rel": front_jitter_rel,
            "theta_p95_deg": float(np.percentile(th[tilt], 95)) if tilt.any() else None,
            "theta_tilt_median_deg": float(np.median(th[tilt])) if tilt.any() else None,
            "theta_max_deg": float(np.max(th[tilt])) if tilt.any() else None,
            "axis_bins": axis_bins, "finger_extension_min": finger_ext_min,
            "series": {"t": np.asarray(times, float).tolist(), "box_short_ratio": (bl / short).tolist(), "edge": edge.tolist(), "theta": th.tolist()}}


def metrics_from_run(det_dir: Path, tracking_cfg: dict, front_skip_s: float) -> dict:
    """Reads a 14-format run folder (results.json + detections.npz); re-tracks with 14's function (cheap, deterministic) for the
    per-frame boxes. theta2d / roles / axis come from the run's own analysis."""
    r = json.loads((det_dir / "results.json").read_text())
    a = r.get("analysis", {})
    if "theta2d_deg" not in a:
        return {"failed": a.get("failed") or "no analysis"}
    W, H = r["normalization"]["output"]["display_w"], r["normalization"]["output"]["display_h"]
    frames = r["frames"]["list"]
    times = np.array([x["time_s"] for x in frames])
    det = load_detections(det_dir / "detections.npz")
    tr = track(det, times, (W, H), tracking_cfg, front_start_s=front_skip_s)
    th = np.array([np.nan if v is None else v for v in a["theta2d_deg"]], float)
    ax = np.array([np.nan if v is None else v for v in a["axis2d_deg"]], float)
    m = metrics_from_arrays(times, tr["present"], tr["box"], W, H, th, a["role"], a.get("auto_transition_s"), a.get("front_jitter_rel"),
                            a["axis_coverage"]["bins"], a["finger_extension"]["min"])
    m["series"]["axis"] = ax.tolist()
    m["series"]["present"] = tr["present"].tolist()
    return m


def conditions_check(m: dict, c: dict) -> dict:
    """Pass / fail per condition; a metric that cannot be computed fails. guide = every condition passes."""
    if "failed" in m:
        return {"guide": False, "items": {}, "reasons": [f"분석 실패: {m['failed']}"]}
    lo, hi = c["theta_tilt_median_range_deg"]
    def ok(x, f):
        return x is not None and bool(f(x))
    items = {
        "box_short_ratio": ok(m["box_short_ratio"], lambda x: x <= c["box_short_ratio_max"]),
        "edge_rate": ok(m["edge_rate"], lambda x: x <= c["edge_rate_max"]),
        "transition": ok(m["transition_s"], lambda x: x >= c["transition_min_s"]),
        "theta_p95": ok(m["theta_p95_deg"], lambda x: x <= c["theta_p95_max_deg"]),
        "theta_tilt_median": ok(m["theta_tilt_median_deg"], lambda x: lo <= x <= hi),
        "axis_bins": ok(m["axis_bins"], lambda x: x >= c["axis_bins_min"]),
        "finger_extension": ok(m["finger_extension_min"], lambda x: x >= c["finger_extension_min"]),
    }
    msg = {"box_short_ratio": lambda: f"손 박스/짧은 변 {cond_f(m['box_short_ratio'], 2)} > {c['box_short_ratio_max']}",
           "edge_rate": lambda: f"가장자리 접촉 {cond_p(m['edge_rate'])} > {100 * c['edge_rate_max']:.0f}%",
           "transition": lambda: ("자동 전환 없음" if m["transition_s"] is None else f"자동 전환 {m['transition_s']:.2f} s < {c['transition_min_s']} s"),
           "theta_p95": lambda: f"θ2d p95 {cond_f(m['theta_p95_deg'], 0)}° > {c['theta_p95_max_deg']:.0f}°",
           "theta_tilt_median": lambda: f"기울기 구간 θ2d 중앙값 {cond_f(m['theta_tilt_median_deg'], 0)}°가 {lo:.0f}–{hi:.0f}° 밖",
           "axis_bins": lambda: f"축 덮임 {m['axis_bins']}/6 < {c['axis_bins_min']}/6",
           "finger_extension": lambda: f"손가락 펴짐 {cond_f(m['finger_extension_min'], 3)} < {c['finger_extension_min']}"}
    return {"guide": all(items.values()), "items": items, "reasons": [msg[k]() for k, v in items.items() if not v]}


def conditions_text(m: dict) -> str:
    if "failed" in m:
        return "분석 실패"
    tr = "없음" if m["transition_s"] is None else f"{m['transition_s']:.1f}초"
    return (f"박스 {cond_f(m['box_short_ratio'], 2)}, 가장자리 {cond_p(m['edge_rate'])}, 전환 {tr}, θ2d p95 {cond_f(m['theta_p95_deg'], 0)}°, "
            f"기울기 중앙값 {cond_f(m['theta_tilt_median_deg'], 0)}°, 최대 {cond_f(m['theta_max_deg'], 0)}°, 덮임 {m['axis_bins']}/6, "
            f"펴짐 {cond_f(m['finger_extension_min'], 3)}, 정면 지터 {cond_p(m['front_jitter_rel'], 2)}")


def cond_f(x, d):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def cond_p(x, d=0):
    return "nan" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}%"


# ====================================================================================================
# ranking  (was HandCalib/protocol_rerun/ranking.py)
# Internal detector quality ranking (not accuracy): category, then the yaml keys in order; exact ties share the mean position.
# ====================================================================================================

def rank_category(d: dict) -> str:
    if "fixed_template" not in d:
        return "FAILED"
    if d["fixed_template"]["at_grid_edge"]:
        return "EDGE"
    return d["verdict"]


def rank_key_values(d: dict) -> dict:
    """The ranking inputs of one detector result (16's summary dict). Missing / not computable -> None."""
    if "fixed_template" not in d:
        return {"sharpness": None, "f_rel_uncertainty": None, "reproj_rms_rel": None, "detect_rate": None}
    sh = d.get("profiles", {}).get("flat17", {}).get("official", {}).get("sharpness")
    unc = None if d["fixed_template"]["at_grid_edge"] else d.get("f_rel_uncertainty")
    return {"sharpness": sh, "f_rel_uncertainty": unc, "reproj_rms_rel": d.get("reproj_rms_rel"), "detect_rate": d.get("detect_rate")}


def _rank_sort_key(cat, vals, rcfg):
    k = [rcfg["category_order"].index(cat)]
    for spec in rcfg["keys"]:
        v = vals.get(spec["name"])
        if v is None or not math.isfinite(v):
            k.append(math.inf)                                   # worst
        else:
            k.append(-v if spec["better"] == "higher" else v)
    return tuple(k)


def rank(dets: dict, rcfg: dict) -> dict:
    """dets: {detector: summary}. Returns {detector: {"rank", "category", "keys"}} with 1 = best; exact ties get the mean position."""
    items = []
    for name, d in dets.items():
        cat = rank_category(d) if "skipped" not in d else "FAILED"
        vals = rank_key_values(d) if "skipped" not in d else rank_key_values({})
        items.append((name, cat, vals, _rank_sort_key(cat, vals, rcfg)))
    items.sort(key=lambda x: x[3])
    out = {}
    i = 0
    while i < len(items):
        j = i
        while j + 1 < len(items) and items[j + 1][3] == items[i][3]:
            j += 1
        r = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            out[items[k][0]] = {"rank": r, "category": items[k][1], "keys": items[k][2], "verdict": dets[items[k][0]].get("verdict")}
        i = j + 1
    return out


def rank_summary(per_video: dict, guide: dict) -> dict:
    """per_video: {video: {detector: rank dict}}; guide: {video: bool}. Mean rank over guide-satisfying videos and category counts."""
    names = sorted({d for v in per_video.values() for d in v})
    out = {}
    for n in names:
        rs = [per_video[v][n]["rank"] for v in per_video if guide.get(v) and n in per_video[v]]
        cats = [per_video[v][n]["category"] for v in per_video if n in per_video[v]]
        verd = [per_video[v][n]["verdict"] for v in per_video if n in per_video[v]]
        out[n] = {"mean_rank_guide": float(np.mean(rs)) if rs else None, "n_guide_videos": len(rs),
                  "verdict_counts": {c: verd.count(c) for c in ("GOOD", "ROUGH", "REJECT")}, "n_edge": cats.count("EDGE"),
                  "n_failed": cats.count("FAILED"), "category_counts": {c: cats.count(c) for c in ("GOOD", "ROUGH", "REJECT", "EDGE", "FAILED")}}
    return out


# ====================================================================================================
# frame_filter  (was HandCalib/protocol_rerun/frame_filter.py)
# Diagnostic frame filter for the 17 pipeline: drop tilt-phase frames with theta2d above a threshold or a box touching the border,
# then re-run 14's analysis core unchanged (front window / template unaffected because only frames after the tilt start change).
# ====================================================================================================

def ff_keep_mask(times, present, theta, box, W, H, tilt_start, fcfg) -> np.ndarray:
    """True for frames the analysis may use. Frames before tilt_start are untouched; tilt-phase frames must pass the filter."""
    t = np.asarray(times, float)
    th = np.asarray(theta, float)
    keep = np.asarray(present, bool).copy()
    tilt = t >= tilt_start
    bad = tilt & ~(np.isfinite(th) & (th <= fcfg["theta_max_deg"]))
    if fcfg.get("exclude_edge"):
        bad |= tilt & edge_flags(box, present, W, H, fcfg.get("edge_margin_px", 2.0))
    return keep & ~bad


def ff_run_one(run_dir: Path, pcfg: dict, fcfg: dict, methods_full=("fixed_template",)) -> dict:
    L0 = load_official(run_dir, pcfg)                          # 14 tracking + analysis (theta2d from the official template)
    a0, tr, W, H, times = L0["a"], L0["tr"], L0["W"], L0["H"], L0["times"]
    if "template" not in a0:
        return {"failed": a0.get("failed")}
    th = np.array([np.nan if v is None else v for v in a0["theta2d_deg"]], float)
    keep = ff_keep_mask(times, tr["present"], th, tr["box"], W, H, a0["tilt_start_s"], fcfg)
    a = analyze(tr["kp"], tr["mask"], keep, tr["box"], tr["cls"], times, (W, H), pcfg, methods_full=methods_full,
                  seed=int(pcfg["experiment"]["seed"]))
    if "per_method" not in a:
        return {"failed": a.get("failed"), "n_kept_tilt": int((keep & (times >= a0["tilt_start_s"])).sum())}
    same_template = np.allclose(np.array(a["template"]), np.array(a0["template"]))
    out = {"same_template": bool(same_template), "tilt_start_s": a0["tilt_start_s"],
           "n_tilt_before": int((tr["present"] & (times >= a0["tilt_start_s"])).sum()),
           "n_tilt_kept": int((keep & (times >= a0["tilt_start_s"])).sum()), "solutions": {}, "per_method": {}}
    for m in ("fixed_template", "joint_delta"):
        s = a["solutions"][m]
        out["solutions"][m] = {"f": s["f"], "at_grid_edge": s["at_grid_edge"], "delta_deg": s.get("delta_deg")}
    for m, pm in a["per_method"].items():
        met = dict(pm["metrics"])
        if fcfg.get("verdict_detect_rate") == "original":
            met["detect_rate"] = float(tr["present"].mean())
        j = judge(met, pcfg["verdict"])
        out["per_method"][m] = {"f": pm["f"], "verdict": j["verdict"], "failed_good": j["failed_good"], "failed_rough": j["failed_rough"],
                                "f_rel_uncertainty": pm["f_rel_uncertainty"], "bootstrap_rel_std": pm["bootstrap"]["rel_std"],
                                "consistency_range": pm["consistency"]["range"], "sharpness": pm["diagnostics"]["sharpness"],
                                "reproj_rms_rel": pm["diagnostics"]["reproj_rms_rel"], "jointsets": pm["jointsets"], "metrics": met}
    # profiles (official + extended grid) of the kept frames, flat17 and palm sets
    L = float(max(W, H))
    off = f_grid(pcfg["estimate"], L)
    ext = np.geomspace(0.2 * L, 5.0 * L, 90)
    est_ids = list(pcfg["joints"]["sets"][pcfg["joints"]["estimate_set"]])
    T = np.array(a["template"])
    used = [i for i, r in enumerate(a["role"]) if r in ("front", "tilt")]
    bl = box_long(tr["box"])
    huber = pcfg["estimate"]["huber_rel"] * np.where(np.isfinite(bl), bl, 0.0)
    cx, cy = a["principal_point"]
    prof = {}
    for sname in ("palm4", "palm5", "palm_proximal9", "flat17"):
        ids = list(pcfg["joints"]["sets"][sname])
        pos = [est_ids.index(j) for j in ids]
        r = {}
        for gname, g in (("official", off), ("extended", ext)):
            C, _ = frame_cost_matrix(T, tr["kp"], tr["mask"], used, pos, ids, g, cx, cy, huber)
            ok = np.isfinite(C).all(1)
            p = C[ok].mean(0) if ok.sum() else np.full(len(g), np.nan)
            k = int(np.nanargmin(p)) if np.isfinite(p).any() else 0
            r[gname] = {"grid": g.tolist(), "profile": p.tolist(), "min_at_px": float(g[k]),
                        "at": "interior" if 0 < k < len(g) - 1 else ("lower" if k == 0 else "upper"), "n_frames": int(ok.sum())}
        prof[sname] = r
    out["profiles"] = prof
    out["grid_px"] = [float(off[0]), float(off[-1])]
    return out


# ====================================================================================================
# segment  (was HandCalib/same_session/segment.py)
# Automatic hand / checkerboard segmentation of one recording.
# ====================================================================================================

def board_found(gray, inner, long_side):
    """findChessboardCornersSB on a downscaled grey frame (either orientation of the inner-corner grid)."""
    h, w = gray.shape
    s = min(1.0, long_side / max(h, w))
    g = cv2.resize(gray, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else gray
    for ps in (tuple(inner), tuple(inner[::-1])):
        ok, _ = cv2.findChessboardCornersSB(g, ps, cv2.CALIB_CB_EXHAUSTIVE)
        if ok:
            return True
    return False


def segment_split(times, hand, board, gap_s):
    """times (n,), hand / board (n,) bools at the scan rate. Hand segment: [0, first board - gap); board segment: [first board, last board].
    Warnings when a board shows up inside the hand segment or a hand inside the board segment."""
    t = np.asarray(times, float)
    hand, board = np.asarray(hand, bool), np.asarray(board, bool)
    if not board.any():
        return {"hand": [float(t[0]), float(t[-1])], "board": None, "warnings": ["체커보드가 검출되지 않았습니다"]}
    i0, i1 = int(np.flatnonzero(board)[0]), int(np.flatnonzero(board)[-1])
    hand_end = float(t[i0] - gap_s)
    seg = {"hand": [float(t[0]), hand_end], "board": [float(t[i0]), float(t[i1])], "warnings": []}
    if hand_end <= t[0]:
        seg["warnings"].append("손 구간이 없습니다(첫 체커보드 검출이 너무 이릅니다)")
    in_hand = (t < hand_end)
    if (board & in_hand).any():
        seg["warnings"].append(f"손 구간에 체커보드 검출 {int((board & in_hand).sum())}장")
    in_board = (t >= t[i0]) & (t <= t[i1])
    if (hand & in_board).any():
        seg["warnings"].append(f"체커보드 구간에 손 검출 {int((hand & in_board).sum())}장 (보드를 든 손일 수 있음)")
    if not hand[in_hand].any():
        seg["warnings"].append("손 구간에서 손이 검출되지 않았습니다")
    seg["hand_detect_rate_in_hand"] = float(hand[in_hand].mean()) if in_hand.any() else None
    seg["board_detect_rate_in_board"] = float(board[in_board].mean())
    return seg


# ====================================================================================================
# cut  (was HandCalib/same_session/cut.py)
# Exact frame cut of the normalised (lossless) video: frames with t0 <= t < t1, re-encoded losslessly, timestamps reset.
# ====================================================================================================

def cut_segment(src: Path, dst: Path, t0: float, t1: float, fps: float) -> dict:
    f0 = int(round(t0 * fps))
    f1 = int(round(t1 * fps))
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-vf",
                    f"select=between(n\\,{f0}\\,{f1 - 1}),setpts=N/FRAME_RATE/TB", "-vsync", "0", "-an", "-map_metadata", "-1",
                    "-c:v", "ffv1", str(dst)], check=True)
    return {"first_frame": f0, "end_frame": f1, "n_frames": f1 - f0, "t0": f0 / fps, "t1": f1 / fps}


# ====================================================================================================
# checkerboard  (was HandCalib/same_session/checkerboard.py)
# Checkerboard reference f of one recording (rules of the experiment specification; video_calib's code was not available).
#
# Sampling: frames of the board segment at sample_fps -> corner detection (findChessboardCornersSB on a downscaled grey frame, then
# cornerSubPix at full resolution) -> up to max_frames frames chosen by farthest-point sampling of a pose descriptor.
# Models (cv2.calibrateCamera):
#   A  fx, fy, cx, cy free; k1 k2 p1 p2 k3
#   B  fx, fy, cx, cy free; k1 k2
#   C  fx = fy, cx, cy free; k1 k2
#   D  fx = fy, principal point fixed at the image centre (W / 2, H / 2); k1 k2
#   E  fx = fy, centre principal point, no distortion (the hand method's and AnyCalib's assumptions)
# f_reference = median fx of the reference models (A-D).
# ====================================================================================================

BOARD_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)


def board_detect(gray, inner, long_side, win):
    """Returns (corners (N, 2) full-resolution pixels, pattern size) or (None, None)."""
    h, w = gray.shape
    s = min(1.0, long_side / max(h, w))
    g = cv2.resize(gray, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else gray
    for ps in (tuple(inner), tuple(inner[::-1])):
        ok, c = cv2.findChessboardCornersSB(g, ps, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if ok:
            c = (c.reshape(-1, 2) / s).astype(np.float32).reshape(-1, 1, 2)
            c = cv2.cornerSubPix(gray, c, (win, win), (-1, -1), BOARD_CRIT)
            return c.reshape(-1, 2).astype(np.float64), ps
    return None, None


def board_object_points(ps, square):
    nx, ny = ps
    g = np.zeros((nx * ny, 3), np.float64)
    g[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) * square
    return g


def _board_init_K(W, H):
    f = float(max(W, H))
    return np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])


def board_pose_descriptor(corners, ps, square, W, H):
    """Rough pose (K = long side, centre pp, no distortion): board normal (3), board centre in normalised image coordinates (2),
    log apparent size (1)."""
    obj = board_object_points(ps, square)
    ok, rv, tv = cv2.solvePnP(obj, corners, _board_init_K(W, H), None, flags=cv2.SOLVEPNP_IPPE)
    R, _ = cv2.Rodrigues(rv)
    n = R[:, 2] * np.sign(R[2, 2] or 1.0)
    c = corners.mean(0)
    size = np.sqrt(cv2.contourArea(cv2.convexHull(corners.astype(np.float32))))
    return np.r_[n, (c[0] - W / 2) / max(W, H) * 2, (c[1] - H / 2) / max(W, H) * 2, np.log(size / max(W, H))]


def farthest_point(desc, k, start=0):
    idx = [start]
    d = np.linalg.norm(desc - desc[start], axis=1)
    while len(idx) < min(k, len(desc)):
        j = int(np.argmax(d))
        idx.append(j)
        d = np.minimum(d, np.linalg.norm(desc - desc[j], axis=1))
    return sorted(idx)


BOARD_FLAGS = {
    "A": 0,
    "B": cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3,
    "C": cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3,
    "D": cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3,
    "E": cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3,
}


def board_calibrate(views, W, H, model):
    """views: list of (obj (N, 3), img (N, 2)). Returns {fx, fy, cx, cy, dist, rms, rvecs, tvecs}."""
    obj = [v[0].astype(np.float32) for v in views]
    img = [v[1].astype(np.float32).reshape(-1, 1, 2) for v in views]
    K0 = _board_init_K(W, H)
    flags = BOARD_FLAGS[model] | cv2.CALIB_USE_INTRINSIC_GUESS
    rms, K, dist, rv, tv = cv2.calibrateCamera(obj, img, (W, H), K0.copy(), np.zeros(5), flags=flags,
                                               criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-10))
    return {"fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]), "dist": np.ravel(dist)[:5].tolist(),
            "rms": float(rms), "rvecs": rv, "tvecs": tv}


def board_f_reference(views, W, H, ref_models):
    fx = [board_calibrate(views, W, H, m)["fx"] for m in ref_models]
    return float(np.median(fx))


def board_analyse(frames, ccfg: dict, W: int, H: int, log=print) -> dict:
    """frames: list of {index, time_s, gray (H, W) uint8}. Returns the full checkerboard result."""
    inner, sq = ccfg["inner_corners"], float(ccfg["square_mm"])
    det = []
    for fr in frames:
        c, ps = board_detect(fr["gray"], inner, ccfg["detect_long_side"], int(ccfg["subpix_window"]))
        if c is not None:
            det.append({"index": fr["index"], "time_s": fr["time_s"], "corners": c, "ps": ps})
    out = {"n_sampled": len(frames), "n_detected": len(det)}
    if len(det) < 5:
        out.update({"verdict": "REJECT", "failed": "검출된 체커보드 프레임이 5장 미만"})
        return out
    desc = np.array([board_pose_descriptor(d["corners"], d["ps"], sq, W, H) for d in det])
    start = int(np.argmin(np.arccos(np.clip(np.abs(desc[:, 2]), 0, 1))))      # most frontal frame first
    sel = farthest_point(desc, int(ccfg["max_frames"]), start)
    views = [(board_object_points(det[i]["ps"], sq), det[i]["corners"]) for i in sel]
    times = np.array([det[i]["time_s"] for i in sel])
    models = {}
    for m in ccfg["models"]:
        r = board_calibrate(views, W, H, m)
        models[m] = {k: r[k] for k in ("fx", "fy", "cx", "cy", "dist", "rms")}
        if m == "D":
            rD = r
    ref = [models[m]["fx"] for m in ccfg["reference_models"]]
    f_ref = float(np.median(ref))
    model_range = float((max(ref) - min(ref)) / f_ref)
    # block bootstrap over the time-ordered selected frames
    rng = np.random.default_rng(0)
    order = np.argsort(times)
    B = int(ccfg["bootstrap_block_frames"])
    blocks = [order[i:i + B] for i in range(0, len(order), B)]
    bs = []
    for _ in range(int(ccfg["n_bootstrap"])):
        pick = [blocks[j] for j in rng.integers(0, len(blocks), len(blocks))]
        vv = [views[i] for blk in pick for i in blk]
        bs.append(board_f_reference(vv, W, H, ccfg["reference_models"]))
    bs = np.array(bs)
    # time segments
    segs = np.array_split(order, int(ccfg["time_segments"]))
    seg_f = [board_f_reference([views[i] for i in s], W, H, ccfg["reference_models"]) if len(s) >= 4 else float("nan") for s in segs]
    segf = [x for x in seg_f if np.isfinite(x)]
    seg_range = float((max(segf) - min(segf)) / f_ref) if len(segf) >= 2 else float("nan")
    # pose / coverage metrics from model D's extrinsics
    tilts = []
    normals = []
    for rv in rD["rvecs"]:
        R, _ = cv2.Rodrigues(rv)
        n = R[:, 2]
        normals.append(n / np.linalg.norm(n))
        tilts.append(float(np.degrees(np.arccos(min(1.0, abs(n[2]) / np.linalg.norm(n))))))
    tilts = np.array(tilts)
    distinct = []
    for n in normals:
        if all(np.degrees(np.arccos(np.clip(abs(n @ m), 0, 1))) >= ccfg["distinct_pose_min_deg"] for m in distinct):
            distinct.append(n)
    gx, gy = ccfg["coverage_grid"]
    cells = np.zeros((gy, gx), int)
    for v in views:
        p = v[1]
        ix = np.clip((p[:, 0] / W * gx).astype(int), 0, gx - 1)
        iy = np.clip((p[:, 1] / H * gy).astype(int), 0, gy - 1)
        np.add.at(cells, (iy, ix), 1)
    coverage = float((cells > 0).mean())
    rms = float(np.median([models[m]["rms"] for m in ccfg["reference_models"]]))
    met = {"tilt_p90_deg": float(np.percentile(tilts, 90)), "tilt_max_deg": float(tilts.max()), "distinct_poses": len(distinct),
           "coverage": coverage, "model_range": model_range, "bootstrap_rel_std": float(np.std(bs, ddof=1) / np.mean(bs)),
           "segment_range": seg_range, "rms_px": rms}
    v = board_verdict(met, ccfg["verdict"])
    out.update({"n_selected": len(sel), "selected_times": times.tolist(), "selected_indices": [int(det[i]["index"]) for i in sel],
                "pattern_size": list(det[0]["ps"]), "models": models, "f_reference_px": f_ref, "f_model_E_px": models.get("E", {}).get("fx"),
                "bootstrap_f": bs.tolist(), "segment_f": seg_f, "metrics": met, "tilts_deg": tilts.tolist(), "coverage_cells": cells.tolist(), **v,
                "image_size": [W, H]})
    return out


def board_verdict(m, vc):
    def ok(t):
        return {"tilt_p90": m["tilt_p90_deg"] >= t["tilt_p90_min_deg"], "coverage": m["coverage"] >= t["coverage_min"],
                "poses": m["distinct_poses"] >= t["poses_min"], "model_range": m["model_range"] <= t["model_range_max"],
                "bootstrap": m["bootstrap_rel_std"] <= t["bootstrap_rel_std_max"],
                "segment_range": bool(np.isfinite(m["segment_range"]) and m["segment_range"] <= t["segment_range_max"]),
                "rms": m["rms_px"] <= t["rms_max_px"]}
    g, r = ok(vc["good"]), ok(vc["rough"])
    v = "GOOD" if all(g.values()) else ("ROUGH" if all(r.values()) else "REJECT")
    return {"verdict": v, "failed_good": [k for k, x in g.items() if not x], "failed_rough": [k for k, x in r.items() if not x]}


# ====================================================================================================
# pseg  (experiment 20: front -> tilt transition detection; new section, 17's functions above are used unchanged)
# The legacy rule (analyze's auto_transition: theta2d >= auto_theta_deg for auto_sustain_s) fires on keypoint noise because theta2d = arccos(s2 / s1)
# is hypersensitive near 0 deg. linear_d uses d = 1 - s2 / s1 (= 1 - cos(theta2d)) against a per-video noise floor taken from the front window,
# a median-smoothed copy of d (segmentation only) and a keypoint-motion condition. Optional, default-off: exclude large-tilt frames, exclude the wrist.
# ====================================================================================================

def affine_ratio(T: np.ndarray, P: np.ndarray) -> float:
    """s2 / s1 of the linear part of the least-squares affine map template T -> observation P (rotation and uniform scale cancel in the ratio)."""
    A = np.c_[T, np.ones(len(T))]
    M, *_ = np.linalg.lstsq(A, P, rcond=None)
    S = np.linalg.svd(M[:2].T, compute_uv=False)
    return float(np.clip(S[1] / S[0], 0.0, 1.0)) if S[0] > 0 else 0.0


def pseg_series(kp, mask, present, box, times, T, est_ids):
    """Per-frame d = 1 - s2 / s1 and the keypoint speed v (1 / s): mean joint displacement between consecutive frames / hand-box long side / dt."""
    t = np.asarray(times, float)
    n = len(t)
    d = np.full(n, np.nan)
    v = np.full(n, np.nan)
    bl = box_long(np.asarray(box, float))
    for i in range(n):
        if present[i]:
            vis = mask[i][est_ids]
            if vis.sum() >= 6:
                d[i] = 1.0 - affine_ratio(T[vis], kp[i][est_ids][vis])
        if i > 0 and present[i] and present[i - 1] and t[i] > t[i - 1]:
            m = mask[i][est_ids] & mask[i - 1][est_ids]
            if m.sum() >= 4:
                disp = float(np.linalg.norm(kp[i][est_ids][m] - kp[i - 1][est_ids][m], axis=-1).mean())
                v[i] = disp / max(float(np.nanmean([bl[i], bl[i - 1]])), 1e-9) / float(t[i] - t[i - 1])
    return d, v


def pseg_median_filter(times, x, window_s: float) -> np.ndarray:
    """Centred running median over |t - t_i| <= window_s / 2 (NaN ignored). Used for the segmentation only."""
    t = np.asarray(times, float)
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        w = x[np.abs(t - t[i]) <= window_s / 2 + 1e-9]
        w = w[np.isfinite(w)]
        if len(w):
            out[i] = float(np.median(w))
    return out


def pseg_transition(times, present, kp, mask, box, T, est_ids, front_idx, p: dict) -> dict:
    """First time after the front window at which the smoothed d stays above d_thr for sustain_s (and, with use_motion, the keypoint speed exceeds
    v_thr within motion_window_s of that start). Returns t_transition (None if not found), d0, d_thr, v0, v_thr and the series."""
    t = np.asarray(times, float)
    d, v = pseg_series(kp, mask, present, box, t, T, est_ids)
    fw = [int(i) for i in front_idx]
    dw, vw = d[fw], v[fw]
    if not np.isfinite(dw).any():
        return {"t_transition": None, "failed": "정면 창에서 d를 계산할 수 없음"}
    d0 = float(np.nanpercentile(dw, p["noise_percentile"]))
    v0 = float(np.nanpercentile(vw, p["noise_percentile"])) if np.isfinite(vw).any() else float("nan")
    d_thr = max(p["k_d"] * d0, 1.0 - math.cos(math.radians(p["theta_min_deg"])))
    v_thr = p["k_v"] * v0
    ds = pseg_median_filter(t, d, p["median_window_s"])
    ok = np.isfinite(ds) & (ds > d_thr)
    t_end = float(t[fw[-1]])
    cand = np.flatnonzero(t > t_end)
    skipped, found, motion_v = [], None, None
    pos = 0
    while pos < len(cand):
        i = cand[pos]
        if ok[i]:
            j = np.flatnonzero((t >= t[i]) & (t < t[i] + p["sustain_s"]))
            if len(j) and ok[j].all():
                mot = True
                if p["use_motion"]:
                    w = np.flatnonzero(np.abs(t - t[i]) <= p["motion_window_s"] + 1e-9)
                    vv = v[w][np.isfinite(v[w])]
                    motion_v = float(vv.max()) if len(vv) else None
                    mot = bool(motion_v is not None and np.isfinite(v_thr) and motion_v > v_thr)
                if mot:
                    found = float(t[i])
                    break
                skipped.append(float(t[i]))
                while pos < len(cand) and ok[cand[pos]]:
                    pos += 1
                continue
        pos += 1
    return {"t_transition": found, "d0": d0, "d_thr": float(d_thr), "v0": v0, "v_thr": float(v_thr), "front_end_s": t_end,
            "motion_v_at_start": motion_v, "skipped_no_motion_s": skipped, "series": {"t": t.tolist(), "d": d.tolist(), "d_smooth": ds.tolist(), "v": v.tolist()}}


def pseg_wrist_cfg(pcfg: dict) -> dict:
    """The pipeline settings with joint 0 (wrist) removed from every joint set (template and f estimation)."""
    c = copy.deepcopy(pcfg)
    for k, ids in c["joints"]["sets"].items():
        c["joints"]["sets"][k] = [j for j in ids if j != 0]
    return c


def pseg_tilt_keep(times, present, theta, tilt_start, max_tilt_deg) -> np.ndarray:
    """True for frames the f estimate may use: tilt-phase frames whose theta2d exceeds max_tilt_deg are dropped (frames without theta2d are left to the analysis)."""
    t = np.asarray(times, float)
    th = np.asarray(theta, float)
    bad = (t >= tilt_start) & np.isfinite(th) & (th > max_tilt_deg)
    return np.asarray(present, bool) & ~bad


def pseg_f_summary(a: dict, present_orig, pcfg: dict, extra: dict | None = None) -> dict:
    """Compact result of one analysis: f of both estimators (grid edge -> side), verdict (detection rate = the original tracker rate), sharpness."""
    if "per_method" not in a:
        return {"failed": a.get("failed", "분석 실패"), **(extra or {})}
    pm = a["per_method"]["fixed_template"]
    hi = a["grid"][-1]
    sol = {m: {"f": a["solutions"][m]["f"], "at_grid_edge": a["solutions"][m]["at_grid_edge"],
               "side": None if not a["solutions"][m]["at_grid_edge"] else ("상한" if a["solutions"][m]["f"] >= hi * (1 - 1e-6) else "하한")} for m in ("fixed_template", "joint_delta")}
    met = dict(pm["metrics"])
    met["detect_rate"] = float(np.asarray(present_orig, bool).mean())
    j = judge(met, pcfg["verdict"])
    return {"solutions": sol, "verdict": j["verdict"], "failed_good": j["failed_good"], "failed_rough": j["failed_rough"], "tilt_start_s": a["tilt_start_s"],
            "n_front_used": a["n_front_used"], "n_tilt_used": a["n_tilt_used"], "f_rel_uncertainty": pm["f_rel_uncertainty"], "bootstrap_rel_std": pm["bootstrap"]["rel_std"],
            "sharpness": pm["diagnostics"]["sharpness"], "reproj_rms_rel": pm["diagnostics"]["reproj_rms_rel"], "theta_tilt": a["theta_tilt"], **(extra or {})}


def pseg_analyze(kp, mask, present, box, cls, times, size_wh, cfg: dict, c20: dict, ccfg: dict, seed: int | None = None, method: str = "linear_d") -> dict:
    """One tracked hand: 17's analysis (unchanged call), both transition detectors, the shooting-guide check with each, and the optional variants."""
    W, H = size_wh
    seed = int(cfg["experiment"]["seed"]) if seed is None else seed
    p = c20
    est_ids = list(cfg["joints"]["sets"][cfg["joints"]["estimate_set"]])
    cache = {}

    def run(cfg_use, tilt_start, pres, key):
        if key not in cache:
            cache[key] = analyze(kp, mask, pres, box, cls, times, size_wh, cfg_use, tilt_start=tilt_start, methods_full=("fixed_template",), seed=seed)
        return cache[key]
    a0 = run(cfg, None, present, (False, None, False))
    if "per_method" not in a0:
        return {"failed": a0.get("failed", "분석 실패")}
    theta = np.array([np.nan if x is None else x for x in a0["theta2d_deg"]], float)
    t_legacy = a0.get("auto_transition_s")
    lin = pseg_transition(times, present, kp, mask, box, np.array(a0["template"]), est_ids, a0["front_window"]["idx"], p)
    t_lin = lin["t_transition"]
    guides = {}
    for name, tv in (("legacy_theta", t_legacy), ("linear_d", t_lin)):
        m = metrics_from_arrays(times, present, box, W, H, theta, a0["role"], tv, a0.get("front_jitter_rel"), a0["axis_coverage"]["bins"], a0["finger_extension"]["min"])
        g = conditions_check(m, ccfg)
        guides[name] = {"transition_s": tv, "guide": g["guide"], "reasons": g["reasons"], "items": g["items"]}
    lin_nm = lin                                              # diagnostic column: the same detector with the motion condition off
    if p["use_motion"]:
        lin_nm = pseg_transition(times, present, kp, mask, box, np.array(a0["template"]), est_ids, a0["front_window"]["idx"], {**p, "use_motion": False})
    m_nm = metrics_from_arrays(times, present, box, W, H, theta, a0["role"], lin_nm["t_transition"], a0.get("front_jitter_rel"), a0["axis_coverage"]["bins"],
                               a0["finger_extension"]["min"])
    g_nm = conditions_check(m_nm, ccfg)
    t_sel = t_lin if method == "linear_d" else t_legacy
    out = {"default": pseg_f_summary(a0, present, cfg, {"front_window": [a0["front_window"].get("t0"), a0["front_window"].get("t1")],
                                                       "auto_transition_s": t_legacy}),
           "legacy_theta": {"t_transition": t_legacy, **guides["legacy_theta"]},
           "linear_d": {k: lin.get(k) for k in ("t_transition", "d0", "d_thr", "v0", "v_thr", "front_end_s", "motion_v_at_start", "skipped_no_motion_s", "failed")} | guides["linear_d"],
           "linear_d_no_motion": {"t_transition": lin_nm["t_transition"], "guide": g_nm["guide"], "reasons": g_nm["reasons"]},
           "method": method, "t_selected": t_sel, "guide_selected": guides[method]["guide"], "variants": {},
           "series": {**lin.get("series", {}), "theta": theta.tolist(), "front": [a0["front_window"].get("t0"), a0["front_window"].get("t1")]}}
    for name, ov in (p.get("variants") or {}).items():
        o = {"phase_mode": "nominal", "exclude_large_tilt": bool(p["exclude_large_tilt"]), "exclude_wrist": bool(p["exclude_wrist"]), **ov}
        cfg_use = pseg_wrist_cfg(cfg) if o["exclude_wrist"] else cfg
        ts = None
        if o["phase_mode"] == "auto":
            if t_sel is None:
                out["variants"][name] = {"skipped": "전환 시각을 검출하지 못함", "options": o}
                continue
            ts = t_sel
        base = run(cfg_use, ts, present, (o["exclude_wrist"], ts, False))
        res, n_before, n_after = base, None, None
        if o["exclude_large_tilt"] and "per_method" in base:
            th = np.array([np.nan if x is None else x for x in base["theta2d_deg"]], float)
            keep = pseg_tilt_keep(times, present, th, base["tilt_start_s"], p["max_tilt_deg"])
            n_before = int((np.asarray(present, bool) & (np.asarray(times) >= base["tilt_start_s"])).sum())
            n_after = int((keep & (np.asarray(times) >= base["tilt_start_s"])).sum())
            res = run(cfg_use, ts, keep, (o["exclude_wrist"], ts, True))
        out["variants"][name] = pseg_f_summary(res, present, cfg_use, {"options": o, "n_tilt_before": n_before, "n_tilt_after": n_after})
    return out


def pseg_job(args) -> dict:
    """(video, detector) job for the process pool: load the 17 run's frame list and detections, analyse, compare with 17's own result."""
    video, det, run_dir, pcfg, c20, ccfg, conf_min, method = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cv2.setNumThreads(1)
    pc = copy.deepcopy(pcfg)
    pc["tracking"]["conf_min"] = float(conf_min)
    d = Path(run_dir) / det
    res = json.loads((d / "results.json").read_text())
    dets = load_detections(d / "detections.npz")
    frames = res["frames"]["list"]
    check_alignment(frames, dets)
    W, H = res["normalization"]["output"]["display_w"], res["normalization"]["output"]["display_h"]
    times = np.array([r["time_s"] for r in frames])
    tr = track(dets, times, (W, H), pc["tracking"], front_start_s=float(pc["phases"]["front_skip_s"]))
    out = pseg_analyze(tr["kp"], tr["mask"], tr["present"], tr["box"], tr["cls"], times, (W, H), pc, c20, ccfg, method=method)
    out.update({"video": video, "detector": det, "size": [W, H], "n_frames": len(frames)})
    if "default" in out and "solutions" in out["default"]:
        s = out["default"]["solutions"]
        a17 = res.get("analysis", {})
        same = {"f_fixed_template": abs(s["fixed_template"]["f"] - res.get("f_hand_px", np.nan)) <= 1e-6 * max(1.0, abs(s["fixed_template"]["f"])),
                "f_joint_delta": abs(s["joint_delta"]["f"] - res.get("f_joint_delta_px", np.nan)) <= 1e-6 * max(1.0, abs(s["joint_delta"]["f"])),
                "verdict": out["default"]["verdict"] == res.get("verdict"),
                "legacy_transition": (out["legacy_theta"]["t_transition"] is None and a17.get("auto_transition_s") is None) or
                                     (out["legacy_theta"]["t_transition"] is not None and a17.get("auto_transition_s") is not None and
                                      abs(out["legacy_theta"]["t_transition"] - a17["auto_transition_s"]) < 1e-9)}
        out["same_as_17"] = {**same, "all": bool(all(same.values())), "f_17": res.get("f_hand_px"), "verdict_17": res.get("verdict")}
    return out


# ====================================================================================================
# grev  (experiment 21: transition defaults, shooting-guide revision, wrist handling; new section, 17's and 20's functions above are used unchanged
# except that Problem / cost / analyze got one optional argument each (weights / wrist_weight, default None = old behaviour))
# - the front window of the noise floor can be searched in a narrower range (front_search_s) and d0 is capped (d0_cap_theta_deg);
# - the guide's tilt rule can be 'frames_median' (enough frames with theta2d <= tilt_max_deg, and their median in range) instead of the p95 limit;
# - wrist_mode keep | exclude | downweight changes the f estimate only (template, PnP cost, uncertainty); transition and guide use the default joint set.
# ====================================================================================================

def grev_front_window(times, present, kp, mask, box, pcfg, search_s) -> dict:
    """front_window with its search range replaced by search_s = [lo, hi] (seconds); the window never reaches outside [lo, hi]."""
    lo, hi = search_s
    return front_window(times, present, kp, mask, box, {**pcfg, "front_skip_s": float(lo), "front_phase_s": float(hi)})


def grev_transition(times, present, kp, mask, box, T, est_ids, front_idx, p: dict) -> dict:
    """pseg_transition with the noise floor capped: d_thr = max(k_d * min(d0, d0_cap), 1 - cos theta_min), d0_cap = 1 - cos d0_cap_theta_deg.
    No d0_cap_theta_deg in p = 20's rule unchanged. d0 itself is reported uncapped; d0_capped says whether the cap acted."""
    cap_deg = p.get("d0_cap_theta_deg")
    if cap_deg is None:
        return {**pseg_transition(times, present, kp, mask, box, T, est_ids, front_idx, p), "d0_cap": None, "d0_capped": False, "warnings": []}
    t = np.asarray(times, float)
    d, _ = pseg_series(kp, mask, present, box, t, T, est_ids)
    dw = d[[int(i) for i in front_idx]]
    if not np.isfinite(dw).any():
        return {**pseg_transition(times, present, kp, mask, box, T, est_ids, front_idx, p), "d0_cap": None, "d0_capped": False, "warnings": []}
    d0 = float(np.nanpercentile(dw, p["noise_percentile"]))
    cap = 1.0 - math.cos(math.radians(float(cap_deg)))
    capped = bool(d0 > cap)
    k_eff = float(p["k_d"]) * cap / d0 if capped else float(p["k_d"])           # k_eff * d0 = k_d * min(d0, cap)
    res = pseg_transition(times, present, kp, mask, box, T, est_ids, front_idx, {**p, "k_d": k_eff})
    floor = 1.0 - math.cos(math.radians(p["theta_min_deg"]))
    if capped:
        res["d_thr"] = float(max(float(p["k_d"]) * cap, floor))
    warn = [f"정면 창에 기울임이 섞였을 수 있음: d0 {d0:.3f} > 상한 {cap:.3f} (θ {cap_deg:.0f}° 환산), 문턱에는 상한값을 씀"] if capped else []
    return {**res, "d0_cap": cap, "d0_capped": capped, "warnings": warn}


def grev_metrics(times, present, box, W, H, theta, role, t_transition, front_jitter_rel, axis_bins, finger_ext_min, tilt_max_deg: float = 45.0) -> dict:
    """metrics_from_arrays plus the new tilt-rule inputs: tilt-phase frames with theta2d <= tilt_max_deg (count) and the median theta2d of those frames."""
    m = metrics_from_arrays(times, present, box, W, H, theta, role, t_transition, front_jitter_rel, axis_bins, finger_ext_min)
    th = np.asarray(theta, float)
    tilt = (np.asarray(role) == "tilt") & np.isfinite(th)
    sel = tilt & (th <= tilt_max_deg)
    m.update({"tilt_frames": int(tilt.sum()), "tilt_frames_le_max": int(sel.sum()), "tilt_median_le_max_deg": float(np.median(th[sel])) if sel.any() else None})
    return m


def grev_conditions_check(m: dict, c: dict) -> dict:
    """conditions_check with a selectable tilt rule. tilt_rule 'p95_max' (default, also when the key is missing) = 17's check unchanged; 'frames_median' replaces
    the theta2d p95 item by: >= min_tilt_frames tilt-phase frames with theta2d <= tilt_max_deg, and the median theta2d of those frames inside the range."""
    if c.get("tilt_rule", "p95_max") == "p95_max" or "failed" in m:
        return conditions_check(m, c)
    lo, hi = c["theta_tilt_median_range_deg"]
    nmin, tmax = int(c["min_tilt_frames"]), float(c["tilt_max_deg"])

    def ok(x, f):
        return x is not None and bool(f(x))
    items = {
        "box_short_ratio": ok(m["box_short_ratio"], lambda x: x <= c["box_short_ratio_max"]),
        "edge_rate": ok(m["edge_rate"], lambda x: x <= c["edge_rate_max"]),
        "transition": ok(m["transition_s"], lambda x: x >= c["transition_min_s"]),
        "tilt_frames": ok(m["tilt_frames_le_max"], lambda x: x >= nmin),
        "theta_tilt_median": ok(m["tilt_median_le_max_deg"], lambda x: lo <= x <= hi),
        "axis_bins": ok(m["axis_bins"], lambda x: x >= c["axis_bins_min"]),
        "finger_extension": ok(m["finger_extension_min"], lambda x: x >= c["finger_extension_min"]),
    }
    msg = {"box_short_ratio": lambda: f"손 박스/짧은 변 {cond_f(m['box_short_ratio'], 2)} > {c['box_short_ratio_max']}",
           "edge_rate": lambda: f"가장자리 접촉 {cond_p(m['edge_rate'])} > {100 * c['edge_rate_max']:.0f}%",
           "transition": lambda: ("자동 전환 없음" if m["transition_s"] is None else f"자동 전환 {m['transition_s']:.2f} s < {c['transition_min_s']} s"),
           "tilt_frames": lambda: f"기울기 구간 θ2d ≤ {tmax:.0f}° 프레임 {m['tilt_frames_le_max']}장 < {nmin}장",
           "theta_tilt_median": lambda: f"θ2d ≤ {tmax:.0f}° 프레임의 θ2d 중앙값 {cond_f(m['tilt_median_le_max_deg'], 0)}°가 {lo:.0f}–{hi:.0f}° 밖",
           "axis_bins": lambda: f"축 덮임 {m['axis_bins']}/6 < {c['axis_bins_min']}/6",
           "finger_extension": lambda: f"손가락 펴짐 {cond_f(m['finger_extension_min'], 3)} < {c['finger_extension_min']}"}
    return {"guide": all(items.values()), "items": items, "reasons": [msg[k]() for k, v in items.items() if not v]}


def grev_template_extent(T: np.ndarray, box_long_front: float) -> dict:
    """Spatial extent of the template points: length along their first principal axis (the hand's long axis) and width along the second, in px of the front
    window and as % of the hand box's long side there."""
    X = np.asarray(T, float) - np.asarray(T, float).mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    proj = X @ Vt.T
    long_px, width_px = float(np.ptp(proj[:, 0])), float(np.ptp(proj[:, 1]))
    return {"n_points": int(len(X)), "long_px": long_px, "width_px": width_px, "long_pct_box": 100.0 * long_px / box_long_front, "width_pct_box": 100.0 * width_px / box_long_front}


def _grev_project(T, obs, vis, f, cx, cy):
    """IPPE fit of the planar template at focal f: reprojected points and the per-point residuals."""
    K = K_of(f, cx, cy)
    r, R, t = frame_fit(T, obs, vis, K)
    X = np.c_[T[vis], np.zeros(int(vis.sum()))] @ R.T + t
    return (K[:2, :2] @ (X[:, :2] / X[:, 2:3]).T + K[:2, 2:3]).T, r


def grev_f_signal(a: dict, kp, mask, ids, f_ref: float, ratios=(0.8, 1.2)) -> dict:
    """Per tilt frame: mean displacement of the reprojected points when the focal is set to ratios[0] * f_ref and ratios[1] * f_ref (pose refitted for each),
    divided by the RMS residual of that frame at f_ref. Median over the tilt frames. Diagnostic of how much of the data can tell f apart from noise."""
    T = np.array(a["template"], float)
    cx, cy = a["principal_point"]
    sig, res, rat = [], [], []
    for i, rl in enumerate(a["role"]):
        if rl != "tilt":
            continue
        obs, vis = np.asarray(kp[i])[ids], np.asarray(mask[i])[ids]
        if vis.sum() < 4:
            continue
        pa, _ = _grev_project(T, obs, vis, f_ref * ratios[0], cx, cy)
        pb, _ = _grev_project(T, obs, vis, f_ref * ratios[1], cx, cy)
        _, r0 = _grev_project(T, obs, vis, f_ref, cx, cy)
        s, r = float(np.linalg.norm(pa - pb, axis=1).mean()), float(np.sqrt((r0 ** 2).mean()))
        sig.append(s), res.append(r), rat.append(s / max(r, 1e-9))
    if not rat:
        return {"n_frames": 0, "signal_px_median": None, "residual_px_median": None, "ratio_median": None}
    return {"n_frames": len(rat), "signal_px_median": float(np.median(sig)), "residual_px_median": float(np.median(res)), "ratio_median": float(np.median(rat))}


def grev_f_run(kp, mask, present, box, cls, times, size_wh, cfg: dict, c: dict, mode: str, a_keep: dict | None, seed: int) -> dict:
    """f estimate (fixed_template and joint_delta, 17's verdict) for one wrist_mode, plus the template extent and the f signal. mode 'keep' = 17's analysis
    (a_keep is reused); 'exclude' removes joint 0 from every joint set; 'downweight' multiplies the wrist's weight in the reprojection cost by wrist_weight.
    exclude_large_tilt (phase_segmentation, default off) drops tilt-phase frames above max_tilt_deg from the estimate."""
    fe, p = c.get("f_estimation") or {}, c["phase_segmentation"]
    cfg_use = pseg_wrist_cfg(cfg) if mode == "exclude" else cfg
    ww = float(fe.get("wrist_weight", 0.3)) if mode == "downweight" else None
    ids = list(cfg_use["joints"]["sets"][cfg_use["joints"]["estimate_set"]])
    base = a_keep if mode == "keep" and a_keep is not None else analyze(kp, mask, present, box, cls, times, size_wh, cfg_use, tilt_start=None,
                                                                         methods_full=("fixed_template",), seed=seed, wrist_weight=ww)
    res, n_before, n_after = base, None, None
    if p.get("exclude_large_tilt") and "per_method" in base:
        th = np.array([np.nan if x is None else x for x in base["theta2d_deg"]], float)
        keep = pseg_tilt_keep(times, present, th, base["tilt_start_s"], p["max_tilt_deg"])
        n_before = int((np.asarray(present, bool) & (np.asarray(times) >= base["tilt_start_s"])).sum())
        n_after = int((keep & (np.asarray(times) >= base["tilt_start_s"])).sum())
        res = analyze(kp, mask, keep, box, cls, times, size_wh, cfg_use, tilt_start=None, methods_full=("fixed_template",), seed=seed, wrist_weight=ww)
    if "per_method" not in res:
        return {"failed": res.get("failed", "분석 실패"), "mode": mode}
    out = pseg_f_summary(res, present, cfg_use, {"mode": mode, "wrist_weight": ww, "n_tilt_before": n_before, "n_tilt_after": n_after, "joint_ids": ids,
                                                 "jointset_range": res["per_method"]["fixed_template"]["jointset_range"]})
    bl = box_long(np.asarray(box, float))
    fidx = [int(i) for i in res["front_window"]["idx"]]
    out["template_extent"] = grev_template_extent(np.array(res["template"]), float(np.nanmedian(bl[fidx])))
    f_ref = c.get("reference_f_px")
    out["f_signal"] = grev_f_signal(res, kp, mask, ids, float(f_ref)) if f_ref else None
    return out


def grev_analyze(kp, mask, present, box, cls, times, size_wh, cfg: dict, c: dict, seed: int | None = None) -> dict:
    """One tracked hand. cfg = the pipeline settings, c = the experiment's settings (phase_segmentation, conditions, optional compare_20 / f_estimation).
    Default analysis = 17's (unchanged call); legacy and configured transitions; 20's transition for comparison; the shooting guide with the configured and with
    20's criteria; the f estimate for every wrist_mode in f_estimation.compare_wrist_modes."""
    W, H = size_wh
    seed = int(cfg["experiment"]["seed"]) if seed is None else seed
    p, ccfg, fe = c["phase_segmentation"], c["conditions"], c.get("f_estimation") or {}
    est_ids = list(cfg["joints"]["sets"][cfg["joints"]["estimate_set"]])
    a0 = analyze(kp, mask, present, box, cls, times, size_wh, cfg, tilt_start=None, methods_full=("fixed_template",), seed=seed)
    if "per_method" not in a0:
        return {"failed": a0.get("failed", "분석 실패")}
    theta = np.array([np.nan if x is None else x for x in a0["theta2d_deg"]], float)
    T = np.array(a0["template"])
    t_legacy = a0.get("auto_transition_s")
    tmax = float(ccfg.get("tilt_max_deg", 45.0))

    def guide_of(t_tr, cc, with_new=False):
        if with_new:
            m = grev_metrics(times, present, box, W, H, theta, a0["role"], t_tr, a0.get("front_jitter_rel"), a0["axis_coverage"]["bins"], a0["finger_extension"]["min"], tmax)
        else:
            m = metrics_from_arrays(times, present, box, W, H, theta, a0["role"], t_tr, a0.get("front_jitter_rel"), a0["axis_coverage"]["bins"], a0["finger_extension"]["min"])
        g = grev_conditions_check(m, cc)
        return {"transition_s": t_tr, "guide": g["guide"], "reasons": g["reasons"], "items": g["items"], "metrics": {k: v for k, v in m.items() if k != "series"}}

    fw0 = a0["front_window"]
    fw = fw0 if not p.get("front_search_s") else grev_front_window(times, present, kp, mask, box, cfg["phases"], p["front_search_s"])
    if len(fw["idx"]) == 0:
        return {"failed": "정면 창을 만들 수 없음"}
    tr = grev_transition(times, present, kp, mask, box, T, est_ids, fw["idx"], p)
    tr_nm = grev_transition(times, present, kp, mask, box, T, est_ids, fw["idx"], {**p, "use_motion": False}) if p["use_motion"] else tr
    new_rule = ccfg.get("tilt_rule", "p95_max") != "p95_max"
    out = {"default": pseg_f_summary(a0, present, cfg, {"front_window": [fw0.get("t0"), fw0.get("t1")], "auto_transition_s": t_legacy}),
           "legacy_theta": {"t_transition": t_legacy, **guide_of(t_legacy, ccfg, new_rule)},
           "transition": {**{k: tr.get(k) for k in ("t_transition", "d0", "d_thr", "v0", "v_thr", "front_end_s", "motion_v_at_start", "skipped_no_motion_s", "failed", "d0_cap",
                                                    "d0_capped", "warnings")},
                          "front_window": [fw.get("t0"), fw.get("t1")], "method": c.get("transition_method", "linear_d"),
                          **guide_of(t_legacy if c.get("transition_method") == "legacy_theta" else tr.get("t_transition"), ccfg, new_rule)},
           "transition_no_motion": {"t_transition": tr_nm.get("t_transition")},
           "series": {**tr.get("series", {}), "theta": theta.tolist(), "front": [fw.get("t0"), fw.get("t1")]}}
    if c.get("compare_20"):
        c20 = c["compare_20"]
        t20 = pseg_transition(times, present, kp, mask, box, T, est_ids, fw0["idx"], c20["phase_segmentation"])
        t20nm = pseg_transition(times, present, kp, mask, box, T, est_ids, fw0["idx"], {**c20["phase_segmentation"], "use_motion": False}) if c20["phase_segmentation"]["use_motion"] else t20
        out["transition_20"] = {**{k: t20.get(k) for k in ("t_transition", "d0", "d_thr", "v0", "v_thr", "skipped_no_motion_s", "failed")}, "front_window": [fw0.get("t0"), fw0.get("t1")],
                                **guide_of(t20.get("t_transition"), c20["conditions"]), "t_no_motion": t20nm.get("t_transition")}
    out["wrist"] = {}
    modes = list(fe.get("compare_wrist_modes") or [fe.get("wrist_mode", "keep")])
    for mode in modes:
        out["wrist"][mode] = grev_f_run(kp, mask, present, box, cls, times, size_wh, cfg, c, mode, a0, seed)
    return out


def grev_job(args) -> dict:
    """(video, detector) job for the process pool: the 17 run's frame list and detections, grev_analyze, and the check that the keep mode (and 20's transition) reproduce
    17's (and 20's) recorded results."""
    video, det, run_dir, pcfg, c, conf_min, ref20 = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cv2.setNumThreads(1)
    pc = copy.deepcopy(pcfg)
    pc["tracking"]["conf_min"] = float(conf_min)
    d = Path(run_dir) / det
    res = json.loads((d / "results.json").read_text())
    dets = load_detections(d / "detections.npz")
    frames = res["frames"]["list"]
    check_alignment(frames, dets)
    W, H = res["normalization"]["output"]["display_w"], res["normalization"]["output"]["display_h"]
    times = np.array([r["time_s"] for r in frames])
    tr = track(dets, times, (W, H), pc["tracking"], front_start_s=float(pc["phases"]["front_skip_s"]))
    out = grev_analyze(tr["kp"], tr["mask"], tr["present"], tr["box"], tr["cls"], times, (W, H), pc, c)
    out.update({"video": video, "detector": det, "size": [W, H], "n_frames": len(frames)})
    if "default" in out and "solutions" in out["default"]:
        s = out["default"]["solutions"]
        same = {"f_fixed_template": abs(s["fixed_template"]["f"] - res.get("f_hand_px", np.nan)) <= 1e-6 * max(1.0, abs(s["fixed_template"]["f"])),
                "f_joint_delta": abs(s["joint_delta"]["f"] - res.get("f_joint_delta_px", np.nan)) <= 1e-6 * max(1.0, abs(s["joint_delta"]["f"])),
                "verdict": out["default"]["verdict"] == res.get("verdict")}
        k = (out.get("wrist") or {}).get("keep")
        if k and "solutions" in k:
            same["keep_equals_default"] = bool(k["solutions"]["fixed_template"]["f"] == s["fixed_template"]["f"] and k["verdict"] == out["default"]["verdict"])
        out["same_as_17"] = {**same, "all": bool(all(same.values())), "f_17": res.get("f_hand_px"), "verdict_17": res.get("verdict")}
        if ref20 is not None:
            t = out.get("transition_20") or {}
            s20 = {"f_fixed_template": abs(s["fixed_template"]["f"] - ref20["default"]["solutions"]["fixed_template"]["f"]) <= 1e-9 * max(1.0, abs(s["fixed_template"]["f"])),
                   "verdict": out["default"]["verdict"] == ref20["default"]["verdict"]}
            if t:
                a, b = t.get("t_transition"), ref20["linear_d"]["t_transition"]
                s20["transition_linear_d"] = bool((a is None and b is None) or (a is not None and b is not None and abs(a - b) < 1e-9))
                s20["d0"] = bool(t.get("d0") is not None and abs(t["d0"] - ref20["linear_d"]["d0"]) < 1e-12)
            out["same_as_20"] = {**s20, "all": bool(all(s20.values()))}
    return out


def grev_adoption(jobs: list, ref_f: float, rules: dict, mode: str, base: str = "keep") -> dict:
    """The fixed adoption rules of experiment 21 for one wrist_mode against `base`, over the (video, detector) pairs (f = fixed_template; a grid-edge f has no value).
    err = |f / ref_f - 1|; no value counts as the worst possible error. A pair is improved when `mode` has a value and the base has none or a larger error, worsened
    in the opposite case (equal or both without value: neither). Rules: 1 grid-edge count not larger; 2 improved pairs >= min_improved_pairs; 3 median error at least
    median_rel_drop lower (and finite); 4 leaving out each video, improved > worsened and the median error lower; 5 no video where all its detectors get worse by
    >= worse_pp percentage points (or lose their value)."""
    def err(j, m):
        w = (j.get("wrist") or {}).get(m)
        if not w or "solutions" not in w or w["solutions"]["fixed_template"]["at_grid_edge"]:
            return None
        return abs(w["solutions"]["fixed_template"]["f"] / ref_f - 1.0)
    pairs = [j for j in jobs if "failed" not in j and mode in (j.get("wrist") or {}) and base in (j.get("wrist") or {})]
    eb, em = [err(j, base) for j in pairs], [err(j, mode) for j in pairs]
    inf = float("inf")
    val = lambda e: inf if e is None else e                                                             # noqa: E731
    improved = lambda b, m_: m_ is not None and (b is None or m_ < b)                                   # noqa: E731
    worsened = lambda b, m_: b is not None and (m_ is None or m_ > b)                                   # noqa: E731
    idx = list(range(len(pairs)))

    def stats(ix):
        mb, mm = float(np.median([val(eb[i]) for i in ix])), float(np.median([val(em[i]) for i in ix]))
        return {"n": len(ix), "improved": sum(improved(eb[i], em[i]) for i in ix), "worsened": sum(worsened(eb[i], em[i]) for i in ix), "median_base": mb, "median_mode": mm}
    allp = stats(idx)
    n_edge = lambda es: sum(e is None for e in es)                                                      # noqa: E731
    r1 = n_edge(em) <= n_edge(eb)
    r2 = allp["improved"] >= int(rules["min_improved_pairs"])
    r3 = bool(np.isfinite(allp["median_mode"]) and allp["median_mode"] <= (1.0 - float(rules["median_rel_drop"])) * allp["median_base"])
    videos = sorted({j["video"] for j in pairs})
    loo = {}
    for v in videos:
        s = stats([i for i in idx if pairs[i]["video"] != v])
        loo[v] = {**s, "direction_2": s["improved"] > s["worsened"], "direction_3": s["median_mode"] < s["median_base"]}
    r4 = all(x["direction_2"] and x["direction_3"] for x in loo.values())
    pp = float(rules["worse_pp"]) / 100.0
    bad_videos = []
    for v in videos:
        ix = [i for i in idx if pairs[i]["video"] == v]
        if ix and all(eb[i] is not None and (em[i] is None or em[i] - eb[i] >= pp) for i in ix):
            bad_videos.append(v)
    r5 = not bad_videos
    return {"mode": mode, "base": base, "n_pairs": len(pairs), "edge_base": n_edge(eb), "edge_mode": n_edge(em), "all": allp, "leave_one_out": loo, "worse_videos": bad_videos,
            "rules": {"1_edge_not_more": r1, "2_improved_pairs": r2, "3_median_drop": r3, "4_leave_one_out": r4, "5_no_video_all_worse": r5}, "candidate": bool(r1 and r2 and r3 and r4 and r5)}


# ====================================================================================================
# fresp  (experiment 22: does the candidate (mediapipe_hands + wrist_mode exclude) respond to f?; new section, earlier functions are used unchanged)
# A1: the estimator is scale-equivariant (keypoints scaled about the image centre -> f scaled by the same factor).
# A2: centre-cropped and enlarged frames (true f = s * f, principal point unchanged), detector re-run on them; AnyCalib re-run on them.
# A3: the candidate on every video of the range, original frames, with the hand's box size in the front window.
# B (distance / lens series) is prepared in the yaml only.
# ====================================================================================================

def fresp_target_videos(input_dir, vcfg: dict, used_in_21: list) -> list:
    """Every IMG_<number> video file in input_dir with its inclusion: number inside number_range [lo, hi] (hi null = open) and not in exclude."""
    lo, hi = vcfg["number_range"]
    excl = {int(x) for x in (vcfg.get("exclude") or [])}
    exts = {e.lower() for e in vcfg.get("extensions", [".mov"])}
    rows = []
    for p in sorted(Path(input_dir).expanduser().iterdir()):
        m = re.fullmatch(r"IMG_(\d+)", p.stem)
        if not m or p.suffix.lower() not in exts:
            continue
        n = int(m.group(1))
        reason = None
        if n in excl:
            reason = (vcfg.get("exclude_reasons") or {}).get(n) or (vcfg.get("exclude_reasons") or {}).get(str(n)) or "제외 목록"
        elif n < lo:
            reason = f"번호 {lo} 미만 (촬영 기준이 다름)"
        elif hi is not None and n > hi:
            reason = f"번호 {hi} 초과"
        rows.append({"video": p.name, "number": n, "included": reason is None, "reason": reason, "used_in_21": p.name in set(used_in_21)})
    return rows


def fresp_scale_track(kp, box, s: float, center) -> tuple:
    """Keypoints and box scaled by s about `center` (x, y): the same picture enlarged about the principal point (A1). No re-detection."""
    c = np.asarray(center, float)
    kp2 = c + float(s) * (np.asarray(kp, float) - c)
    b = np.asarray(box, float)
    box2 = np.c_[c + float(s) * (b[:, :2] - c), c + float(s) * (b[:, 2:] - c)]
    return kp2, box2


def fresp_zoom_matrix(s: float, W: int, H: int) -> np.ndarray:
    """Forward affine map x' = s (x - c) + c of pixel indices about the image centre c = ((W - 1) / 2, (H - 1) / 2), i.e. the optical axis (W / 2, H / 2) in the
    continuous coordinates of the keypoints stays fixed. The crop is (W / s) x (H / s) around that centre, enlarged to W x H."""
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    return np.array([[s, 0.0, (1.0 - s) * cx], [0.0, s, (1.0 - s) * cy]], float)


def fresp_zoom_image(img, s: float, interp: int):
    h, w = img.shape[:2]
    return cv2.warpAffine(img, fresp_zoom_matrix(s, w, h), (w, h), flags=interp, borderMode=cv2.BORDER_REPLICATE)


def fresp_zoom_frames(frames: list, s: float, out_dir: Path, interp: int) -> list:
    """The frame list with every PNG centre-cropped and enlarged by s (same index / frame / time_s; new PNG files and hashes)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in frames:
        img = cv2.imread(r["png"], cv2.IMREAD_COLOR)
        p = out_dir / Path(r["png"]).name
        cv2.imwrite(str(p), fresp_zoom_image(img, s, interp))
        rows.append({**{k: r[k] for k in ("index", "frame", "time_s")}, "png": str(p), "sha256": sha256(p)})
    return rows


def fresp_edge_rate(box, present, W, H) -> float:
    present = np.asarray(present, bool)
    return float(edge_flags(box, present, W, H)[present].mean()) if present.any() else 0.0


def fresp_f_pair(kp, mask, present, box, cls, times, size_wh, cfg: dict, modes, seed: int, ratio_range_scale: float = 1.0, front_idx=None, tilt_start=None) -> dict:
    """f of both estimators for each wrist mode (no uncertainty, no bootstrap). ratio_range_scale multiplies the f grid's range (A1 diagnostic: grid scaled with the picture)."""
    out = {}
    for mode in modes:
        c = pseg_wrist_cfg(cfg) if mode == "exclude" else copy.deepcopy(cfg)
        if ratio_range_scale != 1.0:
            c["estimate"]["f_ratio_range"] = [float(x) * ratio_range_scale for x in c["estimate"]["f_ratio_range"]]
        a = analyze(kp, mask, present, box, cls, times, size_wh, c, front_idx=front_idx, tilt_start=tilt_start, methods_full=(), seed=seed)
        if "solutions" not in a:
            out[mode] = {"failed": a.get("failed", "분석 실패")}
            continue
        hi = a["grid"][-1]
        out[mode] = {m: {"f": a["solutions"][m]["f"], "at_grid_edge": a["solutions"][m]["at_grid_edge"],
                         "side": None if not a["solutions"][m]["at_grid_edge"] else ("상한" if a["solutions"][m]["f"] >= hi * (1 - 1e-6) else "하한")}
                     for m in ("fixed_template", "joint_delta")}
    return out


def fresp_thermal_wait(hi: float = 70.0, lo: float = 65.0, poll_s: float = 15.0, log=print) -> float:
    """GPU thermal guard before a GPU stage: while any visible GPU is >= hi deg C, wait until all are < lo. Returns the seconds waited."""
    ids = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip().isdigit()]

    def temps():
        q = subprocess.check_output(["nvidia-smi", "--query-gpu=index,temperature.gpu", "--format=csv,noheader,nounits"], text=True).strip().splitlines()
        t = {x.split(",")[0].strip(): float(x.split(",")[1]) for x in q}
        return [t[i] for i in (ids or list(t)) if i in t]
    waited = 0.0
    try:
        if max(temps() or [0.0]) >= hi:
            log(f"  GPU 온도 {max(temps()):.0f}°C ≥ {hi:.0f}°C: {lo:.0f}°C 미만이 될 때까지 대기")
            while max(temps() or [0.0]) >= lo:
                time.sleep(poll_s)
                waited += poll_s
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass
    return waited


@contextlib.contextmanager
def fresp_gpu_lock(path):
    """Serialise the GPU stage (AnyCalib) of parallel video jobs with an exclusive file lock."""
    import fcntl
    with open(path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def fresp_zoom_excluded(s: float, edge_rate: float, edge_rate_max: float) -> bool:
    """A zoom above 1 is dropped when more of its detected frames touch the border than edge_rate_max; the 1.0 control is never dropped."""
    return bool(float(s) > 1.0 and edge_rate > edge_rate_max)


def fresp_ratio_status(f_s: dict | None, f_1: dict | None, s: float, tol: float) -> dict:
    """Ratio f(s) / f(1.0) of one estimator (dicts {f, at_grid_edge}); within = |ratio / s - 1| <= tol. A grid-edge value has no ratio."""
    if f_s is None or f_1 is None or f_s.get("at_grid_edge") or f_1.get("at_grid_edge") or not f_1.get("f"):
        return {"ratio": None, "within": None}
    r = f_s["f"] / f_1["f"]
    return {"ratio": float(r), "within": bool(abs(r / s - 1.0) <= tol), "rel_to_s": float(r / s - 1.0)}


def fresp_prepare(args) -> dict:
    """Stage 1, one video: normalise, extract the frames (the 17 pipeline's) and write the centre-cropped / enlarged copies for every zoom factor into frames_root.
    args = (video, input_dir, pcfg, c, frames_root). The frame images stay in frames_root until the run ends (then they are deleted)."""
    video, input_dir, pcfg, c, frames_root = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cv2.setNumThreads(1)
    root = Path(frames_root) / Path(video).stem
    tmp = Path(tempfile.mkdtemp(prefix="hf22n_", dir=frames_root))
    try:
        norm = normalize(Path(input_dir).expanduser() / video, tmp / "n.mkv", pcfg)
        W, H = norm["output"]["display_w"], norm["output"]["display_h"]
        frames, _ = extract_frames(tmp / "n.mkv", root / "orig", norm["output"]["fps"], None, pcfg["frames"]["detect_fps"])
        out = {"video": video, "size": [W, H], "frames": frames, "zoom": {}}
        a2c = c.get("a2") or {}
        if a2c.get("enabled"):
            interp = getattr(cv2, a2c.get("interpolation", "INTER_CUBIC"))
            for z in a2c["zoom_factors"]:
                out["zoom"][str(z)] = fresp_zoom_frames(frames, float(z), root / f"z_{z}", interp)
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def fresp_detect_many(tasks: list, dset: dict, cache_dir, n_workers: int = 1, log=print) -> dict:
    """Stage 2: mediapipe (CPU) on several frame lists with swap_mediapipe_hands unchanged (same model, mode, confidence thresholds, input resolution; cache by frame
    hashes + settings). tasks = [(key, frames)]. n_workers = 1: one list after the other (the old behaviour); > 1: that many lists at the same time, each worker process
    runs its list whole and in the list's own frame order, so the saved keypoints do not depend on which list finishes first. Lists with identical frames (the 1.0 control
    and the original) are detected once. Returns {key: npz path}."""
    from concurrent.futures import ThreadPoolExecutor
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    uniq = {}
    for key, frames in tasks:
        uniq.setdefault(tuple(r["sha256"] for r in frames), frames)

    def one(item):
        sig, frames = item
        npz, _ = swap_mediapipe_hands(frames, dset, cache_dir, None, log)
        return sig, npz
    items = list(uniq.items())
    if int(n_workers) <= 1:
        done = dict(one(it) for it in items)
    else:
        with ThreadPoolExecutor(max_workers=int(n_workers)) as ex:
            done = dict(ex.map(one, items))
    return {key: done[tuple(r["sha256"] for r in frames)] for key, frames in tasks}


def fresp_job(args) -> dict:
    """Stage 3, one video: the original candidate analysis (keep and exclude; 17 detections when a 17 run exists, else the stage-2 detection), A1 and A2 from the stage-2
    detections of the enlarged frames, AnyCalib on them. args = (video, prep, src_dir | None, npz_orig | None, npz_zoom {zoom: path}, pcfg, c, lock_dir, ref21 | None,
    used_in_21, gpu_id | None). gpu_id (a physical GPU index, one fresh process per video) selects the GPU of this video's AnyCalib runs; None keeps the inherited device."""
    video, prep, src_dir, npz_orig, npz_zoom, pcfg, c, lock_dir, ref21, used_in_21, gpu_id = args
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cv2.setNumThreads(1)
    cand = c["candidate"]
    det_name = cand["detector"]
    dset = c["detector_settings"][det_name]
    pc = copy.deepcopy(pcfg)
    pc["tracking"]["conf_min"] = float(dset["conf_min"])
    W, H = prep["size"]
    frames = prep["frames"]
    times = np.array([r["time_s"] for r in frames])
    res = {"video": video, "detector": det_name, "used_in_21": bool(used_in_21), "size": [W, H], "n_frames": len(frames), "a2": {}, "a1": {}}
    if src_dir is not None:
        r17 = json.loads((Path(src_dir) / det_name / "results.json").read_text())
        if [(x["index"], x["sha256"]) for x in frames] != [(x["index"], x["sha256"]) for x in r17["frames"]["list"]]:
            raise RuntimeError(f"{video}: re-created frames differ from the 17 run's frame list")
        dets = load_detections(Path(src_dir) / det_name / "detections.npz")
        res["detections"] = "17 run"
    else:
        dets = load_detections(Path(npz_orig))
        res["detections"] = "new mediapipe run on the normalised frames"
    check_alignment(frames, dets)
    tr = track(dets, times, (W, H), pc["tracking"], front_start_s=float(pc["phases"]["front_skip_s"]))
    seed = int(pc["experiment"]["seed"])
    a3 = bool((c.get("a3") or {}).get("enabled"))
    a_keep = analyze(tr["kp"], tr["mask"], tr["present"], tr["box"], tr["cls"], times, (W, H), pc, tilt_start=None, methods_full=("fixed_template",) if a3 else (), seed=seed)
    if "solutions" not in a_keep:
        res["failed"] = a_keep.get("failed", "분석 실패")
        return res
    cj = {k: c[k] for k in ("phase_segmentation", "f_estimation") if k in c}
    res["original"] = {}
    for mode in (c["f_estimation"]["compare_wrist_modes"] if a3 else []):
        res["original"][mode] = grev_f_run(tr["kp"], tr["mask"], tr["present"], tr["box"], tr["cls"], times, (W, H), pc, cj, mode, a_keep, seed)
    fidx = [int(i) for i in a_keep["front_window"]["idx"]]
    bl = box_long(tr["box"])
    res["hand_box_short_ratio_front"] = float(np.nanmedian(bl[fidx]) / min(W, H))
    res["edge_rate"] = fresp_edge_rate(tr["box"], tr["present"], W, H)
    res["detect_rate"] = float(tr["present"].mean())
    res["frames_table"] = {"index": [int(r["index"]) for r in frames], "time_s": times.tolist(), "present": tr["present"].tolist(),
                           "box_short_ratio": (bl / min(W, H)).tolist(), "theta2d_deg": a_keep["theta2d_deg"], "role": a_keep["role"]}
    if ref21 is not None:
        s21 = {}
        for mode in ("keep", "exclude"):
            a, b = res["original"].get(mode), (ref21.get("wrist") or {}).get(mode)
            if a and b and "solutions" in a and "solutions" in b:
                s21[mode] = bool(a["solutions"]["fixed_template"]["f"] == b["solutions"]["fixed_template"]["f"] and a["verdict"] == b["verdict"])
        res["same_as_21"] = {**s21, "all": bool(s21 and all(s21.values()))}
    modes = list(c["f_estimation"]["compare_wrist_modes"])
    # ---- A1: keypoints and boxes scaled about the image centre, no re-detection
    a1c = c.get("a1") or {}
    if a1c.get("enabled"):
        base = fresp_f_pair(tr["kp"], tr["mask"], tr["present"], tr["box"], tr["cls"], times, (W, H), pc, modes, seed)
        for s in a1c["scales"]:
            kp2, box2 = fresp_scale_track(tr["kp"], tr["box"], s, (W / 2.0, H / 2.0))
            f2 = fresp_f_pair(kp2, tr["mask"], tr["present"], box2, tr["cls"], times, (W, H), pc, modes, seed)
            g2 = fresp_f_pair(kp2, tr["mask"], tr["present"], box2, tr["cls"], times, (W, H), pc, modes, seed, ratio_range_scale=float(s))
            res["a1"][str(s)] = {}
            for mode in modes:
                row = {}
                for m in ("fixed_template", "joint_delta"):
                    b0, b1, b2 = base[mode].get(m), f2[mode].get(m), g2[mode].get(m)
                    ok = b0 and b1 and b2 and not (b0["at_grid_edge"] or b1["at_grid_edge"])
                    row[m] = {"f_1": b0["f"] if b0 else None, "f_s": b1["f"] if b1 else None, "ratio": (b1["f"] / b0["f"]) if ok else None,
                              "ratio_err": (b1["f"] / b0["f"] / s - 1.0) if ok else None,
                              "grid_scaled_ratio_err": (b2["f"] / b0["f"] / s - 1.0) if (ok and not b2["at_grid_edge"]) else None}
                res["a1"][str(s)][mode] = row
    # ---- A2: detections of the enlarged frames (stage 2), the candidate analysis on them, AnyCalib on them
    a2c = c.get("a2") or {}
    if a2c.get("enabled"):
        acfg = copy.deepcopy(pc["anycalib"])
        for s in a2c["zoom_factors"]:
            zfr = prep["zoom"][str(s)]
            zd = load_detections(Path(npz_zoom[str(s)]))
            check_alignment(zfr, zd)
            ztr = track(zd, times, (W, H), pc["tracking"], front_start_s=float(pc["phases"]["front_skip_s"]))
            er = fresp_edge_rate(ztr["box"], ztr["present"], W, H)
            e = {"zoom": float(s), "edge_rate": er, "detect_rate": float(ztr["present"].mean())}
            if fresp_zoom_excluded(float(s), er, float(a2c["edge_rate_max"])):
                e["excluded"] = f"가장자리 접촉 프레임 {100 * er:.1f}% > {100 * float(a2c['edge_rate_max']):.0f}%"
                res["a2"][str(s)] = e
                continue
            e["hand"] = fresp_f_pair(ztr["kp"], ztr["mask"], ztr["present"], ztr["box"], ztr["cls"], times, (W, H), pc, modes, seed)
            if a2c.get("anycalib", True) and pc["anycalib"]["enabled"]:
                with fresp_gpu_lock(Path(lock_dir) / f"gpu_{'' if gpu_id is None else gpu_id}.lock"):
                    e["thermal_wait_s"] = fresp_thermal_wait(float(a2c.get("gpu_pause_c", 70)), float(a2c.get("gpu_resume_c", 65)))
                    ac = anycalib_run(zfr, acfg, float(pc["phases"]["front_phase_s"]), lambda m="": None)
                e["anycalib"] = {m: {"median": v["summary"].get("median"), "n": v["summary"].get("n"), "iqr": v["summary"].get("iqr")} for m, v in ac["models"].items()}
            res["a2"][str(s)] = e
    return res


def fresp_lens_judgment(jobs: list, meta: dict, rules: dict, candidate_mode: str, modes: list) -> dict:
    """B, lens series. 1x estimate = median f (fixed_template, grid-edge values left out) of the 1x videos (every included video that is not a numerator-lens video of the
    metadata); estimated ratio = the numerator-lens video's f / the 1x estimate; reference ratio = its reference f / reference_f_1x_px; within +-ratio_rel_tolerance of
    the reference ratio = "f 변화에 반응함". The verdict applies to the candidate's wrist mode only; the other modes are computed and marked as not judged."""
    r = rules["lens_series"]
    tol = float(r["ratio_rel_tolerance"])
    ok = {j["video"]: j for j in jobs if "failed" not in j}
    lens_vids = [v for v, m in meta.items() if m.get("group") == "lens_series" and m.get("lens") == r["numerator_lens"] and v in ok]
    one_x = [v for v in ok if v not in lens_vids]
    out = {"numerator_videos": lens_vids, "one_x_videos": one_x, "tolerance": tol, "by_video": {}}
    for v in lens_vids:
        ref_ratio = float(meta[v]["reference_f_px"]) / float(r["reference_f_1x_px"])
        lo, hi = ref_ratio * (1.0 - tol), ref_ratio * (1.0 + tol)
        by_mode = {}
        for mode in modes:
            fs = [ok[x]["original"][mode]["solutions"]["fixed_template"] for x in one_x if mode in (ok[x].get("original") or {}) and "solutions" in ok[x]["original"][mode]]
            vals = [x["f"] for x in fs if not x["at_grid_edge"]]
            est1 = float(np.median(vals)) if vals else None
            sol = ((ok[v].get("original") or {}).get(mode) or {}).get("solutions", {}).get("fixed_template")
            f7 = None if sol is None or sol["at_grid_edge"] else sol["f"]
            ratio = None if est1 is None or f7 is None else f7 / est1
            within = None if ratio is None else bool(lo <= ratio <= hi)
            why = None if ratio is not None else ("1배 추정값 없음" if est1 is None else f"{v} f 값 없음 ({sol['side']})" if sol else f"{v} 분석 없음")
            by_mode[mode] = {"judged": mode == candidate_mode, "f_numerator_px": f7, "f_1x_median_px": est1, "n_1x_values": len(vals), "n_1x_videos": len(fs), "ratio": ratio,
                             "reference_ratio": ref_ratio, "range": [lo, hi], "within": within, "unavailable": why,
                             "verdict": None if mode != candidate_mode else ("판정 불가" if within is None else ("f 변화에 반응함" if within else "f 변화에 반응하지 않음"))}
        out["by_video"][v] = by_mode
    return out


# ====================================================================================================
# ferr  (experiment 23: why do the per-video errors differ?; new section, diagnostics only - earlier functions are used unchanged)
# Video-level factors (front window, motion, hand size / position, hand-shape change, detection / estimation), frame-level preferred f from every tilt frame's own cost
# curve, and f re-estimated on subsets fixed in the yaml. New detections are never made; the cached mediapipe (and, for comparison, wilor_yolo / rtmpose) detections are read.
# ====================================================================================================

FERR_FINGERS = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}          # MCP joint of every finger (the PIP / DIP / tip joints follow)


def ferr_long_axis(T: np.ndarray, ids=None) -> np.ndarray:
    """Unit direction of the hand's long axis in template coordinates: mean of the fingertips (8, 12, 16, 20) minus mean of the MCP joints (5, 9, 13, 17) when the joint set
    has them, else the template's first principal axis."""
    T = np.asarray(T, float)
    pos = {j: k for k, j in enumerate(ids)} if ids is not None else {}
    if all(j in pos for j in (5, 9, 13, 17, 8, 12, 16, 20)):
        v = T[[pos[j] for j in (8, 12, 16, 20)]].mean(0) - T[[pos[j] for j in (5, 9, 13, 17)]].mean(0)
        return v / np.linalg.norm(v)
    return np.linalg.svd(T - T.mean(0), full_matrices=False)[2][0]


def ferr_affine(T: np.ndarray, P: np.ndarray) -> tuple:
    """Least-squares affine T -> P (P ~ T @ M + t). Returns M (2, 2), t (2,)."""
    A = np.c_[T, np.ones(len(T))]
    X, *_ = np.linalg.lstsq(A, P, rcond=None)
    return X[:2], X[2]


def ferr_tilt_class(T: np.ndarray, P: np.ndarray, theta_deg: float, cfg: dict, ids=None) -> dict:
    """Tilt kind of one frame. The affine T -> P compresses one template direction v (right singular vector of the smaller singular value); the tilt (rotation) axis is
    perpendicular to v. angle = angle between the rotation axis and the hand's long axis (fingertips minus MCP joints, see ferr_long_axis) = 90 deg - angle(v, long axis). angle < roll_max_deg: roll (rotation about the long
    axis, 굴림); angle > pitch_min_deg: pitch (rotation about the width axis, 숙임); otherwise mixed (혼합). Frames with theta2d below class_min_theta_deg are not classified."""
    M, _ = ferr_affine(T, P)
    _, _, Vt = np.linalg.svd(M.T)
    v = Vt[1]
    c = float(np.degrees(np.arccos(np.clip(abs(float(v @ ferr_long_axis(T, ids))), 0.0, 1.0))))
    ang = 90.0 - c
    if not np.isfinite(theta_deg) or theta_deg < cfg["class_min_theta_deg"]:
        kind = "미분류"
    elif ang < cfg["roll_max_deg"]:
        kind = "굴림"
    elif ang > cfg["pitch_min_deg"]:
        kind = "숙임"
    else:
        kind = "혼합"
    return {"axis_angle_deg": ang, "kind": kind}


def _ferr_bones(ids):
    pos = {j: k for k, j in enumerate(ids)}
    out = []
    for a in FERR_FINGERS.values():
        for j in (a, a + 1, a + 2):
            if j in pos and j + 1 in pos:
                out.append((pos[j], pos[j + 1]))
    for a, b in ((5, 9), (9, 13), (13, 17)):
        if a in pos and b in pos:
            out.append((pos[a], pos[b]))
    return out


def _ferr_angle(u, v) -> float:
    c = float(u @ v / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def ferr_shape(T: np.ndarray, ids, P: np.ndarray, wrist_xy=None) -> dict:
    """Hand-shape indicators of one frame after the best-fit planar tilt (affine template -> frame, fitted on the MCP row and the middle and ring fingers so the index / pinky
    joints that the spread measures cannot be absorbed by the fit) is undone: points Q = (P - t) M^-1 are in the template's coordinates.
    bone_ratio: median over finger segments and MCP-row links of |length in Q / template length - 1| (fraction). spread_deg: angle between the index (MCP -> tip) and the pinky
    direction in Q. wrist_deg: angle between wrist -> middle MCP and the perpendicular of the MCP row in Q (0 = straight; needs the wrist point). Returns NaN where not computable."""
    pos = {j: k for k, j in enumerate(ids)}
    core = [pos[j] for j in (5, 9, 13, 17, 10, 11, 12, 14, 15, 16) if j in pos]            # fit on the MCP row, middle and ring fingers: the index / pinky joints are measured, not fitted
    M, t = ferr_affine(T[core], P[core]) if len(core) >= 6 else ferr_affine(T, P)
    Minv = np.linalg.inv(M)
    Q = (P - t) @ Minv
    out = {"bone_ratio": float("nan"), "spread_deg": float("nan"), "wrist_deg": float("nan")}
    r = [abs(np.linalg.norm(Q[a] - Q[b]) / max(np.linalg.norm(T[a] - T[b]), 1e-9) - 1.0) for a, b in _ferr_bones(ids)]
    if r:
        out["bone_ratio"] = float(np.median(r))
    if all(j in pos for j in (5, 8, 17, 20)):
        out["spread_deg"] = _ferr_angle(Q[pos[8]] - Q[pos[5]], Q[pos[20]] - Q[pos[17]])
    if wrist_xy is not None and np.isfinite(wrist_xy).all() and all(j in pos for j in (5, 9, 17)):
        w = (np.asarray(wrist_xy, float) - t) @ Minv
        row = Q[pos[17]] - Q[pos[5]]
        perp = np.array([-row[1], row[0]])
        ang = _ferr_angle(Q[pos[9]] - w, perp)
        out["wrist_deg"] = float(min(ang, 180.0 - ang))
    return out


def ferr_overlap(kp_frame, mask_frame, box_long: float) -> float:
    """Smallest distance between the PIP / DIP / tip joints of neighbouring fingers (index-middle, middle-ring, ring-pinky), as a fraction of the hand box's long side."""
    names = list(FERR_FINGERS.values())
    best = float("nan")
    for a, b in zip(names[:-1], names[1:]):
        for j in (a + 1, a + 2, a + 3):
            for k in (b + 1, b + 2, b + 3):
                if mask_frame[j] and mask_frame[k]:
                    d = float(np.linalg.norm(kp_frame[j] - kp_frame[k]) / max(box_long, 1e-9))
                    best = d if not np.isfinite(best) else min(best, d)
    return best


def ferr_problem(a: dict, kp, mask, box, cfg_use: dict, size_wh) -> tuple:
    """The estimation problem analyze() solved (same template, frames, Huber widths, the front + tilt frames it used), rebuilt from its result. Returns (Problem, ids)."""
    W, H = size_wh
    ecfg, J = cfg_use["estimate"], cfg_use["joints"]
    ids = list(J["sets"][J["estimate_set"]])
    kp = undistort(np.asarray(kp, float), ecfg, W, H)
    bl = box_long(np.asarray(box, float))
    n = len(bl)
    huber = ecfg["huber_rel"] * np.where(np.isfinite(bl), bl, 0.0)
    cx, cy = principal_point(ecfg, W, H)
    frames = [(kp[i][ids], np.asarray(mask)[i][ids]) for i in range(n)]
    used = [i for i, r in enumerate(a["role"]) if r in ("front", "tilt")]
    return Problem(np.array(a["template"]), frames, cx, cy, huber, ecfg["min_points"]).subset(used), ids


def ferr_frame_curves(pb: Problem, grid: np.ndarray) -> np.ndarray:
    """(n_frames of pb.keep, len(grid)): every frame's own cost curve (cost of that frame alone at every f of the grid), rows in pb.keep order."""
    return np.array([[cost(pb.subset([i]), float(f)) for f in grid] for i in pb.keep])


def ferr_preferred(curve: np.ndarray, grid: np.ndarray, pb_i: Problem) -> dict:
    """Preferred f of one frame (grid minimum, refined by the parabola in log f like fixed_template; at_edge when the minimum is on the grid border) and the sharpness
    min(cost(1.15 f), cost(f / 1.15)) / cost(f)."""
    k = int(np.argmin(curve))
    lf = _parabola(np.log(grid), curve, k)
    f = float(np.exp(np.clip(lf, np.log(grid[0]), np.log(grid[-1]))))
    c0 = cost(pb_i, f)
    return {"f_pref": f, "at_edge": bool(k in (0, len(grid) - 1)), "sharpness": float(min(cost(pb_i, f * 1.15), cost(pb_i, f / 1.15)) / max(c0, 1e-12))}


def ferr_subset_ft(pb: Problem, front_ids, tilt_ids, grid: np.ndarray) -> dict:
    """fixed_template on the front-window frames plus the given tilt frames only (all tilt frames given = the full estimate)."""
    sub = pb.subset(sorted(set(front_ids) | set(tilt_ids)))
    sol = fixed_template(sub, grid)
    hi = grid[-1]
    return {"f": sol["f"], "at_grid_edge": sol["at_grid_edge"], "side": None if not sol["at_grid_edge"] else ("상한" if sol["f"] >= hi * (1 - 1e-6) else "하한"),
            "n_tilt": len([i for i in sub.keep if i in set(tilt_ids)])}


def ferr_bin_table(values, f_pref, at_edge, edges=None, labels=None) -> list:
    """Preferred f per feature bin: rows {label, n, n_edge, median_f (frames not on the grid border), n_with_f}. edges: increasing bin borders (len = bins + 1); labels:
    category labels instead of edges (values are then the labels)."""
    v = list(values)
    f = np.asarray(f_pref, float)
    e = np.asarray(at_edge, bool)
    rows = []
    cats = labels if labels is not None else [(edges[k], edges[k + 1]) for k in range(len(edges) - 1)]
    for cat in cats:
        if labels is not None:
            m = np.array([x == cat for x in v])
            name = str(cat)
        else:
            x = np.asarray(v, float)
            m = np.isfinite(x) & (x >= cat[0]) & (x < cat[1])
            name = f"{cat[0]:.3g}–{cat[1]:.3g}"
        ok = m & ~e
        rows.append({"label": name, "n": int(m.sum()), "n_edge": int((m & e).sum()), "n_with_f": int(ok.sum()), "median_f": float(np.median(f[ok])) if ok.any() else None})
    return rows


def ferr_cached_npz(frames: list, dset: dict, cache_dir) -> Path:
    """Path of the cached mediapipe detection of this frame list (same key as swap_mediapipe_hands). Raises when it is not cached: experiment 23 never detects."""
    settings = {k: dset[k] for k in ("weights", "num_hands", "min_hand_detection_confidence", "min_hand_presence_confidence", "box_margin", "remap")}
    out = Path(cache_dir) / f"mediapipe_hands_{frames_key(frames, settings, 'mediapipe_hands')}.npz"
    if not out.exists():
        raise FileNotFoundError(f"cached detection missing: {out}")
    return out


def ferr_video(args) -> dict:
    """One video: the candidate (mediapipe_hands, wrist_mode exclude, fixed_template) analysis from cached detections, the video-level factors, the frame-level table
    (preferred f of every tilt frame, features, pull) and the subset estimates. args = (video, src_dir | None, input_dir, pcfg, c, cache_dir, tmp_root, ref22 | None, ref_f)."""
    video, src_dir, input_dir, pcfg, c, cache_dir, tmp_root, ref22, ref_f = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cv2.setNumThreads(1)
    fc = c["factors"]
    cand = c["candidate"]
    dset = c["detector_settings"][cand["detector"]]
    pc = copy.deepcopy(pcfg)
    pc["tracking"]["conf_min"] = float(dset["conf_min"])
    res = {"video": video, "detector": cand["detector"]}
    tmp = None
    try:
        if src_dir is not None:
            r17 = json.loads((Path(src_dir) / cand["detector"] / "results.json").read_text())
            times = np.array([x["time_s"] for x in r17["frames"]["list"]])
            W, H = r17["normalization"]["output"]["display_w"], r17["normalization"]["output"]["display_h"]
            dets = load_detections(Path(src_dir) / cand["detector"] / "detections.npz")
            res["detections"] = "17 run"
        else:
            tmp = Path(tempfile.mkdtemp(prefix="hf23_", dir=tmp_root))
            prep = fresp_prepare((video, input_dir, pcfg, {k: v for k, v in c.items() if k != "a2"}, str(tmp)))
            frames = prep["frames"]
            times = np.array([r["time_s"] for r in frames])
            W, H = prep["size"]
            dets = load_detections(ferr_cached_npz(frames, dset, cache_dir))
            check_alignment(frames, dets)
            res["detections"] = "22 run cache (found by the frame hashes)"
        tr = track(dets, times, (W, H), pc["tracking"], front_start_s=float(pc["phases"]["front_skip_s"]))
        seed = int(pc["experiment"]["seed"])
        cfg_ex = pseg_wrist_cfg(pc) if cand["wrist_mode"] == "exclude" else pc
        kp, mask, present, box = tr["kp"], tr["mask"], tr["present"], tr["box"]
        a = analyze(kp, mask, present, box, tr["cls"], times, (W, H), cfg_ex, tilt_start=None, methods_full=("fixed_template",), seed=seed)
        if "per_method" not in a:
            res["failed"] = a.get("failed", "분석 실패")
            return res
        sol, pm = a["solutions"]["fixed_template"], a["per_method"]["fixed_template"]
        f_est = float(sol["f"])
        hi = a["grid"][-1]
        res["ft"] = {"f": f_est, "at_grid_edge": bool(sol["at_grid_edge"]), "side": None if not sol["at_grid_edge"] else ("상한" if f_est >= hi * (1 - 1e-6) else "하한"), "verdict": pm["verdict"]["verdict"]}
        if ref22 is not None:
            s22 = ref22["original"][cand["wrist_mode"]]["solutions"]["fixed_template"]
            res["same_as_22"] = bool(s22["f"] == f_est and ref22["original"][cand["wrist_mode"]]["verdict"] == pm["verdict"]["verdict"])
        pb, ids = ferr_problem(a, kp, mask, box, cfg_ex, (W, H))
        T = np.array(a["template"])
        grid = np.array(a["grid"])
        bl = box_long(box)
        short = float(min(W, H))
        half_diag = float(np.hypot(W, H) / 2.0)
        res["cost_check"] = {"model_cost": float(sol["cost"]), "rebuilt_cost": float(cost(pb, f_est))}
        # ---- frame-level curves
        keep = list(pb.keep)
        curves = ferr_frame_curves(pb, grid)
        prof = np.array(sol["profile_cost"])
        res["curve_mean_check"] = {"max_rel_diff": float(np.max(np.abs(curves.mean(0) - prof) / np.maximum(np.abs(prof), 1e-12))), "n_frames": len(keep)}
        role = a["role"]
        tilt_ids = [i for i in keep if role[i] == "tilt"]
        front_ids = [i for i in keep if role[i] == "front"]
        theta = np.array([np.nan if x is None else x for x in a["theta2d_deg"]], float)
        axis = np.array([np.nan if x is None else x for x in a["axis2d_deg"]], float)
        d_series, v_series = pseg_series(kp, mask, present, box, times, T, ids)
        rms = dict(zip(pm["diagnostics"]["frames_used"], pm["diagnostics"]["frame_rms_px"]))

        def shape_of(i):
            vis = mask[i][ids]
            P = kp[i][ids]
            if vis.sum() < len(ids):
                return {"bone_ratio": float("nan"), "spread_deg": float("nan"), "wrist_deg": float("nan")}
            return ferr_shape(T, ids, P, kp[i][0] if mask[i][0] else None)
        front_shape = [shape_of(i) for i in front_ids]
        ref_spread = float(np.nanmedian([s["spread_deg"] for s in front_shape])) if front_shape else float("nan")
        ref_wrist = float(np.nanmedian([s["wrist_deg"] for s in front_shape])) if front_shape else float("nan")
        res["front_shape"] = {"bone_ratio": float(np.nanmedian([s["bone_ratio"] for s in front_shape])) if front_shape else None, "spread_deg": ref_spread, "wrist_deg": ref_wrist}
        rows = []
        pos_in_keep = {i: k for k, i in enumerate(keep)}
        for i in tilt_ids:
            sh = shape_of(i)
            cl = ferr_tilt_class(T, kp[i][ids], theta[i], fc, ids)
            pr = ferr_preferred(curves[pos_in_keep[i]], grid, pb.subset([i]))
            cx_, cy_ = (box[i][0] + box[i][2]) / 2.0, (box[i][1] + box[i][3]) / 2.0
            d_ref = cost(pb.subset([i]), float(ref_f)) - cost(pb.subset([i]), f_est)
            rows.append({"index": int(i), "time_s": float(times[i]), "theta2d_deg": float(theta[i]), "kind": cl["kind"], "axis_angle_deg": cl["axis_angle_deg"], "direction_deg": float(axis[i]),
                         "resid_rel_pct": float(100.0 * rms[i] / bl[i]) if i in rms and bl[i] > 0 else float("nan"), "bone_ratio_pct": 100.0 * sh["bone_ratio"],
                         "spread_change_deg": abs(sh["spread_deg"] - ref_spread), "wrist_change_deg": abs(sh["wrist_deg"] - ref_wrist),
                         "overlap_rel": ferr_overlap(kp[i], mask[i], bl[i]), "speed": float(v_series[i]), "center_offset_pct": float(100.0 * np.hypot(cx_ - W / 2.0, cy_ - H / 2.0) / half_diag),
                         "f_pref": pr["f_pref"], "pref_at_edge": pr["at_edge"], "sharpness": pr["sharpness"], "pull": float(d_ref)})
        res["frames"] = rows
        res["pull_front_total"] = float(sum(cost(pb.subset([i]), float(ref_f)) - cost(pb.subset([i]), f_est) for i in front_ids))
        res["n_front_frames"] = len(front_ids)
        # ---- video-level factors
        th_t = np.array([r["theta2d_deg"] for r in rows])
        kinds = [r["kind"] for r in rows]
        classified = [k for k in kinds if k != "미분류"]
        fw = [int(i) for i in a["front_window"]["idx"]]
        dfw = d_series[fw]
        jd = a["solutions"]["joint_delta"]
        nm = lambda key: float(np.nanmedian([r[key] for r in rows])) if rows else None      # noqa: E731
        dev = {}
        for other in (c.get("compare_detectors") or []):
            p_ = Path(src_dir) / other / "detections.npz" if src_dir is not None else None
            if p_ is None or not p_.exists():
                dev[other] = None
                continue
            tc = dict(pc["tracking"], conf_min=float(c["detector_settings"][other]["conf_min"]))
            to = track(load_detections(p_), times, (W, H), tc, front_start_s=float(pc["phases"]["front_skip_s"]))
            ds = []
            for i in tilt_ids:
                m_ = mask[i][ids] & to["mask"][i][ids]
                if present[i] and to["present"][i] and m_.sum() >= 4:
                    ds.append(float(np.linalg.norm(kp[i][ids][m_] - to["kp"][i][ids][m_], axis=-1).mean() / bl[i] * 100.0))
            dev[other] = {"median_pct": float(np.median(ds)) if ds else None, "n_frames": len(ds)}
        sig = grev_f_signal(a, kp, mask, ids, f_est)
        res["factors"] = {
            "jd_delta_deg": float(jd["delta_deg"]), "jd_phi_deg": float(jd["phi_deg"]), "d0": float(np.nanpercentile(dfw, 95)) if np.isfinite(dfw).any() else None,
            "front_jitter_rel_pct": 100.0 * float(a["front_jitter_rel"]), "finger_ext_min": float(a["finger_extension"]["min"]), "finger_ext_mean": float(a["finger_extension"]["mean"]),
            "n_tilt_frames": int(a["n_tilt_used"]), "theta_median_deg": float(np.median(th_t)) if len(th_t) else None, "theta_p90_deg": float(np.percentile(th_t, 90)) if len(th_t) else None,
            "theta_over_45_frac": float((th_t > 45.0).mean()) if len(th_t) else None,
            "frac_pitch": float(np.mean([k == "숙임" for k in classified])) if classified else None, "frac_roll": float(np.mean([k == "굴림" for k in classified])) if classified else None,
            "frac_mixed": float(np.mean([k == "혼합" for k in classified])) if classified else None, "frac_unclassified": float(np.mean([k == "미분류" for k in kinds])) if kinds else None,
            "axis_bins": int(a["axis_coverage"]["bins"]), "hand_box_short_ratio_front": float(np.nanmedian(bl[fw]) / short), "center_offset_pct": nm("center_offset_pct"),
            "edge_rate": float(a["edge_frame_rate"]), "bone_ratio_pct": nm("bone_ratio_pct"), "spread_change_deg": nm("spread_change_deg"), "wrist_change_deg": nm("wrist_change_deg"),
            "reproj_rms_rel_pct": 100.0 * float(pm["diagnostics"]["reproj_rms_rel"]), "detector_diff_pct": dev, "sharpness": float(pm["diagnostics"]["sharpness"]),
            "bootstrap_rel_std": None if pm["bootstrap"]["rel_std"] is None or not np.isfinite(pm["bootstrap"]["rel_std"]) else float(pm["bootstrap"]["rel_std"]),
            "f_rel_uncertainty": None if pm["f_rel_uncertainty"] is None or not np.isfinite(pm["f_rel_uncertainty"]) else float(pm["f_rel_uncertainty"]),
            "f_signal_ratio": sig["ratio_median"], "overlap_rel": nm("overlap_rel"), "speed": nm("speed")}
        # ---- subsets fixed in the yaml
        sub = {}
        nmin = int(fc["min_subset_frames"])
        resid = np.array([r["resid_rel_pct"] for r in rows])
        shp = np.array([r["bone_ratio_pct"] for r in rows])
        for s in c["subsets"]:
            kind = s["kind"]
            if kind == "pitch":
                sel = [r["index"] for r in rows if r["kind"] == "숙임"]
            elif kind == "roll":
                sel = [r["index"] for r in rows if r["kind"] == "굴림"]
            elif kind == "theta_range":
                lo, hi_ = s["range"]
                sel = [r["index"] for r in rows if lo <= r["theta2d_deg"] <= hi_]
            elif kind == "residual_low_half":
                sel = [rows[k]["index"] for k in np.argsort(resid, kind="stable")[:len(rows) // 2 + len(rows) % 2]] if rows else []
            elif kind == "shape_low_half":
                sel = [rows[k]["index"] for k in np.argsort(shp, kind="stable")[:len(rows) // 2 + len(rows) % 2]] if rows else []
            else:
                raise ValueError(kind)
            if len(sel) < nmin:
                sub[s["name"]] = {"skipped": f"기울기 프레임 {len(sel)}장 < {nmin}장", "n_tilt": len(sel)}
            else:
                sub[s["name"]] = ferr_subset_ft(pb, front_ids, sel, grid)
        res["subsets"] = sub
        res["subset_all"] = ferr_subset_ft(pb, front_ids, tilt_ids, grid)                  # control: every tilt frame = the full estimate
        return res
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


# ====================================================================================================
# long  (experiment 25: the candidate on long new videos; new section, earlier functions are used unchanged)
# Candidate = mediapipe_hands + wrist_mode exclude + fixed_template (22). New videos only enter after b (the common bias of the baseline videos) is fixed.
# Pooled estimate: the per-video fixed_template cost curves (every video with its own template, same f grid) are summed and minimised like fixed_template.
# H1 noise falls with more data, H2 a common bias exists, H3 b transfers to the new videos, H4 dropping small-tilt frames helps (rules and thresholds in the yaml).
# ====================================================================================================

def long_find_new_videos(input_dir, after_number: int, extensions=(".mov", ".mp4", ".m4v", ".mkv", ".avi")) -> list:
    """Hand videos IMG_<number> in input_dir with a number above after_number: [{video, number, duration_s, width, height}] (technical stream fields only)."""
    rows = []
    for p in sorted(Path(input_dir).expanduser().iterdir()):
        m = re.fullmatch(r"IMG_(\d+)", p.stem)
        if m and p.suffix.lower() in set(extensions) and int(m.group(1)) > int(after_number):
            i = probe(p)
            rows.append({"video": p.name, "number": int(m.group(1)), "duration_s": float(i["duration_s"]), "width": int(i["display_w"]), "height": int(i["display_h"])})
    return rows


def long_pooled_curve(curves) -> np.ndarray:
    """Sum of the per-video fixed_template cost curves (rows = videos, columns = the common f grid)."""
    return np.asarray(curves, float).sum(axis=0)


def long_curve_min(curve, grid) -> dict:
    """f of a cost curve like fixed_template: grid minimum refined by the parabola in log f, clipped to the grid; at_grid_edge when the minimum is on the border."""
    grid = np.asarray(grid, float)
    c = np.asarray(curve, float)
    k = int(np.argmin(c))
    f = float(np.exp(np.clip(_parabola(np.log(grid), c, k), np.log(grid[0]), np.log(grid[-1]))))
    return {"f": f, "at_grid_edge": bool(k in (0, len(grid) - 1)), "side": None if k not in (0, len(grid) - 1) else ("상한" if k == len(grid) - 1 else "하한")}


def long_pooled_estimate(curves, grid) -> dict:
    return long_curve_min(long_pooled_curve(curves), grid)


def long_pooled_bootstrap(curves, grid, n: int, seed: int) -> dict:
    """Video-level bootstrap of the pooled estimate: videos are drawn with replacement n times (fixed seed). Returns the pooled f of every draw, the number of draws on the grid edge."""
    curves = np.asarray(curves, float)
    rng = np.random.default_rng(int(seed))
    fs, edge = [], 0
    for _ in range(int(n)):
        pick = rng.integers(0, len(curves), len(curves))
        r = long_pooled_estimate(curves[pick], grid)
        fs.append(r["f"])
        edge += int(r["at_grid_edge"])
    return {"f": fs, "n_edge": edge, "n": int(n), "seed": int(seed)}


def long_log_ci(fs, ref: float, level: float = 0.95) -> tuple:
    """Percentile interval of log(f / ref) over bootstrap draws."""
    x = np.log(np.asarray(fs, float) / float(ref))
    a = (1.0 - level) / 2.0
    return float(np.quantile(x, a)), float(np.quantile(x, 1.0 - a))


def long_boot_log_std(fs) -> float:
    """Standard deviation of log f over the block-bootstrap estimates of one video (NaN with fewer than 2)."""
    x = np.log(np.asarray(fs, float))
    return float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")


def long_theta_keep(theta, tilt_ids, theta_min_deg: float) -> list:
    """Tilt frames whose theta2d is at least theta_min_deg (frames without theta2d are dropped)."""
    th = np.asarray(theta, float)
    return [int(i) for i in tilt_ids if np.isfinite(th[i]) and th[i] >= float(theta_min_deg)]


def long_h1(new_stds, base_stds, ratio: float) -> dict:
    """H1: the median per-video bootstrap log SD of the new videos <= ratio * the baseline median -> supported."""
    ns = [x for x in new_stds if np.isfinite(x)]
    bs = [x for x in base_stds if np.isfinite(x)]
    if not ns or not bs:
        return {"verdict": "판단 불가", "new_median": None, "base_median": None, "ratio_used": ratio}
    nm, bm = float(np.median(ns)), float(np.median(bs))
    return {"verdict": "지지" if nm <= ratio * bm else "근거 없음", "new_median": nm, "base_median": bm, "ratio_used": ratio, "ratio_observed": nm / bm}


def long_h2(ci: tuple, n_new: int, min_videos: int) -> dict:
    """H2: the 95% interval of log(f_pooled / ref) excludes 0 -> common bias; fewer than min_videos new videos -> not judged (values only)."""
    lo, hi = ci
    if n_new < min_videos:
        return {"verdict": "판단 불가", "reason": f"새 영상 {n_new}개 < {min_videos}개", "ci": [lo, hi]}
    return {"verdict": "공통 편향 있음" if (lo > 0 or hi < 0) else "확인 안 됨", "ci": [lo, hi], "sign": None if not (lo > 0 or hi < 0) else ("+" if lo > 0 else "−")}


def long_h3(f_new, ref: float, b: float, n_new: int, min_videos: int, fraction: float) -> dict:
    """H3: with the correction f / exp(b), the median |error| falls and at least `fraction` of the new videos have a smaller |error| -> effective."""
    f = np.asarray(f_new, float)
    before = np.abs(f / ref - 1.0)
    after = np.abs(f / np.exp(b) / ref - 1.0)
    frac = float(np.mean(after < before)) if len(f) else float("nan")
    mb, ma = float(np.median(before)) if len(f) else float("nan"), float(np.median(after)) if len(f) else float("nan")
    out = {"median_abs_before": mb, "median_abs_after": ma, "fraction_reduced": frac, "n": int(len(f)), "b": float(b)}
    if n_new < min_videos:
        return {**out, "verdict": "판단 불가", "reason": f"새 영상 {n_new}개 < {min_videos}개"}
    return {**out, "verdict": "보정 효과 있음" if (ma < mb and frac >= fraction - 1e-9) else "보정 효과 없음"}


def long_h4(f_orig, f_h4, ref: float, gain: float, fraction: float) -> dict:
    """H4: over the videos with a theta2d >= limit estimate, the median |error| is at least `gain` smaller than the original FT's and at least `fraction` of them improve."""
    fo, fh = np.asarray(f_orig, float), np.asarray(f_h4, float)
    if not len(fo):
        return {"verdict": "판단 불가", "n": 0}
    eo, eh = np.abs(fo / ref - 1.0), np.abs(fh / ref - 1.0)
    mo, mh, frac = float(np.median(eo)), float(np.median(eh)), float(np.mean(eh < eo))
    return {"verdict": "개선" if (mo - mh >= gain - 1e-12 and frac >= fraction - 1e-9) else "근거 없음", "median_abs_orig": mo, "median_abs_h4": mh, "fraction_improved": frac, "n": int(len(fo))}


def long_calibration(f, sigma_log, ref: float, b: float = 0.0) -> dict:
    """Share of videos whose |log(f / ref) - b| lies within 1 and 2 bootstrap log SDs."""
    e = np.abs(np.log(np.asarray(f, float) / ref) - b)
    s = np.asarray(sigma_log, float)
    ok = np.isfinite(s) & (s > 0)
    return {"within_1sigma": float(np.mean(e[ok] <= s[ok])) if ok.any() else None, "within_2sigma": float(np.mean(e[ok] <= 2 * s[ok])) if ok.any() else None, "n": int(ok.sum())}


def long_video(args) -> dict:
    """One video of experiment 25: the candidate (exclude, fixed_template) from mediapipe detections with its cost curve, bootstrap, motion statistics and the theta2d >= limit variant (H4);
    for new videos also the exclusion check, 21's guide and transition, and the comparison estimates (keep, joint_delta, AnyCalib). Baseline videos are read from their 17 runs.
    args = (video, kind 'baseline' | 'new', src_dir | None, prep | None, npz | None, pcfg, c, lock_dir, gpu_id | None)."""
    video, kind, src_dir, prep, npz, pcfg, c, lock_dir, gpu_id = args
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cv2.setNumThreads(1)
    cand, hy, ex = c["candidate"], c["hypotheses"], c["exclusion"]
    dset = c["detector_settings"][cand["detector"]]
    pc = copy.deepcopy(pcfg)
    pc["tracking"]["conf_min"] = float(dset["conf_min"])
    if kind == "baseline":
        r17 = json.loads((Path(src_dir) / cand["detector"] / "results.json").read_text())
        times = np.array([x["time_s"] for x in r17["frames"]["list"]])
        W, H = r17["normalization"]["output"]["display_w"], r17["normalization"]["output"]["display_h"]
        dets = load_detections(Path(src_dir) / cand["detector"] / "detections.npz")
        frames = None
    else:
        frames = prep["frames"]
        times = np.array([r["time_s"] for r in frames])
        W, H = prep["size"]
        dets = load_detections(Path(npz))
        check_alignment(frames, dets)
    tr = track(dets, times, (W, H), pc["tracking"], front_start_s=float(pc["phases"]["front_skip_s"]))
    kp, mask, present, box = tr["kp"], tr["mask"], tr["present"], tr["box"]
    res = {"video": video, "kind": kind, "n_frames": int(len(times)), "duration_s": float(times[-1] - times[0]) if len(times) else 0.0, "size": [W, H]}
    fw = front_window(times, present, kp, mask, box, pc["phases"])
    res["detect_rate"] = float(present.mean())
    res["edge_rate"] = fresp_edge_rate(box, present, W, H)
    res["front_window"] = [fw.get("t0"), fw.get("t1")]
    reasons = []
    if len(fw["idx"]) == 0 or fw["detect_rate"] < float(pc["phases"]["front_min_detect_rate"]):
        reasons.append("유효한 정면 창(1초)이 없음")
    if res["edge_rate"] > float(ex["edge_rate_max"]):
        reasons.append(f"가장자리 접촉 프레임 {100 * res['edge_rate']:.1f}% > {100 * float(ex['edge_rate_max']):.0f}%")
    if 1.0 - res["detect_rate"] > float(ex["detect_fail_max"]):
        reasons.append(f"손 검출 실패 프레임 {100 * (1.0 - res['detect_rate']):.1f}% > {100 * float(ex['detect_fail_max']):.0f}%")
    res["excluded"] = reasons
    if kind == "new" and reasons:
        return res
    seed = int(pc["experiment"]["seed"])
    cfg_ex = pseg_wrist_cfg(pc) if cand["wrist_mode"] == "exclude" else pc
    a = analyze(kp, mask, present, box, tr["cls"], times, (W, H), cfg_ex, tilt_start=None, methods_full=("fixed_template",), seed=seed)
    if "per_method" not in a:
        res["failed"] = a.get("failed", "분석 실패")
        return res
    sol, jd, pm = a["solutions"]["fixed_template"], a["solutions"]["joint_delta"], a["per_method"]["fixed_template"]
    grid = np.array(a["grid"])
    hi = grid[-1]
    side = lambda s: None if not s["at_grid_edge"] else ("상한" if s["f"] >= hi * (1 - 1e-6) else "하한")      # noqa: E731
    bf = pm["bootstrap"]["f"]
    res["ft"] = {"f": float(sol["f"]), "at_grid_edge": bool(sol["at_grid_edge"]), "side": side(sol), "verdict": pm["verdict"]["verdict"], "sharpness": float(pm["diagnostics"]["sharpness"]),
                 "f_rel_uncertainty": float(pm["f_rel_uncertainty"]) if np.isfinite(pm["f_rel_uncertainty"]) else None, "boot_rel_std": float(pm["bootstrap"]["rel_std"]) if np.isfinite(pm["bootstrap"]["rel_std"]) else None,
                 "boot_log_std": long_boot_log_std(bf), "boot_f": [float(x) for x in bf], "n_boot": int(pm["bootstrap"]["n_done"])}
    res["jd"] = {"f": float(jd["f"]), "at_grid_edge": bool(jd["at_grid_edge"]), "side": side(jd)}
    res["curve"] = {"grid": grid.tolist(), "cost": [float(x) for x in sol["profile_cost"]]}
    theta = np.array([np.nan if x is None else x for x in a["theta2d_deg"]], float)
    role = a["role"]
    pb, ids = ferr_problem(a, kp, mask, box, cfg_ex, (W, H))
    front_ids = [i for i in pb.keep if role[i] == "front"]
    tilt_ids = [i for i in pb.keep if role[i] == "tilt"]
    th = theta[tilt_ids]
    res["motion"] = {"tilt_start_s": float(a["tilt_start_s"]), "tilt_last_s": float(times[tilt_ids].max()) if tilt_ids else None, "n_tilt_frames": len(tilt_ids),
                     "tilt_duration_s": float(times[tilt_ids].max() - a["tilt_start_s"]) if tilt_ids else 0.0, "theta_median_deg": float(np.nanmedian(th)) if len(th) else None,
                     "theta_p90_deg": float(np.nanpercentile(th, 90)) if len(th) else None, "theta_over_45_frac": float(np.nanmean(th > 45.0)) if len(th) else None,
                     "theta_max_deg": float(np.nanmax(th)) if len(th) else None, "theta_tilt": [round(float(x), 1) for x in th if np.isfinite(x)]}
    sel = long_theta_keep(theta, tilt_ids, float(hy["h4_theta_min_deg"]))
    if len(sel) >= int(hy["h4_min_frames"]):
        s4 = ferr_subset_ft(pb, front_ids, sel, grid)
        res["h4"] = {"theta_min_deg": float(hy["h4_theta_min_deg"]), "n_tilt": len(sel), "f": s4["f"], "at_grid_edge": s4["at_grid_edge"], "side": s4["side"]}
    else:
        res["h4"] = {"theta_min_deg": float(hy["h4_theta_min_deg"]), "n_tilt": len(sel), "skipped": f"기울기 프레임 {len(sel)}장 < {int(hy['h4_min_frames'])}장"}
    if kind == "baseline":
        return res
    # ---- new videos: per-frame table, 21's guide and transition, comparison estimates
    keepset = set(sel)
    res["frames"] = [{"index": int(i), "time_s": float(times[i]), "theta2d_deg": float(theta[i]), "in_h4_subset": bool(i in keepset)} for i in tilt_ids]
    if c.get("compare_estimates"):
        c21 = {"phase_segmentation": c["guide"]["phase_segmentation"], "conditions": c["guide"]["conditions"], "f_estimation": {"compare_wrist_modes": ["keep"]}, "transition_method": "linear_d"}
        g = grev_analyze(kp, mask, present, box, tr["cls"], times, (W, H), pc, c21, seed)
        if "transition" in g:
            tr21 = g["transition"]
            res["guide"] = {"t_transition": tr21.get("t_transition"), "guide": bool(tr21["guide"]), "reasons": tr21["reasons"], "items": {k: bool(v) for k, v in tr21["items"].items()},
                            "d0": tr21.get("d0"), "d_thr": tr21.get("d_thr"), "d0_capped": tr21.get("d0_capped"), "front_window": tr21.get("front_window")}
            kw = g["wrist"]["keep"]
            res["keep"] = {"ft": kw["solutions"]["fixed_template"], "jd": kw["solutions"]["joint_delta"], "verdict": kw["verdict"]} if "solutions" in kw else {"failed": kw.get("failed")}
        else:
            res["guide"] = {"failed": g.get("failed")}
        acfg = copy.deepcopy(pc["anycalib"])
        if pc["anycalib"]["enabled"] and frames is not None:
            with fresp_gpu_lock(Path(lock_dir) / f"gpu_{'' if gpu_id is None else gpu_id}.lock"):
                fresp_thermal_wait(float(c["anycalib_guard"]["gpu_pause_c"]), float(c["anycalib_guard"]["gpu_resume_c"]))
                ac = anycalib_run(frames, acfg, float(pc["phases"]["front_phase_s"]), lambda m="": None)
            res["anycalib"] = {m: {"median": v["summary"].get("median"), "n": v["summary"].get("n")} for m, v in ac["models"].items()}
    return res


# ====================================================================================================
# gen  (experiment 26: new judgment rule v2, generalisation groups, hand-trajectory effect of the f error; new section, earlier functions are used unchanged)
# Videos carry group (G1 other person, G2 other phone, G3 other lens, G4 gloves, G5 calibration + work), person_id, phone, lens, glove and ref_f. Candidate = 22 / 25's.
# Rating v2 (per video): from the tilt length, the largest theta2d and the bootstrap log SD, not from the sharpness. Verification of v2 and the group verdicts: see gen_verify / gen_group_verdict.
# Trajectory: from the 2D hand centre and the hand size ratio, X = (u - cx) L / l, Y = (v - cy) L / l, Z = f L / l; changing f only scales Z.
# ====================================================================================================

GEN_GROUPS = ("G1", "G2", "G3", "G4", "G5")
GEN_REQUIRED = ("file", "group", "person_id", "phone", "lens", "glove", "ref_f")


def gen_check_videos(entries) -> list:
    """Validates the videos list of the yaml (every entry needs file, group, person_id, phone, lens, glove, ref_f). Stops (SystemExit) with the reason otherwise."""
    if not entries:
        raise SystemExit("videos is empty: give the video entries in the yaml (file, group, person_id, phone, lens, glove, ref_f)")
    for i, e in enumerate(entries):
        miss = [k for k in GEN_REQUIRED if e.get(k) is None]
        if miss:
            raise SystemExit(f"videos[{i}] ({e.get('file')}): missing {miss}")
        if e["group"] not in GEN_GROUPS:
            raise SystemExit(f"videos[{i}] ({e['file']}): group must be one of {GEN_GROUPS}")
        if float(e["ref_f"]) <= 0:
            raise SystemExit(f"videos[{i}] ({e['file']}): ref_f must be positive")
    return list(entries)


def gen_rating(tilt_s, theta_max_deg, boot_log_std, v2: dict) -> str:
    """Rating v2 of one video. '사용 가능': tilt >= tilt_min_s, largest theta2d >= theta_max_min_deg and bootstrap log SD <= usable_log_std_max.
    '참고': both shooting conditions hold and the log SD is <= reference_log_std_max. Otherwise '사용 불가'."""
    if tilt_s is None or theta_max_deg is None or boot_log_std is None or not np.isfinite(boot_log_std):
        return "사용 불가"
    shoot = bool(tilt_s >= float(v2["tilt_min_s"]) and theta_max_deg >= float(v2["theta_max_min_deg"]))
    if shoot and boot_log_std <= float(v2["usable_log_std_max"]):
        return "사용 가능"
    if shoot and boot_log_std <= float(v2["reference_log_std_max"]):
        return "참고"
    return "사용 불가"


def gen_verify(rows: list, ver: dict) -> dict:
    """Verification of rating v2 on videos (rows: dict with rating, f, ref_f, sigma): among the usable videos the share with |log(f / ref_f)| <= within_sigma * sigma must be >= within_share_min and
    their median |error| <= median_abs_err_max. Fewer than min_usable usable videos: not judged."""
    use = [r for r in rows if r["rating"] == "사용 가능" and r.get("f") is not None]
    out = {"n_usable": len(use), "n_videos": len(rows)}
    if use:
        e = np.array([abs(r["f"] / r["ref_f"] - 1.0) for r in use])
        w = np.array([abs(np.log(r["f"] / r["ref_f"])) <= float(ver["within_sigma"]) * r["sigma"] for r in use])
        out.update({"within_share": float(w.mean()), "median_abs_err": float(np.median(e))})
    if len(use) < int(ver["min_usable"]):
        return {**out, "verdict": "판단 불가", "reason": f"사용 가능 영상 {len(use)}개 < {int(ver['min_usable'])}개"}
    ok = out["within_share"] >= float(ver["within_share_min"]) and out["median_abs_err"] <= float(ver["median_abs_err_max"])
    return {**out, "verdict": "판정 v2 검증됨" if ok else "검증 안 됨"}


def gen_group_verdict(curves, grid, f_videos, ref_f: float, gcfg: dict, n_excluded: int = 0, n_total: int | None = None) -> dict:
    """Generalisation verdict of one group. curves = the group's per-video fixed_template cost curves; f_videos = their f (None = no value). Pooled f (curves summed) and a
    video-level bootstrap interval; pass: pooled f within +-pooled_tol of ref_f and at least video_fraction of the videos within +-video_tol; fewer than min_videos videos: not judged."""
    n = len(curves)
    out = {"n_videos": n, "n_excluded": int(n_excluded), "n_total": int(n_total if n_total is not None else n + n_excluded), "ref_f": float(ref_f)}
    if n == 0:
        return {**out, "verdict": "영상 없음"}
    est = long_pooled_estimate(curves, grid)
    b = long_pooled_bootstrap(curves, grid, int(gcfg["bootstrap_n"]), int(gcfg["bootstrap_seed"]))
    inside = [f is not None and abs(f / ref_f - 1.0) <= float(gcfg["video_tol"]) for f in f_videos]
    out.update({"pooled_f": est["f"], "at_grid_edge": est["at_grid_edge"], "side": est["side"], "interval_95": [float(np.quantile(b["f"], 0.025)), float(np.quantile(b["f"], 0.975))],
                "pooled_within_tol": bool(not est["at_grid_edge"] and abs(est["f"] / ref_f - 1.0) <= float(gcfg["pooled_tol"])), "video_share_within_tol": float(np.mean(inside))})
    if n < int(gcfg["min_videos"]):
        return {**out, "verdict": "판단 불가", "reason": f"영상 {n}개 < {int(gcfg['min_videos'])}개"}
    return {**out, "verdict": "통과" if (out["pooled_within_tol"] and out["video_share_within_tol"] >= float(gcfg["video_fraction"]) - 1e-9) else "실패"}


def gen_trajectory(u, v, ell, f: float, L_mm: float, W: float, H: float) -> np.ndarray:
    """Camera coordinates (mm), shape (n, 3): X = (u - cx) L / l, Y = (v - cy) L / l, Z = f L / l (pinhole, principal point at the image centre, l = apparent hand length in px).
    f enters only through Z."""
    u, v, ell = np.asarray(u, float), np.asarray(v, float), np.asarray(ell, float)
    return np.c_[(u - W / 2.0) * L_mm / ell, (v - H / 2.0) * L_mm / ell, f * L_mm / ell]


def gen_smooth_traj(times, P: np.ndarray, window_s: float) -> np.ndarray:
    """0.3 s centred running median of every coordinate (analysis only)."""
    return np.c_[[pseg_median_filter(times, P[:, k], window_s) for k in range(P.shape[1])]].T


def gen_step_vectors(times, P: np.ndarray, step_s: float, max_gap_s: float) -> tuple:
    """Movement vectors between points step_s apart on a regular time grid: (grid index pairs' start times, vectors (m, 3)). A grid point is used when a valid frame lies within max_gap_s."""
    t = np.asarray(times, float)
    ok = np.isfinite(P).all(axis=1)
    t, P = t[ok], P[ok]
    if len(t) < 2:
        return np.zeros(0), np.zeros((0, 3))
    grid = np.arange(t[0], t[-1] + 1e-9, step_s)
    keep = np.array([np.abs(t - g).min() <= max_gap_s for g in grid])
    Q = np.c_[[np.interp(grid, t, P[:, k]) for k in range(3)]].T
    idx = [i for i in range(len(grid) - 1) if keep[i] and keep[i + 1]]
    return grid[idx], np.array([Q[i + 1] - Q[i] for i in idx]).reshape(-1, 3)


def gen_step_metrics(times, P_ref: np.ndarray, P_cmp: np.ndarray, tcfg: dict) -> dict:
    """Errors of a comparison trajectory against the reference trajectory on step_s movement vectors: magnitude error (%), direction error (deg), depth-movement error (mm) with
    their median and 90th percentile, and the median share of the depth component in the reference movement. Steps with a reference movement below min_step_mm are not used."""
    _, dr = gen_step_vectors(times, P_ref, float(tcfg["step_s"]), float(tcfg["max_gap_s"]))
    _, dc = gen_step_vectors(times, P_cmp, float(tcfg["step_s"]), float(tcfg["max_gap_s"]))
    n = min(len(dr), len(dc))
    dr, dc = dr[:n], dc[:n]
    nr, nc = np.linalg.norm(dr, axis=1), np.linalg.norm(dc, axis=1)
    use = nr >= float(tcfg["min_step_mm"])
    out = {"n_steps": int(use.sum())}
    if not use.any():
        return {**out, "magnitude_err_median_pct": None, "magnitude_err_p90_pct": None, "direction_err_median_deg": None, "direction_err_p90_deg": None, "depth_err_median_mm": None, "depth_err_p90_mm": None,
               "depth_share_median": None}
    mag = 100.0 * np.abs(nc[use] / nr[use] - 1.0)
    cos = np.clip((dr[use] * dc[use]).sum(1) / np.maximum(nr[use] * nc[use], 1e-12), -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    dep = np.abs(dc[use, 2] - dr[use, 2])
    share = np.abs(dr[use, 2]) / nr[use]
    return {**out, "magnitude_err_median_pct": float(np.median(mag)), "magnitude_err_p90_pct": float(np.percentile(mag, 90)), "direction_err_median_deg": float(np.median(ang)),
            "direction_err_p90_deg": float(np.percentile(ang, 90)), "depth_err_median_mm": float(np.median(dep)), "depth_err_p90_mm": float(np.percentile(dep, 90)), "depth_share_median": float(np.median(share))}


def gen_present_until(times, present, calib_end_s) -> np.ndarray:
    """Frames the f estimate may use: detected frames up to calib_end_s (None = all). Later frames never enter the estimate (G5: they are the work segment)."""
    p = np.asarray(present, bool)
    return p if calib_end_s is None else (p & (np.asarray(times, float) <= float(calib_end_s)))


def gen_s1(T: np.ndarray, P: np.ndarray) -> float:
    """Larger singular value of the linear part of the least-squares affine map template T -> frame points P (the in-plane scale that tilt does not shrink)."""
    A = np.c_[T, np.ones(len(T))]
    M, *_ = np.linalg.lstsq(A, P, rcond=None)
    return float(np.linalg.svd(M[:2].T, compute_uv=False)[0])


def gen_guide(kp, mask, present, box, cls, times, size_wh, pcfg: dict, c: dict, seed: int) -> dict:
    """21's transition (linear_d) and shooting guide with the guide settings of this experiment (guide v2: hand box <= 0.65), the same steps as grev_analyze without the f estimates."""
    W, H = size_wh
    p, ccfg = c["guide"]["phase_segmentation"], c["guide"]["conditions"]
    est_ids = list(pcfg["joints"]["sets"][pcfg["joints"]["estimate_set"]])
    a0 = analyze(kp, mask, present, box, cls, times, size_wh, pcfg, tilt_start=None, methods_full=(), seed=seed)
    if "solutions" not in a0:
        return {"failed": a0.get("failed", "분석 실패")}
    theta = np.array([np.nan if x is None else x for x in a0["theta2d_deg"]], float)
    fw0 = a0["front_window"]
    fw = fw0 if not p.get("front_search_s") else grev_front_window(times, present, kp, mask, box, pcfg["phases"], p["front_search_s"])
    if len(fw["idx"]) == 0:
        return {"failed": "정면 창을 만들 수 없음"}
    tr_ = grev_transition(times, present, kp, mask, box, np.array(a0["template"]), est_ids, fw["idx"], p)
    m = grev_metrics(times, present, box, W, H, theta, a0["role"], tr_.get("t_transition"), a0.get("front_jitter_rel"), a0["axis_coverage"]["bins"], a0["finger_extension"]["min"],
                     float(ccfg.get("tilt_max_deg", 45.0)))
    g = grev_conditions_check(m, ccfg)
    return {"t_transition": tr_.get("t_transition"), "guide": bool(g["guide"]), "reasons": g["reasons"], "items": {k: bool(v) for k, v in g["items"].items()},
            "box_short_ratio": m.get("box_short_ratio"), "d0": tr_.get("d0"), "d_thr": tr_.get("d_thr")}


def gen_video(args) -> dict:
    """One video of experiment 26 -> f (candidate), bootstrap, motion statistics, guide v2 and the inputs of the trajectory analysis. args = (entry, prep, npz, pcfg, c, seed_unused).
    entry = the yaml video entry; G5 estimates f only from the frames up to calib_end_s and uses the later frames for the trajectory."""
    entry, prep, npz, pcfg, c = args[:5]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cv2.setNumThreads(1)
    cand, ex, tcfg = c["candidate"], c["exclusion"], c["trajectory"]
    dset = c["detector_settings"][cand["detector"]]
    pc = copy.deepcopy(pcfg)
    pc["tracking"]["conf_min"] = float(dset["conf_min"])
    frames = prep["frames"]
    times = np.array([r["time_s"] for r in frames])
    W, H = prep["size"]
    dets = load_detections(Path(npz))
    check_alignment(frames, dets)
    tr = track(dets, times, (W, H), pc["tracking"], front_start_s=float(pc["phases"]["front_skip_s"]))
    kp, mask, present, box = tr["kp"], tr["mask"], tr["present"], tr["box"]
    calib_end = None
    if entry["group"] == "G5":
        calib_end = float(entry.get("calib_end_s") or c.get("calib_end_default_s", 25.0))
    pres = gen_present_until(times, present, calib_end)
    cal = np.ones(len(times), bool) if calib_end is None else (times <= calib_end)
    res = {"video": entry["file"], "group": entry["group"], "person_id": entry["person_id"], "phone": entry["phone"], "lens": entry["lens"], "glove": bool(entry["glove"]), "ref_f": float(entry["ref_f"]),
           "calib_end_s": calib_end, "n_frames": int(len(times)), "duration_s": float(times[-1] - times[0]) if len(times) else 0.0, "size": [W, H],
           "detect_rate": float(present[cal].mean()), "edge_rate": fresp_edge_rate(box[cal], present[cal], W, H)}
    fw = front_window(times, pres, kp, mask, box, pc["phases"])
    res["front_window"] = [fw.get("t0"), fw.get("t1")]
    reasons = []
    if len(fw["idx"]) == 0 or fw["detect_rate"] < float(pc["phases"]["front_min_detect_rate"]):
        reasons.append("유효한 정면 창(1초)이 없음")
    if res["edge_rate"] > float(ex["edge_rate_max"]):
        reasons.append(f"가장자리 접촉 프레임 {100 * res['edge_rate']:.1f}% > {100 * float(ex['edge_rate_max']):.0f}%")
    if 1.0 - res["detect_rate"] > float(ex["detect_fail_max"]):
        reasons.append(f"손 검출 실패 프레임 {100 * (1.0 - res['detect_rate']):.1f}% > {100 * float(ex['detect_fail_max']):.0f}%")
    res["excluded"] = reasons
    if reasons:
        return res
    seed = int(pc["experiment"]["seed"])
    cfg_ex = pseg_wrist_cfg(pc) if cand["wrist_mode"] == "exclude" else pc
    a = analyze(kp, mask, pres, box, tr["cls"], times, (W, H), cfg_ex, tilt_start=None, methods_full=("fixed_template",), seed=seed)
    if "per_method" not in a:
        res["failed"] = a.get("failed", "분석 실패")
        return res
    sol, pm = a["solutions"]["fixed_template"], a["per_method"]["fixed_template"]
    grid = np.array(a["grid"])
    hi = grid[-1]
    bf = pm["bootstrap"]["f"]
    res["ft"] = {"f": float(sol["f"]), "at_grid_edge": bool(sol["at_grid_edge"]), "side": None if not sol["at_grid_edge"] else ("상한" if sol["f"] >= hi * (1 - 1e-6) else "하한"),
                 "verdict": pm["verdict"]["verdict"], "sharpness": float(pm["diagnostics"]["sharpness"]), "boot_log_std": long_boot_log_std(bf),
                 "boot_rel_std": float(pm["bootstrap"]["rel_std"]) if np.isfinite(pm["bootstrap"]["rel_std"]) else None, "n_boot": int(pm["bootstrap"]["n_done"])}
    res["curve"] = {"grid": grid.tolist(), "cost": [float(x) for x in sol["profile_cost"]]}
    theta = np.array([np.nan if x is None else x for x in a["theta2d_deg"]], float)
    role = a["role"]
    tilt = [i for i, r in enumerate(role) if r == "tilt"]
    th = theta[tilt]
    bl = box_long(box)
    fidx = [int(i) for i in a["front_window"]["idx"]]
    res["motion"] = {"tilt_start_s": float(a["tilt_start_s"]), "n_tilt_frames": len(tilt), "tilt_duration_s": float(times[tilt].max() - a["tilt_start_s"]) if tilt else 0.0,
                     "theta_median_deg": float(np.nanmedian(th)) if len(th) else None, "theta_p90_deg": float(np.nanpercentile(th, 90)) if len(th) else None,
                     "theta_max_deg": float(np.nanmax(th)) if len(th) else None, "theta_over_45_frac": float(np.nanmean(th > 45.0)) if len(th) else None}
    res["hand_box_short_ratio_front"] = float(np.nanmedian(bl[fidx]) / min(W, H))
    res["guide"] = gen_guide(kp, mask, pres, box, tr["cls"], times, (W, H), pc, c, seed)
    # ---- trajectory inputs: hand centre (wrist + 4 MCP), size ratio against the front window, hand length l = l_front * s1 / s1_front
    ids = list(cfg_ex["joints"]["sets"][cfg_ex["joints"]["estimate_set"]])
    T = np.array(a["template"])
    front_frames = [int(i) for i in a["template_frames"]]
    s1_front = float(np.median([gen_s1(T, kp[i][ids]) for i in front_frames]))
    lf = [float(np.linalg.norm(kp[i][12] - kp[i][0])) for i in front_frames if mask[i][0] and mask[i][12]]
    ell_front = float(np.median(lf)) if lf else float("nan")
    target = [i for i in range(len(times)) if present[i] and times[i] > calib_end] if calib_end is not None else tilt
    rows, centre = [], [0, 5, 9, 13, 17]
    for i in target:
        vis = mask[i][ids]
        if not (present[i] and vis.sum() >= 6 and mask[i][centre].all()):
            continue
        u, v = kp[i][centre].mean(0)
        s1 = gen_s1(T[vis], kp[i][ids][vis])
        rows.append({"index": int(i), "time_s": float(times[i]), "u": float(u), "v": float(v), "size_ratio": float(s1 / s1_front), "theta2d_deg": None if not np.isfinite(theta[i]) else float(theta[i])})
    res["traj"] = {"rows": rows, "ell_front_px": ell_front, "s1_front": s1_front, "segment": "work (after calib_end_s)" if calib_end is not None else "tilt phase"}
    return res


def gen_existing_rows(cfg: dict) -> list:
    """Existing videos for the descriptive comparison (never a verification): tilt length, largest theta2d, bootstrap log SD, f and its reference f, from the stored results of 22 / 23 / 25."""
    ex = cfg["existing"]
    rows = []
    base = json.loads(resolve(ex["baseline_json"]).read_text())["videos"]
    for j in base:
        rows.append({"video": j["video"], "source": "22 (1x)", "ref_f": float(ex["ref_f_1x"]), "f": j["ft"]["f"], "at_grid_edge": j["ft"]["at_grid_edge"], "sigma": j["ft"]["boot_log_std"],
                     "tilt_s": j["motion"]["tilt_duration_s"], "theta_max": j["motion"]["theta_max_deg"]})
    for j in json.loads(resolve(ex["results_25"]).read_text())["new"]:
        rows.append({"video": j["video"], "source": "25 (1x)", "ref_f": float(ex["ref_f_1x"]), "f": j["ft"]["f"], "at_grid_edge": j["ft"]["at_grid_edge"], "sigma": j["ft"]["boot_log_std"],
                     "tilt_s": j["motion"]["tilt_duration_s"], "theta_max": j["motion"]["theta_max_deg"]})
    j23 = {j["video"]: j for j in json.loads(resolve(ex["results_23"]).read_text())["jobs"]}.get(ex["lens_video"])
    j22 = {j["video"]: j for j in json.loads(resolve(ex["results_22"]).read_text())["jobs"]}.get(ex["lens_video"])
    if j23 and j22:
        s = j22["original"][ex["lens_mode"]]["solutions"]["fixed_template"]
        th = [r["theta2d_deg"] for r in j23["frames"]]
        tt = [r["time_s"] for r in j23["frames"]]
        rows.append({"video": ex["lens_video"], "source": "22 (0.5x)", "ref_f": float(ex["ref_f_lens"]), "f": s["f"], "at_grid_edge": s["at_grid_edge"], "sigma": j23["factors"]["bootstrap_rel_std"],
                     "tilt_s": float(max(tt) - 3.0) if tt else 0.0, "theta_max": float(max(th)) if th else None, "sigma_is_rel_std": True, "size": j22.get("size"),
                     "n_frames": j22.get("n_frames"), "duration_s": None if not j22.get("n_frames") else j22["n_frames"] / 15.0})
    return rows


# ====================================================================================================
# ext  (experiment 27: main-camera 1x extension, G6 distance and G7 left hand; new section, earlier functions are used unchanged)
# Videos carry group (G6 distance 50 / 60 / 75 cm, right hand; G7 left hand), distance_cm, hand and ref_f. Candidate and per-video processing are 26's (gen_video).
# G6 verdict: pooled f per distance (cost curves summed, video-level bootstrap interval), all within +-pooled_tol of ref_f and max / min <= ratio_max. G7: 26's group verdict.
# ====================================================================================================

EXT_GROUPS = ("G6", "G7")
EXT_REQUIRED = ("file", "group", "distance_cm", "hand", "ref_f")


def ext_check_videos(entries) -> list:
    """Validates the videos list of the yaml (every entry needs file, group G6 / G7, distance_cm, hand left / right, ref_f). Stops (SystemExit) with the reason otherwise."""
    if not entries:
        raise SystemExit("videos is empty: give the video entries in the yaml (file, group, distance_cm, hand, ref_f)")
    for i, e in enumerate(entries):
        miss = [k for k in EXT_REQUIRED if e.get(k) is None]
        if miss:
            raise SystemExit(f"videos[{i}] ({e.get('file')}): missing {miss}")
        if e["group"] not in EXT_GROUPS:
            raise SystemExit(f"videos[{i}] ({e['file']}): group must be one of {EXT_GROUPS}")
        if e["hand"] not in ("left", "right"):
            raise SystemExit(f"videos[{i}] ({e['file']}): hand must be left or right")
        if float(e["distance_cm"]) <= 0 or float(e["ref_f"]) <= 0:
            raise SystemExit(f"videos[{i}] ({e['file']}): distance_cm and ref_f must be positive")
    return list(entries)


def ext_video(args) -> dict:
    """One video of experiment 27: 26's gen_video unchanged (the session block of the yaml supplies phone / lens / glove / person_id), plus distance_cm and hand.
    args = (entry, prep, npz, pcfg, c); c needs candidate, exclusion, detector_settings, guide (trajectory is not used)."""
    entry, prep, npz, pcfg, c = args[:5]
    ses = c.get("session") or {}
    e2 = {"person_id": ses.get("person_id"), "phone": ses.get("phone", "main"), "lens": ses.get("lens", "1x"), "glove": bool(ses.get("glove", False)), **entry}
    res = gen_video((e2, prep, npz, pcfg, {**c, "trajectory": c.get("trajectory") or {}}))
    res.update({"distance_cm": float(entry["distance_cm"]), "hand": entry["hand"]})
    if "ft" in res:
        cand = c["candidate"]
        pc = copy.deepcopy(pcfg)
        pc["tracking"]["conf_min"] = float(c["detector_settings"][cand["detector"]]["conf_min"])
        times = np.array([r["time_s"] for r in prep["frames"]])
        W, H = prep["size"]
        tr = track(load_detections(Path(npz)), times, (W, H), pc["tracking"], front_start_s=float(pc["phases"]["front_skip_s"]))
        cfg_ex = pseg_wrist_cfg(pc) if cand["wrist_mode"] == "exclude" else pc
        a = analyze(tr["kp"], tr["mask"], tr["present"], tr["box"], tr["cls"], times, (W, H), cfg_ex, tilt_start=None, methods_full=(), seed=int(pc["experiment"]["seed"]))
        res["dynamics"] = ext_motion(tr["kp"], tr["mask"], tr["present"], tr["box"], times, a, cfg_ex, (W, H))
    return res


def ext_motion(kp, mask, present, box, times, a: dict, cfg_use: dict, size_wh) -> dict:
    """Motion indicators of the tilt-phase frames of one video, defined as in experiment 23 (ferr_video): tilt axis coverage (analyze's 30-degree sectors with >= min_count frames),
    keypoint speed (pseg_series, hand-box lengths per second), bone-length ratio change (percent) and finger-spread change (degrees against the front window's median) of ferr_shape.
    Median and 90th percentile of every per-frame series. a = analyze() result of the candidate (the template and roles it used)."""
    pb, ids = ferr_problem(a, kp, mask, box, cfg_use, size_wh)
    T = np.array(a["template"])
    keep = list(pb.keep)
    role = a["role"]
    tilt_ids = [i for i in keep if role[i] == "tilt"]
    front_ids = [i for i in keep if role[i] == "front"]
    _, v_series = pseg_series(kp, mask, present, box, times, T, ids)

    def shape_of(i):
        if mask[i][ids].sum() < len(ids):
            return {"bone_ratio": float("nan"), "spread_deg": float("nan")}
        return ferr_shape(T, ids, kp[i][ids], kp[i][0] if mask[i][0] else None)
    ref_spread = float(np.nanmedian([shape_of(i)["spread_deg"] for i in front_ids])) if front_ids else float("nan")
    sh = [shape_of(i) for i in tilt_ids]
    series = {"speed": np.array([v_series[i] for i in tilt_ids], float), "bone_ratio_pct": np.array([100.0 * s["bone_ratio"] for s in sh], float),
              "spread_change_deg": np.array([abs(s["spread_deg"] - ref_spread) for s in sh], float)}
    cov = a["axis_coverage"]
    out = {"n_tilt_frames": len(tilt_ids), "axis_bins": int(cov["bins"]), "axis_counts": [int(x) for x in cov["counts"]], "axis_dominant": cov["dominant"]}
    for k, x in series.items():
        x = x[np.isfinite(x)]
        out[k] = {"median": float(np.median(x)) if len(x) else None, "p90": float(np.percentile(x, 90)) if len(x) else None, "n": int(len(x))}
    return out


def ext_rounds(entries) -> dict:
    """Shooting round of every G6 video: the n-th video (in file order) at its distance is round n. {file: n}."""
    seen, out = {}, {}
    for e in sorted((e for e in entries if e["group"] == "G6"), key=lambda e: e["file"]):
        d = float(e["distance_cm"])
        seen[d] = seen.get(d, 0) + 1
        out[e["file"]] = seen[d]
    return out


def ext_ranks(x) -> np.ndarray:
    """Ranks (0-based, ties share their mean rank)."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


def ext_spearman(x, y):
    """Spearman rank correlation, None with fewer than 3 points or a constant input."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return None
    rx, ry = ext_ranks(x), ext_ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def ext_loglog_slope(d, f):
    """Slope of log f on log distance (least squares), None with fewer than 2 distinct distances."""
    d, f = np.asarray(d, float), np.asarray(f, float)
    if len(d) < 2 or len(np.unique(d)) < 2:
        return None
    return float(np.polyfit(np.log(d), np.log(f), 1)[0])


def ext_pooled(curves, grid, gcfg: dict) -> dict:
    """Pooled estimate of a set of cost curves with the video-level bootstrap interval (empty set: n_videos 0 only)."""
    out = {"n_videos": len(curves)}
    if not len(curves):
        return out
    est = long_pooled_estimate(curves, grid)
    b = long_pooled_bootstrap(curves, grid, int(gcfg["bootstrap_n"]), int(gcfg["bootstrap_seed"]))
    return {**out, "pooled_f": est["f"], "at_grid_edge": est["at_grid_edge"], "side": est["side"], "interval_95": [float(np.quantile(b["f"], 0.025)), float(np.quantile(b["f"], 0.975))]}


def ext_distance_verdict(by_distance: dict, grid, ref_f: float, dcfg: dict) -> dict:
    """G6 verdict. by_distance = {distance_cm: {"curves": [fixed_template cost curves of the included videos], "n_total": videos of that distance}}.
    pass: the pooled f of every distance within +-pooled_tol of ref_f and max / min of the pooled f <= ratio_max; fail otherwise; not judged when a distance has fewer than
    min_videos_per_distance included videos; no videos: 영상 없음."""
    out = {"ref_f": float(ref_f), "distances": {}}
    if not by_distance:
        return {**out, "verdict": "영상 없음"}
    for d in sorted(by_distance):
        x = by_distance[d]
        row = ext_pooled(x["curves"], grid, dcfg)
        row.update({"n_total": int(x["n_total"]), "n_excluded": int(x["n_total"]) - len(x["curves"])})
        if "pooled_f" in row:
            row["pooled_within_tol"] = bool(not row["at_grid_edge"] and abs(row["pooled_f"] / ref_f - 1.0) <= float(dcfg["pooled_tol"]))
        out["distances"][d] = row
    short = {d: r["n_videos"] for d, r in out["distances"].items() if r["n_videos"] < int(dcfg["min_videos_per_distance"])}
    if short:
        return {**out, "verdict": "판단 불가", "reason": ", ".join(f"{d:g} cm 영상 {n}개" for d, n in short.items()) + f" < {int(dcfg['min_videos_per_distance'])}개"}
    fs = [r["pooled_f"] for r in out["distances"].values() if not r["at_grid_edge"]]
    out["max_over_min"] = float(max(fs) / min(fs)) if len(fs) == len(out["distances"]) else None
    ok = all(r["pooled_within_tol"] for r in out["distances"].values()) and out["max_over_min"] is not None and out["max_over_min"] <= float(dcfg["ratio_max"]) + 1e-9
    return {**out, "verdict": "통과" if ok else "실패"}


def ext_distance_records(results: list, entries: list, grid, dcfg: dict) -> dict:
    """Record-only quantities of G6 (no verdict): slope of log f on log distance and Spearman rank correlation (video f and pooled f per distance), f by shooting round (pooled over the
    round's videos and the single f values)."""
    g6 = [r for r in results if r["group"] == "G6" and "ft" in r and not r["ft"]["at_grid_edge"]]
    rounds = ext_rounds(entries)
    d_all, f_all = [r["distance_cm"] for r in g6], [r["ft"]["f"] for r in g6]
    out = {"n_points": len(g6), "slope_video": ext_loglog_slope(d_all, f_all), "spearman_video": ext_spearman(d_all, f_all), "rounds": {}}
    pooled = {}
    for d in sorted({r["distance_cm"] for r in g6}):
        c = [r["curve"]["cost"] for r in g6 if r["distance_cm"] == d]
        est = long_pooled_estimate(c, grid)
        if not est["at_grid_edge"]:
            pooled[d] = est["f"]
    out["pooled_by_distance"] = pooled
    out["slope_pooled"] = ext_loglog_slope(list(pooled), list(pooled.values()))
    out["spearman_pooled"] = ext_spearman(list(pooled), list(pooled.values()))
    for k in sorted(set(rounds.values())):
        rv = [r for r in results if r["group"] == "G6" and rounds.get(r["video"]) == k and "ft" in r]
        est = long_pooled_estimate([r["curve"]["cost"] for r in rv], grid) if rv else None
        out["rounds"][k] = {"videos": [r["video"] for r in rv], "n_videos": len(rv), "pooled_f": None if est is None or est["at_grid_edge"] else est["f"],
                            "f_videos": [None if r["ft"]["at_grid_edge"] else r["ft"]["f"] for r in rv]}
    return out


def ext_hand_compare(results: list, grid, gcfg: dict, right_distance_cm: float) -> dict:
    """G7 left-hand videos beside the G6 right-hand videos of the same distance (record only): pooled f with interval, the single f values and the ratio of the pooled f (left / right)."""
    left = [r for r in results if r["group"] == "G7" and "ft" in r]
    right = [r for r in results if r["group"] == "G6" and "ft" in r and abs(r["distance_cm"] - float(right_distance_cm)) < 1e-9]
    out = {}
    for name, rs in (("left", left), ("right", right)):
        d = ext_pooled([r["curve"]["cost"] for r in rs], grid, gcfg)
        d["videos"] = [r["video"] for r in rs]
        d["f_videos"] = [None if r["ft"]["at_grid_edge"] else r["ft"]["f"] for r in rs]
        out[name] = d
    a, b = out["left"], out["right"]
    out["ratio_left_over_right"] = float(a["pooled_f"] / b["pooled_f"]) if "pooled_f" in a and "pooled_f" in b and not a["at_grid_edge"] and not b["at_grid_edge"] else None
    return out


# ====================================================================================================
# wide  (experiment 29: 0.5x lens at about 60 cm, is the f estimate "always read as about 62 cm"?; new section, earlier functions are used unchanged)
# 27 found f x distance ~ constant (1x: about 105,000 px cm = ref_f x 62.5 cm). Hypothesis: the method reads every hand at about 62.5 cm, i.e. f_est = ref_f x 62.5 / distance.
# Verdict on the pooled f of the videos (cost curves summed): within +-support_tol of predicted_f = support, >= refute_high x predicted_f or <= refute_low x predicted_f = refuted, else open.
# ====================================================================================================

WIDE_REQUIRED = ("file", "group", "lens", "distance_cm", "hand", "ref_f")


def wide_check_videos(entries) -> list:
    """Validates the videos list of the yaml (every entry needs file, group, lens, distance_cm, hand left / right, ref_f). Stops (SystemExit) with the reason otherwise."""
    if not entries:
        raise SystemExit("videos is empty: give the video entries in the yaml (file, group, lens, distance_cm, hand, ref_f)")
    for i, e in enumerate(entries):
        miss = [k for k in WIDE_REQUIRED if e.get(k) is None]
        if miss:
            raise SystemExit(f"videos[{i}] ({e.get('file')}): missing {miss}")
        if e["hand"] not in ("left", "right"):
            raise SystemExit(f"videos[{i}] ({e['file']}): hand must be left or right")
        if float(e["distance_cm"]) <= 0 or float(e["ref_f"]) <= 0:
            raise SystemExit(f"videos[{i}] ({e['file']}): distance_cm and ref_f must be positive")
    return list(entries)


def wide_video(args) -> dict:
    """One video of experiment 29: 26's gen_video unchanged (the session block of the yaml supplies phone / glove / person_id, the entry its lens), plus distance_cm, hand, the principal point
    and the front-window keypoint jitter (percent of the hand box). args = (entry, prep, npz, pcfg, c); c needs candidate, exclusion, detector_settings, guide."""
    entry, prep, npz, pcfg, c = args[:5]
    ses = c.get("session") or {}
    e2 = {"person_id": ses.get("person_id"), "phone": ses.get("phone", "main"), "lens": ses.get("lens", "1x"), "glove": bool(ses.get("glove", False)), **entry}
    res = gen_video((e2, prep, npz, pcfg, {**c, "trajectory": c.get("trajectory") or {}}))
    res.update({"distance_cm": float(entry["distance_cm"]), "hand": entry["hand"]})
    if "ft" in res:
        cand = c["candidate"]
        pc = copy.deepcopy(pcfg)
        pc["tracking"]["conf_min"] = float(c["detector_settings"][cand["detector"]]["conf_min"])
        times = np.array([r["time_s"] for r in prep["frames"]])
        W, H = prep["size"]
        tr = track(load_detections(Path(npz)), times, (W, H), pc["tracking"], front_start_s=float(pc["phases"]["front_skip_s"]))
        cfg_ex = pseg_wrist_cfg(pc) if cand["wrist_mode"] == "exclude" else pc
        a = analyze(tr["kp"], tr["mask"], tr["present"], tr["box"], tr["cls"], times, (W, H), cfg_ex, tilt_start=None, methods_full=(), seed=int(pc["experiment"]["seed"]))
        res["front_jitter_pct"] = 100.0 * float(a["front_jitter_rel"])
        res["principal_point"] = [float(x) for x in a["principal_point"]]
    return res


def wide_implied_distance(f, distance_cm, ref_f):
    """Distance (cm) the estimate reads: distance x f / ref_f (None without a value). 62.5 for every f of the hypothesis."""
    return None if f is None or not np.isfinite(f) else float(distance_cm) * float(f) / float(ref_f)


def wide_hypothesis_verdict(pooled: dict, hcfg: dict) -> dict:
    """Verdict of the hypothesis from the pooled estimate (ext_pooled's dict plus n_excluded): fewer than min_videos videos: 판단 불가; pooled f within +-support_tol of predicted_f: 가설 지지;
    >= refute_high x predicted_f or <= refute_low x predicted_f: 가설 반박 (a value on the f grid edge is the clipped edge value and falls in the refuting range); otherwise 판단 보류."""
    p = float(hcfg["predicted_f"])
    lo, hi = p * (1.0 - float(hcfg["support_tol"])), p * (1.0 + float(hcfg["support_tol"]))
    up, low = p * float(hcfg["refute_high"]), p * float(hcfg["refute_low"])
    out = {"predicted_f": p, "support_range": [lo, hi], "refute_above": up, "refute_below": low, "n_videos": int(pooled.get("n_videos", 0)), "n_excluded": int(pooled.get("n_excluded", 0))}
    if out["n_videos"] < int(hcfg["min_videos"]):
        return {**out, "verdict": "판단 불가", "reason": f"영상 {out['n_videos']}개 < {int(hcfg['min_videos'])}개"}
    f = float(pooled["pooled_f"])
    out.update({"pooled_f": f, "at_grid_edge": bool(pooled["at_grid_edge"]), "side": pooled.get("side"), "interval_95": pooled["interval_95"]})
    if lo - 1e-9 <= f <= hi + 1e-9:
        return {**out, "verdict": "가설 지지"}
    if f >= up - 1e-9 or f <= low + 1e-9:
        return {**out, "verdict": "가설 반박"}
    return {**out, "verdict": "판단 보류"}
