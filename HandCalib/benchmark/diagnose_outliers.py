"""Diagnose why circle/mix videos miss the reference f by more than a threshold.

    python diagnose_outliers.py --config ../configs/diagnose_outliers.yaml
    python diagnose_outliers.py --config ../configs/diagnose_outliers.yaml \
        --data-root ../../data_bench --results-path ../runs/benchmark --output-dir ../runs/diagnose_outliers

Targets are the videos of the configured motions whose |error| = |f - reference_f| / reference_f is above
abs_error_threshold in the existing results, plus the partner video of the same session and motion
(circle_01 <-> circle_02). Nothing in calibrator.py or the benchmark scripts is changed: Calibrator is
wrapped (a tap on its Wholebody model) and its estimate() steps are repeated here to expose the values
it keeps inside (homographies, the perspective term, the selected frames).

Outputs in output_dir:
    diagnostics.csv   one row per target video: values listed in the config, error, sign, partner
    pairs.md          per pair: values of the wrong video vs the normal one, large differences marked;
                      pairs where both videos are wrong are collected in their own section
    validation.csv    f recomputed by this module vs the existing result for every scored video
    cache/            per-video keypoints (npz), so a rerun does not detect again
    run.json          config, code version, environment

Definitions used in the output:
    detect_rate          accepted frames (all 21 right-hand scores > 3) / frames of the video
    conf_mean            mean keypoint score over accepted frames
    conf_p10             10th percentile of the per-frame minimum score over accepted frames
    bbox_ratio           median over accepted frames of (long side of the 21-point bbox) / (short side of the frame)
    tilt_*               hypot(H[2,0], H[2,1]) of the template -> image homography, the value calibrator.py
                         ranks frames by; tilt_top_half is the mean over the frames it selects
    n_selected           frames with tilt above the median = the frames calibrateCamera uses
    dir_*                tilt direction of the selected frames in image coordinates (E = +x, S = +y):
                         A^-T (H[2,0], H[2,1]) with A = H[:2,:2] (a local approximation); 8 bins of 45 degrees;
                         dir_R = length of the mean unit vector (1 = one side), dir_R2 = same for doubled angles
                         (1 = along one axis, either sign); diag_share = share in the four diagonal bins;
                         dir_axis_* = the 8 bins folded to 4 axes (E-W, SE-NW, S-N, SW-NE): share of the largest.
                         The direction is relative to calibrator.py's mean-shape template, not to the camera, so
                         its sign can flip between frames; dir_R2 and dir_axis_share are the sign-free values
    shape_resid_*        per-frame rms distance between the homography-projected template and the keypoints,
                         as % of the bbox long side (median and 90th percentile over accepted frames)
    reproj_rms_px        RMS reprojection error returned by calibrateCamera on the selected frames
    boot_*               calibrateCamera repeated on bootstrap resamples of the selected frames
                         (template and selection kept): coefficient of variation and 5th/95th percentile of f
    mix_share_*          share of the selected frames inside each motion segment (needs mix_segments)

mix_segments file (yaml or json): {"<session>/<file>": [{name: twinkle, start_s: 0.0, end_s: 7.5}, ...]};
frame time is frame index / fps. The key may also be the file name alone.
"""
import argparse
import csv
import json
import math
import platform
import re
import subprocess
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from calibrator import FOCAL_ONLY, RIGHT_HAND, Calibrator
from session_rules import HAND_STEMS, find_video

BIN_NAMES = ("E", "SE", "S", "SW", "W", "NW", "N", "NE")
AXIS_NAMES = ("E-W", "SE-NW", "S-N", "SW-NE")
SEGMENT_NAMES = ("twinkle", "nod", "circle")
DIAG_KEYS = ("detection", "tilt", "tilt_direction", "hand_shape", "stability", "mix_segments")

COLUMNS = (
    ["video_id", "session", "file", "motion", "role", "pair_id", "reference_f", "f_existing", "f_recomputed",
     "error", "abs_error", "sign"]
    + ["n_frames", "n_detected", "detect_rate", "conf_mean", "conf_p10", "bbox_ratio"]
    + ["tilt_median", "tilt_p90", "tilt_top_half", "n_selected"]
    + [f"dir_{b}" for b in BIN_NAMES] + ["dir_dominant", "dir_dominant_share", "dir_R", "dir_R2", "diag_share",
                                         "dir_axis_dominant", "dir_axis_share"]
    + ["shape_resid_median_pct", "shape_resid_p90_pct"]
    + ["reproj_rms_px", "boot_cv", "boot_p05", "boot_p95", "boot_fail"]
    + [f"mix_share_{n}" for n in SEGMENT_NAMES] + ["mix_share_other"]
)

# (label, column, format, higher is "larger"), the values compared in pairs.md
COMPARE = (
    ("검출률", "detect_rate", "{:.3f}"), ("신뢰도 평균", "conf_mean", "{:.2f}"), ("신뢰도 하위 10%", "conf_p10", "{:.2f}"),
    ("손 bbox / 화면 짧은 변", "bbox_ratio", "{:.3f}"),
    ("원근 항 중앙값", "tilt_median", "{:.3e}"), ("원근 항 상위 90%", "tilt_p90", "{:.3e}"),
    ("원근 항 상위 절반 평균", "tilt_top_half", "{:.3e}"), ("f 계산에 쓴 프레임 수", "n_selected", "{:.0f}"),
    ("기울임 방향 최다 비율", "dir_dominant_share", "{:.2f}"), ("기울임 방향 쏠림 R", "dir_R", "{:.2f}"),
    ("기울임 한 축 최다 비율", "dir_axis_share", "{:.2f}"),
    ("기울임 한 축 쏠림 R2", "dir_R2", "{:.2f}"), ("대각선 방향 비율", "diag_share", "{:.2f}"),
    ("모양 잔차 중앙값 (% bbox)", "shape_resid_median_pct", "{:.2f}"),
    ("모양 잔차 상위 10% (% bbox)", "shape_resid_p90_pct", "{:.2f}"),
    ("재투영 오차 (px)", "reproj_rms_px", "{:.3f}"), ("부트스트랩 변동계수", "boot_cv", "{:.4f}"),
    ("추정 f", "f_recomputed", "{:.1f}"),
)


# ----------------------------------------------------------------------------- config

def _scalar(text):
    t = text.strip()
    if t in ("", "null", "~", "Null", "NULL"):
        return None
    if t in ("true", "True"):
        return True
    if t in ("false", "False"):
        return False
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "'\"":
        return t[1:-1]
    if t[0] in "[{":
        return _flow(t)
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        return t


def _split_top(text):
    parts, depth, start, quote = [], 0, 0, None
    for i, ch in enumerate(text):
        if quote:
            quote = None if ch == quote else quote
        elif ch in "'\"":
            quote = ch
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    tail = text[start:]
    if tail.strip():
        parts.append(tail)
    return parts


def _flow(text):
    text = text.strip()
    if text[0] == "[":
        return [_scalar(p) for p in _split_top(text[1:-1])]
    out = {}
    for part in _split_top(text[1:-1]):
        key, _, value = part.partition(":")
        out[key.strip()] = _scalar(value)
    return out


def _strip_comment(line):
    quote = None
    for i, ch in enumerate(line):
        if quote:
            quote = None if ch == quote else quote
        elif ch in "'\"":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            return line[:i]
    return line


def mini_yaml(text):
    """The yaml subset the configs of this repo use: nested mappings by indentation, flow lists/maps, scalars."""
    root = {}
    stack = [(-1, root)]
    for raw in text.splitlines():
        line = _strip_comment(raw).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        body = line.strip()
        if body.startswith("- "):
            raise ValueError(f"block lists are not supported by the built-in parser, use [a, b]: {raw!r}")
        key, sep, value = body.partition(":")
        if not sep:
            raise ValueError(f"cannot parse line: {raw!r}")
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child = {}
            parent[key.strip()] = child
            stack.append((indent, child))
        else:
            parent[key.strip()] = _scalar(value)
    return root


def load_config(path):
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        return mini_yaml(text)
    return yaml.safe_load(text)


# ----------------------------------------------------------------------------- inputs

def load_results(path):
    p = Path(path)
    if p.is_file():
        files = [p]
    elif (p / "results.csv").exists():
        files = [p / "results.csv"]
    else:
        files = sorted(p.glob("*/results.csv"))
    if not files:
        raise SystemExit(f"results.csv not found under {p}")
    latest = {}
    for f in files:  # later runs (sorted by name = time) replace earlier rows of the same video
        with open(f, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                row["run"] = f.parent.name
                latest[(row["session"], row["file"])] = row
    return {k: v for k, v in latest.items() if v["status"] == "ok" and v["f"] not in ("", None)}


def read_reference(root, session):
    path = Path(root) / session / "session.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("reference_f")


def git_info():
    def run(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()
    return {"branch": run("rev-parse", "--abbrev-ref", "HEAD"), "commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--", "."))}


# ----------------------------------------------------------------------------- detection (wraps Calibrator)

class _Tap:
    """Records what the Wholebody model returned for the last frame; the result passes through unchanged."""

    def __init__(self, model):
        self.model = model
        self.last = None

    def __call__(self, frame):
        keypoints, scores = self.model(frame)
        self.last = (np.array(keypoints[0, RIGHT_HAND]), np.array(scores[0, RIGHT_HAND]))
        return keypoints, scores


class DiagCalibrator(Calibrator):
    def __init__(self):
        super().__init__("hand")
        self.model = _Tap(self.model)
        self.start()

    def start(self):
        self.reset()
        self.n_frames = 0
        self.indices, self.scores, self.frame_min = [], [], []

    def add(self, frame):
        accepted = len(self.observations)
        index = self.n_frames
        self.n_frames += 1
        super().add(frame)
        _, scores = self.model.last
        self.frame_min.append(float(scores.min()))
        if len(self.observations) > accepted:
            self.indices.append(index)
            self.scores.append(scores)


def detect_video(calibrator, path):
    """Same loop as Calibrator.run() minus the estimate; returns the arrays that estimate() starts from."""
    calibrator.start()
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        calibrator.add(frame)
    cap.release()
    return {
        "points": np.float32(calibrator.observations), "indices": np.int32(calibrator.indices),
        "scores": np.float32(calibrator.scores), "frame_min": np.float32(calibrator.frame_min),
        "n_frames": calibrator.n_frames, "size": np.int32(calibrator.size), "fps": float(fps),
    }


def cache_path(cache_dir, session, file):
    return Path(cache_dir) / f"{session}__{Path(file).stem}.npz"


def load_cache(path, video):
    if not path.exists():
        return None
    with np.load(path) as z:
        if int(z["video_bytes"]) != video.stat().st_size:
            return None
        return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files if k != "video_bytes"}


def save_cache(path, video, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, video_bytes=video.stat().st_size, **data)


# ----------------------------------------------------------------------------- the estimate, step by step

class _Shim:
    """What Calibrator.template() reads from self."""
    target = "hand"
    pattern_size = (7, 7)


def core_estimate(points, size):
    """Calibrator.estimate() repeated with its inside values kept. Same calls, same order."""
    points = np.float32(points)
    template = Calibrator.template(_Shim, points)
    homographies = [cv2.findHomography(template, p)[0] for p in points]
    tilt = np.array([np.hypot(*H[2, :2]) for H in homographies])
    selected = np.flatnonzero(tilt > np.median(tilt))
    object_points = [np.float32(np.c_[template, np.zeros(len(template))])] * len(selected)
    image_points = list(points[selected])
    rms, K = cv2.calibrateCamera(object_points, image_points, tuple(int(v) for v in size), np.eye(3), None,
                                 flags=FOCAL_ONLY)[:2]
    return {"template": template, "H": homographies, "tilt": tilt, "selected": selected,
            "f": float(K[0, 0]), "rms": float(rms)}


def base_estimate(points, size):
    """The unmodified Calibrator.estimate() on the same observations, for the equality check."""
    class Box:
        pass
    box = Box()
    box.observations = list(points)
    box.size = tuple(int(v) for v in size)
    box.template = lambda pts: Calibrator.template(_Shim, pts)
    return float(Calibrator.estimate(box))


def tilt_direction(homographies, selected):
    angles = []
    for i in selected:
        H = homographies[i]
        try:
            d = np.linalg.solve(H[:2, :2].T, H[2, :2])
        except np.linalg.LinAlgError:
            continue
        angles.append(math.atan2(d[1], d[0]))
    angles = np.array(angles)
    out = {}
    if len(angles) == 0:
        return out
    bins = ((np.degrees(angles) + 22.5) % 360 // 45).astype(int)
    hist = np.bincount(bins, minlength=8) / len(angles)
    for name, share in zip(BIN_NAMES, hist):
        out[f"dir_{name}"] = float(share)
    out["dir_dominant"] = BIN_NAMES[int(hist.argmax())]
    out["dir_dominant_share"] = float(hist.max())
    out["dir_R"] = float(abs(np.exp(1j * angles).mean()))
    out["dir_R2"] = float(abs(np.exp(2j * angles).mean()))
    out["diag_share"] = float(hist[1] + hist[3] + hist[5] + hist[7])
    axis = np.bincount(((np.degrees(angles) + 22.5) % 180 // 45).astype(int), minlength=4) / len(angles)
    out["dir_axis_dominant"] = AXIS_NAMES[int(axis.argmax())]
    out["dir_axis_share"] = float(axis.max())
    return out


def shape_residuals(points, template, homographies):
    homog = np.c_[template, np.ones(len(template))]
    long_side = np.ptp(points, axis=1).max(axis=1)
    out = []
    for p, H, size in zip(points, homographies, long_side):
        q = homog @ H.T
        q = q[:, :2] / q[:, 2:3]
        out.append(100 * np.sqrt(((q - p) ** 2).sum(axis=1).mean()) / size)
    return np.array(out)


def bootstrap_f(points, template, selected, size, runs, rng):
    object_one = np.float32(np.c_[template, np.zeros(len(template))])
    values, failed = [], 0
    for _ in range(runs):
        pick = rng.choice(selected, size=len(selected), replace=True)
        try:
            K = cv2.calibrateCamera([object_one] * len(pick), list(points[pick]), tuple(int(v) for v in size),
                                    np.eye(3), None, flags=FOCAL_ONLY)[1]
            f = float(K[0, 0])
        except cv2.error:
            f = float("nan")
        if np.isfinite(f) and f > 0:
            values.append(f)
        else:
            failed += 1
    return np.array(values), failed


def segment_shares(indices, selected, fps, segments):
    if not segments:
        return {}
    times = indices[selected] / fps
    counts = {n: 0 for n in SEGMENT_NAMES}
    other = 0
    for t in times:
        hit = next((s["name"] for s in segments if s["start_s"] <= t < s["end_s"] and s["name"] in counts), None)
        if hit:
            counts[hit] += 1
        else:
            other += 1
    total = max(len(times), 1)
    return {**{f"mix_share_{n}": counts[n] / total for n in SEGMENT_NAMES}, "mix_share_other": other / total}


def diagnose(data, cfg, seed_key, segments):
    """Every diagnostic value of one video from its cached keypoints."""
    flags = cfg["diagnostics"]
    points, size = data["points"], data["size"]
    est = core_estimate(points, size)
    out = {"f_recomputed": est["f"], "n_frames": int(data["n_frames"]), "n_detected": len(points)}
    if flags.get("detection", True):
        scores = data["scores"]
        out["detect_rate"] = len(points) / data["n_frames"]
        out["conf_mean"] = float(scores.mean())
        out["conf_p10"] = float(np.percentile(scores.min(axis=1), 10))
        long_side = np.ptp(points, axis=1).max(axis=1)
        out["bbox_ratio"] = float(np.median(long_side) / min(size))
    if flags.get("tilt", True):
        tilt, sel = est["tilt"], est["selected"]
        out.update(tilt_median=float(np.median(tilt)), tilt_p90=float(np.percentile(tilt, 90)),
                   tilt_top_half=float(tilt[sel].mean()), n_selected=len(sel))
    if flags.get("tilt_direction", True):
        out.update(tilt_direction(est["H"], est["selected"]))
    if flags.get("hand_shape", True):
        resid = shape_residuals(points, est["template"], est["H"])
        out.update(shape_resid_median_pct=float(np.median(resid)),
                   shape_resid_p90_pct=float(np.percentile(resid, 90)))
    if flags.get("stability", True):
        rng = np.random.default_rng(int(cfg["experiment"]["seed"]) + zlib.crc32(seed_key.encode()))
        values, failed = bootstrap_f(points, est["template"], est["selected"], size,
                                     int(cfg.get("bootstrap_runs", 200)), rng)
        out["reproj_rms_px"] = est["rms"]
        out["boot_fail"] = failed
        if len(values) > 1:
            out.update(boot_cv=float(values.std(ddof=1) / values.mean()),
                       boot_p05=float(np.percentile(values, 5)), boot_p95=float(np.percentile(values, 95)))
    if flags.get("mix_segments", True) and segments:
        out.update(segment_shares(data["indices"], est["selected"], data["fps"], segments))
    return out


# ----------------------------------------------------------------------------- outputs

def fmt(value, pattern):
    return "-" if value in (None, "") or (isinstance(value, float) and not math.isfinite(value)) else pattern.format(value)


def relative_gap(a, b):
    if a in (None, "") or b in (None, ""):
        return None
    scale = max(abs(a), abs(b))
    return 0.0 if scale < 1e-12 else abs(a - b) / scale


def compare_table(left, right, left_name, right_name, large):
    lines = [f"| 항목 | {left_name} | {right_name} | 차이 | 표시 |", "|---|---|---|---|---|"]
    flagged = []
    for label, key, pattern in COMPARE:
        a, b = left.get(key), right.get(key)
        if a in (None, "") and b in (None, ""):
            continue
        gap = relative_gap(a, b)
        mark = "●" if gap is not None and gap >= large else ""
        if mark:
            flagged.append(key)
        diff = "-" if gap is None else f"{a - b:+.4g} ({100 * gap:.0f}%)"
        lines.append(f"| {label} | {fmt(a, pattern)} | {fmt(b, pattern)} | {diff} | {mark} |")
    hist = ["| 기울임 방향 | " + " | ".join(BIN_NAMES) + " |", "|---|" + "---|" * len(BIN_NAMES)]
    for name, row in ((left_name, left), (right_name, right)):
        if row.get("dir_E") not in (None, ""):
            hist.append(f"| {name} | " + " | ".join(f"{100 * row[f'dir_{b}']:.0f}" for b in BIN_NAMES) + " |")
    return lines, flagged, hist if len(hist) > 2 else []


def sign_word(error):
    return "과대(+)" if error > 0 else "과소(-)"


def describe(row):
    return f"{row['file']} (오차 {100 * row['error']:+.1f}%, {sign_word(row['error'])}, f {row['f_existing']:.1f})"


def write_pairs(path, one_wrong, both_wrong, cfg, rows_by_id, skipped):
    large = float(cfg.get("large_difference_ratio", 0.3))
    L = ["# 이상치 진단: 같은 세션 짝 영상 비교", "",
         f"- 대상: {', '.join(cfg['motions'])} 영상 중 |오차| > {100 * cfg['abs_error_threshold']:.0f}%",
         f"- 짝: 같은 세션, 같은 동작의 01 ↔ 02. 오차 = (추정 f − reference_f) / reference_f",
         f"- ● 표시: 두 값의 상대 차이 |a−b| / max(|a|,|b|) ≥ {100 * large:.0f}%",
         f"- 값의 정의는 `diagnose_outliers.py` 머리말을 참고하세요.", ""]
    L += [f"## 한쪽만 틀린 짝 ({len(one_wrong)}개)", ""]
    tally = {key: [0, 0, 0] for _, key, _ in COMPARE}  # outlier larger, smaller, flagged
    for wrong, ok in one_wrong:
        L += [f"### {wrong['session']} · {wrong['motion']}", "",
              f"- 틀린 영상: {describe(wrong)}", f"- 정상 영상: {describe(ok)}", ""]
        table, flagged, hist = compare_table(wrong, ok, "틀린 영상", "정상 영상", large)
        L += table + [""] + hist + [""]
        for _, key, _ in COMPARE:
            a, b = wrong.get(key), ok.get(key)
            if a in (None, "") or b in (None, ""):
                continue
            tally[key][0] += a > b
            tally[key][1] += a < b
            tally[key][2] += key in flagged
        L += ["- 크게 다른 항목: " + (", ".join(label for label, key, _ in COMPARE if key in flagged) or "없음"), ""]
    if one_wrong:
        under = sum(w["error"] < 0 for w, _ in one_wrong)
        L += ["## 경향 요약 (한쪽만 틀린 짝)", "",
              f"- 틀린 영상 {len(one_wrong)}개 중 과소 {under}개, 과대 {len(one_wrong) - under}개", "",
              "| 항목 | 틀린 영상이 더 큼 | 틀린 영상이 더 작음 | ● 표시 |", "|---|---|---|---|"]
        for label, key, _ in COMPARE:
            big, small, flagged_count = tally[key]
            if big + small:
                L.append(f"| {label} | {big} | {small} | {flagged_count} |")
        L.append("")
    L += [f"## 둘 다 틀린 짝 ({len(both_wrong)}개)", ""]
    if not both_wrong:
        L += ["없음", ""]
    for first, second in both_wrong:
        same = "같은 방향" if (first["error"] > 0) == (second["error"] > 0) else "반대 방향"
        L += [f"### {first['session']} · {first['motion']} ({same})", "",
              f"- {describe(first)}", f"- {describe(second)}", ""]
        table, flagged, hist = compare_table(first, second, first["file"], second["file"], large)
        L += table + [""] + hist + [""]
    if skipped:
        L += ["## 짝을 만들지 못한 영상", ""] + [f"- {s}" for s in skipped] + [""]
    path.write_text("\n".join(L), encoding="utf-8")


def write_csv(path, rows, columns):
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in columns})


# ----------------------------------------------------------------------------- main

def resolve(base, value):
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


def load_segments(path):
    if not path:
        return {}
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text) if str(path).endswith(".json") else load_config(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root")
    ap.add_argument("--results-path")
    ap.add_argument("--output-dir")
    ap.add_argument("--no-validation", action="store_true", help="skip the recomputation check of non-target videos")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    base = config_path.parent.parent  # HandCalib/, as the other configs' relative paths
    # command-line paths are relative to the current folder, config paths to HandCalib/
    data_root = Path(args.data_root).resolve() if args.data_root else resolve(base, cfg["data_root"])
    results_path = Path(args.results_path).resolve() if args.results_path else resolve(base, cfg["results_path"])
    out = Path(args.output_dir).resolve() if args.output_dir else resolve(base, cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    threshold = float(cfg["abs_error_threshold"])
    motions = list(cfg["motions"])
    excluded = set(cfg.get("exclude_sessions") or [])
    segments_all = load_segments(resolve(base, cfg["mix_segments"]) if cfg.get("mix_segments") else None)
    if cfg.get("pair_mode", "same_session") != "same_session":
        raise SystemExit("pair_mode: only same_session is implemented")

    # --- existing results, errors, targets and partners (no GPU needed)
    results = load_results(results_path)
    refs, videos = {}, {}
    for (session, file), row in sorted(results.items()):
        if session in excluded or Path(file).stem not in HAND_STEMS:
            continue
        if session not in refs:
            refs[session] = read_reference(data_root, session)
        session_dir = data_root / session
        path = find_video(session_dir, Path(file).stem) if session_dir.is_dir() else None
        if refs[session] and path:
            f = float(row["f"])
            videos[(session, file)] = {
                "video_id": f"{session}/{file}", "session": session, "file": file, "path": path,
                "motion": row["motion"], "reference_f": refs[session], "f_existing": f,
                "error": (f - refs[session]) / refs[session], "n_detected_existing": int(row["n_detected"]),
                "run": row["run"],
            }
    print(f"{len(videos)} scored videos in {len({v['session'] for v in videos.values()})} sessions "
          f"(excluded sessions: {sorted(excluded) or 'none'})", flush=True)

    def partner_of(video):
        stem = Path(video["file"]).stem
        other = stem[:-2] + ("02" if stem.endswith("01") else "01")
        return next((v for v in videos.values()
                     if v["session"] == video["session"] and Path(v["file"]).stem == other), None)

    outliers = [v for v in videos.values() if v["motion"] in motions and abs(v["error"]) > threshold]
    one_wrong, both_wrong, skipped, targets = [], [], [], {}
    for v in sorted(outliers, key=lambda x: (x["session"], x["file"])):
        mate = partner_of(v)
        if mate is None:
            skipped.append(f"{v['video_id']}: 같은 세션의 짝 영상 결과가 없음")
            targets[v["video_id"]] = (v, "outlier", None)
            continue
        wrong_both = abs(mate["error"]) > threshold
        targets[v["video_id"]] = (v, "both_outlier" if wrong_both else "outlier", mate["video_id"])
        targets.setdefault(mate["video_id"], (mate, "both_outlier" if wrong_both else "pair", v["video_id"]))
    print(f"{len(outliers)} videos above {100 * threshold:.0f}% in {motions}; {len(targets)} target videos "
          f"(outliers + partners)", flush=True)

    # --- detection for every scored video, targets first; the cache keeps it between runs
    cache_dir = out / "cache"
    order = sorted(videos.values(), key=lambda v: (v["video_id"] not in targets, v["session"], v["file"]))
    if args.no_validation:
        order = [v for v in order if v["video_id"] in targets]
    calibrator, data = None, {}
    t_start = time.time()
    for n, v in enumerate(order, 1):
        cache = cache_path(cache_dir, v["session"], v["file"])
        loaded = load_cache(cache, v["path"])
        if loaded is None:
            if calibrator is None:
                calibrator = DiagCalibrator()
            t0 = time.time()
            loaded = detect_video(calibrator, v["path"])
            save_cache(cache, v["path"], loaded)
            print(f"[{n}/{len(order)}] detected {v['video_id']} ({loaded['n_frames']} frames, "
                  f"{time.time() - t0:.0f}s)", flush=True)
        else:
            print(f"[{n}/{len(order)}] cache {v['video_id']}", flush=True)
        data[v["video_id"]] = loaded

    # --- validation: the same f from the same estimate steps, on every video that was detected
    validation, bad = [], []
    for v in order:
        d = data[v["video_id"]]
        if len(d["points"]) < 6:
            validation.append({"video_id": v["video_id"], "session": v["session"], "file": v["file"],
                               "motion": v["motion"], "note": "too few accepted frames"})
            continue
        est = core_estimate(d["points"], d["size"])
        base_f = base_estimate(d["points"], d["size"])
        diff = 100 * (est["f"] - v["f_existing"]) / v["f_existing"]
        row = {"video_id": v["video_id"], "session": v["session"], "file": v["file"], "motion": v["motion"],
               "f_existing": v["f_existing"], "f_recomputed": round(est["f"], 3), "diff_pct": round(diff, 4),
               "f_calibrator_estimate": round(base_f, 3),
               "diag_vs_calibrator_diff": est["f"] - base_f,
               "n_detected_existing": v["n_detected_existing"], "n_detected_new": len(d["points"]),
               "existing_run": v["run"]}
        validation.append(row)
        if abs(diff) > 0.1:
            bad.append(row)
    write_csv(out / "validation.csv", validation,
              ["video_id", "session", "file", "motion", "f_existing", "f_recomputed", "diff_pct",
               "f_calibrator_estimate", "diag_vs_calibrator_diff", "n_detected_existing", "n_detected_new",
               "existing_run", "note"])
    worst = max((abs(r["diff_pct"]) for r in validation if "diff_pct" in r), default=0.0)
    print(f"validation: {len(validation)} videos, {len(bad)} differ by more than 0.1% (largest {worst:.4f}%)",
          flush=True)

    # --- diagnostics of the targets
    rows_by_id = {}
    for video_id, (v, role, pair_id) in sorted(targets.items()):
        segs = segments_all.get(video_id) or segments_all.get(v["file"]) or []
        t0 = time.time()
        values = diagnose(data[video_id], cfg, video_id, segs)
        row = {k: v[k] for k in ("video_id", "session", "file", "motion", "reference_f", "f_existing", "error")}
        row.update(role=role, pair_id=pair_id or "", abs_error=abs(v["error"]), sign=sign_word(v["error"]), **values)
        rows_by_id[video_id] = row
        print(f"diagnosed {video_id} ({role}, error {100 * v['error']:+.1f}%, {time.time() - t0:.0f}s)", flush=True)
    write_csv(out / "diagnostics.csv", list(rows_by_id.values()), COLUMNS)

    seen = set()
    for video_id, row in rows_by_id.items():
        if row["role"] == "both_outlier" and row["pair_id"] in rows_by_id and row["pair_id"] not in seen:
            seen.add(video_id)
            both_wrong.append((row, rows_by_id[row["pair_id"]]))
        elif row["role"] == "outlier" and row["pair_id"] in rows_by_id:
            one_wrong.append((row, rows_by_id[row["pair_id"]]))
    both_wrong = [(a, b) for a, b in both_wrong]
    write_pairs(out / "pairs.md", one_wrong, both_wrong, cfg, rows_by_id, skipped)

    (out / "run.json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "config_file": str(config_path),
        "config": cfg, "data_root": str(data_root), "results_path": str(results_path), "output_dir": str(out),
        "git": git_info(),
        "environment": {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__},
        "n_scored_videos": len(videos), "n_outliers": len(outliers), "n_targets": len(targets),
        "validation": {"videos": len(validation), "differ_over_0.1pct": len(bad), "largest_abs_diff_pct": worst},
        "seconds": round(time.time() - t_start, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done: {out}", flush=True)


if __name__ == "__main__":
    main()
