"""Extract the pre-registered RGB scene features from the frozen 64-frame set.

Classical OpenCV/NumPy only - no pretrained model, no hand annotation, no
detector box, no ground truth. The feature list was frozen in
experiments/manifests/cam_exp_005_scene_feature_spec_v1.json before any
correlation with the target was computed.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MANIFESTS, RAW, REPO, read_csv, read_json, write_csv  # noqa: E402

FRAMES64 = MANIFESTS / "gigahands_demo_cam_exp_0041_64frames_v1.csv.gz"
SPEC = read_json(MANIFESTS / "cam_exp_005_scene_feature_spec_v1.json")
BORDER = float(SPEC["border_fraction"])

# LSD was removed from the default OpenCV build in some versions; fall back to
# the probabilistic Hough transform and record which one was used.
def make_line_detector():
    try:
        d = cv2.createLineSegmentDetector()
        return d, "LSD"
    except Exception:
        return None, "HOUGHP"


def detect_lines(gray, detector, kind):
    if kind == "LSD":
        lines = detector.detect(gray)[0]
        if lines is None:
            return np.zeros((0, 4), np.float32)
        return lines.reshape(-1, 4).astype(np.float32)
    edges = cv2.Canny(gray, 50, 150)
    diag = float(np.hypot(*gray.shape[:2]))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                            minLineLength=max(diag * 0.03, 20), maxLineGap=6)
    if lines is None:
        return np.zeros((0, 4), np.float32)
    return lines.reshape(-1, 4).astype(np.float32)


def entropy(hist):
    p = hist.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def line_stats(seg, diag, prefix=""):
    """Orientation / length statistics for a set of segments."""
    out = {}
    if seg.shape[0] == 0:
        for k in ("line_count", "long_line_count", "total_line_length_over_area",
                  "median_norm_line_length", "horizontal_line_fraction",
                  "vertical_line_fraction", "diagonal_line_fraction",
                  "line_orientation_entropy",
                  "dominant_line_orientation_strength", "orthogonal_line_support"):
            out[prefix + k] = 0.0
        return out, np.zeros(0), np.zeros(0)
    dx = seg[:, 2] - seg[:, 0]
    dy = seg[:, 3] - seg[:, 1]
    length = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(dy, dx)) % 180.0
    horiz = (ang < 15) | (ang > 165)
    vert = np.abs(ang - 90) < 15
    hist, _ = np.histogram(ang, bins=18, range=(0, 180), weights=length)
    out[prefix + "line_count"] = float(seg.shape[0])
    out[prefix + "long_line_count"] = float((length > 0.10 * diag).sum())
    out[prefix + "median_norm_line_length"] = float(np.median(length) / diag)
    out[prefix + "horizontal_line_fraction"] = float(horiz.mean())
    out[prefix + "vertical_line_fraction"] = float(vert.mean())
    out[prefix + "diagonal_line_fraction"] = float((~horiz & ~vert).mean())
    out[prefix + "line_orientation_entropy"] = entropy(hist)
    out[prefix + "dominant_line_orientation_strength"] = float(
        hist.max() / hist.sum()) if hist.sum() > 0 else 0.0
    # orthogonality support: share of sampled segment pairs 90 +/- 10 deg apart
    n = seg.shape[0]
    if n >= 4:
        rng = np.random.default_rng(0)
        k = min(4000, n * (n - 1) // 2)
        i = rng.integers(0, n, k)
        j = rng.integers(0, n, k)
        keep = i != j
        d = np.abs(ang[i[keep]] - ang[j[keep]])
        d = np.minimum(d, 180 - d)
        out[prefix + "orthogonal_line_support"] = float(
            (np.abs(d - 90) < 10).mean())
    else:
        out[prefix + "orthogonal_line_support"] = 0.0
    return out, length, ang


def vp_consensus(seg, w, h, rng):
    """Largest share of segments agreeing on one vanishing point.

    Simple RANSAC over segment-line intersections. Recorded as
    FEATURE_SKIPPED_UNSTABLE if too few segments.
    """
    n = seg.shape[0]
    if n < 20:
        return float("nan")
    p1 = np.stack([seg[:, 0], seg[:, 1], np.ones(n)], 1)
    p2 = np.stack([seg[:, 2], seg[:, 3], np.ones(n)], 1)
    lines = np.cross(p1, p2)
    best = 0.0
    diag = float(np.hypot(w, h))
    for _ in range(120):
        a, b = rng.integers(0, n, 2)
        if a == b:
            continue
        vp = np.cross(lines[a], lines[b])
        if abs(vp[2]) < 1e-9:
            continue
        vp = vp / vp[2]
        if not np.all(np.isfinite(vp)):
            continue
        mid = np.stack([(seg[:, 0] + seg[:, 2]) / 2,
                        (seg[:, 1] + seg[:, 3]) / 2], 1)
        to_vp = vp[:2][None, :] - mid
        d = np.stack([seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]], 1)
        nv = np.linalg.norm(to_vp, axis=1) * np.linalg.norm(d, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = np.abs((to_vp * d).sum(1) / np.maximum(nv, 1e-9))
        share = float((cos > np.cos(np.radians(2.0))).mean())
        best = max(best, share)
    return best


def frame_features(img, detector, kind, rng):
    h, w = img.shape[:2]
    diag = float(np.hypot(w, h))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    f = {}

    edges = cv2.Canny(gray, 50, 150)
    f["edge_density"] = float(edges.mean() / 255.0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    f["grad_mag_mean"] = float(mag.mean())
    f["grad_mag_p90"] = float(np.percentile(mag, 90))
    ori = (np.degrees(np.arctan2(gy, gx)) % 180.0).ravel()
    ohist, _ = np.histogram(ori, bins=36, range=(0, 180), weights=mag.ravel())
    f["grad_orientation_entropy"] = entropy(ohist)

    seg = detect_lines(gray, detector, kind)
    ls, length, ang = line_stats(seg, diag)
    ls["total_line_length_over_area"] = (float(length.sum() / (w * h))
                                         if length.size else 0.0)
    f.update(ls)
    corners = cv2.goodFeaturesToTrack(gray, 1000, 0.01, 8)
    f["corner_count_over_area"] = float(
        (0 if corners is None else len(corners)) / (w * h) * 1e6)
    f["vp_consensus_strength"] = vp_consensus(seg, w, h, rng)

    f["gray_mean"] = float(gray.mean())
    f["gray_std"] = float(gray.std())
    ghist, _ = np.histogram(gray, bins=256, range=(0, 256))
    f["gray_entropy"] = entropy(ghist)
    f["saturation_mean"] = float(hsv[:, :, 1].mean())
    f["saturation_std"] = float(hsv[:, :, 1].std())
    bs = 16
    hh, ww = gray.shape[0] // bs * bs, gray.shape[1] // bs * bs
    blocks = gray[:hh, :ww].reshape(hh // bs, bs, ww // bs, bs)
    f["local_contrast"] = float(blocks.std(axis=(1, 3)).mean())
    f["laplacian_variance"] = float(cv2.Laplacian(gray, cv2.CV_32F).var())

    # border ring: the outer BORDER fraction on every side
    mask = np.ones((h, w), bool)
    mask[int(h * BORDER):int(h * (1 - BORDER)),
         int(w * BORDER):int(w * (1 - BORDER))] = False
    f["border_edge_density"] = float(edges[mask].mean() / 255.0)
    f["border_grad_mag_mean"] = float(mag[mask].mean())
    f["border_gray_std"] = float(gray[mask].std())
    if seg.shape[0]:
        mx = (seg[:, 0] + seg[:, 2]) / 2
        my = (seg[:, 1] + seg[:, 3]) / 2
        inb = mask[np.clip(my.astype(int), 0, h - 1),
                   np.clip(mx.astype(int), 0, w - 1)]
        bseg = seg[inb]
    else:
        bseg = seg
    bls, blen, _ = line_stats(bseg, diag, prefix="border_")
    f["border_line_count"] = bls["border_line_count"]
    f["border_long_line_count"] = bls["border_long_line_count"]
    f["border_horizontal_line_fraction"] = bls["border_horizontal_line_fraction"]
    f["border_vertical_line_fraction"] = bls["border_vertical_line_fraction"]
    f["border_line_orientation_entropy"] = bls["border_line_orientation_entropy"]
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=0)
    ap.add_argument("--budget-seconds", type=float, default=520.0)
    args = ap.parse_args()

    rows = read_csv(FRAMES64)
    by_view = defaultdict(list)
    for r in rows:
        by_view[(r["sequence"], r["camera"])].append(r)
    views = sorted(by_view)
    if args.count:
        views = views[args.start:args.start + args.count]

    detector, kind = make_line_detector()
    parts = RAW / "_feature_parts"
    parts.mkdir(parents=True, exist_ok=True)
    todo = [v for v in views if not (parts / f"{v[0]}__{v[1]}.csv.gz").exists()]
    print(f"{len(views)} views selected, {len(todo)} to do, line detector={kind}",
          flush=True)

    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    done = 0
    for key in todo:
        if time.perf_counter() - t0 > args.budget_seconds:
            print(f"budget reached after {done} views", flush=True)
            break
        rs = sorted(by_view[key], key=lambda r: int(r["grid_pos"]))
        vid = REPO / rs[0]["video"]
        want = {int(r["frame"]): r for r in rs}
        cap = cv2.VideoCapture(str(vid))
        out, i, hi = [], 0, max(want)
        while i <= hi:
            ok, frame = cap.read()
            if not ok:
                break
            if i in want:
                f = frame_features(frame, detector, kind, rng)
                out.append({"sequence": key[0], "camera_id": key[1],
                            "frame_id": i,
                            "grid_pos": int(want[i]["grid_pos"]),
                            **{f"feature_{k}": round(v, 8) for k, v in f.items()}})
            i += 1
        cap.release()
        write_csv(parts / f"{key[0]}__{key[1]}.csv.gz", out)
        done += 1
        el = time.perf_counter() - t0
        print(f"  [{done}/{len(todo)}] {key[0]}/{key[1]} {len(out)} frames "
              f"{el:.0f}s ({el / done:.1f}s/view)", flush=True)


if __name__ == "__main__":
    main()
