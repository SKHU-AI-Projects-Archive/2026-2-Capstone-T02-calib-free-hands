"""SUPPLEMENTARY_STATIC_STRESS_TEST - run official AnyCam on static clips.

This is NOT a benchmark and AnyCam is not being ranked against the single-frame
models. The question is narrow and mechanical: what does the official
pretrained model, run through its official entry point, actually do when the
camera never moves?

Two conditions per clip:
  A_REAL_STATIC_VIDEO  - a contiguous run of frames from one static camera,
                         with hands and objects moving in the scene
  B_SAME_FRAME_REPEATED - the first frame of the same clip, duplicated to the
                         same length; a scene with literally zero motion

If A behaves like B, then the moving content is contributing nothing and the
output is whatever the model's prior says.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from common import FRAMES64_CSV, REPO, RUN_DIR, read_csv, write_csv, write_json  # noqa: E402

ANYCAM_DIR = REPO / "experiments" / "cache" / "external_models" / "anycam"
CLIP_LEN = 32
N_CLIPS = 16
SEED = 20260923
OUT = RUN_DIR / "results" / "raw"


def pick_clips():
    """Pre-registered clip list: stratified over sequences, random within."""
    rows = read_csv(FRAMES64_CSV)
    views = sorted({(r["sequence"], r["camera"]) for r in rows})
    by_seq = {}
    for v in views:
        by_seq.setdefault(v[0], []).append(v)
    rng = np.random.default_rng(SEED)
    picked = []
    seqs = sorted(by_seq)
    i = 0
    order = {s: [by_seq[s][j] for j in rng.permutation(len(by_seq[s]))] for s in seqs}
    while len(picked) < N_CLIPS:
        for s in seqs:
            if i < len(order[s]) and len(picked) < N_CLIPS:
                picked.append(order[s][i])
        i += 1
    meta = {(r["sequence"], r["camera"]): r for r in rows}
    return [{"sequence": k[0], "camera": k[1],
             "video": meta[k]["video"], "gt_fx": float(meta[k]["gt_fx"]),
             "gt_fy": float(meta[k]["gt_fy"]),
             "image_width": int(meta[k]["image_width"]),
             "image_height": int(meta[k]["image_height"])} for k in picked]


def load_clip(video: Path, n: int, start: int = 0):
    import cv2
    cap = cv2.VideoCapture(str(video))
    frames, i = [], 0
    while len(frames) < n:
        ok, fr = cap.read()
        if not ok:
            break
        if i >= start:
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        i += 1
    cap.release()
    return frames


def focal_from_proj(proj, width: int, height: int):
    """AnyCam's own convention, from fit_video.py:

        gt_normalized_focal = gt_proj[0, 0] / w_ * 2

    so the reported projection entry is 2 * f_px / w. Both the full-width and
    the square-crop reading are recorded, because the default config sets
    `square_crop: True` and the paper does not state which the returned matrix
    refers to.
    """
    p = np.asarray(proj, dtype=float).reshape(-1)
    p00 = float(p[0]) if p.size == 4 else float(np.asarray(proj)[0][0])
    return {
        "proj_00": p00,
        "focal_px_if_full_width": p00 * width / 2.0,
        "focal_px_if_square_crop": p00 * min(width, height) / 2.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=N_CLIPS)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--frame-counts", default="32")
    ap.add_argument("--ba", action="store_true",
                    help="enable bundle-adjustment refinement (official default)")
    args = ap.parse_args()

    sys.path.insert(0, str(ANYCAM_DIR))
    import torch
    torch.hub.set_dir(str(Path.home() / ".cache" / "torch" / "hub"))
    model = torch.hub.load(str(ANYCAM_DIR), "AnyCam", source="local",
                           version="1.0", training_variant="seq8", pretrained=True)
    model = model.cuda() if torch.cuda.is_available() else model

    clips = pick_clips()[:args.limit]
    counts = [int(x) for x in args.frame_counts.split(",")]
    rows, diag = [], []
    for ci, c in enumerate(clips):
        base = load_clip(REPO / c["video"], max(counts))
        if len(base) < max(counts):
            rows.append({**c, "condition": "LOAD_FAILED", "success": 0,
                         "failure_reason": f"only {len(base)} frames decoded"})
            continue
        for n in counts:
            for cond, frames in (("A_REAL_STATIC_VIDEO", base[:n]),
                                 ("B_SAME_FRAME_REPEATED", [base[0]] * n)):
                for rep in range(args.repeats):
                    t0 = time.perf_counter()
                    try:
                        res = model.process_video(frames, ba_refinement=args.ba)
                        proj = res["projection_matrix"]
                        if hasattr(proj, "detach"):
                            proj = proj.detach().cpu().numpy()
                        f = focal_from_proj(proj, c["image_width"], c["image_height"])
                        traj = res.get("trajectory")
                        if hasattr(traj, "detach"):
                            traj = traj.detach().cpu().numpy()
                        traj = np.asarray(traj)
                        # how much camera motion did it think there was?
                        if traj.ndim == 3 and traj.shape[-1] == 4:
                            tr = traj[:, :3, 3]
                            motion = float(np.linalg.norm(tr - tr[0], axis=1).max())
                        else:
                            motion = float("nan")
                        row = {**c, "condition": cond, "n_frames": n, "repeat": rep,
                               "ba_refinement": int(args.ba), "success": 1,
                               **f,
                               "rel_err_full_width_pct":
                                   abs(f["focal_px_if_full_width"] - c["gt_fx"]) / c["gt_fx"] * 100,
                               "rel_err_square_crop_pct":
                                   abs(f["focal_px_if_square_crop"] - c["gt_fx"]) / c["gt_fx"] * 100,
                               "max_translation_norm": motion,
                               "runtime_s": time.perf_counter() - t0,
                               "status": "SUPPLEMENTARY_STATIC_STRESS_TEST"}
                        rows.append(row)
                        ex = res.get("extras") or {}
                        keys = list(ex.keys()) if isinstance(ex, dict) else []
                        diag.append({**c, "condition": cond, "n_frames": n,
                                     "repeat": rep, "extras_keys": ";".join(map(str, keys)),
                                     "proj_00": f["proj_00"],
                                     "result_keys": ";".join(map(str, res.keys()))})
                    except Exception as e:  # noqa: BLE001 - recorded, not hidden
                        rows.append({**c, "condition": cond, "n_frames": n,
                                     "repeat": rep, "success": 0,
                                     "failure_reason": f"{type(e).__name__}: {e}"})
                    print(f"  [{ci+1}/{len(clips)}] {c['sequence']}/{c['camera']} "
                          f"{cond} n={n} rep={rep} -> {rows[-1].get('proj_00')}",
                          flush=True)
    write_csv(OUT / "anycam_static_predictions.csv.gz", rows)
    write_csv(OUT / "anycam_candidate_diagnostics.csv.gz", diag)
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
