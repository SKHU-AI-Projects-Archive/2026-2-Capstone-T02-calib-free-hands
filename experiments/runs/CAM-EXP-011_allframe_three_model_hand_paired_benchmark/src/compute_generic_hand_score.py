"""PHASE B — the generic hand-structure score, computed ONCE per video.

Definition inherited from CAM-EXP-009.4's `M1_SCENE_GENERIC_INDEPENDENT`:
neutral MANO generic prior, 20 connected bones, absolute size removed, WiLoR 2D
observations and unit bone directions, network bone LENGTHS discarded, LEFT and
RIGHT anatomy fitted independently. CAM-EXP-009.4's shared-anatomy M3 term is
NOT used — that run measured its contribution as exactly 0.000 pp.

Fairness. The score is evaluated on an **absolute focal grid**, not on a grid
anchored to any one model's estimate. So the hand evidence for a video is a
single function of absolute focal, and `AnyCalib + Hand`, `GeoCalib + Hand` and
`Perspective Fields + Hand` all read the same function. Only the scene term
differs between them. Computing it once also avoids three redundant solves.

All hand-available frames of the video are used; there is no 8/16/32 sampling.

The reference focal is never read here.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from c011_common import (CACHE, MANIFESTS, MIN_HAND_FRAMES_PER_SIDE,  # noqa: E402
                    N_BONES, RAW, RUNS, SUM, read_csv, read_json, write_csv,
                    write_json)

C0094_SRC = RUNS / "CAM-EXP-009_4_scene_shared_anatomy_focal_refinement" / "src"
sys.path.insert(0, str(C0094_SRC))

WILOR = CACHE / "wilor"
OUT = CACHE / "hand_scores"

# Absolute candidate focal grid (px). Wide enough to contain every model's
# q in [0.5, 1.5] window; frozen before any focal result.
F_MIN, F_MAX, F_N = 400.0, 2400.0, 81


def focal_grid():
    return np.geomspace(F_MIN, F_MAX, F_N)


def load_video_hands(seq, cam, frames):
    """Unit bone directions + 2D, per side. Network bone lengths discarded."""
    from c011_common import BONES
    out = {"left": [], "right": []}
    W = H = None
    for f in frames:
        p = WILOR / ("%s__%s__%06d.npz" % (seq, cam, f))
        if not p.exists():
            continue
        d = np.load(p)
        W, H = int(d["image_width"]), int(d["image_height"])
        for side, want_right in (("left", False), ("right", True)):
            best, bs = None, -1.0
            for j in range(int(d["n_hands"])):
                if bool(int(d["h%d_is_right" % j])) != want_right:
                    continue
                s = float(d["h%d_score" % j])
                if s > bs:
                    best, bs = j, s
            if best is None:
                continue
            kp3 = np.asarray(d["h%d_keypoints_3d" % best], float)
            uv = np.asarray(d["h%d_keypoints_2d" % best], float)
            dirs = np.zeros((N_BONES, 3))
            ok = np.zeros(N_BONES, bool)
            for i, (a, b) in enumerate(BONES):
                v = kp3[b] - kp3[a]
                n = np.linalg.norm(v)
                if np.isfinite(n) and n > 1e-9:
                    dirs[i] = v / n
                    ok[i] = True
            use = np.isfinite(uv).all(1)
            if ok.all() and use.sum() >= 12:
                out[side].append((dirs, uv, use))
    return out, W, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-frames-per-side", type=int, default=0,
                    help="0 = use every hand-available frame (default)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    from joint_anatomy_solver import fit_unit, score_frames

    prior = read_json(MANIFESTS
                      / "cam_exp_0094_generic_mano_prior_v1.json")
    p0 = np.array(prior["p0_generic_prior"], float)
    spec = read_json(MANIFESTS / "cam_exp_0094_synthetic_spec_v1.json")
    lam_g, lam_s = spec["lambda_generic"], spec["lambda_side"]

    obs = read_csv(RAW / "hand_observations.csv.gz")
    by_video = defaultdict(list)
    for r in obs:
        if r["hand_any_available"] == "1":
            by_video[(r["sequence"], r["camera"])].append(int(r["frame"]))

    OUT.mkdir(parents=True, exist_ok=True)
    fs = focal_grid()
    videos = sorted(by_video)
    if args.limit:
        videos = videos[:args.limit]
    # Sharding is pure parallelism over videos: each worker solves a disjoint
    # subset with identical code and identical inputs. It changes no frame, no
    # candidate and no parameter - it is the answer to runtime that the spec
    # asks for instead of reducing frames.
    if args.nshards > 1:
        videos = [v for i, v in enumerate(videos)
                  if i % args.nshards == args.shard]

    rows, t0 = [], time.time()
    for vi, (seq, cam) in enumerate(videos, 1):
        outp = OUT / ("%s__%s.npz" % (seq, cam))
        if outp.exists():
            continue
        frames = sorted(by_video[(seq, cam)])
        hands, W, H = load_video_hands(seq, cam, frames)
        nL, nR = len(hands["left"]), len(hands["right"])
        if nL < MIN_HAND_FRAMES_PER_SIDE or nR < MIN_HAND_FRAMES_PER_SIDE:
            np.savez_compressed(outp, f=fs, score=np.full(len(fs), np.nan),
                                n_left=nL, n_right=nR, eligible=0)
            rows.append({"sequence": seq, "camera": cam, "n_left": nL,
                         "n_right": nR, "eligible": 0,
                         "reason": "insufficient_hand_frames"})
            continue
        if args.max_frames_per_side:
            for k in hands:
                hands[k] = hands[k][:args.max_frames_per_side]
        # deterministic FIT/EVAL split by frame parity within the side's list
        fitL, evL = hands["left"][0::2], hands["left"][1::2]
        fitR, evR = hands["right"][0::2], hands["right"][1::2]
        diag = float(np.hypot(W, H))
        curve = np.full(len(fs), np.nan)
        for k, f in enumerate(fs):
            K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])
            fit = fit_unit(fitL, fitR, K, None, p0, lam_g, lam_s,
                           shared=False, use_generic=True)
            if fit is None:
                continue
            sL = score_frames(evL, K, None, fit["p_L"], diag)
            sR = score_frames(evR, K, None, fit["p_R"], diag)
            if np.isfinite(sL) and np.isfinite(sR):
                curve[k] = 0.5 * (sL + sR)
        np.savez_compressed(outp, f=fs, score=curve, n_left=nL, n_right=nR,
                            eligible=1)
        rows.append({"sequence": seq, "camera": cam, "n_left": nL,
                     "n_right": nR, "eligible": 1,
                     "n_valid_candidates": int(np.isfinite(curve).sum()),
                     "reason": ""})
        if vi % 5 == 0 or vi == len(videos):
            el = time.time() - t0
            print("  %3d/%d videos  %.1f min  (%.1f s/video)"
                  % (vi, len(videos), el / 60, el / max(vi, 1)), flush=True)

    idxp = (RAW / ("hand_score_index_shard%02d.csv.gz" % args.shard)
            if args.nshards > 1 else RAW / "hand_score_index.csv.gz")
    write_csv(idxp, rows)
    write_json(SUM / "hand_score_meta.json", {
        "definition": "CAM-EXP-009.4 M1_SCENE_GENERIC_INDEPENDENT, generic "
                      "anatomy, LEFT/RIGHT independent",
        "shared_anatomy_M3_used": False,
        "grid": {"f_min": F_MIN, "f_max": F_MAX, "f_n": F_N,
                 "absolute": True,
                 "why_absolute": "so the hand evidence is one function of "
                                 "absolute focal, identical for all three "
                                 "scene models"},
        "lambda_generic": lam_g, "lambda_side": lam_s,
        "frames_policy": "all hand-available frames of the video",
        "min_frames_per_side": MIN_HAND_FRAMES_PER_SIDE,
        "reference_focal_read": False,
    })
    print("hand score videos written:", len(rows))


if __name__ == "__main__":
    main()
