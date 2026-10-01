"""PHASE C — cross-fit eligibility, primary video set, folds, global grid.

Everything here is frozen BEFORE any test reference focal is opened. The scene
predictions are read through a focal-blind reader, and the global focal grid is
derived from the scene estimates alone.

Eligibility is structural only: verified RGB, three scene aggregates, and
enough hand observations in BOTH temporal folds on BOTH sides. Nothing is
excluded for performance.
"""
from __future__ import annotations

import itertools
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from c012_common import (BLOCK_SIZE, C011, C011_RAW, DISPLAY_SEQUENCE,  # noqa
                         GLOBAL_GRID_N, LAMBDA_GRID, MANIFESTS,
                         MIN_OBS_PER_FOLD_PER_SIDE, MODELS, N_INNER_FOLDS,
                         N_OUTER_FOLDS, PARTICIPANT_OF, Q_MAX, Q_MIN,
                         SCENE_PRED, SIGMA_SCENE_FLOOR, SUM, TAB,
                         aggregate_video_focal, fnum, global_focal_grid,
                         hash_bucket, q_grid, read_csv, read_focal_blind,
                         scalar_focal, sha256, temporal_fold, write_csv,
                         write_json)
from hand_io import load_video_observations  # noqa: E402


def main():
    # ---------------- usable videos from CAM-EXP-011's verified set
    cams = read_csv(MANIFESTS / "cam_exp_011_usable_cameras_v1.csv")
    videos = sorted({(r["sequence"], r["camera"])
                     for r in cams if r["usable"] == "1"})

    # ---------------- scene aggregates, focal-blind
    scene = {}
    for m, path in SCENE_PRED.items():
        by = defaultdict(list)
        for r in read_focal_blind(path):
            if r["valid"] == "1":
                by[(r["sequence"], r["camera"])].append(
                    fnum(r["focal_pred_px"]))
        agg = {}
        for k, fs in by.items():
            f = aggregate_video_focal(fs)
            lf = np.log([x for x in fs if np.isfinite(x) and x > 0])
            sigma = (max(float(np.median(np.abs(lf - np.median(lf)))) * 1.4826,
                         SIGMA_SCENE_FLOOR) if lf.size else SIGMA_SCENE_FLOOR)
            spread = (100.0 * float(np.percentile(lf, 90)
                                    - np.percentile(lf, 10))
                      if lf.size >= 10 else np.nan)
            agg[k] = {"f_scene": f, "sigma": sigma, "n": len(fs),
                      "spread_pct": spread}
        scene[m] = agg

    # ---------------- hand observations and temporal folds
    obs = read_csv(C011_RAW / "hand_observations.csv.gz")
    frames_by_video = defaultdict(list)
    for r in obs:
        if r["hand_any_available"] == "1":
            frames_by_video[(r["sequence"], r["camera"])].append(
                int(r["frame"]))

    rows, split_rows = [], []
    for (seq, cam) in videos:
        frames = sorted(frames_by_video.get((seq, cam), []))
        counts = {("left", "A"): 0, ("left", "B"): 0,
                  ("right", "A"): 0, ("right", "B"): 0}
        if frames:
            _, idx, W, H = load_video_observations(seq, cam, frames)
            for side in ("left", "right"):
                for f in idx[side]:
                    counts[(side, temporal_fold(f))] += 1
                    split_rows.append({"sequence": seq, "camera": cam,
                                       "hand": side, "frame": f,
                                       "block_id": f // BLOCK_SIZE,
                                       "fold": temporal_fold(f)})
        have_scene = all((seq, cam) in scene[m]
                         and np.isfinite(scene[m][(seq, cam)]["f_scene"])
                         for m in MODELS)
        fold_ok = all(v >= MIN_OBS_PER_FOLD_PER_SIDE for v in counts.values())
        eligible = bool(have_scene and fold_ok)
        reason = ""
        if not eligible:
            if not frames:
                reason = "NO_USABLE_HAND"
            elif not have_scene:
                reason = "MISSING_SCENE_AGGREGATE"
            else:
                bad = [k for k, v in counts.items()
                       if v < MIN_OBS_PER_FOLD_PER_SIDE]
                sides = {s for s, _ in bad}
                reason = ("INSUFFICIENT_BOTH_SIDES_IN_A_FOLD"
                          if sides == {"left", "right"}
                          else "INSUFFICIENT_%s_IN_A_FOLD" % list(sides)[0].upper())
        rec = {"sequence": seq, "display": DISPLAY_SEQUENCE.get(seq, seq),
               "participant": PARTICIPANT_OF.get(seq, ""), "camera": cam,
               "n_hand_frames": len(frames),
               "left_fold_A": counts[("left", "A")],
               "left_fold_B": counts[("left", "B")],
               "right_fold_A": counts[("right", "A")],
               "right_fold_B": counts[("right", "B")],
               "has_all_scene_aggregates": int(have_scene),
               "crossfit_eligible": int(eligible),
               "primary_set": int(eligible),
               "exclusion_reason": reason}
        for m in MODELS:
            a = scene[m].get((seq, cam), {})
            rec["f_scene_%s" % m] = a.get("f_scene", np.nan)
            rec["sigma_%s" % m] = a.get("sigma", np.nan)
            rec["spread_pct_%s" % m] = a.get("spread_pct", np.nan)
        rows.append(rec)

    primary = [r for r in rows if r["primary_set"] == 1]
    write_csv(TAB / "primary_video_set.csv", rows)
    write_csv(TAB / "paired_set_exclusions.csv",
              [r for r in rows if r["primary_set"] == 0])
    split_p = MANIFESTS / "cam_exp_012_temporal_crossfit_v1.csv.gz"
    write_csv(split_p, split_rows)
    prim_p = MANIFESTS / "cam_exp_012_primary_video_set_v1.csv"
    write_csv(prim_p, [{"sequence": r["sequence"], "camera": r["camera"],
                        "participant": r["participant"]} for r in primary])

    # ---------------- global absolute focal grid, from scene estimates only
    lo, hi = [], []
    for r in primary:
        for m in MODELS:
            f = r["f_scene_%s" % m]
            if np.isfinite(f) and f > 0:
                lo.append(Q_MIN * f)
                hi.append(Q_MAX * f)
    f_min, f_max = float(min(lo)), float(max(hi))
    grid = global_focal_grid(f_min, f_max, GLOBAL_GRID_N)
    grid_p = MANIFESTS / "cam_exp_012_global_focal_grid_v1.json"
    write_json(grid_p, {
        "f_min": f_min, "f_max": f_max, "n_points": GLOBAL_GRID_N,
        "spacing": "log", "derived_from": "0.5*f_scene and 1.5*f_scene over "
                                          "the primary videos and all three "
                                          "scene models",
        "reference_focal_used": False,
        "identical_for_every_video": True,
        "grid": [float(x) for x in grid],
    })

    q = q_grid()
    q_p = MANIFESTS / "cam_exp_012_q_grid_v1.json"
    write_json(q_p, {"n": len(q), "q_min": Q_MIN, "q_max": Q_MAX,
                     "contains_exact_one": bool((q == 1.0).any()),
                     "construction": "two log-spaced halves meeting at exactly "
                                     "1.0, duplicate removed",
                     "q": [float(x) for x in q]})

    # ---------------- folds: reuse CAM-EXP-011's outer camera folds
    outer = {r["camera"]: int(r["outer_fold"])
             for r in read_csv(MANIFESTS
                               / "cam_exp_011_outer_camera_folds_v1.csv")}
    cams_p = sorted({r["camera"] for r in primary})
    of_p = MANIFESTS / "cam_exp_012_outer_camera_folds_v1.csv"
    write_csv(of_p, [{"camera": c, "outer_fold": outer.get(
        c, hash_bucket(c, N_OUTER_FOLDS))} for c in cams_p])
    if_p = MANIFESTS / "cam_exp_012_inner_camera_folds_v1.csv"
    write_csv(if_p, [{"camera": c,
                      "outer_fold": outer.get(c, hash_bucket(c, N_OUTER_FOLDS)),
                      "inner_fold": hash_bucket("CAM012_INNER|" + c,
                                                N_INNER_FOLDS)}
                     for c in cams_p])
    lam_p = MANIFESTS / "cam_exp_012_lambda_grid_v1.json"
    write_json(lam_p, {
        "lambda_grid": list(LAMBDA_GRID),
        "lambda_zero_is_a_candidate": False,
        "why": "Hand ON must contribute. lambda = 0 exists only as an "
               "implementation sanity test, never as a tuning candidate; if "
               "every positive lambda is worse on TRAIN, the least-bad "
               "positive lambda is selected rather than falling back to 0.",
        "frozen_before_test_focal_results": True})

    hashes = {p.name: sha256(p) for p in (split_p, prim_p, grid_p, q_p, of_p,
                                          if_p, lam_p)}
    write_json(SUM / "manifest_hashes.json", hashes)

    per_seq = Counter(r["display"] for r in primary)
    excl = Counter(r["exclusion_reason"] for r in rows
                   if r["primary_set"] == 0)
    write_json(SUM / "crossfit_audit.json", {
        "block_size": BLOCK_SIZE,
        "min_obs_per_fold_per_side": MIN_OBS_PER_FOLD_PER_SIDE,
        "usable_videos": len(videos),
        "primary_videos": len(primary),
        "excluded": len(rows) - len(primary),
        "exclusion_counts": dict(excl),
        "per_sequence_primary": dict(per_seq),
        "checks": {"primary_plus_excluded_equals_usable":
                   len(primary) + (len(rows) - len(primary)) == len(videos)},
        "global_grid": {"f_min": f_min, "f_max": f_max, "n": GLOBAL_GRID_N},
        "q_grid_n": len(q), "q_contains_exact_one": bool((q == 1.0).any()),
    })

    print("usable %d -> primary %d (excluded %d)"
          % (len(videos), len(primary), len(videos) - len(primary)))
    for k, v in sorted(excl.items(), key=lambda z: -z[1]):
        print("   %-40s %d" % (k, v))
    print("per sequence:", dict(per_seq))
    print("global focal grid: %.1f - %.1f px, %d points"
          % (f_min, f_max, GLOBAL_GRID_N))
    print("q grid: %d points, exact 1.0 present: %s"
          % (len(q), bool((q == 1.0).any())))
    for k, v in hashes.items():
        print("  %-46s %s" % (k, v[:16]))


if __name__ == "__main__":
    main()
