"""PHASE C — video-level scene focals, common sets, and the six conditions.

Steps, all with the reference focal CLOSED:

  1. per-video scene focal for each model: exp(median(log f_i)) over that
     video's valid frames, on two frame sets
        STRICT  — frames where ALL THREE models succeeded (primary)
        NATIVE  — every frame the model itself succeeded on (secondary)
  2. the fused `+ Hand` estimate, over a q in [0.5, 1.5] window anchored on
     that model's own scene focal, reading the SAME absolute-focal hand score
  3. the six-condition table on the globally paired video set

The `+ Hand` condition keeps the entire scene evidence of the `scene-only`
condition and only adds hand evidence. Frames without a usable hand are never
removed from the scene term.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from c011_common import (CACHE, LAMBDA_GRID, MANIFESTS, MIN_SCENE_COVERAGE,  # noqa
                         q_grid,
                         MODELS, PARTICIPANT_OF, Q_MAX, Q_MIN, Q_N, RAW,
                         SIGMA_SCENE_FLOOR, SUM, TAB, aggregate_video_focal,
                         fnum, hash_bucket, med, read_csv, write_csv,
                         write_json)

PRED = {"ANYCALIB": "anycalib_allframe_predictions.csv.gz",
        "GEOCALIB": "geocalib_allframe_predictions.csv.gz",
        "PERSPECTIVE_FIELDS": "perspective_fields_allframe_predictions.csv.gz"}


def load_predictions():
    out = {}
    for m, fn in PRED.items():
        p = RAW / fn
        if not p.exists():
            print("MISSING:", fn)
            continue
        by = defaultdict(dict)
        for r in read_csv(p):
            by[(r["sequence"], r["camera"])][int(r["frame"])] = (
                fnum(r["focal_pred_px"]) if r["valid"] == "1" else np.nan)
        out[m] = by
    return out


def load_hand_scores():
    out = {}
    for p in sorted((CACHE / "hand_scores").glob("*.npz")):
        seq, cam = p.stem.split("__", 1)
        d = np.load(p)
        if int(d["eligible"]) != 1:
            continue
        out[(seq, cam)] = (d["f"], d["score"], int(d["n_left"]),
                           int(d["n_right"]))
    return out


def fuse(f_scene, sigma, hand_f, hand_s, lam):
    """Fused focal. Scene term in log domain; hand term on the same absolute
    grid, rescaled by its own fixed median so lambda means the same thing
    everywhere (no candidate-dependent min-max rescaling)."""
    qs = q_grid()
    cand = f_scene * qs
    ok = np.isfinite(hand_s)
    if ok.sum() < 5 or not np.isfinite(f_scene) or f_scene <= 0:
        return float("nan"), 1
    hs_raw = np.interp(cand, hand_f[ok], hand_s[ok],
                       left=np.nan, right=np.nan)
    # Normalise by the median WITHIN the candidate window, matching
    # CAM-EXP-009.4. Normalising by the median over the whole absolute grid
    # (which spans ~12x in score) would crush the local variation the fusion
    # actually sees and make the hand term inert - measured at 0 % of videos
    # moved for lambda <= 0.5 before this fix.
    if not np.isfinite(hs_raw).any():
        return float("nan"), 1
    scale = float(np.nanmedian(hs_raw))
    if not np.isfinite(scale) or scale <= 0:
        return float("nan"), 1
    hs = hs_raw / scale
    L_scene = (np.log(qs) / sigma) ** 2
    tot = np.where(np.isfinite(hs), L_scene + lam * hs, np.inf)
    if not np.isfinite(tot).any():
        return float("nan"), 1
    j = int(np.argmin(tot))
    return float(cand[j]), int(j == 0 or j == len(qs) - 1)


def main():
    preds = load_predictions()
    if len(preds) < 3:
        raise SystemExit("all three model prediction files are required")
    hands = load_hand_scores()

    man = [r for r in read_csv(MANIFESTS / "cam_exp_011_allframe_manifest_v1.csv.gz")
           if r["scene_input"] == "1"]
    by_video = defaultdict(set)
    for r in man:
        by_video[(r["sequence"], r["camera"])].add(int(r["frame"]))
    videos = sorted(by_video)

    # ---------------- frame sets
    cov_rows, scene_rows = [], []
    for (seq, cam) in videos:
        frames = by_video[(seq, cam)]
        valid = {}
        for m in MODELS:
            d = preds[m].get((seq, cam), {})
            valid[m] = {f for f in frames
                        if f in d and np.isfinite(d[f]) and d[f] > 0}
        strict = set.intersection(*valid.values()) if valid else set()
        rec = {"sequence": seq, "camera": cam,
               "participant": PARTICIPANT_OF.get(seq, ""),
               "n_scene_input_frames": len(frames),
               "n_three_model_common": len(strict)}
        for m in MODELS:
            rec["n_valid_%s" % m] = len(valid[m])
            rec["coverage_%s" % m] = len(valid[m]) / max(len(frames), 1)
        rec["strict_coverage"] = len(strict) / max(len(frames), 1)
        hv = hands.get((seq, cam))
        rec["hand_eligible"] = int(hv is not None)
        rec["n_hand_left"] = hv[2] if hv else 0
        rec["n_hand_right"] = hv[3] if hv else 0
        cov_rows.append(rec)

        for m in MODELS:
            for setname, fset in (("STRICT", strict), ("NATIVE", valid[m])):
                d = preds[m].get((seq, cam), {})
                fs = [d[f] for f in sorted(fset)]
                f_video = aggregate_video_focal(fs)
                lf = np.log([x for x in fs if np.isfinite(x) and x > 0]) \
                    if fs else np.array([])
                sigma = (max(float(np.median(np.abs(lf - np.median(lf))))
                             * 1.4826, SIGMA_SCENE_FLOOR)
                         if lf.size else SIGMA_SCENE_FLOOR)
                spread = (100.0 * float(np.percentile(lf, 90)
                                        - np.percentile(lf, 10))
                          if lf.size >= 10 else np.nan)
                scene_rows.append({
                    "sequence": seq, "camera": cam,
                    "participant": PARTICIPANT_OF.get(seq, ""),
                    "model": m, "frame_set": setname,
                    "n_frames_used": len(fs),
                    "coverage": len(fs) / max(len(frames), 1),
                    "f_scene": f_video, "sigma_scene": sigma,
                    "within_video_spread_pct": spread,
                    "meets_min_coverage":
                        int(len(fs) / max(len(frames), 1) >= MIN_SCENE_COVERAGE),
                })
    write_csv(RAW / "video_frame_coverage.csv.gz", cov_rows)
    write_csv(RAW / "scene_video_predictions.csv.gz", scene_rows)

    # ---------------- six conditions at every lambda (tuned later)
    six = []
    sc = {(r["sequence"], r["camera"], r["model"], r["frame_set"]): r
          for r in scene_rows}
    for (seq, cam) in videos:
        hv = hands.get((seq, cam))
        for m in MODELS:
            for setname in ("STRICT", "NATIVE"):
                r = sc.get((seq, cam, m, setname))
                if r is None or not np.isfinite(r["f_scene"]):
                    continue
                base = {"sequence": seq, "camera": cam,
                        "participant": PARTICIPANT_OF.get(seq, ""),
                        "model": m, "frame_set": setname,
                        "n_frames_used": r["n_frames_used"],
                        "coverage": r["coverage"],
                        "meets_min_coverage": r["meets_min_coverage"],
                        "within_video_spread_pct":
                            r["within_video_spread_pct"],
                        "f_scene": r["f_scene"]}
                six.append({**base, "condition": "SCENE_ONLY", "lambda": "",
                            "f_pred": r["f_scene"], "boundary": 0,
                            "hand_eligible": int(hv is not None)})
                if hv is None:
                    continue
                for lam in LAMBDA_GRID:
                    f, b = fuse(r["f_scene"], r["sigma_scene"], hv[0], hv[1],
                                lam)
                    six.append({**base, "condition": "SCENE_PLUS_HAND",
                                "lambda": lam, "f_pred": f, "boundary": b,
                                "hand_eligible": 1})
    write_csv(RAW / "six_condition_candidates.csv.gz", six)

    write_json(SUM / "inference_coverage_summary.json", {
        "videos": len(videos),
        "scene_input_frames": sum(len(v) for v in by_video.values()),
        "per_model_valid_frames": {m: sum(r["n_valid_%s" % m]
                                          for r in cov_rows) for m in MODELS},
        "three_model_common_frames": sum(r["n_three_model_common"]
                                         for r in cov_rows),
        "videos_with_hand_score": sum(r["hand_eligible"] for r in cov_rows),
        "min_scene_coverage_rule": MIN_SCENE_COVERAGE,
    })
    print("videos %d | scene frames %d | 3-model common %d | hand-eligible %d"
          % (len(videos), sum(len(v) for v in by_video.values()),
             sum(r["n_three_model_common"] for r in cov_rows),
             sum(r["hand_eligible"] for r in cov_rows)))
    for m in MODELS:
        print("  %-20s valid frames %d" % (m, sum(r["n_valid_%s" % m]
                                                  for r in cov_rows)))


if __name__ == "__main__":
    main()
