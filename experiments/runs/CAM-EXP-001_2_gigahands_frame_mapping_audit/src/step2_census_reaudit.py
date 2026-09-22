"""Step 2 - full QC census, per-camera hand-identity test, and re-audit of the
CAM-EXP-001.1 cases under the confirmed mapping rules.

Answers Q7-Q10: how much of the previous bad set is explained by our own
mapping, what remains, and how the (0,0) pattern should be labelled.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict

import numpy as np

from audit_common import (CAM_EXP_0011, HANDS, RUN_DIR, Sequence, centroid_dist,
                          is_zero_pattern, median_err, read_csv, rel,
                          sequences, write_csv)

log = logging.getLogger("cam-exp-001.2")

# Same bands as CAM-EXP-001.1 so before/after is comparable. These remain
# diagnostic bands, NOT the final quality gate (that is CAM-EXP-001.3).
GOOD_MAX_PX = 10.0
BAD_MIN_PX = 50.0
FRAME_STRIDE = 20          # census stride over chosen frames


def classify(err: float) -> str:
    if not np.isfinite(err):
        return "no_comparison"
    if err <= GOOD_MAX_PX:
        return "good"
    if err >= BAD_MIN_PX:
        return "bad"
    return "medium"


def census() -> tuple:
    """Walk every sequence x camera x frame x hand and record its QC state."""
    rows, swap_rows = [], []
    for seq_dir in sequences():
        s = Sequence(seq_dir)
        cams = s.cameras_2d("left")
        frames = s.union_sorted[::FRAME_STRIDE]
        per_cam_swap = defaultdict(lambda: {"n": 0, "swap": 0})
        for cam_name in cams:
            cam = s.cameras.get(cam_name)
            if cam is None:
                continue
            video_ok = s.video_path(cam_name) is not None
            for f in frames:
                X = {h: s.joints3d(h, f) for h in HANDS}
                G = {h: s.joints2d(h, cam_name, f) for h in HANDS}
                for h in HANDS:
                    oh = "right" if h == "left" else "left"
                    chosen = int(f in s.chosen[h])
                    g = G[h]
                    present = g is not None
                    zero = bool(present and is_zero_pattern(g))
                    err, ncmp = (median_err(cam, X[h], g)
                                 if (present and not zero and chosen) else (float("nan"), 0))
                    d_self = (centroid_dist(cam, X[h], g)
                              if (present and not zero and chosen) else float("nan"))
                    d_other = (centroid_dist(cam, X[oh], g)
                               if (present and not zero and chosen and f in s.chosen[oh])
                               else float("nan"))
                    if not chosen:
                        status, reason = "not_chosen", "frame has no valid 3D pose for this hand"
                    elif not present:
                        status, reason = "missing_2d_row", "no 2D row for this frame"
                    elif zero:
                        status, reason = "invalid_2d_zero", "all 21 joints exactly (0,0)"
                    elif not np.isfinite(err):
                        status, reason = "no_comparison", "too few comparable joints"
                    else:
                        cls = classify(err)
                        swapped = (np.isfinite(d_other) and np.isfinite(d_self)
                                   and d_other < d_self and d_other < 60)
                        if cls == "bad" and swapped:
                            status = "likely_hand_identity_error"
                            reason = ("annotation centroid is closer to the other hand's "
                                      "projection than to its own")
                        elif cls == "bad":
                            status, reason = "bad_unexplained", "large error, not a clean hand swap"
                        else:
                            status, reason = cls, "reprojection agrees with annotation"
                        if cls != "no_comparison":
                            st = per_cam_swap[cam_name]
                            st["n"] += 1
                            st["swap"] += int(bool(swapped))
                    rows.append({
                        "sequence": s.name, "take": s.take, "camera": cam_name,
                        "frame": f, "hand": h, "chosen": chosen,
                        "annotation_present": int(present), "zero_pattern": int(zero),
                        "video_available": int(video_ok),
                        "mapping_valid": 1,       # direct index, verified in step 1
                        "n_compared_joints": ncmp,
                        "reprojection_median_px": round(err, 4) if np.isfinite(err) else "",
                        "own_hand_centroid_px": round(d_self, 2) if np.isfinite(d_self) else "",
                        "opposite_hand_distance_px": round(d_other, 2) if np.isfinite(d_other) else "",
                        "status": status, "reason": reason,
                    })
        for cam_name, st in sorted(per_cam_swap.items()):
            swap_rows.append({
                "sequence": s.name, "camera": cam_name, "n_comparable": st["n"],
                "n_swapped": st["swap"],
                "swap_rate": round(st["swap"] / st["n"], 4) if st["n"] else "",
                "verdict": ("systematic_swap" if st["n"] and st["swap"] / st["n"] > 0.8
                            else "occasional_swap" if st["n"] and st["swap"] / st["n"] > 0.2
                            else "no_systematic_swap"),
            })
    return rows, swap_rows


def reaudit_previous(census_rows: list) -> list:
    """Re-check the CAM-EXP-001.1 representative cases under the audited rules."""
    p = CAM_EXP_0011 / "results" / "raw" / "representative_cases.csv"
    if not p.exists():
        log.warning("previous case list not found: %s", p)
        return []
    prev = read_csv(p)
    out = []
    seq_cache: dict = {}
    for c in prev:
        seq_take = c["sequence"]
        seq_name = seq_take.split("/")[0]
        if seq_name not in seq_cache:
            seq_cache[seq_name] = Sequence(
                [d for d in sequences() if d.name == seq_name][0])
        s = seq_cache[seq_name]
        cam_name, frame, hand = c["camera"], int(c["frame"]), c["hand"]
        cam = s.cameras.get(cam_name)
        old_class = c["view_class"]
        old_err = c["err_median_px"]
        X = s.joints3d(hand, frame)
        g = s.joints2d(hand, cam_name, frame)
        chosen = int(frame in s.chosen[hand])
        oh = "right" if hand == "left" else "left"
        zero = bool(g is not None and is_zero_pattern(g))
        new_err, _ = (median_err(cam, X, g) if (g is not None and not zero and chosen)
                      else (float("nan"), 0))
        d_self = centroid_dist(cam, X, g) if (g is not None and not zero) else float("nan")
        d_other = (centroid_dist(cam, s.joints3d(oh, frame), g)
                   if (g is not None and not zero and frame in s.chosen[oh]) else float("nan"))

        if not chosen:
            new_class, reason = "NOT_CHOSEN_FRAME", "frame is not in chosen_frames for this hand"
        elif zero:
            new_class, reason = "INVALID_OR_MISSING_2D", "all 21 joints exactly (0,0)"
        elif not np.isfinite(new_err):
            new_class, reason = "UNRESOLVED", "insufficient comparable joints"
        elif new_err <= GOOD_MAX_PX:
            new_class, reason = "GOOD_CONFIRMED", "reprojection agrees under the audited mapping"
        elif new_err >= BAD_MIN_PX and np.isfinite(d_other) and d_other < d_self and d_other < 60:
            new_class, reason = ("LIKELY_HAND_IDENTITY_ERROR",
                                 "annotation sits on the other hand; mapping verified correct")
        elif new_err >= BAD_MIN_PX:
            new_class, reason = "UNRESOLVED", "large error with no clean hand-swap explanation"
        else:
            new_class, reason = "MEDIUM_CONFIRMED", "moderate disagreement"

        # Our index mapping is unchanged by this audit, so nothing was "fixed"
        # by re-indexing; record that explicitly rather than implying a repair.
        mapping_changed = 0
        out.append({
            "sequence": seq_take, "camera": cam_name, "frame": frame, "hand": hand,
            "role": c.get("role", ""), "old_class": old_class, "new_class": new_class,
            "old_error_px": old_err,
            "new_error_px": round(new_err, 4) if np.isfinite(new_err) else "",
            "chosen_frame": chosen,
            "opposite_hand_distance_px": round(d_other, 2) if np.isfinite(d_other) else "",
            "mapping_changed": mapping_changed, "reason": reason,
        })
    return out


def main() -> dict:
    log.info("running full QC census (stride=%d frames)", FRAME_STRIDE)
    rows, swap_rows = census()
    write_csv(RUN_DIR / "results" / "raw" / "gigahands_demo_qc_census.csv.gz", rows, gzipped=True)
    write_csv(RUN_DIR / "tables" / "per_camera_hand_identity.csv", swap_rows)

    status_counts = defaultdict(int)
    for r in rows:
        status_counts[r["status"]] += 1
    total = len(rows)
    write_csv(RUN_DIR / "results" / "summary" / "census_status_summary.csv",
              [{"status": k, "n": v, "share": round(v / total, 4)}
               for k, v in sorted(status_counts.items(), key=lambda kv: -kv[1])])

    reaud = reaudit_previous(rows)
    write_csv(RUN_DIR / "results" / "raw" / "previous_cases_reaudit.csv", reaud)

    # before / after comparison against CAM-EXP-001.1 class sizes
    before = {}
    p = CAM_EXP_0011 / "results" / "summary" / "good_vs_bad_summary.csv"
    if p.exists():
        for r in read_csv(p):
            before[r["view_class"]] = int(r["n_views"])
    after = {"good": status_counts["good"], "medium": status_counts["medium"],
             "bad": status_counts["bad_unexplained"] + status_counts["likely_hand_identity_error"],
             "invalid_2d_zero": status_counts["invalid_2d_zero"],
             "likely_hand_identity_error": status_counts["likely_hand_identity_error"],
             "unresolved": status_counts["bad_unexplained"],
             "not_chosen": status_counts["not_chosen"]}
    ba = [{"metric": k, "cam_exp_001_1": before.get(k, ""), "cam_exp_001_2": v,
           "note": "diagnostic bands, not the final quality gate"}
          for k, v in after.items()]
    ba.append({"metric": "zero_sentinel / invalid_2d_zero",
               "cam_exp_001_1": before.get("zero_sentinel", ""),
               "cam_exp_001_2": status_counts["invalid_2d_zero"],
               "note": "different sampling: 001.1 used the CAM-EXP-001 frame subset, "
                       "001.2 strides the full chosen-frame range"})
    write_csv(RUN_DIR / "tables" / "before_after_audit.csv", ba)

    verdict_rows = []
    for r in swap_rows:
        if r["verdict"] != "no_systematic_swap":
            verdict_rows.append(r)
    write_csv(RUN_DIR / "results" / "summary" / "systematic_swap_cameras.csv", verdict_rows)

    res = {"census_rows": total, "status_counts": dict(status_counts),
           "reaudited_cases": len(reaud),
           "systematic_swap_cameras": [r["camera"] for r in verdict_rows
                                       if r["verdict"] == "systematic_swap"]}
    (RUN_DIR / "results" / "summary" / "_census_verdicts.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    log.info("census: %s", dict(status_counts))
    return res


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
