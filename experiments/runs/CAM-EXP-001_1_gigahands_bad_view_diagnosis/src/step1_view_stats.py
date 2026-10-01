"""Step 1 - per-view / per-frame statistics, good-bad definition, case picking.

Reads the CAM-EXP-001 raw per-joint CSV (read-only) and writes the view-level
tables plus the representative-case list that the figures are built from.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict

import numpy as np

from bad_view_common import (RUN_DIR, VIEW_KEYS, load_gigahands_views,
                             otsu_threshold_log, rel, stats, write_csv)

log = logging.getLogger("cam-exp-001.1")


def build(views: dict):
    """Flatten view records into rows, tagging the zero-sentinel views."""
    rows = []
    for key, v in views.items():
        seq, cam, frame, hand = key
        st = stats(v["errors"])
        row = {"sequence": seq, "camera": cam, "frame": frame, "hand": hand,
               "n_valid_joints": v["n_valid"],
               "n_zero_sentinel_joints": v["n_zero"],
               "n_compared_joints": v["n_compared"],
               "zero_sentinel_ratio": round(v["n_zero"] / max(v["n_valid"], 1), 4),
               "is_zero_sentinel_view": int(v["n_zero"] == v["n_valid"] and v["n_valid"] > 0),
               "image_width": v["image_width"], "image_height": v["image_height"],
               "distortion_applied": v["distortion_applied"],
               **{f"err_{k}": val for k, val in st.items()},
               "err_median_including_sentinel": round(
                   float(np.median(v["errors_all"])), 4) if v["errors_all"] else ""}
        # camera family, e.g. brics-odroid-001_cam0 -> brics-odroid-001
        row["camera_family"] = cam.rsplit("_", 1)[0]
        rows.append(row)
    return rows


def classify(rows: list) -> dict:
    """Derive the good/bad split from the observed distribution.

    Zero-sentinel views are set aside first: they carry no annotation at all,
    so ranking them by "error" would only measure how far the hand happens to
    be from the image origin.
    """
    real = [r for r in rows if not r["is_zero_sentinel_view"] and r["err_median_px"] != ""]
    med = np.array([float(r["err_median_px"]) for r in real])
    thr = otsu_threshold_log(med)
    # A medium band around the threshold keeps genuinely ambiguous views from
    # being presented as clean successes or clean failures.
    good_thr = float(np.round(min(thr / 2.0, 10.0), 3))
    bad_thr = float(np.round(max(thr * 2.0, 50.0), 3))
    for r in rows:
        if r["is_zero_sentinel_view"]:
            r["view_class"] = "zero_sentinel"
            continue
        m = float(r["err_median_px"]) if r["err_median_px"] != "" else float("nan")
        r["view_class"] = ("good" if m <= good_thr
                           else "bad" if m >= bad_thr
                           else "medium")
    counts = defaultdict(int)
    for r in rows:
        counts[r["view_class"]] += 1
    return {"otsu_threshold_px": round(thr, 4),
            "good_max_median_px": good_thr,
            "bad_min_median_px": bad_thr,
            "rule": ("Otsu split on log10(per-view median error) over non-sentinel "
                     "views; good = median <= min(otsu/2, 10), bad = median >= "
                     "max(otsu*2, 50), medium in between, zero-sentinel separate."),
            "counts": dict(counts),
            "n_non_sentinel": len(real)}


def group_table(rows: list, key_fn, label: str) -> list:
    g = defaultdict(list)
    for r in rows:
        g[key_fn(r)].append(r)
    out = []
    for k, rs in sorted(g.items()):
        real = [float(r["err_median_px"]) for r in rs
                if not r["is_zero_sentinel_view"] and r["err_median_px"] != ""]
        n = len(rs)
        cls = defaultdict(int)
        for r in rs:
            cls[r["view_class"]] += 1
        out.append({label: k, "n_views": n,
                    "n_zero_sentinel": cls["zero_sentinel"],
                    "zero_sentinel_rate": round(cls["zero_sentinel"] / n, 4),
                    "n_good": cls["good"], "n_medium": cls["medium"],
                    "n_bad": cls["bad"],
                    "bad_rate_excl_sentinel": round(
                        cls["bad"] / max(n - cls["zero_sentinel"], 1), 4),
                    **{f"err_{kk}": vv for kk, vv in stats(real).items()}})
    return out


def pick_cases(rows: list, n_each: int = 6) -> list:
    """Choose cases a human can inspect, with the reason recorded for each."""
    by_key = {tuple(r[k] for k in VIEW_KEYS): r for r in rows}
    good = sorted([r for r in rows if r["view_class"] == "good"],
                  key=lambda r: float(r["err_median_px"]))
    bad = sorted([r for r in rows if r["view_class"] == "bad"],
                 key=lambda r: -float(r["err_median_px"]))
    sent = [r for r in rows if r["view_class"] == "zero_sentinel"]
    cases = []

    def add(r, role, reason, pair_id=""):
        cases.append({**{k: r[k] for k in VIEW_KEYS},
                      "role": role, "reason": reason, "pair_id": pair_id,
                      "view_class": r["view_class"],
                      "err_median_px": r["err_median_px"],
                      "err_mean_px": r["err_mean_px"],
                      "err_p90_px": r["err_p90_px"],
                      "n_compared_joints": r["n_compared_joints"],
                      "zero_sentinel_ratio": r["zero_sentinel_ratio"]})

    # Spread the good cases over distinct sequences so they are not all one take.
    seen = set()
    for r in good:
        if r["sequence"] in seen:
            continue
        seen.add(r["sequence"])
        add(r, "good", f"lowest per-view median error in sequence {r['sequence']}")
        if len(cases) >= n_each:
            break
    for r in good[:n_each]:
        if len(cases) >= n_each:
            break
        add(r, "good", "globally lowest per-view median error")

    n_bad = 0
    seen = set()
    for r in bad:
        if r["sequence"] in seen and n_bad >= 3:
            continue
        seen.add(r["sequence"])
        add(r, "bad", "extreme non-sentinel failure (highest per-view median error)")
        n_bad += 1
        if n_bad >= n_each:
            break

    # Paired cases: same sequence + frame + hand, one good and one bad camera,
    # so the only thing that differs is the viewpoint.
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["sequence"], r["frame"], r["hand"])].append(r)
    pairs = 0
    for (seq, frame, hand), rs in sorted(grouped.items()):
        gs = [r for r in rs if r["view_class"] == "good"]
        bs = [r for r in rs if r["view_class"] == "bad"]
        ss = [r for r in rs if r["view_class"] == "zero_sentinel"]
        if not gs:
            continue
        g0 = min(gs, key=lambda r: float(r["err_median_px"]))
        partner, kind = None, ""
        if bs:
            partner = max(bs, key=lambda r: float(r["err_median_px"]))
            kind = "bad(non-sentinel)"
        elif ss:
            partner = ss[0]
            kind = "zero-sentinel"
        if partner is None:
            continue
        pid = f"pair{pairs + 1:02d}"
        add(g0, "pair_good", f"same sequence/frame/hand as {pid} partner; good camera", pid)
        add(partner, "pair_bad",
            f"same sequence/frame/hand as {pid} partner; {kind} camera", pid)
        pairs += 1
        if pairs >= n_each:
            break

    for r in sent[:2]:
        add(r, "zero_sentinel",
            "hand not detected in this view: every GT 2D joint is exactly (0,0) "
            "with confidence 1.0")
    return cases


def main() -> dict:
    views = load_gigahands_views()
    rows = build(views)
    cls = classify(rows)
    log.info("views=%d  classes=%s  otsu=%.2f px", len(rows), cls["counts"],
             cls["otsu_threshold_px"])

    rows.sort(key=lambda r: (r["err_median_px"] == "", r["err_median_px"] if
                             r["err_median_px"] == "" else float(r["err_median_px"])))
    write_csv(RUN_DIR / "results" / "raw" / "per_view_error_summary.csv", rows)

    # Per-frame: aggregate the two hands of one (sequence, camera, frame).
    per_frame = defaultdict(list)
    for r in rows:
        per_frame[(r["sequence"], r["camera"], r["frame"])].append(r)
    frame_rows = []
    for (seq, cam, frame), rs in sorted(per_frame.items()):
        real = [float(r["err_median_px"]) for r in rs
                if not r["is_zero_sentinel_view"] and r["err_median_px"] != ""]
        frame_rows.append({
            "sequence": seq, "camera": cam, "frame": frame,
            "n_hands": len(rs),
            "n_zero_sentinel_hands": sum(r["is_zero_sentinel_view"] for r in rs),
            "classes": "|".join(sorted(r["view_class"] for r in rs)),
            **{f"err_{k}": v for k, v in stats(real).items()}})
    write_csv(RUN_DIR / "results" / "raw" / "per_frame_error_summary.csv", frame_rows)

    write_csv(RUN_DIR / "tables" / "per_camera_stats.csv",
              group_table(rows, lambda r: r["camera"], "camera"))
    write_csv(RUN_DIR / "tables" / "per_sequence_stats.csv",
              group_table(rows, lambda r: r["sequence"], "sequence"))
    write_csv(RUN_DIR / "tables" / "per_camera_family_stats.csv",
              group_table(rows, lambda r: r["camera_family"], "camera_family"))
    write_csv(RUN_DIR / "tables" / "per_hand_stats.csv",
              group_table(rows, lambda r: r["hand"], "hand"))

    real_sorted = sorted([r for r in rows if not r["is_zero_sentinel_view"]
                          and r["err_median_px"] != ""],
                         key=lambda r: float(r["err_median_px"]))
    write_csv(RUN_DIR / "tables" / "top_good_views.csv", real_sorted[:50])
    write_csv(RUN_DIR / "tables" / "top_bad_views.csv", real_sorted[::-1][:50])

    cases = pick_cases(rows)
    write_csv(RUN_DIR / "results" / "raw" / "representative_cases.csv", cases)

    good_bad = []
    for cname in ("good", "medium", "bad", "zero_sentinel"):
        sel = [r for r in rows if r["view_class"] == cname]
        errs = [float(r["err_median_px"]) for r in sel if r["err_median_px"] != ""]
        good_bad.append({"view_class": cname, "n_views": len(sel),
                         "share_of_all_views": round(len(sel) / len(rows), 4),
                         **{f"view_median_{k}": v for k, v in stats(errs).items()}})
    write_csv(RUN_DIR / "results" / "summary" / "good_vs_bad_summary.csv", good_bad)

    (RUN_DIR / "results" / "summary" / "_classification.json").write_text(
        json.dumps(cls, indent=2), encoding="utf-8")
    return cls


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
