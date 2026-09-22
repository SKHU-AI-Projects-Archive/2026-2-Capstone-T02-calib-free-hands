"""Step 1 - targeted smoke test, control validation and threshold derivation.

Order matters: the clean controls are reconstructed first and their residual
distribution sets the thresholds. Only then are the suspected cases analysed,
so no threshold is tuned on the cases it will later judge.
"""
from __future__ import annotations

import gzip
import csv
import json
import logging
from collections import defaultdict

import numpy as np

from common import (CENSUS_CSV, RUN_DIR, TAKES, rel, write_csv)
import loco
import verdicts

log = logging.getLogger("cam-exp-001.3")

N_PER_GROUP = 20
GROUPS = ("good", "medium", "likely_hand_identity_error", "bad_unexplained")
# Provisional threshold for the control pass only; the operating value is
# derived from the control residual distribution immediately afterwards.
BOOTSTRAP_THRESHOLD_PX = 12.0
BOOTSTRAP_MIN_INLIERS = 3


def load_census() -> list:
    with gzip.open(CENSUS_CSV, "rt", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_targets(census: list, rng) -> dict:
    """Sample each 001.2 status group, spread over sequences and cameras.

    The 001.2 status is used ONLY to choose what to look at. It never enters
    triangulation, inlier selection or the final verdict.
    """
    by_status = defaultdict(list)
    for r in census:
        if r["status"] in GROUPS and r["video_available"] in ("0", "1"):
            by_status[r["status"]].append(r)
    out = {}
    for st in GROUPS:
        pool = by_status.get(st, [])
        if not pool:
            out[st] = []
            continue
        # round-robin over (sequence, camera) so one camera cannot dominate
        buckets = defaultdict(list)
        for r in pool:
            buckets[(r["sequence"], r["camera"])].append(r)
        keys = sorted(buckets)
        rng.shuffle(keys)
        chosen, i = [], 0
        while len(chosen) < N_PER_GROUP and i < 10 * max(1, len(keys)):
            k = keys[i % len(keys)]
            b = buckets[k]
            if b:
                chosen.append(b.pop(0))
            i += 1
        out[st] = chosen[:N_PER_GROUP]
    return out


def run_group(take_by_name, rows, threshold_px, min_inliers, rng, tag) -> list:
    out = []
    for r in rows:
        take = take_by_name[r["sequence"]]
        res = loco.adjudicate(take, int(r["frame"]), r["camera"],
                              threshold_px, min_inliers, rng=rng)
        res = {k: v for k, v in res.items() if not k.startswith("_")}
        res["census_status_001_2"] = r["status"]
        res["group"] = tag
        res["hand_of_interest"] = r["hand"]
        out.append(res)
    return out


def derive_thresholds(control_rows: list) -> dict:
    """Set operating thresholds from the clean-control distribution."""
    def col(rows, key):
        v = [float(r[key]) for r in rows if r.get(key) not in ("", None)]
        return np.asarray(v, dtype=float)

    def col_hand(rows, template):
        """Take the column belonging to each row's own hand of interest.

        A control row was selected because ONE hand was clean in CAM-EXP-001.2;
        the other hand of the same frame may well be a swap case, so mixing both
        hands in would contaminate the clean distribution the thresholds come from.
        """
        v = []
        for r in rows:
            h = r.get("hand_of_interest")
            if h not in ("left", "right"):
                continue
            key = template.format(h=h, o="right" if h == "left" else "left",
                                  H="LL" if h == "left" else "RR")
            if r.get(key) not in ("", None):
                v.append(float(r[key]))
        return np.asarray(v, dtype=float)

    reproj = col_hand(control_rows, "{h}_tri_median_reproj_px")
    same = col_hand(control_rows, "E_{H}_px")
    mpjpe = col_hand(control_rows, "{h}_vs_provided_{h}_mpjpe_mm")
    inl = col_hand(control_rows, "{h}_n_inlier_cameras")
    ang = col_hand(control_rows, "{h}_max_ray_angle_deg")

    def stats(a):
        if a.size == 0:
            return {}
        med = float(np.median(a))
        mad = float(np.median(np.abs(a - med)))
        return {"n": int(a.size), "median": round(med, 4),
                "mad": round(mad, 4),
                "robust_sigma": round(1.4826 * mad, 4),
                "p90": round(float(np.percentile(a, 90)), 4),
                "p95": round(float(np.percentile(a, 95)), 4),
                "p99": round(float(np.percentile(a, 99)), 4),
                "max": round(float(a.max()), 4)}

    s_reproj, s_same, s_mpjpe = stats(reproj), stats(same), stats(mpjpe)
    s_inl, s_ang = stats(inl), stats(ang)

    # Operating points: p99 of the clean controls, so a control sample is
    # almost never rejected, with a floor to avoid an implausibly tight value.
    # Geometric requirements are conditioning floors, not "as good as the best
    # control": demanding a control-percentile number of inlier cameras or ray
    # angle would reject perfectly well-conditioned frames simply for being
    # less redundant than the cleanest ones. The control distribution is
    # reported alongside so the margin is visible.
    p5_inl = float(np.percentile(inl, 5)) if inl.size else 4.0
    thr = {
        # inlier gate for RANSAC, from the control reprojection spread
        "triangulation_inlier_px": round(max(6.0, s_reproj.get("p99", 8.0)), 2),
        # a joint needs this many agreeing cameras to be reconstructed at all
        "min_inliers_for_reconstruction": 4,
        # QC criteria derived from the clean-control distribution
        "same_hand_reprojection_px": round(max(8.0, s_same.get("p99", 12.0)), 2),
        "provided3d_mpjpe_mm": round(max(10.0, s_mpjpe.get("p99", 20.0)), 2),
        "qc_min_inlier_cameras": int(np.clip(np.floor(p5_inl), 4, 8)),
        "min_ray_angle_deg": 30.0,
        # the opposite hand must beat the same hand by this factor to call a swap
        "identity_margin_factor": 3.0,
        "identity_min_gap_px": 30.0,
    }
    return {"control_distributions": {"triangulation_median_reproj_px": s_reproj,
                                      "same_hand_reprojection_px": s_same,
                                      "provided3d_mpjpe_mm": s_mpjpe,
                                      "n_inlier_cameras": s_inl,
                                      "max_ray_angle_deg": s_ang},
            "thresholds": thr,
            "derivation": (
                "Quality thresholds (same_hand_reprojection_px, provided3d_mpjpe_mm) are "
                "the p99 of the clean-control distribution measured at the OPERATING "
                "configuration, with a floor. Geometric requirements "
                "(qc_min_inlier_cameras, min_ray_angle_deg) are conditioning floors "
                "clipped to a small range rather than control percentiles, so that a "
                "well-conditioned but less redundant frame is not rejected for being "
                "less redundant than the cleanest controls. identity_margin_factor is "
                "how many times closer the opposite-hand reconstruction must be, and "
                "identity_min_gap_px the absolute centroid gap also required.")}


def main() -> dict:
    rng = np.random.default_rng(0)
    census = load_census()
    take_by_name = {t.name: t for t in TAKES}
    targets = pick_targets(census, rng)
    log.info("smoke targets: %s", {k: len(v) for k, v in targets.items()})

    # --- control pass first ------------------------------------------------
    control_rows = run_group(take_by_name, targets["good"], BOOTSTRAP_THRESHOLD_PX,
                             BOOTSTRAP_MIN_INLIERS, rng, "good_control")
    thr = derive_thresholds(control_rows)
    log.info("derived thresholds: %s", thr["thresholds"])
    (RUN_DIR / "results" / "summary" / "_thresholds.json").write_text(
        json.dumps(thr, indent=2), encoding="utf-8")

    t = thr["thresholds"]
    # --- re-run controls at the operating threshold, then the other groups --
    rows = run_group(take_by_name, targets["good"], t["triangulation_inlier_px"],
                     t["min_inliers_for_reconstruction"], rng, "good_control")

    # Stage 2: the bootstrap pass used a looser inlier gate, which inflates
    # inlier counts and ray angles. Re-derive the QC criteria from the controls
    # as measured at the operating configuration, keeping the inlier gate fixed.
    thr2 = derive_thresholds(rows)
    fixed = t["triangulation_inlier_px"]
    thr = {"bootstrap": thr, "control_distributions": thr2["control_distributions"],
           "thresholds": {**thr2["thresholds"], "triangulation_inlier_px": fixed},
           "derivation": thr2["derivation"]}
    t = thr["thresholds"]
    log.info("operating thresholds: %s", t)
    (RUN_DIR / "results" / "summary" / "_thresholds.json").write_text(
        json.dumps(thr, indent=2), encoding="utf-8")
    for st, tag in (("medium", "medium_control"),
                    ("likely_hand_identity_error", "suspected_swap"),
                    ("bad_unexplained", "unexplained")):
        rows += run_group(take_by_name, targets[st], t["triangulation_inlier_px"],
                          t["min_inliers_for_reconstruction"], rng, tag)

    for r in rows:
        for h in ("left", "right"):
            case, why = verdicts.adjudicate_case(r, h, t)
            st, why_st = verdicts.qc_status(r, h, case, t)
            r[f"{h}_case"] = case
            r[f"{h}_case_reason"] = why
            r[f"{h}_qc_status"] = st
            r[f"{h}_qc_reason"] = why_st
            r[f"{h}_identity_margin_px"] = round(verdicts.identity_margin_px(r, h), 2)                 if np.isfinite(verdicts.identity_margin_px(r, h)) else ""
    write_csv(RUN_DIR / "results" / "raw" / "smoke_leave_one_camera_out.csv", rows)

    # verdict breakdown for the hand each row was selected for
    from collections import Counter as _C
    tally = _C((r["group"], r.get(f"{r['hand_of_interest']}_case", "")) for r in rows)
    write_csv(RUN_DIR / "results" / "summary" / "smoke_case_summary.csv",
              [{"group": g, "case": c, "n": n} for (g, c), n in sorted(tally.items())])

    ctrl = [r for r in rows if r["group"] == "good_control"]
    ok = [r for r in ctrl if r.get("left_tri_status") == "ok"
          or r.get("right_tri_status") == "ok"]
    # control metrics are computed on each row's own clean hand only
    def _hand_col(rows_, tmpl):
        v = []
        for r in rows_:
            h = r.get("hand_of_interest")
            if h not in ("left", "right"):
                continue
            k = tmpl.format(h=h, o="right" if h == "left" else "left",
                            H="LL" if h == "left" else "RR",
                            X="LR" if h == "left" else "RL")
            if r.get(k) not in ("", None):
                v.append(float(r[k]))
        return np.asarray(v)

    ctrl_metrics = []
    for tmpl, label in (("E_{H}_px", "held-out 2D vs SAME-hand reconstruction (px)"),
                        ("E_{X}_px", "held-out 2D vs OTHER-hand reconstruction (px)"),
                        ("C_{H}_px", "centroid, same hand (px)"),
                        ("C_{X}_px", "centroid, other hand (px)"),
                        ("{h}_vs_provided_{h}_mpjpe_mm", "reconstruction vs provided SAME 3D (mm)"),
                        ("{h}_vs_provided_{o}_mpjpe_mm", "reconstruction vs provided OTHER 3D (mm)"),
                        ("{h}_vs_provided_{h}_root_aligned_mpjpe_mm", "root-aligned MPJPE (mm)"),
                        ("{h}_vs_provided_{h}_wrist_mm", "wrist distance (mm)"),
                        ("{h}_tri_median_reproj_px", "triangulation residual (px)"),
                        ("{h}_n_inlier_cameras", "inlier cameras"),
                        ("{h}_max_ray_angle_deg", "max ray angle (deg)")):
        v = _hand_col(ctrl, tmpl)
        if v.size:
            ctrl_metrics.append({"metric": tmpl, "description": label, "n": int(v.size),
                                 "median": round(float(np.median(v)), 3),
                                 "p90": round(float(np.percentile(v, 90)), 3),
                                 "max": round(float(v.max()), 3)})
    write_csv(RUN_DIR / "tables" / "good_control_metrics.csv", ctrl_metrics)

    control_ok = _control_sanity(ctrl)
    log.info("control sanity: %s", control_ok)
    return {"thresholds": thr, "control_sanity": control_ok,
            "n_smoke_rows": len(rows), "n_control_ok": len(ok)}


def _control_sanity(ctrl: list) -> dict:
    """Controls must reconstruct, match the provided 3D, and not look swapped."""
    def arr(tmpl):
        v = []
        for r in ctrl:
            h = r.get("hand_of_interest")
            if h not in ("left", "right"):
                continue
            k = tmpl.format(h=h, o="right" if h == "left" else "left",
                            H="LL" if h == "left" else "RR",
                            X="LR" if h == "left" else "RL")
            if r.get(k) not in ("", None):
                v.append(float(r[k]))
        return np.asarray(v)
    same, cross = arr("E_{H}_px"), arr("E_{X}_px")
    mp, mpx = arr("{h}_vs_provided_{h}_mpjpe_mm"), arr("{h}_vs_provided_{o}_mpjpe_mm")
    out = {
        "n_control_rows": len(ctrl),
        "same_hand_median_px": round(float(np.median(same)), 3) if same.size else None,
        "cross_hand_median_px": round(float(np.median(cross)), 3) if cross.size else None,
        "provided3d_same_median_mm": round(float(np.median(mp)), 3) if mp.size else None,
        "provided3d_cross_median_mm": round(float(np.median(mpx)), 3) if mpx.size else None,
    }
    out["same_beats_cross"] = bool(same.size and cross.size
                                   and np.median(same) < np.median(cross))
    out["provided3d_agrees"] = bool(mp.size and np.median(mp) < 20.0)
    out["passed"] = bool(out["same_beats_cross"] and out["provided3d_agrees"])
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print(json.dumps(main(), indent=2, default=str))
