"""Real-data reanalysis R0-R5 on GigaHands.

R0 and R1 share one camera model (centre principal point, no distortion,
fx = fy) and differ ONLY in aggregation, so their per-hand profiles are
computed once and aggregated two ways. That is what isolates the hand-weighting
mismatch cleanly.

GT focal MAGNITUDE is never passed to the solver in any condition. R5 receives
the provided fy/fx RATIO only.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import camera_model_solver as S  # noqa: E402
from common import (CACHE, COUNTS, R006_CACHE, R006_FRAMES, RAW, TAB,  # noqa: E402
                    read_csv, write_csv, write_json)

GRID = S.gamma_grid()

# camera models: (id, use_provided_pp, use_provided_dist, use_provided_ratio)
MODELS = {
    "M_CENTER_NODIST": (False, False, False),
    "M_GTPP_NODIST": (True, False, False),
    "M_CENTER_GTDIST": (False, True, False),
    "M_GTPP_GTDIST": (True, True, False),
    "M_GTPP_GTDIST_RATIO": (True, True, True),
}
CONDITIONS = [
    ("R0", "ORIGINAL_REPRODUCTION", "M_CENTER_NODIST", "hand_weighted"),
    ("R1", "FRAME_BALANCED", "M_CENTER_NODIST", "frame_balanced"),
    ("R2", "FRAME_BALANCED_GT_PP_ORACLE", "M_GTPP_NODIST", "frame_balanced"),
    ("R3", "FRAME_BALANCED_GT_DIST_ORACLE", "M_CENTER_GTDIST",
     "frame_balanced"),
    ("R4", "FRAME_BALANCED_GT_PP_DIST_ORACLE", "M_GTPP_GTDIST",
     "frame_balanced"),
    ("R5", "FULL_CAMERA_MODEL_ORACLE_EXCEPT_FOCAL", "M_GTPP_GTDIST_RATIO",
     "frame_balanced"),
]
KEEP_PROFILE_VIEWS = 24          # for figure 06


def camera_metadata():
    """Provided camera parameters per view. FOCAL MAGNITUDE IS NOT RETURNED."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from experiments.src.datasets.gigahands import takes
    meta = {}
    for t in takes():
        for name, c in t.cameras.items():
            meta[(t.name, name)] = {
                "w": int(c.width), "h": int(c.height),
                "cx": float(c.K[0, 2]), "cy": float(c.K[1, 2]),
                "dist": np.asarray(c.dist, float).copy(),
                "fy_over_fx": float(c.K[1, 1] / c.K[0, 0]),
            }
    return meta


def main() -> None:
    z = np.load(R006_CACHE)
    rows = read_csv(R006_FRAMES)
    by_view = defaultdict(list)
    for r in rows:
        by_view[(r["sequence"], r["camera"])].append(
            (int(r["grid_pos"]), int(r["frame"])))

    meta = camera_metadata()
    out, kept_profiles = [], []
    views = sorted(by_view)

    for vi, (seq, cam) in enumerate(views, 1):
        order = [f for _, f in sorted(by_view[(seq, cam)])]
        m = meta[(seq, cam)]
        w, h = m["w"], m["h"]

        # usable hands per frame
        per_frame_hands = {}
        for f in order:
            hs = []
            for hand in ("left", "right"):
                k = f"{seq}|{cam}|{f}|{hand}"
                use = z[k + "|use"]
                if use.sum() < 12:
                    continue
                hs.append((z[k + "|xyz"][use].astype(float),
                           z[k + "|uv"][use].astype(float)))
            per_frame_hands[f] = hs

        # one profile set per CAMERA MODEL (not per condition)
        prof = {}
        for mid, (use_pp, use_d, use_r) in MODELS.items():
            cx = m["cx"] if use_pp else None
            cy = m["cy"] if use_pp else None
            dist = m["dist"] if use_d else None
            ratio = m["fy_over_fx"] if use_r else 1.0
            prof[mid] = {f: [S.profile_hand(xyz, uv, w, h, cx, cy, dist,
                                            ratio, GRID)
                             for xyz, uv in per_frame_hands[f]]
                         for f in order}

        for cid, cname, mid, agg in CONDITIONS:
            for n in COUNTS:
                pfh = [prof[mid][f] for f in order[:n]]
                r = S.solve_view(pfh, w, h, aggregation=agg, grid=GRID)
                out.append({
                    "sequence": seq, "camera": cam, "condition": cid,
                    "condition_name": cname, "camera_model": mid,
                    "aggregation": agg, "n_frames": n,
                    "image_width": w, "image_height": h,
                    "n_frames_used": r["n_frames_used"],
                    "n_hand_observations": r["n_hand_observations"],
                    "gamma_hat": r["gamma"], "focal_hat_px": r["focal_px"],
                    "min_objective_px": r["min_objective_px"],
                    "profile_width": r["profile_width"],
                    "curvature": r["curvature"],
                    "depth_of_minimum_log": r["depth_of_minimum"],
                    "flat_profile": r["flat_profile"],
                    "boundary_solution": r["boundary_solution"],
                    "solver_status": r["status"],
                })
            # keep the N=16 aggregated curve for a representative subset
            if vi <= KEEP_PROFILE_VIEWS and cid in ("R1", "R5"):
                pfh = [prof[mid][f] for f in order[:16]]
                curve = (S.aggregate_frame_balanced(pfh) if agg ==
                         "frame_balanced" else S.aggregate_hand_weighted(pfh))
                if curve is not None:
                    kept_profiles.append({
                        "sequence": seq, "camera": cam, "condition": cid,
                        **{f"g{i}": float(v) for i, v in enumerate(curve)}})
        if vi % 10 == 0:
            print(f"  {vi}/{len(views)} views", flush=True)

    write_csv(RAW / "real_validation_solutions.csv.gz", out)
    write_csv(RAW / "real_objective_profiles.csv.gz", kept_profiles)

    # ---- GT information usage audit --------------------------------------
    usage = []
    for cid, cname, mid, agg in CONDITIONS:
        use_pp, use_d, use_r = MODELS[mid]
        usage.append({
            "condition": cid, "condition_name": cname,
            "uses_gt_focal_magnitude": 0,
            "uses_gt_focal_ratio": int(use_r),
            "uses_gt_principal_point": int(use_pp),
            "uses_gt_distortion": int(use_d),
            "uses_gt_extrinsic": 0,
            "uses_target_2d_in_reference_3d": 0,
            "deployable": int(not (use_pp or use_d or use_r)),
            "diagnostic_only": int(use_pp or use_d or use_r),
        })
    write_csv(TAB / "oracle_information_usage.csv", usage)
    write_json(RAW.parent / "summary" / "real_reanalysis_meta.json", {
        "views": len(views), "conditions": len(CONDITIONS),
        "camera_models": len(MODELS), "rows": len(out),
        "gt_focal_magnitude_ever_a_solver_input": False,
        "note": "R0 and R1 share camera model M_CENTER_NODIST and differ only "
                "in aggregation, so the R0->R1 comparison isolates the "
                "hand-weighting mismatch exactly.",
    })
    print(f"wrote {len(out)} solutions")


if __name__ == "__main__":
    main()
