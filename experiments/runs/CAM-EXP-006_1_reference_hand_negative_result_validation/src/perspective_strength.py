"""ORACLE_PERSPECTIVE_STRENGTH_DIAGNOSTIC.

Was the GigaHands hand actually in a weak-perspective regime? This fits a pose
with the dataset-provided reference focal and measures how deep the hand is
relative to how far away it is.

The provided focal is used ONLY to interpret the cause. It is never an input to
any focal estimator, and this file writes nothing the solver reads.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (R006_CACHE, R006_FRAMES, RAW, SUM, VIEW_TARGETS, fnum,  # noqa: E402
                    read_csv, write_csv, write_json)


def main() -> None:
    import cv2
    z = np.load(R006_CACHE)
    tgt = {(r["sequence"], r["camera"]): r for r in read_csv(VIEW_TARGETS)}
    rows = read_csv(R006_FRAMES)
    by_view = defaultdict(list)
    for r in rows:
        by_view[(r["sequence"], r["camera"])].append(int(r["frame"]))

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from experiments.src.datasets.gigahands import takes
    meta = {}
    for t in takes():
        for name, c in t.cameras.items():
            meta[(t.name, name)] = (int(c.width), int(c.height),
                                    float(c.K[0, 2]), float(c.K[1, 2]))

    out = []
    for (seq, cam), frames in sorted(by_view.items()):
        t = tgt.get((seq, cam))
        if t is None:
            continue
        f_ref = fnum(t["gt_reference_focal_px"])
        w, h, cx, cy = meta[(seq, cam)]
        K = np.array([[f_ref, 0, cx], [0, f_ref, cy], [0, 0, 1.0]])
        for frame in sorted(frames):
            for hand in ("left", "right"):
                k = f"{seq}|{cam}|{frame}|{hand}"
                use = z[k + "|use"]
                if use.sum() < 12:
                    continue
                xyz = z[k + "|xyz"][use].astype(float)
                uv = z[k + "|uv"][use].astype(float)
                try:
                    ok, rvec, tvec = cv2.solvePnP(
                        np.ascontiguousarray(xyz.reshape(-1, 1, 3)),
                        np.ascontiguousarray(uv.reshape(-1, 1, 2)),
                        K, None, flags=cv2.SOLVEPNP_SQPNP)
                except cv2.error:
                    ok = False
                if not ok:
                    continue
                R, _ = cv2.Rodrigues(rvec)
                pc = (R @ xyz.T).T + tvec.reshape(3)
                Z = pc[:, 2]
                if np.any(Z <= 0):
                    continue
                z_med = float(np.median(Z))
                dz = float(Z.max() - Z.min())
                lat = float(max(pc[:, 0].max() - pc[:, 0].min(),
                                pc[:, 1].max() - pc[:, 1].min()))
                diam = float(np.linalg.norm(
                    pc[:, None, :] - pc[None, :, :], axis=-1).max())
                out.append({
                    "sequence": seq, "camera": cam, "frame": frame,
                    "hand": hand,
                    "median_depth_Z": round(z_med, 5),
                    "depth_extent_dZ": round(dz, 5),
                    "lateral_extent": round(lat, 5),
                    "hand_diameter": round(diam, 5),
                    "dZ_over_Z": round(dz / z_med, 6),
                    "diameter_over_Z": round(diam / z_med, 6),
                    "distance_over_diameter": round(z_med / diam, 4)
                    if diam > 0 else np.nan,
                })
    write_csv(RAW / "perspective_strength_cases.csv.gz", out)

    def q(key):
        a = np.array([r[key] for r in out], float)
        a = a[np.isfinite(a)]
        return {"median": round(float(np.median(a)), 5),
                "p10": round(float(np.percentile(a, 10)), 5),
                "p90": round(float(np.percentile(a, 90)), 5)}

    # per-view aggregate, for association with the solver's profile shape
    per_view = defaultdict(list)
    for r in out:
        per_view[(r["sequence"], r["camera"])].append(r)
    view_rows = [{
        "sequence": s, "camera": c,
        "median_dZ_over_Z": round(float(np.median(
            [x["dZ_over_Z"] for x in v])), 6),
        "median_diameter_over_Z": round(float(np.median(
            [x["diameter_over_Z"] for x in v])), 6),
        "median_distance_over_diameter": round(float(np.median(
            [x["distance_over_diameter"] for x in v])), 4),
        "n_cases": len(v),
    } for (s, c), v in sorted(per_view.items())]
    write_csv(SUM / "perspective_strength_summary.csv", view_rows)

    summary = {
        "cases": len(out), "views": len(view_rows),
        "dZ_over_Z": q("dZ_over_Z"),
        "diameter_over_Z": q("diameter_over_Z"),
        "distance_over_diameter": q("distance_over_diameter"),
        "median_depth_Z": q("median_depth_Z"),
        "hand_diameter": q("hand_diameter"),
        "gt_usage": "the dataset-provided reference focal is used ONLY to fit "
                    "the pose for this causal diagnostic; it is never an input "
                    "to any focal estimator",
        "reading": "distance_over_diameter states how many hand-diameters away "
                   "the camera sits. The synthetic distance sweep used the "
                   "same ratio, so the two are directly comparable.",
    }
    write_json(SUM / "perspective_strength_verdict.json", summary)
    print(f"cases {len(out)}, views {len(view_rows)}")
    for k in ("dZ_over_Z", "diameter_over_Z", "distance_over_diameter"):
        print(f"  {k:26s} {summary[k]}")


if __name__ == "__main__":
    main()
