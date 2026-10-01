"""Are the provided extrinsics of the same physical camera comparable across
sequences?

This has to be settled before any camera-orientation descriptor is used, because
each GigaHands take is reconstructed on its own and its world frame need not be
shared. If the frames are not shared, camera pose cannot be compared across
sequences and no top/front/diagonal label may be invented.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, SUM, TAB, load_view_targets, write_csv, write_json  # noqa: E402

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def rot_angle_deg(Ra, Rb) -> float:
    c = (np.trace(Ra @ Rb.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def main() -> None:
    from experiments.src.datasets.gigahands import takes

    views = {(v["sequence"], v["camera"]) for v in load_view_targets()}
    take_by_name = {t.name: t for t in takes()}
    cams = defaultdict(dict)          # camera id -> sequence -> Camera
    for seq, cam in sorted(views):
        t = take_by_name.get(seq)
        if t is None:
            continue
        c = t.cameras.get(cam)
        if c is not None:
            cams[cam][seq] = c

    rows = []
    for cam, per_seq in sorted(cams.items()):
        seqs = sorted(per_seq)
        for a, b in combinations(seqs, 2):
            ca, cb = per_seq[a], per_seq[b]
            # camera centre in world coordinates: C = -R^T t
            Ca = -ca.R.T @ ca.t
            Cb = -cb.R.T @ cb.t
            rows.append({
                "physical_camera_id": cam, "sequence_a": a, "sequence_b": b,
                "rotation_difference_deg": round(rot_angle_deg(ca.R, cb.R), 4),
                "translation_difference_m": round(float(np.linalg.norm(ca.t - cb.t)), 6),
                "camera_centre_difference_m": round(float(np.linalg.norm(Ca - Cb)), 6),
                "fx_a": round(float(ca.K[0, 0]), 3), "fx_b": round(float(cb.K[0, 0]), 3),
                "fx_difference_px": round(float(abs(ca.K[0, 0] - cb.K[0, 0])), 4),
            })
    write_csv(TAB / "extrinsic_consistency_audit.csv", rows)

    rot = np.array([r["rotation_difference_deg"] for r in rows])
    cen = np.array([r["camera_centre_difference_m"] for r in rows])
    fxd = np.array([r["fx_difference_px"] for r in rows])
    # A shared world frame would make the same camera's pose nearly identical in
    # every sequence. A large spread means each take has its own frame.
    comparable = bool(np.median(rot) < 2.0 and np.median(cen) < 0.05)
    verdict = ("COMPARABLE_ACROSS_SEQUENCES" if comparable
               else "NOT_COMPARABLE_ACROSS_SEQUENCES")
    meta = {
        "n_camera_sequence_pairs": len(rows),
        "rotation_difference_deg": {
            "median": float(np.median(rot)), "p90": float(np.percentile(rot, 90)),
            "max": float(rot.max())},
        "camera_centre_difference_m": {
            "median": float(np.median(cen)), "p90": float(np.percentile(cen, 90)),
            "max": float(cen.max())},
        "intrinsic_fx_difference_px": {
            "median": float(np.median(fxd)), "max": float(fxd.max())},
        "verdict": verdict,
        "consequence": (
            "camera orientation descriptors derived from the provided extrinsics "
            "are comparable across sequences and may be used as an ORACLE "
            "DIAGNOSTIC"
            if comparable else
            "each sequence appears to have its own world frame, so the provided "
            "camera pose cannot be compared across sequences. No top/front/"
            "diagonal label is invented, and orientation descriptors are used "
            "only WITHIN a sequence, if at all."),
        "status": "ORACLE_DIAGNOSTIC_ONLY - provided calibration metadata is "
                  "never a deployable feature",
    }
    write_json(SUM / "extrinsic_consistency_verdict.json", meta)
    print(f"{len(rows)} same-camera sequence pairs")
    print(f"  rotation difference  median {np.median(rot):8.3f} deg  max {rot.max():8.3f}")
    print(f"  camera centre diff   median {np.median(cen):8.4f} m    max {cen.max():8.4f}")
    print(f"  fx difference        median {np.median(fxd):8.4f} px   max {fxd.max():8.4f}")
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
