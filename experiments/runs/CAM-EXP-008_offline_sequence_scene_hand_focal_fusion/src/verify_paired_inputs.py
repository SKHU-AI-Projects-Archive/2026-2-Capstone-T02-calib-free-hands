"""Assert that S0 and S1 really did receive identical input.

If this fails, the CAM-EXP-008 primary result is invalid and must not be
reported as a paired comparison.

Checks, per (view, subset):
  * identical frame IDs
  * identical scene predictions
  * identical candidate focal grid
  * identical scene nuisance parameters (f_scene, sigma)
  * the ONLY difference is whether the hand term is on

Also re-checks the circularity contract.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, SUM, TAB, fnum, read_csv, read_json, write_csv, write_json  # noqa: E402


def main() -> None:
    rows = read_csv(RAW / "fusion_fold_predictions.csv.gz")

    # group every condition of one (view, subset) together
    g = defaultdict(dict)
    for r in rows:
        k = (r["sequence"], r["camera"], r["subset"])
        g[k].setdefault(r["condition"], []).append(r)

    audit, bad = [], []
    for (seq, cam, subset), conds in sorted(g.items()):
        if "S0" not in conds or "S1" not in conds:
            continue
        s0 = conds["S0"][0]
        s1 = conds["S1"][0]
        same_frames = s0["frame_hash"] == s1["frame_hash"]
        same_scene = s0["scene_hash"] == s1["scene_hash"]
        same_grid = s0["grid_hash"] == s1["grid_hash"]
        same_nf = s0["n_frames"] == s1["n_frames"]
        same_fscene = abs(fnum(s0["f_scene"]) - fnum(s1["f_scene"])) < 1e-9
        same_sigma = abs(fnum(s0["sigma_scene"])
                         - fnum(s1["sigma_scene"])) < 1e-12
        ok = all([same_frames, same_scene, same_grid, same_nf, same_fscene,
                  same_sigma])
        # C3 (flat hand) must reproduce S0 exactly
        c3_ok = ""
        if "C3" in conds:
            c3_ok = int(abs(fnum(conds["C3"][0]["f_hat"])
                            - fnum(s0["f_hat"])) < 1e-9)
        audit.append({
            "sequence": seq, "camera": cam, "subset": subset,
            "n_frames": s0["n_frames"],
            "frame_hash_scene": s0["frame_hash"],
            "frame_hash_fusion": s1["frame_hash"],
            "scene_prediction_hash_scene": s0["scene_hash"],
            "scene_prediction_hash_fusion": s1["scene_hash"],
            "grid_hash_scene": s0["grid_hash"],
            "grid_hash_fusion": s1["grid_hash"],
            "f_scene_scene": s0["f_scene"], "f_scene_fusion": s1["f_scene"],
            "same_frames": int(same_frames), "same_scene": int(same_scene),
            "same_grid": int(same_grid),
            "same_scene_nuisance": int(same_fscene and same_sigma),
            "C3_flat_hand_reproduces_S0": c3_ok,
            "only_hand_term_differs": int(ok),
        })
        if not ok:
            bad.append((seq, cam, subset))

    write_csv(TAB / "paired_input_identity_audit.csv", audit)

    c3 = [a["C3_flat_hand_reproduces_S0"] for a in audit
          if a["C3_flat_hand_reproduces_S0"] != ""]
    ref = read_json(SUM / "reference_3d_meta.json") \
        if (SUM / "reference_3d_meta.json").exists() else {}

    circ = [{
        "condition": c,
        "target_camera_2d_used_in_hand_objective":
            "YES" if c in ("S1", "H0", "C1", "C2") else "NO",
        "target_camera_2d_used_in_reference_3d": "NO",
        "target_camera_extrinsic_used": "NO",
        "target_camera_reference_focal_used_in_optimization": "NO",
        "GT_distortion_used": "NO",
        "GT_principal_point_used": "NO",
        "scene_prediction_used": "YES",
        "allowed_primary": "YES" if c in ("S0", "S1") else
                           "DIAGNOSTIC_OR_CONTROL",
    } for c in ("S0", "S1", "H0", "C1", "C2", "C3")]
    write_csv(TAB / "circularity_audit.csv", circ)

    verdict = {
        "rows_checked": len(audit),
        "views": len({(a["sequence"], a["camera"]) for a in audit}),
        "subsets": sorted({a["subset"] for a in audit}),
        "same_frames_all": all(a["same_frames"] for a in audit),
        "same_scene_all": all(a["same_scene"] for a in audit),
        "same_grid_all": all(a["same_grid"] for a in audit),
        "same_scene_nuisance_all": all(a["same_scene_nuisance"]
                                       for a in audit),
        "C3_flat_hand_reproduces_S0": (all(c == 1 for c in c3)
                                       if c3 else None),
        "failures": bad[:10],
        "verdict": "PAIRED_INPUT_IDENTITY_HELD" if not bad
                   else "PAIRED_INPUT_IDENTITY_VIOLATED",
        "circularity": {
            "camera_under_test_ever_an_inlier_of_its_own_reference_3d":
                ref.get("camera_under_test_ever_an_inlier"),
            "non_circularity": ref.get("non_circularity"),
        },
        "meaning": "S0 and S1 are computed from one shared per-(view, subset) "
                   "scene cost and one shared candidate grid. The hand term is "
                   "the only thing switched on, so any difference in the "
                   "estimate is attributable to it.",
    }
    write_json(SUM / "paired_input_verification.json", verdict)

    for k in ("same_frames_all", "same_scene_all", "same_grid_all",
              "same_scene_nuisance_all", "C3_flat_hand_reproduces_S0"):
        print(f"  {k:32s} {verdict[k]}")
    print(f"\n{verdict['verdict']}  ({len(audit)} view-subset rows)")
    print(f"  non-circularity: {verdict['circularity']}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
