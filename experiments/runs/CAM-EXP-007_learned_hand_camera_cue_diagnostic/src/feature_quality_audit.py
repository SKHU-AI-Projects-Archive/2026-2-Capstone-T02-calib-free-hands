"""Target-blind feature quality and leakage audits.

Drops are decided on TARGET-INDEPENDENT criteria only: all-NaN, constant, or
near-zero variance. No target is read here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, RAW, TAB, fnum, read_csv, write_csv  # noqa: E402
from run_probes import (F0_KEYS, F0_VIEW, F1_KEYS, F2_KEYS, F3_VIEW,  # noqa: E402
                        assign_group)

GROUP_META = {
    # group: (uses_RGB, uses_hand_model, deployable, note)
    "F0": (1, 1, 1, "detector and crop geometry on the image plane"),
    "F1": (1, 1, 1, "the model's weak-perspective camera head, crop space"),
    "F2": (1, 1, 1, "root-relative predicted hand shape and pose"),
    "F3": (1, 1, 1, "temporal dispersion across a static view"),
    "F4": (1, 1, 1, "frozen image-encoder latent, pooled"),
}


def main() -> None:
    rows = read_csv(RAW / "view_hand_features.csv.gz")
    cols = [c for c in rows[0] if assign_group(c) is not None]

    audit, keep = [], []
    for c in cols:
        v = np.array([fnum(r[c]) for r in rows], float)
        fin = v[np.isfinite(v)]
        nan_rate = 1.0 - fin.size / max(v.size, 1)
        sd = float(np.std(fin)) if fin.size else 0.0
        mean = float(np.mean(fin)) if fin.size else np.nan
        rng = (float(fin.max() - fin.min())) if fin.size else 0.0
        drop = ""
        if fin.size == 0:
            drop = "all_nan"
        elif sd == 0:
            drop = "constant"
        elif sd < 1e-10:
            drop = "near_zero_variance"
        elif nan_rate > 0.5:
            drop = "nan_rate_over_50pct"
        audit.append({
            "feature": c, "group": assign_group(c),
            "n_views": len(rows), "nan_rate": round(nan_rate, 4),
            "mean": round(mean, 6) if np.isfinite(mean) else "",
            "std": round(sd, 8), "range": round(rng, 6),
            "dropped": drop,
            "drop_criterion_is_target_independent": 1,
        })
        if not drop:
            keep.append(c)
    write_csv(TAB / "hand_feature_quality_audit.csv", audit)

    # ---- leakage audit --------------------------------------------------
    lat = CACHE / "view_latent.npz"
    leak = []
    for c in cols:
        g = assign_group(c)
        rgb, hm, dep, note = GROUP_META[g]
        leak.append({
            "feature_group": g, "feature": c,
            "uses_RGB": rgb, "uses_hand_model": hm,
            "uses_dataset_2d_annotation": 0,
            "uses_dataset_3d_annotation": 0,
            "uses_GT_focal": 0, "uses_GT_distortion": 0,
            "uses_GT_extrinsic": 0, "uses_camera_ID": 0,
            "uses_sequence_ID": 0, "uses_anycalib_prediction": 0,
            "uses_pipeline_focal": 0,
            "available_at_deployment": dep,
            "allowed_primary": int(not audit[cols.index(c)]["dropped"]),
            "note": note,
        })
    if lat.exists():
        leak.append({
            "feature_group": "F4", "feature": "LATENT::0..1279 (pooled "
                                              "vit_out)",
            "uses_RGB": 1, "uses_hand_model": 1,
            "uses_dataset_2d_annotation": 0, "uses_dataset_3d_annotation": 0,
            "uses_GT_focal": 0, "uses_GT_distortion": 0,
            "uses_GT_extrinsic": 0, "uses_camera_ID": 0,
            "uses_sequence_ID": 0, "uses_anycalib_prediction": 0,
            "uses_pipeline_focal": 0, "available_at_deployment": 1,
            "allowed_primary": 1,
            "note": "frozen image-encoder representation before the MANO and "
                    "camera heads",
        })
    # the two things that DO touch model predictions, kept separate on purpose
    leak.append({
        "feature_group": "CONTROL_C1", "feature": "log_f_anycalib",
        "uses_RGB": 1, "uses_hand_model": 0,
        "uses_dataset_2d_annotation": 0, "uses_dataset_3d_annotation": 0,
        "uses_GT_focal": 0, "uses_GT_distortion": 0, "uses_GT_extrinsic": 0,
        "uses_camera_ID": 0, "uses_sequence_ID": 0,
        "uses_anycalib_prediction": 1, "uses_pipeline_focal": 0,
        "available_at_deployment": 1, "allowed_primary": 0,
        "note": "SINGLE_FOCAL_PRIOR_CONFOUND_CONTROL - never a hand feature, "
                "never mixed into a primary set",
    })
    leak.append({
        "feature_group": "BASELINE_B3", "feature":
            "training-fold median reference focal",
        "uses_RGB": 0, "uses_hand_model": 0,
        "uses_dataset_2d_annotation": 0, "uses_dataset_3d_annotation": 0,
        "uses_GT_focal": 1, "uses_GT_distortion": 0, "uses_GT_extrinsic": 0,
        "uses_camera_ID": 0, "uses_sequence_ID": 0,
        "uses_anycalib_prediction": 0, "uses_pipeline_focal": 0,
        "available_at_deployment": 0, "allowed_primary": 0,
        "note": "ORACLE_SANITY_CONTROL, uses training target labels, not "
                "deployable",
    })
    write_csv(TAB / "hand_feature_leakage_audit.csv", leak)

    defs = [{"feature": c, "group": assign_group(c),
             "view_aggregation": ("std" if c.endswith("__std")
                                  else "iqr" if c.endswith("__iqr")
                                  else "median" if c.endswith("__median")
                                  else "view-level scalar"),
             "focal_free": 1} for c in cols]
    write_csv(TAB / "hand_feature_definitions.csv", defs)

    dropped = [a for a in audit if a["dropped"]]
    print(f"features {len(cols)}, kept {len(keep)}, dropped {len(dropped)}")
    for d in dropped[:10]:
        print(f"  drop {d['feature']}: {d['dropped']}")
    prim = [r for r in leak if r["allowed_primary"] == 1]
    bad = [r for r in prim if any(r[k] for k in (
        "uses_dataset_2d_annotation", "uses_dataset_3d_annotation",
        "uses_GT_focal", "uses_GT_distortion", "uses_GT_extrinsic",
        "uses_camera_ID", "uses_sequence_ID", "uses_anycalib_prediction",
        "uses_pipeline_focal"))]
    print(f"primary features with a forbidden source: {len(bad)}")


if __name__ == "__main__":
    main()
