"""Feature <-> bias associations, and the oracle camera-property diagnostic.

Effect size with a physical-camera cluster bootstrap CI is the headline. Naive
p-values ignore the cluster structure, so the Benjamini-Hochberg column is a
descriptive supplement only.

Associations are ASSOCIATIONS. Nothing here establishes a cause.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (MANIFESTS, REPO, SUM, benjamini_hochberg, load_view_targets,  # noqa: E402
                    read_csv, read_json, spearman_cluster_ci, write_csv, write_json)

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def deployable_associations():
    views = read_csv(SUM / "view_scene_features.csv.gz")
    kept = read_json(SUM / "_kept_features.json")["kept"]
    spec = read_json(MANIFESTS / "cam_exp_005_scene_feature_spec_v1.json")
    group_of = {f["name"]: g for g, fs in spec["groups"].items() for f in fs}

    cams = [v["physical_camera_id"] for v in views]
    signed = np.array([float(v["anycalib_signed_log_bias"]) for v in views])
    absol = np.array([float(v["anycalib_abs_log_bias"]) for v in views])

    rows = []
    for name in kept:
        x = np.array([float(v[name]) if v.get(name) not in ("", None) else np.nan
                      for v in views], float)
        for target_name, y in (("signed_log_bias", signed),
                               ("abs_log_bias", absol)):
            rho, p, lo, hi, n = spearman_cluster_ci(x, y, cams)
            rows.append({
                "feature": name, "group": group_of.get(name, ""),
                "target": target_name, "spearman_rho": rho,
                "ci_lo_camera_cluster": lo, "ci_hi_camera_cluster": hi,
                "ci_excludes_zero": int(np.isfinite(lo) and np.isfinite(hi)
                                        and (lo > 0 or hi < 0)),
                "naive_p": p, "n_views": n,
                "deployable": "yes",
                "evidence": "ASSOCIATION only - no causal claim",
            })
    for t in ("signed_log_bias", "abs_log_bias"):
        idx = [i for i, r in enumerate(rows) if r["target"] == t]
        q = benjamini_hochberg([rows[i]["naive_p"] for i in idx])
        for j, i in enumerate(idx):
            rows[i]["bh_fdr_q_descriptive"] = float(q[j])
    write_csv(SUM / "feature_bias_associations.csv", rows)
    return rows


def oracle_associations():
    """Provided camera properties vs bias. ORACLE_DIAGNOSTIC_ONLY."""
    from experiments.src.datasets.gigahands import takes
    views = load_view_targets()
    take_by_name = {t.name: t for t in takes()}
    ext_ok = read_json(SUM / "extrinsic_consistency_verdict.json")["verdict"] \
        == "COMPARABLE_ACROSS_SEQUENCES"

    props, cams, signed, absol = defaultdict(list), [], [], []
    for v in views:
        t = take_by_name.get(v["sequence"])
        c = t.cameras.get(v["camera"]) if t else None
        if c is None:
            continue
        K, R, tv, dist = c.K, c.R, c.t, c.dist
        W, H = float(c.width or 0), float(c.height or 0)
        centre = -R.T @ tv
        optical_axis = R.T @ np.array([0.0, 0.0, 1.0])
        props["gt_k1"].append(float(dist[0]) if dist.size > 0 else np.nan)
        props["gt_k2"].append(float(dist[1]) if dist.size > 1 else np.nan)
        props["gt_abs_k1"].append(abs(float(dist[0])) if dist.size > 0 else np.nan)
        props["gt_p1"].append(float(dist[2]) if dist.size > 2 else np.nan)
        props["gt_p2"].append(float(dist[3]) if dist.size > 3 else np.nan)
        props["gt_principal_point_offset_norm"].append(
            float(np.hypot(K[0, 2] - W / 2, K[1, 2] - H / 2) / np.hypot(W, H)))
        props["gt_principal_point_dx_px"].append(float(K[0, 2] - W / 2))
        props["gt_principal_point_dy_px"].append(float(K[1, 2] - H / 2))
        props["gt_fx_over_fy"].append(float(K[0, 0] / K[1, 1]))
        props["gt_reference_focal_px"].append(float(K[0, 0]))
        props["gt_camera_height_world"].append(float(centre[2]))
        props["gt_camera_distance_from_origin"].append(float(np.linalg.norm(centre)))
        props["gt_optical_axis_z"].append(float(optical_axis[2]))
        props["gt_optical_axis_elevation_deg"].append(
            float(np.degrees(np.arcsin(np.clip(optical_axis[2], -1, 1)))))
        cams.append(v["physical_camera_id"])
        signed.append(v["anycalib_signed_log_bias"])
        absol.append(v["anycalib_abs_log_bias"])

    signed = np.asarray(signed, float)
    absol = np.asarray(absol, float)
    rows = []
    for name, vals in props.items():
        x = np.asarray(vals, float)
        orientation = name.startswith(("gt_camera", "gt_optical"))
        for target_name, y in (("signed_log_bias", signed),
                               ("abs_log_bias", absol)):
            rho, p, lo, hi, n = spearman_cluster_ci(x, y, cams)
            rows.append({
                "camera_property": name, "target": target_name,
                "spearman_rho": rho, "ci_lo_camera_cluster": lo,
                "ci_hi_camera_cluster": hi,
                "ci_excludes_zero": int(np.isfinite(lo) and np.isfinite(hi)
                                        and (lo > 0 or hi < 0)),
                "naive_p": p, "n_views": n,
                "ORACLE_DIAGNOSTIC_ONLY": "yes",
                "deployable": "no - provided calibration metadata is not "
                              "available for a new deployment camera",
                "note": ("orientation descriptor; usable only because the "
                         "extrinsic audit found the same physical camera's pose "
                         "consistent across sequences"
                         if orientation and ext_ok else
                         "orientation descriptor - NOT COMPARABLE across "
                         "sequences, interpret with care"
                         if orientation else
                         "the reference focal is inside the target definition, "
                         "so its association with the bias is descriptive and "
                         "partly definitional"
                         if name == "gt_reference_focal_px" else ""),
            })
    write_csv(SUM / "oracle_camera_property_associations.csv", rows)
    return rows


def main() -> None:
    dep = deployable_associations()
    print("=== deployable features vs signed log bias (top |rho|) ===")
    s = [r for r in dep if r["target"] == "signed_log_bias"]
    for r in sorted(s, key=lambda r: -abs(r["spearman_rho"]))[:10]:
        print(f"  {r['feature']:38s} {r['group']:20s} rho={r['spearman_rho']:+.3f} "
              f"[{r['ci_lo_camera_cluster']:+.3f},{r['ci_hi_camera_cluster']:+.3f}] "
              f"excl0={r['ci_excludes_zero']} q={r['bh_fdr_q_descriptive']:.3f}")
    n_sig = sum(r["ci_excludes_zero"] for r in s)
    print(f"  {n_sig}/{len(s)} deployable features have a camera-cluster CI "
          f"excluding zero")

    orc = oracle_associations()
    print("=== ORACLE camera properties vs signed log bias (top |rho|) ===")
    so = [r for r in orc if r["target"] == "signed_log_bias"]
    for r in sorted(so, key=lambda r: -abs(r["spearman_rho"]))[:8]:
        print(f"  {r['camera_property']:34s} rho={r['spearman_rho']:+.3f} "
              f"[{r['ci_lo_camera_cluster']:+.3f},{r['ci_hi_camera_cluster']:+.3f}] "
              f"excl0={r['ci_excludes_zero']}")


if __name__ == "__main__":
    main()
