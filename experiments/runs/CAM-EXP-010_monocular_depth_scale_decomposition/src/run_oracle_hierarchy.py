"""The O0-O6 oracle hierarchy. EVERY correction here is a DIAGNOSTIC.

None of these is a deployable method: each one is fitted using the reference 3D,
which a deployment does not have. Their purpose is to attribute the residual
absolute-3D error to components.

  O0  oracle focal only, cam_t untouched              (the baseline)
  O1  sequence-shared multiplicative z scale
  O2  sequence-shared affine z  (alpha*z + beta)
  O3  sequence-shared constant 3D bias
  O4  sequence-shared xy bias + affine z
  O5  per-frame depth oracle    (z set to reference)
  O6  per-frame root oracle     (T set to reference)

Parameters for O1-O4 are fitted on FIT frames ONLY and frozen before being
applied to EVAL frames, so a frame's own reference never fits its own
correction. All reported numbers are EVAL-frame numbers.

PRIMARY granularity is one correction per sequence-camera video, fitted from
both hands together — that matches the deployment shape (one video, one worker).
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from common import (GATE, MIN_EVAL_FRAMES, MIN_FIT_FRAMES, ORACLES,  # noqa: E402
                    RAW, SUM, TAB, fnum, mad, med, paired_cluster_bootstrap,
                    pct, read_csv, write_csv, write_json)


def load():
    rows = read_csv(RAW / "frame_translation_errors.csv.gz")
    z = np.load(RAW / "pose_residuals.npz")
    R = z["R"]
    idx = {k: i for i, k in enumerate(z["keys"].tolist())}
    for r in rows:
        r["_key"] = "%s|%s|%s|%s" % (r["sequence"], r["camera"], r["hand"],
                                     r["frame"])
        for k in ("x_pred", "y_pred", "z_pred", "x_ref", "y_ref", "z_ref"):
            r[k] = fnum(r[k])
    return rows, R, idx


def fit_params(fit_rows):
    """Robust sequence-level parameters, log-space for the scale."""
    zp = np.array([r["z_pred"] for r in fit_rows])
    zr = np.array([r["z_ref"] for r in fit_rows])
    ok = np.isfinite(zp) & np.isfinite(zr) & (zp > 0) & (zr > 0)
    out = {"n_fit": int(ok.sum())}
    if ok.sum() < MIN_FIT_FRAMES:
        return None
    # O1: alpha = exp(median(log z_ref - log z_pred))
    out["alpha"] = float(np.exp(np.median(np.log(zr[ok]) - np.log(zp[ok]))))
    # O2: robust affine z via Huber IRLS on (z_pred -> z_ref)
    A = np.stack([zp[ok], np.ones(ok.sum())], 1)
    b = zr[ok]
    w = np.ones(len(b))
    coef = np.array([out["alpha"], 0.0])
    for _ in range(25):
        Aw = A * w[:, None]
        coef, *_ = np.linalg.lstsq(Aw, b * w, rcond=None)
        res = b - A @ coef
        s = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-9
        w = np.clip(1.345 * s / np.maximum(np.abs(res), 1e-12), None, 1.0)
    out["alpha_affine"], out["beta_affine"] = float(coef[0]), float(coef[1])
    # O3: constant 3D bias
    for a, p, rr in (("bx", "x_pred", "x_ref"), ("by", "y_pred", "y_ref"),
                     ("bz", "z_pred", "z_ref")):
        d = np.array([r[rr] - r[p] for r in fit_rows], float)
        out[a] = float(np.median(d[np.isfinite(d)]))
    return out


def corrected_T(name, r, par):
    x, y, z = r["x_pred"], r["y_pred"], r["z_pred"]
    if name == "O0_ORACLE_FOCAL_ONLY":
        return np.array([x, y, z])
    if name == "O1_SEQUENCE_MULTIPLICATIVE_Z_SCALE":
        return np.array([x, y, par["alpha"] * z])
    if name == "O2_SEQUENCE_AFFINE_Z":
        return np.array([x, y, par["alpha_affine"] * z + par["beta_affine"]])
    if name == "O3_SEQUENCE_CONSTANT_3D_BIAS":
        return np.array([x + par["bx"], y + par["by"], z + par["bz"]])
    if name == "O4_SEQUENCE_SCALE_PLUS_BIAS":
        return np.array([x + par["bx"], y + par["by"],
                         par["alpha_affine"] * z + par["beta_affine"]])
    if name == "O5_PER_FRAME_DEPTH_ORACLE":
        return np.array([x, y, r["z_ref"]])
    if name == "O6_PER_FRAME_ROOT_TRANSLATION_ORACLE":
        return np.array([r["x_ref"], r["y_ref"], r["z_ref"]])
    raise ValueError(name)


def main():
    rows, R, idx = load()
    by_unit = defaultdict(list)
    for r in rows:
        by_unit[(r["sequence"], r["camera"])].append(r)

    params, out = [], []
    for (seq, cam), rs in sorted(by_unit.items()):
        fit = [r for r in rs if r["split"] == "FIT"]
        ev = [r for r in rs if r["split"] == "EVAL"]
        if len(fit) < MIN_FIT_FRAMES or len(ev) < MIN_EVAL_FRAMES:
            continue
        par = fit_params(fit)
        if par is None:
            continue
        # per-side parameters, secondary diagnostic
        side = {}
        for h in ("left", "right"):
            f2 = [r for r in fit if r["hand"] == h]
            p2 = fit_params(f2) if len(f2) >= MIN_FIT_FRAMES else None
            side[h] = p2["alpha"] if p2 else np.nan
        params.append({"sequence": seq, "camera": cam,
                       "participant": rs[0]["participant"],
                       "outer_fold": rs[0]["outer_fold"],
                       "n_fit": len(fit), "n_eval": len(ev), **par,
                       "alpha_left": side["left"],
                       "alpha_right": side["right"]})

        for r in ev:
            i = idx.get(r["_key"])
            if i is None:
                continue
            Rj = R[i]
            m = np.isfinite(Rj).all(1)
            T_ref = np.array([r["x_ref"], r["y_ref"], r["z_ref"]])
            ra = float(np.mean(np.linalg.norm(Rj[m], axis=1))) * 1000.0
            for name in ORACLES:
                T = corrected_T(name, r, par)
                e = T - T_ref
                out.append({
                    "sequence": seq, "camera": cam, "hand": r["hand"],
                    "frame": r["frame"], "participant": r["participant"],
                    "outer_fold": r["outer_fold"], "oracle": name,
                    "root_error_mm": float(np.linalg.norm(e)) * 1000.0,
                    "xy_error_mm": float(np.hypot(e[0], e[1])) * 1000.0,
                    "depth_error_mm": float(abs(e[2])) * 1000.0,
                    "absolute_mpjpe_mm": float(np.mean(
                        np.linalg.norm(Rj[m] + e, axis=1))) * 1000.0,
                    "root_aligned_mpjpe_mm": ra,
                })
    write_csv(RAW / "sequence_scale_parameters.csv.gz", params)
    write_csv(RAW / "oracle_corrected_frame_results.csv.gz", out)

    # ---- unit-level medians then camera-clustered paired stats
    METRICS = ["root_error_mm", "xy_error_mm", "depth_error_mm",
               "absolute_mpjpe_mm", "root_aligned_mpjpe_mm"]

    def unit_med(oracle, metric):
        by = defaultdict(list)
        for r in out:
            if r["oracle"] == oracle:
                by[(r["sequence"], r["camera"])].append(r[metric])
        return {k: float(np.median(v)) for k, v in by.items()}

    summary, paired = [], {}
    base = {m: unit_med("O0_ORACLE_FOCAL_ONLY", m) for m in METRICS}
    for name in ORACLES:
        rec = {"oracle": name,
               "n_eval_frames": sum(1 for r in out if r["oracle"] == name)}
        for m in METRICS:
            um = unit_med(name, m)
            rec["median_" + m] = med(list(um.values()))
            rec["p75_" + m] = pct(list(um.values()), 75)
            rec["p90_" + m] = pct(list(um.values()), 90)
        summary.append(rec)
        if name == "O0_ORACLE_FOCAL_ONLY":
            continue
        pr = {}
        for m in ("root_error_mm", "depth_error_mm", "absolute_mpjpe_mm"):
            a, b = base[m], unit_med(name, m)
            ks = sorted(set(a) & set(b))
            gains = [a[k] - b[k] for k in ks]
            ptv, lo, hi = paired_cluster_bootstrap(gains, [k[1] for k in ks])
            m0, m1 = med([a[k] for k in ks]), med([b[k] for k in ks])
            pr[m] = {"n_units": len(ks), "O0_median": m0, "median": m1,
                     "relative_reduction_pct": (100.0 * (m0 - m1) / m0
                                                if m0 else np.nan),
                     "paired_median_gain_mm": ptv, "ci95": [lo, hi]}
        paired[name] = pr
    write_csv(SUM / "oracle_hierarchy_summary.csv", summary)
    write_csv(TAB / "oracle_hierarchy_main.csv", summary)
    write_json(SUM / "oracle_paired_summary.json", paired)

    # ---- diagnostic tags, thresholds frozen in common.GATE
    o1 = paired["O1_SEQUENCE_MULTIPLICATIVE_Z_SCALE"]
    o2 = paired["O2_SEQUENCE_AFFINE_Z"]
    stable = bool(
        o1["depth_error_mm"]["relative_reduction_pct"]
        >= GATE["seq_scale_depth_reduction_pct"]
        and o1["root_error_mm"]["relative_reduction_pct"]
        >= GATE["seq_scale_root_reduction_pct"]
        and o1["depth_error_mm"]["ci95"][0] > 0)
    d1 = o1["depth_error_mm"]["median"]
    d2 = o2["depth_error_mm"]["median"]
    affine_adds = bool(np.isfinite(d1) and d1 > 0
                       and (100.0 * (d1 - d2) / d1)
                       >= GATE["affine_adds_value_pct"])
    o6 = next(s for s in summary
              if s["oracle"] == "O6_PER_FRAME_ROOT_TRANSLATION_ORACLE")
    consistency = abs(o6["median_absolute_mpjpe_mm"]
                      - o6["median_root_aligned_mpjpe_mm"])

    write_json(SUM / "oracle_tags.json", {
        "SEQUENCE_DEPTH_SCALE_STABLE": stable,
        "DEPTH_OFFSET_ADDS_VALUE": affine_adds,
        "MULTIPLICATIVE_SCALE_DOMINATES": bool(not affine_adds),
        "O6_absolute_vs_root_aligned_gap_mm": consistency,
        "O6_consistency_ok": bool(consistency
                                  <= GATE["root_oracle_mpjpe_tolerance_mm"]),
        "thresholds": GATE,
        "all_corrections_are_oracle_diagnostics": True,
    })

    for s in summary:
        print("%-38s root %7.1f  xy %7.1f  z %7.1f  absMPJPE %7.1f"
              % (s["oracle"], s["median_root_error_mm"],
                 s["median_xy_error_mm"], s["median_depth_error_mm"],
                 s["median_absolute_mpjpe_mm"]))
    print("\nSEQUENCE_DEPTH_SCALE_STABLE:", stable,
          "| DEPTH_OFFSET_ADDS_VALUE:", affine_adds)
    print("O6 |absMPJPE - rootAligned| = %.3f mm" % consistency)


if __name__ == "__main__":
    main()
