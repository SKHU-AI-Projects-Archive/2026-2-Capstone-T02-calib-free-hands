"""Build the per-view bias target and the camera/sequence structural analysis.

No image is read here. Everything comes from the frozen CAM-EXP-004.1
predictions, so the target cannot drift from what was already reported.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (PRED64, RAW, SUM, TAB, cluster_bootstrap, load_view_targets,  # noqa: E402
                    read_csv, write_csv, write_json)


def structural_summary(views):
    seqs = sorted({v["sequence"] for v in views})
    cams = sorted({v["physical_camera_id"] for v in views})
    per_cam = defaultdict(list)
    for v in views:
        per_cam[v["physical_camera_id"]].append(v)
    return seqs, cams, per_cam


# ------------------------------------------- A. camera repeatability across seq
def camera_repeatability(views, target="anycalib_signed_log_bias"):
    per_cam = defaultdict(list)
    for v in views:
        if np.isfinite(v.get(target, np.nan)):
            per_cam[v["physical_camera_id"]].append(v)
    rows = []
    for cam, vs in sorted(per_cam.items()):
        b = np.array([v[target] for v in vs])
        signs = np.sign(b)
        rows.append({
            "physical_camera_id": cam,
            "n_sequences": len(vs),
            "sequences": ";".join(sorted(v["sequence"] for v in vs)),
            "mean_signed_log_bias": float(b.mean()),
            "median_signed_log_bias": float(np.median(b)),
            "between_sequence_sd_log": float(np.std(b, ddof=1)) if b.size > 1 else float("nan"),
            "min_signed_log_bias": float(b.min()),
            "max_signed_log_bias": float(b.max()),
            "range_log": float(b.max() - b.min()),
            "sign_consistent": int(bool(np.all(signs == signs[0]))),
            "mean_signed_pct": float((np.exp(b.mean()) - 1) * 100),
        })
    write_csv(SUM / "camera_across_sequence_repeatability.csv", rows)
    return rows


def sequence_pair_correlations(views, target="anycalib_signed_log_bias"):
    """For every pair of sequences, correlate the bias of their shared cameras."""
    from scipy import stats
    by_seq = defaultdict(dict)
    for v in views:
        if np.isfinite(v.get(target, np.nan)):
            by_seq[v["sequence"]][v["physical_camera_id"]] = v[target]
    rows = []
    for a, b in combinations(sorted(by_seq), 2):
        common = sorted(set(by_seq[a]) & set(by_seq[b]))
        if len(common) < 4:
            rows.append({"sequence_a": a, "sequence_b": b,
                         "n_common_cameras": len(common),
                         "status": "TOO_FEW_COMMON_CAMERAS"})
            continue
        xa = np.array([by_seq[a][c] for c in common])
        xb = np.array([by_seq[b][c] for c in common])
        pr = stats.pearsonr(xa, xb)
        sp = stats.spearmanr(xa, xb)
        rows.append({
            "sequence_a": a, "sequence_b": b, "n_common_cameras": len(common),
            "pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue),
            "spearman_rho": float(sp.statistic), "spearman_p": float(sp.pvalue),
            "median_abs_difference_log": float(np.median(np.abs(xa - xb))),
            "status": "ok",
        })
    write_csv(SUM / "sequence_pair_camera_correlations.csv", rows)
    return rows


def variance_split(views, target="anycalib_signed_log_bias"):
    """Between-camera vs within-camera-across-sequence variation.

    A one-way decomposition with the physical camera as the grouping factor:

        total  = sum over views of (b_v - grand_mean)^2
        within = sum over views of (b_v - camera_mean)^2
        between= total - within

    and an ICC-like ratio ICC_hat = between_ms_component / (between + within)
    computed from the one-way random-effects mean squares. The design is
    unbalanced and the sample is 40 cameras over 5 sequences, so this is
    reported as a descriptive repeatability index, not an inferential ICC.
    """
    per_cam = defaultdict(list)
    for v in views:
        if np.isfinite(v.get(target, np.nan)):
            per_cam[v["physical_camera_id"]].append(v[target])
    groups = [np.asarray(g) for g in per_cam.values() if len(g) >= 1]
    allv = np.concatenate(groups)
    grand = allv.mean()
    k = len(groups)
    n_tot = allv.size
    ssb = float(sum(g.size * (g.mean() - grand) ** 2 for g in groups))
    ssw = float(sum(((g - g.mean()) ** 2).sum() for g in groups))
    df_b, df_w = k - 1, n_tot - k
    msb = ssb / df_b if df_b > 0 else float("nan")
    msw = ssw / df_w if df_w > 0 else float("nan")
    sizes = np.array([g.size for g in groups], float)
    n0 = ((sizes.sum() - (sizes ** 2).sum() / sizes.sum()) / (k - 1)
          if k > 1 else float("nan"))
    var_b = max((msb - msw) / n0, 0.0) if np.isfinite(n0) and n0 > 0 else float("nan")
    icc = var_b / (var_b + msw) if np.isfinite(var_b) and (var_b + msw) > 0 else float("nan")
    return {
        "target": target, "n_views": int(n_tot), "n_cameras": k,
        "grand_mean_log": float(grand),
        "ss_between_camera": ssb, "ss_within_camera": ssw,
        "pct_ss_between_camera": float(ssb / (ssb + ssw) * 100),
        "ms_between": msb, "ms_within": msw, "n0_effective_group_size": float(n0),
        "icc_like_one_way_random": float(icc),
        "caveat": "unbalanced one-way decomposition over 40 cameras and 5 "
                  "sequences; descriptive repeatability index, not an "
                  "inferential ICC",
    }


# --------------------------------------- B. camera vs sequence descriptive fit
def descriptive_models(views, target="anycalib_signed_log_bias"):
    """M0 / M_SEQ / M_CAM / M_BOTH as plain least-squares group means.

    In-sample explanatory fit only. A one-hot camera model has 40 free
    parameters for 175 views, so its R^2 is an upper bound on what camera
    identity could explain here, not a causal variance share.
    """
    rows = [v for v in views if np.isfinite(v.get(target, np.nan))]
    y = np.array([v[target] for v in rows])
    seq = np.array([v["sequence"] for v in rows])
    cam = np.array([v["physical_camera_id"] for v in rows])

    def design(*factors):
        cols = [np.ones(len(y))]
        for f in factors:
            levels = sorted(set(f))[1:]      # drop one level as the reference
            for lv in levels:
                cols.append((f == lv).astype(float))
        return np.column_stack(cols)

    out = []
    for name, X in (("M0_intercept_only", np.ones((len(y), 1))),
                    ("M_SEQ_sequence_id", design(seq)),
                    ("M_CAM_physical_camera_id", design(cam)),
                    ("M_BOTH_sequence_plus_camera", design(seq, cam))):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        sse = float(((y - pred) ** 2).sum())
        sst = float(((y - y.mean()) ** 2).sum())
        p = X.shape[1]
        r2 = 1 - sse / sst if sst > 0 else float("nan")
        adj = (1 - (1 - r2) * (len(y) - 1) / (len(y) - p)
               if len(y) - p > 0 else float("nan"))
        out.append({
            "model": name, "n_parameters": int(p), "n_views": int(len(y)),
            "sse": sse, "sse_reduction_vs_M0": float(sst - sse),
            "r2_in_sample": float(r2), "adjusted_r2": float(adj),
            "mae_log": float(np.abs(y - pred).mean()),
            "mae_pct_equivalent": float(np.abs(np.exp(np.abs(y - pred)) - 1).mean() * 100),
            "interpretation": "descriptive explanatory fit, in-sample; NOT a "
                              "causal variance share",
            "status": ("DIAGNOSTIC_ONLY - camera identity is not available for a "
                       "new deployment camera"
                       if "CAM" in name else
                       "SEQUENCE_SPECIFIC_DIAGNOSTIC" if "SEQ" in name
                       else "baseline"),
        })
    write_csv(SUM / "camera_sequence_descriptive_models.csv", out)
    return out


def main() -> None:
    views = load_view_targets()
    write_csv(RAW / "view_targets.csv.gz", views)
    seqs, cams, per_cam = structural_summary(views)
    multi = {c: len(v) for c, v in per_cam.items() if len(v) > 1}
    print(f"views={len(views)}  sequences={len(seqs)}  physical cameras={len(cams)}"
          f"  cameras in >1 sequence={len(multi)}")

    b = np.array([v["anycalib_signed_log_bias"] for v in views])
    pt, lo, hi = cluster_bootstrap(b, [v["physical_camera_id"] for v in views])
    print(f"AnyCalib signed log bias: median {pt:+.4f} "
          f"[{lo:+.4f},{hi:+.4f}] (camera-cluster bootstrap) "
          f"= {(np.exp(pt) - 1) * 100:+.2f} %")

    rep = camera_repeatability(views)
    sign_ok = sum(r["sign_consistent"] for r in rep if r["n_sequences"] > 1)
    n_multi = sum(1 for r in rep if r["n_sequences"] > 1)
    print(f"cameras appearing in >1 sequence: {n_multi}; sign-consistent in "
          f"{sign_ok}/{n_multi}")

    pairs = sequence_pair_correlations(views)
    ok = [p for p in pairs if p.get("status") == "ok"]
    if ok:
        rr = np.array([p["pearson_r"] for p in ok])
        print(f"sequence-pair camera bias correlation: pearson r median "
              f"{np.median(rr):+.3f} range [{rr.min():+.3f},{rr.max():+.3f}] "
              f"over {len(ok)} pairs")

    vs = variance_split(views)
    write_json(SUM / "bias_variance_split.json", vs)
    print(f"between-camera SS share {vs['pct_ss_between_camera']:.1f} %, "
          f"ICC-like {vs['icc_like_one_way_random']:.3f}")

    for m in descriptive_models(views):
        print(f"  {m['model']:30s} p={m['n_parameters']:3d} "
              f"R2={m['r2_in_sample']:.3f} adjR2={m['adjusted_r2']:.3f} "
              f"MAE_log={m['mae_log']:.4f}")

    write_csv(TAB / "statistical_units.csv", [
        {"analysis": "per-view bias target",
         "unit": "(sequence, camera) static view", "n": len(views),
         "note": "median over the frozen 64-frame set; frames are repeated "
                 "observations, never independent samples"},
        {"analysis": "association bootstrap",
         "unit": "physical camera cluster", "n": len(cams),
         "note": "all views of a drawn camera enter together"},
        {"analysis": "sensitivity bootstrap", "unit": "sequence", "n": len(seqs),
         "note": "5 clusters - sensitivity only, not a trustworthy CI"},
        {"analysis": "outer validation A", "unit": "sequence fold", "n": len(seqs),
         "note": "content/sequence shift test; the same physical camera can "
                 "appear in train and test"},
        {"analysis": "outer validation B",
         "unit": "physical camera group", "n": len(cams),
         "note": "new-camera test; no view of the held-out camera is in train"},
    ])


if __name__ == "__main__":
    main()
