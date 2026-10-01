"""LINEAR PROBE DIAGNOSTIC.

Asks one question per feature group: does it carry cross-view predictive
information about the per-view bias? It is not a proposed calibration method
and is never called one.

Two outer validations, both frozen in
experiments/manifests/cam_exp_005_outer_splits_v1.json:

  A_LOSO  leave one sequence out           - content / sequence shift
  B_LOCO  leave one physical camera out    - NEW CAMERA, the primary criterion

Imputation, standardisation and the ridge alpha are all fitted inside the
training fold. The held-out fold is never touched before prediction.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (MANIFESTS, RAW, SEED, SUM, cluster_bootstrap, read_csv,  # noqa: E402
                    read_json, write_csv, write_json)

SPEC = read_json(MANIFESTS / "cam_exp_005_probe_spec_v1.json")
ALPHAS = SPEC["alpha_grid"]
TARGET = "anycalib_signed_log_bias"


def load():
    views = read_csv(SUM / "view_scene_features.csv.gz")
    kept = set(read_json(SUM / "_kept_features.json")["kept"])
    fspec = read_json(MANIFESTS / "cam_exp_005_scene_feature_spec_v1.json")
    groups = {g: [f["name"] for f in fs if f["name"] in kept]
              for g, fs in fspec["groups"].items()}
    return views, groups


def matrix(views, names):
    X = np.full((len(views), len(names)), np.nan)
    for j, n in enumerate(names):
        for i, v in enumerate(views):
            try:
                X[i, j] = float(v[n])
            except (KeyError, TypeError, ValueError):
                X[i, j] = np.nan
    return X


def fit_predict(Xtr, ytr, Xte, groups_tr):
    """Ridge with training-fold-only preprocessing and alpha selection."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold

    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xtr = np.where(np.isfinite(Xtr), Xtr, med)
    Xte = np.where(np.isfinite(Xte), Xte, med)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd

    n_groups = len(set(groups_tr))
    k = min(5, n_groups)
    best_a, best_err = ALPHAS[0], np.inf
    if k >= 2:
        gkf = GroupKFold(n_splits=k)
        for a in ALPHAS:
            errs = []
            for itr, iva in gkf.split(Ztr, ytr, groups=groups_tr):
                m = Ridge(alpha=a, random_state=None)
                m.fit(Ztr[itr], ytr[itr])
                errs.append(np.abs(m.predict(Ztr[iva]) - ytr[iva]).mean())
            e = float(np.mean(errs))
            if e < best_err:
                best_a, best_err = a, e
    model = Ridge(alpha=best_a)
    model.fit(Ztr, ytr)
    return model.predict(Zte), best_a


def run_protocol(views, groups, folds, fold_key, protocol_name):
    y = np.array([float(v[TARGET]) for v in views])
    cams = np.array([v["physical_camera_id"] for v in views])
    seqs = np.array([v["sequence"] for v in views])
    keyvals = cams if fold_key == "camera" else seqs

    sets = {"P0_baseline": []}
    for pname, glist in SPEC["feature_sets"].items():
        if pname == "P0_baseline":
            continue
        names = [n for g in glist for n in groups.get(g, [])]
        if names:
            sets[pname] = names
    # diagnostic-only reference models
    diag = {"D_CAMERA_ID": "camera", "D_SEQUENCE_ID": "sequence"}

    rows, preds = [], []
    for pname, names in list(sets.items()) + [(d, None) for d in diag]:
        yhat = np.full(len(views), np.nan)
        alphas = []
        for fold in folds:
            held = fold["held_out_views"]
            mask = np.zeros(len(views), bool)
            hs = {(a, b) for a, b in held}
            for i, v in enumerate(views):
                if (v["sequence"], v["camera"]) in hs:
                    mask[i] = True
            tr, te = ~mask, mask
            if te.sum() == 0 or tr.sum() < 5:
                continue
            if pname == "P0_baseline":
                yhat[te] = y[tr].mean()
                continue
            if pname in diag:
                # one-hot of the grouping variable, fitted on the training fold.
                # For the leave-one-*-out protocol that matches the variable,
                # the held-out level is unseen, so it falls back to the mean -
                # which is exactly the point of the diagnostic.
                src = cams if diag[pname] == "camera" else seqs
                levels = sorted(set(src[tr]))
                Xall = np.array([[1.0 * (s == lv) for lv in levels] for s in src])
                yhat[te], a = fit_predict(Xall[tr], y[tr], Xall[te], cams[tr])
                alphas.append(a)
                continue
            X = matrix(views, names)
            yhat[te], a = fit_predict(X[tr], y[tr], X[te], cams[tr])
            alphas.append(a)

        ok = np.isfinite(yhat)
        err = np.abs(yhat[ok] - y[ok])
        sse = float(((y[ok] - yhat[ok]) ** 2).sum())
        sst = float(((y[ok] - y[ok].mean()) ** 2).sum())
        from scipy import stats
        rho = (float(stats.spearmanr(yhat[ok], y[ok]).statistic)
               if np.std(yhat[ok]) > 0 else float("nan"))
        rows.append({
            "protocol": protocol_name, "feature_set": pname,
            "n_features": 0 if names is None else len(names),
            "n_views_predicted": int(ok.sum()),
            "mae_log_bias": float(err.mean()),
            "median_abs_log_bias_error": float(np.median(err)),
            "r2": float(1 - sse / sst) if sst > 0 else float("nan"),
            "spearman_pred_vs_actual": rho,
            "alpha_mode": (max(set(alphas), key=alphas.count) if alphas else ""),
            "status": ("ORACLE_OR_RIG_SPECIFIC_DIAGNOSTIC"
                       if pname == "D_CAMERA_ID" else
                       "SEQUENCE_SPECIFIC_DIAGNOSTIC" if pname == "D_SEQUENCE_ID"
                       else "deployable feature group"),
        })
        for i, v in enumerate(views):
            if ok[i]:
                preds.append({"protocol": protocol_name, "feature_set": pname,
                              "sequence": v["sequence"], "camera": v["camera"],
                              "actual_log_bias": float(y[i]),
                              "predicted_log_bias": float(yhat[i])})
    return rows, preds


def correction_diagnostic(views, preds, protocol_name, folds):
    """LINEAR_PROBE_CORRECTION_DIAGNOSTIC: f_corr = f_pred / exp(b_hat).

    A diagnostic reading of the probe, not a calibration method.
    """
    by_set = defaultdict(dict)
    for p in preds:
        if p["protocol"] != protocol_name:
            continue
        by_set[p["feature_set"]][(p["sequence"], p["camera"])] = p["predicted_log_bias"]
    base = {(v["sequence"], v["camera"]): v for v in views}
    cams = {(v["sequence"], v["camera"]): v["physical_camera_id"] for v in views}

    rows = []
    orig = np.array([float(v["anycalib_rel_focal_err_pct"]) for v in views])
    rows.append(_summarise("ORIGINAL_uncorrected", protocol_name, orig,
                           [v["physical_camera_id"] for v in views], 0))

    # The decisive control. This rig uses essentially one lens, so an estimator
    # that ignores the image entirely and always outputs the training folds'
    # median reference focal is a very strong "predictor" here. Any probe that
    # does not clearly beat it has not shown a calibration signal - it has shown
    # that it can memorise this rig's focal length. Uses GT, so oracle-only.
    const_err, const_cams = [], []
    for fold in folds:
        hs = {(a, b) for a, b in fold["held_out_views"]}
        tr = [v for v in views if (v["sequence"], v["camera"]) not in hs]
        te = [v for v in views if (v["sequence"], v["camera"]) in hs]
        if not tr or not te:
            continue
        f_const = float(np.median([float(v["gt_reference_focal_px"]) for v in tr]))
        for v in te:
            f_ref = float(v["gt_reference_focal_px"])
            const_err.append(abs(f_const - f_ref) / f_ref * 100)
            const_cams.append(v["physical_camera_id"])
    if const_err:
        r = _summarise("CONSTANT_RIG_FOCAL_ORACLE", protocol_name,
                       np.array(const_err), const_cams, len(const_err))
        r["status"] = ("ORACLE_DIAGNOSTIC_ONLY - ignores the image entirely and "
                       "outputs the training folds' median reference focal; the "
                       "control that any probe must beat on this single-lens rig")
        rows.append(r)
    for pname, d in sorted(by_set.items()):
        errs, cc = [], []
        for key, bhat in d.items():
            v = base[key]
            f_pred = float(v["anycalib_view_focal_px"])
            f_ref = float(v["gt_reference_focal_px"])
            f_corr = f_pred / np.exp(bhat)
            errs.append(abs(f_corr - f_ref) / f_ref * 100)
            cc.append(cams[key])
        rows.append(_summarise(pname, protocol_name, np.array(errs), cc, len(d)))
    return rows


def _summarise(name, protocol, err, cams, n):
    pt, lo, hi = cluster_bootstrap(err, cams, stat=np.median, n_boot=2000)
    return {
        "protocol": protocol, "feature_set": name, "n_views": int(err.size),
        "median_rel_focal_err_pct": float(np.median(err)),
        "median_ci_lo_camera_cluster": lo, "median_ci_hi_camera_cluster": hi,
        "mean_rel_focal_err_pct": float(err.mean()),
        "within_5_pct": float((err <= 5).mean() * 100),
        "within_10_pct": float((err <= 10).mean() * 100),
        "status": "LINEAR_PROBE_CORRECTION_DIAGNOSTIC - not a calibration method",
    }


def main() -> None:
    views, groups = load()
    splits = read_json(MANIFESTS / "cam_exp_005_outer_splits_v1.json")

    all_preds = []
    for protocol, folds, key, out_name in (
            ("A_LOSO", splits["A_LOSO_folds"], "sequence", "probe_summary_loso.csv"),
            ("B_LOCO", splits["B_LOCO_folds"], "camera",
             "probe_summary_leave_camera_out.csv")):
        rows, preds = run_protocol(views, groups, folds, key, protocol)
        write_csv(SUM / out_name, rows)
        all_preds.extend(preds)
        print(f"=== {protocol} ===")
        for r in rows:
            print(f"  {r['feature_set']:16s} n_feat={r['n_features']:3d} "
                  f"MAE={r['mae_log_bias']:.5f} R2={r['r2']:+.3f} "
                  f"rho={r['spearman_pred_vs_actual']:+.3f}  {r['status']}")
    write_csv(RAW / "probe_fold_predictions.csv.gz", all_preds)

    corr = []
    for protocol in ("A_LOSO", "B_LOCO"):
        folds = (splits["A_LOSO_folds"] if protocol == "A_LOSO"
                 else splits["B_LOCO_folds"])
        corr.extend(correction_diagnostic(views, all_preds, protocol, folds))
    write_csv(SUM / "probe_correction_summary.csv", corr)
    print("=== correction diagnostic ===")
    for r in corr:
        print(f"  {r['protocol']:7s} {r['feature_set']:20s} "
              f"median={r['median_rel_focal_err_pct']:6.2f}% "
              f"[{r['median_ci_lo_camera_cluster']:5.2f},"
              f"{r['median_ci_hi_camera_cluster']:5.2f}] "
              f"w5={r['within_5_pct']:5.1f}% w10={r['within_10_pct']:5.1f}%")


if __name__ == "__main__":
    main()
