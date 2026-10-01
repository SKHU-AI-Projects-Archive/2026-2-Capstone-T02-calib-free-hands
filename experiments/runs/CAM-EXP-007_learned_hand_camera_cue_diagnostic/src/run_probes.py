"""PHASE B: load the CAM-EXP-005 target and run every probe, baseline, control.

This is the first CAM-007 file that reads the target. Everything it consumes -
the frame grid, the features, the latent layer, the splits, the probe list -
was frozen before this ran.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ALPHAS, CACHE, CAM005_KEPT, MANIFESTS, PCA_DIM, R005,  # noqa: E402
                    RAW, SEED, SUM, VIEW_TARGETS, fnum, pca_fit,
                    pca_transform, read_csv, read_json, ridge_fit,
                    ridge_predict, write_csv, write_json)

SPLITS = MANIFESTS / "cam_exp_007_outer_splits_v1.json"
SCENE_CSV = R005 / "results" / "summary" / "view_scene_features.csv.gz"
SCENE_DEFS = R005 / "tables" / "scene_feature_definitions.csv"
SCENE_GROUPS = ("F_SCENE_GEOM", "F_IMAGE_GLOBAL", "F_BORDER_SCENE")

PROBES = [
    ("P0", "BASELINE_TRAINING_MEAN", []),
    ("P1", "F0_HAND_VISIBILITY_AND_BOX", ["F0"]),
    ("P2", "F1_CAMERA_HEAD_OUTPUT", ["F1"]),
    ("P3", "F2_EXPLICIT_HAND_GEOMETRY", ["F2"]),
    ("P4", "F3_TEMPORAL_HAND_STABILITY", ["F3"]),
    ("P5", "ALL_EXPLICIT_HAND_OUTPUTS", ["F0", "F1", "F2", "F3"]),
    ("P6", "LEARNED_HAND_LATENT", ["F4"]),
    ("P7", "ALL_EXPLICIT_PLUS_LATENT", ["F0", "F1", "F2", "F3", "F4"]),
    ("P8", "SCENE_PLUS_ALL_EXPLICIT", ["SCENE", "F0", "F1", "F2", "F3"]),
    ("P9", "SCENE_PLUS_LATENT", ["SCENE", "F4"]),
    ("P10", "SCENE_PLUS_ALL_HAND", ["SCENE", "F0", "F1", "F2", "F3", "F4"]),
    ("B2", "CAM005_FROZEN_SCENE_BASELINE", ["SCENE"]),
    ("C1", "ANYCALIB_FOCAL_PRIOR_CONTROL", ["ANYFOCAL"]),
]

F1_KEYS = ("pred_cam_s", "pred_cam_tx", "pred_cam_ty")
F2_KEYS = ("hand_diameter", "depth_extent", "lateral_extent",
           "depth_over_diameter", "palm_normal_z", "finger_spread",
           "bone_ratio_mean", "bone_ratio_std", "global_orient_aa_norm",
           "hand_pose_aa_norm", "hand_pose_aa_mean_abs")
F0_KEYS = ("bbox_area_norm", "bbox_w_over_W", "bbox_h_over_H", "bbox_aspect",
           "bbox_cx_norm", "bbox_cy_norm", "box_size_norm", "det_score")
F0_VIEW = ("availability_rate", "mean_hands_per_frame", "side_consistency",
           "n_hands_std")
F3_VIEW = ("beta_std_mean", "availability_rate", "side_consistency",
           "n_hands_std")


def assign_group(col):
    """Map a hand feature column to its frozen group."""
    if col.endswith("__std"):
        return "F3"
    base = col.split("__")[0]
    if col in F3_VIEW and col not in F0_VIEW:
        return "F3"
    if col in F0_VIEW:
        return "F0"
    if base in F1_KEYS:
        return "F1"
    if base in F2_KEYS or base.startswith("mano_beta_"):
        return "F2"
    if base in F0_KEYS:
        return "F0"
    return None


def inner_cv_alpha(X, y, groups, alphas=ALPHAS, n_splits=5):
    """Grouped CV inside the training fold only. The outer test is unseen."""
    uniq = sorted(set(groups))
    if len(uniq) < 2:
        return alphas[len(alphas) // 2]
    rng = np.random.default_rng(SEED)
    order = list(uniq)
    rng.shuffle(order)
    folds = [order[i::min(n_splits, len(order))]
             for i in range(min(n_splits, len(order)))]
    best, best_err = alphas[0], np.inf
    for a in alphas:
        errs = []
        for f in folds:
            te = np.array([g in f for g in groups])
            if te.all() or (~te).all():
                continue
            w, ym, xm = ridge_fit(X[~te], y[~te], a)
            errs.append(np.mean(np.abs(ridge_predict(X[te], w, ym, xm)
                                       - y[te])))
        if errs and np.mean(errs) < best_err:
            best, best_err = a, float(np.mean(errs))
    return best


def prep_fit(Xtr, is_latent):
    """Imputation + standardisation (+PCA for latent), TRAINING FOLD ONLY."""
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    A = np.where(np.isfinite(Xtr), Xtr, med)
    mu, sd = A.mean(0), A.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    A = (A - mu) / sd
    pca = pca_fit(A, PCA_DIM) if is_latent and A.shape[1] > PCA_DIM else None
    return {"med": med, "mu": mu, "sd": sd, "pca": pca}


def prep_apply(X, p):
    A = np.where(np.isfinite(X), X, p["med"])
    A = (A - p["mu"]) / p["sd"]
    if p["pca"] is not None:
        A = pca_transform(A, *p["pca"])
    return A


def main() -> None:
    # ---------------- features (frozen, target-blind) ------------------
    hand = {(r["sequence"], r["camera"]): r
            for r in read_csv(RAW / "view_hand_features.csv.gz")}
    scene = {(r["sequence"], r["camera"]): r for r in read_csv(SCENE_CSV)}
    kept = set(read_json(CAM005_KEPT)["kept"])
    scene_feats = [r["feature"] for r in read_csv(SCENE_DEFS)
                   if r["group"] in SCENE_GROUPS]
    # EXACT reuse of CAM-EXP-005's frozen deployable scene set (its P6):
    # the 31 features of F_SCENE_GEOM + F_IMAGE_GLOBAL + F_BORDER_SCENE that
    # survived CAM-005's own quality audit. Matching on the exact column name
    # matters - matching on the base name would also pull in the __iqr
    # variants and silently redefine the frozen baseline as 62 features.
    cols0 = list(next(iter(scene.values())).keys())
    sf = set(scene_feats)
    scene_cols = [c for c in cols0 if c in sf and c in kept]
    lat_npz = np.load(CACHE / "view_latent.npz") if (
        CACHE / "view_latent.npz").exists() else None

    hand_cols = [c for c in next(iter(hand.values()))
                 if assign_group(c) is not None]
    group_cols = defaultdict(list)
    for c in hand_cols:
        group_cols[assign_group(c)].append(c)

    # ---------------- target (PHASE B) ----------------------------------
    tgt = {(r["sequence"], r["camera"]): r for r in read_csv(VIEW_TARGETS)}

    views = [v for v in sorted(tgt) if v in hand and v in scene]
    eligible = [v for v in views
                if hand[v]["hand_feature_eligible"] == "1"
                and (lat_npz is None or f"{v[0]}|{v[1]}" in lat_npz)]
    y = np.array([fnum(tgt[v]["anycalib_signed_log_bias"]) for v in eligible])
    f_any = np.array([fnum(tgt[v]["anycalib_view_focal_px"])
                      for v in eligible])
    f_ref = np.array([fnum(tgt[v]["gt_reference_focal_px"]) for v in eligible])
    cams = np.array([tgt[v]["physical_camera_id"] for v in eligible])
    seqs = np.array([v[0] for v in eligible])

    def matrix(groups):
        blocks, names = [], []
        for g in groups:
            if g == "SCENE":
                blocks.append(np.array([[fnum(scene[v][c]) for c in scene_cols]
                                        for v in eligible]))
                names += [f"SCENE::{c}" for c in scene_cols]
            elif g == "F4":
                blocks.append(np.array([lat_npz[f"{v[0]}|{v[1]}"]
                                        for v in eligible]))
                names += [f"LATENT::{i}"
                          for i in range(blocks[-1].shape[1])]
            elif g == "ANYFOCAL":
                blocks.append(np.log(f_any).reshape(-1, 1))
                names += ["log_f_anycalib"]
            else:
                cs = group_cols[g]
                blocks.append(np.array([[fnum(hand[v][c]) for c in cs]
                                        for v in eligible]))
                names += [f"{g}::{c}" for c in cs]
        if not blocks:
            return np.zeros((len(eligible), 0)), []
        return np.hstack(blocks), names

    splits = read_json(SPLITS)
    protocols = {
        "LOCO_PHYSICAL_CAMERA": ("held_out_physical_camera", cams),
        "LOSO_SEQUENCE": ("held_out_sequence", seqs),
    }

    preds_rows, shuffle_rows = [], []
    rng_global = np.random.default_rng(SEED)

    for proto, (key, labels) in protocols.items():
        folds = splits[proto]["folds"]
        for pid, pname, groups in PROBES:
            is_latent = "F4" in groups
            X, names = matrix(groups)
            for fold in folds:
                te = labels == fold[key]
                if te.sum() == 0 or (~te).sum() < 5:
                    continue
                ytr = y[~te]
                if X.shape[1] == 0:
                    yhat = np.full(te.sum(), ytr.mean())
                    alpha = ""
                else:
                    p = prep_fit(X[~te], is_latent)
                    A, B = prep_apply(X[~te], p), prep_apply(X[te], p)
                    alpha = inner_cv_alpha(A, ytr, labels[~te])
                    w, ym, xm = ridge_fit(A, ytr, alpha)
                    yhat = ridge_predict(B, w, ym, xm)
                for j, v in enumerate(np.flatnonzero(te)):
                    preds_rows.append({
                        "protocol": proto, "probe": pid, "probe_name": pname,
                        "fold": fold["fold"], "held_out": fold[key],
                        "sequence": eligible[v][0], "camera": eligible[v][1],
                        "physical_camera_id": cams[v],
                        "n_features": X.shape[1], "alpha": alpha,
                        "y_true": y[v], "y_pred": float(yhat[j]),
                        "f_anycalib": f_any[v], "f_reference": f_ref[v],
                        "f_corrected": float(f_any[v] / np.exp(yhat[j])),
                        "train_bias_median": float(np.median(ytr)),
                    })

        # ---- baselines that are not ridge probes -----------------------
        for fold in folds:
            te = labels == fold[key]
            if te.sum() == 0 or (~te).sum() < 5:
                continue
            ytr = y[~te]
            gb = float(np.median(ytr))
            ref_med = float(np.median(f_ref[~te]))
            for v in np.flatnonzero(te):
                base = {"protocol": proto, "fold": fold["fold"],
                        "held_out": fold[key],
                        "sequence": eligible[v][0], "camera": eligible[v][1],
                        "physical_camera_id": cams[v], "n_features": 0,
                        "alpha": "", "y_true": y[v],
                        "f_anycalib": f_any[v], "f_reference": f_ref[v],
                        "train_bias_median": gb}
                preds_rows.append({**base, "probe": "B0",
                                   "probe_name": "RAW_ANYCALIB",
                                   "y_pred": 0.0,
                                   "f_corrected": f_any[v]})
                preds_rows.append({**base, "probe": "B1",
                                   "probe_name":
                                       "GLOBAL_BIAS_CORRECTED_ANYCALIB",
                                   "y_pred": gb,
                                   "f_corrected": float(f_any[v]
                                                        / np.exp(gb))})
                preds_rows.append({**base, "probe": "B3",
                                   "probe_name":
                                       "TRAINING_FOLD_CONSTANT_REFERENCE"
                                       "_ORACLE",
                                   "y_pred": float(np.log(f_any[v] / ref_med)),
                                   "f_corrected": ref_med})

    # ---------------- controls --------------------------------------------
    key, labels = "held_out_physical_camera", cams
    folds = splits["LOCO_PHYSICAL_CAMERA"]["folds"]
    for pid, pname, groups in [("P5", "ALL_EXPLICIT_HAND_OUTPUTS",
                                ["F0", "F1", "F2", "F3"]),
                               ("P6", "LEARNED_HAND_LATENT", ["F4"]),
                               ("P7", "ALL_EXPLICIT_PLUS_LATENT",
                                ["F0", "F1", "F2", "F3", "F4"])]:
        is_latent = "F4" in groups
        X, _ = matrix(groups)
        for seed_i in range(10):
            rng = np.random.default_rng(SEED + 1000 * seed_i)
            for fold in folds:
                te = labels == fold[key]
                if te.sum() == 0 or (~te).sum() < 5:
                    continue
                Xs = X.copy()
                # permute rows WITHIN each partition; targets untouched
                for part in (np.flatnonzero(~te), np.flatnonzero(te)):
                    bys = defaultdict(list)
                    for i in part:
                        bys[seqs[i]].append(i)
                    for _s, idxs in bys.items():
                        if len(idxs) >= 4:
                            perm = rng.permutation(idxs)
                            Xs[idxs] = X[perm]
                    small = [i for _s, idxs in bys.items() if len(idxs) < 4
                             for i in idxs]
                    if len(small) >= 2:
                        perm = rng.permutation(small)
                        Xs[small] = X[perm]
                ytr = y[~te]
                p = prep_fit(Xs[~te], is_latent)
                A, B = prep_apply(Xs[~te], p), prep_apply(Xs[te], p)
                alpha = inner_cv_alpha(A, ytr, labels[~te])
                w, ym, xm = ridge_fit(A, ytr, alpha)
                yhat = ridge_predict(B, w, ym, xm)
                for j, v in enumerate(np.flatnonzero(te)):
                    shuffle_rows.append({
                        "control": "C2_HAND_FEATURE_VIEW_SHUFFLE",
                        "probe": pid, "seed": seed_i, "fold": fold["fold"],
                        "sequence": eligible[v][0], "camera": eligible[v][1],
                        "physical_camera_id": cams[v],
                        "y_true": y[v], "y_pred": float(yhat[j]),
                        "f_anycalib": f_any[v], "f_reference": f_ref[v],
                        "f_corrected": float(f_any[v] / np.exp(yhat[j])),
                    })

        # C3 random features, matched dimension
        for fold in folds:
            te = labels == fold[key]
            if te.sum() == 0 or (~te).sum() < 5:
                continue
            R = np.vstack([np.random.default_rng(
                abs(hash(f"{eligible[i][0]}|{eligible[i][1]}")) % (2**31)
            ).normal(size=X.shape[1]) for i in range(len(eligible))])
            ytr = y[~te]
            p = prep_fit(R[~te], is_latent)
            A, B = prep_apply(R[~te], p), prep_apply(R[te], p)
            alpha = inner_cv_alpha(A, ytr, labels[~te])
            w, ym, xm = ridge_fit(A, ytr, alpha)
            yhat = ridge_predict(B, w, ym, xm)
            for j, v in enumerate(np.flatnonzero(te)):
                shuffle_rows.append({
                    "control": "C3_RANDOM_FEATURE", "probe": pid,
                    "seed": 0, "fold": fold["fold"],
                    "sequence": eligible[v][0], "camera": eligible[v][1],
                    "physical_camera_id": cams[v],
                    "y_true": y[v], "y_pred": float(yhat[j]),
                    "f_anycalib": f_any[v], "f_reference": f_ref[v],
                    "f_corrected": float(f_any[v] / np.exp(yhat[j])),
                })

        # C4 train-target shuffle
        for fold in folds:
            te = labels == fold[key]
            if te.sum() == 0 or (~te).sum() < 5:
                continue
            ytr = rng_global.permutation(y[~te])
            p = prep_fit(X[~te], is_latent)
            A, B = prep_apply(X[~te], p), prep_apply(X[te], p)
            alpha = inner_cv_alpha(A, ytr, labels[~te])
            w, ym, xm = ridge_fit(A, ytr, alpha)
            yhat = ridge_predict(B, w, ym, xm)
            for j, v in enumerate(np.flatnonzero(te)):
                shuffle_rows.append({
                    "control": "C4_TRAIN_TARGET_SHUFFLE", "probe": pid,
                    "seed": 0, "fold": fold["fold"],
                    "sequence": eligible[v][0], "camera": eligible[v][1],
                    "physical_camera_id": cams[v],
                    "y_true": y[v], "y_pred": float(yhat[j]),
                    "f_anycalib": f_any[v], "f_reference": f_ref[v],
                    "f_corrected": float(f_any[v] / np.exp(yhat[j])),
                })

    write_csv(RAW / "probe_fold_predictions.csv.gz", preds_rows)
    write_csv(RAW / "shuffle_control_predictions.csv.gz", shuffle_rows)
    write_json(SUM / "probe_run_meta.json", {
        "views_with_target_and_features": len(views),
        "views_eligible_common_set": len(eligible),
        "n_scene_features": len(scene_cols),
        "n_hand_feature_columns": len(hand_cols),
        "group_sizes": {g: len(c) for g, c in sorted(group_cols.items())},
        "latent_dim": int(lat_npz[next(iter(lat_npz.files))].shape[0])
        if lat_npz is not None else 0,
        "pca_dim": PCA_DIM,
        "prediction_rows": len(preds_rows),
        "control_rows": len(shuffle_rows),
    })
    print(f"eligible views {len(eligible)}  scene feats {len(scene_cols)}  "
          f"hand cols {len(hand_cols)}")
    print("group sizes:", {g: len(c) for g, c in sorted(group_cols.items())})
    print(f"rows: probes {len(preds_rows)}, controls {len(shuffle_rows)}")


if __name__ == "__main__":
    main()
