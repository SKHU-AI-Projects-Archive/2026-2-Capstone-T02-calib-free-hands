"""Freeze the scene-feature spec, the probe spec and the outer folds.

Run BEFORE any feature is correlated with the target and before any held-out
performance is inspected. The manifests record that, and their hashes go into
config.json.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (MANIFESTS, SEED, TAB, load_view_targets, sha256, write_csv,  # noqa: E402
                    write_json)

FEATURE_SPEC = MANIFESTS / "cam_exp_005_scene_feature_spec_v1.json"
PROBE_SPEC = MANIFESTS / "cam_exp_005_probe_spec_v1.json"
SPLIT_SPEC = MANIFESTS / "cam_exp_005_outer_splits_v1.json"

# Every feature is listed here before extraction. Nothing is added later on the
# strength of how well it correlates.
F_SCENE_GEOM = [
    ("edge_density", "fraction of Canny(50,150) edge pixels"),
    ("grad_mag_mean", "mean Sobel gradient magnitude"),
    ("grad_mag_p90", "90th percentile Sobel gradient magnitude"),
    ("grad_orientation_entropy", "Shannon entropy of the gradient-orientation "
                                 "histogram (36 bins, magnitude-weighted), bits"),
    ("line_count", "LSD line segments detected"),
    ("long_line_count", "segments longer than 10 % of the image diagonal"),
    ("total_line_length_over_area", "sum of segment lengths / (W*H)"),
    ("median_norm_line_length", "median segment length / image diagonal"),
    ("horizontal_line_fraction", "fraction of segments within 15 deg of horizontal"),
    ("vertical_line_fraction", "fraction of segments within 15 deg of vertical"),
    ("diagonal_line_fraction", "fraction of segments that are neither"),
    ("line_orientation_entropy", "Shannon entropy of the length-weighted segment "
                                 "orientation histogram (18 bins over 180 deg), bits"),
    ("dominant_line_orientation_strength", "largest bin share of that histogram"),
    ("corner_count_over_area", "goodFeaturesToTrack corners / (W*H) * 1e6"),
    ("vp_consensus_strength", "largest share of segments consistent with a single "
                              "RANSAC-like vanishing point (skipped if unstable)"),
    ("orthogonal_line_support", "share of segment pairs whose orientations differ "
                                "by 90 +/- 10 deg"),
]
F_IMAGE_GLOBAL = [
    ("gray_mean", "mean of the grayscale image"),
    ("gray_std", "standard deviation of the grayscale image"),
    ("gray_entropy", "Shannon entropy of the 256-bin grayscale histogram, bits"),
    ("saturation_mean", "mean HSV saturation"),
    ("saturation_std", "standard deviation of HSV saturation"),
    ("local_contrast", "mean of the local std in 16x16 blocks"),
    ("laplacian_variance", "variance of the Laplacian (focus/sharpness proxy)"),
]
BORDER_FRACTION = 0.20
F_BORDER_SCENE = [
    ("border_edge_density", "edge density in the outer 20 % border ring"),
    ("border_line_count", "LSD segments whose midpoint lies in the border ring"),
    ("border_long_line_count", "of those, longer than 10 % of the diagonal"),
    ("border_horizontal_line_fraction", "horizontal share among border segments"),
    ("border_vertical_line_fraction", "vertical share among border segments"),
    ("border_line_orientation_entropy", "orientation entropy of border segments"),
    ("border_grad_mag_mean", "mean gradient magnitude in the border ring"),
    ("border_gray_std", "grayscale std in the border ring"),
]
# Model self-outputs already present in the frozen CAM-EXP-004.1 predictions.
F_MODEL_SELF = [
    ("self_pred_focal_px", "median predicted focal over the view (AnyCalib)"),
    ("self_pred_focal_log_iqr", "IQR of log predicted focal within the view"),
    ("self_pred_k1", "median predicted radial k1"),
    ("self_pred_k2", "median predicted radial k2"),
    ("self_pred_k1_iqr", "IQR of predicted k1 within the view"),
    ("self_pp_offset_norm", "median |predicted principal point - image centre| / "
                            "image diagonal"),
    ("self_pp_offset_iqr", "IQR of that offset within the view"),
    ("self_fx_over_fy", "median predicted fx / fy"),
]
MODEL_DISAGREEMENT = [
    ("disagree_abs_log_any_vs_geo", "|log(f_AnyCalib / f_GeoCalib)| per frame, "
                                    "median over the view"),
    ("disagree_signed_log_any_vs_geo", "signed log ratio, median over the view"),
    ("disagree_log_iqr", "IQR of that signed log ratio within the view"),
]


def main() -> None:
    views = load_view_targets()

    feature_spec = {
        "id": "CAM_EXP_005_SCENE_FEATURE_SPEC_V1",
        "created_before_target_analysis": True,
        "frozen_on": "2026-09-24",
        "frames": "the frozen 64-frame nested set per view "
                  "(gigahands_demo_cam_exp_0041_64frames_v1.csv.gz). No frame is "
                  "chosen using the target, the ground truth or any model output.",
        "frame_to_view_aggregation": {
            "primary": "median over the view's frames",
            "secondary": "IQR over the view's frames, for the features marked "
                         "with_iqr",
            "rule": "target-independent; defined before any correlation was seen",
        },
        "border_fraction": BORDER_FRACTION,
        "groups": {
            "F_SCENE_GEOM": [{"name": n, "definition": d} for n, d in F_SCENE_GEOM],
            "F_IMAGE_GLOBAL": [{"name": n, "definition": d} for n, d in F_IMAGE_GLOBAL],
            "F_BORDER_SCENE": [{"name": n, "definition": d} for n, d in F_BORDER_SCENE],
            "F_MODEL_SELF": [{"name": n, "definition": d} for n, d in F_MODEL_SELF],
            "MODEL_DISAGREEMENT": [{"name": n, "definition": d}
                                   for n, d in MODEL_DISAGREEMENT],
        },
        "deployability": {
            "F_SCENE_GEOM": "deployable - classical OpenCV/NumPy on the RGB frame",
            "F_IMAGE_GLOBAL": "deployable - nuisance/scene-condition features, not "
                              "a geometric causal claim",
            "F_BORDER_SCENE": "deployable - generic outer-region statistics. NOT a "
                              "hand segmentation; no hand annotation, detector box "
                              "or hand prediction is used anywhere",
            "F_MODEL_SELF": "deployable - the calibration model's own outputs. "
                            "NOTE: includes the predicted focal, which is "
                            "mathematically related to the target, so this group "
                            "is always reported separately from the scene-only "
                            "groups",
            "MODEL_DISAGREEMENT": "deployable - needs a second calibration model at "
                                  "deployment time; reported as its own diagnostic",
        },
        "forbidden_inputs": [
            "provided 2D or 3D hand annotation", "hand joint geometry", "MANO",
            "predicted hand joints or depth", "hand detector output",
            "ground-truth focal, distortion, principal point or extrinsics",
            "physical camera id", "sequence id",
        ],
        "unstable_feature_policy": "a feature whose implementation is not stable "
                                   "is recorded as FEATURE_SKIPPED_UNSTABLE rather "
                                   "than forced",
        "determinism": {"seed": SEED,
                        "note": "OpenCV LSD and goodFeaturesToTrack are "
                                "deterministic for a fixed input"},
    }
    write_json(FEATURE_SPEC, feature_spec)

    probe_spec = {
        "id": "CAM_EXP_005_PROBE_SPEC_V1",
        "created_before_target_analysis": True,
        "purpose": "A LINEAR PROBE DIAGNOSTIC that asks whether a feature group "
                   "carries cross-view predictive information about the per-view "
                   "bias. It is NOT a proposed calibration method.",
        "model": "Ridge regression (scikit-learn), fitted on the training fold only",
        "target": "signed log focal bias of AnyCalib-gen radial:2 per view",
        "preprocessing": ["training-fold median imputation",
                          "training-fold standardisation",
                          "no statistic of any kind is taken from the held-out fold"],
        "alpha_grid": [0.01, 0.1, 1, 10, 100],
        "alpha_selection": "inner K-fold cross-validation INSIDE the training "
                           "fold only; the held-out fold is never consulted",
        "inner_cv": "GroupKFold by physical camera within the training fold, "
                    "k = min(5, number of training cameras)",
        "seed": SEED,
        "feature_sets": {
            "P0_baseline": "training-fold mean only (intercept)",
            "P1": ["F_SCENE_GEOM"],
            "P2": ["F_IMAGE_GLOBAL"],
            "P3": ["F_BORDER_SCENE"],
            "P4": ["F_MODEL_SELF"],
            "P5": ["MODEL_DISAGREEMENT"],
            "P6": ["F_SCENE_GEOM", "F_IMAGE_GLOBAL", "F_BORDER_SCENE"],
            "P7": ["F_SCENE_GEOM", "F_IMAGE_GLOBAL", "F_BORDER_SCENE",
                   "F_MODEL_SELF", "MODEL_DISAGREEMENT"],
        },
        "diagnostic_only_feature_sets": {
            "D_CAMERA_ID": "one-hot physical camera - ORACLE_OR_RIG_SPECIFIC_"
                           "DIAGNOSTIC, a ceiling-like reference, never deployable",
            "D_SEQUENCE_ID": "one-hot sequence - SEQUENCE_SPECIFIC_DIAGNOSTIC",
        },
        "outer_validation": {
            "A_LOSO": "leave one sequence out (5 folds). A CONTENT/SEQUENCE SHIFT "
                      "test: the same physical camera may appear in train and "
                      "test, so it is NOT a new-camera test.",
            "B_LOCO": "leave one physical camera out (40 folds). The NEW CAMERA "
                      "test, and the primary decision criterion.",
        },
        "metrics": ["MAE of the predicted log bias", "R^2", "Spearman predicted "
                    "vs actual bias",
                    "LINEAR_PROBE_CORRECTION_DIAGNOSTIC: f_corrected = f_pred / "
                    "exp(b_hat), then median relative focal error, within +/-5 %, "
                    "within +/-10 %, signed median"],
        "decision_thresholds_fixed_before_results": {
            "primary_criterion": "Leave-One-Physical-Camera-Out",
            "A_relative_median_error_reduction_pct": 10.0,
            "B_within_5pct_gain_pp": 5.0,
            "direction_must_hold_under": "physical-camera cluster bootstrap",
            "tags": {
                "DEPLOYABLE_RGB_SIGNAL_PROMISING": "A or B met on LOCO and the "
                                                   "direction holds",
                "WEAK_RGB_SIGNAL": "improvement present but below both thresholds",
                "RIG_OR_CAMERA_REPETITION_SIGNAL": "LOSO improves, LOCO does not",
                "NO_USEFUL_DEPLOYABLE_RGB_SIGNAL": "neither improves",
            },
        },
    }
    write_json(PROBE_SPEC, probe_spec)

    # ---- outer folds, frozen by membership -------------------------------
    by_seq = defaultdict(list)
    by_cam = defaultdict(list)
    for v in views:
        key = [v["sequence"], v["camera"]]
        by_seq[v["sequence"]].append(key)
        by_cam[v["physical_camera_id"]].append(key)
    splits = {
        "id": "CAM_EXP_005_OUTER_SPLITS_V1",
        "created_before_performance_inspection": True,
        "n_views": len(views),
        "respects": "experiments/manifests/development_validation_split_v1.json "
                    "(leave-one-sequence-out remains the sequence protocol); this "
                    "file adds the leave-one-physical-camera-out folds that the "
                    "new-camera question needs",
        "A_LOSO_folds": [{"held_out_sequence": s, "n_held_out_views": len(v),
                          "held_out_views": v} for s, v in sorted(by_seq.items())],
        "B_LOCO_folds": [{"held_out_physical_camera": c, "n_held_out_views": len(v),
                          "held_out_views": v} for c, v in sorted(by_cam.items())],
        "caveat_A": "the same physical camera usually appears in the training "
                    "sequences too, so LOSO measures robustness to content, not "
                    "to a new camera",
        "caveat_B": "no view of the held-out camera is in training; this is the "
                    "deployment-relevant test",
    }
    write_json(SPLIT_SPEC, splits)

    # ---- leakage audit ----------------------------------------------------
    rows = []
    for group, feats, deployable in (
            ("F_SCENE_GEOM", F_SCENE_GEOM, True),
            ("F_IMAGE_GLOBAL", F_IMAGE_GLOBAL, True),
            ("F_BORDER_SCENE", F_BORDER_SCENE, True),
            ("F_MODEL_SELF", F_MODEL_SELF, True),
            ("MODEL_DISAGREEMENT", MODEL_DISAGREEMENT, True)):
        for name, _ in feats:
            rows.append({
                "feature": name, "group": group,
                "available_at_deployment": "yes" if deployable else "no",
                "uses_GT": "no", "uses_hand_annotation": "no",
                "uses_hand_prediction": "no", "uses_camera_ID": "no",
                "uses_sequence_ID": "no",
                "allowed_in_primary_probe": "yes",
                "note": ("includes the predicted focal, which is mathematically "
                         "related to the target - reported separately from the "
                         "scene-only groups"
                         if group == "F_MODEL_SELF" else
                         "needs a second calibration model at deployment time"
                         if group == "MODEL_DISAGREEMENT" else ""),
            })
    for name, group in (("physical_camera_one_hot", "D_CAMERA_ID"),
                        ("sequence_one_hot", "D_SEQUENCE_ID")):
        rows.append({
            "feature": name, "group": group, "available_at_deployment": "no",
            "uses_GT": "no", "uses_hand_annotation": "no",
            "uses_hand_prediction": "no",
            "uses_camera_ID": "yes" if "CAMERA" in group else "no",
            "uses_sequence_ID": "yes" if "SEQUENCE" in group else "no",
            "allowed_in_primary_probe": "NO - diagnostic ceiling only",
            "note": "a new deployment camera has no id in any lookup table",
        })
    for name in ("gt_k1", "gt_k2", "gt_principal_point_offset", "gt_fx_over_fy",
                 "gt_camera_orientation_descriptor", "gt_reference_focal"):
        rows.append({
            "feature": name, "group": "ORACLE_CAMERA_PROPERTY",
            "available_at_deployment": "no", "uses_GT": "yes",
            "uses_hand_annotation": "no", "uses_hand_prediction": "no",
            "uses_camera_ID": "no", "uses_sequence_ID": "no",
            "allowed_in_primary_probe": "NO - ORACLE_DIAGNOSTIC_ONLY",
            "note": "descriptive association only; the reference focal in "
                    "particular is inside the target definition and is never a "
                    "predictor",
        })
    write_csv(TAB / "probe_leakage_audit.csv", rows)

    defs = []
    for group, feats in (("F_SCENE_GEOM", F_SCENE_GEOM),
                         ("F_IMAGE_GLOBAL", F_IMAGE_GLOBAL),
                         ("F_BORDER_SCENE", F_BORDER_SCENE),
                         ("F_MODEL_SELF", F_MODEL_SELF),
                         ("MODEL_DISAGREEMENT", MODEL_DISAGREEMENT)):
        for name, d in feats:
            defs.append({"feature": name, "group": group, "definition": d,
                         "view_aggregation": "median (and IQR where named)"})
    write_csv(TAB / "scene_feature_definitions.csv", defs)

    for p in (FEATURE_SPEC, PROBE_SPEC, SPLIT_SPEC):
        print(f"{p.name}  sha256 {sha256(p)[:16]}")
    print(f"pre-registered features: {len(defs)}  "
          f"LOSO folds: {len(splits['A_LOSO_folds'])}  "
          f"LOCO folds: {len(splits['B_LOCO_folds'])}")


if __name__ == "__main__":
    main()
