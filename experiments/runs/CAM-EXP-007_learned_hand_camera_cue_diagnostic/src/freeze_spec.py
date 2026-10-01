"""Freeze every CAM-EXP-007 specification BEFORE the target is read.

Writes the feature spec, latent spec, probe spec, control spec, frame manifest
and outer splits. None of these depends on a focal error, an AnyCalib
prediction or a reference focal.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ALPHAS, CAMERA_MANIFEST, FRAMES16, MANIFESTS,  # noqa: E402
                    MIN_FRAMES_WITH_HAND, N_FRAMES, PCA_DIM, SEED, TAB,
                    read_csv, sha256, write_csv, write_json)

FEATURE_SPEC = MANIFESTS / "cam_exp_007_feature_spec_v1.json"
LATENT_SPEC = MANIFESTS / "cam_exp_007_latent_spec_v1.json"
PROBE_SPEC = MANIFESTS / "cam_exp_007_probe_spec_v1.json"
CONTROL_SPEC = MANIFESTS / "cam_exp_007_control_spec_v1.json"
SPLITS = MANIFESTS / "cam_exp_007_outer_splits_v1.json"
FRAMES = MANIFESTS / "cam_exp_007_frame_manifest_v1.csv.gz"

FEATURE_GROUPS = {
    "F0_HAND_VISIBILITY_AND_BOX": {
        "what": "detector and crop geometry on the image plane",
        "not_a_hand_representation": True,
        "per_frame": [
            "n_hands", "bbox_area_norm", "bbox_w_over_W", "bbox_h_over_H",
            "bbox_aspect", "bbox_cx_norm", "bbox_cy_norm", "box_size_norm",
            "det_score", "crop_kp2d_spread_x", "crop_kp2d_spread_y",
        ],
        "view_aggregate": ["median", "iqr"],
        "extra_view_level": ["availability_rate", "n_frames_with_hand",
                             "mean_hands_per_frame", "side_consistency"],
    },
    "F1_CAMERA_HEAD_OUTPUT": {
        "what": "the model's own weak-perspective camera head, in CROP space",
        "per_frame": ["pred_cam_s", "pred_cam_tx", "pred_cam_ty"],
        "view_aggregate": ["median", "iqr"],
        "never": "pred_cam is never converted to a metric translation, "
                 "because that conversion multiplies in the pipeline focal",
    },
    "F2_EXPLICIT_HAND_GEOMETRY": {
        "what": "root-relative predicted hand shape and pose, focal-free",
        "per_frame": [
            "hand_diameter", "depth_extent", "depth_over_diameter",
            "lateral_extent", "palm_normal_z", "finger_spread",
            "bone_ratio_mean", "bone_ratio_std",
            "mano_beta_0..9 (10)", "global_orient_aa_norm",
            "hand_pose_aa_norm", "hand_pose_aa_mean_abs",
        ],
        "view_aggregate": ["median", "iqr"],
        "side_handling": "LEFT hands are already mirrored into the canonical "
                         "right-hand frame by the model wrapper (x-flip). "
                         "Only side-invariant scalars are used, and no extra "
                         "mirror transform is applied.",
    },
    "F3_TEMPORAL_HAND_STABILITY": {
        "what": "dispersion of the above across the 16 frames of a STATIC view",
        "view_level": [
            "pred_cam_s_std", "pred_cam_s_iqr", "bbox_area_std",
            "box_size_std", "beta_std_mean", "pose_aa_std",
            "hand_diameter_std", "depth_over_diameter_std",
            "det_score_std", "availability_rate", "side_consistency",
            "n_hands_std",
        ],
        "note": "a static camera means these disperse only through the model, "
                "not through camera motion",
    },
    "F4_LEARNED_HAND_LATENT": {
        "what": "the frozen image-encoder representation",
        "see": "cam_exp_007_latent_spec_v1.json",
    },
}

PROBES = [
    ("P0", "BASELINE_TRAINING_MEAN", [], "baseline"),
    ("P1", "F0_HAND_VISIBILITY_AND_BOX", ["F0"], "deployable"),
    ("P2", "F1_CAMERA_HEAD_OUTPUT", ["F1"], "deployable"),
    ("P3", "F2_EXPLICIT_HAND_GEOMETRY", ["F2"], "deployable"),
    ("P4", "F3_TEMPORAL_HAND_STABILITY", ["F3"], "deployable"),
    ("P5", "ALL_EXPLICIT_HAND_OUTPUTS", ["F0", "F1", "F2", "F3"],
     "deployable"),
    ("P6", "LEARNED_HAND_LATENT", ["F4"], "deployable"),
    ("P7", "ALL_EXPLICIT_PLUS_LATENT", ["F0", "F1", "F2", "F3", "F4"],
     "deployable"),
    ("P8", "SCENE_PLUS_ALL_EXPLICIT", ["SCENE", "F0", "F1", "F2", "F3"],
     "deployable"),
    ("P9", "SCENE_PLUS_LATENT", ["SCENE", "F4"], "deployable"),
    ("P10", "SCENE_PLUS_ALL_HAND", ["SCENE", "F0", "F1", "F2", "F3", "F4"],
     "deployable"),
]

BASELINES = [
    {"id": "B0", "name": "RAW_ANYCALIB", "deployable": 1,
     "what": "AnyCalib's view focal with no correction"},
    {"id": "B1", "name": "GLOBAL_BIAS_CORRECTED_ANYCALIB", "deployable": 1,
     "what": "subtract the training folds' median signed log bias; one "
             "constant, no per-view information",
     "role": "PRIMARY_PRACTICAL_BASELINE"},
    {"id": "B2", "name": "CAM005_FROZEN_SCENE_BASELINE", "deployable": 1,
     "what": "CAM-EXP-005's deployable scene-only feature set P6 "
             "(scene geometry + image global + border), reused unchanged",
     "note": "CAM-005's P4 and P7 are NOT used: they contain the model's own "
             "prediction, which is exactly the confound C1 isolates"},
    {"id": "B3", "name": "TRAINING_FOLD_CONSTANT_REFERENCE_ORACLE",
     "deployable": 0, "what": "emit the training cameras' median reference "
                              "focal", "role": "ORACLE_SANITY_CONTROL",
     "uses_training_target_labels": 1},
]

CONTROLS = [
    {"id": "C1", "name": "ANYCALIB_FOCAL_PRIOR_CONTROL",
     "features": ["log_f_anycalib"],
     "role": "SINGLE_FOCAL_PRIOR_CONFOUND_CONTROL",
     "is_hand_signal": 0,
     "why": "the target is log(f_any / f_ref). Handing the probe log(f_any) "
            "lets it exploit the rig's narrow reference-focal prior and look "
            "excellent without any camera cue. Never counted as hand signal, "
            "and never mixed into a primary hand feature set."},
    {"id": "C2", "name": "HAND_FEATURE_VIEW_SHUFFLE",
     "role": "PRIMARY_NEGATIVE_CONTROL", "is_hand_signal": 0,
     "rule": "within each outer fold, permute the feature ROWS among training "
             "views and, separately, among held-out views. Targets are never "
             "shuffled. Permutation is within-sequence where a sequence has "
             "at least 4 views in that partition, otherwise within the whole "
             "partition.",
     "n_seeds": 10},
    {"id": "C3", "name": "RANDOM_FEATURE_CONTROL",
     "role": "SECONDARY_CONTROL", "is_hand_signal": 0,
     "rule": "deterministic Gaussian features of matched dimension, seeded "
             "per view id"},
    {"id": "C4", "name": "TRAIN_TARGET_SHUFFLE",
     "role": "SECONDARY_CONTROL", "is_hand_signal": 0,
     "rule": "shuffle training targets only; test targets stay true"},
]

HYPOTHESES = {
    "H1": "some frozen hand-model explicit output is associated with the "
          "AnyCalib residual bias",
    "H2": "if that signal is a deployable camera cue it must beat B1 under "
          "leave-one-physical-camera-out",
    "H3": "simple box/crop/pred_cam cues and the deep latent may carry "
          "different amounts of information",
    "H4": "if the latent beats the explicit outputs, the network holds a "
          "camera/image prior not exposed in its explicit geometry",
    "H5": "if real hand-view correspondence matters, real features must beat "
          "view-shuffled features",
    "H6": "improvement under LOSO but failure under LOCO-camera indicates "
          "same-rig repetition rather than a generalisable cue",
    "H7": "scene+hand beating scene-only means hand features add information "
          "beyond CAM-005's scene cues",
    "H8": "if nothing beats B1 and the scene baseline, no usable residual-bias "
          "signal was detected in this frozen model",
}

SUCCESS = {
    "primary_protocol": "LOCO_PHYSICAL_CAMERA",
    "HAND_DERIVED_SIGNAL_PROMISING": {
        "condition_1": "versus B1: median relative focal error reduced by "
                       ">= 10 % relative, OR within +-5 % rate up by >= 5 "
                       "percentage points",
        "condition_2": "bias-prediction Spearman positive with a stable "
                       "physical-camera cluster-bootstrap direction; strong "
                       "form requires the CI lower bound > 0",
        "condition_3": "clearly better than HAND_FEATURE_VIEW_SHUFFLE",
        "condition_4": "the corrected focal is not a constant collapse",
        "all_four_required": True,
    },
    "SCENE_PLUS_HAND_SIGNAL_PROMISING": {
        "condition_1": "versus B2 scene-only: >= 10 % relative median error "
                       "reduction OR +5 pp within +-5 %",
        "condition_2": "bias correlation improves in the same direction",
    },
    "LATENT_CAMERA_SIGNAL_PROMISING": {
        "condition": "a latent-containing group (P6, P7, P9, P10) passes the "
                     "primary thresholds, beats shuffle, holds under "
                     "LOCO-camera, and is not a constant collapse",
    },
    "CONSTANT_COLLAPSE": {
        "fires_if": "median error low AND corrected-focal CV near zero AND no "
                    "focal tracking",
        "consequence": "not counted as hand signal",
    },
}

TAGS = {
    "A": "EXPLICIT_HAND_CAMERA_CUE_PROMISING",
    "B": "LATENT_CAMERA_SIGNAL_PROMISING",
    "C": "LATENT_ONLY_CAMERA_SIGNAL",
    "D": "HAND_SIGNAL_INCREMENTAL_TO_SCENE",
    "E": "RIG_REPETITION_SIGNAL_ONLY",
    "F": "SINGLE_FOCAL_PRIOR_CONFOUND",
    "G": "SHUFFLE_CONTROL_NOT_DEGRADED",
    "H": "NO_GENERALIZABLE_HAND_DERIVED_SIGNAL",
    "I": "HAND_FEATURE_COVERAGE_LIMITED",
    "J": "LATENT_EXTRACTION_UNAVAILABLE",
}


def build_frames():
    """Reuse CAM-EXP-006's frozen 16-frame nested grid, unchanged."""
    cam_ok = {(r["sequence"], r["camera"]) for r in read_csv(CAMERA_MANIFEST)
              if r["usable_for_camera_benchmark"] == "1"}
    rows = []
    for r in read_csv(FRAMES16):
        key = (r["sequence"], r["camera"])
        if key in cam_ok and int(r["grid_pos"]) < N_FRAMES:
            rows.append({"sequence": r["sequence"], "camera": r["camera"],
                         "frame": int(r["frame"]),
                         "grid_pos": int(r["grid_pos"])})
    write_csv(FRAMES, rows)
    return rows


def build_splits(rows):
    """Outer splits from camera / sequence membership only."""
    views = sorted({(r["sequence"], r["camera"]) for r in rows})
    # physical camera identity is the camera name itself in this rig
    by_cam = defaultdict(list)
    by_seq = defaultdict(list)
    for s, c in views:
        by_cam[c].append([s, c])
        by_seq[s].append([s, c])
    loco = [{"fold": i, "held_out_physical_camera": c,
             "test_views": v, "n_test_views": len(v)}
            for i, (c, v) in enumerate(sorted(by_cam.items()))]
    loso = [{"fold": i, "held_out_sequence": s,
             "test_views": v, "n_test_views": len(v)}
            for i, (s, v) in enumerate(sorted(by_seq.items()))]
    write_json(SPLITS, {
        "spec_id": "cam_exp_007_outer_splits_v1",
        "created_before_target_analysis": True,
        "built_from": "physical camera and sequence membership only; no "
                      "target value was read",
        "n_views": len(views),
        "LOCO_PHYSICAL_CAMERA": {"role": "PRIMARY", "n_folds": len(loco),
                                 "folds": loco},
        "LOSO_SEQUENCE": {"role": "SECONDARY - content/sequence shift, NOT a "
                                  "new-camera test", "n_folds": len(loso),
                          "folds": loso},
    })
    return views, loco, loso


def main() -> None:
    rows = build_frames()
    views, loco, loso = build_splits(rows)

    write_json(FEATURE_SPEC, {
        "spec_id": "cam_exp_007_feature_spec_v1",
        "created_before_target_analysis": True,
        "seed": SEED, "n_frames_per_view": N_FRAMES,
        "eligibility": {
            "rule": f"a view is HAND_FEATURE_ELIGIBLE if at least "
                    f"{MIN_FRAMES_WITH_HAND} of its {N_FRAMES} frames yield "
                    f"at least one usable hand prediction",
            "otherwise": "HAND_FEATURE_UNAVAILABLE, excluded from the "
                         "hand-probe common set but reported in coverage",
            "frames_are_never_replaced": "a frame with no detection stays in "
                                         "the manifest and becomes an "
                                         "availability feature",
        },
        "frame_to_view_aggregation": {
            "hands_to_frame": "scalar features: median over the hands in that "
                              "frame. vector/latent features: mean over "
                              "hands. Either way a frame contributes exactly "
                              "once, whether it holds one hand or two.",
            "frames_to_view": "median and IQR across frames; the temporal "
                              "group additionally uses std and rates",
        },
        "missingness": "training-fold median imputation; missingness rate may "
                       "itself be a feature",
        "groups": FEATURE_GROUPS,
        "forbidden_in_primary": [
            "absolute camera translation cam_t", "pipeline focal length",
            "any position computed with a focal (5000 px, AnyCalib or "
            "reference)", "dataset-provided 2D or 3D hand annotations",
            "provided MANO", "GT focal / distortion / extrinsics",
            "camera ID", "sequence ID", "AnyCalib's prediction",
        ],
    })

    write_json(LATENT_SPEC, {
        "spec_id": "cam_exp_007_latent_spec_v1",
        "created_before_target_analysis": True,
        "model": "AnyHand-WiLoR",
        "layer_path": "model.backbone",
        "which_output": "index 3 of the backbone's returned tuple (vit_out)",
        "chosen_because": "it is the image encoder's final feature map, "
                          "immediately before the MANO and camera heads "
                          "consume it. Chosen from architecture semantics "
                          "with no target read.",
        "only_one_layer": "exactly one layer is frozen. Probing several and "
                          "reporting the best would be selection on the "
                          "result.",
        "tensor_shape": "(B, 1280, 16, 12)",
        "pooling": "global average pool over the spatial axes -> (B, 1280)",
        "extraction": "forward hook; model source unmodified",
        "verification": "a bit-identity check confirms the hook does not "
                        "change the model's own outputs",
        "dimension_reduction": {
            "method": "PCA by SVD, fit on TRAINING-FOLD views only",
            "primary_dim": PCA_DIM,
            "actual_dim": f"min({PCA_DIM}, n_train_views - 2, input_dim)",
            "not_tuned": "the dimension is never changed after seeing test "
                         "performance; 16 and 64 are secondary sensitivity "
                         "only",
        },
    })

    write_json(PROBE_SPEC, {
        "spec_id": "cam_exp_007_probe_spec_v1",
        "created_before_target_analysis": True,
        "target": {
            "source": "CAM-EXP-005 results/raw/view_targets.csv.gz, column "
                      "anycalib_signed_log_bias",
            "definition": "e_v = median_t log( f_AnyCalib(v,t) / f_ref(v) ) "
                          "for the AnyCalib anycalib_gen radial:2 condition, "
                          "reused EXACTLY as frozen in CAM-EXP-005",
            "why_not_focal_directly": "the rig's reference focal has CV "
                                      "1.90 %, so predicting the focal itself "
                                      "rewards emitting a constant "
                                      "(SINGLE_FOCAL_RIG_CONFOUND)",
        },
        "correction_diagnostic": {
            "formula": "f_corrected = f_anycalib / exp(bias_hat)",
            "label": "LINEAR_PROBE_CORRECTION_DIAGNOSTIC",
            "not_a_method": True,
        },
        "model": {"type": "ridge regression, closed form",
                  "alphas": list(ALPHAS),
                  "alpha_selection": "grouped CV inside the training fold "
                                     "only; the outer test is never consulted",
                  "no_new_network": True},
        "preprocessing_order": [
            "training-fold median imputation",
            "training-fold standardisation",
            "training-fold PCA for latent groups",
            "inner grouped CV alpha selection",
            "full training fit",
            "predict the held-out physical camera",
        ],
        "probes": [{"id": i, "name": n, "groups": g, "status": s}
                   for i, n, g, s in PROBES],
        "baselines": BASELINES,
        "statistical_unit": "(sequence, camera) static view",
        "generalisation_group": "physical camera",
        "bootstrap": "physical-camera cluster bootstrap, 10000 iterations",
        "comparison_rule": "every paired comparison runs on the COMMON "
                           "hand-feature-eligible view set; results are also "
                           "reported on the full benchmark",
        "hypotheses": HYPOTHESES,
        "success_criteria": SUCCESS,
        "decision_tags": TAGS,
    })

    write_json(CONTROL_SPEC, {
        "spec_id": "cam_exp_007_control_spec_v1",
        "created_before_target_analysis": True,
        "controls": CONTROLS,
    })

    write_csv(TAB / "probe_feature_sets.csv",
              [{"probe": i, "name": n, "groups": "+".join(g) or "(none)",
                "status": s} for i, n, g, s in PROBES])
    write_csv(TAB / "baseline_definitions.csv", BASELINES)
    write_csv(TAB / "control_definitions.csv", CONTROLS)
    write_csv(TAB / "statistical_units.csv", [
        {"level": "hand", "role": "repeated observation within a frame"},
        {"level": "frame", "role": "repeated observation within a view"},
        {"level": "view (sequence, camera)", "role": "STATISTICAL UNIT"},
        {"level": "physical camera", "role": "GENERALISATION GROUP and "
                                             "bootstrap cluster"},
        {"level": "sequence", "role": "secondary shift axis (LOSO)"},
    ])

    for p in (FEATURE_SPEC, LATENT_SPEC, PROBE_SPEC, CONTROL_SPEC, SPLITS,
              FRAMES):
        print(f"{p.name}  sha256={sha256(p)[:16]}")
    print(f"\nviews {len(views)}  LOCO folds {len(loco)}  "
          f"LOSO folds {len(loso)}  frames {len(rows)}")


if __name__ == "__main__":
    main()
