"""Freeze every CAM-EXP-008 specification BEFORE the reference focal is read."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CAMERA_MANIFEST, FRAME_COUNTS, HUBER_DELTA, LAMBDAS,  # noqa: E402
                    MANIFESTS, Q_MAX, Q_MIN, Q_N, SEED, SIGMA_FLOOR, TAB,
                    read_csv, sha256, write_csv, write_json)

SPEC = MANIFESTS / "cam_exp_008_scene_hand_spec_v1.json"
GRID = MANIFESTS / "cam_exp_008_focal_grid_v1.json"
LAM = MANIFESTS / "cam_exp_008_lambda_spec_v1.json"
SPLITS = MANIFESTS / "cam_exp_008_outer_splits_v1.json"
DIV = MANIFESTS / "cam_exp_008_pose_diversity_spec_v1.json"
CTRL = MANIFESTS / "cam_exp_008_controls_spec_v1.json"
CAND = MANIFESTS / "cam_exp_008_candidate_frames_v1.csv.gz"

CONDITIONS = [
    {"id": "S0", "name": "SCENE_ONLY_PAIRED", "lambda": 0.0,
     "hand_term": False, "role": "PAIRED BASELINE",
     "note": "its minimum is exactly f_scene by construction"},
    {"id": "S1", "name": "SCENE_PLUS_REAL_HAND_GEOMETRY",
     "lambda": "nested-CV selected", "hand_term": True,
     "role": "PRIMARY TEST"},
    {"id": "H0", "name": "HAND_ONLY_DIAGNOSTIC", "lambda": "inf",
     "hand_term": True, "role": "DIAGNOSTIC, not the headline method",
     "note": "argmin L_hand alone; comparable to the CAM-006 family"},
    {"id": "C1", "name": "SCENE_PLUS_WRONG_FRAME_HAND",
     "hand_term": True, "role": "NEGATIVE CONTROL",
     "note": "same sequence, same anatomical side, deterministically offset "
             "frame: hand size and person preserved, pose correspondence "
             "destroyed"},
    {"id": "C2", "name": "SCENE_PLUS_VIEW_SHUFFLED_HAND_PROFILE",
     "hand_term": True, "role": "NEGATIVE CONTROL",
     "note": "hand profiles permuted across views within a fold; scene "
             "estimate untouched"},
    {"id": "C3", "name": "SCENE_PLUS_FLAT_HAND", "hand_term": True,
     "role": "SANITY", "note": "L_hand == 0; must reproduce S0 exactly"},
]


def main() -> None:
    cam_ok = [(r["sequence"], r["camera"]) for r in read_csv(CAMERA_MANIFEST)
              if r["usable_for_camera_benchmark"] == "1"]
    by_cam, by_seq = defaultdict(list), defaultdict(list)
    for s, c in sorted(cam_ok):
        by_cam[c].append([s, c])
        by_seq[s].append([s, c])

    write_json(SPLITS, {
        "spec_id": "cam_exp_008_outer_splits_v1",
        "created_before_reference_focal_evaluation": True,
        "built_from": "physical camera and sequence membership only",
        "LOCO_PHYSICAL_CAMERA": {
            "role": "PRIMARY", "n_folds": len(by_cam),
            "why": "deployment means a new factory and a new camera. If the "
                   "same physical camera appears in train and test through a "
                   "different sequence, a camera-specific bias can be "
                   "memorised.",
            "folds": [{"fold": i, "held_out_physical_camera": c,
                       "test_views": v}
                      for i, (c, v) in enumerate(sorted(by_cam.items()))]},
        "LOSO_SEQUENCE": {
            "role": "SECONDARY - content shift only, NOT a new-camera test",
            "n_folds": len(by_seq),
            "folds": [{"fold": i, "held_out_sequence": s, "test_views": v}
                      for i, (s, v) in enumerate(sorted(by_seq.items()))]},
    })

    write_json(GRID, {
        "spec_id": "cam_exp_008_focal_grid_v1",
        "created_before_reference_focal_evaluation": True,
        "parameterisation": "q = f / f_scene",
        "q_min": Q_MIN, "q_max": Q_MAX, "n_points": Q_N,
        "spacing": "log-uniform",
        "why": "this experiment refines a scene anchor, it does not solve for "
               "the focal from scratch. The range is wide relative to "
               "AnyCalib's known error scale while avoiding the meaningless "
               "large-focal plateau CAM-EXP-006 mapped.",
        "secondary_sensitivity": {"q_min": 0.25, "q_max": 2.0,
                                  "role": "SECONDARY only"},
        "refinement": "parabolic interpolation in log-q on the three grid "
                      "points around an interior minimum; a boundary minimum "
                      "is reported as BOUNDARY and not refined",
    })

    write_json(LAM, {
        "spec_id": "cam_exp_008_lambda_spec_v1",
        "created_before_reference_focal_evaluation": True,
        "grid": list(LAMBDAS),
        "selection": "grouped CV over TRAINING physical cameras only, inside "
                     "each outer fold. The held-out camera's reference focal "
                     "never influences lambda.",
        "inner_objective": "median relative focal error against the "
                           "dataset-provided reference focal of the inner "
                           "validation cameras",
        "lambda_zero_is_S0": True,
    })

    write_json(DIV, {
        "spec_id": "cam_exp_008_pose_diversity_spec_v1",
        "created_before_reference_focal_evaluation": True,
        "descriptor": "root-relative reference 3D hand, scale-normalised by "
                      "its own bone-length sum, then global rotation removed "
                      "by Procrustes alignment to the view's medoid pose. "
                      "What remains is articulation only.",
        "distance": "Euclidean distance between aligned, scale-normalised "
                    "joint configurations",
        "DIVERSE_N": "greedy farthest-point selection, seeded at the medoid",
        "LOW_DIVERSITY_N": "the N poses nearest the medoid",
        "N_values": [16, 32],
        "target_independence": "no reference focal, no focal error and no "
                               "fusion outcome takes any part in selection",
        "interpretation_rule": "diversity is judged by the PAIRED incremental "
                               "gain G = err(scene-only) - err(scene+hand) "
                               "within each subset, never by comparing "
                               "scene+hand errors across subsets, because the "
                               "subsets contain different frames",
    })

    write_json(CTRL, {"spec_id": "cam_exp_008_controls_spec_v1",
                      "created_before_reference_focal_evaluation": True,
                      "conditions": CONDITIONS,
                      "decisive_rule": "if S1 beats S0 but C1 or C2 gains "
                                       "similarly, the gain is not attributed "
                                       "to hand geometry"})

    write_json(SPEC, {
        "spec_id": "cam_exp_008_scene_hand_spec_v1",
        "created_before_reference_focal_evaluation": True,
        "seed": SEED,
        "question": "does adding sequence-level hand geometry evidence to a "
                    "scene calibration anchor estimate the sequence-shared "
                    "focal better than the scene evidence alone, on exactly "
                    "the same input?",
        "deployment_setting": {
            "input": "one already-recorded video file",
            "realtime": False,
            "whole_video_available": True,
            "one_video": "one continuous recording, one physical camera, no "
                         "cut, no resolution change, no crop change, no "
                         "digital zoom",
            "intrinsics": "shared across the whole sequence",
            "mount": "tripod / fixed; small vibration allowed, no front->side "
                     "viewpoint change",
            "varies_between_factories": ["hand size", "worker", "worktable",
                                         "background", "camera", "distance",
                                         "scene layout"],
        },
        "unit": "SEQUENCE_CAMERA_UNIT = one (sequence, camera) pair = one "
                "offline calibration unit",
        "shared_vs_varying": {
            "shared_across_sequence": "K, and specifically the scalar focal f",
            "free_per_frame": "hand rotation and translation R_t, T_t",
            "note": "small tripod vibration is NOT modelled as an intrinsics "
                    "variation",
        },
        "primary_parameter": "focal only. fy/fx ratio, principal point and "
                             "radial distortion are taken from the scene "
                             "estimator at sequence level and held IDENTICAL "
                             "in S0 and S1.",
        "scene_model": "AnyCalib anycalib_gen / radial:2 (deterministic in "
                       "the CAM-EXP-004.1 reproducibility test)",
        "scene_focal": "f_scene = median over the common frames of per-frame "
                       "pred_fx",
        "scene_nuisance": {
            "r_fy": "median(pred_fy / pred_fx)",
            "cx": "median(pred_cx)", "cy": "median(pred_cy)",
            "k1": "median(k1)", "k2": "median(k2)",
            "used_in": "BOTH S0 and S1, identically",
        },
        "why_distortion_is_modelled": "CAM-EXP-006.1 showed that ignoring "
                                      "this rig's distortion (median k1 "
                                      "-0.392) substantially inflated the "
                                      "focal error. The primary hand "
                                      "objective therefore uses the "
                                      "scene-estimated distortion, never GT.",
        "scene_cost": {
            "form": "Huber(log(f / f_scene) / sigma_scene)",
            "delta": HUBER_DELTA,
            "sigma_scene": "1.4826 * MAD(log per-frame pred_fx), floored",
            "sigma_floor": SIGMA_FLOOR,
            "property": "at lambda = 0 the minimum is exactly f_scene",
        },
        "hand_cost": {
            "per_hand": "median over joints of the reprojection error in px",
            "per_frame": "median over the hands in that frame, so a two-hand "
                         "frame does not get double weight (the CAM-EXP-006 "
                         "aggregation mismatch is not repeated)",
            "per_sequence": "median over frames",
            "normalisation": "L_hand(f) = C_hand(f) - min_f C_hand(f), in "
                             "pixels. NOT per-view min-max normalised, which "
                             "would amplify a flat profile into false "
                             "confidence.",
        },
        "fused": "L_total(f) = L_scene(f) + lambda * L_hand(f)",
        "hand_geometry_source": {
            "name": "SEQUENCE_CONSISTENT_REFERENCE_HAND",
            "base": "OTHER_CAMERA_ONLY_REFERENCE_3D (target camera excluded "
                    "before reconstruction)",
            "bone_lengths": "per sequence and anatomical side, the median "
                            "bone length over the sequence; one worker's bones "
                            "do not change between frames",
            "scale": "REMOVED. The template is normalised by its own "
                     "bone-length sum, so no absolute hand-size prior is "
                     "used - factories differ in hand size.",
            "articulation": "per-frame bone DIRECTIONS from the reference "
                            "reconstruction, re-assembled along the kinematic "
                            "tree with the shared normalised bone lengths",
            "status": "INTERNAL ORACLE / INFORMATION CEILING, not deployable",
        },
        "why_oracle_hand_first": "CAM-EXP-007 found no usable residual signal "
                                 "in predicted-hand outputs. This run asks the "
                                 "prior question: if the hand geometry were "
                                 "good, would fusing it help at all? If even "
                                 "the ceiling does not help, there is little "
                                 "reason to implement the same formulation "
                                 "with a noisier predicted hand.",
        "hand_only_is_not_the_method": "H0 may fail while S1 succeeds. Scene "
                                       "narrows the candidate region; the "
                                       "hand only has to express a preference "
                                       "inside it.",
        "frame_counts": {"nested": list(FRAME_COUNTS),
                         "plus": "ALL_COMMON",
                         "nesting": "N8 subset of N16 subset of N32 subset of "
                                    "N64 subset of ALL_COMMON",
                         "wording": "offline evidence accumulation, NOT "
                                    "realtime update"},
        "conditions": CONDITIONS,
        "statistical_unit": "SEQUENCE_CAMERA_UNIT",
        "bootstrap": "physical-camera cluster bootstrap, 10000 iterations",
        "evaluation_target": "dataset-provided reference focal",
        "success_criteria": {
            "SCENE_HAND_FUSION_PROMISING": [
                "S1 vs S0 on the paired common set: median relative focal "
                "error down >= 10 % relative, OR within +-5 % up >= 5 pp",
                "the paired improvement direction holds under the "
                "physical-camera cluster bootstrap; strong form requires the "
                "CI to exclude zero",
                "real hand fusion clearly beats the C1 wrong-frame and C2 "
                "shuffled controls",
                "the output is not a constant focal collapse",
            ],
            "all_required": True,
            "SCENE_HAND_FUSION_STRONG": "additionally: controls degrade, the "
                                        "gain is visible on unseen physical "
                                        "cameras, N or pose diversity give a "
                                        "coherent trend, the S2 stress subset "
                                        "keeps the direction, and the "
                                        "secondary E2 anchor agrees",
        },
        "decision_tags": {
            "STRONG": "SCENE_HAND_FUSION_STRONG",
            "PROMISING": "SCENE_HAND_FUSION_PROMISING",
            "LOCAL": "HAND_SIGNAL_LOCAL_BUT_NOT_GENERALIZABLE",
            "DIVERSITY": "POSE_DIVERSITY_SIGNAL_ONLY",
            "ORACLE": "ORACLE_NUISANCE_REQUIRED",
            "NONE": "NO_INCREMENTAL_HAND_GEOMETRY_SIGNAL",
        },
        "forbidden": [
            "GT focal as an optimisation input",
            "GT distortion or GT principal point in the primary condition",
            "target-camera 2D inside the reference 3D",
            "target-camera extrinsics",
            "absolute hand-size prior",
            "frame selection by focal error, reference focal or fusion "
            "outcome",
            "hand observation filtering by whether it improves the focal",
            "opening the external confirmatory holdout",
        ],
        "honesty_rules": [
            "a negative result is reported unchanged",
            "a positive result is scoped to the evaluated GigaHands rig and "
            "this formulation, and is never called a deployable method",
            "if S0 and S1 differ in even one frame, the primary comparison is "
            "declared invalid",
        ],
    })

    write_csv(TAB / "statistical_units.csv", [
        {"level": "joint", "role": "repeated observation within a hand"},
        {"level": "hand", "role": "repeated observation within a frame"},
        {"level": "frame", "role": "repeated observation within a sequence"},
        {"level": "SEQUENCE_CAMERA_UNIT", "role": "STATISTICAL UNIT"},
        {"level": "physical camera",
         "role": "GENERALISATION GROUP and bootstrap cluster"},
        {"level": "sequence", "role": "secondary shift axis (LOSO)"},
    ])
    write_csv(TAB / "scene_parameter_definitions.csv", [
        {"parameter": "f_scene", "definition": "median per-frame pred_fx",
         "used_in": "S0 and S1", "source": "AnyCalib anycalib_gen/radial:2"},
        {"parameter": "r_fy", "definition": "median(pred_fy / pred_fx)",
         "used_in": "S0 and S1", "source": "same"},
        {"parameter": "cx", "definition": "median(pred_cx)",
         "used_in": "S0 and S1", "source": "same"},
        {"parameter": "cy", "definition": "median(pred_cy)",
         "used_in": "S0 and S1", "source": "same"},
        {"parameter": "k1", "definition": "median(k1)",
         "used_in": "S0 and S1", "source": "same"},
        {"parameter": "k2", "definition": "median(k2)",
         "used_in": "S0 and S1", "source": "same"},
        {"parameter": "sigma_scene",
         "definition": f"max(1.4826*MAD(log pred_fx), {SIGMA_FLOOR})",
         "used_in": "S0 and S1", "source": "same"},
    ])
    write_csv(TAB / "control_definitions.csv", CONDITIONS)

    for p in (SPEC, GRID, LAM, SPLITS, DIV, CTRL, CAND):
        if p.exists():
            print(f"{p.name}  sha256={sha256(p)[:16]}")
    print(f"\nLOCO folds {len(by_cam)}  LOSO folds {len(by_seq)}")


if __name__ == "__main__":
    main()
