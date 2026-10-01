"""Freeze every CAM-EXP-006 specification BEFORE any solver output is seen.

Four artefacts are written, each hashed into the run's provenance:

  cam_exp_006_reference_hand_spec_v1.json    reference-3D construction
  cam_exp_006_reference_hand_frames_v1.csv.gz nested N = 1,2,4,8,16 frame grid
  cam_exp_006_controls_spec_v1.json          negative controls
  cam_exp_006_evaluation_spec_v1.json        metrics, thresholds, decision tags

Nothing in this file depends on a CAM-006 result. Run it once, commit the
frozen files, and only then run the solver.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (LOCO_MIN_INLIER_CAMERAS, LOCO_THRESHOLD_PX, MANIFESTS,  # noqa: E402
                    SEED, sha256, write_json)

SPEC = MANIFESTS / "cam_exp_006_reference_hand_spec_v1.json"
CONTROLS = MANIFESTS / "cam_exp_006_controls_spec_v1.json"
EVAL = MANIFESTS / "cam_exp_006_evaluation_spec_v1.json"
FRAMES = MANIFESTS / "cam_exp_006_reference_hand_frames_v1.csv.gz"


REFERENCE_SPEC = {
    "spec_id": "cam_exp_006_reference_hand_spec_v1",
    "seed": SEED,
    "reference_3d": {
        "name": "OTHER_CAMERA_ONLY_REFERENCE_3D",
        "what_it_is": "a 3D hand triangulated from the dataset-provided 2D "
                      "observations of every OTHER calibrated camera in the "
                      "same sequence, with the camera under test excluded "
                      "before reconstruction",
        "what_it_is_not": [
            "not ground truth 3D",
            "not independent physical ground truth",
            "not a dataset-provided 3D annotation",
        ],
        "implementation": "experiments/runs/CAM-EXP-001_3_gigahands_multiview_"
                          "triangulation/src/loco.py :: reconstruct(), reused "
                          "unchanged",
        "threshold_px": LOCO_THRESHOLD_PX,
        "min_inlier_cameras": LOCO_MIN_INLIER_CAMERAS,
        "min_ok_joints_required": 12,
    },
    "non_circularity_contract": [
        "the camera under test is removed from the observation set BEFORE "
        "reconstruction, so its own 2D cannot shape the reference hand it is "
        "later tested against",
        "GT focal length, GT extrinsics and GT distortion are NEVER solver "
        "inputs; the solver sees only the reference 3D, the provided 2D of the "
        "camera under test, and the image size",
        "the principal point is FIXED at (W/2, H/2), it is not read from the "
        "provided calibration",
        "no calibration model output (AnyCalib, GeoCalib, E2) enters the "
        "solver; those appear only in the evaluation table",
        "the bilateral apparent-size ratio between the two hands is NEVER "
        "used as a focal estimator",
    ],
    "scope": {
        "type": "INTERNAL_REFERENCE_HAND_DIAGNOSTIC",
        "question": "how much focal information is recoverable, at the very "
                    "best, from reference hand geometry plus one view's 2D",
        "explicitly_not": "a deployable calibration method; the reference 3D "
                          "is unavailable at deployment time by construction",
    },
    "forbidden_in_this_run": [
        "training any hand or calibration network",
        "using predicted-hand cues from WiLoR or AnyHand (that is CAM-007)",
        "modifying E2",
        "building a scene-aware corrector",
        "opening the FINAL_CONFIRMATORY_HOLDOUT",
    ],
}


SOLVER_SPEC = {
    "name": "PROFILED_PNP_FOCAL_DIAGNOSTIC",
    "unknowns": "one scalar focal f, shared by fx and fy, per (view, frame set)",
    "nuisance": "per-hand rotation R and translation T, re-fitted at every "
                "candidate f by cv2.solvePnP (SQPNP, no distortion model)",
    "intrinsics": "K(f) = [[f,0,W/2],[0,f,H/2],[0,0,1]]; principal point FIXED "
                  "at the image centre, never read from provided calibration",
    "grid": {"gamma_min": 0.25, "gamma_max": 5.0, "n_points": 121,
             "spacing": "log-uniform",
             "gamma_definition": "gamma = f / max(W, H)"},
    "objective": "median over joints of the reprojection error in px, then "
                 "the mean over the frames in the set, in log space",
    "selection": "argmin over the grid, refined by a parabolic fit on the "
                 "three grid points around the minimum, in log-gamma",
    "identifiability_flags": [
        "FLAT_PROFILE if the objective at the grid ends is within 10 % of the "
        "minimum, i.e. the data does not constrain f",
        "BOUNDARY_SOLUTION if the refined optimum lands on the first or last "
        "grid point",
    ],
}


CONTROLS_SPEC = {
    "spec_id": "cam_exp_006_controls_spec_v1",
    "seed": SEED,
    "purpose": "a focal that can be read off a hand must come from the hand's "
               "3D SHAPE. These controls destroy that shape in three different "
               "ways while leaving everything else identical. If the solver "
               "still recovers the focal under a control, the signal is not "
               "geometric and the positive result is an artefact.",
    "controls": [
        {"id": "JOINT_PERMUTATION",
         "what": "randomly permute the 21 reference-3D joints before the "
                 "solve, keeping the 2D order fixed",
         "destroys": "the joint correspondence, hence all shape information",
         "expected_if_signal_is_real": "the solver should fail or return a "
                                       "flat, uninformative profile"},
        {"id": "WRONG_POSE",
         "what": "replace the reference 3D by the reference 3D of the SAME "
                 "hand in a DIFFERENT, temporally distant frame of the same "
                 "view",
         "destroys": "the match between 3D pose and 2D observation, while "
                     "keeping a real hand's scale and bone lengths",
         "expected_if_signal_is_real": "large focal error; this is the "
                                       "sharpest control because hand SIZE is "
                                       "preserved"},
        {"id": "PLANARIZED",
         "what": "orthogonally project the reference 3D onto its own best-fit "
                 "plane before the solve",
         "destroys": "the depth extent that makes perspective, hence focal, "
                     "observable at all",
         "expected_if_signal_is_real": "the profile should become flat or the "
                                       "optimum should run to the grid "
                                       "boundary, because a planar target "
                                       "cannot separate focal from depth"},
    ],
    "reporting_rule": "every control is reported with the same metrics as the "
                      "main condition, in the same table, whatever the outcome",
}


EVAL_SPEC = {
    "spec_id": "cam_exp_006_evaluation_spec_v1",
    "seed": SEED,
    "unit_of_aggregation": "(sequence, camera) view; frames are NEVER pooled "
                           "across sequences",
    "primary_metric": "relative focal error in percent, "
                      "100 * |f_hat - f_ref| / f_ref, where f_ref is "
                      "GT_EFFECTIVE_FOCAL = the dataset-provided native fx",
    "secondary_metrics": ["signed log error log(f_hat / f_ref)",
                          "per-view bias and within-view spread, decomposed as "
                          "in CAM-EXP-004",
                          "share of views flagged FLAT_PROFILE or "
                          "BOUNDARY_SOLUTION"],
    "frame_counts": [1, 2, 4, 8, 16],
    "uncertainty": "cluster bootstrap over the 40 physical cameras, 10000 "
                   "iterations, 95 % percentile interval",
    "comparators": [
        {"id": "ANYCALIB_N8", "role": "PRIMARY_SINGLE_IMAGE_BASELINE"},
        {"id": "GEOCALIB_N8", "role": "SECONDARY_SINGLE_IMAGE_BASELINE"},
        {"id": "E2_N8", "role": "FROZEN_EXPLORATORY_ENSEMBLE, not a proposed "
                                "method"},
        {"id": "CONSTANT_RIG_FOCAL_ORACLE",
         "role": "PRE_REGISTERED_POST_HOC_ORACLE_SANITY_CONTROL",
         "what": "predict, for every view, the median GT reference focal of "
                 "the whole rig",
         "why": "the GigaHands rig has a single-focal confound "
                "(SINGLE_FOCAL_RIG_CONFOUND, reference focal CV 1.90 % across "
                "175 views). Any method that does not beat this constant has "
                "demonstrated no per-view focal information at all.",
         "not_a_method": "it uses the evaluation targets and could not be "
                         "formed without them"},
    ],
    "pre_registered_thresholds": {
        "H1_reference_hand_carries_focal_information":
            "the main condition must beat CONSTANT_RIG_FOCAL_ORACLE on median "
            "relative focal error, with a cluster-bootstrap 95 % interval on "
            "the paired difference that excludes zero",
        "H2_the_information_is_geometric":
            "every control must be worse than the main condition by at least a "
            "factor of 2 in median relative focal error",
        "H3_more_frames_help":
            "N = 16 must beat N = 1 by at least 2 percentage points of median "
            "relative focal error",
    },
    "decision_tags": {
        "A": "REFERENCE_HAND_FOCAL_INFORMATION_CONFIRMED",
        "B": "REFERENCE_HAND_FOCAL_INFORMATION_ABSENT",
        "C": "SIGNAL_PRESENT_BUT_NOT_GEOMETRIC (a control also succeeds)",
        "D": "SIGNAL_EXPLAINED_BY_SINGLE_FOCAL_RIG_CONFOUND (the constant "
             "oracle is not beaten)",
        "E": "WEAKLY_IDENTIFIABLE (a large share of FLAT_PROFILE or "
             "BOUNDARY_SOLUTION flags)",
        "F": "SATURATED_BY_FEW_FRAMES (H3 fails)",
        "G": "BLOCKED (the solve could not be run as specified)",
        "H": "INCONCLUSIVE_UNDER_THIS_RIG",
    },
    "honesty_rules": [
        "a result opposite to the hypothesis is reported unchanged",
        "no frame, view, threshold or grid point is chosen by looking at the "
        "focal error",
        "the ceiling reported here is an UPPER BOUND under an oracle reference "
        "hand and must never be described as achievable performance",
    ],
}


def main() -> None:
    for path, obj in [(SPEC, REFERENCE_SPEC | {"solver": SOLVER_SPEC}),
                      (CONTROLS, CONTROLS_SPEC), (EVAL, EVAL_SPEC)]:
        write_json(path, obj)
        print(f"{path.name}  sha256={sha256(path)[:16]}")
    if not FRAMES.exists():
        print(f"NOTE: {FRAMES.name} is written by src/build_frame_manifest.py")


if __name__ == "__main__":
    main()
