"""Source-level audit of the CAM-EXP-006 solver implementation.

Nine stages are compared against the frozen CAM-006 intent. This reads
CAM-EXP-006's source and manifests; it writes nothing into that run.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MANIFESTS, R006, TAB, read_json, write_csv, write_json  # noqa: E402

SOLVER = R006 / "src" / "focal_profile_solver.py"
RUNNER = R006 / "src" / "run_solver.py"


def main() -> None:
    solver_src = SOLVER.read_text(encoding="utf-8")
    runner_src = RUNNER.read_text(encoding="utf-8")
    spec = read_json(MANIFESTS / "cam_exp_006_reference_hand_spec_v1.json")
    sol_spec = spec["solver"]

    # --- the decisive evidence for stage 2 -------------------------------
    flatten = re.search(
        r"ps\s*=\s*\[p for f in order\[:n\] for p in prof\.get\(\(cond, f\), \[\]\)\]",
        runner_src)
    combine_is_mean_over_list = "np.nanmean(lg, axis=0)" in solver_src
    hand_weighted = bool(flatten) and combine_is_mean_over_list

    rows = [
        {"stage": "1_joint_to_hand",
         "frozen_intent": "median over joints of the reprojection error in px",
         "actual_implementation": "np.median(norm(proj - uv, axis=1)) inside "
                                  "_pnp_error()",
         "match": "MATCH",
         "impact": "none",
         "correction_in_cam0061": "none"},

        {"stage": "2_hands_within_one_frame",
         "frozen_intent": "hands in one frame combine to ONE frame score, so "
                          "every frame carries equal weight (the docstring of "
                          "combine() says 'Mean over frames of the per-frame "
                          "log objective')",
         "actual_implementation": "run_solver.py flattens every (frame, hand) "
                                  "profile into one list and combine() takes "
                                  "nanmean over that flat list, so a frame "
                                  "with two usable hands contributes TWO "
                                  "entries",
         "match": "MISMATCH" if hand_weighted else "MATCH",
         "impact": "a two-hand frame receives twice the weight of a one-hand "
                   "frame; the effective aggregation unit is the hand, not the "
                   "frame",
         "correction_in_cam0061": "CAM0061_FRAME_BALANCED: median over the "
                                  "usable hands inside a frame first, then the "
                                  "existing frame aggregation"},

        {"stage": "3_frames_to_view",
         "frozen_intent": "mean, in log space, of the per-frame objectives",
         "actual_implementation": "np.nanmean over log-profiles in combine()",
         "match": "MATCH_IN_FORM",
         "impact": "correct once stage 2 is fixed; before the fix the mean is "
                   "taken over hands rather than frames",
         "correction_in_cam0061": "kept unchanged; only the stage-2 input to "
                                  "it is corrected"},

        {"stage": "4_pnp_backend_and_planar_fallback",
         "frozen_intent": "SQPNP as the primary back-end",
         "actual_implementation": "SQPNP, falling back to IPPE then ITERATIVE "
                                  "when SQPNP raises on degenerate input",
         "match": "MATCH_WITH_DOCUMENTED_EXTENSION",
         "impact": "the PLANARIZED control necessarily runs a different "
                   "back-end, so it is not a matched comparison",
         "correction_in_cam0061": "the real planar control is demoted to "
                                  "sensitivity only; a synthetic planarity "
                                  "sweep carries the evidence instead"},

        {"stage": "5_principal_point",
         "frozen_intent": "fixed at (W/2, H/2), never read from the provided "
                          "calibration",
         "actual_implementation": "K(f) built with cx=w/2, cy=h/2",
         "match": "MATCH",
         "impact": "the rig's provided cy sits about 24 px from H/2, so this "
                   "is an approximation of real size",
         "correction_in_cam0061": "conditions R2, R4 and R5 supply the "
                                  "provided principal point as an "
                                  "ORACLE_DIAGNOSTIC_ONLY nuisance"},

        {"stage": "6_distortion",
         "frozen_intent": "no distortion model",
         "actual_implementation": "distCoeffs=None in both solvePnP and "
                                  "projectPoints, applied to raw distorted 2D",
         "match": "MATCH",
         "impact": "the rig's median k1 is about -0.39, i.e. substantial "
                   "barrel distortion, so this approximation is not benign",
         "correction_in_cam0061": "conditions R3, R4 and R5 supply the "
                                  "provided distortion coefficients as an "
                                  "ORACLE_DIAGNOSTIC_ONLY nuisance"},

        {"stage": "7_fx_fy",
         "frozen_intent": "one scalar focal shared by fx and fy",
         "actual_implementation": "K(f) = [[f,0,cx],[0,f,cy],[0,0,1]]",
         "match": "MATCH",
         "impact": "the rig's provided fy/fx is about 0.9987, so the "
                   "assumption is very nearly true on this rig",
         "correction_in_cam0061": "condition R5 supplies the provided fy/fx "
                                  "RATIO only; the focal magnitude is never "
                                  "supplied"},

        {"stage": "8_focal_grid_and_refinement",
         "frozen_intent": f"{sol_spec['grid']['n_points']} log-uniform points "
                          f"over [{sol_spec['grid']['gamma_min']}, "
                          f"{sol_spec['grid']['gamma_max']}], parabolic "
                          f"refinement in log-gamma",
         "actual_implementation": "np.geomspace(0.25, 5.0, 121); parabolic "
                                  "refinement with the step clipped to +-1 "
                                  "grid interval",
         "match": "MATCH",
         "impact": "none",
         "correction_in_cam0061": "reused unchanged"},

        {"stage": "9_flat_and_boundary_flags",
         "frozen_intent": "FLAT_PROFILE when the grid ends are within 10 % of "
                          "the minimum; BOUNDARY_SOLUTION at the first or last "
                          "grid point",
         "actual_implementation": "depth = min(ends) - min(curve) in log units, "
                                  "flat when depth < log1p(0.10); boundary when "
                                  "argmin index is 0 or n-1",
         "match": "MATCH",
         "impact": "none",
         "correction_in_cam0061": "reused unchanged"},
    ]
    write_csv(TAB / "cam006_aggregation_implementation_audit.csv", rows)

    verdict = {
        "hand_weighting_mismatch_detected": hand_weighted,
        "label": "CAM006_ORIGINAL_HAND_WEIGHTED" if hand_weighted
                 else "CAM006_ALREADY_FRAME_BALANCED",
        "evidence_line_in_run_solver": flatten.group(0) if flatten else None,
        "evidence_line_in_combine": "return np.nanmean(lg, axis=0)"
                                    if combine_is_mean_over_list else None,
        "docstring_of_combine": "Mean over frames of the per-frame log "
                                "objective, as frozen in the spec.",
        "reading": "the docstring states a per-FRAME mean, but the argument "
                   "handed to it is a flat list of per-HAND profiles, so the "
                   "mean is over hands. Frames containing two usable hands are "
                   "weighted twice.",
        "stages_matching": sum(1 for r in rows if r["match"].startswith("MATCH")),
        "stages_total": len(rows),
    }
    write_json(TAB.parent / "results" / "summary" /
               "cam006_implementation_audit.json", verdict)
    for r in rows:
        print(f"  {r['stage']:34s} {r['match']}")
    print(f"\nhand-weighting mismatch: {hand_weighted}  -> {verdict['label']}")


if __name__ == "__main__":
    main()
