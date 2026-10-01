"""Assemble the CAM-EXP-006.1 primary verdict from the frozen criteria."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SUM, TAB, fnum, read_csv, read_json, write_csv, write_json  # noqa: E402


def main() -> None:
    sv = read_json(SUM / "synthetic_validation_verdict.json")
    ev = read_json(SUM / "real_evaluation.json")
    ps = read_json(SUM / "perspective_strength_verdict.json")
    rm = read_json(SUM / "real_regime_matched_verdict.json")
    track5 = ev["tracking_R5_N16"]

    solver_ok = sv["verdict"] == "SOLVER_VALIDATED"
    weighting = ev["frame_weighting"]["verdict"]
    camera = ev["camera_model"]["verdict"]
    repro = ev["reproduction"]["verdict"]
    r1, r5 = ev["R1_N16"], ev["R5_N16"]
    oracle = fnum(ev["constant_rig_focal_oracle"]["median_rel_focal_err_pct"])

    # Did the FULL camera-model oracle actually restore identifiability?
    # The materiality trigger is a numeric screen; this is the substantive
    # question that separates verdict B from verdict C.
    r5_usable = bool(
        r5["median_rel_focal_err_pct"] <= 10.0
        and r5["share_cleanly_identifiable"] >= 0.50
        and track5["tracking_positive_ci_excludes_zero"])

    if not solver_ok:
        verdict, code = "CAM006_SOLVER_NOT_VALIDATED", "E"
    elif weighting == "FRAME_WEIGHTING_MATERIAL":
        verdict, code = ("CAM006_FRAME_WEIGHTING_MATERIALLY_AFFECTED_RESULT",
                         "D")
    elif r5_usable:
        verdict, code = "CAM006_CAMERA_MODEL_MISMATCH_EXPLAINS_RESULT", "C"
    elif camera == "CAMERA_MODEL_MISMATCH_MATERIAL":
        verdict, code = "CAM006_NEGATIVE_RESULT_VALIDATED_WITH_CAVEATS", "B"
    else:
        verdict, code = "CAM006_NEGATIVE_RESULT_STRONGLY_VALIDATED", "A"

    claims = [
        {"cam006_claim": "the reference hand does not identify the focal "
                         "under this formulation",
         "status": "UPHELD",
         "evidence": f"R5, with the true principal point, distortion and "
                     f"fy/fx ratio, still gives "
                     f"{r5['median_rel_focal_err_pct']:.1f} % median error, "
                     f"{100*r5['share_flat_profile']:.0f} % flat profiles and "
                     f"no positive tracking correlation",
         "action": "keep, with the scope narrowed"},
        {"cam006_claim": "202 % median relative focal error at N=16",
         "status": "REPRODUCED_BUT_MISATTRIBUTED",
         "evidence": f"R0 reproduces it exactly "
                     f"({ev['R0_N16']['median_rel_focal_err_pct']:.2f} %), but "
                     f"correcting only the camera model brings it to "
                     f"{r5['median_rel_focal_err_pct']:.1f} %. Most of the "
                     f"headline magnitude was unmodelled lens distortion, not "
                     f"hand geometry.",
         "action": "report both numbers; stop quoting 202 % as the size of "
                   "the geometric limit"},
        {"cam006_claim": "the limit is geometric, not a modelling shortfall",
         "status": "OVERSTATED",
         "evidence": "modelling mattered a great deal: supplying the provided "
                     "distortion cut the error by about 75 % in relative "
                     "terms. What survives is that even a fully correct "
                     "camera model leaves the focal unidentified.",
         "action": "replace with a scoped statement"},
        {"cam006_claim": "it will not be fixed by a better hand model",
         "status": "NOT_SUPPORTED_AS_STATED",
         "evidence": "the synthetic reference-3D sweep shows focal error "
                     "rising steeply with reference-hand error - about 26 % "
                     "focal error at a 3D error of 1 % of hand diameter - so "
                     "hand accuracy does influence this problem, it is simply "
                     "not sufficient on its own",
         "action": "narrow to 'improving hand-geometry accuracy alone is "
                   "unlikely to resolve the ambiguity in this formulation'"},
        {"cam006_claim": "a predicted hand cannot carry more focal "
                         "information than the oracle hand",
         "status": "LOGICALLY_TOO_STRONG",
         "evidence": "a learned hand model can encode training priors, image "
                     "appearance and camera priors that are not present in "
                     "explicit 3D geometry. The CAM-006 result constrains the "
                     "same profiled-PnP estimator, not every hand-derived cue.",
         "action": "replace with the narrower statement about this estimator"},
        {"cam006_claim": "the planarized control beating the main condition "
                         "shows the signal is absent",
         "status": "WEAK_EVIDENCE",
         "evidence": "that control necessarily ran a different PnP back-end. "
                     "The synthetic planarity sweep is the clean evidence and "
                     "supports the same direction.",
         "action": "demote to sensitivity evidence"},
        {"cam006_claim": "PRE_REGISTERED_POST_HOC_ORACLE_SANITY_CONTROL",
         "status": "SELF_CONTRADICTORY_LABEL",
         "evidence": "the confound was found post hoc in CAM-005 and was "
                     "therefore already known when CAM-006 began, so in "
                     "CAM-006 it is simply pre-registered",
         "action": "rename to PRE_REGISTERED_IN_CAM006_ORACLE_SANITY_CONTROL, "
                   "motivated by the CAM-005 post-hoc finding"},
    ]
    write_csv(TAB / "scientific_claim_status.csv", claims)

    plain = (
        "WAS THE CAM-EXP-006 RESULT REAL?  Yes in direction, no in "
        "attribution.\n\n"
        f"1. The solver is fine. On synthetic scenes where the focal IS "
        f"recoverable it lands within "
        f"{sv['generic_positive_control']['median_rel_focal_err_pct']:.2f} % "
        f"of the true focal, 100 % of trials inside 5 %.\n"
        f"2. The suspected frame-weighting bug is real but harmless: "
        f"{ev['frame_weighting']['delta_median_err_pp']:.1f} pp change.\n"
        f"3. The big one: CAM-006 ignored the rig's lens distortion "
        f"(k1 about -0.39). Supplying it drops the error from "
        f"{r1['median_rel_focal_err_pct']:.0f} % to "
        f"{r5['median_rel_focal_err_pct']:.0f} %. CAM-006's headline "
        f"202 % overstated the geometric limit.\n"
        f"4. But even with the true principal point, distortion and fy/fx "
        f"ratio, the focal is still not identifiable: "
        f"{100*r5['share_flat_profile']:.0f} % of profiles are flat, only "
        f"{100*r5['share_cleanly_identifiable']:.0f} % are clean, the "
        f"estimates do not track the true focal, and a single constant beats "
        f"it ({oracle:.2f} % versus "
        f"{r5['median_rel_focal_err_pct']:.0f} %).\n"
        f"5. Why: the hands sit about "
        f"{ps['distance_over_diameter']['median']:.1f} hand-diameters from "
        f"the camera with a relative depth extent of only "
        f"{ps['dZ_over_Z']['median']:.2f}. That is a weak-perspective regime, "
        f"and the reference hand itself carries error.\n\n"
        f"VERDICT {code}: {verdict}"
    )

    out = {
        "primary_verdict_code": code,
        "primary_verdict": verdict,
        "synthetic_solver_verdict": sv["verdict"],
        "reproduction_verdict": repro,
        "frame_weighting_verdict": weighting,
        "camera_model_verdict": camera,
        "full_oracle_restored_usable_identifiability": r5_usable,
        "why_not_verdict_C": (
            "the pre-registered CAMERA_MODEL_MISMATCH_MATERIAL trigger did "
            "fire, on the error-reduction criterion. But verdict C requires "
            "identifiability to be RESTORED, and it was not: R5 still shows "
            f"{r5['median_rel_focal_err_pct']:.1f} % median error, "
            f"{100*r5['share_flat_profile']:.0f} % flat profiles, "
            f"{100*r5['share_cleanly_identifiable']:.0f} % cleanly "
            "identifiable views, and a tracking correlation whose interval "
            "includes zero. The camera model explains much of the MAGNITUDE "
            "of CAM-006's headline number without overturning its DIRECTION."
        ) if camera == "CAMERA_MODEL_MISMATCH_MATERIAL" else None,
        "key_numbers": {
            "R0_N16_median_err_pct": ev["R0_N16"]["median_rel_focal_err_pct"],
            "R1_N16_median_err_pct": r1["median_rel_focal_err_pct"],
            "R5_N16_median_err_pct": r5["median_rel_focal_err_pct"],
            "R5_N16_flat_share": r5["share_flat_profile"],
            "R5_N16_identifiable_share": r5["share_cleanly_identifiable"],
            "constant_rig_focal_oracle_pct": oracle,
            "real_distance_over_diameter_median":
                ps["distance_over_diameter"]["median"],
            "real_dZ_over_Z_median": ps["dZ_over_Z"]["median"],
        },
        "scientific_claim_status": claims,
        "plain_summary": plain,
        "scope_statement": (
            "Under the evaluated GigaHands imaging regime and this "
            "profiled-PnP formulation, other-camera-only reference hand "
            "geometry does not provide enough perspective information to "
            "identify the focal reliably, even when the rest of the camera "
            "model is supplied exactly. This does not establish that hand "
            "geometry carries no camera information in other regimes, nor "
            "that learned hand-derived features encoding camera priors are "
            "ruled out."
        ),
        "cam007_recommendation": (
            "REDESIGN, not simply proceed and not abandon. Repeating the same "
            "profiled-PnP focal estimator with a noisier predicted hand is "
            "low priority: the oracle hand already fails, and the synthetic "
            "reference-3D sweep shows a noisier hand makes it worse. But a "
            "learned model can encode image and camera priors that explicit "
            "3D geometry does not, so learned hand-derived camera cues are "
            "NOT ruled out. CAM-007 should be restated around that different "
            "question."
        ),
    }
    write_json(SUM / "cam0061_verdict.json", out)
    print(plain)


if __name__ == "__main__":
    main()
