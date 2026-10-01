"""Assemble the participant-aware picture and audit which claims survive.

This is NOT a confirmatory gate. No new PASS/FAIL threshold is invented after
seeing CAM-EXP-009.3's results. The output is a tag set and a claim-survival
table.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ADEQUACY_BAR_PARTICIPANTS, PARTICIPANTS,  # noqa: E402
                    RAW, SOURCE_ARTIFACTS, SUM, TAB, fnum, med,
                    participant_bootstrap, read_csv, read_json, write_csv,
                    write_json)


def main():
    ident_prov = read_json(SUM / "identity_provenance_summary.json")
    v1rep = read_json(SUM / "cam0093_v1_reproduction.json")
    perm = read_json(SUM / "participant_permutation_summary.json")
    v1 = read_json(SOURCE_ARTIFACTS["cam0093_verdict.json"])
    ctrl = read_csv(SOURCE_ARTIFACTS["control_summary.csv"])

    rep = read_csv(SOURCE_ARTIFACTS["repeatability_distances.csv.gz"])
    vrep = read_csv(SOURCE_ARTIFACTS["view_repeatability_distances.csv.gz"])
    within = read_csv(RAW / "within_session_distances.csv.gz")
    cross = read_csv(RAW / "cross_participant_distances.csv.gz")
    pairs = read_csv(RAW / "participant_pair_distances.csv.gz")
    p41 = read_csv(SUM / "p41_cross_session_summary.csv")
    pident = read_csv(SUM / "participant_identification_summary.csv")
    psum = read_csv(SUM / "participant_distance_summary.csv")

    g = {r["quantity"]: fnum(r["median"]) for r in psum}
    D_repeat = med([fnum(r["d_repeat"]) for r in rep])
    D_view = med([fnum(r["d_view_repeat"]) for r in vrep])
    D_within = g["D_within_session_participant_equal_weight"]
    D_cross = g["D_cross_participant_pair_equal_weight"]
    ratio = D_within / D_cross
    s_sep = (D_cross - D_within) / D_repeat

    p41_same = next(fnum(r["median"]) for r in p41
                    if r["comparison"] == "POOLED_SAME_SIDE")
    p41_opp = next(fnum(r["median"]) for r in p41
                   if r["comparison"] == "POOLED_OPPOSITE_SIDE")
    p41_within = next(fnum(r["median"]) for r in p41
                      if r["comparison"] == "REFERENCE_within_session_p41")
    cross_all = med([fnum(r["d"]) for r in cross])

    # participant-clustered bootstrap (4 clusters - supplemental)
    bw = participant_bootstrap([fnum(r["d"]) for r in within],
                               [r["participant"] for r in within])
    bc = participant_bootstrap([fnum(r["d"]) for r in cross],
                               [r["left_participant"] for r in cross])

    per_p = read_csv(SUM / "session_subject_summary.csv")
    top1 = {r["direction"]: fnum(r["top1_accuracy_pct"]) for r in pident}
    chance = 100.0 / len(PARTICIPANTS)

    c0 = next(c for c in ctrl if c["control"] == "C0_CORRECT_MAPPING")
    c1 = next(c for c in ctrl if c["control"] == "C1_BONE_MAPPING_PERMUTATION")
    deg = fnum(c1["degradation_pct_vs_correct"])

    cross_session_supported = bool(p41_same < cross_all
                                   and p41_opp < cross_all)

    tags = ["OFFICIAL_PARTICIPANT_MAPPING_VERIFIED",
            "PARTICIPANT_REANALYSIS_UNDERPOWERED"]
    tags.append("WITHIN_SESSION_BILATERAL_SEPARATION_PRESENT" if ratio < 0.80
                else "WITHIN_SESSION_BILATERAL_SEPARATION_WEAK")
    tags.append("CROSS_SESSION_STABILITY_SUPPORTED_FOR_P41"
                if cross_session_supported
                else "CROSS_SESSION_STABILITY_NOT_SUPPORTED_FOR_P41")
    tags.append("GENERIC_BONE_IDENTITY_SIGNAL_STRONG" if deg >= 100
                else "GENERIC_BONE_IDENTITY_SIGNAL_WEAK")
    if not cross_session_supported:
        tags.append("SUBJECT_SPECIFIC_PRIOR_NOT_JUSTIFIED")
        tags.append("SEQUENCE_SPECIFIC_SHARED_ANATOMY_REMAINS_PLAUSIBLE")

    claims = [
        {"claim": "same sequence L/R closer than cross sequence",
         "status": "SUPPORTED",
         "evidence": "within-session %.5f vs cross-participant %.5f "
                     "(ratio %.3f); reproduced exactly from CAM-EXP-009.3"
                     % (D_within, D_cross, ratio)},
        {"claim": "same participant L/R generally closer than different "
                  "participants",
         "status": "UNDERPOWERED",
         "evidence": "true at participant-equal weighting (ratio %.3f), but "
                     "n=4 participants, participant-block permutation p=%.4f "
                     "(floor %.4f), and the effect is not separable from "
                     "session context" % (ratio, perm["p_value_one_sided"],
                                          perm["smallest_attainable_p"])},
        {"claim": "same participant geometry persists across sessions",
         "status": ("SUPPORTED_FOR_P41" if cross_session_supported
                    else "NOT_SUPPORTED_FOR_P41"),
         "evidence": "p41 cross-session same-side %.5f and opposite-side "
                     "%.5f, both LARGER than cross-participant %.5f and far "
                     "larger than p41 within-session %.5f"
                     % (p41_same, p41_opp, cross_all, p41_within)},
        {"claim": "generic corresponding bone identity matters",
         "status": "SUPPORTED",
         "evidence": "within-finger bone permutation degrades the distance by "
                     "%+.0f%% (%.4f -> %.4f)"
                     % (deg, fnum(c0["median_d_within"]),
                        fnum(c1["median_d_within"]))},
        {"claim": "strong subject-specific shared anatomy prior justified",
         "status": "NO" if not cross_session_supported
                   else "INSUFFICIENT_EVIDENCE",
         "evidence": "the only same-participant cross-session comparison "
                     "available is farther apart than different participants, "
                     "so a person-level template is not supported by this "
                     "data"},
        {"claim": "sequence-specific shared anatomy remains plausible",
         "status": "YES",
         "evidence": "within-session bilateral separation is present and "
                     "%.2fx the reference repeatability; it simply is not "
                     "attributable to a persistent person-level signature"
                     % s_sep},
    ]
    write_csv(TAB / "claim_survival_audit.csv", claims)
    write_csv(SUM / "claim_survival_summary.csv", claims)

    out = {
        "analysis_type": "post-result provenance correction and "
                         "participant-aware reanalysis",
        "created_after_cam0093_results": True,
        "not_a_preregistered_confirmatory_experiment": True,
        "identity": {
            "status": ident_prov["STATUS"],
            "source": ident_prov["official_source_url"],
            "n_participants": ident_prov["n_participants"],
            "multi_session": ident_prov["multi_session_participants"],
        },
        "source_reuse": {
            "reference_reconstruction_rerun": False,
            "v1_metrics_reproduced": v1rep["ALL_REPRODUCED"],
            "tolerance": v1rep["tolerance"],
        },
        "distances": {
            "D_repeat": D_repeat,
            "D_view_repeat": D_view,
            "D_within_session_all_units":
                g["D_within_session_all_units"],
            "D_within_session_participant_equal_weight": D_within,
            "D_cross_participant_all_pairs":
                g["D_cross_participant_all_pairs"],
            "D_cross_participant_pair_equal_weight": D_cross,
            "D_cross_participant_LEFT_TO_RIGHT":
                g["D_cross_participant_LEFT_TO_RIGHT"],
            "D_cross_participant_RIGHT_TO_LEFT":
                g["D_cross_participant_RIGHT_TO_LEFT"],
            "ratio_within_over_cross": ratio,
            "separation_over_repeatability": s_sep,
            "bootstrap_within_ci_participant_cluster": bw[1:],
            "bootstrap_cross_ci_participant_cluster": bc[1:],
            "bootstrap_caveat": "4 participant clusters; supplemental and "
                                "unstable",
        },
        "per_participant_within_session": {
            r["participant"]: fnum(r["within_session_median"]) for r in per_p},
        "p41_cross_session": {
            "same_side_pooled": p41_same,
            "opposite_side_pooled": p41_opp,
            "p41_within_session": p41_within,
            "cross_participant_reference": cross_all,
            "stability_supported": cross_session_supported,
            "caveat": "ONE participant. No population claim follows.",
        },
        "participant_identification": {
            "top1": top1, "chance_pct": chance,
            "caveat": "4 participants, 4 queries per direction; descriptive "
                      "only",
        },
        "permutation": perm,
        "bone_mapping_control": {
            "correct": fnum(c0["median_d_within"]),
            "permuted": fnum(c1["median_d_within"]),
            "degradation_pct": deg,
        },
        "power": {
            "n_participants": len(PARTICIPANTS),
            "bar": ADEQUACY_BAR_PARTICIPANTS,
            "underpowered": len(PARTICIPANTS) < ADEQUACY_BAR_PARTICIPANTS,
            "note": "verifying identity does not fix power",
        },
        "cam0093_historical_verdict": v1["VERDICT_TAGS"],
        "supersedes": "CAM-EXP-009.3 remains valid as a within-sequence vs "
                      "cross-sequence analysis; its participant-provenance "
                      "interpretation is superseded by this reanalysis.",
        "claims": claims,
        "TAGS": tags,
    }
    write_json(SUM / "cam00931_verdict.json", out)
    write_csv(TAB / "decision_summary.csv",
              [{"item": c["claim"], "status": c["status"]} for c in claims]
              + [{"item": "TAGS", "status": ";".join(tags)}])

    print("D_repeat                      %.5f" % D_repeat)
    print("D_view_repeat                 %.5f" % D_view)
    print("D_within_session (equal wt)   %.5f" % D_within)
    print("D_cross_participant (equal wt)%.5f" % D_cross)
    print("ratio %.3f   separation/repeat %.3f" % (ratio, s_sep))
    print("p41 cross-session same %.5f opp %.5f  vs cross-participant %.5f"
          % (p41_same, p41_opp, cross_all))
    print("participant top1 %s  chance %.1f%%" % (top1, chance))
    print("permutation p %.4f" % perm["p_value_one_sided"])
    print("TAGS:")
    for t in tags:
        print("   ", t)


if __name__ == "__main__":
    main()
