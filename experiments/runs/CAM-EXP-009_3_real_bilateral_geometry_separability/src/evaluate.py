"""PHASE B step 5 — apply the frozen criteria and assign the verdict.

Every threshold comes from GATE in common.py, frozen before any distance was
computed. The verdict taxonomy is the one fixed in the spec.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (EPS, GATE, MANIFESTS, RAW, SUM, TAB,  # noqa: E402
                    cluster_bootstrap, fnum, read_csv, read_json, write_csv,
                    write_json)


def med(vals):
    v = [x for x in vals if np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def seq_level(rows, key, seqkey="sequence"):
    """Median across cameras within a sequence: camera views are never counted
    as independent participants."""
    by = defaultdict(list)
    for r in rows:
        by[r[seqkey]].append(fnum(r[key]))
    return {s: med(v) for s, v in by.items()}


def main():
    spec = read_json(MANIFESTS / "cam_exp_0093_geometry_spec_v1.json")
    cov = read_json(SUM / "coverage_audit.json")
    pid = read_json(SUM / "participant_identity_audit.json")

    rep = read_csv(RAW / "repeatability_distances.csv.gz")
    vrep = read_csv(RAW / "view_repeatability_distances.csv.gz")
    within = read_csv(RAW / "same_subject_distances.csv.gz")
    cross = read_csv(RAW / "cross_subject_distances.csv.gz")
    margins = read_csv(RAW / "margins.csv.gz")
    ident = read_csv(RAW / "identification.csv.gz")
    ctrl = read_csv(SUM / "control_summary.csv")
    perm = read_json(SUM / "permutation_summary.json")
    csame = read_csv(RAW / "cross_sequence_same_subject.csv.gz")

    cross_p = [c for c in cross if c["in_primary_cross"] == "1"]

    D_repeat = med([fnum(r["d_repeat"]) for r in rep])
    D_view = med([fnum(r["d_view_repeat"]) for r in vrep])
    D_within = med([fnum(r["d_within"]) for r in within])
    D_cross = med([fnum(r["d_cross"]) for r in cross_p])

    ratio = D_within / D_cross if D_cross else float("nan")
    S_sep = (D_cross - D_within) / max(D_repeat, EPS)

    # sequence-level aggregation (primary unit)
    sw = seq_level(within, "d_within")
    sr = seq_level(rep, "d_repeat")
    sc = defaultdict(list)
    for c in cross_p:
        sc[c["left_sequence"]].append(fnum(c["d_cross"]))
    sc = {k: med(v) for k, v in sc.items()}

    # margins
    m_rows = [r for r in margins]
    frac_nearest = float(np.mean([fnum(r["m_nearest"]) > 0 for r in m_rows]))
    frac_median = float(np.mean([fnum(r["m_median"]) > 0 for r in m_rows]))
    by_dir = {}
    for d in sorted(set(r["direction"] for r in m_rows)):
        sub = [r for r in m_rows if r["direction"] == d]
        by_dir[d] = {
            "median_m_nearest": med([fnum(r["m_nearest"]) for r in sub]),
            "median_m_median": med([fnum(r["m_median"]) for r in sub]),
            "frac_m_nearest_positive":
                float(np.mean([fnum(r["m_nearest"]) > 0 for r in sub])),
            "n": len(sub),
        }

    # identification
    ident_out = {}
    for d in sorted(set(r["direction"] for r in ident)):
        sub = [r for r in ident if r["direction"] == d]
        nc = med([fnum(r["n_candidates"]) for r in sub])
        ident_out[d] = {
            "top1_accuracy_pct": 100.0 * np.mean([r["correct"] == "1"
                                                  for r in sub]),
            "chance_pct": 100.0 / nc if nc else float("nan"),
            "n_trials": len(sub),
            "median_n_candidates": nc,
        }
    all_top1 = 100.0 * np.mean([r["correct"] == "1" for r in ident])
    chance = 100.0 / med([fnum(r["n_candidates"]) for r in ident])

    # bootstrap, clustered on sequence
    bw = cluster_bootstrap([fnum(r["d_within"]) for r in within],
                           [r["sequence"] for r in within])
    bc = cluster_bootstrap([fnum(c["d_cross"]) for c in cross_p],
                           [c["left_sequence"] for c in cross_p])
    br = cluster_bootstrap([fnum(r["d_repeat"]) for r in rep],
                           [r["sequence"] for r in rep])

    c1 = next((c for c in ctrl
               if c["control"] == "C1_BONE_MAPPING_PERMUTATION"), {})
    deg = fnum(c1.get("degradation_pct_vs_correct"))
    drop = fnum(c1.get("top1_drop_pp"))

    n_participants = cov["unique_candidate_participants"]
    underpowered = n_participants < GATE["min_participants_for_power"]
    provenance_unresolved = pid["STATUS"] == "SAME_SUBJECT_PROVENANCE_UNRESOLVED"

    crit = {
        "c1_ratio": bool(np.isfinite(ratio)
                         and ratio <= GATE["same_over_cross_ratio"]),
        "c2_nearest_margin": bool(
            frac_nearest >= GATE["nearest_margin_fraction"]),
        "c3_s_sep": bool(np.isfinite(S_sep) and S_sep >= GATE["s_sep_min"]),
        "c4_bone_control": bool(
            (np.isfinite(deg) and deg >= GATE["control_degradation_pct"])
            or (np.isfinite(drop) and drop >= GATE["control_top1_drop_pp"])),
        "c5_permutation_direction": bool(
            perm["observed"] > perm["null_median"]),
    }
    supported = all(crit.values())
    strong = (supported
              and np.isfinite(ratio) and ratio <= GATE["strong_ratio"]
              and frac_nearest >= GATE["strong_nearest_fraction"]
              and all_top1 > chance)

    tags = []
    if strong:
        tags.append("REAL_REFERENCE_BILATERAL_SIGNAL_STRONG")
    elif supported:
        tags.append("REAL_REFERENCE_BILATERAL_SIGNAL_SUPPORTED")
    elif np.isfinite(ratio) and ratio < 1.0:
        tags.append("REAL_REFERENCE_BILATERAL_SIGNAL_WEAK"
                    if S_sep >= 0.5 else "REFERENCE_VARIABILITY_DOMINATES")
    else:
        tags.append("NO_REAL_REFERENCE_BILATERAL_SEPARABILITY")
    if np.isfinite(S_sep) and S_sep < 1.0 and "REFERENCE_VARIABILITY_DOMINATES" \
            not in tags:
        tags.append("REFERENCE_VARIABILITY_COMPARABLE_TO_SEPARATION")
    if underpowered:
        tags.append("SUBJECT_SEPARABILITY_UNDERPOWERED")
    if provenance_unresolved:
        tags.append("SAME_SUBJECT_PROVENANCE_UNRESOLVED")

    cs_same = med([fnum(r["d"]) for r in csame
                   if r["comparison"] == "SAME_SIDE"])
    cs_opp = med([fnum(r["d"]) for r in csame
                  if r["comparison"] == "OPPOSITE_SIDE"])

    out = {
        "contrast_naming": "WITHIN_SEQUENCE vs CROSS_SEQUENCE. Participant "
                           "identity could not be established from shipped "
                           "evidence, so this is NOT reported as same-subject "
                           "vs cross-subject.",
        "population": {
            "sequences": cov["total_sequences"],
            "candidate_participants": n_participants,
            "physical_cameras": cov["physical_cameras"],
            "sequence_camera_units": cov["sequence_camera_units"],
            "eligible_rows": cov["eligible_rows"],
        },
        "distances": {
            "D_repeat_median": D_repeat, "D_repeat_ci": br[1:],
            "D_view_repeat_median": D_view,
            "D_within_median": D_within, "D_within_ci": bw[1:],
            "D_cross_median": D_cross, "D_cross_ci": bc[1:],
            "ratio_within_over_cross": ratio,
            "separation": D_cross - D_within,
            "S_sep": S_sep,
        },
        "sequence_level": {"within": sw, "cross": sc, "repeat": sr},
        "margins": {"frac_m_nearest_positive": frac_nearest,
                    "frac_m_median_positive": frac_median,
                    "by_direction": by_dir},
        "identification": {"combined_top1_pct": all_top1,
                           "chance_pct": chance, "by_direction": ident_out},
        "controls": ctrl,
        "permutation": perm,
        "cross_sequence_same_candidate_participant": {
            "median_same_side": cs_same, "median_opposite_side": cs_opp,
            "n": len(csame),
            "caveat": "one sequence pair only; descriptive, and its subject "
                      "relationship is exactly what is unverified",
        },
        "criteria": crit,
        "SUPPORTED": supported,
        "STRONG": strong,
        "VERDICT_TAGS": tags,
        "underpowered": underpowered,
        "thresholds": GATE,
    }
    write_json(SUM / "cam0093_verdict.json", out)

    write_csv(SUM / "separability_summary.csv", [
        {"metric": "D_repeat_median", "value": D_repeat},
        {"metric": "D_view_repeat_median", "value": D_view},
        {"metric": "D_within_median", "value": D_within},
        {"metric": "D_cross_median", "value": D_cross},
        {"metric": "ratio_within_over_cross", "value": ratio},
        {"metric": "S_sep", "value": S_sep},
        {"metric": "frac_m_nearest_positive", "value": frac_nearest},
        {"metric": "top1_pct", "value": all_top1},
        {"metric": "chance_pct", "value": chance},
    ])
    write_csv(SUM / "identification_summary.csv",
              [dict(direction=k, **v) for k, v in ident_out.items()])
    write_csv(TAB / "decision_summary.csv",
              [{"criterion": k, "passed": v} for k, v in crit.items()]
              + [{"criterion": "VERDICT", "passed": ";".join(tags)}])

    print("D_repeat      %.4f" % D_repeat)
    print("D_view_repeat %.4f" % D_view)
    print("D_within      %.4f" % D_within)
    print("D_cross       %.4f" % D_cross)
    print("ratio         %.3f   S_sep %.3f" % (ratio, S_sep))
    print("m_nearest>0   %.1f%%  top1 %.1f%% (chance %.1f%%)"
          % (100 * frac_nearest, all_top1, chance))
    for k, v in crit.items():
        print("  %-26s %s" % (k, v))
    print("VERDICT:", tags)


if __name__ == "__main__":
    main()
