"""PHASE A freeze — units, splits, controls, permutation plan, thresholds.

Everything written here is fixed BEFORE any distance is computed.

One design decision is forced by the Phase A identity audit and is recorded
here rather than discovered later. No shipped GigaHands metadata declares a
participant field, so the `p<NN>` prefix cannot be used to assert that two
sequences are the same person. The primary contrast is therefore stated in the
terms the evidence actually supports:

    WITHIN_SEQUENCE   left and right of the SAME sequence and camera
    CROSS_SEQUENCE    left and right from DIFFERENT sequences, same camera

and the one pair that shares a name prefix (`p41-boxing-0021` /
`p41-plant-0004`) is held out of the primary cross set, because it is exactly
the pair whose subject relationship is ambiguous. It is reported separately as
the cross-sequence same-candidate-participant diagnostic.
"""
from __future__ import annotations

import collections
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BONES, GATE, MANIFESTS, MAX_FRAMES_PER_UNIT,  # noqa: E402
                    MIN_FRAMES_PER_HALF, MIN_FRAMES_PER_SIDE, N_BOOTSTRAP,
                    N_PERMUTATIONS, QC, QC_PASS_STATUSES, SUM, TAB,
                    finger_permutation, frame_half, read_csv, sha256,
                    write_csv, write_json)

SEC_PER_RECONSTRUCT = 0.122      # measured before any result


def select_frames(frames):
    """Deterministic, target-independent frame cap: evenly spaced by index.

    Never selects frames by fit quality, distance or any result.
    """
    f = sorted(set(frames))
    if len(f) <= MAX_FRAMES_PER_UNIT:
        return f
    idx = np.linspace(0, len(f) - 1, MAX_FRAMES_PER_UNIT)
    return [f[int(round(i))] for i in idx]


def main():
    qc = read_csv(QC)
    elig = [r for r in qc if r["qc_status"] in QC_PASS_STATUSES
            and r["triangulation_success"] == "1"]

    by_unit = collections.defaultdict(list)
    for r in elig:
        by_unit[(r["sequence"], r["camera"], r["hand"])].append(int(r["frame"]))

    units, split_rows = [], []
    for (seq, cam, hand), frames in sorted(by_unit.items()):
        if len(frames) < MIN_FRAMES_PER_SIDE:
            continue
        sel = select_frames(frames)
        halves = [frame_half(seq, cam, hand, f) for f in sel]
        nA, nB = halves.count(0), halves.count(1)
        units.append({
            "sequence": seq, "camera": cam, "hand": hand,
            "participant_group": seq.split("-")[0],
            "n_eligible_frames": len(frames), "n_selected_frames": len(sel),
            "n_half_A": nA, "n_half_B": nB,
            "repeatability_eligible": int(min(nA, nB) >= MIN_FRAMES_PER_HALF),
        })
        for f, h in zip(sel, halves):
            split_rows.append({"sequence": seq, "camera": cam, "hand": hand,
                               "frame": f, "half": "A" if h == 0 else "B"})

    units_p = MANIFESTS / "cam_exp_0093_eligible_units_v1.csv.gz"
    split_p = MANIFESTS / "cam_exp_0093_repeatability_split_v1.csv.gz"
    write_csv(units_p, units)
    write_csv(split_p, split_rows)

    # identity mapping, carried forward with its confidence grade intact
    idrows = read_csv(TAB / "participant_identity_audit.csv")
    id_p = MANIFESTS / "cam_exp_0093_identity_mapping_v1.csv"
    write_csv(id_p, idrows)

    seqs = sorted(set(u["sequence"] for u in units))
    groups = collections.defaultdict(list)
    for s in seqs:
        groups[s.split("-")[0]].append(s)
    ambiguous_pairs = [sorted(p) for g in groups.values() if len(g) > 1
                       for p in itertools.combinations(sorted(g), 2)]

    controls = {
        "C1_BONE_MAPPING_PERMUTATION": {
            "definition": "right-hand bone indices permuted WITHIN each finger "
                          "before the distance is taken",
            "permutation": [int(x) for x in finger_permutation()],
            "why_within_finger": "a global shuffle would also destroy the "
                                 "coarse size ordering of the bones, so its "
                                 "degradation could be explained without any "
                                 "anatomical correspondence",
            "expected": "WITHIN_SEQUENCE distance increases",
            "frozen_before_results": True,
        },
        "C2_SEQUENCE_LABEL_PERMUTATION": {
            "definition": "right-hand sequence labels permuted across "
                          "sequences, camera structure preserved",
            "n_permutations": N_PERMUTATIONS,
            "exact_if_small": "with 5 sequences all 5! = 120 relabelings are "
                              "enumerable, so the null is computed EXACTLY "
                              "rather than sampled",
            "expected": "the within-sequence advantage disappears",
            "frozen_before_results": True,
        },
    }

    n_recon = sum(u["n_selected_frames"] for u in units)
    spec = {
        "experiment": "CAM-EXP-009.3",
        "title": "Real Reference-Geometry Bilateral Separability Diagnostic",
        "estimates_focal": False,
        "created_before_distance_results": True,
        "research_question": "In the GigaHands reference geometry, is a "
                             "sequence's LEFT normalised bone structure closer "
                             "to its OWN RIGHT than to a RIGHT from a "
                             "different sequence?",
        "geometry_source": {
            "name": "OTHER_CAMERA_ONLY_REFERENCE_3D",
            "code": "CAM-EXP-001.3 src/loco.py, reused unchanged",
            "target_camera_excluded_before_reconstruction": True,
            "threshold_px": 8.0, "min_inlier_cameras": 4,
        },
        "topology": {
            "n_bones": len(BONES), "bones": [list(b) for b in BONES],
            "verified": "connected bones only; arbitrary joint pairs such as "
                        "4-8, 4-12 or 8-20 span no bone and are excluded",
        },
        "representation": {
            "step1": "20 connected bone lengths per frame per hand",
            "step2": "p_b = l_b / sum_b l_b  (absolute hand size removed)",
            "step3": "z_b = log(p_b + 1e-8)  (eps frozen, matches CAM-009.2)",
            "template": "bone-wise median of z over a unit's eligible frames",
        },
        "primary_distance": "median_b |z_A,b - z_B,b|",
        "secondary_distances": ["mean_abs_log", "raw_proportion_l1", "cosine",
                                "finger_internal"],
        "eligibility": {
            "source": "gigahands_demo_qc_v1.csv.gz (frozen CAM-EXP-001.3 QC)",
            "accepted_qc_status": list(QC_PASS_STATUSES),
            "requires_triangulation_success": True,
            "min_frames_per_side": MIN_FRAMES_PER_SIDE,
            "min_frames_per_repeatability_half": MIN_FRAMES_PER_HALF,
            "max_frames_per_unit": MAX_FRAMES_PER_UNIT,
            "frame_cap_rule": "evenly spaced by frame index; deterministic and "
                              "target-independent",
            "no_performance_based_filtering": "frames are never removed for "
                                              "having a large left/right "
                                              "distance, an unusual bone, or "
                                              "any metric outcome",
        },
        "repeatability_split": "SHA-256 parity of "
                               "(sequence|camera|hand|frame); deterministic "
                               "and independent of fit quality",
        "identity": {
            "STATUS": "SAME_SUBJECT_PROVENANCE_UNRESOLVED",
            "consequence": "the primary contrast is stated as WITHIN_SEQUENCE "
                           "vs CROSS_SEQUENCE, not same-subject vs "
                           "cross-subject",
            "ambiguous_pairs_excluded_from_primary_cross": ambiguous_pairs,
            "ambiguous_pair_reported_as":
                "CROSS_SEQUENCE_SAME_CANDIDATE_PARTICIPANT_DIAGNOSTIC",
        },
        "cross_pairing": {
            "camera_matched": True,
            "why": "so that WITHIN and CROSS comparisons are not separated "
                   "artificially by camera/view reconstruction differences",
            "unmatched_pairs": "secondary only",
            "aggregation": "camera views are NEVER counted as independent "
                           "participants; distances are aggregated to "
                           "sequence-camera, then sequence level",
        },
        "controls": controls,
        "statistics": {
            "permutation": "exact over sequence relabelings (5! = 120)",
            "bootstrap": {"n": N_BOOTSTRAP, "cluster": "sequence"},
            "primary_unit": "sequence-camera, aggregated to sequence",
        },
        "gate": GATE,
        "underpowered_rule": "if unique participants < %d, tag "
                             "SUBJECT_SEPARABILITY_UNDERPOWERED, report "
                             "descriptively, and do not conclude from a "
                             "p-value alone" % GATE["min_participants_for_power"],
        "compute": {
            "measured_seconds_per_reconstruction": SEC_PER_RECONSTRUCT,
            "planned_reconstructions": n_recon,
            "estimated_minutes": round(n_recon * SEC_PER_RECONSTRUCT / 60, 1),
            "fixed_from_runtime_before_any_result": True,
        },
        "never_used": ["dataset reference focal", "AnyCalib", "GeoCalib",
                       "E2", "any candidate focal grid", "scene fusion",
                       "WiLoR or any learned hand latent",
                       "the FINAL_CONFIRMATORY_HOLDOUT"],
    }
    spec_p = MANIFESTS / "cam_exp_0093_geometry_spec_v1.json"
    ctrl_p = MANIFESTS / "cam_exp_0093_controls_spec_v1.json"
    perm_p = MANIFESTS / "cam_exp_0093_permutation_spec_v1.json"
    write_json(spec_p, spec)
    write_json(ctrl_p, controls)
    write_json(perm_p, {
        "primary_test": "exact permutation over sequence relabelings",
        "n_sequences": len(seqs), "n_relabelings": 120,
        "statistic": "median(CROSS) - median(WITHIN), sequence-level",
        "n_permutations_if_sampled": N_PERMUTATIONS,
        "seed_namespace": "CAM0093_PERM",
        "frozen_before_results": True,
    })

    write_csv(TAB / "primary_metric_definition.csv", [
        {"name": "D_primary", "tier": "PRIMARY",
         "formula": "median_b |log(p_L,b+1e-8) - log(p_R,b+1e-8)|",
         "note": "one primary metric; secondaries are sensitivity only"},
        {"name": "mean_abs_log", "tier": "SECONDARY",
         "formula": "mean_b |z_A,b - z_B,b|", "note": ""},
        {"name": "raw_proportion_l1", "tier": "SECONDARY",
         "formula": "mean_b |p_A,b - p_B,b|", "note": ""},
        {"name": "cosine", "tier": "SECONDARY",
         "formula": "1 - cos(p_A, p_B)", "note": ""},
        {"name": "finger_internal", "tier": "SECONDARY",
         "formula": "median_b |log r_A - log r_B|, r normalised per finger",
         "note": ""},
    ])
    write_csv(TAB / "statistical_units.csv", [
        {"level": "frame", "independent": "NO",
         "note": "frames within a unit are highly correlated"},
        {"level": "sequence-camera", "independent": "NO",
         "note": "49 cameras observe the same hands; not independent people"},
        {"level": "sequence", "independent": "PARTIAL",
         "note": "primary aggregation level"},
        {"level": "candidate participant", "independent": "UNVERIFIED",
         "note": "identity provenance unresolved; only 4 candidate groups"},
    ])

    hashes = {p.name: sha256(p) for p in (spec_p, ctrl_p, perm_p, units_p,
                                          split_p, id_p)}
    write_json(SUM / "manifest_hashes.json", hashes)

    print("eligible side-units      %d" % len(units))
    print("repeatability-eligible   %d"
          % sum(u["repeatability_eligible"] for u in units))
    print("planned reconstructions  %d  (~%.0f min)"
          % (n_recon, n_recon * SEC_PER_RECONSTRUCT / 60))
    print("ambiguous prefix pairs   %s" % ambiguous_pairs)
    for k, v in hashes.items():
        print("  %-52s %s" % (k, v[:16]))


if __name__ == "__main__":
    main()
