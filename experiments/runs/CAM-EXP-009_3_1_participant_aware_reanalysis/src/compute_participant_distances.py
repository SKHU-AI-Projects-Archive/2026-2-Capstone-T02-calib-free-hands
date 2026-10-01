"""Participant-aware distances, with the three quantities kept separate.

  D_WITHIN_SESSION                  same participant AND same session
  D_SAME_PARTICIPANT_CROSS_SESSION  same participant, different session (p41)
  D_CROSS_PARTICIPANT               different participants

Weighting rules, stated before the numbers:
  * camera views are aggregated to a sequence-camera, then to a sequence;
  * p41's two sessions are averaged to ONE participant value, so p41 does not
    carry double weight;
  * cross-participant distances are summarised per ORDERED PARTICIPANT PAIR
    first, so each participant pair gets equal weight regardless of how many
    sequences it contributes.
"""
from __future__ import annotations

import itertools
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (MULTI_SESSION, PARTICIPANT_OF, PARTICIPANTS,  # noqa: E402
                    RAW, SOURCE_ARTIFACTS, SUM, d_primary, med, read_csv,
                    write_csv, zof)


def load_templates():
    rows = read_csv(SOURCE_ARTIFACTS["geometry_templates.csv.gz"])
    full = {}
    for r in rows:
        if r["template_type"] == "FULL":
            full[(r["sequence"], r["camera"], r["hand"])] = zof(r)
    return full


def main():
    full = load_templates()
    cams = sorted(set(c for (_s, c, _h) in full))
    seqs = sorted(set(s for (s, _c, _h) in full))

    # ---------------------------------------------- within session
    within = []
    for cam in cams:
        for s in seqs:
            L = full.get((s, cam, "left"))
            R = full.get((s, cam, "right"))
            if L is None or R is None:
                continue
            within.append({"participant": PARTICIPANT_OF[s], "sequence": s,
                           "camera": cam, "d": d_primary(L, R)})
    write_csv(RAW / "within_session_distances.csv.gz", within)

    # sequence -> participant, p41's two sessions averaged to one value
    by_seq = defaultdict(list)
    for r in within:
        by_seq[r["sequence"]].append(r["d"])
    seq_within = {s: med(v) for s, v in by_seq.items()}
    part_within = {}
    for p in PARTICIPANTS:
        ss = [s for s in seqs if PARTICIPANT_OF[s] == p]
        part_within[p] = med([seq_within[s] for s in ss if s in seq_within])

    # ---------------------------------------------- cross participant
    cross_raw, ordered = [], defaultdict(list)
    for cam in cams:
        for sl in seqs:
            L = full.get((sl, cam, "left"))
            if L is None:
                continue
            for sr in seqs:
                if PARTICIPANT_OF[sl] == PARTICIPANT_OF[sr]:
                    continue          # excludes p41<->p41 by construction
                R = full.get((sr, cam, "right"))
                if R is None:
                    continue
                d = d_primary(L, R)
                cross_raw.append({
                    "left_participant": PARTICIPANT_OF[sl],
                    "right_participant": PARTICIPANT_OF[sr],
                    "left_sequence": sl, "right_sequence": sr,
                    "camera": cam, "direction": "LEFT_TO_RIGHT", "d": d})
                ordered[(PARTICIPANT_OF[sl], PARTICIPANT_OF[sr],
                         "LEFT_TO_RIGHT")].append(d)
            # reverse direction: this sequence's RIGHT vs others' LEFT
            Rq = full.get((sl, cam, "right"))
            if Rq is None:
                continue
            for sr in seqs:
                if PARTICIPANT_OF[sl] == PARTICIPANT_OF[sr]:
                    continue
                Lt = full.get((sr, cam, "left"))
                if Lt is None:
                    continue
                d = d_primary(Rq, Lt)
                cross_raw.append({
                    "left_participant": PARTICIPANT_OF[sl],
                    "right_participant": PARTICIPANT_OF[sr],
                    "left_sequence": sl, "right_sequence": sr,
                    "camera": cam, "direction": "RIGHT_TO_LEFT", "d": d})
                ordered[(PARTICIPANT_OF[sl], PARTICIPANT_OF[sr],
                         "RIGHT_TO_LEFT")].append(d)
    write_csv(RAW / "cross_participant_distances.csv.gz", cross_raw)

    pair_rows = []
    for (pi, pj, direction), v in sorted(ordered.items()):
        pair_rows.append({"left_participant": pi, "right_participant": pj,
                          "direction": direction, "n": len(v),
                          "median_d": med(v)})
    write_csv(RAW / "participant_pair_distances.csv.gz", pair_rows)

    # PRIMARY cross-participant distribution: participant-pair medians,
    # equal weight per pair
    pair_med_lr = [r["median_d"] for r in pair_rows
                   if r["direction"] == "LEFT_TO_RIGHT"]
    pair_med_rl = [r["median_d"] for r in pair_rows
                   if r["direction"] == "RIGHT_TO_LEFT"]
    pair_bidir = defaultdict(list)
    for r in pair_rows:
        pair_bidir[(r["left_participant"], r["right_participant"])].append(
            r["median_d"])
    bidir = {k: med(v) for k, v in pair_bidir.items()}

    # ---------------------------------------------- summary
    summary = [
        {"quantity": "D_within_session_all_units", "n": len(within),
         "median": med([r["d"] for r in within])},
        {"quantity": "D_within_session_participant_equal_weight",
         "n": len(PARTICIPANTS),
         "median": med(list(part_within.values()))},
        {"quantity": "D_cross_participant_all_pairs", "n": len(cross_raw),
         "median": med([r["d"] for r in cross_raw])},
        {"quantity": "D_cross_participant_pair_equal_weight",
         "n": len(bidir), "median": med(list(bidir.values()))},
        {"quantity": "D_cross_participant_LEFT_TO_RIGHT",
         "n": len(pair_med_lr), "median": med(pair_med_lr)},
        {"quantity": "D_cross_participant_RIGHT_TO_LEFT",
         "n": len(pair_med_rl), "median": med(pair_med_rl)},
    ]
    write_csv(SUM / "participant_distance_summary.csv", summary)

    per_p = [{"participant": p, "n_sequences":
              sum(1 for s in seqs if PARTICIPANT_OF[s] == p),
              "within_session_median": part_within[p],
              "sequences": ";".join(s for s in seqs
                                    if PARTICIPANT_OF[s] == p)}
             for p in PARTICIPANTS]
    write_csv(SUM / "session_subject_summary.csv", per_p)

    for r in summary:
        print("%-52s n=%-6s %.5f" % (r["quantity"], r["n"], r["median"]))
    print()
    for r in per_p:
        print("  %-5s seq=%d  within_session %.5f"
              % (r["participant"], r["n_sequences"],
                 r["within_session_median"]))


if __name__ == "__main__":
    main()
