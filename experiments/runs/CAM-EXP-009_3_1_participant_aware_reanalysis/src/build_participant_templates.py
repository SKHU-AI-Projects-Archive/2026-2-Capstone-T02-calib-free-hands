"""Participant-level templates and top-1 identification. SECONDARY diagnostic.

A participant template aggregates that participant's sequence-level templates
with EQUAL WEIGHT per sequence, so p41's two sessions count once between them.

This is a participant-level descriptive diagnostic. It is NOT a
session-independent proof: p41's own template is an average of two sessions
that the cross-session diagnostic shows disagree strongly.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (N_BONES, PARTICIPANT_OF, PARTICIPANTS, RAW,  # noqa: E402
                    SOURCE_ARTIFACTS, SUM, d_primary, read_csv, write_csv,
                    zof)


def main():
    rows = read_csv(SOURCE_ARTIFACTS["geometry_templates.csv.gz"])
    full = defaultdict(list)
    for r in rows:
        if r["template_type"] == "FULL":
            full[(r["sequence"], r["hand"])].append(zof(r))

    # sequence-level template = median across that sequence's cameras
    seq_t = {k: np.median(np.vstack(v), axis=0) for k, v in full.items()}

    # participant template = median across that participant's sequences
    part_t = {}
    for p in PARTICIPANTS:
        for hand in ("left", "right"):
            ts = [seq_t[(s, hand)] for s in sorted(PARTICIPANT_OF)
                  if PARTICIPANT_OF[s] == p and (s, hand) in seq_t]
            if ts:
                part_t[(p, hand)] = np.median(np.vstack(ts), axis=0)

    out = []
    for (p, hand), z in sorted(part_t.items()):
        rec = {"participant": p, "hand": hand,
               "n_sequences": sum(1 for s in PARTICIPANT_OF
                                  if PARTICIPANT_OF[s] == p)}
        for b in range(N_BONES):
            rec["z%02d" % b] = float(z[b])
        out.append(rec)
    write_csv(RAW / "participant_templates.csv.gz", out)

    # distance matrix + top-1, both directions
    mat, ident = [], []
    for direction, qh, th in (("LEFT_TO_RIGHT", "left", "right"),
                              ("RIGHT_TO_LEFT", "right", "left")):
        correct = 0
        for p in PARTICIPANTS:
            Q = part_t.get((p, qh))
            if Q is None:
                continue
            ds = {q: d_primary(Q, part_t[(q, th)]) for q in PARTICIPANTS
                  if (q, th) in part_t}
            for q, d in ds.items():
                mat.append({"direction": direction, "query_participant": p,
                            "target_participant": q, "same": int(p == q),
                            "d": d})
            best = min(ds, key=ds.get)
            correct += int(best == p)
            ident.append({"direction": direction, "participant": p,
                          "predicted": best, "correct": int(best == p),
                          "d_same": ds.get(p, np.nan),
                          "n_candidates": len(ds)})
    write_csv(RAW / "participant_distance_matrix.csv.gz", mat)

    summ = []
    for direction in ("LEFT_TO_RIGHT", "RIGHT_TO_LEFT"):
        sub = [r for r in ident if r["direction"] == direction]
        n = len(sub)
        acc = 100.0 * sum(r["correct"] for r in sub) / n if n else np.nan
        summ.append({"direction": direction, "n_participants": n,
                     "top1_accuracy_pct": acc,
                     "chance_pct": 100.0 / len(PARTICIPANTS),
                     "n_correct": sum(r["correct"] for r in sub)})
    allc = sum(r["correct"] for r in ident)
    summ.append({"direction": "COMBINED", "n_participants": len(ident),
                 "top1_accuracy_pct": 100.0 * allc / len(ident),
                 "chance_pct": 100.0 / len(PARTICIPANTS),
                 "n_correct": allc})
    write_csv(SUM / "participant_identification_summary.csv", summ)
    write_csv(SUM / "participant_identification_detail.csv", ident)

    for r in summ:
        print("%-16s n=%-3s top1 %.1f%% (chance %.1f%%)  correct=%s"
              % (r["direction"], r["n_participants"], r["top1_accuracy_pct"],
                 r["chance_pct"], r["n_correct"]))
    print("\nN_PARTICIPANTS = 4 -> descriptive only.")


if __name__ == "__main__":
    main()
