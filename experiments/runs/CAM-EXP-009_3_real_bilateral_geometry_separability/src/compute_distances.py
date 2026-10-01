"""PHASE B step 3 — the three distances, plus margins and identification.

  D_repeat       same sequence, camera, hand; half A template vs half B
  D_view_repeat  same sequence, hand; camera i template vs camera j
  D_within       same sequence, same camera; LEFT vs RIGHT
  D_cross        different sequences, SAME camera; LEFT vs RIGHT

Naming: the contrast is WITHIN_SEQUENCE vs CROSS_SEQUENCE, because the Phase A
audit could not establish participant identity from shipped evidence. The one
sequence pair sharing a name prefix is flagged and excluded from the primary
cross set.

Cross pairs are camera-matched so that the two contrasts are not separated
artificially by camera/view reconstruction differences.
"""
from __future__ import annotations

import itertools
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (MANIFESTS, N_BONES, RAW, SECONDARY, d_primary,  # noqa: E402
                    finger_permutation, read_csv, read_json, write_csv)

OTHER = ("left", "right")


def zof(r):
    return np.array([float(r["z%02d" % b]) for b in range(N_BONES)], float)


def load():
    rows = read_csv(RAW / "geometry_templates.csv.gz")
    full, halves = {}, {}
    for r in rows:
        k = (r["sequence"], r["camera"], r["hand"])
        if r["template_type"] == "FULL":
            full[k] = zof(r)
        else:
            halves[k + (r["half"],)] = zof(r)
    return full, halves


def main():
    spec = read_json(MANIFESTS / "cam_exp_0093_geometry_spec_v1.json")
    ambiguous = {tuple(sorted(p)) for p in
                 spec["identity"]["ambiguous_pairs_excluded_from_primary_cross"]}
    perm = finger_permutation()
    full, halves = load()

    # ---------------------------------------------------- repeatability
    rep = []
    for (seq, cam, hand) in sorted(set(full)):
        a = halves.get((seq, cam, hand, "A"))
        b = halves.get((seq, cam, hand, "B"))
        if a is None or b is None:
            continue
        rec = {"sequence": seq, "camera": cam, "hand": hand,
               "participant_group": seq.split("-")[0],
               "d_repeat": d_primary(a, b)}
        for n, f in SECONDARY.items():
            rec["sec_" + n] = f(a, b)
        rep.append(rec)
    write_csv(RAW / "repeatability_distances.csv.gz", rep)

    # ------------------------------------------- view-to-view repeatability
    byseqhand = defaultdict(list)
    for (seq, cam, hand) in full:
        byseqhand[(seq, hand)].append(cam)
    vrep = []
    for (seq, hand), cams in sorted(byseqhand.items()):
        cams = sorted(cams)
        for ci, cj in itertools.combinations(cams, 2):
            vrep.append({
                "sequence": seq, "hand": hand, "camera_i": ci, "camera_j": cj,
                "participant_group": seq.split("-")[0],
                "d_view_repeat": d_primary(full[(seq, ci, hand)],
                                           full[(seq, cj, hand)]),
            })
    write_csv(RAW / "view_repeatability_distances.csv.gz", vrep)

    # ---------------------------------------------------- within-sequence
    within = []
    for (seq, cam, hand) in sorted(set(full)):
        if hand != "left":
            continue
        R = full.get((seq, cam, "right"))
        if R is None:
            continue
        L = full[(seq, cam, "left")]
        rec = {"sequence": seq, "camera": cam,
               "participant_group": seq.split("-")[0],
               "d_within": d_primary(L, R),
               "d_within_bonepermuted": d_primary(L, R, perm)}
        for n, f in SECONDARY.items():
            rec["sec_" + n] = f(L, R)
        within.append(rec)
    write_csv(RAW / "same_subject_distances.csv.gz", within)

    # ---------------------------------------------------- cross-sequence
    cross = []
    for cam in sorted(set(c for (_s, c, _h) in full)):
        lefts = [(s, full[(s, cam, "left")]) for (s, c2, h) in full
                 if c2 == cam and h == "left"]
        rights = [(s, full[(s, cam, "right")]) for (s, c2, h) in full
                  if c2 == cam and h == "right"]
        for (sl, L) in sorted(lefts):
            for (sr, R) in sorted(rights):
                if sl == sr:
                    continue
                amb = tuple(sorted((sl, sr))) in ambiguous
                cross.append({
                    "camera": cam, "left_sequence": sl, "right_sequence": sr,
                    "left_group": sl.split("-")[0],
                    "right_group": sr.split("-")[0],
                    "same_candidate_participant": int(amb),
                    "in_primary_cross": int(not amb),
                    "d_cross": d_primary(L, R),
                    "d_cross_bonepermuted": d_primary(L, R, perm),
                })
    write_csv(RAW / "cross_subject_distances.csv.gz", cross)

    # ---------------------------------------------------- margins + top-1
    # For each LEFT template, compare its own RIGHT against every other
    # sequence's RIGHT on the SAME camera. Reverse direction also run.
    margins, ident = [], []
    for direction in ("LEFT_TO_RIGHT", "RIGHT_TO_LEFT"):
        qh, th = ("left", "right") if direction == "LEFT_TO_RIGHT" \
            else ("right", "left")
        for cam in sorted(set(c for (_s, c, _h) in full)):
            targets = {s: full[(s, cam, th)] for (s, c2, h) in full
                       if c2 == cam and h == th}
            for (s, c2, h) in sorted(full):
                if c2 != cam or h != qh or s not in targets:
                    continue
                Q = full[(s, cam, qh)]
                own = d_primary(Q, targets[s])
                others = {t: d_primary(Q, z) for t, z in targets.items()
                          if t != s}
                # the ambiguous same-prefix partner is not a valid "other"
                prim = {t: d for t, d in others.items()
                        if tuple(sorted((s, t))) not in ambiguous}
                if not prim:
                    continue
                nearest_t = min(prim, key=prim.get)
                d_near = prim[nearest_t]
                d_med = float(np.median(list(prim.values())))
                best = min([(own, s)] + [(d, t) for t, d in prim.items()])
                margins.append({
                    "direction": direction, "camera": cam, "sequence": s,
                    "participant_group": s.split("-")[0],
                    "d_same": own, "d_nearest_other": d_near,
                    "d_median_other": d_med,
                    "m_nearest": d_near - own, "m_median": d_med - own,
                    "nearest_other_sequence": nearest_t,
                    "n_candidates": len(prim) + 1,
                })
                ident.append({
                    "direction": direction, "camera": cam, "sequence": s,
                    "predicted_sequence": best[1],
                    "correct": int(best[1] == s),
                    "n_candidates": len(prim) + 1,
                })
    write_csv(RAW / "margins.csv.gz", margins)
    write_csv(RAW / "identification.csv.gz", ident)

    # ------------------------------------ cross-sequence same-candidate-group
    cs = []
    for (sl, sr) in sorted(ambiguous):
        for cam in sorted(set(c for (_s, c, _h) in full)):
            for h1 in OTHER:
                for h2 in OTHER:
                    A = full.get((sl, cam, h1))
                    B = full.get((sr, cam, h2))
                    if A is None or B is None:
                        continue
                    cs.append({
                        "sequence_a": sl, "sequence_b": sr, "camera": cam,
                        "hand_a": h1, "hand_b": h2,
                        "comparison": ("SAME_SIDE" if h1 == h2
                                       else "OPPOSITE_SIDE"),
                        "d": d_primary(A, B),
                    })
    write_csv(RAW / "cross_sequence_same_subject.csv.gz", cs)

    print("repeat %d | view_repeat %d | within %d | cross %d (primary %d) | "
          "margins %d | ident %d | cross_seq_same_group %d"
          % (len(rep), len(vrep), len(within), len(cross),
             sum(c["in_primary_cross"] for c in cross), len(margins),
             len(ident), len(cs)))


if __name__ == "__main__":
    main()
