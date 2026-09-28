"""Participant-block permutation, exact over 4! = 24 relabelings.

p41's two sessions always move together as ONE block, which is the correction
over CAM-EXP-009.3's sequence-level 5! = 120 permutation.

The floor is 1/24 = 0.0417 and there are 4 participant units, so this is coarse
and SUPPORTING ONLY. No conclusion is placed on it.
"""
from __future__ import annotations

import itertools
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (MANIFESTS, PARTICIPANT_OF, PARTICIPANTS,  # noqa: E402
                    RAW, SOURCE_ARTIFACTS, SUM, d_primary, med, read_csv,
                    write_csv, write_json, zof)


def main():
    rows = read_csv(SOURCE_ARTIFACTS["geometry_templates.csv.gz"])
    full = {(r["sequence"], r["camera"], r["hand"]): zof(r)
            for r in rows if r["template_type"] == "FULL"}
    cams = sorted(set(c for (_s, c, _h) in full))
    seqs = sorted(set(s for (s, _c, _h) in full))

    def statistic(relabel):
        """relabel: participant -> participant whose RIGHT hands are treated
        as 'own'. p41's two sessions inherit one block label together."""
        w, x = [], []
        for cam in cams:
            for sl in seqs:
                L = full.get((sl, cam, "left"))
                if L is None:
                    continue
                own_p = relabel[PARTICIPANT_OF[sl]]
                for sr in seqs:
                    R = full.get((sr, cam, "right"))
                    if R is None:
                        continue
                    d = d_primary(L, R)
                    if PARTICIPANT_OF[sr] == own_p:
                        w.append(d)
                    else:
                        x.append(d)
        if not w or not x:
            return float("nan")
        return float(med(x) - med(w))

    identity = {p: p for p in PARTICIPANTS}
    obs = statistic(identity)

    null, recs = [], []
    for i, perm in enumerate(itertools.permutations(PARTICIPANTS)):
        rel = dict(zip(PARTICIPANTS, perm))
        v = statistic(rel)
        null.append(v)
        recs.append({"index": i, "relabeling": ";".join(
            "%s->%s" % (k, rel[k]) for k in PARTICIPANTS),
            "is_identity": int(all(rel[k] == k for k in PARTICIPANTS)),
            "statistic": v})
    write_csv(RAW / "participant_permutation_results.csv.gz", recs)

    null = np.array(null, float)
    ok = np.isfinite(null)
    p_value = float((null[ok] >= obs).sum() / ok.sum())

    out = {
        "test": "exact permutation over PARTICIPANT blocks",
        "n_participant_blocks": len(PARTICIPANTS),
        "n_permutations": int(ok.sum()),
        "p41_sessions_move_as_one_block": True,
        "statistic": "median(different-participant) - median(same-participant),"
                     " camera-matched",
        "observed": obs,
        "null_median": float(np.median(null[ok])),
        "null_max": float(np.max(null[ok])),
        "p_value_one_sided": p_value,
        "identity_permutation_included_in_null": True,
        "smallest_attainable_p": 1.0 / int(ok.sum()),
        "interpretation": "SUPPORTING ONLY. With 4 participant blocks the "
                          "floor is 1/24 = 0.0417 and the test is coarse. No "
                          "conclusion in this reanalysis rests on it.",
    }
    write_json(SUM / "participant_permutation_summary.json", out)
    write_json(MANIFESTS / "cam_exp_00931_permutation_spec_v1.json", {
        "blocks": PARTICIPANTS, "n_permutations": 24,
        "p41_sessions_in_same_block": True,
        "created_after_cam0093_results": True,
        "role": "supporting only",
    })

    print("observed      %+.6f" % obs)
    print("null median   %+.6f" % out["null_median"])
    print("null max      %+.6f" % out["null_max"])
    print("exact p       %.4f  (floor %.4f, %d permutations)"
          % (p_value, out["smallest_attainable_p"], int(ok.sum())))


if __name__ == "__main__":
    main()
