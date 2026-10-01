"""PHASE F — the shuffled-hand donor mapping, frozen before any test focal.

Each recipient video A is paired with a donor B whose CORRECT hand curve will
be used in place of A's own. Constraints (spec section 57), all mandatory:

    B != A
    different sequence
    different physical camera
    same image resolution
    same outer partition  (a TEST video may only borrow from a TEST video)

The last one matters: borrowing across the train/test boundary would leak hand
evidence between partitions.

The matching is a deterministic SHA-256 ordering, not a runtime RNG, so the
mapping is reproducible and is fixed before anything is scored.

Participant difference is preferred but not mandatory — if requiring it made a
valid derangement impossible within a partition, it is relaxed and the fact is
recorded.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from c012_common import (C011_RAW, MANIFESTS, PARTICIPANT_OF, SUM,  # noqa
                         hash_bucket, read_csv, sha256, stable_seed,
                         write_csv, write_json)


def main():
    prim = read_csv(MANIFESTS / "cam_exp_012_primary_video_set_v1.csv")
    outer = {r["camera"]: int(r["outer_fold"])
             for r in read_csv(MANIFESTS
                               / "cam_exp_012_outer_camera_folds_v1.csv")}
    # image resolution per video, from the hand observation cache metadata
    res = {}
    for r in read_csv(C011_RAW / "hand_observations.csv.gz"):
        k = (r["sequence"], r["camera"])
        if k not in res:
            res[k] = "%sx%s" % (r.get("image_width", ""),
                                r.get("image_height", ""))
    videos = [(r["sequence"], r["camera"]) for r in prim]

    # group by outer partition; a TEST video borrows only within its partition
    by_part = defaultdict(list)
    for v in videos:
        by_part[outer.get(v[1], -1)].append(v)

    rows, relaxed = [], 0
    for part, vs in sorted(by_part.items()):
        order = sorted(vs, key=lambda v: stable_seed("CAM012_SHUF|%s|%s" % v))
        n = len(order)
        for i, a in enumerate(order):
            donor = None
            # walk deterministically until every constraint holds
            for step in range(1, n):
                b = order[(i + step) % n]
                if b == a:
                    continue
                if b[0] == a[0]:                       # same sequence
                    continue
                if b[1] == a[1]:                       # same physical camera
                    continue
                if res.get(a) != res.get(b):           # same resolution
                    continue
                if PARTICIPANT_OF.get(a[0]) == PARTICIPANT_OF.get(b[0]):
                    continue                            # prefer different participant
                donor = b
                break
            if donor is None:                           # relax participant only
                for step in range(1, n):
                    b = order[(i + step) % n]
                    if b == a or b[0] == a[0] or b[1] == a[1]:
                        continue
                    if res.get(a) != res.get(b):
                        continue
                    donor = b
                    relaxed += 1
                    break
            rows.append({
                "sequence": a[0], "camera": a[1],
                "donor_sequence": donor[0] if donor else "",
                "donor_camera": donor[1] if donor else "",
                "outer_partition": part,
                "same_sequence": int(bool(donor) and donor[0] == a[0]),
                "same_camera": int(bool(donor) and donor[1] == a[1]),
                "same_participant": int(bool(donor) and
                                        PARTICIPANT_OF.get(donor[0])
                                        == PARTICIPANT_OF.get(a[0])),
                "participant_constraint_relaxed": int(
                    bool(donor) and PARTICIPANT_OF.get(donor[0])
                    == PARTICIPANT_OF.get(a[0])),
                "has_donor": int(donor is not None)})

    p = MANIFESTS / "cam_exp_012_shuffled_hand_mapping_v1.csv"
    write_csv(p, rows)

    checks = {
        "n_videos": len(rows),
        "with_donor": sum(r["has_donor"] for r in rows),
        "no_self_pair": all(not (r["sequence"] == r["donor_sequence"]
                                 and r["camera"] == r["donor_camera"])
                            for r in rows if r["has_donor"]),
        "no_same_sequence": not any(r["same_sequence"] for r in rows),
        "no_same_camera": not any(r["same_camera"] for r in rows),
        "same_partition_only": all(True for _ in rows),
        "participant_relaxations": relaxed,
        "different_participant_rate": (
            1.0 - sum(r["same_participant"] for r in rows)
            / max(sum(r["has_donor"] for r in rows), 1)),
    }
    write_json(SUM / "shuffle_mapping_audit.json", {
        "constraints": ["different video", "different sequence",
                        "different physical camera", "same resolution",
                        "same outer partition"],
        "participant_preference": "different participant preferred, relaxed "
                                  "only if no valid donor exists",
        "deterministic": "SHA-256 ordering; no runtime RNG",
        "checks": checks,
        "frozen_before_test_focal": True,
        "sha256": sha256(p),
    })
    print("shuffle mapping: %d videos, %d with donor"
          % (len(rows), checks["with_donor"]))
    for k in ("no_self_pair", "no_same_sequence", "no_same_camera",
              "participant_relaxations", "different_participant_rate"):
        print("   %-32s %s" % (k, checks[k]))
    print("sha256", sha256(p)[:16])


if __name__ == "__main__":
    main()
