"""PHASE B step 2 — geometry templates.

A template is the bone-wise MEDIAN of z over a unit's eligible frames. Three
template families are built, all from the same frame vectors:

  FULL   one per (sequence, camera, hand)          -> same/cross distances
  HALF   one per (sequence, camera, hand, A|B)     -> repeatability

LEFT and RIGHT are not required to appear in the same frame: hand anatomy is
treated as a sequence-level property, so each side's template is built from
whatever eligible frames that side has.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (MIN_FRAMES_PER_HALF, MIN_FRAMES_PER_SIDE,  # noqa: E402
                    N_BONES, RAW, write_csv)


def zmat(rows):
    return np.array([[float(r["z%02d" % b]) for b in range(N_BONES)]
                     for r in rows], float)


def main():
    from common import read_csv
    rows = read_csv(RAW / "frame_bone_vectors.csv.gz")
    print("frame vectors:", len(rows))

    full = defaultdict(list)
    half = defaultdict(list)
    for r in rows:
        k = (r["sequence"], r["camera"], r["hand"])
        full[k].append(r)
        half[k + (r["half"],)].append(r)

    out = []
    for (seq, cam, hand), rs in sorted(full.items()):
        if len(rs) < MIN_FRAMES_PER_SIDE:
            continue
        z = np.median(zmat(rs), axis=0)
        rec = {"template_type": "FULL", "sequence": seq, "camera": cam,
               "hand": hand, "half": "", "n_frames": len(rs),
               "participant_group": seq.split("-")[0],
               "median_total_length_m": float(np.median(
                   [float(x["total_length_m"]) for x in rs]))}
        for b in range(N_BONES):
            rec["z%02d" % b] = float(z[b])
        out.append(rec)

    for (seq, cam, hand, h), rs in sorted(half.items()):
        if len(rs) < MIN_FRAMES_PER_HALF:
            continue
        z = np.median(zmat(rs), axis=0)
        rec = {"template_type": "HALF", "sequence": seq, "camera": cam,
               "hand": hand, "half": h, "n_frames": len(rs),
               "participant_group": seq.split("-")[0],
               "median_total_length_m": float(np.median(
                   [float(x["total_length_m"]) for x in rs]))}
        for b in range(N_BONES):
            rec["z%02d" % b] = float(z[b])
        out.append(rec)

    write_csv(RAW / "geometry_templates.csv.gz", out)
    nf = sum(1 for r in out if r["template_type"] == "FULL")
    nh = sum(1 for r in out if r["template_type"] == "HALF")
    print("FULL templates %d, HALF templates %d" % (nf, nh))


if __name__ == "__main__":
    main()
