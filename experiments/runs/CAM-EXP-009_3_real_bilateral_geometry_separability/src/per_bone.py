"""PHASE B step 6 — per-bone decomposition. DESCRIPTIVE ONLY.

Which bones are stable across a repeat of the same hand, and which separate
sequences? This is reported to inform a future design. It is NEVER used to
re-select bones and recompute the primary metric of this run: the primary
distance stays over all 20 frozen bones.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BONES, EPS, MANIFESTS, N_BONES, RAW, SUM,  # noqa: E402
                    read_csv, read_json, write_csv)

NAMES = ["thumb_mcp", "thumb_pip", "thumb_dip", "thumb_tip",
         "index_mcp", "index_pip", "index_dip", "index_tip",
         "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
         "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
         "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"]


def zof(r):
    return np.array([float(r["z%02d" % b]) for b in range(N_BONES)], float)


def main():
    spec = read_json(MANIFESTS / "cam_exp_0093_geometry_spec_v1.json")
    ambiguous = {tuple(sorted(p)) for p in
                 spec["identity"]["ambiguous_pairs_excluded_from_primary_cross"]}
    rows = read_csv(RAW / "geometry_templates.csv.gz")
    full, half = {}, {}
    for r in rows:
        k = (r["sequence"], r["camera"], r["hand"])
        (full if r["template_type"] == "FULL" else half)[
            k if r["template_type"] == "FULL" else k + (r["half"],)] = zof(r)

    rep = [[] for _ in range(N_BONES)]
    wit = [[] for _ in range(N_BONES)]
    cro = [[] for _ in range(N_BONES)]

    for (seq, cam, hand) in full:
        a, b = half.get((seq, cam, hand, "A")), half.get((seq, cam, hand, "B"))
        if a is not None and b is not None:
            d = np.abs(a - b)
            for i in range(N_BONES):
                rep[i].append(d[i])

    cams = sorted(set(c for (_s, c, _h) in full))
    seqs = sorted(set(s for (s, _c, _h) in full))
    for cam in cams:
        for s in seqs:
            L = full.get((s, cam, "left"))
            R = full.get((s, cam, "right"))
            if L is not None and R is not None:
                d = np.abs(L - R)
                for i in range(N_BONES):
                    wit[i].append(d[i])
            if L is None:
                continue
            for t in seqs:
                if t == s or tuple(sorted((s, t))) in ambiguous:
                    continue
                R2 = full.get((t, cam, "right"))
                if R2 is None:
                    continue
                d = np.abs(L - R2)
                for i in range(N_BONES):
                    cro[i].append(d[i])

    out = []
    for i in range(N_BONES):
        vr = float(np.median(rep[i])) if rep[i] else np.nan
        vw = float(np.median(wit[i])) if wit[i] else np.nan
        vc = float(np.median(cro[i])) if cro[i] else np.nan
        out.append({
            "bone_index": i, "bone": NAMES[i],
            "joints": "%d-%d" % BONES[i],
            "V_repeat": vr, "V_within": vw, "V_cross": vc,
            "separation_within_to_cross": vc - vw,
            "reliability_ratio_R_b": vc / max(vr, EPS),
            "within_over_cross": vw / vc if vc else np.nan,
        })
    write_csv(SUM / "per_bone_summary.csv", out)

    print("%-12s %8s %8s %8s %8s" % ("bone", "repeat", "within", "cross", "R_b"))
    for r in out:
        print("%-12s %8.4f %8.4f %8.4f %8.2f"
              % (r["bone"], r["V_repeat"], r["V_within"], r["V_cross"],
                 r["reliability_ratio_R_b"]))
    print("\nDESCRIPTIVE ONLY - the primary distance keeps all 20 bones.")


if __name__ == "__main__":
    main()
