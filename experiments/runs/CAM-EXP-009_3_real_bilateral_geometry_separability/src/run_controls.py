"""PHASE B step 4 — negative controls and the permutation test.

C1 BONE MAPPING PERMUTATION
    right-hand bone indices permuted within each finger before the distance.
    If the distance uses real anatomical correspondence, the within-sequence
    distance must get worse and identification must fall.

C2 SEQUENCE LABEL PERMUTATION
    the right-hand sequence labels are relabelled. With 5 sequences all
    5! = 120 relabelings are enumerable, so the null is computed EXACTLY
    rather than sampled. The identity permutation is included in the null, as
    it must be for a valid permutation p-value.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (MANIFESTS, N_BONES, RAW, SUM, d_primary,  # noqa: E402
                    finger_permutation, read_csv, read_json, write_csv,
                    write_json)


def zof(r):
    return np.array([float(r["z%02d" % b]) for b in range(N_BONES)], float)


def main():
    spec = read_json(MANIFESTS / "cam_exp_0093_geometry_spec_v1.json")
    ambiguous = {tuple(sorted(p)) for p in
                 spec["identity"]["ambiguous_pairs_excluded_from_primary_cross"]}
    perm = finger_permutation()

    rows = read_csv(RAW / "geometry_templates.csv.gz")
    full = {(r["sequence"], r["camera"], r["hand"]): zof(r)
            for r in rows if r["template_type"] == "FULL"}
    cams = sorted(set(c for (_s, c, _h) in full))
    seqs = sorted(set(s for (s, _c, _h) in full))

    # ------------------------------------------------ C1 bone mapping
    ctrl = []
    for mapping, tag in ((None, "C0_CORRECT_MAPPING"),
                         (perm, "C1_BONE_MAPPING_PERMUTATION")):
        d_within, correct, total = [], 0, 0
        for cam in cams:
            targets = {s: full[(s, cam, "right")] for s in seqs
                       if (s, cam, "right") in full}
            for s in seqs:
                if (s, cam, "left") not in full or s not in targets:
                    continue
                L = full[(s, cam, "left")]
                own = d_primary(L, targets[s], mapping)
                d_within.append(own)
                cand = {t: d_primary(L, z, mapping)
                        for t, z in targets.items()
                        if t == s or tuple(sorted((s, t))) not in ambiguous}
                if len(cand) < 2:
                    continue
                total += 1
                correct += int(min(cand, key=cand.get) == s)
        ctrl.append({
            "control": tag,
            "median_d_within": float(np.nanmedian(d_within)),
            "n_within": len(d_within),
            "top1_accuracy_pct": 100.0 * correct / total if total else np.nan,
            "n_identification_trials": total,
        })
    base = ctrl[0]["median_d_within"]
    ctrl[1]["degradation_pct_vs_correct"] = (
        100.0 * (ctrl[1]["median_d_within"] - base) / base if base else np.nan)
    ctrl[1]["top1_drop_pp"] = (ctrl[0]["top1_accuracy_pct"]
                               - ctrl[1]["top1_accuracy_pct"])
    write_csv(RAW / "control_distances.csv.gz", ctrl)
    write_csv(SUM / "control_summary.csv", ctrl)

    # ------------------------------------------------ C2 label permutation
    # statistic: sequence-level median(CROSS) - median(WITHIN)
    def statistic(relabel):
        """relabel maps a left sequence -> the sequence whose RIGHT it is
        paired with as 'its own'."""
        w, x = [], []
        for cam in cams:
            for s in seqs:
                if (s, cam, "left") not in full:
                    continue
                L = full[(s, cam, "left")]
                own_seq = relabel[s]
                if (own_seq, cam, "right") not in full:
                    continue
                w.append(d_primary(L, full[(own_seq, cam, "right")]))
                for t in seqs:
                    if t == own_seq or (t, cam, "right") not in full:
                        continue
                    if tuple(sorted((own_seq, t))) in ambiguous:
                        continue
                    x.append(d_primary(L, full[(t, cam, "right")]))
        if not w or not x:
            return float("nan")
        return float(np.median(x) - np.median(w))

    identity = {s: s for s in seqs}
    obs = statistic(identity)

    null = []
    for p in itertools.permutations(seqs):
        null.append(statistic(dict(zip(seqs, p))))
    null = np.array(null, float)
    ok = np.isfinite(null)
    # one-sided: how often does a relabeling match or beat the observed
    # within-sequence advantage?
    p_value = float((null[ok] >= obs).sum() / ok.sum())

    perm_rows = [{"relabeling_index": i, "statistic": float(v)}
                 for i, v in enumerate(null)]
    write_csv(RAW / "permutation_results.csv.gz", perm_rows)
    out = {
        "test": "exact permutation over sequence relabelings",
        "n_sequences": len(seqs),
        "n_relabelings": int(ok.sum()),
        "statistic": "median(CROSS) - median(WITHIN), camera-matched",
        "observed": obs,
        "null_median": float(np.median(null[ok])),
        "null_max": float(np.max(null[ok])),
        "p_value_one_sided": p_value,
        "identity_permutation_included_in_null": True,
        "note": "with 5 sequences the smallest attainable p-value is 1/120 = "
                "0.0083, and the permutation unit is the SEQUENCE, of which "
                "there are only 5. The test is therefore weak by construction "
                "regardless of the outcome.",
    }
    write_json(SUM / "permutation_summary.json", out)

    print("C0 median d_within %.4f  top1 %.1f%%"
          % (ctrl[0]["median_d_within"], ctrl[0]["top1_accuracy_pct"]))
    print("C1 median d_within %.4f  top1 %.1f%%  degradation %+.1f%%  "
          "top1 drop %+.1f pp"
          % (ctrl[1]["median_d_within"], ctrl[1]["top1_accuracy_pct"],
             ctrl[1]["degradation_pct_vs_correct"], ctrl[1]["top1_drop_pp"]))
    print("C2 observed %+.5f  null median %+.5f  p = %.4f"
          % (obs, out["null_median"], p_value))


if __name__ == "__main__":
    main()
