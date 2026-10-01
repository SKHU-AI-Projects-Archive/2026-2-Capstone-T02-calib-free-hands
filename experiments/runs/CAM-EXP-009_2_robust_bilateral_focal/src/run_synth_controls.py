"""Negative controls C0-C4 for the selected method.

A cue that identifies the focal must LOSE that ability when the bilateral
correspondence it relies on is destroyed. If a control does as well as the
correct pairing, the apparent signal is not bilateral structure.

  C0  correct                 the reference condition
  C1  wrong bone mapping      right-hand bones permuted within each finger
  C2  subject swap            right hand from a different synthetic subject
  C3  temporal shuffle        right frames from an unrelated sequence
  C4  fixed-length invariant  bone lengths held fixed at the candidate-
                              independent value; the CAM-EXP-009 naive
                              formulation, which must be exactly flat

C4 is an ORACLE DIAGNOSTIC: it is handed the true bone lengths, so it is not a
method that could ever be run on real data. Its only job is to demonstrate the
invariance - a score built from lengths that do not depend on the candidate
focal cannot vary with the candidate focal. No other control receives any true
quantity.

C1 is a within-finger permutation rather than a global one: a global shuffle
would also destroy the rough size ordering of the bones, so its degradation
could be explained without any bilateral correspondence at all.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c92_common import (COMBINED_MODERATE, MANIFESTS, RAW,  # noqa: E402
                        read_csv, read_json, stable_seed, write_csv)
import build_candidate_shape_profiles as bcsp  # noqa: E402
import generate_stress_synthetic as gss  # noqa: E402
import robust_bilateral_scores as rbs  # noqa: E402
from run_synth_dev import F_NOMINAL_SYNTH, pick  # noqa: E402

CONTROLS = ["C0_CORRECT", "C1_WRONG_BONE_MAPPING", "C2_SUBJECT_SWAP",
            "C3_TEMPORAL_SHUFFLE", "C4_FIXED_LENGTH_INVARIANT"]
N_SUBJ = 8


def finger_permutation(rng):
    """Permute bones WITHIN each finger. Preserves the coarse size ordering."""
    idx = np.arange(20)
    for f in range(5):
        blk = idx[4 * f:4 * f + 4].copy()
        while True:
            q = rng.permutation(blk)
            if not np.array_equal(q, blk):
                break
        idx[4 * f:4 * f + 4] = q
    return idx


def eval_curve(pL, pR, qs, method, mapping=None):
    s = np.array([rbs.score(method, pL[i], pR[i], mapping)
                  for i in range(len(qs))], float)
    return pick(qs, s) + (s,)


def main():
    sel = read_json(MANIFESTS / "cam_exp_0092_selected_method_v1.json")
    method = sel["SELECTED_METHOD"]
    recs, t0 = [], time.time()
    for s in range(N_SUBJ):
        rng = np.random.default_rng(stable_seed("CAM0092_CTRL|S%03d" % s))
        seq = gss.make_sequence(rng, n_frames=12, **COMBINED_MODERATE)
        alt = gss.make_sequence(
            np.random.default_rng(stable_seed("CAM0092_CTRL_ALT|S%03d" % s)),
            n_frames=12, **COMBINED_MODERATE)
        prof = bcsp.build(seq, F_NOMINAL_SYNTH)
        prof_alt = bcsp.build(alt, F_NOMINAL_SYNTH)
        qs = prof["q"]
        f_true = seq["f_true"]
        pL, pR = prof["p_left_eval"], prof["p_right_eval"]

        # C3: the right side refitted from frames of an unrelated sequence.
        shuf = dict(seq)
        shuf_frames = list(alt["right"])
        pR_shuf = bcsp.build({"K": seq["K"], "dist": seq["dist"],
                              "left": seq["left"], "right": shuf_frames},
                             F_NOMINAL_SYNTH)["p_right_eval"]

        # C4: candidate-independent fixed lengths, normalised. The score cannot
        # depend on q, so its curve must be exactly flat.
        lL = seq["lL_true"] / seq["lL_true"].sum()
        lR = seq["lR_true"] / seq["lR_true"].sum()
        flat = np.array([rbs.score(method, lL, lR)] * len(qs), float)

        for c in CONTROLS:
            if c == "C0_CORRECT":
                f_hat, b, curve = eval_curve(pL, pR, qs, method)
            elif c == "C1_WRONG_BONE_MAPPING":
                f_hat, b, curve = eval_curve(
                    pL, pR, qs, method, finger_permutation(rng))
            elif c == "C2_SUBJECT_SWAP":
                f_hat, b, curve = eval_curve(
                    pL, prof_alt["p_right_eval"], qs, method)
            elif c == "C3_TEMPORAL_SHUFFLE":
                f_hat, b, curve = eval_curve(pL, pR_shuf, qs, method)
            else:
                curve = flat
                f_hat, b = pick(qs, curve)
            rng_span = (float(np.nanmax(curve) - np.nanmin(curve))
                        if np.isfinite(curve).any() else float("nan"))
            recs.append({
                "subject": s, "control": c, "method": method,
                "f_true": f_true, "f_hat": f_hat,
                "err_pct": (abs(f_hat - f_true) / f_true * 100.0
                            if np.isfinite(f_hat) else float("nan")),
                "boundary": b, "curve_range": rng_span,
            })
        print("subject %d  %.1f min" % (s, (time.time() - t0) / 60), flush=True)
    write_csv(RAW / "control_trials.csv.gz", recs)
    print("wrote", len(recs), "rows")


if __name__ == "__main__":
    main()
