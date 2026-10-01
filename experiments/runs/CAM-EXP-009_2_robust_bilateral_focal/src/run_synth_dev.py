"""DEV run: every method M0-M6 on the 7-cell DEV_CHALLENGE_SET.

The DEV set exists for ONE purpose - to select a single method. Its numbers
are never reported as evidence that the cue works; that is what the disjoint
TEST set is for.

The candidate grid is anchored on F_NOMINAL_SYNTH, an arbitrary constant that
is NOT the true focal. The true focal therefore sits at q = 0.9, away from the
grid centre, so a method cannot score well merely by preferring the middle of
the search range.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c92_common import (MANIFESTS, RAW, read_csv, write_csv)  # noqa: E402
import build_candidate_shape_profiles as bcsp  # noqa: E402
import generate_stress_synthetic as gss  # noqa: E402
import robust_bilateral_scores as rbs  # noqa: E402

F_NOMINAL_SYNTH = 1000.0          # arbitrary anchor; the true focal is 900


def curve(profiles, method, tag):
    pL = profiles["p_left_%s" % tag]
    pR = profiles["p_right_%s" % tag]
    return np.array([rbs.score(method, pL[i], pR[i])
                     for i in range(len(profiles["q"]))], float)


def pick(q, s, f_nominal=F_NOMINAL_SYNTH):
    """argmin of a score curve -> (focal, boundary flag)."""
    ok = np.isfinite(s)
    if ok.sum() < 3:
        return float("nan"), 1
    idx = np.flatnonzero(ok)
    j = idx[int(np.argmin(s[idx]))]
    boundary = int(j == idx[0] or j == idx[-1])
    return float(q[j] * f_nominal), boundary


def run_trial(row, methods, n_frames=12):
    kw = dict(dist_over_diam=float(row["dist_over_diam"]),
              asym_kind=row["asym_kind"], asym=float(row["asym"]),
              noise_px=float(row["noise_px"]),
              artic_deg=float(row["artic_deg"]),
              missing=float(row["missing"]),
              visibility=float(row["visibility"]))
    rng = np.random.default_rng(int(row["seed"]))
    seq = gss.make_sequence(rng, n_frames=n_frames, **kw)
    prof = bcsp.build(seq, F_NOMINAL_SYNTH)
    f_true = seq["f_true"]
    out = []
    for m in methods:
        rec = {"trial_id": row["trial_id"], "cell": row["cell"],
               "subject": row["subject"], "method": m, "f_true": f_true}
        for tag in ("fit", "eval"):
            s = curve(prof, m, tag)
            f_hat, b = pick(prof["q"], s)
            rec["f_hat_%s" % tag] = f_hat
            rec["err_pct_%s" % tag] = (abs(f_hat - f_true) / f_true * 100.0
                                       if np.isfinite(f_hat) else float("nan"))
            rec["boundary_%s" % tag] = b
        out.append(rec)
    return out


def main():
    rows = read_csv(MANIFESTS / "cam_exp_0092_synthetic_dev_trials_v1.csv.gz")
    recs, t0 = [], time.time()
    for i, r in enumerate(rows, 1):
        recs.extend(run_trial(r, rbs.METHODS))
        if i % 5 == 0 or i == len(rows):
            print("%3d/%d  %.1f min" % (i, len(rows), (time.time() - t0) / 60),
                  flush=True)
    write_csv(RAW / "dev_trials.csv.gz", recs)
    print("wrote", len(recs), "rows")


if __name__ == "__main__":
    main()
