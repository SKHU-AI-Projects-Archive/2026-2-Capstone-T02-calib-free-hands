"""Candidate-conditioned bone-proportion profiles p_L(q), p_R(q).

The bone fit is the only expensive part of this experiment and it is shared:
for each (trial, candidate focal q) the left and right proportions are fitted
ONCE, and every score in the M0-M6 family is then evaluated on that same pair.
No method ever sees an input another method did not see.

Two independent profiles are built per trial:

  FIT   frames (even frame index)  - used to choose the focal
  EVAL  frames (odd frame index)   - an independent set of frames of the SAME
                                     subject, refitted from scratch

The split is by frame parity, fixed before any result was produced. It is not
a held-out generalisation test in the usual sense: CAM-EXP-009 already showed
that a hand fitted at the wrong focal is *consistently* wrong, so other frames
of the same hand do not expose that bias. Its purpose here is narrower - to
show the chosen focal is not an artefact of the particular frames fitted.

The true focal, the true pose and the true bone vector are never passed in.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c92_common import MIN_FIT_FRAMES, q_grid  # noqa: E402
from corrected_bone_fitter import fit_bones  # noqa: E402  (CAM-009.1)

PIPELINE_BASELINE_Q = 1.0


def split_frames(frames):
    """Deterministic parity split. Never depends on fit quality."""
    fit = [f for i, f in enumerate(frames) if i % 2 == 0]
    ev = [f for i, f in enumerate(frames) if i % 2 == 1]
    return fit, ev


def _profile(frames, dist, cx, cy, qs, f_nominal, warm=True):
    """Fit p(q) across the candidate grid for one side, one frame set."""
    out = np.full((len(qs), 20), np.nan)
    px = np.full(len(qs), np.nan)
    if len(frames) < MIN_FIT_FRAMES:
        return out, px
    prev = None
    for i, q in enumerate(qs):
        f = f_nominal * q
        K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
        l, ideal, _raw = fit_bones(frames, K, dist, dof="D20",
                                   l_init=prev if warm else None)
        if l is not None and np.isfinite(l).all():
            out[i] = l
            px[i] = ideal
            prev = l
    return out, px


def build(seq, f_nominal, qs=None, warm=True):
    """One trial -> FIT and EVAL profiles for both sides.

    `f_nominal` is the focal the grid is expressed relative to. It is supplied
    by the caller as the PIPELINE BASELINE, never as the true focal.
    """
    qs = q_grid() if qs is None else np.asarray(qs, float)
    cx, cy = seq["K"][0, 2], seq["K"][1, 2]
    dist = seq["dist"]
    res = {"q": qs}
    for side in ("left", "right"):
        fit_f, ev_f = split_frames(seq[side])
        for tag, fr in (("fit", fit_f), ("eval", ev_f)):
            p, px = _profile(fr, dist, cx, cy, qs, f_nominal, warm)
            res[f"p_{side}_{tag}"] = p
            res[f"px_{side}_{tag}"] = px
            res[f"n_{side}_{tag}"] = len(fr)
    return res
