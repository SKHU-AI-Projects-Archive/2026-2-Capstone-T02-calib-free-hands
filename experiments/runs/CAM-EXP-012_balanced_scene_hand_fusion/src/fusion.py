"""Loss-scale normalisation and fusion. The core of CAM-EXP-012.

Both curves are CENTRED at q = 1 so a constant offset cannot matter:

    S(q) = L_scene(q) - L_scene(1)
    H(q) = L_hand(f_scene * q) - L_hand(f_scene)

No min-max or range normalisation is applied — that would make lambda mean
something different in every video.

The scale factor alpha is measured at a FIXED log-focal offset of +/- 0.05
(about +/- 5 %), deliberately NOT at an adjacent grid point. An adjacent-point
measurement would make alpha depend on grid resolution, because the scene loss
is quadratic near q = 1 while the hand loss can be locally linear.

    D_scene_video = ( |S(q-)| + |S(q+)| ) / 2
    D_hand_video  = ( |H(q-)| + |H(q+)| ) / 2
    alpha         = median_train(D_scene_video) / median_train(D_hand_video)

alpha never sees a reference focal, a focal error, or whether a correction
moves toward the truth. It only equalises two numerical magnitudes.

Fusion:

    L_total(q) = S(q) + lambda * alpha * H(q)

The SAME alpha (from the correct-hand curves) and the SAME selected lambda are
used for the shuffled and wrong-bone controls, so the only thing that changes
between conditions is the hand curve itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c012_common import (ALPHA_DEGENERATE_THRESHOLD, EPS, Q_MINUS,  # noqa
                         Q_PLUS, med, q_grid, scene_loss)


def centred_scene(q, sigma):
    """S(q), analytic. S(1) = 0 exactly."""
    return scene_loss(q, sigma) - scene_loss(1.0, sigma)


def centred_hand(q, f_scene, hand_f, hand_s):
    """H(q) on the candidate ratio axis, centred at q = 1.

    Interpolated from the GLOBAL absolute grid, which every video shares.
    Returns NaN outside the grid's support rather than extrapolating.
    """
    ok = np.isfinite(hand_s)
    if ok.sum() < 5:
        return np.full(np.shape(q), np.nan)
    f = np.asarray(f_scene, float) * np.asarray(q, float)
    inside = (f >= hand_f[ok].min()) & (f <= hand_f[ok].max())
    out = np.full(np.shape(f), np.nan)
    out[inside] = np.interp(f[inside], hand_f[ok], hand_s[ok])
    base = (np.interp(f_scene, hand_f[ok], hand_s[ok])
            if hand_f[ok].min() <= f_scene <= hand_f[ok].max() else np.nan)
    return out - base


def local_magnitudes(f_scene, sigma, hand_f, hand_s):
    """(D_scene, D_hand) at the fixed +/- 5 % log-focal offset."""
    qs = np.array([Q_MINUS, Q_PLUS])
    S = centred_scene(qs, sigma)
    Hq = centred_hand(qs, f_scene, hand_f, hand_s)
    d_scene = float(np.mean(np.abs(S)))
    d_hand = float(np.mean(np.abs(Hq))) if np.isfinite(Hq).all() else np.nan
    return d_scene, d_hand


def alpha_from_train(pairs):
    """pairs: [(D_scene, D_hand)] over TRAIN videos -> (alpha, D_s, D_h)."""
    ds = med([p[0] for p in pairs])
    dh = med([p[1] for p in pairs])
    if not np.isfinite(dh) or dh <= ALPHA_DEGENERATE_THRESHOLD:
        return float("nan"), ds, dh
    return float(ds / dh), ds, dh


def fuse(f_scene, sigma, hand_f, hand_s, alpha, lam):
    """-> (f_pred, boundary_flag). lam = 0 reproduces f_scene exactly."""
    qs = q_grid()
    S = centred_scene(qs, sigma)
    if lam == 0 or not np.isfinite(alpha):
        j = int(np.argmin(S))
        return float(f_scene * qs[j]), int(j in (0, len(qs) - 1))
    H = centred_hand(qs, f_scene, hand_f, hand_s)
    tot = np.where(np.isfinite(H), S + lam * alpha * H, np.inf)
    if not np.isfinite(tot).any():
        return float("nan"), 1
    j = int(np.argmin(tot))
    return float(f_scene * qs[j]), int(j in (0, len(qs) - 1))
