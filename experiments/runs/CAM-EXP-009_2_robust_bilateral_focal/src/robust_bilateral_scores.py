"""The frozen robust bilateral score family M0-M6.

All of them consume the SAME candidate-conditioned fitted bone proportions
p_L(f), p_R(f). The expensive bone fitting therefore happens once per
(trial, candidate focal) and every score is computed from it. No score gets a
different input from any other.

Both vectors are already normalised to sum 1, so overall hand size is removed
and a person with larger hands is not penalised. A separate global LEFT/RIGHT
scale parameter r_LR would be REDUNDANT for exactly that reason - it is
audited in `audit_r_LR_redundancy()` and not used.

This family is frozen before the DEV run. Nothing may be added after seeing
results.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c92_common import EPS  # noqa: E402

N_BONES = 20
FINGERS = [list(range(4 * f, 4 * f + 4)) for f in range(5)]
TRIM_FRACTION = 0.20          # drop the 4 worst-matching of 20 bones
HUBER_TAUS = (0.01, 0.02, 0.05)


def log_diff(pL, pR, mapping=None):
    """Relative per-bone mismatch in log-proportion space."""
    a = np.asarray(pL, float)
    b = np.asarray(pR, float)
    if mapping is not None:
        b = b[np.asarray(mapping, int)]
    return np.log(a + EPS) - np.log(b + EPS)


def finger_internal(p):
    """Per-finger internal segment proportions: each chain sums to 1.

    Removes a per-finger length difference between the sides, so only the
    *internal* segmentation of each finger is compared.
    """
    p = np.asarray(p, float)
    out = np.zeros_like(p)
    for f in FINGERS:
        s = p[f].sum()
        out[f] = p[f] / s if s > 0 else 0.0
    return out


def _huber(z, tau):
    a = np.abs(np.asarray(z, float)) / tau
    return np.where(a <= 1.0, 0.5 * a ** 2, a - 0.5)


def score(name, pL, pR, mapping=None):
    """One robust bilateral score. Lower is better agreement."""
    if pL is None or pR is None:
        return float("nan")
    pL = np.asarray(pL, float)
    pR = np.asarray(pR, float)
    if not (np.isfinite(pL).all() and np.isfinite(pR).all()):
        return float("nan")

    if name == "M0_BASELINE_RAW_L1":
        b = pR if mapping is None else pR[np.asarray(mapping, int)]
        return float(np.mean(np.abs(pL - b)))

    d = log_diff(pL, pR, mapping)

    if name == "M1_LOG_L1":
        return float(np.mean(np.abs(d)))
    if name == "M2_LOG_MEDIAN":
        return float(np.median(np.abs(d)))
    if name == "M3_LOG_TRIMMED":
        a = np.sort(np.abs(d))
        keep = int(round(len(a) * (1.0 - TRIM_FRACTION)))
        return float(np.mean(a[:max(keep, 1)]))
    if name.startswith("M4_HUBER_"):
        tau = {"M4_HUBER_001": 0.01, "M4_HUBER_002": 0.02,
               "M4_HUBER_005": 0.05}[name]
        return float(np.mean(_huber(d, tau)))
    if name == "M5_FINGER_INTERNAL_RATIO":
        rL, rR = finger_internal(pL), finger_internal(pR)
        dd = log_diff(rL, rR, mapping)
        return float(np.median(np.abs(dd)))
    if name == "M6_FINGER_HUBER":
        rL, rR = finger_internal(pL), finger_internal(pR)
        dd = log_diff(rL, rR, mapping)
        return float(np.mean(_huber(dd, 0.02)))
    raise ValueError(name)


METHODS = [
    "M0_BASELINE_RAW_L1",
    "M1_LOG_L1",
    "M2_LOG_MEDIAN",
    "M3_LOG_TRIMMED",
    "M4_HUBER_001",
    "M4_HUBER_002",
    "M4_HUBER_005",
    "M5_FINGER_INTERNAL_RATIO",
    "M6_FINGER_HUBER",
]

DEFINITIONS = [
    {"method": "M0_BASELINE_RAW_L1", "robust": 0,
     "formula": "mean_b |p_L,b - p_R,b|",
     "note": "closest to CAM-EXP-009.1; the comparison baseline, not a robust "
             "method"},
    {"method": "M1_LOG_L1", "robust": 0,
     "formula": "mean_b |log(p_L,b+eps) - log(p_R,b+eps)|",
     "note": "relative rather than absolute mismatch; large and small bones "
             "made comparable"},
    {"method": "M2_LOG_MEDIAN", "robust": 1,
     "formula": "median_b |d_b|",
     "note": "a few strongly asymmetric bones cannot dominate"},
    {"method": "M3_LOG_TRIMMED", "robust": 1,
     "formula": "mean of the smallest 80 % of |d_b| (4 of 20 dropped)",
     "note": "trim fraction frozen at 0.20"},
    {"method": "M4_HUBER_001", "robust": 1, "formula": "mean huber(d, 0.01)",
     "note": "quadratic below tau, linear above"},
    {"method": "M4_HUBER_002", "robust": 1, "formula": "mean huber(d, 0.02)",
     "note": ""},
    {"method": "M4_HUBER_005", "robust": 1, "formula": "mean huber(d, 0.05)",
     "note": ""},
    {"method": "M5_FINGER_INTERNAL_RATIO", "robust": 1,
     "formula": "median_b |log r_L - log r_R| with r normalised inside each "
                "finger",
     "note": "tolerant to a per-finger length difference between the sides"},
    {"method": "M6_FINGER_HUBER", "robust": 1,
     "formula": "mean huber(finger-internal d, 0.02)", "note": ""},
]


def audit_r_LR_redundancy(n_trials=200, seed=12345):
    """Is a global LEFT/RIGHT scale parameter redundant?

    Both proportion vectors are constrained to sum to 1. Multiplying one side
    by a scalar r and renormalising returns the identical vector, so r cannot
    change any score in this family. Verified numerically rather than asserted.
    """
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_trials):
        pL = rng.dirichlet(np.ones(N_BONES))
        pR = rng.dirichlet(np.ones(N_BONES))
        r = float(rng.uniform(0.5, 2.0))
        scaled = (pR * r) / (pR * r).sum()
        for m in METHODS:
            a, b = score(m, pL, pR), score(m, pL, scaled)
            if np.isfinite(a) and np.isfinite(b):
                worst = max(worst, abs(a - b))
    return {
        "max_abs_score_change_under_global_right_scale": float(worst),
        "redundant": bool(worst < 1e-9),
        "reason": "p_L and p_R are each normalised to sum 1, so a global "
                  "LEFT/RIGHT scale ratio is absorbed by the normalisation "
                  "and cannot change any score. A separate r_LR nuisance "
                  "parameter is therefore NOT introduced.",
    }
