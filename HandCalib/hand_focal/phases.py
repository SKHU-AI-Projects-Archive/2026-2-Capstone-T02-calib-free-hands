"""Front window and tilt phase."""
from __future__ import annotations

import numpy as np

from .track import box_long


def front_window(times, present, kp, mask, box, pcfg) -> dict:
    """Lowest-motion contiguous window of front_window_s inside [front_skip_s, front_phase_s) with detection rate >= the minimum."""
    t = np.asarray(times)
    cand = np.flatnonzero((t >= pcfg["front_skip_s"]) & (t < pcfg["front_phase_s"]))
    bl = box_long(box)
    speed = np.full(len(t), np.nan)
    for i in range(1, len(t)):
        if present[i] and present[i - 1]:
            m = mask[i] & mask[i - 1]
            if m.sum() >= 4:
                speed[i] = float(np.median(np.linalg.norm(kp[i, m] - kp[i - 1, m], axis=-1)) / max(np.nanmean([bl[i], bl[i - 1]]), 1e-9))
    best, warn = None, []
    for s in cand:
        idx = cand[(t[cand] >= t[s]) & (t[cand] < t[s] + pcfg["front_window_s"] - 1e-9)]
        if t[s] + pcfg["front_window_s"] > pcfg["front_phase_s"] + 1e-9:
            break
        rate = float(present[idx].mean()) if len(idx) else 0.0
        sp = float(np.nanmean(speed[idx])) if np.isfinite(speed[idx]).any() else np.inf
        key = (rate < pcfg["front_min_detect_rate"], sp)
        if best is None or key < best[0]:
            best = (key, idx, rate, sp)
    if best is None:
        return {"idx": np.zeros(0, int), "detect_rate": 0.0, "speed_rel": float("nan"), "warnings": ["정면 구간에 창을 만들 프레임이 없습니다"]}
    if best[2] < pcfg["front_min_detect_rate"]:
        warn.append(f"정면 창의 검출률이 낮습니다({best[2]:.2f})")
    return {"idx": best[1], "detect_rate": best[2], "speed_rel": best[3], "t0": float(t[best[1][0]]), "t1": float(t[best[1][-1]]), "warnings": warn}


def auto_transition(times, theta, present, pcfg):
    """First time at which theta2d stays >= auto_theta_deg for auto_sustain_s."""
    t = np.asarray(times)
    ok = present & np.isfinite(theta) & (theta >= pcfg["auto_theta_deg"])
    for i in range(len(t)):
        if not ok[i]:
            continue
        j = np.flatnonzero((t >= t[i]) & (t < t[i] + pcfg["auto_sustain_s"]))
        if len(j) and ok[j].all():
            return float(t[i])
    return None


def tilt_indices(times, present, t_start) -> np.ndarray:
    t = np.asarray(times)
    return np.flatnonzero((t >= t_start) & present)
