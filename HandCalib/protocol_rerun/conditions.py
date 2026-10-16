"""Shooting-condition metrics (14's indicators) from one detector's run, and the pass / fail check against the yaml thresholds."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hand_focal import track as TR
from hand_focal.detectors import adapters as DA


def edge_flags(box, present, W, H, margin=2.0):
    e = np.zeros(len(box), bool)
    for i in np.flatnonzero(present):
        b = box[i]
        e[i] = bool(b[0] <= margin or b[1] <= margin or b[2] >= W - margin or b[3] >= H - margin)
    return e


def metrics_from_arrays(times, present, box, W, H, theta, role, auto_transition_s, front_jitter_rel, axis_bins, finger_ext_min) -> dict:
    """All inputs are per-frame arrays of one tracked hand (box (n, 4); theta in degrees, NaN where not computed; role 'tilt' marks
    the tilt phase) plus 14's scalar indicators. Returns the condition metrics (None where not computable)."""
    present = np.asarray(present, bool)
    box = np.asarray(box, float)
    bl = TR.box_long(box)
    short = float(min(W, H))
    edge = edge_flags(box, present, W, H)
    th = np.asarray(theta, float)
    tilt = (np.asarray(role) == "tilt") & np.isfinite(th)
    return {"box_short_ratio": float(np.nanmedian(bl[present]) / short) if present.any() else None,
            "edge_rate": float(edge[present].mean()) if present.any() else None,
            "transition_s": auto_transition_s, "front_jitter_rel": front_jitter_rel,
            "theta_p95_deg": float(np.percentile(th[tilt], 95)) if tilt.any() else None,
            "theta_tilt_median_deg": float(np.median(th[tilt])) if tilt.any() else None,
            "theta_max_deg": float(np.max(th[tilt])) if tilt.any() else None,
            "axis_bins": axis_bins, "finger_extension_min": finger_ext_min,
            "series": {"t": np.asarray(times, float).tolist(), "box_short_ratio": (bl / short).tolist(), "edge": edge.tolist(), "theta": th.tolist()}}


def metrics_from_run(det_dir: Path, tracking_cfg: dict, front_skip_s: float) -> dict:
    """Reads a 14-format run folder (results.json + detections.npz); re-tracks with 14's function (cheap, deterministic) for the
    per-frame boxes. theta2d / roles / axis come from the run's own analysis."""
    r = json.loads((det_dir / "results.json").read_text())
    a = r.get("analysis", {})
    if "theta2d_deg" not in a:
        return {"failed": a.get("failed") or "no analysis"}
    W, H = r["normalization"]["output"]["display_w"], r["normalization"]["output"]["display_h"]
    frames = r["frames"]["list"]
    times = np.array([x["time_s"] for x in frames])
    det = DA._load(det_dir / "detections.npz")
    tr = TR.track(det, times, (W, H), tracking_cfg, front_start_s=front_skip_s)
    th = np.array([np.nan if v is None else v for v in a["theta2d_deg"]], float)
    ax = np.array([np.nan if v is None else v for v in a["axis2d_deg"]], float)
    m = metrics_from_arrays(times, tr["present"], tr["box"], W, H, th, a["role"], a.get("auto_transition_s"), a.get("front_jitter_rel"),
                            a["axis_coverage"]["bins"], a["finger_extension"]["min"])
    m["series"]["axis"] = ax.tolist()
    m["series"]["present"] = tr["present"].tolist()
    return m


def check(m: dict, c: dict) -> dict:
    """Pass / fail per condition; a metric that cannot be computed fails. guide = every condition passes."""
    if "failed" in m:
        return {"guide": False, "items": {}, "reasons": [f"분석 실패: {m['failed']}"]}
    lo, hi = c["theta_tilt_median_range_deg"]
    def ok(x, f):
        return x is not None and bool(f(x))
    items = {
        "box_short_ratio": ok(m["box_short_ratio"], lambda x: x <= c["box_short_ratio_max"]),
        "edge_rate": ok(m["edge_rate"], lambda x: x <= c["edge_rate_max"]),
        "transition": ok(m["transition_s"], lambda x: x >= c["transition_min_s"]),
        "theta_p95": ok(m["theta_p95_deg"], lambda x: x <= c["theta_p95_max_deg"]),
        "theta_tilt_median": ok(m["theta_tilt_median_deg"], lambda x: lo <= x <= hi),
        "axis_bins": ok(m["axis_bins"], lambda x: x >= c["axis_bins_min"]),
        "finger_extension": ok(m["finger_extension_min"], lambda x: x >= c["finger_extension_min"]),
    }
    msg = {"box_short_ratio": lambda: f"손 박스/짧은 변 {_f(m['box_short_ratio'], 2)} > {c['box_short_ratio_max']}",
           "edge_rate": lambda: f"가장자리 접촉 {_p(m['edge_rate'])} > {100 * c['edge_rate_max']:.0f}%",
           "transition": lambda: ("자동 전환 없음" if m["transition_s"] is None else f"자동 전환 {m['transition_s']:.2f} s < {c['transition_min_s']} s"),
           "theta_p95": lambda: f"θ2d p95 {_f(m['theta_p95_deg'], 0)}° > {c['theta_p95_max_deg']:.0f}°",
           "theta_tilt_median": lambda: f"기울기 구간 θ2d 중앙값 {_f(m['theta_tilt_median_deg'], 0)}°가 {lo:.0f}–{hi:.0f}° 밖",
           "axis_bins": lambda: f"축 덮임 {m['axis_bins']}/6 < {c['axis_bins_min']}/6",
           "finger_extension": lambda: f"손가락 펴짐 {_f(m['finger_extension_min'], 3)} < {c['finger_extension_min']}"}
    return {"guide": all(items.values()), "items": items, "reasons": [msg[k]() for k, v in items.items() if not v]}


def short_text(m: dict) -> str:
    if "failed" in m:
        return "분석 실패"
    tr = "없음" if m["transition_s"] is None else f"{m['transition_s']:.1f}초"
    return (f"박스 {_f(m['box_short_ratio'], 2)}, 가장자리 {_p(m['edge_rate'])}, 전환 {tr}, θ2d p95 {_f(m['theta_p95_deg'], 0)}°, "
            f"기울기 중앙값 {_f(m['theta_tilt_median_deg'], 0)}°, 최대 {_f(m['theta_max_deg'], 0)}°, 덮임 {m['axis_bins']}/6, "
            f"펴짐 {_f(m['finger_extension_min'], 3)}, 정면 지터 {_p(m['front_jitter_rel'], 2)}")


def _f(x, d):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def _p(x, d=0):
    return "nan" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}%"
