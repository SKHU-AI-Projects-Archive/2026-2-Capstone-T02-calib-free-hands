"""GOOD / ROUGH / REJECT for the hand f (thresholds from the yaml, fixed before the first run) and Korean re-shoot advice."""
from __future__ import annotations

import numpy as np


def _le(x, t):
    return bool(np.isfinite(x) and x <= t)


def _ge(x, t):
    return bool(np.isfinite(x) and x >= t)


def judge(m: dict, vcfg: dict) -> dict:
    g, r = vcfg["good"], vcfg["rough"]
    checks_good = {
        "detect_rate": _ge(m["detect_rate"], g["detect_rate_min"]),
        "front_frames": m["front_frames"] >= g["front_frames_min"],
        "finger_extension": (not g["finger_extension_required"]) or bool(m["finger_extension_ok"]),
        "theta_p90": _ge(m["theta_p90"], g["theta_p90_min_deg"]),
        "axis_bins": m["axis_bins6"] >= g["axis_bins_min"],
        "reproj_rms": _le(m["reproj_rms_rel"], g["reproj_rms_rel_max"]),
        "jointset_range": _le(m["jointset_range"], g["jointset_range_max"]),
        "bootstrap": _le(m["bootstrap_rel_std"], g["bootstrap_rel_std_max"]),
        "consistency": _le(m["consistency_range"], g["consistency_range_max"]),
        "delta": _le(abs(m["delta_deg"]), g["delta_abs_max_deg"]),
        "sharpness": _ge(m["sharpness"], g["sharpness_min"]),
    }
    checks_rough = {
        "detect_rate": _ge(m["detect_rate"], r["detect_rate_min"]),
        "theta_p90": _ge(m["theta_p90"], r["theta_p90_min_deg"]),
        "axis_bins": m["axis_bins6"] >= r["axis_bins_min"],
        "reproj_rms": _le(m["reproj_rms_rel"], r["reproj_rms_rel_max"]),
        "jointset_range": _le(m["jointset_range"], r["jointset_range_max"]),
    }
    if m["n_target_frames"] < vcfg["min_target_frames"]:
        v = "REJECT"
    elif all(checks_good.values()):
        v = "GOOD"
    elif all(checks_rough.values()):
        v = "ROUGH"
    else:
        v = "REJECT"
    return {"verdict": v, "good_checks": checks_good, "rough_checks": checks_rough, "failed_good": [k for k, ok in checks_good.items() if not ok],
            "failed_rough": [k for k, ok in checks_rough.items() if not ok]}


def advice(m: dict, j: dict) -> list:
    a = []
    f = set(j["failed_good"])
    if "theta_p90" in f:
        a.append(f"기울기가 부족합니다(p90 {m['theta_p90']:.0f}°). 손을 30~45°까지 크게 기울이세요")
    if "axis_bins" in f:
        dom = m.get("axis_dominant")
        a.append("상하 기울기만 있고 좌우가 없습니다" if dom == "vertical_only" else "좌우 기울기만 있고 상하가 없습니다" if dom == "horizontal_only"
                 else "기울기 방향이 일부에 몰려 있습니다. 상하좌우로 고르게 기울이세요")
    if m["front_jitter_px"] == m["front_jitter_px"] and ("front_frames" in f or m["front_jitter_rel"] > 0.01):
        a.append(f"정면 구간에서 손이 충분히 정지하지 않았습니다(지터 {m['front_jitter_px']:.1f} px). 손을 폰과 평행하게 1초 이상 멈추세요")
    if "finger_extension" in f:
        a.append("손가락이 펴지지 않았습니다")
    if m.get("edge_frame_rate", 0) > 0.05:
        a.append("손이 화면 가장자리에 걸렸습니다. 영상 중앙에서 시작하세요")
    if np.isfinite(m.get("method_diff", np.nan)) and m["method_diff"] > 0.04:
        a.append(f"정면 가정 민감도가 큽니다(두 방식 차이 {100 * m['method_diff']:.1f}%)")
    return a


SHOOTING_GUIDE = [
    "정면 구간: 손바닥을 렌즈 쪽이 아니라 폰 화면과 평행하게 두고, 손을 영상 중앙에서 1초 이상 멈춥니다(처음 약 3초).",
    "그 뒤 상하좌우로 30~45°까지 천천히 기울입니다(한 방향만이 아니라 네 방향 모두).",
    "거리는 30~50 cm, 흐려지지 않게 천천히 움직이고 손가락은 쫙 편 상태를 유지합니다.",
]
