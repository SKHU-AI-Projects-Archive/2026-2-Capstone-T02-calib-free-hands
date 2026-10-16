"""Diagnostic frame filter for the 17 pipeline: drop tilt-phase frames with theta2d above a threshold or a box touching the border,
then re-run 14's analysis core unchanged (front window / template unaffected because only frames after the tilt start change)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hand_focal import analysis as A
from hand_focal import diagnostic as D
from hand_focal import estimate as E
from hand_focal import verdict as V
from hand_focal.track import box_long
from protocol_rerun.conditions import edge_flags


def keep_mask(times, present, theta, box, W, H, tilt_start, fcfg) -> np.ndarray:
    """True for frames the analysis may use. Frames before tilt_start are untouched; tilt-phase frames must pass the filter."""
    t = np.asarray(times, float)
    th = np.asarray(theta, float)
    keep = np.asarray(present, bool).copy()
    tilt = t >= tilt_start
    bad = tilt & ~(np.isfinite(th) & (th <= fcfg["theta_max_deg"]))
    if fcfg.get("exclude_edge"):
        bad |= tilt & edge_flags(box, present, W, H, fcfg.get("edge_margin_px", 2.0))
    return keep & ~bad


def run_one(run_dir: Path, pcfg: dict, fcfg: dict, methods_full=("fixed_template",)) -> dict:
    L0 = D.load_official(run_dir, pcfg)                          # 14 tracking + analysis (theta2d from the official template)
    a0, tr, W, H, times = L0["a"], L0["tr"], L0["W"], L0["H"], L0["times"]
    if "template" not in a0:
        return {"failed": a0.get("failed")}
    th = np.array([np.nan if v is None else v for v in a0["theta2d_deg"]], float)
    keep = keep_mask(times, tr["present"], th, tr["box"], W, H, a0["tilt_start_s"], fcfg)
    a = A.analyze(tr["kp"], tr["mask"], keep, tr["box"], tr["cls"], times, (W, H), pcfg, methods_full=methods_full,
                  seed=int(pcfg["experiment"]["seed"]))
    if "per_method" not in a:
        return {"failed": a.get("failed"), "n_kept_tilt": int((keep & (times >= a0["tilt_start_s"])).sum())}
    same_template = np.allclose(np.array(a["template"]), np.array(a0["template"]))
    out = {"same_template": bool(same_template), "tilt_start_s": a0["tilt_start_s"],
           "n_tilt_before": int((tr["present"] & (times >= a0["tilt_start_s"])).sum()),
           "n_tilt_kept": int((keep & (times >= a0["tilt_start_s"])).sum()), "solutions": {}, "per_method": {}}
    for m in ("fixed_template", "joint_delta"):
        s = a["solutions"][m]
        out["solutions"][m] = {"f": s["f"], "at_grid_edge": s["at_grid_edge"], "delta_deg": s.get("delta_deg")}
    for m, pm in a["per_method"].items():
        met = dict(pm["metrics"])
        if fcfg.get("verdict_detect_rate") == "original":
            met["detect_rate"] = float(tr["present"].mean())
        j = V.judge(met, pcfg["verdict"])
        out["per_method"][m] = {"f": pm["f"], "verdict": j["verdict"], "failed_good": j["failed_good"], "failed_rough": j["failed_rough"],
                                "f_rel_uncertainty": pm["f_rel_uncertainty"], "bootstrap_rel_std": pm["bootstrap"]["rel_std"],
                                "consistency_range": pm["consistency"]["range"], "sharpness": pm["diagnostics"]["sharpness"],
                                "reproj_rms_rel": pm["diagnostics"]["reproj_rms_rel"], "jointsets": pm["jointsets"], "metrics": met}
    # profiles (official + extended grid) of the kept frames, flat17 and palm sets
    L = float(max(W, H))
    off = E.f_grid(pcfg["estimate"], L)
    ext = np.geomspace(0.2 * L, 5.0 * L, 90)
    est_ids = list(pcfg["joints"]["sets"][pcfg["joints"]["estimate_set"]])
    T = np.array(a["template"])
    used = [i for i, r in enumerate(a["role"]) if r in ("front", "tilt")]
    bl = box_long(tr["box"])
    huber = pcfg["estimate"]["huber_rel"] * np.where(np.isfinite(bl), bl, 0.0)
    cx, cy = a["principal_point"]
    prof = {}
    for sname in ("palm4", "palm5", "palm_proximal9", "flat17"):
        ids = list(pcfg["joints"]["sets"][sname])
        pos = [est_ids.index(j) for j in ids]
        r = {}
        for gname, g in (("official", off), ("extended", ext)):
            C, _ = D.frame_cost_matrix(T, tr["kp"], tr["mask"], used, pos, ids, g, cx, cy, huber)
            ok = np.isfinite(C).all(1)
            p = C[ok].mean(0) if ok.sum() else np.full(len(g), np.nan)
            k = int(np.nanargmin(p)) if np.isfinite(p).any() else 0
            r[gname] = {"grid": g.tolist(), "profile": p.tolist(), "min_at_px": float(g[k]),
                        "at": "interior" if 0 < k < len(g) - 1 else ("lower" if k == 0 else "upper"), "n_frames": int(ok.sum())}
        prof[sname] = r
    out["profiles"] = prof
    out["grid_px"] = [float(off[0]), float(off[-1])]
    return out
