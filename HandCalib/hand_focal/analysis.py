"""Analysis core shared by real videos and the synthetic study: tracked 2D keypoints -> template, theta2d, f (both estimators),
diagnostics, uncertainty and the verdict inputs. Uses 2D keypoints, their mask and the box only."""
from __future__ import annotations

import time

import cv2
import numpy as np

from . import estimate as E
from . import phases as PH
from . import template as TP
from . import uncertainty as U
from . import verdict as V
from .track import box_long

METHODS = ("fixed_template", "joint_delta")


def principal_point(ecfg, W, H):
    pp = ecfg.get("principal_point", "center")
    return (W / 2.0, H / 2.0) if pp == "center" else (float(pp[0]), float(pp[1]))


def undistort(kp, ecfg, W, H):
    """Optional distortion [k1, k2, p1, p2] with the convention K_d = [[L, 0, cx], [0, L, cy]] (L = long side), applied once."""
    d = ecfg.get("distortion")
    if not d:
        return kp
    cx, cy = principal_point(ecfg, W, H)
    L = float(max(W, H))
    Kd = np.array([[L, 0, cx], [0, L, cy], [0, 0, 1.0]])
    flat = kp.reshape(-1, 1, 2).astype(np.float64)
    ok = np.isfinite(flat).all(-1).ravel()
    out = flat.copy()
    out[ok] = cv2.undistortPoints(flat[ok], Kd, np.asarray(d, float), P=Kd)
    return out.reshape(kp.shape)


def _theta(T, kp, mask, present, ids):
    n = len(kp)
    th, ax = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        if not present[i]:
            continue
        v = mask[i, ids]
        if v.sum() >= 6:
            th[i], ax[i] = TP.affine_tilt(T[v], kp[i, ids][v])
    return th, ax


def axis_coverage(theta, axis, ids, theta_min=10.0, min_count=2):
    a = np.asarray([axis[i] for i in ids if np.isfinite(theta[i]) and theta[i] >= theta_min])
    counts = np.bincount((a // 30).astype(int) % 6, minlength=6) if len(a) else np.zeros(6, int)
    bins = int((counts >= min_count).sum())
    horiz = counts[[0, 5]].sum()                 # compressed image axis near 0 / 180 deg: left-right tilt
    vert = counts[[2, 3]].sum()                  # near 90 deg: up-down tilt
    dom = None
    if len(a) and vert > 0.85 * len(a):
        dom = "vertical_only"
    elif len(a) and horiz > 0.85 * len(a):
        dom = "horizontal_only"
    return {"counts": counts.tolist(), "bins": bins, "dominant": dom, "theta_min_deg": theta_min, "min_count": min_count}


def analyze(kp, mask, present, box, cls, times, size_wh, cfg, front_idx=None, tilt_start=None, methods_full=("fixed_template",),
            n_boot=None, seed=0, budget_s=None, log=None) -> dict:
    t_start_clock = time.time()
    timings = {}
    W, H = size_wh
    J, ecfg, pcfg = cfg["joints"], cfg["estimate"], cfg["phases"]
    est_ids = list(J["sets"][J["estimate_set"]])
    kp = undistort(np.asarray(kp, float), ecfg, W, H)
    mask, present = np.asarray(mask, bool), np.asarray(present, bool)
    times = np.asarray(times, float)
    n = len(times)
    bl = box_long(np.asarray(box, float))
    warnings = []

    # ---- front window and template
    if front_idx is None:
        fw = PH.front_window(times, present, kp, mask, box, pcfg)
        warnings += fw["warnings"]
    else:
        fw = {"idx": np.asarray(front_idx, int), "detect_rate": float(present[front_idx].mean()), "speed_rel": float("nan"),
              "t0": float(times[front_idx[0]]), "t1": float(times[front_idx[-1]]), "warnings": []}
    fr_full = [int(i) for i in fw["idx"] if present[i] and mask[i, est_ids].all()]
    base = {"warnings": warnings, "front_window": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in fw.items()},
            "front_frames": len(fr_full), "detect_rate": float(present.mean()) if n else 0.0, "n_frames": n}
    if len(fr_full) < 3:
        base["warnings"].append("모든 관절이 보이는 정면 프레임이 3장 미만이라 템플릿을 만들 수 없습니다")
        base["failed"] = "no_template"
        return base
    tpl = TP.build(kp[fr_full][:, est_ids], int(J["template_align_iters"]))
    T = tpl["template"]
    jitter_px = tpl["jitter_rms_px"]
    jitter_rel = jitter_px / float(np.nanmedian(bl[fr_full]))
    ext = TP.finger_extension(kp[fr_full])
    if ext["min"] < pcfg["finger_extension_min"]:
        base["warnings"].append(f"정면 구간 손가락 펴짐 지표가 낮습니다(최소 {ext['min']:.3f} < {pcfg['finger_extension_min']})")

    # ---- theta2d, phases
    theta, axis = _theta(T, kp, mask, present, est_ids)
    t_auto = PH.auto_transition(times, theta, present, pcfg)
    if tilt_start is None:
        t_nom = float(pcfg["front_phase_s"])
        if t_auto is not None and abs(t_auto - t_nom) > pcfg["auto_warn_diff_s"]:
            base["warnings"].append(f"자동 전환 시각({t_auto:.2f} s)이 명목 {t_nom:.1f} s와 {pcfg['auto_warn_diff_s']} s 넘게 다릅니다")
        tilt_start = t_auto if (pcfg["phase_mode"] == "auto" and t_auto is not None) else t_nom
    tilt_all = PH.tilt_indices(times, present, tilt_start)

    # ---- estimation problem (front window frames with all joints + tilt frames)
    cx, cy = principal_point(ecfg, W, H)
    huber = ecfg["huber_rel"] * np.where(np.isfinite(bl), bl, 0.0)
    used = sorted(set(fr_full) | set(int(i) for i in tilt_all))

    def problem(set_ids):
        pos = [est_ids.index(j) for j in set_ids]
        frames = [(kp[i][set_ids], mask[i][set_ids]) for i in range(n)]
        return E.Problem(T[pos], frames, cx, cy, huber, ecfg["min_points"]).subset(used)

    pb = problem(est_ids)
    tilt_ids = [i for i in pb.keep if i in set(int(x) for x in tilt_all)]
    front_ids = [i for i in pb.keep if i in set(fr_full)]
    grid = E.f_grid(ecfg, max(W, H))
    sigma2 = max(jitter_px ** 2, 0.25)
    t0 = time.time()
    sols = {m: E.run(pb, m, grid, ecfg, sigma2) for m in METHODS}
    timings["estimate_s"] = time.time() - t0

    # ---- per-frame bookkeeping
    role = np.array(["none"] * n, dtype=object)
    reason = np.array([""] * n, dtype=object)
    keep = set(pb.keep)
    for i in range(n):
        if not present[i]:
            reason[i] = "not_detected"
        elif i in set(fr_full):
            role[i] = "front"
        elif i in set(int(x) for x in fw["idx"]):
            reason[i] = "front_window_missing_joints"
        elif times[i] < tilt_start:
            reason[i] = "front_phase_outside_window"
        elif i in keep:
            role[i] = "tilt"
        else:
            reason[i] = "too_few_points"
    edge = np.zeros(n, bool)
    for i in np.flatnonzero(present):
        b = box[i]
        edge[i] = bool(b[0] <= 2 or b[1] <= 2 or b[2] >= W - 2 or b[3] >= H - 2)
    cls_p = np.asarray(cls)[present]
    lr = None
    if len(cls_p):
        vals, cnt = np.unique(cls_p, return_counts=True)
        lr = {"majority_cls": int(vals[np.argmax(cnt)]), "consistency": float(cnt.max() / cnt.sum()), "counts": {int(v): int(c) for v, c in zip(vals, cnt)}}
    th_tilt = theta[tilt_ids] if tilt_ids else np.zeros(0)
    th_tilt = th_tilt[np.isfinite(th_tilt)]
    cov = axis_coverage(theta, axis, tilt_ids)
    out = dict(base)
    out.update({
        "template": T.tolist(), "template_frames": fr_full, "obs_front_example": kp[fr_full[len(fr_full) // 2]][est_ids].tolist(), "front_jitter_px": jitter_px, "front_jitter_rel": jitter_rel, "finger_extension": ext,
        "finger_extension_ok": bool(ext["min"] >= pcfg["finger_extension_min"]),
        "theta2d_deg": theta.tolist(), "axis2d_deg": axis.tolist(), "tilt_start_s": float(tilt_start), "auto_transition_s": t_auto,
        "phase_mode": pcfg["phase_mode"] if front_idx is None else "given",
        "theta_tilt": {"median": float(np.median(th_tilt)) if len(th_tilt) else float("nan"), "p90": float(np.percentile(th_tilt, 90)) if len(th_tilt) else float("nan"),
                       "max": float(th_tilt.max()) if len(th_tilt) else float("nan")},
        "axis_coverage": cov, "n_target_frames": len(pb.keep), "n_front_used": len(front_ids), "n_tilt_used": len(tilt_ids),
        "role": role.tolist(), "reason": reason.tolist(), "edge_frame_rate": float(edge[present].mean()) if present.any() else 0.0,
        "lr": lr, "hand_size_px": {"box_long_median": float(np.nanmedian(bl[present])) if present.any() else float("nan")},
        "sigma2_px2": sigma2, "grid": grid.tolist(), "solutions": sols, "principal_point": [cx, cy],
        "method_diff_rel": abs(sols["fixed_template"]["f"] - sols["joint_delta"]["f"]) / sols["fixed_template"]["f"],
    })
    # ---- full uncertainty, diagnostics and verdict per method
    ucfg = cfg["uncertainty"]
    nb = int(ucfg["n_bootstrap"] if n_boot is None else n_boot)
    budget = float(ucfg["time_budget_s"] if budget_s is None else budget_s)
    box_med = float(np.nanmedian(bl[pb.keep]))
    out["per_method"] = {}
    for m in methods_full:
        t1 = time.time()
        sol = sols[m]
        start = (np.log(sol["f"]), sol["delta_deg"], sol["phi_deg"])
        diag = E.diagnostics(pb, sol, box_med, est_ids, tilt_ids)
        js = {}
        for name in J["uncertainty_sets"]:
            sub = problem(list(J["sets"][name]))
            js[name] = E.run(sub, m, grid, ecfg, sigma2, start=start if m == "joint_delta" else None)["f"] if len(sub.keep) >= 3 else float("nan")
        js_range = U.rel_range(list(js.values()), sol["f"])
        boot = U.block_bootstrap(pb, front_ids, tilt_ids, m, grid, ecfg, sigma2, sol, nb, int(ucfg["block_frames"]), seed, budget)
        cons = U.consistency(pb, front_ids, tilt_ids, axis, m, grid, ecfg, sigma2, sol)
        comb = U.combine(boot["rel_std"], js_range, cons["range"], out["method_diff_rel"])
        metrics = {"detect_rate": out["detect_rate"], "front_frames": out["front_frames"], "finger_extension_ok": out["finger_extension_ok"],
                   "theta_p90": out["theta_tilt"]["p90"], "axis_bins6": cov["bins"], "axis_dominant": cov["dominant"], "reproj_rms_rel": diag["reproj_rms_rel"],
                   "jointset_range": js_range, "bootstrap_rel_std": boot["rel_std"], "consistency_range": cons["range"], "delta_deg": sol["delta_deg"],
                   "sharpness": diag["sharpness"], "n_target_frames": out["n_target_frames"], "front_jitter_px": jitter_px, "front_jitter_rel": jitter_rel,
                   "edge_frame_rate": out["edge_frame_rate"], "method_diff": out["method_diff_rel"]}
        j = V.judge(metrics, cfg["verdict"])
        out["per_method"][m] = {"f": sol["f"], "diagnostics": diag, "jointsets": js, "jointset_range": js_range, "bootstrap": boot,
                                "consistency": cons, "f_rel_uncertainty": comb, "metrics": metrics, "verdict": j, "advice": V.advice(metrics, j),
                                "time_s": time.time() - t1}
    timings["total_s"] = time.time() - t_start_clock
    out["timings"] = timings
    return out
