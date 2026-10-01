"""Evaluate the CAM-EXP-008 oracle camera-nuisance diagnostic.

Answers one question: was CAM-008's hand profile shallow and biased toward
q = 1.184 BECAUSE the scene-estimated nuisance parameters were wrong?

Reads the CAM-008 reference focal only for evaluation, exactly as CAM-008 did.
CAM-EXP-008's own outputs are never written.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
RUN = HERE.parents[1]
C8 = RUN.parent / "CAM-EXP-008_offline_sequence_scene_hand_focal_fusion"
sys.path.insert(0, str(C8 / "src"))
sys.path.insert(0, str(RUN / "src"))

from common import (MANIFESTS, RAW, SUM, TAB, VIEW_TARGETS,  # noqa: E402
                    cluster_bootstrap, fnum, huber, mad_sigma, read_csv,
                    read_json, write_csv, write_json)

SUBSETS = MANIFESTS / "cam_exp_008_frame_subsets_v1.csv.gz"
SPLITS = MANIFESTS / "cam_exp_008_outer_splits_v1.json"
PROF = {"N0": C8 / "cache" / "hand_profiles" / "real"}
for c in ("N1", "N2", "N3", "N4"):
    PROF[c] = RUN / "cache" / "nuisance_profiles" / c
PRIMARY = "ALL_COMMON"
LAMBDAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


def hand_cost(curves, frames_of_curve, keep):
    byf = defaultdict(list)
    for c, fr in zip(curves, frames_of_curve):
        if fr in keep:
            byf[fr].append(c)
    if not byf:
        return None
    fc = [np.nanmedian(np.asarray(v, float), axis=0) for v in byf.values()]
    C = np.nanmedian(np.asarray(fc, float), axis=0)
    if not np.isfinite(C).any():
        return None
    return C - np.nanmin(C)


def pick(curve, f_abs):
    c = np.asarray(curve, float)
    if not np.isfinite(c).any():
        return float("nan"), "NO_SOLUTION"
    i = int(np.nanargmin(c))
    if i == 0 or i == len(c) - 1:
        return float(f_abs[i]), "BOUNDARY"
    lg = np.log(f_abs)
    y0, y1, y2 = c[i - 1], c[i], c[i + 1]
    den = y0 - 2 * y1 + y2
    if np.isfinite(den) and abs(den) > 1e-12:
        d = float(np.clip(0.5 * (y0 - y2) / den, -1.0, 1.0))
        return float(np.exp(lg[i] + d * (lg[i + 1] - lg[i]))), "ok"
    return float(f_abs[i]), "ok"


def main() -> None:
    subs = read_csv(SUBSETS)
    keep_of = defaultdict(set)
    for r in subs:
        if r["subset"] == PRIMARY:
            keep_of[(r["sequence"], r["camera"])].add(int(r["frame"]))
    tgt = {(r["sequence"], r["camera"]): r for r in read_csv(VIEW_TARGETS)}
    splits = read_json(SPLITS)

    shape_rows, est_rows = [], []
    for cond, d in PROF.items():
        if not d.exists():
            continue
        for p in sorted(d.glob("*.npz")):
            seq, cam = p.stem.split("__")
            if (seq, cam) not in tgt:
                continue
            z = np.load(p, allow_pickle=True)
            cur = np.asarray(z["curves"], float)
            if cur.size == 0:
                continue
            grid = np.asarray(z["grid"], float)
            f_anchor = float(z["f_scene"])
            f_abs = grid * f_anchor
            keys = [str(k) for k in z["keys"]]
            fr = [int(k.split("|")[0]) for k in keys]
            keep = keep_of[(seq, cam)]
            L = hand_cost(cur, fr, keep)
            if L is None:
                continue
            ref = fnum(tgt[(seq, cam)]["gt_reference_focal_px"])
            sfr = np.asarray(z["scene_frames"], int)
            sfx = np.asarray(z["scene_fx"], float)
            m = np.isin(sfr, sorted(keep))
            if m.sum() < 2:
                continue
            f_scene = float(np.median(sfx[m]))
            sigma = mad_sigma(np.log(sfx[m]))
            Ls = huber(np.log(f_abs / f_scene) / sigma)

            i = int(np.nanargmin(L))
            q_pref = float(grid[i])
            shape_rows.append({
                "condition": cond, "sequence": seq, "camera": cam,
                "physical_camera_id": tgt[(seq, cam)]["physical_camera_id"],
                "L_hand_max_px": float(np.nanmax(L)),
                "preferred_q": q_pref,
                "boundary": int(i in (0, len(L) - 1)),
                # does the hand point toward the reference focal from f_scene?
                "correct_direction": int(
                    np.sign(q_pref - 1.0) == np.sign(ref / f_scene - 1.0)),
                "f_scene": f_scene, "f_reference": ref,
            })
            f0, _ = pick(Ls, f_abs)
            row = {"condition": cond, "sequence": seq, "camera": cam,
                   "physical_camera_id": tgt[(seq, cam)]["physical_camera_id"],
                   "f_scene_only": f0, "f_reference": ref}
            for lam in LAMBDAS:
                fh, _ = pick(Ls + lam * L, f_abs)
                row[f"f_lam_{lam}"] = fh
            est_rows.append(row)

    write_csv(RAW / "nuisance_profile_shape.csv", shape_rows)
    write_csv(RAW / "nuisance_focal_estimates.csv", est_rows)

    # ---- profile shape per condition ------------------------------------
    summ = []
    for cond in PROF:
        s = [r for r in shape_rows if r["condition"] == cond]
        if not s:
            continue
        summ.append({
            "condition": cond, "n_views": len(s),
            "L_hand_max_px_median": round(float(np.median(
                [r["L_hand_max_px"] for r in s])), 4),
            "preferred_q_median": round(float(np.median(
                [r["preferred_q"] for r in s])), 4),
            "boundary_rate": round(float(np.mean(
                [r["boundary"] for r in s])), 4),
            "correct_direction_rate": round(float(np.mean(
                [r["correct_direction"] for r in s])), 4),
        })
    write_csv(SUM / "nuisance_profile_shape_summary.csv", summ)

    # ---- fused focal error, lambda chosen on TRAINING cameras only -------
    fused = []
    by_cond = defaultdict(list)
    for r in est_rows:
        by_cond[r["condition"]].append(r)
    for cond, rs in sorted(by_cond.items()):
        idx = {(r["sequence"], r["camera"]): r for r in rs}
        vs = sorted(idx)
        gains, e0s, e1s, cams = [], [], [], []
        for fold in splits["LOCO_PHYSICAL_CAMERA"]["folds"]:
            ho = fold["held_out_physical_camera"]
            test = [v for v in vs if idx[v]["physical_camera_id"] == ho]
            train = [v for v in vs if v not in test]
            if not test or len(train) < 5:
                continue
            best, berr = None, np.inf
            for lam in LAMBDAS:
                e = [100 * abs(fnum(idx[v][f"f_lam_{lam}"])
                               - fnum(idx[v]["f_reference"]))
                     / fnum(idx[v]["f_reference"]) for v in train]
                e = [x for x in e if np.isfinite(x)]
                if len(e) < 5:
                    continue
                if np.median(e) < berr:
                    best, berr = lam, float(np.median(e))
            if best is None:
                continue
            for v in test:
                ref = fnum(idx[v]["f_reference"])
                e0 = 100 * abs(fnum(idx[v]["f_scene_only"]) - ref) / ref
                e1 = 100 * abs(fnum(idx[v][f"f_lam_{best}"]) - ref) / ref
                e0s.append(e0)
                e1s.append(e1)
                gains.append(e0 - e1)
                cams.append(idx[v]["physical_camera_id"])
        if not gains:
            continue
        g, glo, ghi = cluster_bootstrap(np.array(gains), cams)
        m0, _, _ = cluster_bootstrap(np.array(e0s), cams)
        m1, _, _ = cluster_bootstrap(np.array(e1s), cams)
        fused.append({
            "condition": cond, "n_views": len(gains),
            "scene_only_median_err_pct": round(m0, 4),
            "scene_plus_hand_median_err_pct": round(m1, 4),
            "median_paired_gain_pp": round(g, 4),
            "gain_ci_lo": round(glo, 4), "gain_ci_hi": round(ghi, 4),
            "relative_reduction_pct": round(100 * (m0 - m1) / m0, 4),
            "share_improved": round(float(np.mean(np.array(gains) > 0)), 4),
        })
    write_csv(SUM / "nuisance_fused_summary.csv", fused)

    n0 = next((r for r in summ if r["condition"] == "N0"), None)
    n4 = next((r for r in summ if r["condition"] == "N4"), None)
    f0 = next((r for r in fused if r["condition"] == "N0"), None)
    f4 = next((r for r in fused if r["condition"] == "N4"), None)

    sharpened = bool(n0 and n4 and n4["L_hand_max_px_median"]
                     > 2 * n0["L_hand_max_px_median"])
    debiased = bool(n0 and n4
                    and abs(n4["preferred_q_median"] - 1.0)
                    < 0.5 * abs(n0["preferred_q_median"] - 1.0))
    materially_better = bool(f0 and f4
                             and f4["relative_reduction_pct"] >= 10)
    verdict = ("CAM008_NUISANCE_LIMITED"
               if (sharpened and debiased and materially_better)
               else "CAM008_NEGATIVE_ROBUST_TO_CAMERA_NUISANCE")

    write_json(SUM / "closure_verdict.json", {
        "question": "was CAM-EXP-008's hand profile shallow and biased toward "
                    "q = 1.184 because the scene-estimated principal point, "
                    "distortion and aspect ratio were wrong?",
        "profile_shape": summ,
        "fused": fused,
        "N4_sharpened_profile": sharpened,
        "N4_reduced_q_bias": debiased,
        "N4_materially_better_fusion": materially_better,
        "verdict": verdict,
        "scope": "N1-N4 are ORACLE_DIAGNOSTIC_ONLY. The provided focal "
                 "MAGNITUDE was never used. CAM-EXP-008's own results are "
                 "unchanged.",
    })

    print(f"{'cond':5s} {'Lmax px':>9s} {'pref q':>8s} {'bnd':>6s} "
          f"{'dir ok':>7s}")
    for r in summ:
        print(f"{r['condition']:5s} {r['L_hand_max_px_median']:9.3f} "
              f"{r['preferred_q_median']:8.3f} {r['boundary_rate']:6.2f} "
              f"{r['correct_direction_rate']:7.2f}")
    print()
    for r in fused:
        print(f"{r['condition']:5s} scene {r['scene_only_median_err_pct']:7.3f}%"
              f"  +hand {r['scene_plus_hand_median_err_pct']:7.3f}%"
              f"  gain {r['median_paired_gain_pp']:+7.4f} pp"
              f"  relred {r['relative_reduction_pct']:+6.2f}%")
    print(f"\n{verdict}")


if __name__ == "__main__":
    main()
