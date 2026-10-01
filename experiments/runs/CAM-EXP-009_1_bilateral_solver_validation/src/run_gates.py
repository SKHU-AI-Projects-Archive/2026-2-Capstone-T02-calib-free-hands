"""Gates G0-G4. The corrected full sweep runs only if all of them pass.

G0  camera-model round trip (raw and ideal)
G1  true-geometry projection + PnP at the true focal
G2  bone fitter basic validation at q = 1.0, three initialisations
G3  synthetic determinism, process-independent seeds
G4  candidate-focal scoring conventions internally consistent
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DIST_REAL, F_TRUE, GATE, N_BONES, N_MULTISTART, RAW,  # noqa: E402
                    SUM, arr_sha, assemble, make_K, q_grid, stable_seed,
                    write_csv, write_json)
from corrected_bone_fitter import (bilateral_distance, fit_bones,  # noqa: E402
                                   multistart_inits, pose_ideal, reproj_ideal,
                                   reproj_raw, score_fixed_shape,
                                   undistort_once)
from synthetic import make_sequence


def g0_camera_roundtrip():
    import cv2
    rows = []
    for t in range(10):
        rng = np.random.default_rng(stable_seed("G0", t))
        seq = make_sequence(rng, n_frames=4, dist=DIST_REAL, noise_px=0.0)
        K = make_K(F_TRUE)
        for side in ("left", "right"):
            for dirs, uv_raw, use, Xc_true in seq[side + "_full"]:
                # A: re-project the true camera-frame geometry, raw space
                proj, _ = cv2.projectPoints(
                    np.ascontiguousarray(Xc_true.reshape(-1, 1, 3),
                                         np.float64),
                    np.zeros(3), np.zeros(3), K,
                    DIST_REAL.reshape(1, -1))
                raw_err = np.linalg.norm(
                    proj.reshape(-1, 2) - uv_raw, axis=1)
                # B: undistort the raw pixels, compare to a pinhole projection
                uvi = undistort_once(uv_raw, K, DIST_REAL)
                z = Xc_true[:, 2]
                pin = np.stack([K[0, 0] * Xc_true[:, 0] / z + K[0, 2],
                                K[1, 1] * Xc_true[:, 1] / z + K[1, 2]], 1)
                ideal_err = np.linalg.norm(pin - uvi, axis=1)
                rows.append({
                    "trial": t, "side": side,
                    "raw_median_px": float(np.median(raw_err)),
                    "raw_max_px": float(np.max(raw_err)),
                    "ideal_median_px": float(np.median(ideal_err)),
                    "ideal_max_px": float(np.max(ideal_err))})
    write_csv(RAW / "camera_roundtrip_trials.csv.gz", rows)
    rm = float(np.median([r["raw_median_px"] for r in rows]))
    rx = float(np.max([r["raw_max_px"] for r in rows]))
    im = float(np.median([r["ideal_median_px"] for r in rows]))
    ix = float(np.max([r["ideal_max_px"] for r in rows]))
    ok = (rm < GATE["raw_roundtrip_median_px"]
          and rx < GATE["raw_roundtrip_max_px"]
          and im < GATE["ideal_roundtrip_median_px"]
          and ix < GATE["ideal_roundtrip_max_px"])
    res = {"raw_median_px": rm, "raw_max_px": rx,
           "ideal_median_px": im, "ideal_max_px": ix,
           "thresholds": {k: GATE[k] for k in GATE if "roundtrip" in k},
           "verdict": "PASS" if ok else "CAMERA_MODEL_VALIDATION_FAILED"}
    write_json(SUM / "camera_model_validation.json", res)
    return ok, res


def g1_true_geometry():
    rows = []
    for t in range(10):
        rng = np.random.default_rng(stable_seed("G1", t))
        seq = make_sequence(rng, n_frames=8, dist=DIST_REAL, noise_px=0.0)
        K = make_K(F_TRUE)
        for side, ltrue in (("left", seq["lL_true"]),
                            ("right", seq["lR_true"])):
            l = ltrue / ltrue.sum()
            i, r = score_fixed_shape(seq[side], K, DIST_REAL, l)
            rows.append({"trial": t, "side": side,
                         "ideal_px": i, "raw_px": r})
    write_csv(RAW / "true_geometry_trials.csv.gz", rows)
    im = float(np.nanmedian([r["ideal_px"] for r in rows]))
    rm = float(np.nanmedian([r["raw_px"] for r in rows]))
    ok = (im <= GATE["true_geometry_median_px"]
          and rm <= GATE["true_geometry_median_px"])
    res = {"ideal_median_px": im, "raw_median_px": rm,
           "threshold_px": GATE["true_geometry_median_px"],
           "verdict": "PASS" if ok else "POSE_OR_PROJECTION_VALIDATION_FAILED"}
    write_json(SUM / "true_geometry_validation.json", res)
    return ok, res


def g2_bone_fitter():
    rows = []
    for t in range(10):
        rng = np.random.default_rng(stable_seed("G2", t))
        seq = make_sequence(rng, n_frames=12, dist=DIST_REAL, noise_px=0.0)
        K = make_K(F_TRUE)
        mrng = np.random.default_rng(stable_seed("G2_ms", t))
        inits = multistart_inits(mrng, N_MULTISTART)
        for side, ltrue in (("left", seq["lL_true"]),
                            ("right", seq["lR_true"])):
            lt = ltrue / ltrue.sum()
            # I0 uniform
            l0, i0, r0 = fit_bones(seq[side], K, DIST_REAL)
            # I1 multi-start, pick the best IDEAL reprojection (no GT used)
            best = None
            allL = []
            for v in inits:
                lm, im, rm = fit_bones(seq[side], K, DIST_REAL, l_init=v)
                if lm is None:
                    continue
                allL.append(lm)
                if best is None or im < best[1]:
                    best = (lm, im, rm)
            # I2 oracle init, DIAGNOSTIC ONLY
            l2, i2, r2 = fit_bones(seq[side], K, DIST_REAL, l_init=lt)
            spread = np.nan
            if len(allL) > 1:
                A = np.stack(allL)
                d = np.abs(A[:, None, :] - A[None, :, :]).mean(-1)
                spread = float(np.median(d[np.triu_indices(len(A), 1)]))
            rows.append({
                "trial": t, "side": side,
                "I0_ideal_px": i0, "I0_raw_px": r0,
                "I0_bone_L1": float(np.mean(np.abs(l0 - lt)))
                if l0 is not None else np.nan,
                "I0_cos": float(np.dot(l0, lt)
                                / (np.linalg.norm(l0) * np.linalg.norm(lt)))
                if l0 is not None else np.nan,
                "I1_ideal_px": best[1] if best else np.nan,
                "I1_bone_L1": float(np.mean(np.abs(best[0] - lt)))
                if best else np.nan,
                "I2_ideal_px": i2, "I2_raw_px": r2,
                "I2_bone_L1": float(np.mean(np.abs(l2 - lt)))
                if l2 is not None else np.nan,
                "multistart_shape_spread_L1": spread,
            })
    write_csv(RAW / "bone_fitter_trials.csv.gz", rows)

    def med(k):
        return float(np.nanmedian([r[k] for r in rows]))

    i0, i1, i2 = med("I0_ideal_px"), med("I1_ideal_px"), med("I2_ideal_px")
    spread = med("multistart_shape_spread_L1")
    thr = GATE["true_geometry_median_px"]
    if i2 > thr:
        tag = "BONE_FITTER_IMPLEMENTATION_INVALID"
    elif i0 > thr:
        tag = "BONE_FITTER_LOCAL_MINIMUM"
    else:
        tag = "PASS"
    shape_nonid = bool(i0 <= thr and spread > 0.01)
    res = {"I0_uniform_ideal_px": i0, "I1_multistart_ideal_px": i1,
           "I2_oracle_init_ideal_px": i2,
           "I0_bone_L1": med("I0_bone_L1"), "I2_bone_L1": med("I2_bone_L1"),
           "I0_cosine": med("I0_cos"),
           "multistart_shape_spread_L1": spread,
           "shape_nonidentifiable": shape_nonid,
           "threshold_px": thr, "verdict": tag}
    write_json(SUM / "bone_fitter_validation.json", res)
    return tag == "PASS", res


def g3_determinism():
    """Process-independent determinism of the synthetic generator."""
    import json as _j
    import subprocess
    code = (
        "import sys;sys.path.insert(0,r'%s')\n"
        "import numpy as np, json\n"
        "from common import stable_seed, arr_sha, DIST_REAL\n"
        "from synthetic import make_sequence\n"
        "out=[]\n"
        "for t in range(10):\n"
        "    rng=np.random.default_rng(stable_seed('G3',t))\n"
        "    s=make_sequence(rng,n_frames=6,dist=DIST_REAL,noise_px=1.0)\n"
        "    out.append([arr_sha(np.stack([f[1] for f in s['left']])),\n"
        "                arr_sha(s['lL_true'])])\n"
        "print(json.dumps(out))"
    ) % str(Path(__file__).resolve().parent)
    outs = []
    for _ in range(2):
        p = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True)
        outs.append(p.stdout.strip().splitlines()[-1] if p.stdout else "")
    ok = bool(outs[0]) and outs[0] == outs[1]
    res = {"identical_across_processes": ok,
           "n_checked": 10,
           "seeding": "sha256 of the condition name; Python hash() is never "
                      "used because it is salted per process",
           "verdict": "SYNTHETIC_DETERMINISM_PASS" if ok else "FAILED"}
    write_json(SUM / "determinism_validation.json", res)
    return ok, res


def g4_scoring_consistency():
    """Do the ideal and raw scores rank candidate focals consistently?"""
    from common import spearman_safe
    grid = q_grid()
    rows = []
    for t in range(4):
        rng = np.random.default_rng(stable_seed("G4", t))
        seq = make_sequence(rng, n_frames=10, dist=DIST_REAL, noise_px=0.0)
        li, lr = [], []
        for q in grid[::8]:
            K = make_K(q * F_TRUE)
            l = seq["lL_true"] / seq["lL_true"].sum()
            i, r = score_fixed_shape(seq["left"], K, DIST_REAL, l)
            li.append(i)
            lr.append(r)
        rows.append({"trial": t,
                     "spearman_ideal_vs_raw": spearman_safe(li, lr),
                     "ideal_argmin_q": float(grid[::8][int(np.nanargmin(li))]),
                     "raw_argmin_q": float(grid[::8][int(np.nanargmin(lr))])})
    write_csv(RAW / "scoring_consistency_trials.csv.gz", rows)
    sp = float(np.nanmedian([r["spearman_ideal_vs_raw"] for r in rows]))
    ok = sp > 0.8
    res = {"median_spearman_ideal_vs_raw": sp,
           "ideal_argmin_q_median": float(np.nanmedian(
               [r["ideal_argmin_q"] for r in rows])),
           "raw_argmin_q_median": float(np.nanmedian(
               [r["raw_argmin_q"] for r in rows])),
           "verdict": "PASS" if ok
                      else "DISTORTION_SCORE_CONVENTION_MISMATCH"}
    write_json(SUM / "scoring_consistency.json", res)
    return ok, res


def main() -> None:
    gates, results = {}, {}
    for name, fn in (("G0_camera_roundtrip", g0_camera_roundtrip),
                     ("G1_true_geometry", g1_true_geometry),
                     ("G2_bone_fitter", g2_bone_fitter),
                     ("G3_determinism", g3_determinism),
                     ("G4_scoring_consistency", g4_scoring_consistency)):
        ok, res = fn()
        gates[name] = ok
        results[name] = res
        print(f"  {name:26s} {'PASS' if ok else 'FAIL'}   "
              f"{res.get('verdict', '')}")
    allok = all(gates.values())
    write_json(SUM / "gates.json", {
        "gates": gates, "results": results,
        "all_passed": allok,
        "consequence": "the corrected full sweep runs only if all gates pass; "
                       "otherwise FULL_SWEEP_NOT_RUN",
    })
    write_csv(__import__("common").TAB / "pass_fail_gates.csv",
              [{"gate": k, "passed": int(v),
                "verdict": results[k].get("verdict", "")}
               for k, v in gates.items()])
    print(f"\nALL GATES {'PASS' if allok else 'FAIL'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
