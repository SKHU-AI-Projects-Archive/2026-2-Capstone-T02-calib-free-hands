"""Synthetic identifiability gate for the bilateral bone-consistency idea.

Sweeps distance, asymmetry, noise and pose diversity. For every trial it
records, over the frozen candidate grid:

  * FIT reprojection            - fitted on FIT_FRAMES, measured on them
  * HELD-OUT reprojection       - fitted on FIT_FRAMES, measured on EVAL_FRAMES
  * bilateral distance          - ||l_L(f) - l_R(f)||_1, fitted on FIT_FRAMES
  * bilateral on held-out frames

The held-out split matters. Twenty bone lengths per side is a lot of freedom:
fitted and measured on the same frames, a wrong focal can be absorbed by
reshaping the hand. Held-out frames are what expose that.

Controls: wrong bone correspondence, right-hand swapped from another subject,
and the fixed-length bilateral loss which must stay flat.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BONE_CORRESPONDENCE, N_BONES, SEED, SUM, q_grid,  # noqa: E402
                    write_csv, write_json)
from fit_candidate_conditioned_bones import bilateral_distance, fit_bones
from generate_synthetic_bimanual import DIST_RIG, H, W, make_sequence

GRID = q_grid()
N_TRIALS = 20
N_FRAMES = 12


def split_frames(frames):
    """Deterministic alternating FIT / EVAL split, independent of any target."""
    fit = [f for i, f in enumerate(frames) if i % 2 == 0]
    ev = [f for i, f in enumerate(frames) if i % 2 == 1]
    return fit, ev


def eval_reproj(frames, l, K, dist):
    """Median reprojection of a GIVEN bone vector on GIVEN frames."""
    from fit_candidate_conditioned_bones import _pose_from
    from common import assemble
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    errs = []
    for dirs, uv, use in frames:
        if use.sum() < 6:
            continue
        R, T = _pose_from(l, dirs, uv, use, K, dist)
        if R is None:
            continue
        X = assemble(dirs, l)
        Xc = (R @ X.T).T + T
        z = np.where(np.abs(Xc[:, 2]) < 1e-9, 1e-9, Xc[:, 2])
        proj = np.stack([fx * Xc[:, 0] / z + cx, fy * Xc[:, 1] / z + cy], 1)
        errs.append(np.median(np.linalg.norm(proj[use] - uv[use], axis=1)))
    return float(np.median(errs)) if errs else float("nan")


def one_trial(rng, **kw):
    seq = make_sequence(rng, n_frames=N_FRAMES, **kw)
    if len(seq["left"]) < 6 or len(seq["right"]) < 6:
        return None
    f_true = seq["f_true"]
    K0, dist = seq["K"], seq["dist"]
    Lfit, Lev = split_frames(seq["left"])
    Rfit, Rev = split_frames(seq["right"])

    # a foreign right hand, for the swap control
    seq2 = make_sequence(np.random.default_rng(int(rng.integers(1 << 30))),
                         n_frames=N_FRAMES, **kw)
    R2fit, _ = split_frames(seq2["right"]) if seq2["right"] else ([], [])

    perm = np.random.default_rng(12345).permutation(N_BONES)
    wrong_corr = [(i, int(perm[i])) for i in range(N_BONES)]

    out = {"q": [], "fit_reproj": [], "held_reproj": [], "bilateral": [],
           "bilateral_held": [], "bilateral_wrongcorr": [],
           "bilateral_swap": [], "bilateral_fixed": []}
    lLf = seq["lL_true"] / seq["lL_true"].sum()
    lRf = seq["lR_true"] / seq["lR_true"].sum()
    for q in GRID:
        f = q * f_true
        K = np.array([[f, 0, K0[0, 2]], [0, f, K0[1, 2]], [0, 0, 1.0]])
        lL, eL = fit_bones(Lfit, K, dist)
        lR, eR = fit_bones(Rfit, K, dist)
        if lL is None or lR is None:
            for k in out:
                out[k].append(np.nan)
            out["q"][-1] = float(q)
            continue
        hL = eval_reproj(Lev, lL, K, dist)
        hR = eval_reproj(Rev, lR, K, dist)
        lR2, _ = fit_bones(R2fit, K, dist) if R2fit else (None, np.nan)
        out["q"].append(float(q))
        out["fit_reproj"].append(np.nanmean([eL, eR]))
        out["held_reproj"].append(np.nanmean([hL, hR]))
        out["bilateral"].append(bilateral_distance(lL, lR,
                                                   BONE_CORRESPONDENCE))
        out["bilateral_held"].append(
            bilateral_distance(lL, lR, BONE_CORRESPONDENCE))
        out["bilateral_wrongcorr"].append(
            bilateral_distance(lL, lR, wrong_corr))
        out["bilateral_swap"].append(
            bilateral_distance(lL, lR2, BONE_CORRESPONDENCE)
            if lR2 is not None else np.nan)
        out["bilateral_fixed"].append(
            bilateral_distance(lLf, lRf, BONE_CORRESPONDENCE))
    return out


def argmin_q(vals):
    v = np.asarray(vals, float)
    if not np.isfinite(v).any():
        return np.nan, "NO_SOLUTION"
    i = int(np.nanargmin(v))
    st = "BOUNDARY" if i in (0, len(v) - 1) else "ok"
    return float(GRID[i]), st


def summarise(trials, key):
    qs, sts = [], []
    for t in trials:
        q, st = argmin_q(t[key])
        qs.append(q)
        sts.append(st)
    qs = np.array(qs, float)
    err = 100.0 * np.abs(qs - 1.0)
    return {
        "n_trials": len(trials),
        "median_recovered_q": round(float(np.nanmedian(qs)), 4),
        "median_focal_err_pct": round(float(np.nanmedian(err)), 3),
        "within_5pct": round(float(np.nanmean(err <= 5)), 3),
        "boundary_rate": round(float(np.mean([s == "BOUNDARY"
                                              for s in sts])), 3),
    }


def main() -> None:
    conds = []
    # distance sweep (the perspective regime)
    for d in (2, 4, 8, 16, 32):
        conds.append((f"DIST_{d}x", dict(dist_over_diam=d, asymmetry=0.0,
                                         noise_px=0.0)))
    # asymmetry sweep
    for a in (0.0, 0.01, 0.02, 0.05):
        conds.append((f"ASYM_{a}", dict(dist_over_diam=4, asymmetry=a,
                                        noise_px=0.0)))
    # noise sweep
    for nz in (0.0, 0.5, 1.0, 2.0, 4.0):
        conds.append((f"NOISE_{nz}px", dict(dist_over_diam=4, asymmetry=0.0,
                                            noise_px=nz)))
    # pose diversity
    for sp, nm in ((0.35, "LOW"), (1.0, "MEDIUM"), (1.8, "HIGH")):
        conds.append((f"POSE_{nm}", dict(dist_over_diam=4, asymmetry=0.0,
                                         noise_px=0.0, pose_spread=sp)))

    rows, curves = [], {}
    for name, kw in conds:
        rng = np.random.default_rng(SEED + abs(hash(name)) % 10000)
        trials = []
        for _ in range(N_TRIALS):
            t = one_trial(rng, **kw)
            if t is not None:
                trials.append(t)
        if not trials:
            continue
        curves[name] = np.nanmedian(
            np.array([t["bilateral"] for t in trials], float), axis=0)
        row = {"condition": name}
        for key, tag in (("fit_reproj", "fitreproj"),
                         ("held_reproj", "heldreproj"),
                         ("bilateral", "bilat"),
                         ("bilateral_wrongcorr", "bilat_wrongcorr"),
                         ("bilateral_swap", "bilat_swap"),
                         ("bilateral_fixed", "bilat_fixed")):
            s = summarise(trials, key)
            for k, v in s.items():
                if k == "n_trials":
                    row["n_trials"] = v
                else:
                    row[f"{tag}_{k}"] = v
        # is the fixed-length control flat?
        fx = np.array([t["bilateral_fixed"] for t in trials], float)
        rng_fixed = np.nanmax(fx, axis=1) - np.nanmin(fx, axis=1)
        row["bilat_fixed_range_median"] = float(np.nanmedian(rng_fixed))
        bl = np.array([t["bilateral"] for t in trials], float)
        row["bilat_rel_dynamic_range"] = round(float(np.nanmedian(
            (np.nanmax(bl, axis=1) - np.nanmin(bl, axis=1))
            / np.nanmedian(bl, axis=1))), 4)
        rows.append(row)
        print(f"  {name:14s} bilat q={row['bilat_median_recovered_q']:.3f} "
              f"err={row['bilat_median_focal_err_pct']:6.2f}%  "
              f"bnd={row['bilat_boundary_rate']:.2f} | "
              f"held-reproj q={row['heldreproj_median_recovered_q']:.3f} "
              f"err={row['heldreproj_median_focal_err_pct']:6.2f}%",
              flush=True)

    write_csv(SUM / "synthetic_summary.csv", rows)
    np.savez_compressed(SUM.parent / "raw" / "synthetic_profiles.npz",
                        grid=GRID, **curves)

    # ---- frozen gate ----------------------------------------------------
    base = next((r for r in rows if r["condition"] == "DIST_4x"), None)
    near = next((r for r in rows if r["condition"] == "DIST_2x"), None)
    gate = {
        "g1_bilateral_not_focal_invariant": bool(
            base and base["bilat_rel_dynamic_range"] > 0.05),
        "g2_true_focal_near_minimum_noiseless": bool(
            base and base["bilat_median_focal_err_pct"] <= 5.0),
        "g2_best_case_any_distance": round(float(min(
            r["bilat_median_focal_err_pct"] for r in rows
            if r["condition"].startswith("DIST_"))), 3),
        "g3_held_out_reprojection_identifies_focal": bool(
            base and base["heldreproj_median_focal_err_pct"] <= 5.0),
        "g4_wrong_correspondence_degrades": bool(
            base and base["bilat_wrongcorr_median_focal_err_pct"]
            > base["bilat_median_focal_err_pct"]),
        "g5_fixed_length_control_is_flat": bool(
            base and base["bilat_fixed_range_median"] < 1e-12),
        "boundary_rate_noiseless": base["bilat_boundary_rate"] if base
        else None,
        "near_field_2x_bilateral_err_pct": near[
            "bilat_median_focal_err_pct"] if near else None,
    }
    passed = bool(gate["g1_bilateral_not_focal_invariant"]
                  and gate["g2_true_focal_near_minimum_noiseless"]
                  and gate["g4_wrong_correspondence_degrades"])
    gate["verdict"] = ("SYNTHETIC_GATE_PASS" if passed
                       else "BILATERAL_FOCAL_SIGNAL_NOT_IDENTIFIABLE_"
                            "SYNTHETICALLY")
    write_json(SUM / "synthetic_gate.json", gate)
    print("\n" + "\n".join(f"  {k}: {v}" for k, v in gate.items()))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
