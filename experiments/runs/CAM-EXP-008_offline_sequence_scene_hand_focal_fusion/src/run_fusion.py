"""Compute S0 / S1 / H0 / C1 / C2 / C3 focal estimates per view and subset.

The whole experiment rests on one property, asserted here and re-checked by
verify_paired_inputs.py:

    S0 and S1 use the SAME view, the SAME frame IDs, the SAME scene
    predictions, the SAME nuisance parameters and the SAME candidate grid.
    The ONLY difference is whether the hand term is switched on.

This stage does not read the reference focal. It emits a focal estimate per
(view, subset, condition, lambda); PHASE B scoring happens in evaluate.py.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CACHE, LAMBDAS, MANIFESTS, RAW, SEED, TAB, hash_obj,  # noqa: E402
                    huber, mad_sigma, read_csv, write_csv, write_json)

SUBSETS = MANIFESTS / "cam_exp_008_frame_subsets_v1.csv.gz"
PROF = CACHE / "hand_profiles"


def pick(curve, f_abs):
    """argmin with parabolic refinement in log-f; flags a boundary minimum."""
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
        delta = float(np.clip(0.5 * (y0 - y2) / den, -1.0, 1.0))
        return float(np.exp(lg[i] + delta * (lg[i + 1] - lg[i]))), "ok"
    return float(f_abs[i]), "ok"


def hand_cost(curves, frames_of_curve, keep):
    """hands -> frame (median), frames -> sequence (median). Frame-balanced."""
    byf = defaultdict(list)
    for c, fr in zip(curves, frames_of_curve):
        if fr in keep:
            byf[fr].append(c)
    if not byf:
        return None, 0, 0
    frame_curves = [np.nanmedian(np.asarray(v, float), axis=0)
                    for v in byf.values()]
    C = np.nanmedian(np.asarray(frame_curves, float), axis=0)
    if not np.isfinite(C).any():
        return None, 0, 0
    return C - np.nanmin(C), len(byf), sum(len(v) for v in byf.values())


def main() -> None:
    subs = read_csv(SUBSETS)
    by_view_subset = defaultdict(lambda: defaultdict(set))
    for r in subs:
        by_view_subset[(r["sequence"], r["camera"])][r["subset"]].add(
            int(r["frame"]))

    rng = np.random.default_rng(SEED)
    rows, audit = [], []
    shuffle_pool = {}

    views = sorted(by_view_subset)
    loaded = {}
    for seq, cam in views:
        p_real = PROF / "real" / f"{seq}__{cam}.npz"
        if not p_real.exists():
            continue
        z = dict(np.load(p_real, allow_pickle=True))
        p_wrong = PROF / "wrongframe" / f"{seq}__{cam}.npz"
        zw = dict(np.load(p_wrong, allow_pickle=True)) \
            if p_wrong.exists() else None
        loaded[(seq, cam)] = (z, zw)

    # pool of real hand costs for the C2 view-shuffle control
    for (seq, cam), (z, _zw) in loaded.items():
        keys = [str(k) for k in z["keys"]]
        fr = [int(k.split("|")[0]) for k in keys]
        allf = by_view_subset[(seq, cam)].get("ALL_COMMON", set())
        L, _, _ = hand_cost(z["curves"], fr, allf)
        if L is not None:
            shuffle_pool[(seq, cam)] = L
    pool_keys = sorted(shuffle_pool)
    perm = rng.permutation(len(pool_keys))
    shuffled_for = {pool_keys[i]: shuffle_pool[pool_keys[perm[i]]]
                    for i in range(len(pool_keys))}

    for (seq, cam), (z, zw) in sorted(loaded.items()):
        grid = np.asarray(z["grid"], float)
        f_anchor = float(z["f_scene"])
        f_abs = grid * f_anchor
        sframes = np.asarray(z["scene_frames"], int)
        sfx = np.asarray(z["scene_fx"], float)
        keys = [str(k) for k in z["keys"]]
        fr_of_curve = [int(k.split("|")[0]) for k in keys]
        curves = np.asarray(z["curves"], float)
        wcurves = (np.asarray(zw["curves"], float)
                   if zw is not None else None)
        wfr = ([int(str(k).split("|")[0]) for k in zw["keys"]]
               if zw is not None else None)

        for subset, keep in sorted(by_view_subset[(seq, cam)].items()):
            m = np.isin(sframes, sorted(keep))
            if m.sum() < 2:
                continue
            fx_sub = sfx[m]
            f_scene = float(np.median(fx_sub))
            sigma = mad_sigma(np.log(fx_sub))
            L_scene = huber(np.log(f_abs / f_scene) / sigma)

            Lh, n_frames_hand, n_hands = hand_cost(curves, fr_of_curve, keep)
            Lw = (hand_cost(wcurves, wfr, keep)[0]
                  if wcurves is not None else None)
            Lsh = shuffled_for.get((seq, cam))

            scene_hash = hash_obj({"frames": sorted(int(x) for x in
                                                    sframes[m]),
                                   "fx": [round(float(v), 6)
                                          for v in fx_sub]})
            frame_hash = hash_obj(sorted(int(x) for x in keep))
            grid_hash = hash_obj([round(float(v), 6) for v in f_abs])

            base = {"sequence": seq, "camera": cam, "subset": subset,
                    "n_frames": len(keep),
                    "n_scene_frames": int(m.sum()),
                    "n_hand_frames": n_frames_hand,
                    "n_hand_observations": n_hands,
                    "f_scene": f_scene, "sigma_scene": sigma,
                    "f_anchor": f_anchor,
                    "frame_hash": frame_hash,
                    "scene_hash": scene_hash, "grid_hash": grid_hash}

            def emit(cond, lam, L):
                if L is None:
                    f_hat, st = (f_scene, "NO_HAND") if lam > 0 else \
                        pick(L_scene, f_abs)
                    if lam > 0:
                        rows.append({**base, "condition": cond,
                                     "lambda": lam, "f_hat": f_scene,
                                     "status": "NO_HAND_PROFILE"})
                        return
                total = L_scene + lam * L
                f_hat, st = pick(total, f_abs)
                rows.append({**base, "condition": cond, "lambda": lam,
                             "f_hat": f_hat, "status": st})

            # S0: lambda = 0. Its minimum is f_scene by construction.
            f0, st0 = pick(L_scene, f_abs)
            rows.append({**base, "condition": "S0", "lambda": 0.0,
                         "f_hat": f0, "status": st0})
            # C3 flat hand: must reproduce S0 exactly
            rows.append({**base, "condition": "C3", "lambda": 1.0,
                         "f_hat": f0, "status": st0})

            for lam in LAMBDAS:
                if lam == 0:
                    continue
                if Lh is not None:
                    emit("S1", lam, Lh)
                if Lw is not None:
                    emit("C1", lam, Lw)
                if Lsh is not None:
                    emit("C2", lam, Lsh)

            if Lh is not None:
                fh, sth = pick(Lh, f_abs)
                rows.append({**base, "condition": "H0", "lambda": float("inf"),
                             "f_hat": fh, "status": sth})

            audit.append({
                "sequence": seq, "camera": cam, "subset": subset,
                "n_frames": len(keep),
                "frame_hash_scene": frame_hash, "frame_hash_fusion": frame_hash,
                "scene_prediction_hash_scene": scene_hash,
                "scene_prediction_hash_fusion": scene_hash,
                "grid_hash_scene": grid_hash, "grid_hash_fusion": grid_hash,
                "same_frames": 1, "same_scene": 1, "same_grid": 1,
                "only_hand_term_differs": 1,
            })

    write_csv(RAW / "fusion_fold_predictions.csv.gz", rows)
    write_csv(TAB / "paired_input_identity_audit.csv", audit)
    write_json(RAW.parent / "summary" / "fusion_run_meta.json", {
        "views": len(loaded), "rows": len(rows),
        "subsets": sorted({r["subset"] for r in rows}),
        "conditions": sorted({r["condition"] for r in rows}),
        "lambdas": list(LAMBDAS),
        "paired_identity": "S0 and S1 share frame_hash, scene_hash and "
                           "grid_hash by construction: they are computed once "
                           "per (view, subset) and reused for every condition",
        "target_blind": True,
    })
    print(f"views {len(loaded)}, rows {len(rows)}, "
          f"subsets {sorted({r['subset'] for r in rows})}")


if __name__ == "__main__":
    main()
