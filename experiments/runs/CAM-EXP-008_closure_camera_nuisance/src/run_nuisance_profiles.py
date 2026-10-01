"""CAM-EXP-008 closure: the oracle camera-nuisance diagnostic that was skipped.

Question: was CAM-008's hand profile shallow and biased toward q = 1.184
BECAUSE the scene-estimated principal point / distortion / aspect ratio were
wrong?

Everything is held identical to CAM-008: the same sequence-camera units, the
same frame IDs, the same dataset-provided 2D, the same reference 3D, the same
candidate focal grid. The ONLY thing that changes is which camera nuisance
parameters the hand reprojection uses.

    N0  scene cx,cy   scene k1,k2   scene fy/fx     (= CAM-008 primary)
    N1  PROVIDED cx,cy  scene dist    scene ratio
    N2  scene cx,cy     PROVIDED dist scene ratio
    N3  PROVIDED cx,cy  PROVIDED dist scene ratio
    N4  PROVIDED cx,cy  PROVIDED dist PROVIDED ratio

N1-N4 are ORACLE_DIAGNOSTIC_ONLY. The provided focal MAGNITUDE is never used.

CAM-EXP-008's own outputs are read-only; nothing here writes into that run.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
RUN = HERE.parents[1]
C8 = RUN.parent / "CAM-EXP-008_offline_sequence_scene_hand_focal_fusion"
sys.path.insert(0, str(C8 / "src"))
sys.path.insert(0, str(RUN / "src"))

from common import (CACHE, MANIFESTS, RAW, REPO, SUM, fnum, q_grid,  # noqa: E402
                    read_csv, write_csv, write_json)
from compute_hand_profiles import (profile_one, rebuild_hand,  # noqa: E402
                                   scene_params, view_template)

CAND = MANIFESTS / "cam_exp_008_candidate_frames_v1.csv.gz"
REF_DIR = C8 / "cache" / "reference_3d"
SCENE_DIR = C8 / "cache" / "scene_pred"
OUT_ROOT = RUN / "cache" / "nuisance_profiles"

CONDITIONS = {
    "N1": dict(pp=True, dist=False, ratio=False),
    "N2": dict(pp=False, dist=True, ratio=False),
    "N3": dict(pp=True, dist=True, ratio=False),
    "N4": dict(pp=True, dist=True, ratio=True),
}


def provided_cameras():
    """Provided camera metadata. FOCAL MAGNITUDE IS DELIBERATELY NOT RETURNED."""
    sys.path.insert(0, str(REPO))
    from experiments.src.datasets.gigahands import takes
    meta = {}
    for t in takes():
        for name, c in t.cameras.items():
            meta[(t.name, name)] = {
                "cx": float(c.K[0, 2]), "cy": float(c.K[1, 2]),
                "k1": float(c.dist[0]), "k2": float(c.dist[1]),
                "r_fy": float(c.K[1, 1] / c.K[0, 0]),
            }
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=list(CONDITIONS))
    args = ap.parse_args()
    cfg = CONDITIONS[args.condition]

    prov = provided_cameras()
    cand = read_csv(CAND)
    by = defaultdict(list)
    for r in cand:
        by[(r["sequence"], r["camera"])].append(int(r["frame"]))
    views = sorted(by)

    outdir = OUT_ROOT / args.condition
    outdir.mkdir(parents=True, exist_ok=True)
    grid = q_grid()
    meta = []

    for vi, (seq, cam) in enumerate(views, 1):
        out = outdir / f"{seq}__{cam}.npz"
        if out.exists():
            continue
        ref = REF_DIR / f"{seq}__{cam}.npz"
        sc = SCENE_DIR / f"{seq}__{cam}.csv.gz"
        if not ref.exists() or not sc.exists() or (seq, cam) not in prov:
            continue
        sp = scene_params([r for r in read_csv(sc) if r["success"] == "1"])
        if sp is None:
            continue
        pv = prov[(seq, cam)]

        # the anchor f_scene is UNCHANGED, so the candidate grid is identical
        cx = pv["cx"] if cfg["pp"] else sp["cx"]
        cy = pv["cy"] if cfg["pp"] else sp["cy"]
        k1 = pv["k1"] if cfg["dist"] else sp["k1"]
        k2 = pv["k2"] if cfg["dist"] else sp["k2"]
        rfy = pv["r_fy"] if cfg["ratio"] else sp["r_fy"]

        K_list = [np.array([[q * sp["f_scene"], 0, cx],
                            [0, rfy * q * sp["f_scene"], cy],
                            [0, 0, 1.0]]) for q in grid]
        dist = [k1, k2, 0.0, 0.0]

        store = dict(np.load(ref))
        frames = sorted(by[(seq, cam)])
        templates = {s: view_template(store, frames, s)
                     for s in ("left", "right")}

        curves, keys = [], []
        for fr in frames:
            for side in ("left", "right"):
                k = f"{fr}|{side}"
                if k + "|xyz" not in store or templates[side] is None:
                    continue
                use = store[k + "|use"]
                if use.sum() < 12:
                    continue
                H = rebuild_hand(np.asarray(store[k + "|xyz"], float),
                                 templates[side])
                if H is None:
                    continue
                uv = np.asarray(store[k + "|uv"], float)
                m = use & np.isfinite(uv).all(1)
                if m.sum() < 12:
                    continue
                c = profile_one(H[m], uv[m], K_list, dist)
                if np.isfinite(c).any():
                    curves.append(c.astype(np.float32))
                    keys.append(f"{fr}|{side}|{fr}")

        np.savez_compressed(
            out, curves=np.asarray(curves, np.float32),
            keys=np.array(keys), grid=grid.astype(np.float32),
            f_scene=sp["f_scene"], r_fy=rfy, cx=cx, cy=cy, k1=k1, k2=k2,
            scene_frames=sp["frames"].astype(np.int32),
            scene_fx=sp["fx"].astype(np.float32),
            scene_fy=sp["fy"].astype(np.float32),
            log_fx=sp["log_fx"].astype(np.float32))
        meta.append({"sequence": seq, "camera": cam,
                     "condition": args.condition,
                     "n_hand_observations": len(curves),
                     "cx": round(cx, 3), "cy": round(cy, 3),
                     "k1": round(k1, 5), "r_fy": round(rfy, 6)})
        if vi % 20 == 0:
            print(f"  {vi}/{len(views)} views ({args.condition})", flush=True)

    if meta:
        write_csv(RAW / f"nuisance_profile_meta_{args.condition}.csv", meta)
    print(f"done ({args.condition}): {len(meta)} views written")


if __name__ == "__main__":
    main()
