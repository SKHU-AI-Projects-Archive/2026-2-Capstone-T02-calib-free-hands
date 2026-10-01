"""Build the SEQUENCE_CONSISTENT_REFERENCE_HAND and its focal profiles.

TARGET-BLIND. Reads no reference focal and no focal error.

Per (sequence, camera, anatomical side):
  1. bone lengths are the median over that view's own reference
     reconstructions - one worker's bones do not change between frames;
  2. the template is normalised by its own bone-length sum, so NO absolute
     hand-size prior is used (factories differ in hand size);
  3. each frame keeps its own bone DIRECTIONS and is re-assembled along the
     kinematic tree with those shared normalised lengths.

The template for a view is built only from that view's own
OTHER_CAMERA_ONLY reconstructions, which never saw the target camera's 2D.
Borrowing another view's reconstruction would re-admit the target camera.

Then, for every candidate focal f = q * f_scene, a PnP reprojection profile is
computed using the SCENE-estimated principal point and radial distortion.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CACHE, MANIFESTS, RAW, SUM, fnum, q_grid, read_csv,  # noqa: E402
                    write_csv, write_json)

CAND = MANIFESTS / "cam_exp_008_candidate_frames_v1.csv.gz"
REF_DIR = CACHE / "reference_3d"
SCENE_DIR = CACHE / "scene_pred"
OUT_DIR = CACHE / "hand_profiles"

# 21-joint hand kinematic tree, matching the dataset convention verified by
# the CAM-EXP-006 manual-QC overlays (parent of j is PARENT[j], root = 0)
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
         (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
PARENT = {c: p for p, c in BONES}
ORDER = [c for _, c in BONES]          # parents always precede children


def scene_params(rows):
    """Sequence-level scene nuisance estimates. Identical for S0 and S1."""
    fx = np.array([fnum(r["pred_fx"]) for r in rows])
    fy = np.array([fnum(r["pred_fy"]) for r in rows])
    ok = np.isfinite(fx) & (fx > 0)
    if ok.sum() == 0:
        return None
    fx, fy = fx[ok], fy[ok]
    frames = np.array([int(r["frame"]) for r in rows])[ok]
    return {
        "frames": frames, "fx": fx, "fy": fy,
        "f_scene": float(np.median(fx)),
        "r_fy": float(np.median(fy / fx)),
        "cx": float(np.median([fnum(r["pred_cx"]) for r in rows])),
        "cy": float(np.median([fnum(r["pred_cy"]) for r in rows])),
        "k1": float(np.median([fnum(r["k1"], 0.0) for r in rows])),
        "k2": float(np.median([fnum(r["k2"], 0.0) for r in rows])),
        "log_fx": np.log(fx),
        "n_scene_frames": int(ok.sum()),
    }


def rebuild_hand(xyz, lengths):
    """Re-assemble a hand from its own bone directions and shared lengths."""
    out = np.zeros((21, 3), float)
    for j in ORDER:
        p = PARENT[j]
        d = xyz[j] - xyz[p]
        n = np.linalg.norm(d)
        if n < 1e-9 or not np.isfinite(n):
            return None
        out[j] = out[p] + (d / n) * lengths[(p, j)]
    return out


def view_template(store, frames, side):
    """Median bone lengths for one view and side, normalised to unit sum."""
    per_bone = defaultdict(list)
    for fr in frames:
        k = f"{fr}|{side}"
        if k + "|xyz" not in store:
            continue
        x = store[k + "|xyz"]
        use = store[k + "|use"]
        for (p, c) in BONES:
            if use[p] and use[c]:
                d = np.linalg.norm(x[c] - x[p])
                if np.isfinite(d) and d > 0:
                    per_bone[(p, c)].append(d)
    if len(per_bone) < len(BONES):
        return None
    L = {b: float(np.median(v)) for b, v in per_bone.items()}
    tot = sum(L.values())
    if not np.isfinite(tot) or tot <= 0:
        return None
    return {b: v / tot for b, v in L.items()}      # scale removed


def profile_one(xyz, uv, K_list, dist):
    """Reprojection error in px at every candidate focal, for one hand."""
    import cv2
    obj = np.ascontiguousarray(xyz.reshape(-1, 1, 3), np.float64)
    img = np.ascontiguousarray(uv.reshape(-1, 1, 2), np.float64)
    d = np.asarray(dist, np.float64).reshape(1, -1)
    out = np.full(len(K_list), np.nan)
    for i, K in enumerate(K_list):
        ok = False
        for flag in (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_IPPE,
                     cv2.SOLVEPNP_ITERATIVE):
            try:
                ok, rvec, tvec = cv2.solvePnP(obj, img, K, d, flags=flag)
            except cv2.error:
                ok = False
            if ok:
                break
        if not ok:
            continue
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, d)
        out[i] = float(np.median(np.linalg.norm(
            proj.reshape(-1, 2) - uv, axis=1)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrong-frame", action="store_true",
                    help="C1 control: pair 2D with a temporally offset hand")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tag = "wrongframe" if args.wrong_frame else "real"
    outdir = OUT_DIR / tag
    outdir.mkdir(parents=True, exist_ok=True)

    cand = read_csv(CAND)
    by = defaultdict(list)
    for r in cand:
        by[(r["sequence"], r["camera"])].append(int(r["frame"]))
    views = sorted(by)
    if args.limit:
        views = views[:args.limit]

    grid = q_grid()
    meta_rows = []
    for vi, (seq, cam) in enumerate(views, 1):
        out = outdir / f"{seq}__{cam}.npz"
        if out.exists():
            continue
        ref = REF_DIR / f"{seq}__{cam}.npz"
        sc = SCENE_DIR / f"{seq}__{cam}.csv.gz"
        if not ref.exists() or not sc.exists():
            continue
        srows = [r for r in read_csv(sc) if r["success"] == "1"]
        sp = scene_params(srows)
        if sp is None:
            continue
        store = dict(np.load(ref))
        frames = sorted(by[(seq, cam)])

        K_list = []
        for q in grid:
            f = q * sp["f_scene"]
            K_list.append(np.array([[f, 0, sp["cx"]],
                                    [0, sp["r_fy"] * f, sp["cy"]],
                                    [0, 0, 1.0]]))
        dist = [sp["k1"], sp["k2"], 0.0, 0.0]

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
                src_fr = fr
                if args.wrong_frame:
                    # deterministic temporal offset: same sequence, same side,
                    # a different frame. Hand size and person are preserved;
                    # pose correspondence is destroyed.
                    cands = [g for g in frames
                             if f"{g}|{side}|xyz" in store and g != fr]
                    if not cands:
                        continue
                    src_fr = cands[(frames.index(fr) + len(cands) // 2)
                                   % len(cands)]
                    k2 = f"{src_fr}|{side}"
                    u2 = store[k2 + "|use"]
                    both = use & u2
                    if both.sum() < 12:
                        continue
                    xyz_raw = store[k2 + "|xyz"]
                    sel = both
                else:
                    xyz_raw = store[k + "|xyz"]
                    sel = use
                H = rebuild_hand(np.asarray(xyz_raw, float),
                                 templates[side])
                if H is None:
                    continue
                uv = np.asarray(store[k + "|uv"], float)
                m = sel & np.isfinite(uv).all(1)
                if m.sum() < 12:
                    continue
                c = profile_one(H[m], uv[m], K_list, dist)
                if np.isfinite(c).any():
                    curves.append(c.astype(np.float32))
                    keys.append(f"{fr}|{side}|{src_fr}")

        np.savez_compressed(
            out, curves=np.asarray(curves, np.float32),
            keys=np.array(keys), grid=grid.astype(np.float32),
            f_scene=sp["f_scene"], r_fy=sp["r_fy"], cx=sp["cx"], cy=sp["cy"],
            k1=sp["k1"], k2=sp["k2"], log_fx=sp["log_fx"].astype(np.float32),
            scene_frames=sp["frames"].astype(np.int32),
            scene_fx=sp["fx"].astype(np.float32),
            scene_fy=sp["fy"].astype(np.float32))
        meta_rows.append({"sequence": seq, "camera": cam,
                          "n_hand_observations": len(curves),
                          "n_frames": len(frames),
                          "f_scene": round(sp["f_scene"], 3),
                          "n_scene_frames": sp["n_scene_frames"]})
        if vi % 5 == 0:
            print(f"  {vi}/{len(views)} views ({tag})", flush=True)

    if meta_rows:
        write_csv(RAW / f"hand_profile_meta_{tag}.csv", meta_rows)
        write_json(SUM / f"hand_profile_meta_{tag}.json", {
            "condition": tag, "views": len(meta_rows),
            "hand_observations": sum(r["n_hand_observations"]
                                     for r in meta_rows),
            "grid_points": len(grid), "target_blind": True,
        })
    print(f"done ({tag}): {len(meta_rows)} views written")


if __name__ == "__main__":
    main()
