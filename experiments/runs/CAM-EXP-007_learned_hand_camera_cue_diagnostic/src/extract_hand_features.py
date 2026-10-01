"""Build per-view hand features from the cached model outputs.

TARGET-BLIND BY CONSTRUCTION. This module imports nothing that touches the
target, the reference focal or any AnyCalib prediction. Grep it: there is no
reference to VIEW_TARGETS anywhere.

Aggregation, exactly as frozen:
  hands -> frame : median over the hands in that frame (mean for vectors), so
                   a frame counts once whether it holds one hand or two
  frames -> view : median and IQR, plus the temporal-stability statistics
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CACHE, MANIFESTS, MIN_FRAMES_WITH_HAND, N_FRAMES, RAW,  # noqa: E402
                    SUM, read_csv, write_csv, write_json)

FRAMES = MANIFESTS / "cam_exp_007_frame_manifest_v1.csv.gz"
NPZ_DIR = CACHE / "hand_outputs"
OUT_VIEW = RAW / "view_hand_features.csv.gz"
OUT_LATENT = CACHE / "view_latent.npz"

# MANO kinematic chain used for bone-ratio statistics
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
         (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
TIPS = [4, 8, 12, 16, 20]


def iqr(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, 75) - np.percentile(a, 25)) if a.size else np.nan


def safe(fn, a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(fn(a)) if a.size else np.nan


def hand_scalars(z, p, W, H):
    """Focal-free scalar features for one detected hand."""
    f = {}
    bbox = z.get(p + "_bbox")
    if bbox is not None:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
        f["bbox_area_norm"] = (w * h) / (W * H)
        f["bbox_w_over_W"] = w / W
        f["bbox_h_over_H"] = h / H
        f["bbox_aspect"] = w / h
        f["bbox_cx_norm"] = ((x1 + x2) / 2) / W
        f["bbox_cy_norm"] = ((y1 + y2) / 2) / H
    bs = z.get(p + "_box_size")
    if bs is not None:
        f["box_size_norm"] = float(np.asarray(bs).reshape(-1)[0]) / max(W, H)
    sc = z.get(p + "_score")
    if sc is not None:
        f["det_score"] = float(sc)

    pc = z.get(p + "_pred_cam")
    if pc is not None:
        pc = np.asarray(pc, float).reshape(-1)
        is_right = float(z.get(p + "_is_right", 1.0))
        # mirror tx the same way the wrapper does, so left and right hands
        # share one convention. This is the model's own convention, not an
        # extra transform invented here.
        mult = 2.0 * is_right - 1.0
        f["pred_cam_s"] = pc[0]
        f["pred_cam_tx"] = mult * pc[1]
        f["pred_cam_ty"] = pc[2]

    kp = z.get(p + "_kp3d_rootrel")
    if kp is not None:
        k = np.asarray(kp, float)
        k = k - k[0]                      # enforce root-relative
        d = np.linalg.norm(k[:, None, :] - k[None, :, :], axis=-1)
        diam = float(d.max())
        f["hand_diameter"] = diam
        f["depth_extent"] = float(k[:, 2].max() - k[:, 2].min())
        f["lateral_extent"] = float(max(k[:, 0].max() - k[:, 0].min(),
                                        k[:, 1].max() - k[:, 1].min()))
        f["depth_over_diameter"] = (f["depth_extent"] / diam
                                    if diam > 0 else np.nan)
        # palm plane normal, z component: how face-on the hand is
        try:
            palm = k[[0, 5, 17]]
            n = np.cross(palm[1] - palm[0], palm[2] - palm[0])
            nn = np.linalg.norm(n)
            f["palm_normal_z"] = float(abs(n[2] / nn)) if nn > 0 else np.nan
        except Exception:                                    # noqa: BLE001
            f["palm_normal_z"] = np.nan
        tips = k[TIPS]
        f["finger_spread"] = float(np.linalg.norm(
            tips[:, None, :] - tips[None, :, :], axis=-1).max() / diam) \
            if diam > 0 else np.nan
        bl = np.array([np.linalg.norm(k[a] - k[b]) for a, b in BONES])
        bl = bl / (diam if diam > 0 else 1.0)
        f["bone_ratio_mean"] = float(bl.mean())
        f["bone_ratio_std"] = float(bl.std())

    beta = z.get(p + "_mano_shape")
    if beta is not None:
        for i, v in enumerate(np.asarray(beta, float).reshape(-1)[:10]):
            f[f"mano_beta_{i}"] = float(v)
    pose = z.get(p + "_mano_pose")
    if pose is not None:
        pa = np.asarray(pose, float).reshape(-1)
        f["global_orient_aa_norm"] = float(np.linalg.norm(pa[:3]))
        f["hand_pose_aa_norm"] = float(np.linalg.norm(pa[3:]))
        f["hand_pose_aa_mean_abs"] = float(np.abs(pa[3:]).mean())
    return f


def main() -> None:
    frames = read_csv(FRAMES)
    by_view = defaultdict(list)
    for r in frames:
        by_view[(r["sequence"], r["camera"])].append(int(r["frame"]))

    idx = {(r["sequence"], r["camera"], int(r["frame"])): r
           for r in read_csv(RAW / "hand_inference_index.csv.gz")}

    view_rows, latents, coverage = [], {}, []
    for (seq, cam), fr in sorted(by_view.items()):
        npz = NPZ_DIR / f"{seq}__{cam}.npz"
        z = dict(np.load(npz)) if npz.exists() else {}
        per_frame, per_frame_lat = [], []
        n_hands_list, sides, n_ok = [], [], 0

        for f in sorted(fr):
            meta = idx.get((seq, cam, f), {})
            W = int(meta.get("image_width") or 1280)
            H = int(meta.get("image_height") or 720)
            ks = sorted({int(k.split("_")[1]) for k in z
                         if k.startswith(f"{f}_") and k.endswith("_bbox")})
            n_hands_list.append(len(ks))
            if not ks:
                per_frame.append({})
                continue
            n_ok += 1
            hs = [hand_scalars(z, f"{f}_{k}", W, H) for k in ks]
            keys = set().union(*[set(h) for h in hs])
            # hands -> frame: median, so a two-hand frame still counts once
            per_frame.append({k: safe(np.median, [h.get(k, np.nan)
                                                  for h in hs])
                              for k in keys})
            for k in ks:
                sides.append(float(z.get(f"{f}_{k}_is_right", np.nan)))
            lat = [z[f"{f}_{k}_latent"] for k in ks
                   if f"{f}_{k}_latent" in z]
            if lat:
                per_frame_lat.append(np.mean(np.stack(lat, 0), axis=0))

        eligible = n_ok >= MIN_FRAMES_WITH_HAND
        coverage.append({
            "sequence": seq, "camera": cam,
            "n_frames": len(fr), "n_frames_with_hand": n_ok,
            "availability_rate": round(n_ok / max(len(fr), 1), 4),
            "mean_hands_per_frame": round(float(np.mean(n_hands_list)), 4),
            "has_latent": int(bool(per_frame_lat)),
            "hand_feature_eligible": int(eligible),
            "reason_excluded": "" if eligible
            else f"only {n_ok}/{len(fr)} frames with a usable hand",
        })

        row = {"sequence": seq, "camera": cam,
               "n_frames_with_hand": n_ok,
               "availability_rate": n_ok / max(len(fr), 1),
               "mean_hands_per_frame": float(np.mean(n_hands_list)),
               "n_hands_std": float(np.std(n_hands_list)),
               "side_consistency": (float(max(np.nanmean(sides),
                                              1 - np.nanmean(sides)))
                                    if sides else np.nan),
               "hand_feature_eligible": int(eligible)}

        keys = sorted(set().union(*[set(d) for d in per_frame]) if per_frame
                      else set())
        for k in keys:
            vals = [d.get(k, np.nan) for d in per_frame]
            row[f"{k}__median"] = safe(np.median, vals)
            row[f"{k}__iqr"] = iqr(vals)
        # F3 temporal stability
        for k in ("pred_cam_s", "pred_cam_tx", "pred_cam_ty",
                  "bbox_area_norm", "box_size_norm", "hand_diameter",
                  "depth_over_diameter", "det_score", "hand_pose_aa_norm"):
            if k in keys:
                row[f"{k}__std"] = safe(np.std, [d.get(k, np.nan)
                                                 for d in per_frame])
        bstd = [safe(np.std, [d.get(f"mano_beta_{i}", np.nan)
                              for d in per_frame]) for i in range(10)]
        row["beta_std_mean"] = safe(np.mean, bstd)
        view_rows.append(row)

        if per_frame_lat:
            latents[f"{seq}|{cam}"] = np.mean(
                np.stack(per_frame_lat, 0), axis=0).astype(np.float32)

    write_csv(OUT_VIEW, view_rows)
    write_csv(SUM / "hand_feature_coverage.csv", coverage)
    if latents:
        np.savez_compressed(OUT_LATENT, **latents)

    elig = sum(c["hand_feature_eligible"] for c in coverage)
    write_json(SUM / "feature_extraction_meta.json", {
        "views_total": len(coverage),
        "views_hand_feature_eligible": elig,
        "views_with_latent": len(latents),
        "latent_dim": int(next(iter(latents.values())).shape[0])
        if latents else 0,
        "min_frames_with_hand_required": MIN_FRAMES_WITH_HAND,
        "n_frames_per_view": N_FRAMES,
        "n_view_feature_columns": len(view_rows[0]) - 3 if view_rows else 0,
        "target_blind": True,
    })
    print(f"views {len(coverage)}, eligible {elig}, "
          f"latent views {len(latents)}, "
          f"feature columns {len(view_rows[0]) if view_rows else 0}")


if __name__ == "__main__":
    main()
