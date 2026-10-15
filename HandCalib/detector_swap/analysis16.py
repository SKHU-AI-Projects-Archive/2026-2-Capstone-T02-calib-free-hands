"""Diagnostics on top of 14's per-detector run: cost profiles (official and extended grid) per joint set, and 2D differences
between detectors on the same frames and hand. Profile minima are diagnostics, never f values."""
from __future__ import annotations

import numpy as np

from hand_focal import diagnostic as D
from hand_focal import estimate as E
from hand_focal.track import box_long


def grids(pcfg, cfg16, L):
    off = E.f_grid(pcfg["estimate"], L)
    lo, hi = cfg16["profiles"]["extended_ratio_range"]
    ext = np.geomspace(lo * L, hi * L, int(cfg16["profiles"]["extended_n"]))
    return off, ext


def stats(grid, c):
    ok = np.isfinite(c)
    if ok.sum() < 3:
        return {"interior": None, "at": "none", "min_at_px": None, "sharpness": None}
    k = int(np.nanargmin(c))
    at = "interior" if 0 < k < len(c) - 1 else ("lower" if k == 0 else "upper")
    lg = np.log(grid)
    side = [np.interp(lg[k] + d, lg[ok], c[ok], left=np.nan, right=np.nan) for d in (np.log(1.15), -np.log(1.15))]
    side = [s for s in side if np.isfinite(s)]
    return {"interior": at == "interior", "at": at, "min_at_px": float(grid[k]), "sharpness": float(min(side) / c[k]) if side and c[k] > 0 else None}


def profiles(run_dir, pcfg, cfg16):
    L0 = D.load_official(run_dir, pcfg)
    a, tr, W, H = L0["a"], L0["tr"], L0["W"], L0["H"]
    if "template" not in a:
        return {"failed": a.get("failed")}, L0
    L = float(max(W, H))
    off, ext = grids(pcfg, cfg16, L)
    J = pcfg["joints"]
    est_ids = list(J["sets"][J["estimate_set"]])
    T = np.array(a["template"])
    role = np.array(a["role"])
    used = [i for i in range(len(role)) if role[i] in ("front", "tilt")]
    cx, cy = a["principal_point"]
    bl = box_long(tr["box"])
    huber = pcfg["estimate"]["huber_rel"] * np.where(np.isfinite(bl), bl, 0.0)
    out = {"official_grid": off.tolist(), "extended_grid": ext.tolist(), "n_used": len(used), "sets": {}}
    for s in cfg16["profiles"]["joint_sets"]:
        ids = list(J["sets"][s])
        pos = [est_ids.index(j) for j in ids]
        res = {}
        for gname, g in (("official", off), ("extended", ext)):
            C, _ = D.frame_cost_matrix(T, tr["kp"], tr["mask"], used, pos, ids, g, cx, cy, huber)
            ok = np.isfinite(C).all(1)
            prof = C[ok].mean(0) if ok.sum() else np.full(len(g), np.nan)
            res[gname] = {"profile": prof.tolist(), "n_frames": int(ok.sum()), **stats(g, prof)}
        out["sets"][s] = res
    return out, L0


def diff2d(L0s: dict, pcfg, cfg16):
    """Pairs of detectors: per joint 2D difference (B - A) on frames where both track the same hand (target-box IoU >= min).
    Radial = component along the direction from A's hand centroid to A's joint (positive: B's point lies further out)."""
    from hand_focal.track import iou
    names = [n for n in L0s if L0s[n] is not None and "template" in L0s[n]["a"]]
    bins = cfg16["diff2d"]["theta_bins_deg"]
    est_ids = list(pcfg["joints"]["sets"][pcfg["joints"]["estimate_set"]])
    out = {}
    for i, A in enumerate(names):
        for B in names[i + 1:]:
            ta, tb = L0s[A]["tr"], L0s[B]["tr"]
            th = np.array([np.nan if v is None else v for v in L0s[A]["a"]["theta2d_deg"]], float)
            rows = {b: {"d": [], "rel": [], "rad": [], "per_joint": {j: [] for j in range(21)}, "frames": 0} for b in range(len(bins) - 1)}
            n_pair = 0
            pa, pb_, pm = [], [], []
            for f in range(len(th)):
                if not (ta["present"][f] and tb["present"][f]) or not np.isfinite(th[f]):
                    continue
                if iou(ta["box"][f], tb["box"][f]) < cfg16["diff2d"]["same_hand_iou_min"]:
                    continue
                m = ta["mask"][f] & tb["mask"][f]
                if m.sum() < 4:
                    continue
                n_pair += 1
                pa.append(ta["kp"][f])
                pb_.append(tb["kp"][f])
                pm.append(m)
                b = int(np.clip(np.searchsorted(bins, th[f], side="right") - 1, 0, len(bins) - 2))
                rows[b]["frames"] += 1
                bl = float(box_long(ta["box"][f]))
                ctr = ta["kp"][f][[j for j in est_ids if ta["mask"][f][j]]].mean(0)
                for j in np.flatnonzero(m):
                    d = tb["kp"][f][j] - ta["kp"][f][j]
                    u = ta["kp"][f][j] - ctr
                    nu = np.linalg.norm(u)
                    rows[b]["d"].append(float(np.linalg.norm(d)))
                    rows[b]["rel"].append(float(np.linalg.norm(d) / bl))
                    rows[b]["rad"].append(float(d @ u / nu / bl) if nu > 1e-6 else np.nan)
                    rows[b]["per_joint"][j].append(float(np.linalg.norm(d) / bl))
            tab = []
            for b in range(len(bins) - 1):
                r = rows[b]
                tab.append({"theta": [bins[b], bins[b + 1]], "frames": r["frames"], "n_points": len(r["d"]),
                            "median_px": float(np.median(r["d"])) if r["d"] else None, "median_rel": float(np.median(r["rel"])) if r["rel"] else None,
                            "p90_rel": float(np.percentile(r["rel"], 90)) if r["rel"] else None,
                            "radial_median_rel": float(np.nanmedian(r["rad"])) if r["rad"] else None,
                            "per_joint_median_rel": {j: (float(np.median(v)) if v else None) for j, v in r["per_joint"].items()}})
            out[f"{A}|{B}"] = {"a": A, "b": B, "n_frames": n_pair, "by_theta": tab,
                               "order_assignment": order_assignment(np.array(pa), np.array(pb_), np.array(pm)) if pa else None}
    return out


def order_assignment(kpa, kpb, mask):
    """Median distance D[i, j] between point i of detector B and point j of detector A over shared frames; the point order of the two
    detectors agrees when the optimal one-to-one assignment (Hungarian) of D is the identity."""
    from scipy.optimize import linear_sum_assignment
    D = np.full((21, 21), np.nan)
    for i in range(21):
        for j in range(21):
            ok = mask[:, i] & mask[:, j]
            if ok.any():
                D[i, j] = float(np.median(np.linalg.norm(kpb[ok, i] - kpa[ok, j], axis=-1)))
    Dc = np.where(np.isfinite(D), D, 1e9)
    r, c = linear_sum_assignment(Dc)
    perm = c[np.argsort(r)].tolist()
    return {"identity": perm == list(range(21)), "assignment": perm, "row_argmin": [int(np.argmin(Dc[i])) for i in range(21)]}
